"""
================================================================================
EGPO Encoder Runner - 专家引导编码器强化学习训练器
================================================================================

核心功能:
1. 专家引导: 前200次迭代使用专家动作线性衰减混合
2. 观测编码: 处理分离的obs(67), vgf(30), terrain(143)观测结构
3. 特权训练: CNN编码地形，为Phase 2的Estimator/LSTM准备

与OnPolicyRunner的区别:
- 支持分离观测结构(obs_dict)
- 集成专家动作混合(alpha_t衰减)
- 使用EGPOEncoder算法(BC loss + PPO loss)
- 初始化存储时硬编码维度: actor_obs=[97], critic_obs=[143]

训练流程:
1. env.reset_separate() → {obs, vgf, terrain}
2. encode_obs() → obs_splice(97), terrain_latent(32)
3. act() → mixed_action = α×expert + (1-α)×agent
4. env.step() → reward, next_obs
5. update() → PPO + BC Loss

适用场景:
- Phase 1: EGPO训练，深度相机关闭，Raycast启用
- 后续Phase 2/3需要切换到Estimator/LSTM模式
================================================================================
"""
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticEncoder
from rsl_rl.algorithms import EGPO, EGPOEncoder
from rsl_rl.env import VecEnv
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import deque
import time
import os
import statistics


class EGPOEncoderRunner(OnPolicyRunner):
    def __init__(self,
                 env:VecEnv,
                 train_cfg,
                 log_dir,
                 device='cpu'):
        print(f"________>rsl from current dir, using device {device}<____________")
        # super().__init__(env,train_cfg,log_dir,device)
        # #因为初始化的算法变为EGPO，所以重写init函数
        # 因为要添加一个专家，也是actorcritic类实现的，所以额外初始化一个expert:ActorCritic 实际只用到了其中的actor部分
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic

        actor_critic: ActorCriticEncoder = actor_critic_class( self.env.num_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO EGPO EGPOEncoder
        self.alg: EGPOEncoder = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #额外添加的expert
        # self.expert = ActorCritic(self.env.num_obs,num_critic_obs ,self.env.num_actions, **self.policy_cfg).to(self.device)
        # self.expert.load_state_dict(torch.load(self.cfg["expert_path"],weights_only=True))
        # self.expert.eval()
        # expert的干预时间

        # ============================================================
        # 初始化存储: 硬编码维度 (EGPO Encoder架构)
        # ============================================================
        # actor_obs_shape = [67+30] = [97]  - obs + vgf拼接
        # critic_obs_shape = [11×13] = [143] - Raycast地形高度图
        # 
        # 注意: 这里硬编码了维度，不使用env.num_privileged_obs
        # 因为EGPO架构对actor和critic输入有特殊要求:
        # - Actor: 需要obs_splice(97) + terrain_latent(32) = 129
        # - Critic: CNN直接处理11×13地形图 → 32维latent
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs+30], [11*13], [self.env.num_actions])
        # self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # print("reset env")
        obs_dict = self.env.reset_separate()  # 返回观测字典
    
    def learn(self,num_learning_iterations, init_at_random_ep_len=False):
        #因为要使用expert_actions的进行插值，所以重写learn函数, 并且alg返回中多了一项bc_loss
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir,flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 从观测字典获取分离的观测
        obs_dict = self.env.reset_separate()
        obs = obs_dict['proprioception']
        obs_vgf = obs_dict['privileged']
        obs_terrain = obs_dict['terrain']

        expert_actions=self.env.get_expert_actions() #从环境提供的获取专家动作
        # expert_actions = self.expert.act_inference(obs) #从自身初始化获取的专家动作

        # critic_obs = privileged_obs if privileged_obs is not None else obs
        # obs, critic_obs, expert_actions = obs.to(self.device), critic_obs.to(self.device), expert_actions.to(self.device)
        obs, obs_vgf,obs_terrain,expert_actions = obs.to(self.device), obs_vgf.to(self.device), obs_terrain.to(self.device), expert_actions.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer=deque(maxlen=100)
        lenbuffer=deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs,dtype=torch.float,device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs,dtype=torch.float,device=self.device)


        tot_iter=self.current_learning_iteration+num_learning_iterations
        for it in range(self.current_learning_iteration,tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if it<self.alg.expert_interface_iter: #专家退出后，不再计算其参考动作
                        expert_actions=self.env.get_expert_actions()
                        expert_actions = expert_actions.to(self.device)

                    obs_splice,obs_terrain_latent = self.alg.encode_obs(obs,obs_vgf,obs_terrain)
                    actions = self.alg.act(obs_splice,obs_terrain_latent,expert_actions,it)

                    obs_dict, rewards, dones, infos = self.env.step_separate(actions)  # 返回观测字典
                    obs = obs_dict['proprioception']
                    obs_vgf = obs_dict['privileged']
                    obs_terrain = obs_dict['terrain']

                    obs, obs_vgf, obs_terrain = obs.to(self.device), obs_vgf.to(self.device), obs_terrain.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0


                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_splice,obs_terrain_latent)
            
            # mean_value_loss, mean_surrogate_loss, \
            # mean_estimator_loss, mean_lstm_encoder_loss, \
            # = self.alg.update(it,tot_iter) #传入it 和tot_iter用于计算衰减因子
            mean_value_loss, mean_surrogate_loss,mean_BC_loss,\
            mean_estimator_loss, mean_lstm_encoder_loss=self.alg.update(it,tot_iter)

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self._save_training_summary()


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        #
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/bc_loss', locs['mean_BC_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_estimator_loss',locs['mean_estimator_loss'], locs['it'])
        self.writer.add_scalar('Loss/lstm_encoder_loss', locs['mean_lstm_encoder_loss'], locs['it'])
        #
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)


        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_BC_loss']:.4f}\n"""
                          f"""{'estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'lstm encoder loss:':>{pad}} {locs['mean_lstm_encoder_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

