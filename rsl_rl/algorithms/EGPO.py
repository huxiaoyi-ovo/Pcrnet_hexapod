from rsl_rl.algorithms import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.distributions import Normal
#需要修改act方法，接受参考的expert_action
class EGPO(PPO):
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 expert_interface_iter=200,
                 expert_alpha_min=0.1,
                 expert_alpha_schedule="cosine"
                 ):
        super().__init__(actor_critic,
                         num_learning_epochs,
                         num_mini_batches,
                         clip_param,
                         gamma,
                         lam,
                         value_loss_coef,
                         entropy_coef,
                         learning_rate,
                         max_grad_norm,
                         use_clipped_value_loss,
                         schedule,
                         desired_kl,
                         device)
        # self.tic1=time.time()
        # self.tic2=time.time()
        self.expert_interface_iter = expert_interface_iter #专家干预的迭代次数
        self.expert_alpha_min = expert_alpha_min
        self.expert_alpha_schedule = expert_alpha_schedule

        
    def _get_expert_alpha(self, it):
        if self.expert_interface_iter <= 0:
            return 0.0
        progress = min(float(it) / self.expert_interface_iter, 1.0)
        if self.expert_alpha_schedule == "cosine":
            alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            alpha = 1.0 - progress
        return max(alpha, self.expert_alpha_min)

    def act(self,obs,critic_obs,expert_action,it):
        
        agent_actions=self.actor_critic.act(obs).detach()
        self.transition.exeprt_actions = expert_action.detach()

        # self.transition.actions = expert_action.detach()
        # self.transition.actions = agent_actions.detach()

        alpha_t = self._get_expert_alpha(it)
        # # alpha_t = math.exp(-(it/200.0)/0.2)#测试专家动作一下非线性衰减
        self.transition.actions = alpha_t*expert_action.detach() + (1-alpha_t)*agent_actions.detach()


        # #计算专家的策略下mix_action 的 log_prob
        # if it<300:
        #     alpha_t=(300-it)/300.0
        # elif it<600:
        #     alpha_t=(it-300)/300.0
        # #构造专家的分布
        # mixed_norm=expert_action.detach() * (1-alpha_t) + self.actor_critic.action_mean * alpha_t
        # mixed_std = torch.ones_like(expert_action)*0.01 * (1-alpha_t) + self.actor_critic.action_std * alpha_t
        # mixed_dist = Normal(mixed_norm, mixed_std)
        # self.transition.actions = mixed_dist.sample()


        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.actions_log_prob = mixed_dist.log_prob(self.transition.actions).sum(dim=-1).detach()
        # self.transition.action_mean = mixed_norm
        # self.transition.action_sigma = mixed_std

        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # if time.time()-self.tic1>3.0:
            # print("expert action=",expert_action[0])
            # print("agent action=",agent_actions[0])
            # print("(agent_action-expert_action).mean()=",torch.norm(agent_actions-expert_action,dim=1).mean())
            # print("iteration num=",it)
            # print("action log prob.shape=",self.transition.actions_log_prob.shape) #nums_env*18
            # print("==========>agent action log prob=",torch.mean(self.actor_critic.get_actions_log_prob(agent_actions),dim=0))
            # min_log_prob = torch.min(self.actor_critic.get_actions_log_prob(expert_action))
            # max_log_prob = torch.max(self.actor_critic.get_actions_log_prob(expert_action))

            # print("============>expert action log prob mean()=",torch.mean(self.actor_critic.get_actions_log_prob(expert_action),dim=0))
            # print("=============>expert action log prob min()=",min_log_prob,"\n=============>expert action log prob max()=",max_log_prob)
            # print("============>mixed action log prob=",torch.mean(self.actor_critic.get_actions_log_prob(self.transition.actions),dim=0))
            # self.tic1 = time.time()        
        return self.transition.actions
    def update(self,it,tot_iter):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_BC_loss = 0.0
        mean_ratio = 0.0
        mean_advantage = 0.0

        mean_actor_mean = 0.0
        mean_actor_min = 1.0e3
        mean_actor_max = -1.0e3

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch,expert_actions_batch,hid_states_batch, masks_batch in generator:


                agent_actions_batch=self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy


                #BC
                expert_alpha = self._get_expert_alpha(it)
                # BC_loss_fn = torch.nn.MSELoss()
                # BC_loss = BC_loss_fn(agent_actions_batch,expert_actions_batch)*alpha*5.0
                # BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)*alpha*0.5
                BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)*expert_alpha*2.0
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

                #clip ratio
                ratio = torch.clamp(ratio, 0.1,10.0)

                # #监测ratio异常
                # abnorm_mask = (ratio>100.0) | (ratio<0.001)
                # if abnorm_mask.any():
                #     print("==============>Detecting abnormal ratio! corresponding variables<============")
                #     print("abnorm number=",abnorm_mask.sum().item())
                #     print("abnorm ratio\n",ratio[abnorm_mask])
                #     print("old_actions_log_prob_batch\n",torch.squeeze(old_actions_log_prob_batch[abnorm_mask]))
                #     print("current_log_prob_batch\n",actions_log_prob_batch[abnorm_mask])


                #     print("expert action batch")
                #     print(actions_batch[abnorm_mask])

                #     print("current agent policy network")
                #     print("mean\n",mu_batch[abnorm_mask])
                #     print("std\n",sigma_batch[abnorm_mask])
                #     print("log_prob\n",self.actor_critic.distribution.log_prob(actions_batch)[abnorm_mask])




                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() * alpha

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                #给探索因子加上一个时间衰减项,专家退出前，适当提高这一项权重，避免方差下降太快导致失去探索性
                entropy_alpha = math.exp(-(it/tot_iter) / 0.4)
                # if it<150:
                #     loss=self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()+BC_loss
                # else:
                loss = surrogate_loss + self.value_loss_coef * value_loss - entropy_alpha * self.entropy_coef * entropy_batch.mean() + BC_loss

                # Gradient stepv  
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                step=True
                for name, param in self.actor_critic.named_parameters():
                    if 'actor' in name:
                        if param.grad is not None:  # 确认这个参数确实有梯度
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"NaN or Inf detected in gradient of {name}")    
                                step=False
                #因为一部分动作是从专家采样来的，因此插值后偶尔出现梯度nan的情况，只需要不在这个时候step就可以
                if step: 
                    self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.abs().item()
                mean_BC_loss += BC_loss.item()
                mean_ratio += ratio.mean().item()
                mean_advantage += advantages_batch.abs().mean().item()


                # for name, param in self.actor_critic.named_parameters():
                #     if 'actor' in name:
                #         nan_param_mask = torch.isnan(param)
                #         if nan_param_mask.any():
                #             print(f"find nan in actor {name} ")
                #             print(f"actor {name} shape {param.shape}, has nan param number {nan_param_mask.sum()}")
                #         mean_actor_mean += param.mean().item()
                #         mean_actor_min = param.min().item() if mean_actor_min>param.min().item() else mean_actor_min
                #         mean_actor_max = param.max().item() if mean_actor_max<param.max().item() else mean_actor_max



                        # mean_actor_min
        

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_BC_loss /= num_updates
        mean_ratio /= num_updates
        mean_advantage /= num_updates

        mean_actor_mean /= num_updates


        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_BC_loss, mean_ratio, mean_advantage, mean_actor_mean, mean_actor_min, mean_actor_max
