
"""正态分布曲线绘制"""
# from matplotlib import pyplot as plt
# import numpy as np

# import torch
# from torch.distributions import Normal

# dist_1 = Normal(torch.tensor([1.2]),torch.tensor([0.1]))
# dist_2 = Normal(torch.tensor([1.5]),torch.tensor([0.15]))
# x = torch.arange(0.75,2,0.005)
# y = torch.exp(dist_1.log_prob(x))
# z = torch.exp(dist_2.log_prob(x))

# plt.plot(x, y, label='N(miu=1.2, sigma=0.1)', color='blue')
# plt.plot(x, z, label='N(miu=1.5, sigma=0.15)', color='red')

# # 在x=1.6处画竖线
# plt.axvline(x=1.7, color='green', linestyle='--', label='x = 1.6')

# # 计算并标注交点值
# y_val = torch.exp(dist_1.log_prob(torch.tensor(1.7))).item()
# z_val = torch.exp(dist_2.log_prob(torch.tensor(1.7))).item()

# plt.text(1.7, y_val +0.2, f'y = {y_val:.4f}', color='blue', ha='left')
# plt.text(1.7, z_val + 0.2, f'z = {z_val:.4f}', color='red', ha='left')
# plt.legend()
# plt.show()


"""从log中将actor_critic的参数读取并单独保存下来"""
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO
from legged_gym.utils import class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.modules import ActorCritic, ActorCriticEncoder
import torch

device = 'cuda'
cfg = HexGroundCfg()
policy_cfg = HexGroundCfgPPO.policy()
policy_cfg_dict = class_to_dict(policy_cfg)
#将actor的参数单独保存
actor_critic=ActorCritic(cfg.env.num_observations,cfg.env.num_privileged_obs,
                         cfg.env.num_actions,**policy_cfg_dict).to(device)

on_policy_state_dict = torch.load(
    f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Oct15_15-06-56_/model_400.pt",weights_only=False)
actor_critic.load_state_dict(on_policy_state_dict['model_state_dict'])

state_dict = actor_critic.actor.state_dict()
prefixed_state_dict = {f"actor.{k}": v for k, v in state_dict.items()}
torch.save(prefixed_state_dict, f"{LEGGED_GYM_ROOT_DIR}/agents/EGPO_400.pt")

#将actor，obs_vgf_estimator, lstm, lstm_fc参数保存下来
# actor_critic_encoder = ActorCriticEncoder(cfg.env.num_observations,cfg.env.num_actions,**policy_cfg_dict)
# on_policy_state_dict = torch.load(f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Sep19_11-23-03_ /model_200.pt",weights_only=True)
# actor_critic_encoder.load_state_dict(on_policy_state_dict['model_state_dict'])
# modify_state_dict={}
# for param, value in actor_critic_encoder.state_dict().items():
#     if 'actor' in param or 'lstm' in param or 'lstm_fc' in param:
#         if 'actor_obs_priv_estimator' in param:
#             param=param.replace('actor_obs_priv_estimator','obs_vgf_estimator')
#         modify_state_dict[param]=value

# torch.save(modify_state_dict,f"{LEGGED_GYM_ROOT_DIR}/agents/encoder_200.pt")

"""测试torch功能"""
# import torch
# a=torch.rand(1,3)
# a *= a>0.5
# print(a)

"""绘图，查看exp(-a/b)中b对曲线"""
# from matplotlib import pyplot as plt
# import numpy as np
# import torch
# import math
# x = np.arange(1, 200, 0.1)
# x = x/200.0
# y = np.exp(-x/0.2)
# z = np.exp(-x/0.5)
# w = np.exp(-x/0.9)

# # plt.plot(x, y, label='b=0.2')
# # plt.plot(x, z, label='b=0.5')
# # plt.plot(x, w, label='b=0.9')

# err = torch.arange(0,1.2,0.001)

# near_rew=torch.exp(-(err+0.2)/0.6)
# far_rew=torch.tanh((err+0.2)/0.6)
# w=torch.sigmoid((err-0.15)*10)
# # w=0.8
# plt.plot(err,w)

# #绘制err一直大于0.1的情况
# smooth_near=w*near_rew + (1-w)*far_rew
# smooth_far = (1-w)*near_rew + w*far_rew

# plt.plot(err,smooth_near,label='near')
# plt.plot(err,smooth_far,label='far')

#     # w = torch.sigmoid(k * (err - d))
    
#     # # 靠近奖励：指数型，err 小时高
#     # near_rew = torch.exp(-err / sigma)
    
#     # # 大步奖励：tanh 型，随 step 增大而增长
#     # far_rew = torch.tanh(step / L)
    
#     # # 平滑组合
#     # rew = (1 - w) * near_rew + w * far_rew

# plt.legend()
# plt.show()


# print(torch.cuda.get_device_name(1))  # GPU 名称


"""测试并行环境"""
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.optim as optim
# import os
# from torch.nn.parallel import DistributedDataParallel as DDP


# def example(rank, world_size):
#     print("rank=",rank)
#     # create default process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     # create local model
#     model = nn.Linear(10, 10).to(rank)
#     # construct DDP model
#     ddp_model = DDP(model, device_ids=[rank])
#     # define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # forward pass
#     outputs = ddp_model(torch.randn(20, 10).to(rank))
#     labels = torch.randn(20, 10).to(rank)
#     # backward pass
#     loss_fn(outputs, labels).backward()
#     # update parameters
#     optimizer.step()

# def main():
#     world_size = 3
#     mp.spawn(example,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True)

# if __name__=="__main__":
#     # Environment variables which need to be
#     # set when using c10d's default "env"
#     # initialization mode.
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     main()




"""先获取专家数据，使用智能体计算专家动作在相同状态分布下的概率密度"""

# from legged_gym.envs import HexGroundCfg,HexGroundCfgPPO

# from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.utils.helpers import class_to_dict,get_load_path,update_cfg_from_args
# from legged_gym.utils import get_args
# from rsl_rl.modules import ActorCritic

# import os
# import torch
# from torch.utils.data import TensorDataset, DataLoader

# device='cuda:0'


# if os.path.exists(LEGGED_GYM_ROOT_DIR+'/resources/expert_data/bc_episode_0.pth'):
#     print("Find expert data, loading......")
#     buffer = torch.load(LEGGED_GYM_ROOT_DIR+'/resources/expert_data/bc_episode_0.pth',weights_only=True)
#     print("Load expert data done!")
# dataset = TensorDataset(buffer['obs'].to(device), buffer['expert_actions'].to(device))
# dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# #加载智能体参数
# env_cfg=HexGroundCfg()
# train_cfg=HexGroundCfgPPO()
# args=get_args()
# env_cfg,train_cfg = update_cfg_from_args(env_cfg,train_cfg,args) 
# train_cfg=class_to_dict(train_cfg)

# actor_critic = ActorCritic(env_cfg.env.num_observations,env_cfg.env.num_privileged_obs,
#                             env_cfg.env.num_actions,**train_cfg['policy']).to(device)
# actor_critic.eval()
# log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg['runner']['experiment_name'])
# resume_path = get_load_path(log_root, load_run=train_cfg['runner']['load_run'], checkpoint=train_cfg['runner']['checkpoint'])
# print("============>load model {}<===============".format(resume_path))
# actor_critic.load_state_dict(torch.load(resume_path,weights_only=True)['model_state_dict'])

# #计算数据
# log_prob_list=[]
# for obs, expert_actions in dataloader:
#     actor_critic.update_distribution(obs)
#     log_probs=actor_critic.get_actions_log_prob(expert_actions).mean()
#     log_prob_list.append(log_probs)
# print(torch.exp(torch.tensor(log_prob_list).mean()))


"""测试参数传递功能"""
# def func1(param1,param2,param3,param4,param5):
#     print("{}; {}; {}; {}; {}".format(param1,param2,param3,param4,param5),)

# param_dict={"param3":123.4,"param5":'aaaaa',"param4":[1,2,3]}

# func1(1,2,**param_dict)

"""遍历slow bag，将其中异常的数据删除后保存"""
# import rosbag
# import rosbag
# leg_names = ['RF','RM','RB','LF','LM','LB']
# topics = []
# for leg_name in leg_names:
#     topics.append(['/'+leg_name+'/sita_des','/'+leg_name+'/sita_cur'])

# input_bag = "/home/ubuntu/valerian_ws/BIH_ws/bag/120_180kp/slow.bag"
# output_bag = "/home/ubuntu/valerian_ws/BIH_ws/bag/120_180kp/slow_clean.bag"
# # output_bag=rosbag.Bag(output_bag, 'a')
# sita_des=[]
# sita_cur=[]
# dot_sita_cur=[]
# torques=[]
# j=0
# des_topics=None
# des_msg=None
# des_t=None
# with rosbag.Bag(output_bag, 'w') as outbag:
#     for pairs in topics:
#         for topic, msg, t in rosbag.Bag(input_bag).read_messages(pairs):
#             j+=1
#             normal_msg = True
#             if 'des' in topic:
#                 sita_des = list(msg.data[1:4])
#                 des_t = t
#                 des_msg = msg
#                 des_topics = topic
#             if 'cur' in topic:

#                 sita_cur = list(msg.data[0:3])
#                 torques = list(msg.data[3:6])
#                 dot_sita_cur = list(msg.data[6:9])
#                 for i in range(3):
#                     if abs(((sita_des[i] - sita_cur[i]) * 120 - dot_sita_cur[i] * 0.8) - torques[i]) > 40:
#                         print(abs(((sita_des[i] - sita_cur[i]) * 120 - dot_sita_cur[i] * 0.8) - torques[i]))
#                         normal_msg = False
#                 if normal_msg:
#                     j+=1
#                     outbag.write(topic, msg, t)
#                     outbag.write(des_topics, des_msg, des_t)

# print(f"✅ 已保存到: {output_bag},number={j}")

"""检查bag中数据"""
# import torch
# from pathlib import Path
# import rosbag
# import matplotlib.pyplot as plt
# def process_bag_data(bag_path:str,index:int,device):
#     bag_names=list(Path(bag_path).rglob('*.bag'))
#     bag_names=["/home/ubuntu/valerian_ws/BIH_ws/bag/120_180kp/slow_fast.bag"]
#     print(bag_names)
#     leg_names=['RF','RM','RB','LF','LM','LB']
#     topics=[] #RF sita_des RF sita_cur
#     for leg_name in leg_names:
#         topics.append(['/'+leg_name+'/sita_des','/'+leg_name+'/sita_cur'])

    
#     sita_des=[]
#     sita_cur=[]
#     dot_sita_cur=[]
#     torque_cur=[]
#     t_cur_list=[]
#     t_des_list=[]
#     for bag_name in bag_names:
#         with rosbag.Bag(bag_name,'r') as bag:
#             for pairs in topics:
#                 for topic, msg, t in bag.read_messages(pairs):
#                     msg=list(msg.data)
#                     # print(topic)
#                     if 'des' in topic:
#                         sita_des.append(msg[1:4])
#                         t_des_list.append(t.to_sec())
#                     elif 'cur' in topic:
#                         sita_cur.append(msg[0:3])
#                         torque_cur.append(msg[3:6])
#                         dot_sita_cur.append(msg[6:9])
#                         t_cur_list.append(t.to_sec())
    


#     pos_err = (torch.tensor(sita_des))-(torch.tensor(sita_cur))
#     vel_err = -torch.tensor(dot_sita_cur)
#     torques = torch.tensor(torque_cur)
#     t_cur = torch.tensor(t_cur_list,dtype=torch.float64)
#     t_des = torch.tensor(t_des_list,dtype=torch.float64)
#     return pos_err.to(device), vel_err.to(device), torques.to(device), t_cur.to(device), t_des.to(device)


# pos_err,vel_err,torques, t_cur, t_des=process_bag_data("/home/ubuntu/valerian_ws/BIH_ws/bag/120kp1kd",0,'cpu')
# pos_err = pos_err.view(-1)
# vel_err = vel_err.view(-1)
# torques = torques.view(-1)

# ideal_torques=pos_err*120+0.8*vel_err
# big_difference = (ideal_torques-torques).abs()>40
# # print(big_difference.sum())
# # print(f"pos_err\n{pos_err[big_difference]}\nvel_err\n{vel_err[big_difference]}\ntorques\n{torques[big_difference]}")
# fig = plt.figure()
# # # # 添加一个三维坐标轴
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(pos_err[big_difference], vel_err[big_difference], torques[big_difference])
# ax.scatter(pos_err[::20], vel_err[::20], torques[::20])
# # ax.scatter(pos_err, vel_err, torques)

# # ax.scatter(x, y, torques_pred.detach().cpu().numpy())
# # ax.scatter(x, y, ideal_torques.cpu().numpy())

# # 设置坐标轴标签
# ax.set_xlabel('pos_err (m) Axis')
# ax.set_ylabel('vel_err (m/s) Axis')
# ax.set_zlabel('ideal_torques (Nm) Axis')
# plt.show()




