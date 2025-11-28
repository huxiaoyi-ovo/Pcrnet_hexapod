import os
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from legged_gym import LEGGED_GYM_ROOT_DIR
# 指定 .tfevents 文件路径

"""绘制奖励函数和episode时长图"""
# events_file1 = f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Sep08_22-40-03_/events.out.tfevents.1757342404.uss.35791.0"
# events_file2 = f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Sep09_11-24-36_/events.out.tfevents.1757388278.uss.134229.0"

# # 加载 TensorBoard 事件文件
# ea1 = event_accumulator.EventAccumulator(events_file1)
# ea2 = event_accumulator.EventAccumulator(events_file2)
# ea1.Reload()
# ea2.Reload()

# reward_tag = 'Train/mean_reward'
# episode_length_tag = 'Train/mean_episode_length'

# # 获取 'Train/mean_reward' 数据
# scalars_reward = ea1.Scalars(reward_tag)
# steps_reward1, values_reward1 = zip(*[(x.step, x.value) for x in scalars_reward][:200])
# scalars_reward = ea2.Scalars(reward_tag)
# steps_reward2, values_reward2 = zip(*[(x.step, x.value) for x in scalars_reward][:200])
# # 获取 'Train/mean_episode_length' 数据
# scalars_length = ea1.Scalars(episode_length_tag)
# steps_length1, values_length1 = zip(*[(x.step, x.value*0.02) for x in scalars_length][:200])
# scalars_length = ea2.Scalars(episode_length_tag)
# steps_length2, values_length2 = zip(*[(x.step, x.value*0.02) for x in scalars_length][:200])
# # 创建图形
# plt.rcParams.update({'font.size': 14})
# fig, ax1 = plt.subplots(figsize=(8,5))
# # fig, ax1 = plt.subplots()

# # 绘制 'Train/mean_reward' 曲线
# ax1.plot(steps_reward1, values_reward1, color="#E62A2A", label='EGPO:r',linewidth=2)
# ax1.set_xlabel('Iterations',fontsize=18)
# ax1.set_ylabel('Mean Reward',fontsize=18)
# ax1.tick_params(axis='y')
# ax1.plot(steps_reward2, values_reward2, color="#580C0C", label='PPO:r',linewidth=2)


# # 创建一个共享的第二个 y 轴
# ax2 = ax1.twinx()
# # 绘制 'Train/mean_episode_length' 曲线
# ax2.plot(steps_length1, values_length1, color="#2BB449", label='EGPO:$T_{episode}$',linewidth=2)
# ax2.plot(steps_length2, values_length2, color="#0B611C", label='PPO:$T_{episode}$',linewidth=2)
# ax2.set_ylabel('Episode Length(s)',fontsize=18)
# ax2.tick_params(axis='y')

# # 添加图例
# ax1.legend(bbox_to_anchor=(0.6, 0.1),loc='lower left')
# ax2.legend(bbox_to_anchor=(0.6, 0.3),loc='lower left')

# # 设置图表标题

# # 显示图表
# plt.show()

"""绘制专家和EGPO和BC的运动速度和关节角度图"""
# import numpy as np
# from scipy.signal import lfilter

# # def extract_q_data(ea,title):
# #     q1 = ea.Scalars(title+'/q1')
# #     q2 = ea.Scalars(title+'/q2')
# #     q3 = ea.Scalars(title+'/q3')
# #     steps_q1, q1 = zip(*[(x.step*0.02, x.value) for x in q1])
# #     _, q2 = zip(*[(x.step, x.value) for x in q2])
# #     _, q3 = zip(*[(x.step, x.value) for x in q3])
# #     # steps_q1=int(float(steps_q1)*0.02)
# #     return q1,q2,q3,steps_q1
# def extract_v_data(ea,title):
#     vx=ea.Scalars(title+'/vx')
#     vy=ea.Scalars(title+'/vy')
#     steps,vx=zip(*[(x.step*0.02,x.value) for x in vx])
#     _,vy=zip(*[(x.step*0.02,x.value) for x in vy])
#     return vx,vy,steps
# # 事件文件路径
# log_dir=f"{LEGGED_GYM_ROOT_DIR}/logs/motion_data/"
# event_files=[]
# for file in os.listdir(log_dir):
#     if file.endswith('.0'):
#         event_files.append(file)
# print(event_files)

# file_labels=['EGPO','BC','Expert']
# # file_labels=['Expert','BC']
# # file_labels=['Expert']
# ea_list=[]
# # fig_q=plt.figure("joint")
# plt.rcParams.update({'font.size': 16})
# fig_v=plt.figure("Velocity",figsize=(10,6))
# color_list=[['#2BB449',"#058623"],["#7466EB","#271DB9"],["#DB6E6E","#861A1A"]]
# cmd_v_list=[[0.4,0],[0,0.55],[0.4,0.5]]
# alpha=0.93 #平滑因子
# for i,event_file in enumerate(event_files):
#     ea=event_accumulator.EventAccumulator(log_dir+event_file)
#     ea.Reload()
#     ea_list.append(ea)
#     # q1,q2,q3,steps=extract_q_data(ea_list[i],file_labels[i])
#     line_style='-'
#     line_width=1.5
#     # if i==0:
#     #     line_style='--'
#     #     line_width=1.5

#     # plt.plot(steps,q1,label=file_labels[i]+'/q1',ls=line_style)
#     # plt.plot(steps,q2,label=file_labels[i]+'/q2',ls=line_style)
#     # plt.plot(steps,q3,label=file_labels[i]+'/q3',ls=line_style)
#     vx,vy,steps=extract_v_data(ea_list[i],file_labels[i])
#     vx,vy=np.array(vx),np.array(vy)
#     vx_smooth = lfilter([1 - alpha], [1, -alpha], vx)
#     vy_smooth = lfilter([1 - alpha], [1, -alpha], vy)
#     #计算误差的大小

#     er1x=np.linalg.norm(vx_smooth[200:350]-0.4)
#     er1y=np.linalg.norm(vy_smooth[200:350])

#     er2x=np.linalg.norm(vx_smooth[550:700])
#     er2y=np.linalg.norm(vy_smooth[550:700]-0.6)  

#     er3x=np.linalg.norm(vx_smooth[800:1050]-0.3)
#     er3y=np.linalg.norm(vy_smooth[800:1050]-0.45)  

#     print(er1x,er1y,er2x,er2y,er3x,er3y)

#     print((er1x+er1y+er2x+er2y+er3x+er3y)/3.0)
#     # plt.figure("Velocity")
#     # plt.plot(steps,vx,label=file_labels[i]+'/vx',linewidth=line_width,ls=line_style)
#     plt.plot(steps,vx_smooth,label=file_labels[i]+': $v_x$',linewidth=line_width,ls=line_style,color=color_list[i][0])
#     # plt.plot(steps,vy,label=file_labels[i]+'/vy',linewidth=line_width,ls=line_style)
#     plt.plot(steps,vy_smooth,label=file_labels[i]+': $v_y$',linewidth=line_width,ls=line_style,color=color_list[i][1])
#     # plt.ylim(-0.75, 0.75) 
# # 虚线位置
# for x in (4, 11, 18):
#     plt.axvline(x, color='black', linestyle='--', linewidth=1)

# # 实线位置
# for x in (7, 14, 21):
#     plt.axvline(x, color='black', linestyle='-', linewidth=1)
# plt.plot([0,7,7,14,14,21],[0.4,0.4,0,0,0.4,0.4],ls='-.',linewidth=2,label="cmd: $v_x$")
# plt.plot([0,7,7,14,14,21],[0,0,0.55,0.55,0.5,0.5],ls='-.',linewidth=2,label="cmd: $v_y$")
# plt.xlabel('time(s)',fontsize=20)
# plt.ylabel('velocity(m/s)',fontsize=20)
# plt.legend(
#     loc="lower center",    # 图例在图的上方（center表示居中）
#     bbox_to_anchor=(0.5, -0.03),  # 调整图例位置 (x=0.5表示居中, y=-0.1表示往下移到图外)
#     ncol=4,                # 图例排成几列，这里4列
#     frameon=True          # 去掉图例边框
# )

# plt.show()


"""绘制多个奖励函数"""
# import numpy as np
# from scipy.signal import lfilter
# smooth_factor=0.7
# plt.rcParams.update({'font.size': 18})
# plt.figure(figsize=(10,5))
# root_file=f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/"
# file_names=['Sep10_13-44-29_ /events.out.tfevents.1757483070.uss.53815.0',
#            'Sep10_13-47-32_ /events.out.tfevents.1757483253.uss.54420.0',
#            'Sep10_14-43-36_ /events.out.tfevents.1757486618.uss.62989.0',
#            'Sep10_15-46-09_ /events.out.tfevents.1757490370.uss.80411.0']
# event_files=[]
# file_labels=['BC+PPO','PPO','EGPO','Expert']
# color_list=["#E91414","#058623","#1902E6","#7726D3"]
# for file in file_names:
#     event_files.append(root_file+file)
# for i,file in enumerate(event_files):
#     ea=event_accumulator.EventAccumulator(file)
#     ea.Reload()
#     scalars=ea.Scalars('Train/mean_reward')
#     steps, rewards=zip(*[(x.step, x.value) for x in scalars][:2000])
#     plt.plot(steps,rewards,linewidth=4.0,alpha=0.3,color=color_list[i])
#     rewards_smoothed = lfilter([1 - smooth_factor], [1, -smooth_factor], rewards)
#     plt.plot(steps,rewards_smoothed,label=file_labels[i],linewidth=1.0,color=color_list[i])
# plt.xlabel("Iterations")
# plt.ylabel("Mean Reward")
# plt.ylim(0,60)
# plt.legend()
# plt.show()


"""从bag中读取速度数据，然后计算速度误差"""
import rosbag
from pathlib import Path
import numpy as np
bag_name="/home/ubuntu/valerian_ws/BIH_ws/bag/velocity/2025-09-18-15-04-28.bag"
total_err=0.0
i=0.0
vx_des=[]
vy_des=[]
vx_real=[]
vy_real=[]
with rosbag.Bag(bag_name,'r') as bag:
    for topic, msg, t in bag.read_messages(topics=["/base_msg"]):
        msg=np.array(msg.data)
        print("msg=",msg)
        total_err +=np.linalg.norm(msg[0:2]-msg[3:5])/3.0
        i+=1.0
print("ave_err=",total_err/i)

'#271DB9'