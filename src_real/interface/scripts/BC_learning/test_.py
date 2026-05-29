# import os
# import time
# from datetime import datetime
# from isaacgym.torch_utils import *
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import math
# from hex_cfg import HexCfg
# from kinematic import Kinematic

# from BC_learning.Agent_utils import BC_Agent,Gait_Agent

"""批次保存数据与加载数据"""
# # 假设 generate_pairs(num) 是生成 num 对输入输出的函数
# def generate_pairs(num):
#     inputs = torch.randn(num, 90,device='cuda')  # 输入数据：形状为 (num, 90)
#     outputs = torch.randn(num, 30,device='cuda')  # 输出数据：形状为 (num, 30)
#     return inputs.clone(), outputs.clone()

# # 设置每个文件保存的样本数量
# # data generation
# batch_size = 50
# num_samples = 1000000
# num_batches = num_samples // batch_size

# format_date=datetime.now().strftime('%Y-%m-%d,%H:%M:%S')
# file_name=os.getcwd()+'/bag/'+format_date

# # # 逐批次保存数据
# def DataSave():
#     all_data={'input':[],'output':[]}
#     inputs_batch=torch.zeros(batch_size,90,device='cuda')
#     out_batch=torch.zeros(batch_size,30,device='cuda')
#     for batch_idx in range(num_batches):
#         inputs, outputs = generate_pairs(batch_size)  # 生成 50 对输入输出
#         inputs_batch=torch.cat([inputs_batch,inputs],dim=0)
#         out_batch=torch.cat([out_batch,outputs],dim=0)
#         if batch_idx % 1000 == 0:
#             all_data['input']=inputs_batch
#             all_data['output']=out_batch
#             inputs_batch=inputs.clone()
#             out_batch=outputs.clone()
#             pt_file_name=file_name+str(batch_idx)+'.pt'
#             torch.save(all_data,pt_file_name)

# data load
# root_path=os.getcwd()
# file_name=root_path+'/bag/2025-04-07,17:31:54-.pt'
# print(file_name)
# data=torch.load(file_name,weights_only=True)
# print(data['action'].shape)
# print(data['velocity'])
# obs=data['obs'] #type: torch.Tensor
# action=data['action'] #type: torch.Tensor
# print("obs.shape:",obs.shape)
# print("action.shape:",action.shape)
# dataset=TensorDataset(obs,action)
# train_loader=DataLoader(dataset,batch_size=64,shuffle=True)


# for input, target in train_dataset:
#     print(input.shape, target.shape)


# """四元数旋转测试"""
# axis=[0,0,1]
# rotate_angle=math.pi/2
# q1=torch.tensor([[math.sin(rotate_angle/2.0)*axis[0],
#                  math.sin(rotate_angle/2.0)*axis[1],
#                  math.sin(rotate_angle/2.0)*axis[2],
#                  math.cos(rotate_angle/2.0)]],dtype=torch.float32)

# vec=torch.tensor([[1,1,0.0]])
# rotated_vec=quat_rotate(q1,vec)
# vec_1=torch.tensor([[0,1,0.0]])
# print(quat_rotate_inverse(q1,vec_1))

# a=torch.rand(3,2)
# c=torch.zeros(2,3,2)
# print(a)
# print(torch.norm(c-a,p=2,dim=2))
#dim test
# a=torch.rand(6,3)
# print(a.sum(dim=0))

# env_nums=1
# hex_cfg=HexCfg("hex_cfg.yaml")
# kin=Kinematic(hex_cfg.link.l1,hex_cfg.link.l2,hex_cfg.link.l3,'cpu')
# joint=torch.tensor([[math.pi/9.0,0.7,-2.14]],dtype=torch.float32)
# pos=torch.zeros_like(joint)
# kin.ForwardKin(joint,pos)
# print(pos)
# a=torch.zeros(1)
# b=torch.ones(1,dtype=torch.bool)
# print(a[b])

# env_nums=6
# command=torch.rand(env_nums,2,dtype=torch.float32)
# c=torch.rand(env_nums,dtype=torch.float32)<0.5
# command[c]=1000
# print(command)
# norm=torch.ones(env_nums,6,2,dtype=torch.float32)
# print(command)
# res=torch.bmm(norm,command.unsqueeze(-1)).squeeze(-1)<1

# command = torch.zeros(6,5,dtype=torch.float32,device='cuda:0') 
# print(command.shape[0])
# print(torch.arange(command.shape[0],dtype=torch.int32,device='cuda:0'))

"""单个输入输出对测试"""
# # data_file=os.getcwd()+'/bag/test_data_set/2025-04-10,12:02:33-.pt'
# data_file=os.getcwd()+'/bag/test_data_set/2025-04-10,11:45:02-.pt'
# # data_file=os.getcwd()+'/bag/train_data_set/2025-04-09,16:09:47-.pt'
# # data_file=os.getcwd()+'/bag/2025-04-09,20:29:23-.pt'
# dataset=torch.load(data_file,weights_only=True)
# obs=dataset['obs']
# action=dataset['action']

# obs_q=obs[:,13:31]
# obs_torq=obs[:,49:67]
# obs_suction_force=obs[:,67:73]
# obs_last_ad=obs[:,101:107]

# obs_new=torch.cat([obs_q,obs_torq,obs_suction_force],dim=1)
# # obs_new=torch.cat([obs_q,obs_torq],dim=1)
# action_new=action[:,24:30]

# test_dataset=TensorDataset(obs_new,action_new)
# test_loader=DataLoader(test_dataset,batch_size=128,shuffle=True)

# # # model=Gait_Agent().to('cuda')
# # # model.load_state_dict(torch.load(os.getcwd()+'/bag/gait.pth',weights_only=True))

# model=Gait_Agent().to('cuda')
# model.load_state_dict(torch.load(os.getcwd()+'/bag/sampled_dataset_gait.pth',weights_only=True))
# model.eval()

# criterion = torch.nn.MSELoss()
# # 初始化性能指标
# total_loss = 0.0
# correct_predictions = 0

# # 测试模型
# with torch.no_grad():  # 禁用梯度计算
#     for inputs, labels in test_loader:
#         # 前向传播
#         outputs = model(inputs)
        
#         # 计算损失
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
        
#         outputs=(outputs>0.8).to(torch.int32)
#         labels=labels.to(torch.int32)
#         correct_predictions += ((torch.abs(outputs-labels)).sum(dim=1)<1).sum().item()


# # 计算平均损失和准确率
# average_loss = total_loss / (len(test_loader))
# accuracy = correct_predictions / (len(test_loader)*128)

# print(f"平均损失: {average_loss:.4f}")
# print(f"准确率: {accuracy:.4f}")

# relu=torch.nn.ReLU()
# a=torch.tensor([-10,22,3,-4],dtype=torch.float32)
# print(torch.nn.functional.relu(a))

# a=torch.rand(1,3,4,device='cuda:0')
# a_flat=a.view(12)
# torch.save(a_flat,'test.pt')

"""在两个线程中发布消息"""
# import time
# import threading
# import rospy
# from std_msgs.msg import String

# def pub_msg(topic_name):
#     time_start=time.time()
#     pub=rospy.Publisher(topic_name,String,queue_size=10)
#     msg=String()
#     msg.data="thread:"+topic_name
#     rate=rospy.Rate(1)
#     while not rospy.is_shutdown():
#         msg.data="thread:"+topic_name+":->time: "+(time.time()-time_start).__str__()
#         pub.publish(msg)
#         rate.sleep()

# def main():
#     rospy.init_node('pub_msg',anonymous=True)
#     t1=threading.Thread(target=pub_msg,args=("/topic1",))
#     t2=threading.Thread(target=pub_msg,args=("/topic2",))

#     t1.start()
#     t2.start()

#     t1.join()
#     t2.join()
# if __name__=="__main__":
#     main()


import torch
a=torch.rand(2,3,6)
print(torch.max(a,dim=2))

