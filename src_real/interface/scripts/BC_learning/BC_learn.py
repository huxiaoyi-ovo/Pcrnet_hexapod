from BC_learning.Agent_utils import BC_Agent, BC_Agent_Dist
import math, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
def NormalizeData(obs,action):

    pass

def LoadData(path):
    data_dict=torch.load(path,weights_only=True)
    #load raw data obs:batch_size*77 action:batch_size*30
    obs=data_dict['obs'] # type: torch.Tensor
    action=data_dict['action'] # type: torch.Tensor


    # obs_root_q_qdot_qtorq=obs[:,0:67]
    # obs_command=obs[:,73:77]
    # obs=torch.cat([obs_root_q_qdot_qtorq,obs_command],dim=1)

    obs=obs[:,0:77]

    print(obs.shape)

    #

    # # obs_command_lastq_lastad=obs[:,73:107]
    # obs=torch.cat([obs_root_q_qdot_qtorq,obs_command],dim=1)
    # obs_root=obs[:,0:13]
    # obs_q=obs[:,13:31]
    # obs_torq=obs[:,49:67]
    # obs_suction_force=obs[:,67:73]
    # obs_last_ad=obs[:,101:107]

    # obs_new=torch.cat([obs_q,obs_torq,obs_suction_force],dim=1)
    # action_new=action[:,24:30]

    # obs_q_qdot_torq=obs[:,13:67]
    # obs_command=obs[:,73:77]
    # obs=torch.cat([obs_q_qdot_torq,obs_command],dim=1)
    # action=action[:,0:24]
    
    train_dataset=TensorDataset(obs,action)
    # train_dataset=TensorDataset(obs_new,action_new)
    train_loader=DataLoader(train_dataset,batch_size=512,shuffle=True)
    return train_loader

#步态切换数据集：吸附状态(3个保持 4个保持 5个保持 6个保持) & 吸附状态切换 各占50%
def CreateGaitData(path):
    data_set=torch.load(path,weights_only=True)
    obs=data_set['obs'] #type: torch.Tensor
    action=data_set['action'] #type: torch.Tensor

    #找到状态切换与状态保持的mask
    different_mask=(obs[:,101:107]-action[:,24:30]).any(dim=1)
    different_index=torch.where(different_mask)[0]
    same_index_list=[]
    for i in [3,4,5,6]:
        index=torch.where( (action[:,24:30].sum(dim=1)==i) & (~different_mask) )[0]
        same_index_list.append(index)

    #如果比 切换状态数量/5少 则全部采取，否则随机采样 切换状态数量/5 的样本
    sampled_same_index_list=[]
    for i in range(4):
        if same_index_list[i].shape[0]<int(different_index.shape[0]/4):
            sampled_same_index_list.append(same_index_list[i])
        else:
            sampled_index=torch.randperm(int(different_index.shape[0]/4))
            sampled_same_index_list.append(same_index_list[i][sampled_index])

    sampled_same_index=torch.cat([sampled_same_index_list[i] for i in range(4)],dim=0)
    print("sampled_same_index.shape=",sampled_same_index.shape)
    sampled_obs=torch.cat([obs[sampled_same_index],obs[different_index]],dim=0)
    sampled_action=torch.cat([action[sampled_same_index],action[different_index]],dim=0)
    #only contain q_cur q_torque suction_force
    # sampled_part_obs=torch.cat([sampled_obs[:,13:31],sampled_obs[:,49:67],sampled_obs[:,67:73]],dim=1)
    sampled_part_obs=torch.cat([sampled_obs[:,13:31],sampled_obs[:,49:67]],dim=1)
    sampled_part_action=sampled_action[:,24:30]
    

    train_dataset=TensorDataset(sampled_part_obs,sampled_part_action)
    train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
    return train_loader
    # for i in range(4):
    #     print("len ",i,"=",len(sampled_same_index_list[i]))
    #     print(sampled_same_index_list[i][0:10])



file_name=os.getcwd()+"/bag/expert_dataset/"+"2025-05-19,20:47:58-.pt"
train_loader=LoadData(file_name)

"""网络输出动作 损失采用norm(a_e-a_pi)均方差"""
# agent=BC_Agent().to('cuda')
# loss_fn=nn.MSELoss()
# optimizer=optim.Adam(agent.parameters(),lr=agent.lr)
# num_epochs=10
# for epoch in range(num_epochs):
#     agent.train()
#     running_loss=0.0
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()

#         outputs=agent(inputs)

#         loss=loss_fn(outputs,targets)

#         loss.backward()

#         optimizer.step()

#         running_loss+=loss.item()

#     print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")
#     print("train_loader_len=",len(train_loader))

# torch.save(agent.state_dict(),os.getcwd()+'/bag/agent/BC_model.pth')
# print("Model saved to BC_model.pth")

"""网络输出动作分布 损失采用E[-log(pi(a_e|s))]"""
agent=BC_Agent_Dist().to('cuda')
optimizer=optim.Adam(agent.parameters(),lr=agent.lr)

def neg_log_prob(miu:torch.Tensor,log_sigma:torch.Tensor,actions:torch.Tensor)->torch.Tensor:
    # print("sigma=",torch.exp(log_sigma)[0:5])

    res=(0.5*math.log(math.pi*2) + log_sigma+ 0.5*(actions-miu)**2/torch.exp(2.0*log_sigma))
    # print("res stats:", res.min().item(), res.max().item(), res.mean().item())
    return res.sum(dim=1).mean()
    # dist=Normal(miu,torch.exp(log_sigma))
    # log_prob=dist.log_prob(actions)
    # print("log_prob stats:", log_prob.min().item(), log_prob.max().item(), log_prob.mean().item())
    # print("log_sigma stats:", log_sigma.min().item(), log_sigma.max().item(), log_sigma.mean().item())
    # return -log_prob.sum(dim=1).mean()



num_epochs=30
for epoch in range(num_epochs):
    agent.train()
    running_loss=0.0
    for obs, actions in train_loader:
        optimizer.zero_grad()

        miu,log_sigma=agent(obs)
        loss=neg_log_prob(miu,log_sigma,actions)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")
    print("train_loader_len=",len(train_loader))

torch.save(agent.state_dict(),os.getcwd()+'/bag/agent/BC_model_dist.pth')
print("Model saved to BC_model_dist.pth")

# norm_dist=Normal(0,1)
# print(-norm_dist.log_prob(torch.tensor([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])))


