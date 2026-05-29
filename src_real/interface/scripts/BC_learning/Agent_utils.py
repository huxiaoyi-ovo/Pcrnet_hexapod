# from hex_gym_env import HexEnv
# from curv_adapt_multi import CurvAdapt_multi
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from hex_cfg import HexCfg
import random
import math
from typing import Tuple
def RandomCommand(command:torch.Tensor,command_count,command_count_max):
    if command_count.sum()==0:
        #at the beginning of command generation, resample all commands
        resample_index=torch.arange(command.shape[0],dtype=torch.int32,device='cuda:0')  
        command_count.fill_(1)
    else:
        resample_mask=command_count>=command_count_max
        command_count[resample_mask]=0
        command_count[~resample_mask]+=1
        command_count_max[resample_mask]=torch.randint(200,400,(resample_mask.sum(),),dtype=torch.int32,device='cuda:0')
        resample_index=torch.where(resample_mask)[0]
    direc=-1.0 if random.random()<0.5 else 1.0
    # command[resample_index,4]=direc*((torch.rand(len(resample_index),dtype=torch.float32,device='cuda:0')-0.5)*0.4+HexCfg.max_vec.y)
    # command[resample_index,2]=direc*((torch.rand(len(resample_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y)

    #0~15: vec_x; 16~31: vec_y; 32~47: omega_move; 48~63:vec_x vec_y; 64~79:  vec_y omega
    vec_x_index=resample_index[resample_index<24]
    vec_y_index=resample_index[(resample_index>=24)&(resample_index<48)]
    vec_omega_index=resample_index[(resample_index>=48)&(resample_index<72)]
    # vec_xy_index=resample_index[(resample_index>=48)&(resample_index<64)]
    # vec_y_omega_index=resample_index[(resample_index>=64)&(resample_index<80)]

    # vec_x_index=resample_index[resample_index<4]
    # vec_y_index=resample_index[(resample_index>=4)&(resample_index<8)]
    # vec_omega_index=resample_index[(resample_index>=8)&(resample_index<16)]
    # vec_xy_index=resample_index[(resample_index>=16)&(resample_index<24)]
    # vec_y_omega_index=resample_index[(resample_index>=24)&(resample_index<32)]

    command[vec_x_index,1]=(torch.rand(len(vec_x_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x
    command[vec_y_index,2]=(torch.rand(len(vec_y_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y
    command[vec_omega_index,4]=(torch.rand(len(vec_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move
    # command[vec_xy_index,1]=(torch.rand(len(vec_xy_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x*0.5
    # command[vec_xy_index,2]=(torch.rand(len(vec_xy_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y*0.5
    # command[vec_y_omega_index,2]=(torch.rand(len(vec_y_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y*0.5
    # command[vec_y_omega_index,4]=(torch.rand(len(vec_y_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move*0.5
    command[resample_index,1:5]=direc*command[resample_index,1:5]

    # command[:,2]=1.5
    # 0~7: vec_x; 8~15: vec_y; 16~23: omega_move 
    # 24~31: vec_x vec_y; 32~39: vec_x omega_move; 40~47: vec_y omega_move
    # 48~55: vec_x vec_y omega_move
    # vec_x_index=resample_index[resample_index<8]
    # vec_y_index=resample_index[(resample_index>=8)&(resample_index<16)]
    # vec_omega_index=resample_index[(resample_index>=16)&(resample_index<24)]
    # vec_xy_index=resample_index[(resample_index>=24)&(resample_index<32)]
    # vec_x_omega_index=resample_index[(resample_index>=32)&(resample_index<40)]
    # vec_y_omega_index=resample_index[(resample_index>=40)&(resample_index<48)]
    # vec_xyomega_index=resample_index[(resample_index>=48)&(resample_index<56)]
    # command[vec_x_index,1]=(torch.rand(len(vec_x_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x
    # command[vec_y_index,2]=(torch.rand(len(vec_y_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y
    # command[vec_omega_index,4]=(torch.rand(len(vec_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move
    # command[vec_xy_index,1]=(torch.rand(len(vec_xy_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x
    # command[vec_xy_index,2]=(torch.rand(len(vec_xy_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y
    # command[vec_x_omega_index,1]=(torch.rand(len(vec_x_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x
    # command[vec_x_omega_index,4]=(torch.rand(len(vec_x_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move
    # command[vec_y_omega_index,2]=(torch.rand(len(vec_y_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y
    # command[vec_y_omega_index,4]=(torch.rand(len(vec_y_omega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move
    # command[vec_xyomega_index,1]=(torch.rand(len(vec_xyomega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.x
    # command[vec_xyomega_index,2]=(torch.rand(len(vec_xyomega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.y
    # command[vec_xyomega_index,4]=(torch.rand(len(vec_xyomega_index),dtype=torch.float32,device='cuda:0')-0.5)*0.2+HexCfg.max_vec.omega_move
    # command[resample_index,1:4]=direc*command[resample_index,1:4]

def CatObs(env,command,env_nums)->torch.Tensor:
    obs_root,_=env.GetRootStates() #4+3+3+3=13 0-13
    obs_q=(env.dof_pos.view(env_nums,6,7)[...,0:3]).reshape(-1,18) #18 13-31
    obs_q_dot=(env.dof_vel.reshape(env_nums,6,7)[...,0:3]).reshape(-1,18)#18 31-49
    obs_torq=(env.torques.view(env_nums,6,7)[...,0:3]).reshape(-1,18)#18 49-67
    obs_force=env.suction_force_z#6 67-73
    obs_command=command[:,1:5]#4 73-77

    obs_last_q=env.last_command_dof_pos.view(env_nums,6,7)[...,0:4].reshape(-1,24)# 24 77-101
    obs_last_ad=env.last_command_adhesions.clone()# 6 101-107
    # obs=torch.cat([obs_root,obs_q,obs_q_dot,obs_torq,obs_force,obs_command,obs_last_q,obs_last_ad],dim=1) #77+30=107
    # obs=torch.cat([obs_root,obs_q,obs_q_dot,obs_torq,obs_command,obs_last_q,obs_last_ad],dim=1) #71+30=101
    obs=torch.cat([obs_root,obs_q,obs_q_dot,obs_torq,obs_force,obs_command],dim=1) #77
    # obs=torch.cat([obs_root,obs_q,obs_q_dot,obs_torq,obs_command],dim=1) #71
    # obs=torch.cat([obs_q,obs_q_dot,obs_torq,obs_command],dim=1) #58
    return obs 

def CatAction(expert,env_nums):
    action_q=expert.q_des.reshape(-1,24) #24
    action_ad=expert.adhesions.clone() #6
    action=torch.cat([action_q,action_ad],dim=1) #24+6=30
    # return action_q
    return action 



class BC_Agent(nn.Module):
    def __init__(self, input_dim=71, output_dim=30, hidden_dim1=256, hidden_dim2=256, hidden_dim3=256, lr=0.0005):
    # def __init__(self, input_dim=71, output_dim=30, hidden_dim1=512, hidden_dim2=128, lr=0.0005):
    # def __init__(self, input_dim=58, output_dim=24, hidden_dim1=512, hidden_dim2=128, lr=0.0005):
    # def __init__(self, input_dim=58, output_dim=24, hidden_dim1=256, hidden_dim2=256, hidden_dim3=256, lr=0.0005):
        super(BC_Agent,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim1)
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        # self.fc3=nn.Linear(hidden_dim2,output_dim)
        self.fc3=nn.Linear(hidden_dim2,hidden_dim3)
        self.fc4=nn.Linear(hidden_dim3,output_dim)
        self.droupout=nn.Dropout(p=0.5)
        self.lr=lr
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.droupout(x)
        x=F.relu(self.fc2(x))
        x=self.droupout(x)
        # x=self.fc3(x)
        x=F.relu(self.fc3(x))
        x=self.droupout(x)
        x=self.fc4(x)
        return x

class Gait_Agent(nn.Module):
    def __init__(self, input_dim=42,output_dim=6,hidden_dim1=128,hidden_dim2=64,lr=0.0005):
        super(Gait_Agent,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim1)
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        self.fc3=nn.Linear(hidden_dim2,output_dim)
        self.droupout=nn.Dropout(p=0.5)
        self.lr=lr
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.droupout(x)
        x=F.relu(self.fc2(x))
        x=self.droupout(x)
        x=self.fc3(x)
        return x

#输出为30个动作正态分布的均值和方差
class BC_Agent_Dist(nn.Module):
    def __init__(self,input_dim=77,output_dim=30,hidden_dim1=256,hidden_dim2=256,hidden_dim3=256,lr=0.0005):
        super(BC_Agent_Dist,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim1)
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        self.fc3=nn.Linear(hidden_dim2,hidden_dim3)
        self.miu_linear=nn.Linear(hidden_dim3,output_dim)
        # self.log_std_linear=nn.Linear(hidden_dim3,output_dim)
        self.log_std=torch.nn.Parameter(torch.ones(output_dim)*math.log(0.5))
        self.droupout=nn.Dropout(p=0.5)
        self.lr=lr

        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        # x=self.droupout(x)
        x=F.relu(self.fc2(x))
        # x=self.droupout(x)
        x=F.relu(self.fc3(x))
        # x=self.droupout(x)
        miu=self.miu_linear(x)

        log_std=torch.clamp(self.log_std,min=math.log(0.5),max=math.log(2.0))

        return miu,log_std

    def sample_action(self,obs,deterministic=False):
        miu,log_std=self.forward(obs)
        if deterministic:
            return miu
        else:
            return miu+torch.exp(log_std)*torch.randn_like(miu)



#Low Level Controller; Including 1 critic 1 actor 1 base_obs_encoder 1 vec_estimator 1 terrain_encoder 
class LLC(nn.Module):
    def __init__(self):
        super(LLC,self).__init__()
        # raw obs include: 
        # 1. base(76) [quat(4), angular_velocity(3), linear_acc(3), q(18), q_dot(18), q_torque(18), suction_force(6), command(6)] 
        # 2. priviliged(6) [linear_velocity(3), gravity(3)]
        # 3. terrain(32) 200 is the number of terrain points
        self.net_dict=nn.ModuleDict()

        # define priviliged estimator input:76 output:6 and loss function
        self.priv_estimator=nn.Sequential(
            nn.Linear(76,128),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,6))
        self.net_dict['priv_estimator']=self.priv_estimator
        self.priviliged_loss_fn=nn.MSELoss()

        # define terrain encoder input:n*3 output:256
        self.terrain_feature_extractor=nn.Sequential(
            nn.Conv1d(3,64,1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,128,1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024),nn.ReLU())
        self.terrain_post_mlp=nn.Sequential(
            nn.Linear(1024,512),nn.ReLU(),
            nn.Linear(512,256))

        # define critic input:82 output:1
        self.critic=nn.Sequential(
            nn.Linear(82,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,1))
        self.net_dict['critic']=self.critic
        
        # define actor input:82 output:30
        self.actor=nn.Sequential(
            nn.Linear(82,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,30))
        self.actor_log_std=nn.Parameter(torch.ones(30)*math.log(0.5))
        self.net_dict['actor']=self.actor

        # define optimizer
        self.lr_list=[1e-3,1e-3,1e-3]
        self.optimizer_dict={name:Adam(net.parameters(),lr=lr)
                             for (name,net),lr in zip(self.net_dict.items(),self.lr_list)}
        self.optimizer_dict['actor'].add_param_group({'params':self.actor_log_std})

    def _encode_terrain(self,terrain_obs:torch.Tensor)->torch.Tensor:
        # terrain_obs: B*n*3
        terrain_obs=terrain_obs.transpose(1,2) #B*3*n
        feature=self.terrain_feature_extractor(terrain_obs) #B*1024*n
        feature=torch.max(feature,dim=2)[0] #B*1024
        feature=self.terrain_post_mlp(feature) #B*256
        return feature

    # def _encode_obs(self,base_obs:torch.Tensor,terrain_obs:torch.Tensor)->torch.Tensor:
    #     #en: encoded
    #     # en_terrain_obs=self._encode_terrain(terrain_obs)
    #     priv_obs=self.priv_estimator(base_obs)
    #     # en_obs=torch.cat([base_obs,priv_obs,en_terrain_obs]) #B*(73+6+256) 335
    #     en_obs=torch.cat([base_obs,priv_obs]) #B*(73+6) 79
    #     return en_obs
    
    def _encode_obs(self,base_obs:torch.Tensor,priv_obs:torch.Tensor,terrain_obs:torch.Tensor)->torch.Tensor:
        en_obs=torch.cat([base_obs,priv_obs],dim=1)
        return en_obs

    def _train_vec_estimator(self):
        pass

    def get_v_a(self,base_obs:torch.Tensor,priv_obs:torch.Tensor,terrain_obs:torch.Tensor,deterministic=False)->Tuple[torch.Tensor,torch.Tensor]:
        obs_encoded=self._encode_obs(base_obs,priv_obs,terrain_obs)
        action_mean=self.net_dict['actor'](obs_encoded)
        value=self.critic(obs_encoded).detach()
        if deterministic:
            a=action_mean
        else:
            self.actor_log_std=torch.clamp(self.actor_log_std,min=math.log(0.5),max=math.log(2.0))
            a=action_mean+torch.exp(self.actor_log_std)*torch.randn_like(action_mean)
        return value,a




        

class EGPO_Agent(nn.Module):
    def __init__(self, input_dim=75,hidden_dim1=512,hidden_dim2=256,hidden_dim3=128,output_dim=18):
    # def __init__(self, input_dim=75,hidden_dim1=256,hidden_dim2=256,hidden_dim3=256,output_dim=18):
        super(EGPO_Agent,self).__init__()
        layers=[nn.Linear(input_dim,hidden_dim1),nn.ELU(alpha=1.0),
                nn.Linear(hidden_dim1,hidden_dim2),nn.ELU(alpha=1.0),
                nn.Linear(hidden_dim2,hidden_dim3),nn.ELU(alpha=1.0),
                nn.Linear(hidden_dim3,output_dim)]
        # layers=[nn.Linear(input_dim,hidden_dim1),nn.ReLU(),
        #         nn.Linear(hidden_dim1,hidden_dim2),nn.ReLU(),
        #         nn.Linear(hidden_dim2,hidden_dim3),nn.ReLU(),
        #         nn.Linear(hidden_dim3,output_dim)]        
        self.actor = nn.Sequential(*layers)


    def forward(self,x):
        return self.actor(x)
    # def act(self,obs):
    #     return self.actor(obs)  

#管理obs历史,负责拼接成LSTM需要的输入形式
#不需要维护done,在手柄控重设为初始化时,重置valid_hist_index
class StorageMemory():
    def __init__(self,num_hist=20,obs_shape=[67+12],device='cpu'):
        self.num_hist=num_hist
        self.device=device

        self.obs_hist=torch.zeros(num_hist,*obs_shape,device=self.device)
        self.mask=torch.zeros(num_hist,dtype=torch.bool,device=self.device)
        self.valid_hist_index=num_hist #有效索引是最后一位之后

    def update_obs_hist(self,obs):
        self.obs_hist[:-1,...]=self.obs_hist[1:,...].clone()
        self.obs_hist[-1].copy_(obs)
        if self.valid_hist_index>0:
            self.valid_hist_index -= 1


    def get_current_obs_hist(self):
        #返回obs维度为 H*D 10*67 ,返回mask维度为H 10
        #valid_hist_index之后所有值都是有效值
        
        self.mask[self.valid_hist_index:]=True

        return self.obs_hist, self.mask #
    def reset(self):
        self.valid_hist_index=self.num_hist
        self.mask.fill_(False)
    




    pass
class ActorCriticEncoder(nn.Module):
    def __init__(self,
                 num_obs=67,
                 num_actions=18,
                 actor_hidden_dims=[512,256,128]
                 ):
        super(ActorCriticEncoder,self).__init__()
        num_terrain_obs_latent=32
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs+12+num_terrain_obs_latent, actor_hidden_dims[0]))
        actor_layers.append(nn.ELU())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims)-1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]))
                actor_layers.append(nn.ELU())
        self.actor = nn.Sequential(*actor_layers)

        self.obs_vgf_estimator = nn.Sequential(
            nn.Linear(num_obs, 64),nn.ReLU(),
            nn.Linear(64, 32),nn.ReLU(),
            nn.Linear(32, 12)
        )
        # obs history -> terrain hidden vector encoder
        # 采用LSTM网络,输入：num_actor_obs + 12
        self.lstm = nn.LSTM(input_size=num_obs+12,
                                    hidden_size=64,
                                    num_layers=2,
                                    batch_first=True,
                                    bidirectional=True)
        self.lstm_fc = nn.Linear(128,32)
                
    def LSTM_encode(self, obs_batch, mask_batch):

        # obs_batch: [seq_len, num_obs+12]
        # mask_batch: [seq_len]
        obs_batch = obs_batch.unsqueeze(0)
        mask_batch = mask_batch.unsqueeze(0)
        # 由于数据特点，所有输入进来的obs_batch都是头部填充0，而lstm的数据处理要求尾部填充0
        # 因此采用以下方式：先把obs_batch和 mask_batch的 seq_len维度逆转，再采用双向的lstm
        # 这样就变成了尾部填充，同时lstm输出也同时包含正向和逆向，由于特征向量同时包含两部分，所以是等效的
        obs_batch_flipped = obs_batch.flip(dims=[1])

        #查看翻转情况，做出判断
        # print("obs_batch\n",obs_batch[0,:,0])
        # print("obs_batch_flipped\n",obs_batch_flipped[0,:,0])
        # mask_batch_flipped = mask_batch.flip(dims=[1])
        # print("mask_batch\n",mask_batch[0])
        # print("mask_batch_flipped\n",mask_batch_flipped[0])


        lengths = torch.sum(mask_batch,dim=1).to('cpu').to(torch.long) #pytorch接口要求
        # print("length zeros index=",torch.where(lengths==0)[0])
        padded_obs_batch =nn.utils.rnn.pack_padded_sequence(obs_batch_flipped, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) =self.lstm(padded_obs_batch)
        return self.lstm_fc( torch.cat([h_n[-2],h_n[-1]],dim=1) ).squeeze(0)
    def act_interence(self,obs,storage:StorageMemory):
        obs = self.Encode_obs(obs,storage)
        return self.actor(obs)
    
    def Encode_obs(self,obs,storage:StorageMemory):
        obs_vgf_estimates=self.obs_vgf_estimator(obs)
        # obs_vgf_estimates.fill_(0.0)
        obs_splice=torch.cat([obs,obs_vgf_estimates])
        storage.update_obs_hist(obs_splice)
        obs_hist,mask=storage.get_current_obs_hist()
        # print("obs_hist=\n",obs_hist[:,0])
        # print("mask\n",mask)
        obs_terrain_lstm=self.LSTM_encode(obs_hist,mask)
        # obs_terrain_lstm.fill_(0.0)
        return torch.cat([obs_splice,obs_terrain_lstm])


if __name__=='__main__':
    import rospy
    from std_msgs.msg import Float32MultiArray
    rospy.init_node("agent_testing")
    pub=rospy.Publisher("/test_agent",Float32MultiArray,queue_size=10)
    msg=Float32MultiArray()

    device='cuda:0'
    agent=ActorCriticEncoder()
    agent.to(device)

    # print(agent.state_dict())
    # storage=StorageMemory(device=device)


    # i=0
    # while not rospy.is_shutdown():
    #     if i==2 or i==8:
    #         storage.reset()
    #         obs=torch.zeros(67,dtype=torch.float,device=device)
    #         print("reset---------------------")
        
    #     obs=torch.rand(67,dtype=torch.float,device=device)
    #     actions=agent.act_interence(obs,storage)
    #     i+=1
    #     if i==10:
    #         exit(0)

    