#添加以下encoder
#obs_priv_estimator: MLP: 从obs观测推算基座线速度v，重力加速度分量g
#obs_terrain_encoder: TCN: 从历史obs,v,g编码地形特征向量
#terrain_encoder: MLP: 从地形高度编码特征向量

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticEncoder(nn.Module):
    #num_obs 不包含 v g f 和地形 特征向量
    #地形向量固定为 11(rows)*17(cols) 
    def __init__(self, num_obs,
                       num_actions,
                       actor_hidden_dims=[256,256,256],
                       critic_hidden_dims=[256,256,256],
                       activation='elu',
                       init_noise_std = 1.0):
        super(ActorCriticEncoder,self).__init__()
        

        activation = get_activation(activation)

        num_terrain_obs_latent = 32


        # actor
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs+30+num_terrain_obs_latent, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims)-1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        self.distributions = None
        self.std = nn.Parameter(init_noise_std*torch.ones(num_actions))
        # critic
        critic_layers = []
        critic_layers.append(nn.Linear(num_obs+30+num_terrain_obs_latent, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims)-1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l+1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # actor_obs -> [v,g,contact_force] estimator
        # 采用固定的MLP网络
        self.actor_obs_priv_estimator = nn.Sequential(
            nn.Linear(num_obs, 64),nn.ReLU(),
            nn.Linear(64, 32),nn.ReLU(),
            nn.Linear(32, 30)
        )
        # obs history -> terrain hidden vector encoder
        # 采用LSTM网络,输入：num_actor_obs + 12
        self.lstm = nn.LSTM(input_size=num_obs+30,
                                    hidden_size=64,
                                    num_layers=2,
                                    batch_first=True,
                                    bidirectional=True)
        self.lstm_fc = nn.Linear(128,32)
        

        # terrain height -> terrain hidden vector encoder
        # terrain encoder输入要求应该为[batch_size,1,heigh 11,width 13]
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(1,4,3,padding=1,stride=1),nn.ReLU(), #[b, 4, 11,13]
            nn.Conv2d(4,8,3,padding=1,stride=2),nn.ReLU(), #[b, 8, 6,7]
            nn.Flatten(),
            nn.Linear(8*6*7,num_terrain_obs_latent)
        )
        # self.terrain_encoder = nn.Sequential(
        #     nn.Linear(11*13,256),nn.ReLU(),
        #     nn.Linear(256,128),nn.ReLU(),
        #     nn.Linear(128,num_terrain_obs_latent)
        # )
    def LSTM_encode(self, obs_batch, mask_batch):
        # obs_batch: [batch_size, seq_len, num_obs+12]
        # mask_batch: [batch_size, seq_len]
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


        lengths = torch.sum(mask_batch,dim=1).to('cpu').to(torch.long) #'cpu'是pytorch接口要求，这里只计算mask中有效数量，因此不需要翻转
        # print("length zeros index=",torch.where(lengths==0)[0])
        padded_obs_batch =nn.utils.rnn.pack_padded_sequence(obs_batch_flipped, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) =self.lstm(padded_obs_batch)
        return self.lstm_fc( torch.cat([h_n[-2],h_n[-1]],dim=1) )
    
    @property
    def action_mean(self):
        return self.distributions.mean
    @property
    def action_std(self):
        return self.distributions.stddev
    @property
    def entropy(self):
        return self.distributions.entropy().sum(dim=-1)
    
    def encode_terrain(self,terrain_batch):
        terrain_batch = terrain_batch.reshape(-1,1,11,13)
        return self.terrain_encoder(terrain_batch)


    def update_distributions(self,obs):
        mean = self.actor(obs)
        self.distributions = Normal(mean, mean*0. + self.std)

    def act(self,obs,obs_terrain_latent):
        self.update_distributions(torch.cat([obs,obs_terrain_latent],dim=-1))
        return self.distributions.sample()
    
    def get_actions_log_prob(self,actions):
        return self.distributions.log_prob(actions).sum(dim=-1)
    
    def act_inference(self,obs,obs_terrain_latent):
        return self.actor(torch.cat([obs,obs_terrain_latent],dim=-1))
    
    def evaluate(self,obs,obs_terrain_latent):
        return self.critic(torch.cat([obs,obs_terrain_latent],dim=-1))
    
    def reset(self):
        #lstm会默认重置参数为0，不需要显示调用
        pass

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None