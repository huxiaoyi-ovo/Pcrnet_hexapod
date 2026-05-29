from hex_gym_env import HexEnv
from hex_cfg import HexCfg
from kinematic import Kinematic
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math

from BC_learning.Agent_utils import BC_Agent, BC_Agent_Dist,CatObs, RandomCommand

def GetJointPos(joint_pos:torch.Tensor):
    """joint_pos:(env_nums,6,4)"""
    joint_pos[...,0]=0
    joint_pos[...,1]=0.7
    joint_pos[...,2]=-2.14
    joint_pos[:,[1,3],0]=-math.pi/9.0
    joint_pos[:,[0,4],0]=math.pi/9.0
    # LB 0 LF 1 RM 5
    # joint_pos[:,[0,1,5],1]=1.2
    # joint_pos[:,[2,3,4],2]=-2.0
    # random_joint=(torch.rand((env_nums,6,3),dtype=torch.float32,device='cuda:0')-0.5)
    # joint_pos[...,0:3]=joint_pos[...,0:3]+random_joint
    joint_pos_flat=joint_pos.view(-1,4)
    joint_pos_flat[:,3]=-(joint_pos_flat[:,1]+joint_pos_flat[:,2])-math.pi/2.0

def GetAdhesions(adhesions:torch.Tensor):
    pass
    ad_index=torch.tensor([0,1,5],dtype=torch.int32,device='cuda:0')
    adhesions[:,ad_index]=True




if __name__=="__main__":
    #create envs and experts(curv_multi)
    env_nums=72
    # env_nums=1
    hex_cfg=HexCfg("hex_cfg.yaml")
    hex_env=HexEnv(hex_cfg,env_nums,'cuda:0')


    joint_pos_cur_env=hex_env.dof_pos.view(env_nums,6,7)[...,0:3]
    joint_torques_cur_env=hex_env.torques.view(env_nums,6,7)[...,0:3]
    joint_pos_des_env = torch.zeros((env_nums,42),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env_part = joint_pos_des_env.view(env_nums,6,7)[...,0:4]
    adhesions_env = torch.zeros(env_nums,6,dtype=torch.bool,device='cuda:0')
    suction_force_env = hex_env.suction_force_z 
    reset_buff=torch.zeros(env_nums,dtype=torch.bool,device='cuda:0')



    command = torch.zeros(env_nums,5,dtype=torch.float32,device='cuda:0') 
    command_count_max=torch.randint(50,200,(env_nums,),dtype=torch.int32,device='cuda:0')
    command_count=torch.zeros(env_nums,dtype=torch.int32,device='cuda:0')
    # command[:,0]=1
    # command[:,1]=1.0
    # command[:,2]=0.6
    


    # model=BC_Agent().to('cuda:0')
    # # model.load_state_dict(torch.load(os.getcwd()+'/bag/agent/BC_model.pth',weights_only=True))
    # model.load_state_dict(torch.load(os.getcwd()+'/bag/agent/dagger_agent.pth',weights_only=True))
    # model.eval()

    model_dist=BC_Agent_Dist().to('cuda:0')
    # model_dist.load_state_dict(torch.load(os.getcwd()+'/bag/agent/BC_model_dist.pth',weights_only=True))
    model_dist.load_state_dict(torch.load(os.getcwd()+'/bag/agent/dagger_agent_dist.pth',weights_only=True))
    model_dist.eval()
    print("model std:",model_dist.log_std.exp())

    for i in range(5):
        hex_env.ResetIdx(torch.arange(env_nums,dtype=torch.int32,device='cuda:0'))
        time.sleep(0.1)
    GetJointPos(joint_pos_des_env_part)
    GetAdhesions(adhesions_env)

    for _ in range(50):
        hex_env.SetJointPos(joint_pos_des_env)
        hex_env.SetAdhesions(adhesions_env)
        hex_env.Simulate()

    sim_paused=False
    count=0
    while not hex_env.gym.query_viewer_has_closed(hex_env.viewer):
    # while True:
        for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
            if evt.action=='pause' and evt.value>0:
                sim_paused= not sim_paused
            if evt.action=='reset' and evt.value>0:
                print("reset env")
                env_ids=torch.arange(env_nums,dtype=torch.int32,device='cuda:0')
                # env_ids=torch.tensor([0],dtype=torch.int32,device='cuda:0')
                hex_env.ResetIdx(env_ids)
                GetJointPos(joint_pos_des_env_part)
                GetAdhesions(adhesions_env)
                for _ in range(50):
                    hex_env.SetJointPos(joint_pos_des_env)
                    hex_env.SetAdhesions(adhesions_env)
                    hex_env.Simulate()

        if sim_paused:
            hex_env.gym.step_graphics(hex_env.sim)
            hex_env.gym.draw_viewer(hex_env.viewer,hex_env.sim,False)
        else:
            RandomCommand(command,command_count,command_count_max)

            obs=CatObs(hex_env,command,env_nums)
            #action form BC_Agent
            # action=model(obs)
            # joint_pos_des_env_part[:]=action[:,0:24].reshape(env_nums,6,4)
            # adhesions_env[:]=action[:,24:30].reshape(env_nums,6)>0.6
            #action form BC_Agent_Dist
            action=model_dist.sample_action(obs,True)
        

            joint_pos_des_env_part[:]=action[:,0:24].reshape(env_nums,6,4)
            adhesions_env[:]=action[:,24:30].reshape(env_nums,6)>0.6

            hex_env.SetJointPos(joint_pos_des_env)
            hex_env.SetAdhesions(adhesions_env)
            hex_env.Simulate()
            count+=1
            # print("count",count)
            #

        