# Behavioural Cloning (BC) for Transfer Learning
from hex_gym_env import HexEnv
from curv_adapt import CurvAdapt
from hex_cfg import HexCfg
from usr_command import Command
import torch
import time
import random

# def RandomCommand(command_list,env_nums,device):
#     reset_index=command_duration_counts>torch.randint(low=100,high=400,size=(env_nums,),device=device)
    
#     if reset_index.any():
#         for i in reset_index.nonzero(as_tuple=False).tolist():
#             command_list[i].vec[0]=(random.random()*2-1.0)*hex_cfg.max_vec.x
#             command_list[i].vec[1]=(random.random()*2-1.0)*hex_cfg.max_vec.y
#             command_list[i].omega=(random.random()*2-1.0)*hex_cfg.max_vec.omega_move
#     command_duration_counts[~reset_index].add_(1)
#     # print("reset_index:",reset_index)
#     # print("command_duration_counts:",command_duration_counts)
#     command_duration_counts[reset_index]=0

if __name__=="__main__":
    env_nums=2
    hex_cfg=HexCfg("hex_cfg.yaml")
    hex_env=HexEnv(hex_cfg,env_nums,'cuda:0')
    hex_curv_list=[] #type: list[CurvAdapt]
    command_list=[] #type: list[Command]

    joint_pos_cur_env=hex_env.dof_pos.view(env_nums,6,7)[...,0:3]
    joint_torques_cur_env=hex_env.torques.view(env_nums,6,7)[...,0:3]
    joint_pos_des_env= torch.zeros((env_nums,6,7),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env_part=joint_pos_des_env[...,0:4]
    joint_pos_des_env_flat =joint_pos_des_env.view(env_nums,42)
    adhesions_env=torch.zeros(env_nums,6,dtype=torch.bool,device='cuda:0')
    suction_force_env=hex_env.suction_force_z 
    reset_buff=torch.zeros(env_nums,dtype=torch.bool,device='cuda:0')

    for i in range(env_nums):
        joint_pos_des=joint_pos_des_env_part[i]
        joint_pos_cur=joint_pos_cur_env[i]
        joint_torques_cur=joint_torques_cur_env[i]
        adhesions=adhesions_env[i]
        suction_force=suction_force_env[i]
        curv=CurvAdapt("yaml_file","cuda:0",
                    joint_pos_des,
                    joint_pos_cur,
                    joint_torques_cur,
                    adhesions,
                    suction_force)
        hex_curv_list.append(curv)

        command=Command()
        command.set_init=True
        command.vec=[0,0.1,0]
        command.omega=0.0
        command_list.append(command)

    command_duration_counts=torch.zeros(env_nums,dtype=torch.int32,device='cuda:0')

    sim_paused=False
    loop_count=0

    for i in range(5):
        hex_env.ResetIdx(torch.arange(env_nums,dtype=torch.int32,device='cuda:0'))
        time.sleep(0.1)

    while not hex_env.gym.query_viewer_has_closed(hex_env.viewer):
        for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
            if evt.action=='pause' and evt.value>0:
                sim_paused= not sim_paused
            if evt.action=='reset' and evt.value>0:
                print("reset env")
                hex_env.ResetIdx(torch.arange(env_nums,dtype=torch.int32,device='cuda:0'))
        # when and how to reset the envs
        # for i in range(env_nums):
        #     if hex_curv_list[i].ProcessCommand(RandomCommand()):
        #         reset_buff[i]=1
        # reset_idx=reset_buff.nonzero(as_tuple=False).flatten()
        # reset_buff.zero_()
        # if reset_idx.shape[0]>0:
        #     hex_env.ResetIdx(reset_idx)
        #     print("reset env_ids:",reset_idx)
        if sim_paused:
            hex_env.gym.step_graphics(hex_env.sim)
            hex_env.gym.draw_viewer(hex_env.viewer,hex_env.sim,True)
        else:
            # RandomCommand(command_list,env_nums,'cuda:0')
            for i in range(env_nums):
                if loop_count<500:
                    command_list[i].vec=[0,0,0]
                    command_list[i].omega=0.2
                else:
                    command_list[i].vec=[0,0.1,0]
                    command_list[i].omega=0.0
                loop_count+=1


                hex_curv_list[i].ProcessCommand(command_list[i])

            hex_env.SetJointPos(joint_pos_des_env_flat)
            hex_env.SetAdhesions(adhesions_env)
            hex_env.Simulate()
            hex_env.GetRootStates()
            # loop_count+=1
            # if loop_count==20:
            #     exit()
        # time.sleep(0.01)
        


        # print("hex_env.dof_states:",hex_env.dof_states[0:4,0])
        # print("hex_env.dof_pos:",hex_env.dof_pos[0,0:4])
        # print("hex_curv[0].joint_pos_cur[0]:",hex_curv_list[0].joint_pos_cur[0])