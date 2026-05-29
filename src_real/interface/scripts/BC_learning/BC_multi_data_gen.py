from hex_gym_env import HexEnv
from curv_adapt_multi import CurvAdapt_multi
from hex_cfg import HexCfg
from datetime import datetime
import torch
import time
import os
from BC_learning.Agent_utils import RandomCommand, CatObs,CatAction,BC_Agent


# import rospy
# from std_msgs.msg import Float64MultiArray

import pandas as pd
import numpy as np

def continue_condition(render_mode,hex_env:HexEnv):
    global sim_paused
    if render_mode=="human":
        for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
            if evt.action=='pause' and evt.value>0:
                sim_paused= not sim_paused
            if evt.action=='reset' and evt.value>0:
                print("reset env")
                hex_env.ResetIdx(env_ids)
                command[env_ids,0].fill_(1)
                curv_multi.ProcessCommand(command)
        return not hex_env.gym.query_viewer_has_closed(hex_env.viewer)
    else:
        return True


if __name__=="__main__":
    #create ros related to plot curve
    # rospy.init_node('BC_play',anonymous=True)
    # pub_curve=rospy.Publisher('/joint_pos',Float64MultiArray,queue_size=10)
    # _pos_msg=Float64MultiArray()# [expert_des joint0 agent_des joint0]

    #create envs and experts(curv_multi)
    # env_nums=64
    env_nums=72
    # env_nums=1
    hex_cfg=HexCfg("hex_cfg.yaml")
    hex_env=HexEnv(hex_cfg,env_nums,'cuda:0')
    hex_curv_list=[] #type: list[CurvAdapt_multi]

    joint_pos_cur_env=hex_env.dof_pos.view(env_nums,6,7)[...,0:3]
    joint_torques_cur_env=hex_env.torques.view(env_nums,6,7)[...,0:3]
    # joint_pos_des_env= torch.zeros((env_nums,6,7),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env = torch.zeros((env_nums,42),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env_part = joint_pos_des_env.view(env_nums,6,7)[...,0:4]
    adhesions_env = torch.zeros(env_nums,6,dtype=torch.bool,device='cuda:0')
    suction_force_env = hex_env.suction_force_z 
    reset_buff=torch.zeros(env_nums,dtype=torch.bool,device='cuda:0')


    curv_multi=CurvAdapt_multi("yaml_file","cuda:0",env_nums,
                               joint_pos_des_env_part,
                               joint_pos_cur_env,
                               joint_torques_cur_env,
                               adhesions_env,
                               suction_force_env)
    #create agent and load model
    # agent=BC_Agent().to('cuda:0')
    # agent.load_state_dict(torch.load(os.getcwd()+'/bag/one_policy_agent.pth',weights_only=True))
    # agent.eval()
    # min_max_dict=torch.load(os.getcwd()+'/bag/dagger_min_max.pt',weights_only=True)
    # obs_min=min_max_dict['obs_min']
    # obs_max=min_max_dict['obs_max']
    # action_min=min_max_dict['action_min']
    # action_max=min_max_dict['action_max']
    #
    agent_expert=0

    #create command
    sim_paused=False
    # env_nums*[set_init vx vy vz omega]
    command = torch.zeros(env_nums,5,dtype=torch.float32,device='cuda:0') 
    command_count_max=torch.randint(50,200,(env_nums,),dtype=torch.int32,device='cuda:0')
    command_count=torch.zeros(env_nums,dtype=torch.int32,device='cuda:0')
    command[:,0]=1
    # command[:,1]=0.2
    # command[:,2]=0.3

    #create files to save data
    # joint_data={'e_t':[],'e_k':[],'e_a':[],'e_f':[],'a_t':[],'a_k':[],'a_a':[],'a_f':[]}

    file_root=os.getcwd()+'/bag/expert_dataset/'
    #set batch size of a single file, set batches to loop
    batch_size=40000
    # file_num=4
    batch_index=0
    file_index=0
    obs_list=[]
    action_list=[]

    env_ids=torch.arange(env_nums,dtype=torch.int32,device='cuda:0')
    # reset the envs
    for i in range(5):
        hex_env.ResetIdx(env_ids)
        time.sleep(0.1)
    # start simulation
    while continue_condition(hex_cfg.render_mode,hex_env):
        if sim_paused:
            hex_env.gym.step_graphics(hex_env.sim)
            hex_env.gym.draw_viewer(hex_env.viewer,hex_env.sim,True)
        else:
            """record expert and agent action to .csv file"""
            """
            # #reset every 5s
            # if batch_index%500==0:
            #     #choose expert or agent action 
            #     agent_expert=1-agent_expert
            #     for i in range (5):
            #         hex_env.ResetIdx(env_ids)
            #         curv_multi.ResetJoint(env_ids)
            #         time.sleep(0.1)
            #     #set robot initial state
            #     for _ in range(100):
            #         curv_multi.ProcessCommand(command)
            #         curv_multi.set_init_done.zero_()
            #         hex_env.SetJointPos(joint_pos_des_env)
            #         hex_env.SetAdhesions(adhesions_env)
            #         hex_env.Simulate()
            #     curv_multi.set_init_done.fill_(200) #

            #     file_index+=1
            #     if file_index>file_num and file_index%2!=0:
            #         df=pd.DataFrame(joint_data)
            #         df.to_csv(file_root+'joint_data.csv')
            #         print("save file to",file_root+'joint_data.csv')
            # #     #set random start state
            # #     random_joint=(torch.rand((env_nums,6,3),dtype=torch.float32,device='cuda:0')-0.5)*0.7
            # #     curv_multi.q_init[...,0:3]=curv_multi.q_init[...,0:3]+random_joint
            # #     curv_multi.adhesions[:]=torch.rand((env_nums,6),device='cuda:0')<0.5
            # #     curv_multi._GetFootAngle(curv_multi.q_init_flat)
            # #     curv_multi.kin.ForwardKin(curv_multi.q_init_flat,curv_multi.B_e_init.view(-1,4))
            # #     random_stance_group=(torch.rand(env_nums,dtype=torch.float32,device='cuda:0')<0.5).to(torch.int32)
            # #     stance_index=(curv_multi.gaits_groups[random_stance_group]).view(-1,3)
            # #     curv_multi.gaits.fill_(0)
            # #     curv_multi.gaits[torch.arange(env_nums,dtype=torch.int32,device='cuda:0').unsqueeze(1),stance_index]=1
            #     # print("reset gaits\n",curv_multi.gaits[:10])
            # if agent_expert==0:
            #     # RandomCommand(command,command_count,command_count_max)
            #     moving_env_mask=curv_multi.ProcessCommand(command)
            #     q_des_0=curv_multi.q_des[0].detach().cpu().numpy()
            #     joint_data['e_t'].append(q_des_0[0,0])
            #     joint_data['e_k'].append(q_des_0[0,1])
            #     joint_data['e_a'].append(q_des_0[0,2])
            #     joint_data['e_f'].append(q_des_0[0,3])
            #     # print("use expert action")
            # if agent_expert==1:
            #     obs=CatObs(hex_env,command,env_nums)
            #     obs=(obs-obs_min)/(obs_max-obs_min)
            #     action=agent(obs)
            #     action=action*(action_max-action_min)+action_min
            #     joint_pos_des_env_part[:]=action[:,0:24].reshape(env_nums,6,4)
            #     adhesions_env[:]=action[:,24:30].reshape(env_nums,6)>0.6
            #     action_=action.detach().cpu().numpy()
            #     joint_data['a_t'].append(action_[0,0])
            #     joint_data['a_k'].append(action_[0,1])  
            #     joint_data['a_a'].append(action_[0,2])
            #     joint_data['a_f'].append(action_[0,3])
            #     print("agent action",action_[0,0],action_[0,1],action_[0,2],action_[0,3])

            # hex_env.SetJointPos(joint_pos_des_env)
            # hex_env.SetAdhesions(adhesions_env)
            # hex_env.Simulate()
            # batch_index+=1
            # # if batch_index%100==0:
            # #     print("adhesions\n",adhesions_env)
            """


            """generate expert data to .pt file"""
            if batch_index%2000==0:
                for i in range (5):
                    hex_env.ResetIdx(env_ids)
                    time.sleep(0.1)
                #set robot initial state
                for _ in range(50):
                    command[:,0].fill_(1)
                    curv_multi.ProcessCommand(command)
                    hex_env.SetJointPos(joint_pos_des_env)
                    hex_env.SetAdhesions(adhesions_env)
                    hex_env.Simulate()
                curv_multi.set_init_done.fill_(True)
                print("===============Finish init, start movine=================")
                command[:,0].fill_(0)
            RandomCommand(command,command_count,command_count_max)
            curv_multi.ProcessCommand(command)
            # print("B_e_des\n",curv_multi.B_e_des[0],"\nq_e_des\n",curv_multi.q_des[0])
            # print("\n")
            # for i,rb_name in enumerate(hex_env.rb_names):
            #     if rb_name=='l_lb_toe':
            #         print(rb_name,"=",hex_env.contact_forces[0,i,:])

            obs=CatObs(hex_env,command,env_nums)
            action=CatAction(curv_multi,env_nums)
            obs_list.append(obs.clone())
            action_list.append(action.clone())
            #print to examine data
            # _joint_cur=obs[0,13:31].reshape(6,3)
            # pos=torch.zeros(6,3,dtype=torch.float32,device='cuda:0')
            # curv_multi.kin.ForwardKin(_joint_cur,pos)
            # print(">>>>>>>>>>>>>>>>>>>In data record<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # print("this command \n","\n q_des=\n", curv_multi.q_des,"\n adhesions=\n",curv_multi.adhesions)
            # print("last command \n","\n q_des=\n",hex_env.last_command_dof_pos,"\n adhesions=\n",hex_env.last_command_adhesions)
            # print("")
            # print("-----------obs-----------------")
            # print("suction_force=",obs[0,67:73])
            # print("joint cur\n",_joint_cur,"\n B_e_cur\n",pos,"\n suction_force\n",obs[0,67:73])
            # print("joint torques\n",obs[0,49:67].reshape(6,3))
            # print("last q_des\n",obs[0,77:101].reshape(6,4),"\n adhesions\n",obs[0,101:107])
            # print("-----------action------------------")
            # _joint_des=action[0,0:24].reshape(6,4)
            # curv_multi.kin.ForwardKin(_joint_des[:,0:3],pos)
            # print("joint des\n",_joint_des,"\n adhesions\n",action[0,24:30])
            
            #step simulation
            hex_env.SetJointPos(joint_pos_des_env)
            hex_env.SetAdhesions(adhesions_env)
            hex_env.Simulate()
            

            #empty cache
            del obs, action
            torch.cuda.empty_cache()

            batch_index+=1
            if batch_index%2000==0:
                print("save ",batch_index/batch_size*100,"%"," data")
            if batch_index>batch_size:
                # if file_index<=file_num:
                #     if batch_index>=batch_size:
                obs_tensor_batch=torch.cat(obs_list,dim=0)
                action_tensor_batch=torch.cat(action_list,dim=0)
                format_date=datetime.now().strftime('%Y-%m-%d,%H:%M:%S')
                file_name=file_root+format_date+'-.pt'
                torch.save({'obs':obs_tensor_batch,'action':action_tensor_batch},file_name)
                print("save data to file:",file_name)
                exit(0)
            # time.sleep(0.5)


        