from hex_gym_env import HexEnv
from curv_adapt_multi import CurvAdapt_multi
from hex_cfg import HexCfg
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import signal
import matplotlib.pyplot as plt
import math

from BC_learning.Agent_utils import RandomCommand, CatObs, CatAction,BC_Agent, BC_Agent_Dist


def PostExit(singla,frame):
    torch.save(agent.state_dict(),os.getcwd()+'/bag/agent/dagger_agent_dist.pth')
    print("Capture ctrl+c, save model to",os.getcwd()+'/bag/agent/dagger_agent_dist.pth')
    plt.plot(loss_list, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error Over Epochs')
    plt.legend()
    plt.show()
    exit(0)

def neg_log_prob(miu:torch.Tensor, log_sigma:torch.Tensor, actions:torch.Tensor)->torch.Tensor:
    res=(0.5*math.log(math.pi*2) + log_sigma+ 0.5*(actions-miu)**2/torch.exp(2.0*log_sigma))
    return res.sum(dim=1).mean()

if __name__=="__main__":
    #set file path
    file_root=os.getcwd()+'/bag/'
    #create envs
    # env_nums=80
    # env_nums=128
    env_nums=72
    # env_nums=1
    hex_cfg=HexCfg("hex_cfg.yaml")
    hex_env=HexEnv(hex_cfg,env_nums,'cuda:0')

    #create experts (curv_adapt_multi)
    joint_pos_cur_env=hex_env.dof_pos.view(env_nums,6,7)[...,0:3]
    joint_torques_cur_env=hex_env.torques.view(env_nums,6,7)[...,0:3]
    # joint_pos_des_env= torch.zeros((env_nums,6,7),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env = torch.zeros((env_nums,42),dtype=torch.float32,device='cuda:0')
    joint_pos_des_env_part = joint_pos_des_env.view(env_nums,6,7)[...,0:4]
    adhesions_env = torch.zeros(env_nums,6,dtype=torch.bool,device='cuda:0')
    suction_force_env = hex_env.suction_force_z  #env_nums*6
    reset_buff=torch.zeros(env_nums,dtype=torch.bool,device='cuda:0')
    curv_multi=CurvAdapt_multi("yaml_file","cuda:0",env_nums,
                               joint_pos_des_env_part,
                               joint_pos_cur_env,
                               joint_torques_cur_env,
                               adhesions_env,
                               suction_force_env)
    #create experts for state action mark
    _joint_pos_cur=torch.zeros_like(joint_pos_cur_env)
    _joint_torques_cur=torch.zeros_like(joint_torques_cur_env)
    _joint_pos_des_part=torch.zeros_like(joint_pos_des_env_part)
    _adhesions=torch.zeros_like(adhesions_env)
    _suction_force=torch.zeros_like(suction_force_env)
    mark_curv_multi=CurvAdapt_multi("yaml_file","cuda:0",env_nums,
                                    _joint_pos_des_part,
                                    _joint_pos_cur,
                                    _joint_torques_cur,
                                    _adhesions,
                                    _suction_force)
    mark_curv_multi.set_init_done.fill_(True)

    #create agent
    # agent=BC_Agent().to('cuda:0')
    # agent.load_state_dict(torch.load(os.getcwd()+'/bag/agent/BC_model.pth',weights_only=True))
    # agent.train()


    agent=BC_Agent_Dist().to('cuda:0')
    agent.load_state_dict(torch.load(os.getcwd()+'/bag/agent/BC_model_dist.pth',weights_only=True))
    agent.train()

    optimizer=optim.Adam(agent.parameters(),lr=agent.lr)

    #create mixed policy sampler

    #init command buffer
    sim_paused=False
    # env_nums*[set_init vx vy vz omega]
    command = torch.zeros(env_nums,5,dtype=torch.float32,device='cuda:0') 
    command_count_max=torch.randint(50,200,(env_nums,),dtype=torch.int32,device='cuda:0')
    command_count=torch.zeros(env_nums,dtype=torch.int32,device='cuda:0')
    command[:,0]=1


    #
    # obs_min=torch.zeros(77,dtype=torch.float32,device='cuda:0')
    # obs_max=torch.zeros(77,dtype=torch.float32,device='cuda:0')
    # action_min=torch.zeros(30,dtype=torch.float32,device='cuda:0')
    # action_max=torch.zeros(30,dtype=torch.float32,device='cuda:0')
    #set batch size of a single file, set batches to loop
    batch_index=0
    # p=0.99
    p=0.5
    action_list=[]
    obs_list=[]
    # state_traj_dict={'joint_cur':[],'command':[],'B_e_des':[]}
    loss_list=[]
    mode='three_policy'
    # mode='two_policy'
    # mode='one_policy'

    env_reset_ids=torch.arange(env_nums,dtype=torch.int32,device='cuda:0')
    # reset the envs
    for _ in range(5):
        hex_env.ResetIdx(env_reset_ids)
    for _ in range(50):
        command[:,0].fill_(1)
        curv_multi.ProcessCommand(command)
        hex_env.SetJointPos(joint_pos_des_env)
        hex_env.SetAdhesions(adhesions_env)
        hex_env.Simulate()
    curv_multi.set_init_done.fill_(True)
    command[:,0].fill_(0)
    # start simulation 
    # set ctrl+c programmer
    signal.signal(signal.SIGINT,PostExit)

    loop_count=0
    while True:
    # try:
    # while not hex_env.gym.query_viewer_has_closed(hex_env.viewer):
    #     for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
    #         if evt.action=='pause' and evt.value>0:
    #             sim_paused= not sim_paused
    #         if evt.action=='reset' and evt.value>0:
    #             print("reset env")
    #             hex_env.ResetIdx(env_reset_ids)
    #             curv_multi.ResetJoint(env_reset_ids)
        if sim_paused:
            hex_env.gym.step_graphics(hex_env.sim)
            hex_env.gym.draw_viewer(hex_env.viewer,hex_env.sim,True)
        else:
            RandomCommand(command,command_count,command_count_max)
            moving_env_mask=curv_multi.ProcessCommand(command)
            collect_min_max_len=300
            # collect_expert_len=500
            #env set initial ready, begin online tranning
            if batch_index<=collect_min_max_len:
                hex_env.SetJointPos(joint_pos_des_env)
                hex_env.SetAdhesions(adhesions_env)
                hex_env.Simulate()
                obs=CatObs(hex_env,command,env_nums)
                # print("obs[0]\n",obs[0])
                # expert_action=torch.cat([curv_multi.q_des.reshape(-1,24),curv_multi.adhesions.clone()],dim=1)
                expert_action=CatAction(curv_multi,env_nums)
                # #use 1000 samples to acuqire min max for normalization
                # if batch_index==1:
                #     obs_min.copy_(obs.min(dim=0).values)
                #     obs_max.copy_(obs.max(dim=0).values)
                #     action_min.copy_(expert_action.min(dim=0).values)
                #     action_max.copy_(expert_action.max(dim=0).values)
                # else:
                #     obs_min=torch.min(obs.min(dim=0).values,obs_min)
                #     obs_max=torch.max(obs.max(dim=0).values,obs_max)+1e-6
                #     action_min=torch.min(expert_action.min(dim=0).values,action_min)
                #     action_max=torch.max(expert_action.max(dim=0).values,action_max)+1e-6
                # #record min max
                # if batch_index==collect_min_max_len:
                #     min_max_dict={'obs_min':obs_min,'obs_max':obs_max,'action_min':action_min,'action_max':action_max}
                #     torch.save(min_max_dict,file_root+'dagger_min_max.pt')
                #     print("save min max to ",file_root+'dagger_min_max.pt')
                # if batch_index>collect_min_max_len:
                    # obs=(obs-obs_min)/(obs_max-obs_min)
                    # expert_action=(expert_action-action_min)/(action_max-action_min) 
                obs_list.append(obs.clone().detach())
                action_list.append(expert_action.clone().detach())
            else:
                # reset env, start from initial state
                # if batch_index==1001
                for _ in range(5):
                    hex_env.ResetIdx(env_reset_ids)
                for _ in range(50):
                    command[:,0].fill_(1)
                    curv_multi.ProcessCommand(command)
                    hex_env.SetJointPos(joint_pos_des_env)
                    hex_env.SetAdhesions(adhesions_env)
                    hex_env.Simulate()
                curv_multi.set_init_done.fill_(True)
                command[:,0].fill_(0)
                print("========set init=======")
                    # print("set_init_done=",curv_multi.set_init_done)

                # print("policy p\n",pow(p,(batch_index-500)/2.0))
                print("obs_list len:",len(obs_list))
                print("action_list len:",len(action_list))

                for i in range(500):
                    # for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
                    #     print("query viewer action events")
                    #     if evt.action=='pause' and evt.value>0:
                    #         print("sim_paused")
                    #         sim_paused= not sim_paused
                    # while sim_paused:
                    #     hex_env.gym.step_graphics(hex_env.sim)
                    #     hex_env.gym.draw_viewer(hex_env.viewer,hex_env.sim,True)
                    #     for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
                    #         if evt.action=='pause' and evt.value>0:
                    #             sim_paused= not sim_paused
                    #

                    # RandomCommand(command,command_count,command_count_max)
                    # command[:,1]=0.15
                    # command[:,2]=0.1
                    """use mixed policy to continuously interact with env, collect obs, meanwhile mark states with mark expert action"""
                    if mode=='three_policy':
                        #use mixed policy to generate T-step trajectory
                        obs=CatObs(hex_env,command,env_nums)
                        # obs=(obs-obs_min)/(obs_max-obs_min)
                        """I first mark this state with expert action, collect obs and action """
                        if i>10: #mark_curv_multi does not perform well at the beginning
                            # print("------------->In mark expert action<-----------------")
                            #1.acquire current joint and start positon of B_e_des=B_e_cur
                            mark_curv_multi.stance_group_index.copy_(curv_multi.stance_group_index)
                            mark_curv_multi.q_cur[:]=joint_pos_cur_env[:]
                            mark_curv_multi.kin.ForwardKin(mark_curv_multi.q_cur_flat,mark_curv_multi.B_e_cur_flat)
                            # mark_curv_multi.B_e_des.copy_(mark_curv_multi.B_e_cur)
                            mark_curv_multi.B_e_des.copy_(curv_multi.B_e_des)
                            mark_curv_multi.set_init_done.fill_(True)
                            # print("command:",command[0])
                            # print("B_e_des=B_e_cur:\n",mark_curv_multi.B_e_des[0])
                            #
                            #2.judge gait by suction_force LB LF LM RB RF RM  0-1-5 or 2-4-6 adhesions set to zero, Gaitplanning will change gait
                            A_group_force=torch.norm(suction_force_env[:,[0,1,5]],p=1,dim=1) #env_nums
                            B_group_force=torch.norm(suction_force_env[:,[2,3,4]],p=1,dim=1)
                            stance_group_index_env=torch.zeros(env_nums,dtype=torch.int32,device='cuda:0')
                            # print("A_group_force<B_group_force:\n",A_group_force<B_group_force)
                            stance_group_index_env[A_group_force<B_group_force]=1
                            stance_leg_index=(mark_curv_multi.gaits_groups[stance_group_index_env]).view(-1,3)
                            mark_curv_multi.gaits.fill_(0)
                            mark_curv_multi.gaits[torch.arange(env_nums,dtype=torch.int32,device='cuda:0').unsqueeze(1),stance_leg_index]=1
                            mark_curv_multi.adhesions.fill_(0)
                            # print("suction_force:",suction_force_env[0])
                            # print("gaits:",mark_curv_multi.gaits[0])

                            #3. judeg swing_reach_point 
                            # judeg if leg swing reach the point, use x y dimension (swing_init_point-B_e_cur).dot(command) 
                            # calculate vector from B_e_cur to swing_init_point in xy dimension
                            swing_cur_init_norm=(mark_curv_multi.swing_init_point.unsqueeze(0)-mark_curv_multi.B_e_cur[...,0:3])[...,0:2] #env_nums*6*2
                            # need transfer the vector to R coordinate, for Right legs, remain same for Left legs, reverse both x and y 
                            swing_cur_init_norm[:,0:3,:]=-swing_cur_init_norm[:,0:3,:]
                            #for those vx vy not equal to 0
                            vec_xy_norm=command[:,1:3].unsqueeze(1).expand(-1,6,-1).clone()#env_nums*6*2
                            #for those omega not equal to 0 v_xy_norm=torch.cross([0,0,omega],[swing_init_point[0],swing_init_point[1],0])
                            omega=command[:,4].unsqueeze(1) #env_nums*1
                            R_swing_init_point=mark_curv_multi._B2R(mark_curv_multi.swing_init_point.unsqueeze(0))#1*6*3
                            x=R_swing_init_point[...,0]#1*6
                            y=R_swing_init_point[...,1]#1*6
                            cross_result=torch.stack((-omega*y,omega*x),dim=2)#env_nums*6*2
                            omega_mask=command[:,4]!=0

                            vec_xy_norm[omega_mask]=cross_result[omega_mask]
                            dot_res=(swing_cur_init_norm*vec_xy_norm).sum(dim=-1) #env_nums*6

                            # dot_res=torch.bmm(swing_cur_init_norm,vec_xy_norm).squeeze(-1)#env_nums*6
                            mark_curv_multi.swing_reach_point.fill_(1)
                            # for swing legs, if swing_init_point-B_e_cur dot command>0 & B_e_z<=0.028, set need back to reach point
                            mark_curv_multi.swing_reach_point[(dot_res>0)&(mark_curv_multi.B_e_cur[...,2]<-0.020)]=0
                            # print("dot res=",dot_res[0])
                            # print("mark_curv_multi.B_e_cur[...,2]=",mark_curv_multi.B_e_cur[0,:,2])
                            # print("swing_reach_point:",mark_curv_multi.swing_reach_point[0])


                            #test mark_curv_multi:
                            # if i%100==0:
                            # print("-->mixed policy<--")
                            # if A_group_force[0]==B_group_force[0]:
                            #     print("------------->In mark expert action<-----------------")

                            #     for _ in range(100):
                            #         mark_curv_multi.q_cur[:]=joint_pos_cur_env[:]
                            #         mark_curv_multi.suction_forces[:]=suction_force_env[:]
                            #         mvoing_mask=mark_curv_multi.ProcessCommand(command)
                            #         joint_pos_des_env_part[:]=mark_curv_multi.q_des[:]
                            #         adhesions_env[:]=mark_curv_multi.adhesions[:]
                            #         hex_env.SetJointPos(joint_pos_des_env)
                            #         hex_env.SetAdhesions(adhesions_env)
                            #         hex_env.Simulate()
                            #         time.sleep(0.2)   
                            #         print("B_e_des\n",mark_curv_multi.B_e_des[0])
                            #         print("swing_reach_point\n",mark_curv_multi.swing_reach_point[0])
                            #         print("gaits\n",mark_curv_multi.gaits[0])
                            #         print("adhesions\n",mark_curv_multi.adhesions[0])
                            #         print("suction_force\n",mark_curv_multi.suction_forces[0],"\n")
                            #     #after test, set curv_multi B_e_des
                            #     curv_multi.stance_group_index.copy_(mark_curv_multi.stance_group_index)
                            #     curv_multi.B_e_des.copy_(mark_curv_multi.B_e_des)
                            #     curv_multi.gaits.copy_(mark_curv_multi.gaits)
                            #     curv_multi.adhesions.copy_(mark_curv_multi.adhesions)
                            #     curv_multi.swing_reach_point.copy_(mark_curv_multi.swing_reach_point)

                            # 4. use mark_curv_multi to mark state with action
                            mvoing_mask=mark_curv_multi.ProcessCommand(command)
                            expert_action=CatAction(mark_curv_multi,env_nums)

                            #5. put obs and mark expert action into buffer
                            obs_list.append(obs.clone().detach())
                            action_list.append(expert_action.clone().detach())
                            if len(obs_list)>15000: 
                                del obs_list[0]
                                del action_list[0]
                                torch.cuda.empty_cache()

                        """II. then use mixed policy generate action to interact with env, which will lead robot to new state"""
                        # print("--------------------->In mixed policy <-----------------")
                        # print("gaits=",curv_multi.gaits[0],"\n")
                        agent_action=agent(obs) #agent action
                        agent_action=agent.sample_action(obs,deterministic=False)

                        curv_multi.ProcessCommand(command) #expert action
                        # expert_action=torch.cat([curv_multi.q_des.reshape(-1,24),curv_multi.adhesions.clone()],dim=1)
                        expert_action=CatAction(curv_multi,env_nums)
                        # expert_action=(expert_action-action_min)/(action_max-action_min)
                        # action=expert_action.clone()
                        action=agent_action.clone()
                        choose_expert_mask=torch.rand(env_nums,dtype=torch.float32,device='cuda:0')<p#pow(p,(batch_index-500)/2.0)
                        action[choose_expert_mask]=expert_action[choose_expert_mask]

                        joint_pos_des_env_part[:]=action[:,0:24].reshape(env_nums,6,4)
                        adhesions_env[:]=action[:,24:30].reshape(env_nums,6)>0.8
                        # adhesions_env[:]=curv_multi.adhesions[:]
                        hex_env.SetJointPos(joint_pos_des_env)
                        hex_env.SetAdhesions(adhesions_env)
                        hex_env.Simulate()
                        # time.sleep(0.1)
                        # print("obs after sim:\n",obs)
                        
                        del obs, action, expert_action, agent_action
                        torch.cuda.empty_cache()
                    
                    """use mixed policy to interact with env for one step, use expert to interact with env from current state, and collect obs and action"""
                    if mode=='two_policy':
                        #collect obs
                        obs=CatObs(hex_env,command,env_nums)
                        # obs=(obs-obs_min)/(obs_max-obs_min)
                        #generate agent action
                        agent_action=agent(obs) #agent action
                        action=agent_action.clone()
                        #expert action
                        curv_multi.ProcessCommand(command)
                        expert_action=CatAction(curv_multi,env_nums)
                        #generate mixed policy
                        # choose_expert_mask=torch.rand(env_nums,dtype=torch.float32,device='cuda:0')<pow(p,(batch_index-500)/2.0)
                        choose_expert_mask=torch.rand(env_nums,dtype=torch.float32,device='cuda:0')<p
                        action[choose_expert_mask]=expert_action[choose_expert_mask]
                        # action=action*(action_max-action_min)+action_min
                        joint_pos_des_env_part[:]=action[:,0:24].reshape(env_nums,6,4)
                        # adhesions_env[:]=action[:,24:30].reshape(env_nums,6)>0.6
                        adhesions_env[:]=curv_multi.adhesions[:]
                        hex_env.SetJointPos(joint_pos_des_env)
                        hex_env.SetAdhesions(adhesions_env)
                        hex_env.Simulate()
                        # from current state, use expert to generate right action
                        if i>0 and i%10==0:
                            print("use expert interact with env, collecting data")
                            curv_multi.B_e_des.copy_(curv_multi.B_e_cur)
                            for j in range(10):
                                obs=CatObs(hex_env,command,env_nums)
                                # obs=(obs-obs_min)/(obs_max-obs_min)
                                curv_multi.ProcessCommand(command)
                                joint_pos_des_env_part[:]=curv_multi.q_des[:]
                                adhesions_env[:]=curv_multi.adhesions[:]
                                hex_env.SetJointPos(joint_pos_des_env)
                                hex_env.SetAdhesions(adhesions_env)
                                hex_env.Simulate()
                                expert_action=CatAction(curv_multi,env_nums)
                                # expert_action=(expert_action-action_min)/(action_max-action_min)                                    
                                obs_list.append(obs.clone().detach())
                                action_list.append(expert_action.clone().detach())
                                if len(obs_list)>15000:
                                    del obs_list[0]
                                    del action_list[0]
                                    torch.cuda.empty_cache()      
                            curv_multi.B_e_des.copy_(curv_multi.B_e_cur)     
                        del obs, action, agent_action, expert_action
                        torch.cuda.empty_cache()                     

                    """purely use expert demonstation to collect data"""
                    if mode=='one_policy':
                        obs=CatObs(hex_env,command,env_nums)
                        # print(" command=",obs[0,73:77])
                        # q_cur=obs[0,13:31].reshape(6,3)
                        # pos=torch.zeros_like(q_cur,device='cuda:0')
                        # curv_multi.kin.ForwardKin(q_cur,pos)
                        # print("pos_cur=:\n",pos)


                        # obs=(obs-obs_min)/(obs_max-obs_min)
                        curv_multi.ProcessCommand(command)
                        expert_action=CatAction(curv_multi,env_nums)
                        
                        # q_des=expert_action[0,0:24].reshape(6,4)[...,0:3]
                        # curv_multi.kin.ForwardKin(q_des,pos)
                        # print("q_des:\n",q_des)
                        # print("pos_command=:\n",pos)

                        # expert_action=(expert_action-action_min)/(action_max-action_min)
                        obs_list.append(obs.clone().detach())
                        action_list.append(expert_action.clone().detach())
                        hex_env.SetJointPos(joint_pos_des_env)
                        hex_env.SetAdhesions(adhesions_env)
                        hex_env.Simulate()
                        if len(obs_list)>15000:
                            del obs_list[0]
                            del action_list[0]
                            torch.cuda.empty_cache()
                        del obs, expert_action
                        torch.cuda.empty_cache()
                #train agent for nums_epidoses
                # if (batch_index>collect_min_max_len and batch_index%100==0) | (batch_index>collect_min_max_len):
            if (batch_index>collect_min_max_len):
                # print("obs_list len:",len(obs_list))
                # print("action_list len:",len(action_list))
                dataset=TensorDataset(torch.cat(obs_list,dim=0),torch.cat(action_list,dim=0))
                dataloader=DataLoader(dataset,batch_size=64,shuffle=True)
                for _ in range(2):
                    running_loss=0.0
                    for inputs,targets in dataloader:
                        agent.zero_grad()
                        # outputs=agent(inputs)
                        # loss=F.mse_loss(outputs,targets)
                        miu,log_sigma=agent(inputs)
                        loss=neg_log_prob(miu,log_sigma,targets)
                        loss.backward()
                        optimizer.step()
                        running_loss+=loss.item()
                        loss_list.append(loss.item())
                        agent.zero_grad()
                    print("loss:",running_loss/len(dataloader))
                    print("\n")
                del dataset, dataloader
                torch.cuda.empty_cache()

            if len(obs_list)>14990:
                loop_count+=1
            if loop_count>200:
                torch.save(agent.state_dict(),file_root+'dagger_agent_dist.pth')
                print("save model to ",file_root+'dagger_agent_dist.pth')
                exit()

            batch_index+=1
                


    # except:
    #     torch.save(agent.state_dict(),file_root+'dagger_agent.pth')
    #     print("save model to ",file_root+'dagger_agent.pth')
    #     hex_env.gym.destroy_viewer(hex_env.viewer)
 

        