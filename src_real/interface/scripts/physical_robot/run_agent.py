from BC_learning.Agent_utils import EGPO_Agent, CatObs
from hex_cfg import HexCfg
import torch
import os
import rospy
from std_msgs.msg import Float64MultiArray
from interface.msg import joy_command
import time
import argparse
import threading

class motor_mode:
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2


def ProcessCommand(msg:joy_command):
    global agent, last_actions, tic
    if msg.disable_torque:
        for i in range(6):
            q_des_msg_list[i].data=[motor_mode.Disable,0,0,0,0,0,0,-999]
            q_des_pub_list[i].publish(q_des_msg_list[i])
        return

    command=torch.tensor([[msg.set_init,msg.x_vec,msg.y_vec,msg.w_twist]],dtype=torch.float32,device=device)
    # v_range=[0.6,0.7,1.0]
    v_range=[1.0,1.2,2.0]
    command[:,1]=command[:,1]* v_range[0]* 2.0
    command[:,2]=command[:,2]* v_range[1]* 2.0 
    command[:,3]=command[:,3]* v_range[2]* 0.25
    #拼接观测，要乘上观测值对应的系数
    # obs = torch.cat([
    #     quat,
    #     omega * 0.25,
    #     acc,
    #     (q_cur-q_default).reshape(1,-1),
    #     q_dot_cur.reshape(1,-1) * 0.05,
    #     q_torq.reshape(1,-1) * 0.1,
    #     command[:,1:]
    # ],dim=1)
    obs = torch.cat([
        last_actions*0.5,
        (q_cur-q_default).reshape(1,-1),
        q_dot_cur.reshape(1,-1) * 0.05,
        q_torq.reshape(1,-1) * 0.1,
        command[:,1:]
    ],dim=1)
    # print("obs.shape=",obs.shape)

    with torch.inference_mode():
        action=agent(obs)
        last_actions = action
        #还原到关节期望角度
        action = action.reshape(6,3)*0.5+q_default
        # action = action*0.2+q_default
    # for _ in range(4):
        for i in range(6):
            if msg.set_init==0:
                q_des_msg_list[i].data=[motor_mode.Traj_follow,action[i,0].item(),action[i,1].item(),action[i,2].item(),0,0,0,-999]
            else:
                q_des_msg_list[i].data=[motor_mode.Pos_vel,q_default[i,0],q_default[i,1],q_default[i,2],0,0,0,-999]
            q_des_pub_list[i].publish(q_des_msg_list[i])
        print("before")
        rospy.sleep(rospy.Duration(0.004))
        print("after")
    # print("-------------")
    tic = time.time()


def UpdateQState(q_state_msgs:Float64MultiArray,index):
    global device, q_cur, q_dot_cur, q_torq,tic
    q_cur[:,index,0:3]=torch.tensor(q_state_msgs.data[0:3],dtype=torch.float32,device=device)
    q_torq[:,index,0:3]=torch.tensor(q_state_msgs.data[3:6],dtype=torch.float32,device=device)
    q_dot_cur[:,index,0:3]=torch.tensor(q_state_msgs.data[6:9],dtype=torch.float32,device=device)
    print(f"{leg_name_list[index]} time interval={time.time()-tic}")

def UpdateRootState(root_state_msgs:Float64MultiArray):
    global device, acc, omega, quat
    quat[0]=torch.tensor(root_state_msgs.data[0:4],dtype=torch.float32,device=device)
    omega[0]=torch.tensor(root_state_msgs.data[4:7],dtype=torch.float32,device=device)
    acc[0]=torch.tensor(root_state_msgs.data[7:10],dtype=torch.float32,device=device)

    # print("quat=",quat)
    # print("omega=",omega)
    # print("acc=",acc)

if __name__ == '__main__':
    tic = time.time()
    toc = None

    parser=argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default='cpu',help='Device that actor and encoder running, default cpu')
    parser.add_argument("--agent",type=str,default=None,help="Choose an agent in file ~/agents")
    args=parser.parse_args()
    if args.agent == None:
        print("Please specify a agent using --agent=EGPO.pt for example")
        exit(0)
    device=args.device

    rospy.init_node('run_agent', anonymous=True)
    hex_cfg=HexCfg("---")

    # device='cpu'
    device = 'cuda:0'
    q_default = torch.zeros(6,3,dtype=torch.float32,device=device)
    q_des=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_dot_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_torq=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    #填充关节默认角度
    for i in range(6):
        if i==0 or i==4:
            thigh=0.5
        elif i==1 or i==3:
            thigh=-0.5
        else:
            thigh=0.0
        knee=0.67
        ankle=-2.2
        q_default[i,0]=thigh
        q_default[i,1]=knee
        q_default[i,2]=ankle
        
    #IMU观测
    acc = torch.zeros(1,3,dtype=torch.float32,device=device)
    omega = torch.zeros(1,3,dtype=torch.float32,device=device)
    quat = torch.zeros(1,4,dtype=torch.float32,device=device)
    #上一次动作
    last_actions=torch.zeros(1,18,dtype=torch.float32,device=device)
    
    # agent=BC_Agent()
    agent = EGPO_Agent()

    model_path='/home/nvidia/agents/'+args.agent
    agent.load_state_dict(torch.load(model_path)) #为了方便复制粘贴,放到主目录下方
    agent.eval()
    agent.to(device)


    command_sub=rospy.Subscriber('/usr/command',joy_command,ProcessCommand,queue_size=10)
    imu_sub = rospy.Subscriber('/imu/model_states',Float64MultiArray,UpdateRootState,queue_size=10)
    q_des_pub_list=[]
    q_cur_sub_list=[]
    q_des_msg_list=[] #Float64MultiArray()
    q_state_dict={'q_des':[],'q_cur':[],'q_dot_cur':[],'torq_cur':[]}
    leg_name_list=['LB','LF','LM','RB','RF','RM']
    for i,leg_name in enumerate(leg_name_list):
        q_des_pub_list.append(rospy.Publisher('/'+leg_name+'/sita_des',Float64MultiArray,queue_size=10))
        q_cur_sub_list.append(rospy.Subscriber('/'+leg_name+'/sita_cur',Float64MultiArray,UpdateQState,callback_args=i))
        q_des_msg_list.append(Float64MultiArray())
    print("------------------->run curv adapt ready<-------------------")
    print(f" device ={device} ; load agent from {model_path}")
    rospy.spin()