#所有的关节指令从同一个话题发出[flat, q1, q2, q3, v1, v2, v3, servo, ...] 1+42=43
import rospy, os, time, argparse, torch
import numpy  as np
from BC_learning.Agent_utils import EGPO_Agent
from interface.msg import joy_command
from std_msgs.msg import Float64MultiArray



class motor_mode:
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2


def PubCommand(_):
    global q_des_msgs, left_cur_sub,right_cur_sub
    q_des_pub.publish(q_des_msgs)
    

def ProcessCommand(msg:joy_command):
    global agent, last_actions, q_des_msgs,tic
    if l_cur_time<0.01 or r_cur_time<0.01:
        print("waiting for latest cur")
        # return
    if msg.disable_torque:
        q_des_msgs.data[0]=motor_mode.Disable
        q_des_pub.publish(q_des_msgs)
        return
    
    command=torch.tensor([[msg.set_init,msg.x_vec,msg.y_vec,msg.w_twist]],dtype=torch.float32,device=device)
    v_range=[0.8,1.2,1.5]
    # v_range=[1.0,1.2,2.0]
    command[:,1]=command[:,1]* v_range[0]* 2.0
    command[:,2]=command[:,2]* v_range[1]* 2.0 
    command[:,3]=command[:,3]* v_range[2]* 0.25
    obs = torch.cat([
        last_actions*0.5,
        (q_cur-q_default).reshape(1,-1),
        q_dot_cur.reshape(1,-1) * 0.05,
        q_torq.reshape(1,-1) * 0.1,
        command[:,1:]],dim=1)
    with torch.inference_mode():
        action=agent(obs)
        last_actions = action
        #还原到关节期望角度
        action = action.reshape(6,3)*0.5+q_default

    if msg.set_init==0:
        q_des=torch.cat([action,torch.zeros(size=(6,4),dtype=torch.float,device=device)],dim=1)
        q_des[:,6]=-999        
        q_des_msgs.data[0]=motor_mode.Traj_follow
        q_des_msgs.data[1:]=q_des.view(-1)
    else:
        q_des=torch.cat([q_default,torch.zeros(size=(6,4),dtype=torch.float,device=device)],dim=1)
        q_des[:,6]=-999
        q_des_msgs.data[0]=motor_mode.Pos_vel
        q_des_msgs.data[1:]=q_des.view(-1)
    # print(time.time()-tic)
    tic = time.time()
    q_des_pub.publish(q_des_msgs)
    test_pub.publish(q_des_msgs)
    #这里注销掉,然后在定时器中再打开可以确保收到的是0.2s左右的
    print(f"process r time={r_cur_time}, l time={l_cur_time}")
    #由于电机是发一次收到一次状态,因此采用一个定时器来获取执行指令0.018s后的状态
    # rospy.Timer(rospy.Duration(0.013),PubCommand,oneshot=True)
    rospy.Timer(rospy.Duration(0.008),PubCommand,oneshot=True)
    rospy.Timer(rospy.Duration(0.008),PubCommand,oneshot=True)


def UpdateQState(msg:Float64MultiArray,lr_index):
    global q_cur, q_dot_cur, q_torq, tic, r_cur_time, l_cur_time
    msg_data=torch.tensor(msg.data,device=device).reshape(3,9)
    if lr_index=='l':
        q_cur[0,0:3]=msg_data[:,0:3]
        q_torq[0,0:3]=msg_data[:,3:6]
        q_dot_cur[0,0:3]=msg_data[:,6:9]
        l_cur_time=time.time()-tic
        print("lll")

    elif lr_index=='r':
        q_cur[0,3:6]=msg_data[:,0:3]
        q_torq[0,3:6]=msg_data[:,3:6]
        q_dot_cur[0,3:6]=msg_data[:,6:9]
        r_cur_time=time.time()-tic
        print("rrr")
    




def UpdateRootState(root_state_msgs:Float64MultiArray):
    global device, acc, omega, quat
    quat[0]=torch.tensor(root_state_msgs.data[0:4],dtype=torch.float32,device=device)
    omega[0]=torch.tensor(root_state_msgs.data[4:7],dtype=torch.float32,device=device)
    acc[0]=torch.tensor(root_state_msgs.data[7:10],dtype=torch.float32,device=device)

if __name__=='__main__':
    rospy.init_node("run_agent2",anonymous=True)
    q_des_msgs = Float64MultiArray()
    q_des_msgs.data =[0]*43
    tic=time.time()
    l_cur_time=1.0
    r_cur_time=1.0

    parser=argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default='cpu',help='Device that actor and encoder running, default cpu')
    parser.add_argument("--agent",type=str,default=None,help="Choose an agent in file ~/agents")
    args,_unknow=parser.parse_known_args()
    if args.agent == None:
        print("Please specify a agent using --agent=EGPO.pt for example")
        exit(0)
    device=args.device

    q_default = torch.zeros(6,3,dtype=torch.float32,device=device)
    q_des=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_dot_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_torq=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    #上一次动作
    last_actions=torch.zeros(1,18,dtype=torch.float32,device=device)    
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
    #IMU观测 LB LM LF RB RM RF
    acc = torch.zeros(1,3,dtype=torch.float32,device=device)
    omega = torch.zeros(1,3,dtype=torch.float32,device=device)
    quat = torch.zeros(1,4,dtype=torch.float32,device=device)


    agent = EGPO_Agent()
    # model_path=os.getcwd()+'/src/agent/'+args.agent
    model_path='/home/nvidia/agents/'+args.agent
    agent.load_state_dict(torch.load(model_path)) #为了方便复制粘贴,放到主目录下方
    agent.eval()
    agent.to(device)

    command_sub=rospy.Subscriber('/usr/command',joy_command,ProcessCommand,queue_size=1)
    imu_sub = rospy.Subscriber('/imu/model_states',Float64MultiArray,UpdateRootState,queue_size=1)
    # q_cur_sub = rospy.Subscriber('/sita_cur',Float64MultiArray,UpdateQState,queue_size=10)
    left_cur_sub=rospy.Subscriber('/left_sita_cur',Float64MultiArray,UpdateQState,callback_args='l',queue_size=1)
    right_cur_sub=rospy.Subscriber('/right_sita_cur',Float64MultiArray,UpdateQState,callback_args='r',queue_size=1)

    test_pub = rospy.Publisher('test',Float64MultiArray,queue_size=10)
    
    q_des_pub = rospy.Publisher('/sita_des',Float64MultiArray,queue_size=1)

    print(f" device ={device} ; load agent from {model_path}")
    print("------------------->run agent2 ready<-------------------")
    rospy.spin()




















