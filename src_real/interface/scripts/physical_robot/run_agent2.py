#所有的关节指令从同一个话题发出[flat, q1, q2, q3, v1, v2, v3, servo, ...] 1+42=43
import threading
import rospy, os, time, argparse, torch
import numpy  as np
from BC_learning.Agent_utils import EGPO_Agent
from interface.msg import joy_command
from std_msgs.msg import Bool, Float64MultiArray



class motor_mode:
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2


LOWLEVEL_HZ = 50.0
PCR_TIMEOUT = 0.3
PCR_CMD_DEADBAND = 1e-4
MANUAL_DEADBAND = 1e-6
pcr_enabled = False
latest_pcr_cmd = None
latest_pcr_stamp = 0.0
hardware_fault = False
left_feedback_received = False
right_feedback_received = False


def PublishDefaultStand():
    q_des=torch.cat([q_default,torch.zeros(size=(6,4),dtype=torch.float,device=device)],dim=1)
    q_des[:,6]=-999
    q_des_msgs.data[0]=motor_mode.Pos_vel
    q_des_msgs.data[1:]=q_des.view(-1)
    q_des_pub.publish(q_des_msgs)


def FeedbackReady():
    if not left_feedback_received or not right_feedback_received:
        rospy.logwarn_throttle(
            1.0,
            "[run_agent2] waiting for initial motor feedback: left=%d right=%d",
            int(left_feedback_received),
            int(right_feedback_received),
        )
        return False
    return True


def ProcessCommand(msg:joy_command, source="legacy"):
    global agent, last_actions, q_des_msgs,tic
    with command_lock:
        if hardware_fault:
            rospy.logerr_throttle(1.0, "[run_agent2] hardware fault latched; command rejected")
            return
        if msg.disable_torque:
            q_des_msgs.data[0]=motor_mode.Disable
            q_des_pub.publish(q_des_msgs)
            return
        if msg.stop or msg.disable_pump or msg.action_valve:
            PublishDefaultStand()
            print(f"process source={source}, safety stand, stop={int(msg.stop)}, disable_pump={int(msg.disable_pump)}, action_valve={int(msg.action_valve)}")
            return
        if msg.set_init:
            PublishDefaultStand()
            print(f"process source={source}, set_init=1, default stand")
            return
        if not FeedbackReady():
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
            PublishDefaultStand()
        # print(time.time()-tic)
        tic = time.time()
        if msg.set_init==0:
            q_des_pub.publish(q_des_msgs)
        test_pub.publish(q_des_msgs)
        #这里注销掉,然后在定时器中再打开可以确保收到的是0.2s左右的
        status = f"process source={source}, set_init={int(msg.set_init)}, x={msg.x_vec:.3f}, y={msg.y_vec:.3f}, yaw={msg.w_twist:.3f}, r time={r_cur_time}, l time={l_cur_time}"
        if source.startswith("pcr"):
            rospy.loginfo_throttle(1.0, status)
        else:
            print(status)
def ClearPcrCache():
    global latest_pcr_cmd, latest_pcr_stamp
    latest_pcr_cmd = None
    latest_pcr_stamp = 0.0


def ManualCommandIsIdle(msg:joy_command):
    has_flag = (
        msg.set_init or msg.moving or msg.disable_pump or msg.disable_torque or
        msg.action_valve or msg.stop or msg.change_mode
    )
    has_motion = (
        abs(msg.x_vec) > MANUAL_DEADBAND or
        abs(msg.y_vec) > MANUAL_DEADBAND or
        abs(msg.z_vec) > MANUAL_DEADBAND or
        abs(msg.w_twist) > MANUAL_DEADBAND
    )
    return (not has_flag) and (not has_motion)


def ManualCommandCb(msg:joy_command):
    global pcr_enabled
    with command_lock:
        if hardware_fault:
            return
        if msg.change_mode:
            pcr_enabled = True
            ClearPcrCache()
            print("[run_agent2] PCR speed input enabled by manual change_mode")
            return

        if ManualCommandIsIdle(msg):
            return

        if pcr_enabled:
            print("[run_agent2] manual command overrides PCR speed input")
        pcr_enabled = False
        ClearPcrCache()
        ProcessCommand(msg, source="manual")


def BuildPcrSpeedCommand(cmd_data):
    cmd = joy_command()
    cmd.set_init = False
    cmd.x_vec = float(cmd_data["x_vec"])
    cmd.y_vec = float(cmd_data["y_vec"])
    cmd.z_vec = float(cmd_data["z_vec"])
    cmd.w_twist = float(cmd_data["w_twist"])
    return cmd


def PcrCommandCb(msg:joy_command):
    global latest_pcr_cmd, latest_pcr_stamp
    with command_lock:
        if hardware_fault:
            return
        x_vec = float(np.clip(msg.x_vec, -1.0, 1.0))
        y_vec = float(np.clip(msg.y_vec, -1.0, 1.0))
        w_twist = float(np.clip(msg.w_twist, -1.0, 1.0))
        latest_pcr_stamp = time.time()
        if abs(x_vec) <= PCR_CMD_DEADBAND and abs(y_vec) <= PCR_CMD_DEADBAND and abs(w_twist) <= PCR_CMD_DEADBAND:
            latest_pcr_cmd = None
            return
        latest_pcr_cmd = {
            "x_vec": x_vec,
            "y_vec": y_vec,
            "z_vec": 0.0,
            "w_twist": w_twist,
        }


def PcrControlTick(_):
    global pcr_enabled
    with command_lock:
        if hardware_fault or not pcr_enabled:
            return

        if latest_pcr_cmd is None:
            rospy.loginfo_throttle(1.0, "[PCR] enabled, waiting for first command")
            return

        now = time.time()
        if now - latest_pcr_stamp > PCR_TIMEOUT:
            pcr_enabled = False
            ClearPcrCache()
            rospy.logwarn("[PCR] command timeout, disable PCR output")
            return

        ProcessCommand(BuildPcrSpeedCommand(latest_pcr_cmd), source="pcr_50hz")


def LegacyCommandCb(msg:joy_command):
    global pcr_enabled
    with command_lock:
        if hardware_fault:
            return
        if pcr_enabled:
            pcr_enabled = False
            ClearPcrCache()
        ProcessCommand(msg, source="legacy")


def HardwareFaultCb(msg:Bool):
    global hardware_fault, pcr_enabled
    if not msg.data:
        return
    with command_lock:
        if hardware_fault:
            return
        hardware_fault = True
        pcr_enabled = False
        ClearPcrCache()
        q_des_msgs.data[0] = motor_mode.Disable
        q_des_pub.publish(q_des_msgs)
        rospy.logfatal(
            "[run_agent2] CAN serial fault latched; Disable published once and all commands rejected until node restart"
        )


def UpdateQState(msg:Float64MultiArray,lr_index):
    global q_cur, q_dot_cur, q_torq, tic, r_cur_time, l_cur_time
    global left_feedback_received, right_feedback_received
    if len(msg.data) != 27:
        rospy.logerr_throttle(
            1.0,
            "[run_agent2] invalid %s motor feedback length: %d",
            lr_index,
            len(msg.data),
        )
        return
    msg_data=torch.tensor(msg.data,device=device).reshape(3,9)
    if not torch.isfinite(msg_data).all():
        rospy.logerr_throttle(1.0, "[run_agent2] non-finite %s motor feedback rejected", lr_index)
        return
    if lr_index=='l':
        q_cur[0,0:3]=msg_data[:,0:3]
        q_torq[0,0:3]=msg_data[:,3:6]
        q_dot_cur[0,0:3]=msg_data[:,6:9]
        l_cur_time=time.time()-tic
        left_feedback_received = True

    elif lr_index=='r':
        q_cur[0,3:6]=msg_data[:,0:3]
        q_torq[0,3:6]=msg_data[:,3:6]
        q_dot_cur[0,3:6]=msg_data[:,6:9]
        r_cur_time=time.time()-tic
        right_feedback_received = True
    




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
    parser.add_argument("--legacy_command_topic",type=str,default="",help="Optional single command topic for old one-topic mode")
    parser.add_argument("--manual_command_topic",type=str,default="/usr/command_manual")
    parser.add_argument("--pcr_command_topic",type=str,default="/usr/command_pcr")
    args,_unknow=parser.parse_known_args()
    if args.agent == None:
        print("Please specify a agent using --agent=EGPO.pt for example")
        exit(0)
    device=args.device
    pcr_enabled = False
    latest_pcr_cmd = None
    latest_pcr_stamp = 0.0
    hardware_fault = False
    left_feedback_received = False
    right_feedback_received = False
    command_lock = threading.RLock()

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
    # Joint/state order: LB LF LM RB RF RM
    acc = torch.zeros(1,3,dtype=torch.float32,device=device)
    omega = torch.zeros(1,3,dtype=torch.float32,device=device)
    quat = torch.zeros(1,4,dtype=torch.float32,device=device)


    agent = EGPO_Agent()
    # model_path=os.getcwd()+'/src/agent/'+args.agent
    model_path='/home/nvidia/agents/'+args.agent
    agent.load_state_dict(torch.load(model_path)) #为了方便复制粘贴,放到主目录下方
    agent.eval()
    agent.to(device)

    test_pub = rospy.Publisher('test',Float64MultiArray,queue_size=10)
    q_des_pub = rospy.Publisher('/sita_des',Float64MultiArray,queue_size=1)
    hardware_fault_sub = rospy.Subscriber(
        '/can_control/serial_fault',
        Bool,
        HardwareFaultCb,
        queue_size=1,
    )

    if args.legacy_command_topic:
        command_sub=rospy.Subscriber(args.legacy_command_topic,joy_command,LegacyCommandCb,queue_size=1)
        print(f"[run_agent2] legacy command topic: {args.legacy_command_topic}")
    else:
        manual_sub=rospy.Subscriber(args.manual_command_topic,joy_command,ManualCommandCb,queue_size=1)
        pcr_sub=rospy.Subscriber(args.pcr_command_topic,joy_command,PcrCommandCb,queue_size=1)
        print(f"[run_agent2] manual topic: {args.manual_command_topic}")
        print(f"[run_agent2] PCR topic: {args.pcr_command_topic}")
        print(f"[run_agent2] manual change_mode enables PCR speed input; any other manual command disables it")
        print("[run_agent2] manual and PCR commands use one /sita_des publish per control update")
    imu_sub = rospy.Subscriber('/imu/model_states',Float64MultiArray,UpdateRootState,queue_size=1)
    # q_cur_sub = rospy.Subscriber('/sita_cur',Float64MultiArray,UpdateQState,queue_size=10)
    left_cur_sub=rospy.Subscriber('/left_sita_cur',Float64MultiArray,UpdateQState,callback_args='l',queue_size=1)
    right_cur_sub=rospy.Subscriber('/right_sita_cur',Float64MultiArray,UpdateQState,callback_args='r',queue_size=1)

    pcr_control_timer = rospy.Timer(rospy.Duration(1.0 / LOWLEVEL_HZ), PcrControlTick)

    print(f" device ={device} ; load agent from {model_path}")
    print("------------------->run agent2 ready<-------------------")
    rospy.spin()


