#!/usr/bin/python3
import rospy
from std_msgs.msg import Float64MultiArray
from interface.msg import joy_command
from hex_cfg import HexCfg
from curv_adapt_multi import CurvAdapt_multi
import torch
import random 
import threading
import queue
import argparse
Lock=threading.Lock()
process_deque=queue.Queue(maxsize=6)
class motor_mode:
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2

def ProcessCommand(msg:joy_command):
    global device
    with Lock:
        if msg.disable_torque:
            q_des_msgs.data[0]=motor_mode.Disable
            q_des_pub.publish(q_des_msgs)
            return

        command=torch.tensor([[msg.set_init,msg.x_vec,msg.y_vec,0,msg.w_twist]],dtype=torch.float32,device=device)

        command[:,1]=command[:,1]*hex_cfg.max_vec.x
        command[:,2]=command[:,2]*hex_cfg.max_vec.y
        command[:,3]=0.0
        command[:,4]=command[:,4]*hex_cfg.max_vec.omega_move
        if curv_multi.cfg.high_body:
            command = 0.3*command #防止过快
        if msg.change_mode:
            curv_multi.cfg.high_body=not curv_multi.cfg.high_body
        print("command:",command)
        curv_multi.ProcessCommand(command)
        # print("curv_multi.set_init_done:",curv_multi.set_init_done)
        #use 
        # print("pub command")
        #指令处理50HZ,控制200HZ
        # for _ in range(4):
        if msg.set_init==0:
            joint_command=torch.cat([q_des[0,:,:3],torch.zeros(size=(6,4),dtype=torch.float,device=device)],dim=1)
            joint_command[:,6] = q_des[0,:,3] #舵机关节角度
            #当前顺序是RF RM RB
            joint_command[:,6]=-999        
            q_des_msgs.data[0]=motor_mode.Traj_follow
            q_des_msgs.data[1:]=joint_command.view(-1)
        else:
            joint_command=torch.cat([q_des[0,:,:3],torch.zeros(size=(6,4),dtype=torch.float,device=device)],dim=1)
            joint_command[:,6] = q_des[0,:,3]
            joint_command[:,6]=-999
            q_des_msgs.data[0]=motor_mode.Pos_vel
            q_des_msgs.data[1:]=joint_command.view(-1)        
        q_des_pub.publish(q_des_msgs)
        # for i in range(6):
        #     # print("q_des[{}]={}".format(i,q_des[0,i]))
        #     q_pos_des=q_des[0,i,0:4].cpu().tolist()
        #     if curv_multi.gaits[0,i]:
        #         # q_pos_des[3]=-999#for stance state, set sevo disable  舵机断势能保护
        #         pass

        #     if msg.set_init==0: #2 represent traj follow mode
        #         q_des_msg_list[i].data=[motor_mode.Traj_follow,q_pos_des[0],q_pos_des[1],q_pos_des[2],0,0,0,q_pos_des[3]]
        #     else: #1 represent pos vel mode
        #         q_des_msg_list[i].data=[motor_mode.Pos_vel,q_pos_des[0],q_pos_des[1],q_pos_des[2],0.0,0.0,0.0,q_pos_des[3]]
        #         # q_des_msg_list[i].data=[motor_mode.Traj_follow,q_pos_des[0],q_pos_des[1],q_pos_des[2],0,0,0,q_pos_des[3]]
        #     q_des_pub_list[i].publish(q_des_msg_list[i])
            # time.sleep(0.002)
            # print(leg_name_list[i],":q_des_msg=",q_des_msg_list[i].data)

        GetSuctionForce(adhesions,suction_force)
        #记录电机信息
        # if curv_multi.set_init_done.all():
        #     #t time step:
        #     q_state_dict['q_cur'].append(q_cur.clone().squeeze(0))
        #     q_state_dict['q_dot_cur'].append(q_dot_cur.clone().squeeze(0))
        #     q_state_dict['q_des'].append(q_des.clone().squeeze(0))
        #     #t-1 time step
        #     q_state_dict['torq_cur'].append(q_torq.clone().squeeze(0))
        #     if (len(q_state_dict['torq_cur'])==2000):
        #         suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #         torch.save(q_state_dict,os.getcwd()+'/bag/motor_data/motor_raw_'+suffix+'.pt')
        #         print("save motor data to bag/motor_data/motor_raw_"+suffix+".pt")
        #         for key in q_state_dict.keys():
        #             q_state_dict[key]=[]
        #     # print("----------->save q_cur q_des<-------------")
        #     # print("q_des_t\n",q_des)
        #     # print("q_cur\n",q_cur)
        #     # print("q_dot_cur\n",q_dot_cur)

# def UpdateQState(q_state_msgs:Float64MultiArray,index):
#     #实物交互更新
#     global device
#     # q_cur[:,index,0:3]=torch.tensor(q_state_msgs.data[0:3],dtype=torch.float32,device=device)
#     # q_torq[:,index,0:3]=torch.tensor(q_state_msgs.data[3:6],dtype=torch.float32,device=device)
#     # q_dot_cur[:,index,0:3]=torch.tensor(q_state_msgs.data[6:9],dtype=torch.float32,device=device)
#     #仿真交互跟新
#     q_cur[:,index,0:3]=torch.tensor(q_state_msgs.data[0:3],dtype=torch.float32,device=device)
def UpdateQState(msg:Float64MultiArray,lr_index):
    global q_cur, q_dot_cur, q_torq
    msg_data=torch.tensor(msg.data,device=device).reshape(3,9)
    if lr_index=='l':
        q_cur[0,0:3]=msg_data[:,0:3]
        q_torq[0,0:3]=msg_data[:,3:6]
        q_dot_cur[0,0:3]=msg_data[:,6:9]
        print("lll")

    elif lr_index=='r':
        q_cur[0,3:6]=msg_data[:,0:3]
        q_torq[0,3:6]=msg_data[:,3:6]
        q_dot_cur[0,3:6]=msg_data[:,6:9]
        print("rrr")
def GetSuctionForce(adhesions:torch.Tensor,suction_force:torch.Tensor):
    #需要吸附
    add_force_mask=adhesions &(suction_force<=hex_cfg.max_suck_force)    
    suction_force[add_force_mask]+=20+(random.random()-0.5)*10
    suction_force[suction_force>hex_cfg.max_suck_force]=hex_cfg.max_suck_force
    #需要释放
    sub_force_mask=(~adhesions) & (suction_force>=0)
    suction_force[sub_force_mask]-=20+(random.random()-0.5)*10
    suction_force[suction_force<0]=0
    

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default='cpu',help='Device that actor and encoder running, default cpu')
    args=parser.parse_args()
    device=args.device

    rospy.init_node('run_curv_adapt',anonymous=True)
    hex_cfg=HexCfg("config.yaml")
    command_sub=rospy.Subscriber('/usr/command',joy_command,ProcessCommand,queue_size=10)
    # device='cpu'
    print("device=",device)
    q_des=torch.zeros(1,6,4,dtype=torch.float32,device=device)
    q_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_dot_cur=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    q_torq=torch.zeros(1,6,3,dtype=torch.float32,device=device)
    adhesions=torch.zeros(1,6,dtype=torch.bool,device=device)
    suction_force=torch.zeros(1,6,dtype=torch.float32,device=device)
    curv_multi=CurvAdapt_multi("config.yaml",device,1,
                               q_des,q_cur,q_torq,adhesions,suction_force)
    
    left_cur_sub=rospy.Subscriber('/left_sita_cur',Float64MultiArray,UpdateQState,callback_args='l',queue_size=1)
    right_cur_sub=rospy.Subscriber('/right_sita_cur',Float64MultiArray,UpdateQState,callback_args='r',queue_size=1)
    q_des_pub = rospy.Publisher('/sita_des',Float64MultiArray,queue_size=1)
    q_des_msgs = Float64MultiArray()
    q_des_msgs.data =[0]*43

    # q_cur_flat=q_cur.view(18)
    # q_torq_flat=q_torq.view(18)
    # q_des_flat=q_des[...,0:3].view(18)
    # q_des_pub_list=[]
    # q_cur_sub_list=[]
    # q_des_msg_list=[] #Float64MultiArray()
    # q_state_dict={'q_des':[],'q_cur':[],'q_dot_cur':[],'torq_cur':[]}
    # leg_name_list=['LB','LF','LM','RB','RF','RM']
    # for i,leg_name in enumerate(leg_name_list):
    #     q_des_pub_list.append(rospy.Publisher('/'+leg_name+'/sita_des',Float64MultiArray,queue_size=10))
    #     q_cur_sub_list.append(rospy.Subscriber('/'+leg_name+'/sita_cur',Float64MultiArray,UpdateQState,callback_args=i))
    #     q_des_msg_list.append(Float64MultiArray())
    # print("------------------->run curv adapt ready<-------------------")
    rospy.spin()

