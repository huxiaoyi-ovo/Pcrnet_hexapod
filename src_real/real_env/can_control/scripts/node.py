import rospy
from DM_CAN import *
from std_msgs.msg import Float64MultiArray
import serial
import threading
import time
import numpy as np
angle_list=[0,0,0,0,0,0,0,0,0]
speed_list=[0,0,0,0,0,0,0,0,0]
Lock=threading.Lock()
class part(IntEnum):
    left = 0
    right = 1

class modelist(IntEnum):
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2

class leglist(IntEnum):
    Front = 1
    Middle = 2
    Back = 3
class target_joint:
    def __init__(self, body_part: int):
        if body_part == body_part.right:

           self.status_cur = "right_sita_cur"

           self.fro_ID = 0x01
           self.mid_ID = 0x04
           self.bac_ID = 0x07
           self.pn_list = [1,-1,1]

        elif body_part == body_part.left:

           self.status_cur = "left_sita_cur"
           self.fro_ID = 0x01
           self.mid_ID = 0x04
           self.bac_ID = 0x07
           self.pn_list = [1,1,-1]

class leg_list:
    def __init__(self, num: int):
        self.motor=[]
        Motor1 = Motor(DM_Motor_Type.DM4340,num,0x10+num)
        Motor2 = Motor(DM_Motor_Type.DM4340,num+0x01,0x11+num)
        Motor3 = Motor(DM_Motor_Type.DM4340,num+0x02,0x12+num)
        self.motor.append(Motor1)
        self.motor.append(Motor2)
        self.motor.append(Motor3)

class rosnode:
    def __init__(self,node_name='leg_control',body_part=part.left,port='/dev/ttyACM0',record_txt=False):
        target_joints = target_joint(body_part)
        self.pn_list = target_joints.pn_list
        rospy.init_node(node_name, anonymous=True)

        self.sita_cur_pub = rospy.Publisher(target_joints.status_cur, Float64MultiArray, queue_size=1)
        self.sita_des_sub = rospy.Subscriber("/sita_des", Float64MultiArray, self.sita_des)

        print("node init------------")
        self.rate = rospy.Rate(200) # 100hz

        self.fro_leg = leg_list(target_joints.fro_ID)
        self.mid_leg = leg_list(target_joints.mid_ID)
        self.bac_leg = leg_list(target_joints.bac_ID)
        self.leglist_= [self.bac_leg,self.fro_leg,self.mid_leg]

        self.body_part = body_part
        self.rec_flag=False
        self.serial_device = serial.Serial(port, 921600, timeout=0.5)

        #创建用于写入的文件流
        self.record_txt=record_txt
        if self.record_txt:
            self.file_name = target_joints.status_cur.split('_')[0]
            with open('/home/nvidia/BIH_ws/hex_sim2/bag/'+self.file_name+'.txt','w') as file:
                file.close()


        print("serial is opened-----------")
        self.MC=MotorControl(self.serial_device)

        for a in self.fro_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("front leg is opened--------")
        for a in self.mid_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("midle leg is opened--------")
        for a in self.bac_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("back leg is opened--------")
    def leg_is_enable(self,leg_num:leg_list):
        for i in leg_num.motor:
            # print("motor is enable ",i.isEnable)
            return i.isEnable
            
    def leg_set_enable(self,leg_num:leg_list,ablestatus=True):
        if ablestatus == True:
            for a in leg_num.motor:
                self.MC.enable(a)
                pass
        else:
            for i in leg_num.motor:
                self.MC.disable(i)
                pass

    def leg_set_mode(self,leg_num:leg_list,mode:Control_Type):
        for i in leg_num.motor:
            self.MC.switchControlMode(i,mode)
            time.sleep(1)

            pass
        time.sleep(1)

    def limit_check(self,data:Float64MultiArray):
        limit_data=list(data)
        return limit_data
        for i in range(3):
            if limit_data[i+1]>3.1415926/2:
                limit_data[i+1]=3.1415926/2
                limit_data[i+4]=0

            elif limit_data[i+1]<-3.1415926/2:
                limit_data[i+1]=-3.1415926/2
                limit_data[i+4]=0

        return limit_data
    def motor_staus_pub(self,leg_num:leg_list):
        status = []
        n=0
        for i in leg_num.motor:
            status.append(i.getPosition()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            status.append(i.getTorque()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            status.append(i.getVelocity()*self.pn_list[n])
            n=n+1
        return status

    def return_position(self):
        des_poses=[]
        for leg in self.leglist_:
            for t in range(3):
                des_poses.append(leg.motor[t].getPosition()*self.pn_list[t])
        return des_poses
    def data2leg(self,data:float):
        # print("IN DATA 2 LEG")
        if data[0]==modelist.Disable:
            for leg in self.leglist_:
                if self.leg_is_enable(leg):
                    self.leg_set_enable(leg,False)
                else :
                    pass
        else:
            for leg in self.leglist_:
                if not self.leg_is_enable(leg):
                    self.leg_set_enable(leg,True)
        
            if data[0]==modelist.Zero_Torque:
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)
                        pass
                for leg in self.leglist_:
                    for count_ in range(3):
                        self.MC.controlMIT(leg.motor[count_], 0, 0, 0, 0, 0)
                pass

            elif data[0]==modelist.Pos_vel:
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)

                for leg in self.leglist_:
                    for t in range(3):
                        self.MC.controlMIT(leg.motor[t],0,0,0,0,0)
                        self.MC.recv()
                        time.sleep(0.001)
                # print("IN HERE-----")
                des_poses = np.array(list(data[1:4]) + list(data[7:10]) + list(data[13:16]),dtype=np.float64)
                cur_poses = np.array(self.return_position(), dtype=np.float64)
                
                dis_norm=(des_poses-cur_poses)/np.linalg.norm(cur_poses-des_poses)
                reach_target=False
                while not reach_target:
                    if np.linalg.norm(cur_poses-des_poses)>0.02:
                        cur_poses=dis_norm*0.02+cur_poses
                    else:
                        cur_poses=des_poses
                        reach_target=True

                    for leg_idx, leg in enumerate(self.leglist_):
                        for joint_idx in range(3):
                            idx = leg_idx * 3 + joint_idx
                            target = cur_poses[idx] * self.pn_list[joint_idx]
                            self.MC.controlMIT(leg.motor[joint_idx], 120, 1.0, target, 0, 0)
                            self.MC.recv()
                            time.sleep(0.001)
                            print("in here---------")

            elif data[0]==modelist.Traj_follow:
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)
                        pass
                for leg_idx, leg in enumerate(self.leglist_):
                    for i in range(3):
                        self.MC.controlMIT(leg.motor[i],100,1.0,data[i+1+leg_idx*6]*self.pn_list[i],data[i+4+leg_idx*6],0)
                        self.MC.recv()
                        sleep(0.00001)
            else:
                print("setting mode was wrong")

                
                



    
    def sita_des(self,data):
        with Lock:
            status_data=Float64MultiArray()
            des_data=[]
            set_data=[0.0]*19
            set_data[0]=data.data[0]
            set_data[1:7]=data.data[1+self.body_part*21:7+self.body_part*21]
            set_data[7:13]=data.data[8+self.body_part*21:14+self.body_part*21]
            set_data[13:19]=data.data[15+self.body_part*21:21+self.body_part*21]
            
            self.data2leg(set_data)
            # 扁平化拼接状态并发布
            status_flat = []
            status_flat.extend(self.motor_staus_pub(self.bac_leg))
            status_flat.extend(self.motor_staus_pub(self.fro_leg))
            status_flat.extend(self.motor_staus_pub(self.mid_leg))
            status_data.data = status_flat
            self.sita_cur_pub.publish(status_data)

            #将期望值和当前值写入一个文件中
            # if self.record_txt:
            #     if data.data[0]==modelist.Traj_follow:
            #         with open('/home/nvidia/BIH_ws/hex_sim2/bag/'+self.file_name+'.txt','a') as file:
            #             file.write(f"des={des_data};cur={status_flat}\n")
            #             file.close()





    def send_data(self):
        if self.fro_leg.motor[0].isEnable and self.fro_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.fro_leg.motor[i], 100, 1, angle_list[i], speed_list[i], 0)
                self.MC.recv()
                sleep(0.0002)
        if self.mid_leg.motor[0].isEnable and self.mid_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.mid_leg.motor[i], 100, 1, angle_list[i+3], speed_list[i+3], 0)
                self.MC.recv()
                sleep(0.0002)
        if self.bac_leg.motor[0].isEnable and self.bac_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.bac_leg.motor[i], 100, 1, angle_list[i+6], speed_list[i+6], 0)
                self.MC.recv()
                sleep(0.0002)
        # print("in here----------",selaaf.rec_flag)


            
        pass


    def run(self):
        rospy.spin()