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
           self.fro_des = "/RF/sita_des"
           self.mid_des = "/RM/sita_des"
           self.bac_des = "/RB/sita_des"

           self.fro_cur = "/RF/sita_cur"
           self.mid_cur = "/RM/sita_cur"
           self.bac_cur = "/RB/sita_cur"

           self.fro_ID = 0x01
           self.mid_ID = 0x04
           self.bac_ID = 0x07
           self.pn_list = [1,-1,1]

        elif body_part == body_part.left:
           self.fro_des = "/LF/sita_des"
           self.mid_des = "/LM/sita_des"
           self.bac_des = "/LB/sita_des"

           self.fro_cur = "/LF/sita_cur"
           self.mid_cur = "/LM/sita_cur"
           self.bac_cur = "/LB/sita_cur"

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
    def __init__(self,node_name='leg_control',body_part=part.left,port='/dev/ttyACM0'):
        target_joints = target_joint(body_part)
        self.pn_list = target_joints.pn_list
        rospy.init_node(node_name, anonymous=True)

        self.fro_sita_cur_pub = rospy.Publisher(target_joints.fro_cur, Float64MultiArray, queue_size=1)
        self.mid_sita_cur_pub = rospy.Publisher(target_joints.mid_cur, Float64MultiArray, queue_size=1)
        self.bac_sita_cur_pub = rospy.Publisher(target_joints.bac_cur, Float64MultiArray, queue_size=1)

        self.sita_des_sub = rospy.Subscriber(target_joints.fro_des, Float64MultiArray, self.fro_sita_des)
        self.sita_des_sub = rospy.Subscriber(target_joints.mid_des, Float64MultiArray, self.mid_sita_des)
        self.sita_des_sub = rospy.Subscriber(target_joints.bac_des, Float64MultiArray, self.bac_sita_des)
        print("node init------------")
        self.rate = rospy.Rate(200) # 100hz

        self.fro_leg = leg_list(target_joints.fro_ID)
        self.mid_leg = leg_list(target_joints.mid_ID)
        self.bac_leg = leg_list(target_joints.bac_ID)

        self.rec_flag=False
        self.serial_device = serial.Serial(port, 921600, timeout=0.5)
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
        limit_data=list(data.data)
        return limit_data
        for i in range(3):
            if limit_data[i+1]>3.1415926/2:
                limit_data[i+1]=3.1415926/2
                limit_data[i+4]=0

            elif limit_data[i+1]<-3.1415926/2:
                limit_data[i+1]=-3.1415926/2
                limit_data[i+4]=0

        return limit_data
    def motor_staus_pub(self,leg_num:leg_list,whichleg:leglist):
        data=Float64MultiArray()
        n=0
        for i in leg_num.motor:
            data.data.append(i.getPosition()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            data.data.append(i.getTorque()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            data.data.append(i.getVelocity()*self.pn_list[n])
            n=n+1
        # print("get data is ",data.data)
        if whichleg==leglist.Front:
            self.fro_sita_cur_pub.publish(data)
        elif whichleg==leglist.Middle:
            self.mid_sita_cur_pub.publish(data)
        elif whichleg==leglist.Back:
            self.bac_sita_cur_pub.publish(data)
        else:
            print("Leg Error")
    
    def data2leg(self,leg_num:leg_list,data:Float64MultiArray,listnum:int):
        if data.data[0]==modelist.Disable:
            # print("Disable leg--------")
            # self.leg_set_enable(leg_num,False)
            if self.leg_is_enable(leg_num):
                # print("Di leg--------")
                self.leg_set_enable(leg_num,False)
            else:
                pass
        else:
            if not self.leg_is_enable(leg_num):
                self.leg_set_enable(leg_num,True)

                #zero torque mode
            if data.data[0]==modelist.Zero_Torque:
                if leg_num.motor[0].NowControlMode!=Control_Type.MIT:
                    self.leg_set_mode(leg_num,Control_Type.MIT)
                    # self.leg_set_enable(leg_num,True)
                    pass
                for i in range(3):
                    self.MC.controlMIT(leg_num.motor[i], 0, 0, 0, 0, 0)
                pass

                #pos_vel mode
            elif data.data[0]==modelist.Pos_vel:
                pass
                if leg_num.motor[0].NowControlMode!=Control_Type.MIT:
                    self.leg_set_mode(leg_num,Control_Type.MIT)
                cur_poses=np.zeros(3,dtype=np.float64)
                des_poses=np.array([data.data[1],data.data[2],data.data[3]],dtype=np.float64)
                for i in range(3):
                        self.MC.controlMIT(leg_num.motor[i], 0, 0, 0, 0, 0)
                        self.MC.recv()
                        cur_poses[i]=leg_num.motor[i].getPosition()*self.pn_list[i]
                        if cur_poses[i]==0:
                            print("Fail to read pos from motor")
                            exit(0)
                        print("cur_poses[{}]={}".format(i,cur_poses[i]))
                        time.sleep(0.001)
                dis_norm=(des_poses-cur_poses)/np.linalg.norm(cur_poses-des_poses)
                reach_target=False
                while not reach_target:
                    if np.linalg.norm(cur_poses-des_poses)>0.02:
                        cur_poses=dis_norm*0.02+cur_poses
                    else:
                        cur_poses=des_poses
                        reach_target=True
                    for i in range(3):
                        self.MC.controlMIT(leg_num.motor[i],120,1.0,cur_poses[i]*self.pn_list[i],0,0)
                        self.MC.recv()
                    time.sleep(0.005)
            elif data.data[0]==modelist.Traj_follow:
                if leg_num.motor[0].NowControlMode!=Control_Type.MIT:
                    print("now in Traj_follow mode")
                    # self.leg_set_enable(leg_num,False)
                    # time.sleep(0.1)
                    self.leg_set_mode(leg_num,Control_Type.MIT)
                    # self.leg_set_enable(leg_num,True)
                    # time.sleep(0.1)
                    pass
                limit_data=self.limit_check(data)
                for i in range(3):
                # i=2
                    angle_list[listnum+i]=limit_data[i+1]*self.pn_list[i]
                    speed_list[listnum+i]=limit_data[i+4]
                    self.MC.controlMIT(leg_num.motor[i], 100, 1.0, limit_data[i+1]*self.pn_list[i], limit_data[i+4], 0)
                    self.MC.recv()
                    sleep(0.00001)
                # print("set data is ",limit_data[7], limit_data[8], limit_data[i+1]*self.pn_list[i], limit_data[i+4], 0.1)
                    # self.MC.controlMIT(leg_num.motor[i], 0.0, 0.0, 0,0,5)
                # rospy.sleep(0.003)
            else:
                print("Fro Mode Error")
        self.rec_flag=True
    
    def fro_sita_des(self,data):
        with Lock:
            self.data2leg(self.fro_leg,data,0)
            self.motor_staus_pub(self.fro_leg,leglist.Front)

    def mid_sita_des(self,data):
        with Lock:
            self.data2leg(self.mid_leg,data,3)
            self.motor_staus_pub(self.mid_leg,leglist.Middle)

    def bac_sita_des(self,data):
        with Lock:
            self.data2leg(self.bac_leg,data,6)
            self.motor_staus_pub(self.bac_leg,leglist.Back)


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
    # def thread_job(self):
    #     rospy.spin()

    def run(self):
        rospy.spin()
        # thread = threading.Thread(target=self.thread_job)
        # thread.start()
        # while not rospy.is_shutdown():
        #     self.send_data()
        #     self.motor_staus_pub(self.fro_leg,leglist.Front)
        #     self.motor_staus_pub(self.mid_leg,leglist.Middle)
        #     self.motor_staus_pub(self.bac_leg,leglist.Back)
        #     # print("run")
        #     self.rate.sleep()
