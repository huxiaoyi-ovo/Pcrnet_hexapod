#!/usr/bin/python3
import math
from DM_CAN import *
import serial
import time
import random
import torch
import os
import rospy
from std_msgs.msg import Float32MultiArray
if __name__ == '__main__':
    rospy.init_node('dm_motor_test', anonymous=True)
    pub = rospy.Publisher('/test', Float32MultiArray, queue_size=10)
    msg=Float32MultiArray()
    msg.data=[0]*9 #[pos_err vel_err torq_cur]

    serial_l=serial.Serial('/dev/DM_r',921600, timeout=0.5)
    # serial_r=serial.Serial('/dev/DM_r',921600, timeout=0.5)
    motor_ctrl_l=MotorControl(serial_l)
    # motor_ctrl_r=MotorControl(serial_r)
    motors=[]
    for i in range(9):
        motors.append(Motor(DM_Motor_Type.DM4340,0x01+i,0x11+i))
        motor_ctrl_l.addMotor(motors[i])
        # motor_ctrl_r.addMotor(motors[i])
        motor_ctrl_l.enable(motors[i])
        time.sleep(0.1)
        # motor_ctrl_r.enable(motors[i])
    q_state_dict={'q_des':[],'q_cur':[],'q_dot_cur':[],'torq_cur':[]}
    i=0
    q_des=torch.zeros(9,dtype=torch.float32)
    q_cur=torch.zeros_like(q_des)
    q_dot_cur=torch.zeros_like(q_des)
    torq_cur=torch.zeros_like(q_des)

    #set back to zeros
    for j in range(9):
        motor_ctrl_l.controlMIT(motors[j], 0, 0, 0, 0, 0)
        motor_ctrl_l.recv()
        time.sleep(0.0002)
        q_cur[j]=float(motors[j].getPosition())    
    while True:
        if (q_cur).norm()>0.1:
            q_des=((0-q_cur)/torch.norm(q_cur))*0.1+q_cur
        else:
            q_des.fill_(0)
        for j in range(9):
            print("control q_des:",q_des[j].item())
            motor_ctrl_l.controlMIT(motors[j], 150, 2, q_des[j].item(), 0, 0)
            motor_ctrl_l.recv()
            time.sleep(0.0002)
            q_cur[j]=float(motors[j].getPosition())
        print("q_cur=",q_cur)
        print("q_des=",q_des)   
        if (q_cur).norm()<0.1:
            break
        time.sleep(0.001)

    start_tic=time.time()
    i=0
    while not rospy.is_shutdown():
        i=i+1
        #resample max_vec
        # if i%200==0 or i==1:
        #     speed_item_list=[]
        #     amplitude_item_list=[]
        #     for j in range(18):
        #         speed_item_list.append((random.random()-0.5))
        #         amplitude_item_list.append((random.random()-0.5))
        
        for j in range(9):
            if j%3==0:
                q_des[j]=math.sin((time.time()-start_tic)*3)*0.6
            if j==1 or j==2:
                q_des[j]=math.sin((time.time()-start_tic)*3.2)*1.7
            if j==4 or j==5:
                q_des[j]=math.sin((time.time()-start_tic)*3.4)*1.7
            if j==7 or j==8:
                q_des[j]=math.sin((time.time()-start_tic)*3.6)*1.7
            if j==2 or j==5 or j==8:
                q_des[j]=math.sin((time.time()-start_tic)*3)*0.2

            motor_ctrl_l.controlMIT(motors[j], 150, 2, q_des[j].item(), 0, 0)
            motor_ctrl_l.recv()
            time.sleep(0.0002)
            q_cur[j]=float(motors[j].getPosition())
            q_dot_cur[j]=float(motors[j].getVelocity())
            torq_cur[j]=float(motors[j].getTorque())
        q_state_dict['q_des'].append(q_des.clone())
        q_state_dict['q_cur'].append(q_cur.clone())
        q_state_dict['q_dot_cur'].append(q_dot_cur.clone())
        q_state_dict['torq_cur'].append(torq_cur.clone())
        #pub msg to visualize
        for j in range(3):
            msg.data[j]=(q_des[j]-q_cur[j]).item()
            msg.data[j+3]=(-q_dot_cur[j]).item()
            msg.data[j+6]=(torq_cur[j]).item()
        pub.publish(msg)
        # for j in range(9):
            # else:
            #     q=math.cos(time.time()*speed_item_list[j-9])*amplitude_item_list[j-9]
            #     motor_ctrl_r.controlMIT(motors[j-9], 150, 2, q, 0, 0)
            #     motor_ctrl_r.recv()
            #     time.sleep(0.0002)
            #     q_state_dict['q_des'].append(q)
            #     q_state_dict['q_cur'].append(motors[j-9].getPosition())
            #     q_state_dict['q_dot_cur'].append(motors[j-9].getVelocity())
            #     q_state_dict['torq_cur'].append(motors[j-9].getTorque())                
        if i == 2000:
            torch.save(q_state_dict,os.getcwd()+'/bag/motor_data/sin_curve_data.pt')
            break
        time.sleep(0.01)
    serial_l.close()
    # serial_r.close()
    exit()


    # Motor1=Motor(DM_Motor_Type.DM4340,0x01,0x11)
    # Motor2=Motor(DM_Motor_Type.DM4340,0x02,0x12)
    # Motor3=Motor(DM_Motor_Type.DM4340,0x03,0x13)

    # serial_device = serial.Serial('/dev/DM_r', 921600, timeout=0.5)
    # MotorControl1=MotorControl(serial_device)
    # MotorControl1.addMotor(Motor1)
    # MotorControl1.addMotor(Motor2)
    # MotorControl1.addMotor(Motor3)

    # # # MotorControl1.addMotor(Motor2)
    # # if MotorControl1.switchControlMode(Motor1,Control_Type.POS_VEL):
    # #     print("switch POS_VEL success")
    # # if MotorControl1.switchControlMode(Motor2,Control_Type.POS_VEL):
    # #     print("switch POS_VEL success")
    # # if MotorControl1.switchControlMode(Motor3,Control_Type.POS_VEL):
    # #     print("switch POS_VEL success")
    # # # if MotorControl1.switchControlMode(Motor2,Control_Type.VEL):
    # # #     print("switch VEL success")
    # # print("sub_ver:",MotorControl1.read_motor_param(Motor1,DM_variable.sub_ver))
    # # print("Gr:",MotorControl1.read_motor_param(Motor1,DM_variable.Gr))

    # # # if MotorControl1.change_motor_param(Motor1,DM_variable.KP_APR,54):
    # # #     print("write success")
    # # print("PMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.PMAX))
    # # print("MST_ID:",MotorControl1.read_motor_param(Motor1,DM_variable.MST_ID))
    # # print("VMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.VMAX))
    # # print("TMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.TMAX))
    # # # MotorControl1.save_motor_param(Motor1)
    # # # MotorControl1.save_motor_param(Motor2)
    # MotorControl1.enable(Motor1)
    # MotorControl1.enable(Motor2)
    # MotorControl1.enable(Motor3)

    # # MotorControl1.enable(Motor2)
    # i=0
    # while i<10000:
    #     q=math.sin(time.time()*1.5)
    #     i=i+1
    #     # MotorControl1.control_pos_force(Motor1, 10, 1000,100)
    #     # MotorControl1.control_Vel(Motor1, q*5)
    #     # MotorControl1.control_Pos_Vel(Motor1,0.2,0.1)
    #     # MotorControl1.control_Pos_Vel(Motor2,0.2,0.1)
    #     # MotorControl1.control_Pos_Vel(Motor3,0.1,0.1)

    #     # print("Motor1:","POS:",Motor1.getPosition(),"VEL:",Motor1.getVelocity(),"TORQUE:",Motor1.getTorque())
    #     MotorControl1.controlMIT(Motor1, 150, 2, 0, 0, 0)
    #     MotorControl1.recv()
    #     time.sleep(0.0002)
    #     MotorControl1.controlMIT(Motor2, 150, 2, q, 0, 0)
    #     MotorControl1.recv()
    #     time.sleep(0.0002)
    #     MotorControl1.controlMIT(Motor3, 150, 2, q, 0, 0)
    #     MotorControl1.recv()

    #     Motor1.getPosition()
    #     Motor1.getVelocity()
    #     Motor1.getTorque()

    #     time.sleep(0.01)
    #     # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)

    # #语句结束关闭串口
    # serial_device.close()