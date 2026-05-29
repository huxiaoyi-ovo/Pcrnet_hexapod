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
    # raw_data=torch.load("/home/nvidia/BIH_ws/hex_sim2/bag/motor_data/sin_curve_data_1.pt") 
    # print(raw_data.keys())
    # q_cur=torch.cat(raw_data['q_cur'],dim=0)
    # print((q_cur==0).sum())
    # q_des=torch.cat(raw_data['q_des'],dim=0)
    
    serial_l=serial.Serial('/dev/DM_l',921600, timeout=0.5)
    # # serial_r=serial.Serial('/dev/DM_r',921600, timeout=0.5)
    motor_ctrl_l=MotorControl(serial_l)
    
    # # motor_ctrl_r=MotorControl(serial_r)
    motors=[]
    for i in range(1):
        motors.append(Motor(DM_Motor_Type.DM4340,0x01+i,0x11+i))
        motor_ctrl_l.addMotor(motors[i])
        # motor_ctrl_r.addMotor(motors[i])
        motor_ctrl_l.enable(motors[i])
        time.sleep(0.1)
        # motor_ctrl_r.enable(motors[i])
    toqrue=0.0        
    for i in range(1000):
        q=math.sin(i*0.2)*0.5
        if i%100==0:
            toqrue+=0.1
        # motor_ctrl_l.controlMIT(motors[0], 150, 2, q, 0, 0)
        motor_ctrl_l.controlMIT(motors[0], 0, 0, 0, 0, toqrue)
        motor_ctrl_l.recv()
        print(f"motors.torque={motors[0].getTorque()},motors.velocity={motors[0].getVelocity()}")
        time.sleep(0.02)