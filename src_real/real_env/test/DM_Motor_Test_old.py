#!/usr/bin/env python3
import math
from can_control.scripts.DM_CAN_old import *
import serial
import time
import rospy
from std_msgs.msg import Float32MultiArray
if __name__ == "__main__":
    rospy.init_node("dm_mode")
    que_pub = rospy.Publisher("motor_que",Float32MultiArray,queue_size=10)
    vel_pub = rospy.Publisher("motor_vel",Float32MultiArray,queue_size=10)
    pos_pub = rospy.Publisher("motor_pos",Float32MultiArray,queue_size=10)
    
    Motor1=Motor(DM_Motor_Type.DM4310,0x01,0x11)
    Motor2=Motor(DM_Motor_Type.DM4310,0x02,0x12)
    Motor3=Motor(DM_Motor_Type.DM4310,0x03,0x13)

    serial_device = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
    MotorControl1=MotorControl(serial_device)
    MotorControl1.addMotor(Motor1)
    MotorControl1.addMotor(Motor2)
    MotorControl1.addMotor(Motor3)

    MotorControl1.enable(Motor1)
    MotorControl1.enable(Motor2)
    MotorControl1.enable(Motor3)
    time.sleep(1)
    MotorControl1.switchControlMode(Motor1,Control_Type.POS_VEL)
    MotorControl1.switchControlMode(Motor2,Control_Type.POS_VEL)
    MotorControl1.switchControlMode(Motor3,Control_Type.POS_VEL)

    MotorControl1.save_motor_param(Motor1)
    MotorControl1.save_motor_param(Motor2)
    MotorControl1.save_motor_param(Motor3)
    n=0
    while not rospy.is_shutdown():
        n=n+1
        q=math.sin(time.time())

        MotorControl1.control_Pos_Vel(Motor1,q*15,2)
        # time.sleep(0.001)
        MotorControl1.control_Pos_Vel(Motor2,q*10,2)
        # time.sleep(0.001)
        MotorControl1.control_Pos_Vel(Motor3,q*5,2)
        que_msg = Float32MultiArray()
        vel_msg = Float32MultiArray()
        pos_msg = Float32MultiArray()

        que_msg.data.append(Motor1.getTorque())
        que_msg.data.append(Motor2.getTorque())
        que_msg.data.append(Motor2.getTorque())

        vel_msg.data.append(Motor1.getVelocity())
        vel_msg.data.append(Motor2.getVelocity())
        vel_msg.data.append(Motor3.getVelocity())

        pos_msg.data.append(Motor1.getPosition())
        pos_msg.data.append(Motor2.getPosition())
        pos_msg.data.append(Motor3.getPosition())
        que_pub.publish(que_msg)
        vel_pub.publish(vel_msg)
        pos_pub.publish(pos_msg)
        # print(Motor2.getTorque())

        # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)
        print(n)
        if n==20000:
            MotorControl1.disable(Motor1)
            MotorControl1.disable(Motor2)
            MotorControl1.disable(Motor3)
            serial_device.close()
            break
        time.sleep(0.001)
#语句结束关闭串口
serial_device.close()
