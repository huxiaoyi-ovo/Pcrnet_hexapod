#!/usr/bin/env python3
import math
import serial
import time
import rospy
from std_msgs.msg import Float64MultiArray
from can_control.msg import Sitades

if __name__ == "__main__":
    rospy.init_node("zxc_body")
    que_pub = rospy.Publisher("/LF/sita_des",Float64MultiArray,queue_size=1)
    vel_pub = rospy.Publisher("/LM/sita_des",Float64MultiArray,queue_size=1)
    pos_pub = rospy.Publisher("/LB/sita_des",Float64MultiArray,queue_size=1)

    # vel_pub = rospy.Publisher("motor_vel",Float32MultiArray,queue_size=10)
    # pos_pub = rospy.Publisher("motor_pos",Float32MultiArray,queue_size=10)
    
    # Motor1=Motor(DM_Motor_Type.DM4310,0x01,0x11)
    # Motor2=Motor(DM_Motor_Type.DM4310,0x02,0x12)
    # Motor3=Motor(DM_Motor_Type.DM4310,0x03,0x13)

    # serial_device = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
    # MotorControl1=MotorControl(serial_device)
    # MotorControl1.addMotor(Motor1)
    # MotorControl1.addMotor(Motor2)
    # MotorControl1.addMotor(Motor3)

    # MotorControl1.enable(Motor1)
    # MotorControl1.enable(Motor2)
    # MotorControl1.enable(Motor3)
    # time.sleep(1)
    # MotorControl1.switchControlMode(Motor1,Control_Type.MIT)
    # MotorControl1.switchControlMode(Motor2,Control_Type.POS_VEL)
    # MotorControl1.switchControlMode(Motor3,Control_Type.VEL)

    # MotorControl1.save_motor_param(Motor1)
    # MotorControl1.save_motor_param(Motor2)
    # MotorControl1.save_motor_param(Motor3)
    n=-0.3
    delaa=0.0003
    while not rospy.is_shutdown():
        if n>0.3:
            delaa=-0.0003
        if n<-0.3:
            delaa=0.0003
        n=n+delaa
        q=math.sin(time.time())/2

        # MotorControl1.control_Pos_Vel(Motor1,q*15,2)
        # time.sleep(0.001)
        # MotorControl1.control_Pos_Vel(Motor2,q*10,2)
        # time.sleep(0.001)
        # MotorControl1.control_Pos_Vel(Motor3,q*5,2)

        # MotorControl1.controlMIT(Motor1, 50, 0.3, 10*q, 0, 0.1)

        # MotorControl1.control_Pos_Vel(Motor2,q*10,2)

        # MotorControl1.control_Vel(Motor3, q*5)


        que_msg = Float64MultiArray()
        vel_msg = Float64MultiArray()
        pos_msg = Float64MultiArray()

        que_msg.data.append(2)
        que_msg.data.append(q)
        que_msg.data.append(q)
        que_msg.data.append(q)
        que_msg.data.append(math.cos(time.time())/2)
        que_msg.data.append(math.cos(time.time())/2)
        que_msg.data.append(math.cos(time.time())/2)
        que_msg.data.append(0.5)
        que_msg.data.append(5)
        # vel_msg.data.append(Motor1.getVelocity())
        # vel_msg.data.append(Motor2.getVelocity())
        # vel_msg.data.append(Motor3.getVelocity())
        vel_msg.data=que_msg.data
        pos_msg.data=que_msg.data
        # pos_msg.data.append(Motor1.getPosition())
        # pos_msg.data.append(Motor2.getPosition())
        # pos_msg.data.append(Motor3.getPosition())
        que_pub.publish(que_msg)
        vel_pub.publish(vel_msg)
        pos_pub.publish(pos_msg)
        # print(Motor2.getTorque())

        # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)
        print(n)
        if n>30000000:
            # MotorControl1.disable(Motor1)
            # MotorControl1.disable(Motor2)
            # MotorControl1.disable(Motor3)
            # serial_device.close()
            break
        time.sleep(0.003)
#语句结束关闭串口
# serial_device.close()
