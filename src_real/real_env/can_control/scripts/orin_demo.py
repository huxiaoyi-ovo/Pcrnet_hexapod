import math
from can_control.scripts.DM_CAN_old import *
import serial
import time

Motor1=Motor(DM_Motor_Type.DM4340,0x02,0x12)
# Motor2=Motor(DM_Motor_Type.DM4310,0x05,0x15)
serial_device = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
MotorControl1=MotorControl(serial_device)
MotorControl1.addMotor(Motor1)
# MotorControl1.addMotor(Motor2)
# MotorControl1.enable(Motor1)
# MotorControl1.enable(Motor2)
time.sleep(1)
# MotorControl1.switchControlMode(Motor1,Control_Type.MIT)
MotorControl1.switchControlMode(Motor1,Control_Type.POS_VEL)
time.sleep(0.01)
MotorControl1.enable(Motor1)
time.sleep(0.01)
MotorControl1.read_motor_param(Motor1,0x0a)
time.sleep(0.01)

MotorControl1.read_motor_param(Motor1,0x0a)
time.sleep(0.01)

MotorControl1.read_motor_param(Motor1,0x0a)

# MotorControl1.disable(Motor1)
# MotorControl1.save_motor_param(Motor1)
for i in range(100):
# while True:
    q=math.sin(time.time())
    # MotorControl1.control_pos_force(Motor1, 10, 1000,100)
    # MotorControl1.control_Vel(Motor1, q*5)
    # MotorControl1.control_Pos_Vel(Motor1,0.0,0.1)
    # MotorControl1.controlMIT(Motor1, 50, 0.3, 0.0, 0, 0.1)
    # MotorControl1.control(Motor2, 0, 0, 0, 0, 0.1)
    # print(Motor1.getPosition())
    # MotorControl1.recv(Motor1)

    # print(Motor2.getTorque())
    # time.sleep(0.01)
    # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)

#语句结束关闭串口
serial_device.close()