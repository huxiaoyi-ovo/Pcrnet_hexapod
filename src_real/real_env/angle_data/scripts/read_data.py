#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np
from datetime import datetime
import csv
import os

class ServoController:
    def __init__(self):
        rospy.init_node('servo_controller')
        
        # 初始化参数
        self.num_servos = 6
        self.min_angle = -1.0  # 最小弧度
        self.max_angle = 1.0   # 最大弧度
        
        # 为每个舵机设置不同的频率(Hz)和速度(弧度/秒)
        self.frequencies = [25.0, 50.0, 80.0, 100.0, 130.0, 150.0]
        # 速度改为弧度/秒 (原来的度数除以约57.3并调整)
        self.speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        
        # 初始化舵机状态
        self.current_angles = [0.0] * self.num_servos
        self.target_angles = [0.0] * self.num_servos
        self.directions = [1] * self.num_servos  # 1表示增加角度，-1表示减少角度
        self.last_pub_times = [rospy.Time.now()] * self.num_servos
        self.next_target_angles=[0,0,0,0,0,0]
        # 创建发布者
        self.publishers = []
        for i in range(self.num_servos):
            pub = rospy.Publisher('/LF/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)
            pub = rospy.Publisher('/LM/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)
            pub = rospy.Publisher('/LB/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)
            pub = rospy.Publisher('/RF/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)
            pub = rospy.Publisher('/RM/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)
            pub = rospy.Publisher('/RB/sita_des', Float64MultiArray, queue_size=1)
            self.publishers.append(pub)

        # 创建订阅者
        rospy.Subscriber('/servo_pos', Float64MultiArray, self.feedback_callback)
        # 创建数据记录文件
        self.data_files = []
        self.csv_writers = []
        for i in range(self.num_servos):
            filename = f'servo_{i+1}_data.csv'
            file = open(filename, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 
                           'Current_Target(rad)', 
                           'Actual_Angle(rad)', 
                           'Error(rad)',
                           'Next_Target(rad)'])
            self.data_files.append(file)
            self.csv_writers.append(writer)

        rospy.loginfo("Servo Controller initialized. Operating in radians mode (-1 to 1 rad)")

    def feedback_callback(self, msg):
        """接收舵机反馈的当前角度（弧度）"""
        for i in range(min(len(msg.data), self.num_servos)):
            self.current_angles[i] = msg.data[i]

    def calculate_next_target(self, servo_idx, time_diff):
        """计算下一个时刻的目标角度"""
        current_target = self.target_angles[servo_idx]
        angle_change = self.speeds[servo_idx] * time_diff * self.directions[servo_idx]
        next_target = current_target + angle_change

        # 检查是否需要改变方向
        if next_target >= self.max_angle:
            next_target = self.max_angle
            self.directions[servo_idx] = -1
        elif next_target <= self.min_angle:
            next_target = self.min_angle
            self.directions[servo_idx] = 1

        return next_target

    def update_servo(self, servo_idx):
        """更新单个舵机的目标角度（弧度）"""
        current_time = rospy.Time.now()
        time_diff = (current_time - self.last_pub_times[servo_idx]).to_sec()
        
        # 检查是否需要更新（基于频率）
        if time_diff >= (1.0 / self.frequencies[servo_idx]):
            # 更新当前目标角度
            self.target_angles[servo_idx] = self.next_target_angles[servo_idx]
            listnowangle = self.next_target_angles
            # 计算下一个目标角度
            self.next_target_angles[servo_idx] = self.calculate_next_target(
                servo_idx, 
                1.0 / self.frequencies[servo_idx]  # 使用固定的时间间隔来计算下一目标
            )
    
            # 发布控制命令
            msg = Float64MultiArray()
            msg.data = [0,0,0,0,0,0,0,self.target_angles[servo_idx]]
            self.publishers[servo_idx].publish(msg)
            self.last_pub_times[servo_idx] = current_time
            # 计算误差
            error = listnowangle[servo_idx] - self.current_angles[servo_idx]

            # 记录数据
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.csv_writers[servo_idx].writerow([
                timestamp,
                round(listnowangle[servo_idx], 4),    # 当前目标角度
                round(self.current_angles[servo_idx], 4),   # 实际角度
                round(error, 4),                            # 误差
                round(self.next_target_angles[servo_idx], 4) # 下一时刻目标角度
            ])


    def run(self):
        """主运行循环"""
        rate = rospy.Rate(2000)  # 100Hz的更新率
        
        # 初始化下一目标角度
        for i in range(self.num_servos):
            self.next_target_angles[i] = self.calculate_next_target(
                i, 
                1.0 / self.frequencies[i]
            )
        
        try:
            while not rospy.is_shutdown():
                for i in range(self.num_servos):
                    self.update_servo(i)
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            # 关闭所有数据文件
            for file in self.data_files:
                file.close()
            rospy.loginfo("Data files closed successfully")

    def __del__(self):
        """确保文件被正确关闭"""
        for file in self.data_files:
            if not file.closed:
                file.close()

if __name__ == '__main__':
    try:
        controller = ServoController()
        controller.run()
    except rospy.ROSInterruptException:
        pass