#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray

# 定义回调函数
def callback_topic1(data):
    write_to_file('topic1', data.data)

def callback_topic2(data):
    write_to_file('topic2', data.data)

def callback_topic3(data):
    write_to_file('topic3', data.data)

# 将数据写入文件
def write_to_file(topic_name, values):
    with open('output.txt', 'a') as file:
# 格式化数据
        line = f"{topic_name} " + " ".join(map(str, values)) + "\n"
        file.write(line)

def listener():
# 初始化 ROS 节点
    rospy.init_node('multiarray_listener', anonymous=True)

# 订阅三个话题
    rospy.Subscriber("motor_que", Float32MultiArray, callback_topic1)
    rospy.Subscriber("motor_vel", Float32MultiArray, callback_topic2)
    rospy.Subscriber("motor_pos", Float32MultiArray, callback_topic3)

# 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass