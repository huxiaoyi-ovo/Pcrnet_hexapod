#!/usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float32MultiArray
import threading
des = 0.0
cur = 0.0

def des_callback(data):
    des=data.data[1]
def cur_callback(data):
    des=data.data[0]
def thread_job():
    rospy.spin()
if __name__ == "__main__":
    rospy.init_node("delta_node")

    pub = rospy.Publisher("/RF/delta", Float32MultiArray, queue_size=10)
    sub = rospy.Subscriber("/RF/sita_des", Float32MultiArray, des_callback)
    sub = rospy.Subscriber("/RF/sita_cur", Float32MultiArray, cur_callback)
    rate = rospy.Rate(200) # 100hz

    thread = threading.Thread(target=thread_job)
    thread.start()
    while not rospy.is_shutdown():
        data=Float32MultiArray()
        data.data=[des-cur]
        print(des-cur)
        pub.publish(data)
        # self.motor_staus_pub(self.mid_leg,leglist.Middle)
        # self.motor_staus_pub(self.bac_leg,leglist.Back)
        rate.sleep()

