#!/usr/bin/python3
# import DrEmpower_can as servo

#发布关节期望角度
import time
import rospy
from std_msgs.msg import Float64MultiArray

rospy.init_node("pub_angle")

name_list=["RF","RM","RB","LF","LM","LB"]
pub_list=[]
pos_msg = Float64MultiArray()
pos_msg.data=[2,0,0.2,-1.1,0,0,0,-999]
for name in name_list:
    pub_list.append(rospy.Publisher("/"+name+"/sita_des",Float64MultiArray,queue_size=10))

rate=rospy.Rate(50)

while not rospy.is_shutdown():
    for i,name in enumerate(name_list):
        if name== "RF" or name == "LB":
            pos_msg.data[1]=0.5
        elif name == "LF" or name == "RB":
            pos_msg.data[1]=-0.5
        else:
            pos_msg.data[1]=0.0
        pub_list[i].publish(pos_msg)
    rate.sleep()
    
