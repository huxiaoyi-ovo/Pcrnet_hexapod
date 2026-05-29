/*
舵机出厂速度单位是0.0146rpm，速度为V=1500
*/

#include <iostream>
#include "SCServo.h"
#include <ros/ros.h>
#include <unistd.h>
#include <string>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <thread>
#include <vector>
SCSCL sc;
ros::Publisher servo_pos;
double rad2deg=2.3561;
int servo_speed=1000;
void LF_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(1,40,0);
	else
	{
		static int LF=0;
		ros::param::get("LF_servo",LF);
		sc.WritePos(1, 1000-(msg->data[7]/rad2deg*500+500+LF), 0, servo_speed);
	}
}
void LM_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(2,40,0);
	else
	{
		static int LM=0;
		ros::param::get("LM_servo",LM);
		sc.WritePos(2, 1000-(msg->data[7]/rad2deg*500+500+LM) , 0, servo_speed);
	}
}
void LB_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(3,40,0);
	else
	{
		static int LB=0;
		ros::param::get("LB_servo",LB);
		sc.WritePos(3, 1000-(msg->data[7]/rad2deg*500+500+LB) , 0, servo_speed);
	}
}
void RF_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(4,40,0);
	else
	{
		static int RF=0;
		ros::param::get("RF_servo",RF);
		sc.WritePos(4, msg->data[7]/rad2deg*500+500+RF , 0, servo_speed);
	}
}
void RM_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(5,40,0);
	else
	{
		static int RM=0;
		ros::param::get("RM_servo",RM);
		sc.WritePos(5, msg->data[7]/rad2deg*500+500+RM , 0, servo_speed);
	}
}
void RB_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg){
	if(msg->data[7]==-999)sc.writeByte(6,40,0);
	else 
	{
		static int RB=0;
		ros::param::get("RB_servo",RB);
		sc.WritePos(6, (msg->data[7])/rad2deg*500+500+RB , 0, servo_speed);
	}
	printf("receive in here\n");
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "write_pos");
	ros::NodeHandle n;
	ros::Subscriber LF_sub = n.subscribe("/LF/sita_des", 1, LF_Callback);
	ros::Subscriber LM_sub = n.subscribe("/LM/sita_des", 1, LM_Callback);
	ros::Subscriber LB_sub = n.subscribe("/LB/sita_des", 1, LB_Callback);
	ros::Subscriber RF_sub = n.subscribe("/RF/sita_des", 1, RF_Callback);
	ros::Subscriber RM_sub = n.subscribe("/RM/sita_des", 1, RM_Callback);
	ros::Subscriber RB_sub = n.subscribe("/RB/sita_des", 1, RB_Callback);
	
	servo_pos = n.advertise<std_msgs::Float64MultiArray>("/servo_pos", 1000);
	std::string port = "/dev/ttyTHS0";
	// std::string port = "/dev/ttyUSB0";

    if(!sc.begin(1000000, port.c_str())){
        std::cout<<"Failed to init scscl motor!"<<std::endl;
        return 0;
    }
	// thread t1(&SCServo::ReadPosThread, &sc);
	while(ros::ok()){
		printf("in here -------%d \n",sc.ReadPos(1));
	std_msgs::Float64MultiArray msg;
	msg.data={0,0,0,0,0,0};
	msg.data[0]=float(500-sc.ReadPos(1))/500*2.3561;
	usleep(100);
	msg.data[1]=float(500-sc.ReadPos(2))/500*2.3561;
	usleep(100);
	msg.data[2]=float(500-sc.ReadPos(3))/500*2.3561;
	usleep(100);
	msg.data[3]=float(sc.ReadPos(4)-500)/500*2.3561;
	usleep(100);
	msg.data[4]=float(sc.ReadPos(5)-500)/500*2.3561;
	usleep(100);
	msg.data[5]=float(sc.ReadPos(6)-500)/500*2.3561;
	usleep(100);
	servo_pos.publish(msg);
	ros::spinOnce();
	}
	// ros::spin();
	// sc.WritePos(3, 200 , 0, 500);
	// sleep(1000000);
	sc.end();
	return 1;
}

