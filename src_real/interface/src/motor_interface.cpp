/*Receive command from leg publisher and convert into gazebo or feite servo command
  Update state of all joint and pub them to leg through topic*/
#include"ros/ros.h"
#include"std_msgs/Float64MultiArray.h"
#include"std_msgs/Float64.h"
#include"control_msgs/JointControllerState.h"
#include"geometry_msgs/WrenchStamped.h"
#include<map>
#include<memory>
// #include"servo_sdk/SMS_STS.h"
#include<math.h>
using namespace std;


typedef map<string,pair<ros::Subscriber,ros::Publisher>> Bridge;
typedef map<string,vector<double>> LegData;
typedef map<string,double> MotorData;

// SMS_STS sm_st;
string port,robot_name;
// u8 ID[9];
// u8 Acc[9];
// s16 Pos[9];
// u16 Vec[9];
int vec, acc, start_tag, end_tag, enable_torq;
vector<int> direction_list;
// map<u8,int> servo_direction;
// map<u8,s16> servo_media;
double rad2byte_rate, byte2rad_rate;

void MotorBridge(const control_msgs::JointControllerStateConstPtr& motor_state_msgs, string& joint_name, shared_ptr<LegData> leg_ptr){
    //E.g. joint_name = rf_thigh
    string leg_sub_name = joint_name.substr(0,2);
    for(int i=0; i<2; ++i) leg_sub_name[i] = leg_sub_name[i]-32;//Change into UpperCase
    string joint_sub_name = joint_name.substr(3);
    int joint_key;
    if(joint_sub_name == "thigh") joint_key = 0;
    else if(joint_sub_name == "knee") joint_key = 1;
    else if(joint_sub_name == "ankle") joint_key = 2;
    else if(joint_sub_name == "foot" ) joint_key = 3;
    else return;
    leg_ptr->find(leg_sub_name)->second[joint_key] = motor_state_msgs->process_value;
}
void MotorFTBridge(const geometry_msgs::WrenchStampedConstPtr& FT_msgs,string& joint_ft_name ,shared_ptr<LegData>& leg_ptr){
    //E.g. joint_ft_name = rf_thigh_sensor
    string leg_sub_name = joint_ft_name.substr(0,2);
    for(int i=0; i<2; ++i) leg_sub_name[i] = leg_sub_name[i] - 32;
    string joint_sub_name = joint_ft_name.substr(3,joint_ft_name.find('_',3)-3);
    int joint_key;
    if(joint_sub_name == "thigh") joint_key = 4;
    else if(joint_sub_name == "knee") joint_key = 5;
    else if(joint_sub_name == "ankle") joint_key = 6;
    else if(joint_sub_name == "foot") joint_key = 7;
    else return;
    leg_ptr->find(leg_sub_name)->second[joint_key] = FT_msgs->wrench.torque.z;
    // cout<< motor_ft_ptr->find(joint_ft_name)->second <<endl;
}
void LegBridge(const std_msgs::Float64MultiArrayConstPtr& leg_command_msgs,string leg_name, shared_ptr<MotorData>& motor_ptr, shared_ptr<LegData> leg_ptr){
    if(leg_command_msgs->data.size()!=8){ //mode sita1 sita2 sita3 dot_sita1 dot_sita2 dot_sita3 sita4
        ROS_WARN("leg_command_msgs type error, expected:[mode, q1,q2,q3,dot_q1,dot_q2,dot_q3,q4]");
        return;
    }
    for(int i=0; i<2; ++i) leg_name[i] = leg_name[i]+32;//Change into LowerCase


    string rf_tag = leg_name.substr(0,1);
    vector<string>joint_name({"thigh","knee","ankle"});
    for(int i=0; i<3; ++i){
        motor_ptr->find(leg_name+"_"+joint_name[i])->second = leg_command_msgs->data[i+1];
        // if(i<3){
        //     motor_ptr->find(leg_name+"_"+joint_name[i])->second = leg_command_msgs->data[i+1];
        //     // cout<<motor_ptr->find(leg_name+"_"+joint_name[i])->second<<endl;
        // }
        // else if(i==3){
        //     //Support State
        //     motor_ptr->find(leg_name+"_foot")->second = (abs(leg_command_msgs->data[3])-abs(leg_command_msgs->data[2]))-M_PI_2;
        //     //Clamp State
        //     // motor_ptr->find(leg_name+"_foot")->second = (abs(leg_command_msgs->data[3])+abs(leg_command_msgs->data[2]))-M_PI;
        //     // motor_ptr->find(leg_name+"_foot")->second = (knee_+ankle_)-M_PI_2;
        // }
    }
    motor_ptr->find(leg_name+"_foot")->second = leg_command_msgs->data.back();
    motor_ptr->find(leg_name+"_ball1")->second = 0;
    motor_ptr->find(leg_name+"_ball2")->second = 0;
}
void LegStatePub(shared_ptr<LegData> leg_state_ptr,shared_ptr<MotorData> motor_ft_ptr ,Bridge& leg_bridge){
    std_msgs::Float64MultiArray leg_msgs;
    leg_msgs.data.resize(8);
    for(auto bridge: leg_bridge){
        for(int i=0; i<8; ++i){
            leg_msgs.data[i]=leg_state_ptr->find(bridge.first)->second[i];
        }
        bridge.second.second.publish(leg_msgs);
    }
}
void MotroCommandPub(shared_ptr<MotorData>& motor_command_ptr, Bridge& motor_bridge){
    std_msgs::Float64 motor_msgs;
    for(auto bridge: motor_bridge){
        motor_msgs.data = motor_command_ptr->find(bridge.first)->second;
        bridge.second.second.publish(motor_msgs);
    }
}


// void RealMotorBridge(const std_msgs::Float64MultiArrayConstPtr& motor_command_msgs, string leg_name){
//     if(motor_command_msgs->data.size() != 4) {
//         ROS_WARN("Wrong type of motor_command_msgs");
//         return;
//     }
//     int servo_tag;
//     string lr_tag = leg_name.substr(0,1);
//     string pose_tag = leg_name.substr(1,1);
//     int joint_tag,lr_num;
//     if(lr_tag == "R") lr_num = 0;
//     else if(lr_tag == "L" ) lr_num = 9;

//     if(pose_tag == "F") joint_tag = 1;
//     else if(pose_tag == "M" ) joint_tag =4;
//     else if(pose_tag == "B" ) joint_tag =7;

//     if(lr_tag == "L" && pose_tag == "F") joint_tag = 7;
//     if(lr_tag == "L" && pose_tag == "B") joint_tag = 1;

//     enable_torq = motor_command_msgs->data[0];
//     for(int i=1; i<4; ++i) {
//         servo_tag = joint_tag+lr_num+i-1;
//         Pos[joint_tag-1+i-1] = servo_media[servo_tag]+servo_direction[servo_tag]*s16(
//             motor_command_msgs->data[i]*rad2byte_rate
//         );
//     }
// }
// void ServoIO(){
//     if(enable_torq == -1){
//         for(int i=0; i<9; ++i){
//             sm_st.EnableTorque(ID[i],0);
//         }
//     }
//     else if(enable_torq == 1 || enable_torq == 2)
//     {
//         sm_st.SyncWritePosEx(ID,sizeof(ID),Pos,Vec,Acc);
//     }
// }

int main(int argc, char** argv){
    ros::init(argc,argv,"motro_interface");
    ros::NodeHandle nh;
    ros::Duration rate(0.001);
    nh.getParam("VEC_SERVO",vec);
    nh.getParam("ACC_SERVO",acc);
    nh.getParam("robot_name",robot_name);
    if(argc != 2){
        ROS_ERROR("Wrong usage of motor_interface, run with 'motor_interface simu/real_l/real_r'");
        return 0;
    }    
    string args = argv[1];

    if(args.substr(0,4) == "simu"){
        cout<<"Motor interface running simu\n";
        //interface with sim
        vector<string> joint_name_list,joint_ft_name_list ,leg_list, joint_list;
        Bridge motor_bridge, leg_bridge;
        map<string,ros::Subscriber> motor_ft_bridge;
        std_msgs::Float64MultiArray leg_msgs;
        std_msgs::Float64 motor_msgs;
        ros::Publisher leg_state_pub, motor_command_pub;
        leg_list.assign({"LF","LM","LB","RF","RM","RB"});
        joint_list.assign({"thigh","knee","ankle","foot","ball1","ball2"});
        joint_name_list.resize(36);
        joint_ft_name_list.resize(12);
        int k=0,m=0;
        for(int i=0; i<6; ++i){
            for(int j=0; j<6; ++j){
                string leg_name = leg_list[i];
                for(char& key:leg_name) key += 32; //change into lower cast
                joint_name_list[k]=leg_name+"_"+joint_list[j];
                k += 1;
                if(j==1 || j==2){
                    joint_ft_name_list[m]=leg_name+"_"+joint_list[j]+"_sensor";
                    m += 1;
                }
            }
        }
        
        auto motor_ptr = make_shared<MotorData>();
        auto leg_ptr = make_shared<LegData>();
        auto motor_ft_ptr = make_shared<MotorData>();
        for(string leg_name: leg_list) leg_ptr->insert(make_pair(leg_name,vector<double>(8,0)));
        for(string joint_name: joint_name_list) motor_ptr->insert(make_pair(joint_name,0));
        for(string joint_ft_name: joint_ft_name_list) motor_ft_ptr->insert(make_pair(joint_ft_name,0));
        
        for(string leg_name: leg_list){
            leg_bridge.insert(
                make_pair(
                    leg_name,
                    make_pair(
                        nh.subscribe<std_msgs::Float64MultiArray>("/"+leg_name+"/sita_des",10,boost::bind(&LegBridge,_1,leg_name,motor_ptr,leg_ptr)),
                        nh.advertise<std_msgs::Float64MultiArray>("/"+leg_name+"/sita_cur",10)
                    )
                )
            );
        }
        for(string joint_name: joint_name_list){
            motor_bridge.insert(
                make_pair(
                    joint_name,
                    make_pair(
                        nh.subscribe<control_msgs::JointControllerState>(
                            "/"+robot_name+"/"+joint_name+"/state",
                            20,boost::bind(&MotorBridge,_1,joint_name,leg_ptr)),
                        nh.advertise<std_msgs::Float64>("/"+robot_name+"/"+joint_name+"/command",10)
                    )
                )
            );
        }

        int i=0;
        for(string joint_ft_name: joint_ft_name_list){
            motor_ft_bridge.insert(
                make_pair(
                    joint_ft_name, nh.subscribe<geometry_msgs::WrenchStamped>(
                        joint_ft_name,10,boost::bind(&MotorFTBridge,_1,joint_ft_name,leg_ptr))
                )
            );
            i++;
        }
        ros::Timer timer_leg_pub = nh.createTimer(ros::Duration(0.01),boost::bind(&LegStatePub,leg_ptr,motor_ft_ptr,leg_bridge));
        ros::Timer timer_motor_pub = nh.createTimer(ros::Duration(0.01),boost::bind(&MotroCommandPub,motor_ptr,motor_bridge));
        ros::spin();
    }
    else if(args.substr(0,4) == "real"){
        ROS_WARN("please uncomment the code and make the include/servo_sdk file");
    //     vector<int>servo_media_byte;
    //     nh.getParam("servo_media_byte",servo_media_byte);
    //     for(int i=1; i<=18; ++i){
    //         servo_media.insert({u8(i),s16(servo_media_byte[i-1])});
    //     }
    //     direction_list.assign({1,2,4,5,7,8,12,15,18});
    //     for(int i=1; i<=18; ++i){
    //         int direction=1;
    //         for(int dir:direction_list){
    //             if(i==dir){
    //                 direction = -1;
    //                 break;
    //             }
    //         }
    //         servo_direction.insert({i,direction});
    //         direction=1;
    //     }
    //     rad2byte_rate = 4096.0/(M_PI*2.0);
    //     byte2rad_rate = (M_PI*2.0)/4096.0;
    //     vector<string> leg_name_list_r({"RF","RM","RB"});
    //     vector<string> leg_name_list_l({"LF","LM","LB"});
    //     Bridge servo_bridge;
    //     shared_ptr<vector<string>> leg_list_ptr;
    //     for(int i=0; i<9; ++i){
    //         Acc[i] = acc;
    //         Vec[i] = vec;
    //     }
    //     if(args.substr(5) == "r"){
    //         start_tag = 1; end_tag = 9;
    //         port = "/dev/servo_r";
    //         for(int i=0; i<9; ++i) ID[i] = u8(i+1);
    //         leg_list_ptr = make_shared<vector<string>>(leg_name_list_r);
    //     }
    //     else if(args.substr(5) == "l"){
    //         start_tag = 10; end_tag = 18;
    //         port = "/dev/servo_l";
    //         for(int i=0; i<9; ++i) ID[i] = u8(i+10);
    //         leg_list_ptr = make_shared<vector<string>>(leg_name_list_l);
    //     }
    //     for(auto leg_name:*leg_list_ptr){
    //         servo_bridge.insert(make_pair(leg_name,make_pair(
    //             nh.subscribe<std_msgs::Float64MultiArray>(leg_name+"/sita_des",10,boost::bind(&RealMotorBridge,_1,leg_name)),
    //             nh.advertise<std_msgs::Float64MultiArray>(leg_name+"/sita_cur",10)
    //         )));
    //     }
    //     if(sm_st.begin(1000000,port.c_str())){
    //         while(ros::ok()){
    //             ServoIO();
    //             rate.sleep();
    //             ros::spinOnce();
    //         }
    //     }
    //     else{
    //         ROS_ERROR("Failed to connect to servo %s",args.substr(5).c_str());
    //         return 0;
        // }        
    }
    return 0;
}