#include"ros/ros.h"
#include"std_msgs/Int16MultiArray.h"
#include"std_msgs/Float64MultiArray.h"
#include"gazebo_msgs/ApplyBodyWrench.h"
#include"gazebo_msgs/GetLinkState.h"
#include"gazebo_msgs/ContactsState.h"
#include"Eigen/Core"
#include"Eigen/LU"
#include"Eigen/Geometry"

using namespace std;

std_msgs::Float64MultiArray suck_state_msg,pv_pub_msg;
std_msgs::Float64MultiArray contacts_state_array;

// std_msgs::Int16MultiArray pv_w_msg;

vector<bool> p_w(6);
vector<bool> v_w(6);
// vector<double> suck_state(6);
vector<double> pv_temp(2);
vector<int> pv_command;

Eigen::Matrix3d m_w_f;
ros::ServiceClient sucker_client, link_state_client;
ros::Publisher contacts_state_pub;
gazebo_msgs::ApplyBodyWrench sucker_srv;
gazebo_msgs::GetLinkState link_state_srv;
vector<string> sucker_name_list({"rf","rm","rb","lf","lm","lb"});
vector<string> leg_name_list;
map<string,ros::Subscriber> contacts_bridge;

string robot_name;
void receive_pv_command(const std_msgs::Int16MultiArrayConstPtr& pv_command_msg, ros::Publisher pv_state_pub){
    int j =0;  uint32_t io_p[6],io_v[6];
    pv_command.assign(pv_command_msg->data.begin(),pv_command_msg->data.end());
    std::swap(pv_command[6],pv_command[10]);
    std::swap(pv_command[7],pv_command[11]);
    for(int i=0; i<pv_command_msg->data.size(); i = i+2){
        p_w[j] = pv_command[i];
        v_w[j] = pv_command[i+1];
        for(char k=0;k<6;k++){
            pv_temp[0] = pv_temp[0]*10+p_w[k];
            pv_temp[1] = pv_temp[1]*10+v_w[k];
        }
        pv_pub_msg.data = pv_temp;
        pv_state_pub.publish(pv_pub_msg);
        pv_temp[0] = 0; 
        pv_temp[1] = 0;
        j=j+1;
    }
}

void serial_suck_callback(const std_msgs::Float64MultiArray::ConstPtr& stm32_adc_msg){
    suck_state_msg.data = stm32_adc_msg->data;
    std::swap(suck_state_msg.data[5],suck_state_msg.data[3]);
}

void pub_suck_read(const ros::TimerEvent& e,ros::Publisher& s_state_pub){
    s_state_pub.publish(suck_state_msg);
}

void Simu_WritePV(const std_msgs::Int16MultiArrayConstPtr& pv_command_msg){   
    for(int i=0; i<6; ++i){
        link_state_srv.request.link_name=sucker_name_list[i];
        link_state_srv.request.reference_frame="world";
        if(link_state_client.call(link_state_srv)){
            Eigen::Quaterniond q_w_f(link_state_srv.response.link_state.pose.orientation.w,
                                     link_state_srv.response.link_state.pose.orientation.x,
                                     link_state_srv.response.link_state.pose.orientation.y,
                                     link_state_srv.response.link_state.pose.orientation.z);
            m_w_f = q_w_f.normalized().toRotationMatrix();
            sucker_srv.request.body_name = robot_name+"::"+sucker_name_list[i];
            sucker_srv.request.reference_frame="world";
            sucker_srv.request.duration = ros::Duration(-1.0);
            Eigen::Vector3d sucker_force;
            if(robot_name == "hexapod_model") sucker_force<<200,0,0;
            else if(robot_name == "hexapod_model921") sucker_force<<80,0,0;
            else if(robot_name == "hexapod_model121") sucker_force<<0,0,-80;
            else if(robot_name == "hex_v4") sucker_force<<0,0,-80;
            if(pv_command_msg->data[i*2]){
                sucker_force = m_w_f*sucker_force;
                sucker_srv.request.wrench.force.x = sucker_force(0);
                sucker_srv.request.wrench.force.y = sucker_force(1);
                sucker_srv.request.wrench.force.z = sucker_force(2);
            }
            else{
                sucker_srv.request.wrench.force.x = 0;
                sucker_srv.request.wrench.force.y = 0;
                sucker_srv.request.wrench.force.z = 0;
            }
            sucker_client.call(sucker_srv);
            // cout<<"call apply force\n";
        }
        else{
            ROS_WARN("call link state failed");
        }
    }
}
void Simu_ReadContact(const gazebo_msgs::ContactsStateConstPtr& contacts_msg, string& suck_name){
    //[suck1 contact number total force, suck2 contact number total force, ...]
    for(int i=0; i<6; ++i){
        cout<<"suck_name="<<suck_name<<endl;
        if (suck_name == leg_name_list[i]){
            contacts_state_array.data[i*2] = contacts_msg->states[0].depths.size();
            contacts_state_array.data[i*2+1] = contacts_msg->states[0].total_wrench.force.z;
            cout<<"contacts_state_array.data["<<i<<"*2]=="<<contacts_state_array.data[i*2];
            break;
        }
    }
}

int main(int argc,char** argv){
    ros::init(argc,argv,"pv_interface");
    ros::NodeHandle nh;
    string args = argv[1];
    nh.getParam("robot_name",robot_name);
    if((args.substr(0,4)!="simu" && args.substr(0,4)!="real" )){
        ROS_ERROR("Wrong usage of pv_interface run with 'pv_interface simu or real'!");
        cout<<args.substr(0,4)<<endl;
        return 0;
    }
    if(args.substr(0,4) == "real"){
        suck_state_msg.data.resize(6);
        ros::Publisher pv_state_pub = nh.advertise<std_msgs::Float64MultiArray>("/serial/pv_state_from_ros",10);
        ros::Subscriber pv_sub = nh.subscribe<std_msgs::Int16MultiArray>("/usr/pv_command",1,boost::bind(&receive_pv_command,_1,pv_state_pub));
        //读串口AD订阅者定义
        ros::Subscriber s_state_sub = nh.subscribe<std_msgs::Float64MultiArray>("/serial/s_state_from_stm32",10,serial_suck_callback);
        // 发布串口IO 定义
        ros::Publisher s_state_pub = nh.advertise<std_msgs::Float64MultiArray>("/serial/s_state",1);
        ros::Timer suck_read = nh.createTimer(ros::Duration(0.02),
                                            boost::bind(&pub_suck_read,_1,s_state_pub));
        ros::spin();
    }
    else if(args.substr(0,4) == "simu"){
        leg_name_list = sucker_name_list;
        sucker_client = nh.serviceClient<gazebo_msgs::ApplyBodyWrench>("/gazebo/apply_body_wrench");
        link_state_client = nh.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");
        if(robot_name == "hexapod_model") for (string& suck:sucker_name_list) suck = "l_"+suck+"_foot";
        else if(robot_name == "hexapod_model921" || robot_name=="hexapod_model121" || robot_name=="hex_v4") for (string& suck:sucker_name_list) suck = "l_"+suck+"_suck";
        ros::Subscriber pv_sub = nh.subscribe<std_msgs::Int16MultiArray>("/usr/pv_command",10,Simu_WritePV);
        
        // contacts_state_array.data.resize(12);
        // contacts_state_pub = nh.advertise<std_msgs::Float64MultiArray>("/contacts_state",10);
        // for(string suck_name: leg_name_list){
        //     //add suck subscriber
        //     contacts_bridge.insert(make_pair(
        //         suck_name,
        //         nh.subscribe<gazebo_msgs::ContactsState>("/hexapod_model921/"+suck_name+"_contacts",10,boost::bind(&Simu_ReadContact,_1,suck_name))
        //     ));
        // }
        // ros::Timer contact_timer = nh.createTimer(ros::Duration(0.01),boost::bind(&Write_ContactState,_1,contacts_state_pub));
        ros::spin();
        // ros::Duration rate(0.01);
        // while (ros::ok())
        // {
        //     contacts_state_pub.publish(contacts_state_array);
        //     rate.sleep();
        //     ros::spinOnce();
        // }
    }

    return 0;
}