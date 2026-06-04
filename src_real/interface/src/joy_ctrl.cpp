#include"sensor_msgs/Joy.h"
#include"ros/ros.h"
#include<gazebo_msgs/ModelStates.h>
#include<gazebo_msgs/SetModelState.h>
#include<gazebo_msgs/SetPhysicsProperties.h>
#include"geometry_msgs/PoseStamped.h"
#include"interface/joy_command.h"
#include"std_msgs/Int32MultiArray.h"
#include"ros/time.h"
#include"Eigen/Eigen"
#include<unistd.h>
#include<time.h>

#define RAD_STEP 0.174 //直接设定机器人位姿时角度转动的量
#define VEL_MODE 0  //机器人以三角步态正常运动
#define POS_MODE 1  //直接设定机器人位置
using namespace std;
// double VEC_X_MAX, VEC_Y_MAX, VEC_Z_MAX,ORE_W_MAX;
bool init_flag = false;
// bool s_init_flag = false;
bool m_init_flag = false;
bool command_mod = VEL_MODE;
bool enable_gravity = true;
bool call_gravity_flag = false;
vector<double> model_state(7);  //x y z  x y z w
vector<double> axes(8);
vector<int> buttons(15);
ros::Time tic1,tic2;
clock_t cl1,cl2,cl3;
string robot_name;
gazebo_msgs::SetModelState set_model_srv;
Eigen::Vector3d ea;
Eigen::Quaterniond q;
// void GetParam(ros::NodeHandle& nh){
//     nh.getParam("VEC_X_MAX",VEC_X_MAX);
//     nh.getParam("VEC_Y_MAX",VEC_Y_MAX);
//     nh.getParam("VEC_Z_MAX",VEC_Z_MAX);
//     nh.getParam("OMEGA_MOVE",ORE_W_MAX);
//     nh.getParam("robot_name",robot_name);
// }
void joy_callback(const sensor_msgs::JoyConstPtr& joy_msg, ros::ServiceClient& set_physic_client,ros::ServiceClient& set_model_client,
    ros::Publisher& neg_pre_pub)
{
    for(int i=0; i<8; ++i) {
        axes[i] = joy_msg->axes[i];
    }
    for(int j=0; j<15; j++) {
        buttons[j] = joy_msg->buttons[j];
    }
    //button x 设置负压传感器数值
    //cout<<"buttons[3]="<<buttons[3]<<"time="<<double(clock()-cl1)/CLOCKS_PER_SEC<<endl;
    if((buttons[3]==1 || buttons[11]==1)&& double(clock()-cl1)/CLOCKS_PER_SEC>0.01) {
        cl1 = clock();
        std_msgs::Int32MultiArray neg_msgs;
        if(buttons[3]){
            neg_msgs.data.assign({80,80});
        }
        else if(buttons[11]){
            neg_msgs.data.assign({2800,2800});
        }
        neg_pre_pub.publish(neg_msgs);
        ROS_WARN("Publish negative_pressure command, p1=%d, p2=%d",neg_msgs.data[0],neg_msgs.data[1]);

    }
    // cout<<axes[0]<<" "<<axes[1]<<" "<<axes[2]<<" "<<axes[3]<<endl;
    //中间左上角按键切换步态或者上帝模式
    if(buttons[10] ==1 && double(clock()-cl1)/CLOCKS_PER_SEC > 0.01 ){
        cl1 = clock();
        command_mod = !command_mod;
        if(command_mod == VEL_MODE)ROS_WARN("vel mode now!");
        else if(command_mod == POS_MODE ) ROS_WARN("pos mode now!");
    }
    //中间右上按键添加/取消重力
    if(buttons[11] == 1 && double(clock()-cl3)/CLOCKS_PER_SEC > 0.01 ){
        cl3 = clock();
        enable_gravity = !enable_gravity;
        if(enable_gravity)ROS_WARN("enable gravity now!");
        else ROS_WARN("disable gravity now!");
        call_gravity_flag = true;
    }
    if(call_gravity_flag){
        gazebo_msgs::SetPhysicsProperties set_physic_srv;
        set_physic_srv.request.time_step = 0.001;
        set_physic_srv.request.max_update_rate = 1200.0;
        set_physic_srv.request.ode_config.auto_disable_bodies = false;
        set_physic_srv.request.ode_config.sor_pgs_iters = 50;
        set_physic_srv.request.ode_config.sor_pgs_w = 1.3;
        set_physic_srv.request.ode_config.contact_surface_layer=0.001;
        set_physic_srv.request.ode_config.contact_max_correcting_vel = 100.0;
        set_physic_srv.request.ode_config.erp = 0.2;
        set_physic_srv.request.ode_config.max_contacts = 20;

        set_physic_srv.request.gravity.z=-9.8*double(enable_gravity);
        set_physic_client.call(set_physic_srv);
        call_gravity_flag = false;
    }
    if(command_mod==POS_MODE){
        bool set_state=false;
        if((buttons[3]==1 || buttons[0]==1)&& double(clock()-cl2)/CLOCKS_PER_SEC >0.01){
            set_state=true;
            cl2 = clock();
            for(int i=0; i<6; ++i)model_state[i]=0;
            if(buttons[3]==1){
                //"button "X" set model back to support state
                // model_state[0]=0.01; model_state[1]=-5.6; model_state[2]=0.47;
                // ea<<-M_PI_2,0,1.25;
                //button "X" set to the A* search initial state
                model_state[0]=0.8; model_state[1]=-(6+3); model_state[2]=2.3;
                ea<<0,0,0;
            }
            else if(buttons[0]==1){
                //"button "A" set model back to clamp state
                // model_state[0]=9.409498; model_state[1]=-9.860572; model_state[2]=2.067374;
                // ea<<-M_PI_2,0,0;  
                model_state[0]=0.2; model_state[1]=-0.89; model_state[2]=0.47;
                ea<<-M_PI_2,0,1.25;        
            }
        }
        else{
            int axes_nd = ceil(abs(axes[0])+abs(axes[1])+abs(axes[2])+abs(axes[3])+abs(axes[6])+abs(axes[7]));
            if(axes_nd){
                set_state=true;
                model_state[0] = model_state[0] + axes[0]*-0.1;
                model_state[1] = model_state[1] + axes[1]*0.1;
                model_state[2] = model_state[2] + axes[3]*0.1;
                q.w()=model_state[6];
                q.x()=model_state[3];
                q.y()=model_state[4];
                q.z()=model_state[5];
                Eigen::Matrix3d rx = q.toRotationMatrix();
                ea = rx.eulerAngles(2,1,0);
                ea[0] = ea[0] + axes[6]*RAD_STEP;
                ea[1] = ea[1] + axes[2]*-RAD_STEP;
                ea[2] = ea[2] + axes[7]*-RAD_STEP;
                for(int i=0; i<3; ++i){
                    if( ea[i] >= 2*M_PI)ea[i] = ea[i] - 2*M_PI;
                    else if(ea[i] <= -2*M_PI) ea[i] = ea[i] + 2*M_PI; 
                }
            }
        }
        if(set_state){
            q = Eigen::AngleAxisd(ea[0],Eigen::Vector3d::UnitZ())*
                Eigen::AngleAxisd(ea[1],Eigen::Vector3d::UnitY())*
                Eigen::AngleAxisd(ea[2],Eigen::Vector3d::UnitX());
            set_model_srv.request.model_state.pose.position.x = model_state[0];
            set_model_srv.request.model_state.pose.position.y = model_state[1];
            set_model_srv.request.model_state.pose.position.z = model_state[2];                    
            set_model_srv.request.model_state.pose.orientation.x=q.x();
            set_model_srv.request.model_state.pose.orientation.y=q.y();
            set_model_srv.request.model_state.pose.orientation.z=q.z();
            set_model_srv.request.model_state.pose.orientation.w=q.w();
            model_state[3]=q.x(); model_state[4]=q.y(); model_state[5]=q.z(); model_state[6]=q.w();
            set_model_client.call(set_model_srv);   
        }
    }    
}

//只做机器人当前状态更新
void model_state_callback(const gazebo_msgs::ModelStatesConstPtr& model_state_msg ){
    int i =0;
    for(int i= 0; i<model_state_msg->name.size(); ++i){
        if(model_state_msg->name[i] == robot_name){
            model_state[0] = model_state_msg->pose[i].position.x;
            model_state[1] = model_state_msg->pose[i].position.y;
            model_state[2] = model_state_msg->pose[i].position.z;
            model_state[3] = model_state_msg->pose[i].orientation.x;
            model_state[4] = model_state_msg->pose[i].orientation.y;
            model_state[5] = model_state_msg->pose[i].orientation.z;
            model_state[6] = model_state_msg->pose[i].orientation.w;
            break;
        }
    }
}

void pub_joy_command(const ros::TimerEvent& time_event, ros::Publisher& com_pub){
    //stop attach
    //disable torque or stop pump
    if(command_mod == VEL_MODE){
        interface::joy_command joy_setpoint;
        double cmd_x = abs(axes[0]) <= 0.15 ? 0.0 : -axes[0];
        double cmd_y = abs(axes[1]) <= 0.15 ? 0.0 : axes[1];
        double cmd_z = abs(axes[3]) <= 0.15 ? 0.0 : -axes[3];
        double cmd_yaw = abs(axes[2]) <= 0.15 ? 0.0 : axes[2];
        bool vel_axes_active = cmd_x != 0.0 || cmd_y != 0.0 || cmd_z != 0.0 || cmd_yaw != 0.0;
        if(buttons[6]){
            init_flag=0;
            // s_init_flag = 0;
            m_init_flag = 0;
            joy_setpoint.disable_pump = 1;
            com_pub.publish(joy_setpoint);
        }
        else if(buttons[7]){
            init_flag=0;
            // s_init_flag = 0;
            m_init_flag = 0;
            joy_setpoint.disable_torque = 1;
            com_pub.publish(joy_setpoint);
        }
        else if(buttons[13]){
            init_flag=0;
            // s_init_flag = 0;
            m_init_flag = 0;
            joy_setpoint.action_valve = 1;
            com_pub.publish(joy_setpoint);
        }
        else if(buttons[1]){
            //button B enter initial set
            init_flag = 1;
            // s_init_flag = 0;
            m_init_flag = 0;
            joy_setpoint.set_init=1;
            com_pub.publish(joy_setpoint);
        }
        // else if(buttons[3]&&init_flag){
        //     //standing init "button X"
        //     joy_setpoint.set_init=1;
        //     joy_setpoint.standing = 1;
        //     s_init_flag = 1;
        //     com_pub.publish(joy_setpoint);
        // }
        else if(buttons[4]&&init_flag){
            //moving init "button Y"
            joy_setpoint.set_init=1;
            joy_setpoint.moving = 1;
            m_init_flag = 1;
            com_pub.publish(joy_setpoint);
        }        
        else if(buttons[0]&&m_init_flag){
            //button A publish stop command
            joy_setpoint.stop=1;
            com_pub.publish(joy_setpoint);
        }
        // else if(buttons[14]&&init_flag&&!m_init_flag){
            //只有初始化按下B键才可以进行切换模式

        else if(buttons[14]&&m_init_flag&&!vel_axes_active){
            //按下右侧轮盘按钮即可切换
            joy_setpoint.change_mode=1;
            com_pub.publish(joy_setpoint);
            //切换模式后需要重新初始化
            // init_flag=0;
        }        
        else if(m_init_flag){
            //calculate x y desvelocity
            if(vel_axes_active){
                joy_setpoint.x_vec = cmd_x;
                joy_setpoint.y_vec = cmd_y;
                joy_setpoint.z_vec = cmd_z;
                joy_setpoint.w_twist = cmd_yaw;
                // joy_setpoint.set_init=1;
                // joy_setpoint.moving = 1;
                com_pub.publish(joy_setpoint);
            }
        }        
    }

}

int main(int argc, char** argv){
    ros::init(argc,argv,"joy_ctrl");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    std::string command_topic;
    private_nh.param<std::string>("command_topic", command_topic, "/usr/command_manual");
    // GetParam(nh);
    set_model_srv.request.model_state.model_name=robot_name;
    set_model_srv.request.model_state.reference_frame="world";    
    tic1 = ros::Time::now();
    tic2 = ros::Time::now();
    cl1 = clock(); cl2 = clock(); cl3=clock();
    // ros::Publisher com_pub = nh.advertise<geometry_msgs::PoseStamped>("usr/command",1);
    ros::Publisher com_pub = nh.advertise<interface::joy_command>(command_topic,10);
    ROS_WARN("joy_ctrl publishes joy_command to %s", command_topic.c_str());
    ros::Publisher neg_pre_pub = nh.advertise<std_msgs::Int32MultiArray>("/negative_pressure",1);
    ros::ServiceClient set_model_client = nh.serviceClient<gazebo_msgs::SetModelState>("gazebo/set_model_state");
    ros::ServiceClient set_physic_client = nh.serviceClient<gazebo_msgs::SetPhysicsProperties>("gazebo/set_physics_properties");
    ros::Subscriber joy_sub = nh.subscribe<sensor_msgs::Joy>("/joy",10,boost::bind(&joy_callback,_1,set_physic_client,set_model_client,neg_pre_pub));
    ros::Subscriber model_state_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states",1,model_state_callback);
    ros::Timer pub_commad = nh.createTimer(ros::Duration(0.02),boost::bind(&pub_joy_command,_1,com_pub));
    // ros::Duration rate(0.02);
    // sleep(3);
    // ROS_WARN("joy contol ready~");
    while(ros::ok()){
        //接受手柄信息
        usleep(0.01*1000000);
        ros::spinOnce();
    }

    return 0;
}
