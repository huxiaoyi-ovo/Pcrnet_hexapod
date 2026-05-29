// Visualize working range of each leg as polyhedron and trajectory of foot end.
#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP
#include<ros/ros.h>
#include<sensor_msgs/JointState.h>
#include<control_msgs/JointControllerState.h>
#include<geometry_msgs/PoseStamped.h>
#include<geometry_msgs/Point.h>
#include<tf2_msgs/TFMessage.h>
#include<gazebo_msgs/ModelStates.h>
#include<visualization_msgs/Marker.h>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<Eigen/Geometry>
#include<fstream>
#include"quickhull.hpp"
using namespace std;

class VISUALIZER{
private:
    ros::NodeHandle nh;
    //construct polyhedron
    vector<double> unfold_vertexes; //Vertexes of working range polyhedron of leg
    vector<Eigen::Vector3d> vertexes;
    vector<Eigen::Matrix3Xd> vPolys;
    vector<Eigen::Matrix3Xd> all_vPolys;
    vector<Eigen::Matrix4d> R_T_Bs;
    double polyhedron_h1, polyhedron_h2, polyhedron_h3;

    //vis polyhedron relative
    ros::Publisher edge_pub;
    ros::Publisher mesh_pub;
    ros::Publisher traj_pub;
    ros::Publisher e_cur_pub;
    ros::Publisher surface_pub;
    visualization_msgs::Marker meshMarker;
    visualization_msgs::Marker edgeMarker;
    visualization_msgs::Marker PointMarker;
    visualization_msgs::Marker surfaceMarker;
    ros::Timer polyhedron_timer;
    //vis robot state relative
    ros::Publisher joints_state_pub;
    ros::Publisher robot_state_pub;
    ros::Publisher robot_des_state_pub;
    ros::Publisher robot_euler_pub;
    ros::Publisher blade_state_pub;
    ros::Subscriber fitten_plane_sub;
    
    ros::Subscriber robot_state_sub;
    ros::Subscriber robot_des_state_sub;
    ros::Timer timer1,timer2;
    sensor_msgs::JointState joints_state_msg;
    geometry_msgs::PoseStamped euler_msgs;

    tf2_msgs::TFMessage robot_state_msg;
    tf2_msgs::TFMessage robot_des_state_msg;
    map<string,ros::Subscriber> motors_bridge;

    Eigen::Matrix4d W_T_R;
    map<string,visualization_msgs::Marker> leg2marker;
    vector<string> joint_names,leg_names; 
    string robot_name;
    string motor_name;
    fstream body_pos_log;
    Eigen::Vector3d euler_angles_ave;
    int count;
    ros::Time tic;
public:
    VISUALIZER(const ros::NodeHandle & nh_): nh(nh_){

    }
    inline void TrajPointVisInit(){
        //trajectory
        traj_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/traj",100);
        e_cur_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/e_cur",100);
        leg_names={"RF","RM","RB","LF","LM","LB"};
        visualization_msgs::Marker traj_marker;
        traj_marker.header.frame_id="body";
        traj_marker.id=0;
        traj_marker.type=visualization_msgs::Marker::LINE_STRIP;
        traj_marker.action=visualization_msgs::Marker::ADD;
        traj_marker.scale.x=0.005;
        traj_marker.color.a=1.0;
        traj_marker.color.r=0.8;
        traj_marker.color.g=0.5;
        traj_marker.color.b=0.0;
        traj_marker.pose.orientation.w=1.0;
        traj_marker.pose.orientation.x=0.0;
        traj_marker.pose.orientation.y=0.0;
        traj_marker.pose.orientation.z=0.0;
        for(string name:leg_names){
            traj_marker.ns=name+"_traj";
            leg2marker.insert(make_pair(name,traj_marker));
        }

        PointMarker.header.frame_id="body";
        PointMarker.ns="point";
        PointMarker.type=visualization_msgs::Marker::SPHERE;
        PointMarker.action=visualization_msgs::Marker::ADD;
        PointMarker.scale.x=0.05;
        PointMarker.scale.y=0.05;
        PointMarker.scale.z=0.05;
        PointMarker.color.a=1.0;
        PointMarker.color.r=1.0;
        PointMarker.color.g=0.0;
        PointMarker.color.b=0.0;
    }
    inline void VisTraj(vector<Eigen::Vector3d>&trajs,Eigen::Matrix4d& R_T_B, string& name){
        visualization_msgs::Marker& traj_marker = leg2marker[name];
        Eigen::Affine3d R_A_B(R_T_B);
        geometry_msgs::Point p_msg;
        traj_marker.action=visualization_msgs::Marker::DELETE;
        traj_pub.publish(traj_marker);
        traj_marker.points.clear();
        for(auto point:trajs){
            Eigen::Vector3d R_p=R_A_B*point;
            p_msg.x=R_p[0];
            p_msg.y=R_p[1];
            p_msg.z=R_p[2];
            traj_marker.points.push_back(p_msg);
        }
        traj_marker.action=visualization_msgs::Marker::ADD;
        traj_marker.header.stamp=ros::Time::now();
        traj_pub.publish(traj_marker);
    }
    inline void VisPoint(Eigen::Vector3d& point, Eigen::Matrix4d& R_T_B){

    }
    inline void StateVisInit(){
        //log file for body pose
        body_pos_log.open("/home/val/BIH_ws/hexapod_sim/bag/body_pos_real.txt",ios::out);
        body_pos_log.close();
        euler_angles_ave<<0,0,0;
        count=1;
        tic=ros::Time::now();
        //polyherdon
        edge_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/edge",100);
        mesh_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/mesh",100);
        this->ConstructPolys();
        polyhedron_timer= nh.createTimer(ros::Duration(0.05),&VISUALIZER::VisPolytope,this);
        //fitten plane
        surface_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/surface",100);
        fitten_plane_sub = nh.subscribe<geometry_msgs::PoseStamped>("/fitten_plane",10,&VISUALIZER::Fitten_Plane,this);
        surfaceMarker.header.frame_id="dummy_link";
        surfaceMarker.ns="surface";
        surfaceMarker.type=visualization_msgs::Marker::TRIANGLE_LIST;
        surfaceMarker.action=visualization_msgs::Marker::ADD;
        surfaceMarker.scale.x=1;
        surfaceMarker.scale.y=1;
        surfaceMarker.scale.z=1;
        surfaceMarker.color.a=0.05;
        surfaceMarker.color.r=0.00;
        surfaceMarker.color.g=0.00;
        surfaceMarker.color.b=1.00;
        surfaceMarker.id=1;
        surfaceMarker.pose.orientation.w=1.0;



        //joints states
        leg_names={"rf","rm","rb","lf","lm","lb"};
        joint_names={"thigh","knee","ankle","foot","ball1","ball2","suck"};
        joints_state_msg.name.resize(leg_names.size()*joint_names.size());
        joints_state_msg.position.resize(leg_names.size()*joint_names.size());
        int i = 0;
        nh.getParam("robot_name",robot_name);
        for(string leg:leg_names){
            for(string joint:joint_names){
                motor_name = "j_"+leg+"_"+joint;
                joints_state_msg.name[i]=motor_name;
                motors_bridge.insert(make_pair(
                    motor_name,
                    nh.subscribe<control_msgs::JointControllerState>("/"+robot_name+"/"+leg+"_"+joint+"/state",10,boost::bind(&VISUALIZER::MotorBridge,this,_1,motor_name))));
                i+=1;
            }
        }
        //visualize joint state in rviz
        joints_state_pub = nh.advertise<sensor_msgs::JointState>("/joint_states",10);
        timer1 = nh.createTimer(ros::Duration(0.01),&VISUALIZER::JointState_timer,this);
        //visualize robot state
        robot_state_msg.transforms.resize(1);
        robot_state_msg.transforms[0].header.frame_id="dummy_link";
        robot_state_msg.transforms[0].child_frame_id = "body";
        robot_state_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states",10,&VISUALIZER::RobotState,this);        
        robot_des_state_msg.transforms.resize(1);
        robot_des_state_msg.transforms[0].header.frame_id="dummy_link";
        robot_des_state_msg.transforms[0].child_frame_id = "des_state";
        robot_state_pub = nh.advertise<tf2_msgs::TFMessage>("/tf",10);
        robot_des_state_pub = nh.advertise<tf2_msgs::TFMessage>("/tf",10);
        blade_state_pub = nh.advertise<tf2_msgs::TFMessage>("/tf",10);
        robot_des_state_sub = nh.subscribe<geometry_msgs::PoseStamped>("/robot_des_state",10,&VISUALIZER::RobotDesState,this);

        timer2=nh.createTimer(ros::Duration(0.1),&VISUALIZER::BladeState_timer,this);
        robot_euler_pub = nh.advertise<geometry_msgs::PoseStamped>("/robot_euler",1);

    }
    inline void ConstructPolys(){
        //calculate R_T_B for every six legs;
        double ofx, ofy, oflen;
        nh.getParam("OFFSET_X",ofx);
        nh.getParam("OFFSET_Y",ofy);
        nh.getParam("OFFSET_LEN",oflen);
        vector<double> x_array({ofx,oflen,ofx,-ofx,-oflen,-ofx});
        vector<double> y_array({ofy,0,-ofy,ofy,0,-ofy});
        R_T_Bs.resize(6);
        for(int i=0; i<6; ++i){
            R_T_Bs[i] = Eigen::Matrix4d::Identity();
            R_T_Bs[i].block<3,1>(0,3) = Eigen::Vector3d(x_array[i],y_array[i],0);
            if(i>2){
                R_T_Bs[i](0,0) = -1;
                R_T_Bs[i](1,1) = -1;
            }
        }

        //get vertex from param and construct 4 V-representation convex polyhedron
        nh.getParam("vertexes",unfold_vertexes);
        nh.getParam("polyhedron_h1",polyhedron_h1);
        nh.getParam("polyhedron_h2",polyhedron_h2);
        nh.getParam("polyhedron_h3",polyhedron_h3);
        vertexes.resize((unfold_vertexes.size()/3)*2-2);
        int j=0;
        //construct vertexes from unfold_vertexes
        for(int i=0; i<unfold_vertexes.size(); i+=3){
            if(i<20){
                vertexes[j]<<unfold_vertexes[i],-polyhedron_h1,unfold_vertexes[i+2];
                vertexes[j+1]<<unfold_vertexes[i],polyhedron_h1,unfold_vertexes[i+2];
                j += 2;
            }
            else if(i == 21 || i == 24){
                vertexes[j]<<unfold_vertexes[i],unfold_vertexes[i+1],unfold_vertexes[i+2];
                j++;
            }
            else if( i== 27 ){
                vertexes[j]<<unfold_vertexes[i],polyhedron_h2,unfold_vertexes[i+2];
                vertexes[j+1]<<unfold_vertexes[i],-polyhedron_h2,unfold_vertexes[i+2];
                j += 2;
            }
            else if( i==30 ){
                vertexes[j]<<unfold_vertexes[i],polyhedron_h3,unfold_vertexes[i+2];
                vertexes[j+1]<<unfold_vertexes[i],-polyhedron_h3,unfold_vertexes[i+2];
                j += 2;
            }
        }
        vPolys.resize(4);
        vPolys[0].resize(3,8); vPolys[1].resize(3,10); vPolys[2].resize(3,6); vPolys[3].resize(3,6);
        vector<Eigen::Vector3d> R_vertexes((unfold_vertexes.size()/3)*2-2);
        for(int id=0; id<6; ++id){
            for(int j=0; j<vertexes.size(); ++j){
                Eigen::Vector4d norm_vertex(vertexes[j].x(),vertexes[j].y(),vertexes[j].z(),1);
                norm_vertex = R_T_Bs[id]*norm_vertex;
                R_vertexes[j] << norm_vertex(0),norm_vertex(1),norm_vertex(2);
            }
            for(int i=0; i<vertexes.size(); ++i){
                if(i<8){
                    vPolys[0].col(i) = R_vertexes[i];
                }
                if(i>3 && i<14){
                    vPolys[1].col(i-4) = R_vertexes[i];
                }
                if(i>9 && i< 16){
                    vPolys[2].col(i-10) = R_vertexes[i];
                }
                if(i>13){
                    vPolys[3].col(i-14) = R_vertexes[i];
                }
            }
            for(int i=0; i<4; ++i){
                all_vPolys.push_back(vPolys[i]);
            }
        }
        
        Eigen::Matrix3Xd mesh(3,0), curTris(3,0), oldTris(3,0);
        for (size_t id =0; id<all_vPolys.size(); id++){
            oldTris = mesh;
            Eigen::Matrix3Xd vPoly;
            vPoly = all_vPolys[id];
            quickhull::QuickHull<double> tinyQH;
            const auto polyhull = tinyQH.getConvexHull(vPoly.data(),vPoly.cols(),false,true);
            const auto &idxBuffer = polyhull.getIndexBuffer();
            int hNum = idxBuffer.size()/3;

            curTris.resize(3,hNum * 3);
            for (int i=0; i<hNum*3; ++i){
                curTris.col(i) = vPoly.col(idxBuffer[i]);
            }
            mesh.resize(3,oldTris.cols()+curTris.cols());
            mesh.leftCols(oldTris.cols()) = oldTris;
            mesh.rightCols(curTris.cols()) = curTris;
        }
        // RVIZ support tris for visualization
        meshMarker.header.stamp = ros::Time::now();
        meshMarker.id = 0;
        meshMarker.header.frame_id = "body";
        meshMarker.pose.orientation.w = 1.00;
        meshMarker.action = visualization_msgs::Marker::ADD;
        meshMarker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        meshMarker.ns = "mesh";
        meshMarker.color.r = 0.00;
        meshMarker.color.g = 0.00;
        meshMarker.color.b = 1.00;
        meshMarker.color.a = 0.05;
        meshMarker.scale.x = 1;
        meshMarker.scale.y = 1;
        meshMarker.scale.z = 1;

        edgeMarker = meshMarker;
        edgeMarker.type = visualization_msgs::Marker::LINE_LIST;
        edgeMarker.ns = "edge";
        edgeMarker.color.r = 0.00;
        edgeMarker.color.g = 1.00;
        edgeMarker.color.b = 1.00;
        edgeMarker.color.a = 0.1;
        edgeMarker.scale.x = 0.001;
        edgeMarker.scale.y = 0.001;
        edgeMarker.scale.z = 0.001;

        geometry_msgs::Point point;

        int ptnum = mesh.cols();

        for (int i = 0; i < ptnum; i++)
        {
            point.x = mesh(0, i);
            point.y = mesh(1, i);
            point.z = mesh(2, i);
            meshMarker.points.push_back(point);
        }

        for (int i = 0; i < ptnum / 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                point.x = mesh(0, 3 * i + j);
                point.y = mesh(1, 3 * i + j);
                point.z = mesh(2, 3 * i + j);
                edgeMarker.points.push_back(point);
                point.x = mesh(0, 3 * i + (j + 1) % 3);
                point.y = mesh(1, 3 * i + (j + 1) % 3);
                point.z = mesh(2, 3 * i + (j + 1) % 3);
                edgeMarker.points.push_back(point);
            }
        }
    }
    inline void VisPolytope(const ros::TimerEvent& e){
        meshMarker.header.stamp = ros::Time::now();
        edgeMarker.header.stamp = ros::Time::now();
        mesh_pub.publish(meshMarker);
        edge_pub.publish(edgeMarker);
    }   
    inline void RobotState(const gazebo_msgs::ModelStatesConstPtr& msg){
        int tag=0;
        for(int j=0; j< msg->name.size(); j++){
            if (msg->name[j] == robot_name){
                tag = j;
                break;
            }
        }
        robot_state_msg.transforms[0].header.stamp = ros::Time::now();
        robot_state_msg.transforms[0].transform.translation.x = msg->pose[tag].position.x;   
        robot_state_msg.transforms[0].transform.translation.y = msg->pose[tag].position.y;   
        robot_state_msg.transforms[0].transform.translation.z = msg->pose[tag].position.z;  

        robot_state_msg.transforms[0].transform.rotation.x = msg->pose[tag].orientation.x; 
        robot_state_msg.transforms[0].transform.rotation.y = msg->pose[tag].orientation.y; 
        robot_state_msg.transforms[0].transform.rotation.z = msg->pose[tag].orientation.z; 
        robot_state_msg.transforms[0].transform.rotation.w = msg->pose[tag].orientation.w; 
        robot_state_pub.publish(robot_state_msg);
        Eigen::Quaterniond q(msg->pose[tag].orientation.w,msg->pose[tag].orientation.x,msg->pose[tag].orientation.y,msg->pose[tag].orientation.z);
        Eigen::Matrix3d rot= q.toRotationMatrix();
        Eigen::Vector3d euler_angle=rot.eulerAngles(2,1,0).array()/M_PI*180.0;
        W_T_R.block<3,3>(0,0)=rot;
        W_T_R.block<3,1>(0,3)=Eigen::Vector3d(msg->pose[tag].position.x,msg->pose[tag].position.y,msg->pose[tag].position.z);
        
        // euler_angle(2) = std::atan2(rot(2, 1), rot(2, 2));
        // euler_angle(1) = std::atan2(-rot(2, 0), std::sqrt(rot(2, 1) * rot(2, 1) + rot(2, 2) * rot(2, 2)));
        // euler_angle(0) = std::atan2(rot(1, 0), rot(0, 0));
        // for(int k=0;k<3;k++){
            // if(abs(euler_angle(k))>M_PI/2.0){
            //     if(euler_angle(k)>0) euler_angle(k)-=M_PI;
            //     else euler_angle(k)+=M_PI;
            // }
        // }
        euler_angles_ave+=euler_angle;
        count+=1;
        if((ros::Time::now()-tic).toSec()>=0.01){
            body_pos_log.open("/home/val/BIH_ws/hexapod_sim/bag/body_pos_real.txt",ios::app);

            euler_angles_ave=euler_angles_ave/double(count);
            // body_pos_log<<ros::Time::now().toSec()<<" "<<euler_angles_ave.transpose()<<endl;
            body_pos_log<<std::fixed<<ros::Time::now().toSec()<<" "<<q.coeffs().transpose()<<endl;
            count=0;
            tic=ros::Time::now();
            euler_msgs.pose.position.x=euler_angles_ave(0);
            euler_msgs.pose.position.y=euler_angles_ave(1);
            euler_msgs.pose.position.z=euler_angles_ave(2);
            euler_msgs.header.stamp = ros::Time::now();
            robot_euler_pub.publish(euler_msgs);
            euler_angles_ave<<0,0,0;
            
            body_pos_log.close();
        }
    } 
    inline void RobotDesState(const geometry_msgs::PoseStampedConstPtr& msg){
        //Q1代表了从dummy_link到body的变换
        Eigen::Quaterniond Q1(robot_state_msg.transforms[0].transform.rotation.w,
                              robot_state_msg.transforms[0].transform.rotation.x,
                              robot_state_msg.transforms[0].transform.rotation.y,
                              robot_state_msg.transforms[0].transform.rotation.z);
        Eigen::Affine3d trans1=Eigen::Translation3d(robot_state_msg.transforms[0].transform.translation.x,
                                                    robot_state_msg.transforms[0].transform.translation.y,
                                                    robot_state_msg.transforms[0].transform.translation.z)*Q1;
        //Q2代表了从body到des_state的变换
        Eigen::Quaterniond Q2(msg->pose.orientation.w,msg->pose.orientation.x,msg->pose.orientation.y,msg->pose.orientation.z);
        Eigen::Affine3d trans2 = Eigen::Translation3d(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z)*Q2;
        
        //Q3代表了从dummy_link到des_state的变换
        Eigen::Affine3d trans3 = trans1*trans2;
        Eigen::Quaterniond Q3(trans3.rotation());
        robot_des_state_msg.transforms[0].header.stamp = ros::Time::now();
        // robot_des_state_msg.transforms[0].transform.translation.x = trans3.translation().x();
        // robot_des_state_msg.transforms[0].transform.translation.y = trans3.translation().y();
        // robot_des_state_msg.transforms[0].transform.translation.z = trans3.translation().z();
        robot_des_state_msg.transforms[0].transform.rotation.x = Q3.x();
        robot_des_state_msg.transforms[0].transform.rotation.y = Q3.y();
        robot_des_state_msg.transforms[0].transform.rotation.z = Q3.z();
        robot_des_state_msg.transforms[0].transform.rotation.w = Q3.w();

        robot_des_state_pub.publish(robot_des_state_msg);

    }
    inline void Fitten_Plane(const geometry_msgs::PoseStampedConstPtr& msg){
        double a=msg->pose.orientation.x;
        double b=msg->pose.orientation.y;
        double c=msg->pose.orientation.z;
        double d=msg->pose.orientation.w;
        geometry_msgs::Point point;
        Eigen::Vector4d w_point;

        surfaceMarker.color.r=0.00;
        surfaceMarker.color.g=1.00;
        surfaceMarker.color.b=0.00;
        surfaceMarker.color.a=1.0;

        vector<double> x_array({0.1,0.1,-0.1, 0.1,-0.1,-0.1});
        vector<double> y_array({0.2,-0.2,-0.2, 0.2,-0.2,0.2});
        // vector<double> x_array({0.1,0.1,-0.1, -0.1});
        // vector<double> y_array({0.2,-0.2,-0.2, 0.2});        
        for(int i=0;i<6;i++){
            w_point=W_T_R*(Eigen::Vector4d (x_array[i],y_array[i],-(d+a*x_array[i]+b*y_array[i])/c,1));
            point.x=w_point.x();
            point.y=w_point.y();
            point.z=w_point.z();
            surfaceMarker.points.push_back(point);
        }
        surfaceMarker.header.stamp = ros::Time::now();
        surface_pub.publish(surfaceMarker);
    }
    
    inline void JointState_timer(const ros::TimerEvent& e){
        joints_state_msg.header.stamp = ros::Time::now();
        joints_state_pub.publish(joints_state_msg);
    }
    inline void BladeState_timer(const ros::TimerEvent& e){
        tf2_msgs::TFMessage blade_state_msg;
        blade_state_msg.transforms.resize(1);
        blade_state_msg.transforms[0].header.stamp = ros::Time::now();
        blade_state_msg.transforms[0].header.frame_id = "dummy_link";
        blade_state_msg.transforms[0].child_frame_id = "curve_surface";
        Eigen::AngleAxisd rollAngle(1.57,Eigen::Vector3d::UnitX());
        Eigen::Matrix3d R_roll=rollAngle.toRotationMatrix();
        Eigen::Quaterniond Q_roll(R_roll);
        blade_state_msg.transforms[0].transform.rotation.x=Q_roll.x();
        blade_state_msg.transforms[0].transform.rotation.y=Q_roll.y();
        blade_state_msg.transforms[0].transform.rotation.z=Q_roll.z();
        blade_state_msg.transforms[0].transform.rotation.w=Q_roll.w();
        blade_state_pub.publish(blade_state_msg);


    }
    inline void MotorBridge(const control_msgs::JointControllerStateConstPtr& msgs,string& motor_name){
        int j=0;
        for(int i=0; i<(joints_state_msg.name.size()); ++i){
            if(joints_state_msg.name[i]==motor_name) {
                j = i;
                break;
            }
        }
        //关节实际角度
        joints_state_msg.position[j] = msgs->process_value;
        //关节期望角度
        // joints_state_msg.position[j] = msgs->set_point;
    }
};

#endif
