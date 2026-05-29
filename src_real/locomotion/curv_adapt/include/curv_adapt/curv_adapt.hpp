#ifndef TRAJ_SEG_CAL_HPP
#define TRAJ_SEG_CAL_HPP
#include<ros/ros.h>
#include<chrono>
#include<vector>
#include<fstream>
#include<iostream>
#include <iomanip>
// #include "control/color_cout.h"
#include"utils/ColorCout.h"
#include"utils/quickhull.hpp"
#include"utils/visualizer.hpp"
#include"interface/joy_command.h"
#include"utils/kinematic.hpp"
//for hexapod_model 921 and 121 
#include<std_msgs/Float64MultiArray.h>
#include<std_msgs/Int16MultiArray.h>
#include<std_msgs/Int32MultiArray.h>
#include<gazebo_msgs/ContactsState.h>
#include<gazebo_msgs/ModelStates.h>
#include<geometry_msgs/PoseStamped.h>

//for hexapod_model703
#include"utils/robot_polyhedra.hpp"

using namespace std;
#define RF "RF"
#define RM "RM"
#define RB "RB"
#define LF "LF"
#define LM "LM"
#define LB "LB"
#define STANCE "STANCE"
#define SWING "SWING"
#define ATTACH "ATTACH"
#define ADJUST "ADJUST"
#define RETRACE "RETRACE"
#define STANCE_DONE "STANCE_DONE"
#define SWING_DONE "SWING_DONE"
#define ATTACH_DONE "ATTACH_DONE"
#define ADJUST_DONE "ADJUST_DONE"
#define RETRACE_DONE "RETRACE_DONE"
#define WAIT_ADSORB "WAIT_ADSORB"
#define WAIT_RELEASE "WAIT_RELEASE"
#define FAILED "FAILED"
#define WAITING "WAITING"
//motor mode
#define TRAJ_FELLOW 2
#define TRAPEZOID   1
#define DISABLE    -1
//机器人运行模式
#define SUPPORT_MODE "SUPPORT_MODE"
#define CLAMP_MODE "CLAMP_MODE"
//腿部极坐标在机器人中的位置，由安装位置决定
double R_B_X,R_B_Y;
double RETRACE_H, RETRACE_LEN, SUPPORT_H, SUPPORT_LEN;
double CLAMP_H,CLAMP_RETRACE_LEN,CLAMP_SUPPORT_LEN;
//机器人回收、接触、调整时的速度
double VEC_VERTICAL,VEC_ADJ=0.08;
//机器人调整时的角速度
double OMEGA_ADJ=0.15,OMEGA_MOVE=0.15;
bool STOP_FLAG=false;
double KP=50;
double ros_time_start;
string RUNNING_MODE=SUPPORT_MODE;
// string RUNNING_MODE=CLAMP_MODE;
string ROBOT_NAME;
string MODE;


bool ConfigParam(ros::NodeHandle& nh){
    nh.getParam("OFFSET_X",R_B_X);
    nh.getParam("OFFSET_Y",R_B_Y);
    if(R_B_X==0){
        RED("Load Param Failed, Please Load Robot Param First(e.g. osparam load src/hexapod_model121/config/robot_config.yaml)");
        return false;
    }
    nh.getParam("SUPPORT_H", SUPPORT_H);
    nh.getParam("SUPPORT_LEN", SUPPORT_LEN);
    nh.getParam("RETRACE_H",RETRACE_H);
    nh.getParam("RETRACE_LEN", RETRACE_LEN);
    nh.getParam("KP",KP);
    nh.getParam("CLAMP_H",CLAMP_H);
    nh.getParam("CLAMP_RETRACE_LEN",CLAMP_RETRACE_LEN);
    nh.getParam("CLAMP_SUPPORT_LEN",CLAMP_SUPPORT_LEN);
    nh.getParam("VEC_VERTICAL",VEC_VERTICAL);
    nh.getParam("VEC_ADJ",VEC_ADJ);
    nh.getParam("OMEGA_ADJ",OMEGA_ADJ);
    nh.getParam("OMEGA_MOVE",OMEGA_MOVE);
    nh.getParam("robot_name",ROBOT_NAME);
    return true;
}



class POLYHEDRA3{
//构建机器人运动活动范围
private:
    vector<double> unfold_vertexes;
    vector<Eigen::Vector3d> vertexes;
    //each col represents coordinate of vertex point
    vector<Eigen::Matrix3Xd> vPolys;
    //each row represent a b c d sign of sign=-1 a*x+b*y+c*z+d<0 and sign=1 a*x+b*y+c*z+d>0
    vector<Eigen::Matrix<double,-1,5>> hPolys;
    double polyhedron_h1, polyhedron_h2, polyhedron_h3;
    ros::NodeHandle nh;
public:
    POLYHEDRA3(){}
    POLYHEDRA3(const ros::NodeHandle& n):nh(n){
        n.getParam("vertexes",unfold_vertexes);
        n.getParam("polyhedron_h1",polyhedron_h1);
        n.getParam("polyhedron_h2",polyhedron_h2);
        n.getParam("polyhedron_h3",polyhedron_h3);
        if(unfold_vertexes.empty()) ROS_ERROR("Failed to get vertexes param!");
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

        //Construct v-polyhedron from vertexes
        vPolys.resize(4);
        vPolys[0].resize(3,8); vPolys[1].resize(3,10); vPolys[2].resize(3,6); vPolys[3].resize(3,6);
        for(int i=0; i<vertexes.size(); ++i){
            if(i<8){
                vPolys[0].col(i) = vertexes[i];
            }
            if(i>3 && i<14){
                vPolys[1].col(i-4) = vertexes[i];
            }
            if(i>9 && i< 16){
                vPolys[2].col(i-10) = vertexes[i];
            }
            if(i>13){
                vPolys[3].col(i-14) = vertexes[i];
            }
        }   

        //Claculate inner points for each polyhedron
        vector<Eigen::Vector3d> polyhedra_inner_points(4);
        for(int i=0;i<4;++i)
        {
            for(int j=0;j<3;++j)
            {
                polyhedra_inner_points[i](j)=vPolys[i].row(j).mean();
            }
        }

        //Convert v-representation polyhedra to h-representation polyhedra
        hPolys.resize(4);
        //get triangle mesh surface representation
        quickhull::QuickHull<double> tinyQH;
        Eigen::Matrix3Xd tris;
        for(int i=0;i<4;++i)
        {   
            //calculate triangle vertexes
            const auto polyhull = tinyQH.getConvexHull(vPolys[i].data(),vPolys[i].cols(),false,true);
            const auto &idexBuffer = polyhull.getIndexBuffer();
            int hNum = idexBuffer.size()/3;
            tris.resize(3,hNum*3);
            for(int j=0;j<hNum*3;++j)
            {
                tris.col(j)=vPolys[i].col(idexBuffer[j]);
            }
            //every row represent a b c d sign of sign=-1 a*x+b*y+c*z<d and sign=1 a*x+b*y+c*z>d 
            Eigen::Matrix<double,-1,5> cur_coeffs; 
            cur_coeffs.resize(hNum,5);
            int cof_idx=0;
            //calculate plane for every triangle
            for(int k=0;k<hNum*3;k+=3)
            {
                Eigen::Vector3d p1,p2;  //vector from three points on the plane
                Eigen::Vector3d norm;   //norm of this plane
                double sign;
                double d;
                p1 = tris.col(k+1)-tris.col(k);
                p2 = tris.col(k+2)-tris.col(k);
                norm = p1.cross(p2).normalized();
                d = -norm.dot(tris.col(k));
                sign = norm.dot(polyhedra_inner_points[i])+d >=0? 1:-1;
                cur_coeffs.row(cof_idx)<<norm.x(),norm.y(),norm.z(),d,sign;
                cof_idx++;
            }
            //find out the same plane and delete them
            vector<Eigen::Matrix<double,1,5>> unique_row_array;
            unique_row_array.push_back(cur_coeffs.row(0));
            for(int i=0;i<cur_coeffs.rows();++i){
                bool find = false;
                for(auto &unique_row:unique_row_array){
                    if((unique_row-cur_coeffs.row(i)).norm()<=0.001){
                        find = true;
                        break;
                    }
                }
                if(!find){
                    unique_row_array.push_back(cur_coeffs.row(i));
                }
            }
            Eigen::Matrix<double,-1,5> unique_coeffs;
            unique_coeffs.resize(unique_row_array.size(),5);
            for(int i=0;i<unique_row_array.size();++i){
                unique_coeffs.row(i)=unique_row_array[i];
            }
            // cout<<"coeffs["<<i<<"]:\n"<<cur_coeffs<<endl;
            // cout<<"unique_coffe["<<i<<"]:\n"<<unique_coeffs<<endl;
            
            // hPolys[i].resize(cur_coeffs.rows(),5);
            hPolys[i] = unique_coeffs;     
        }            
    }

    bool InPolyHedra(const Eigen::Vector3d& p){
        bool in_poly=true;
        for(auto hpoly:hPolys){
            bool break_flag=false;
            for(int i=0; i<hpoly.rows(); ++i){
                int dot_flag; 
                double dot_res = hpoly.row(i).block<1,3>(0,0).dot(p)+hpoly.row(i)(3);
                if(dot_res == 0) continue;
                dot_flag = dot_res>0? 1:-1;
                if(dot_flag != hpoly.row(i)(4)){
                    in_poly = false;
                    break_flag = true;
                    break;
                }
            }
            if(!break_flag) return true;
        }
        return in_poly;
    }

};

shared_ptr<POLYNOIMAL> TRAJ_PTR;
shared_ptr<KINEMATIC> KIN_PTR;
shared_ptr<POLYHEDRA3> HEDRA_PTR;
shared_ptr<VISUALIZER> VIS_PTR;
shared_ptr<ROBOT_POLYHEDRA2> HEDRA2_PTR;


class LEG{
public:
    string name;
    string lower_name;
    string state;               //腿部当前状态
    string des_state;           //腿部目标状态，达到该状态后，便等待用户进一步指令
    double e_error;
    double fai_0, fai_t;
    double step_len;
    double rou;
    double negative_pressure;
    vector<double> fai_traj;
    vector<double> fai_t_traj;
    int state_dir;
    int traj_pub_index;             //带时间戳需要执行的轨迹索引
    int traj_cal_index;             //可行性检查时轨迹索引
    int LR_dir;                     //左右腿坐标系方向，1为右腿，-1为左腿
    int contact_number;
    int p_state, v_state;
    int count;
    bool traj_execute_done;         //轨迹当前位置与目标位置误差小于阈值，则认为轨迹执行完毕
    bool traj_pub_done;
    Eigen::Vector3d B_e_cur,sita_cur;               //当前的轨迹末端与当前角度值
    Eigen::Vector3d B_e_0, B_e_t,B_e_last;                   //一段轨迹起点终点腿部末端的位置
    Eigen::Vector3d R_b;
    Eigen::Vector3d torque_cur;
    Eigen::Vector3d contact_force;
    Eigen::Vector3d acc,vcc,vcc_last;
    Eigen::Vector3d B_e_contact;                   //用于记录stace_done时足端的位置，当adjust时，使用adjust和该点拟合平面
    // Eigen::Vector3d sita_retrace;
    Eigen::Matrix4d B_T_R, R_T_B;
    vector<Eigen::Vector3d> B_e_t_traj;               //当前轨迹,带入时间戳后的轨迹
    vector<Eigen::Vector3d> B_e_traj;
    vector<Eigen::Vector3d> dot_e_t_traj;             //当前轨迹，带入时间戳的速度
    // vector<Eigen::Vector3d> sita_retrace_t_traj;
    ros::NodeHandle n;
    ros::Publisher sita_des_pub;
    ros::Publisher debug_msgs_pub;
    ros::Timer joint_ctrl_timer;                    //发布关节控制指令
    ros::Subscriber joint_state_sub, contact_state_sub;  
    std_msgs::Float64MultiArray sita_des_msg;
    geometry_msgs::PoseStamped debug_msgs;
    fstream debug_file;
public:
    LEG(const ros::NodeHandle& nh, Eigen::Vector3d R_b_, string name_):
    n(nh),name(name_),R_b(R_b_)
    {
        for(char c:name) lower_name.push_back(tolower(c));   
        R_T_B=Eigen::Matrix4d::Identity();
        R_T_B.block<3,1>(0,3)=R_b;
        //提取name的第一个字符，如果为R，给LR_dir赋值1，否则赋值-1
        if(name[0]=='R') {
            LR_dir=1;
        }
        else if(name[0]=='L') {
            LR_dir=-1;
            R_T_B(0,0)=-1;
            R_T_B(1,1)=-1;            
        }
        B_T_R=R_T_B.inverse();
        traj_pub_index=0;
        step_len = 0.09;
        negative_pressure = 0.0;
        contact_number=0;
        traj_execute_done=false;
        traj_pub_done=false;
        acc<<0,0,0;
        vcc<<0,0,0;
        vcc_last<<0,0,0;
        B_e_last<<0,0,0;
        count=0;
        joint_state_sub=n.subscribe<std_msgs::Float64MultiArray>("/"+name+"/sita_cur",1,&LEG::JointUpdate,this);
        contact_state_sub=n.subscribe<gazebo_msgs::ContactsState>("/"+ROBOT_NAME+"/"+lower_name+"_contacts",1,&LEG::ContactState,this);
        sita_des_pub=n.advertise<std_msgs::Float64MultiArray>("/"+name+"/sita_des",1);
        debug_msgs_pub=n.advertise<geometry_msgs::PoseStamped>("/"+name+"/debug_msgs",1);
        // joint_ctrl_timer=n.createTimer(ros::Duration(0.06),&LEG::Sita_TrajCal,this);
        joint_ctrl_timer=n.createTimer(ros::Duration(0.03),&LEG::Sita_TrajCal,this);
        //记录数据用
        if(this->name==RF){
            // debug_file.open("/home/val/BIH_ws/hexapod_sim/bag/"+this->name+".txt",ios::out);
            // debug_file.close();
            // debug_file.open("/home/val/BIH_ws/hexapod_sim/bag/"+this->name+".txt",ios::app);
        }

    }
    void Leg_DesPosCal(Eigen::Vector3d& vec_des, double& omega_des){
        Eigen::Vector3d M_e;
        double step_len_cur=0;
        // double fai_step=0.0174533;
        // double vec_step=0.001;
        traj_cal_index=0;
        B_e_traj.clear();
        B_e_traj.push_back(B_e_0);
        if(state == STANCE || state==SWING){
            B_e_0(0)=B_e_0(0)*LR_dir;
            B_e_0(1)=B_e_0(1)*LR_dir;            
            if(state==STANCE) {
                state_dir=-1;
            }
            else {
                // vec_des=vec_des*0.2;
                // omega_des=omega_des*0.2;
                state_dir = 1;
            }
            //计算足端E点在旋转中心的位置，转化为极坐标并得到fai和rou
            if(omega_des!=0){
                M_e<<B_e_0(0)+R_b(0)+vec_des(1)/omega_des,
                     B_e_0(1)+R_b(1)-vec_des(0)/omega_des,
                     B_e_0(2);
                // cout<<this->name<<" "<<this->state<<":M_e="<<M_e.transpose()<<" B_e_0="<<B_e_0.transpose()<<endl;
                rou = sqrt(pow(M_e(0),2)+pow(M_e(1),2));
                fai_0 = atan(M_e(1)/M_e(0));
                if(M_e(0)<0) fai_0 += M_PI;
                fai_traj.clear();
                fai_traj.push_back(fai_0);
                //计算足端期望位置，超出可行范围或者走过距离大于设定补偿时，停止计算，将最长可行步长放入fai_index,为了避免奇异点出现，只允许计算1000次以内
                while(traj_cal_index<1000){
                    fai_t=fai_0+state_dir*omega_des*0.01*(traj_cal_index+1);
                    step_len_cur=abs(fai_t-fai_0)*rou;
                    //由于swing状态腿部是从回收到中点后开始摆动，因此swing状态下，步长限制为step_len/2.0
                    if(this->state==SWING) if(step_len_cur>=step_len/2.0) break;
                    else if(step_len_cur>=step_len) break;
                    fai_traj.push_back(fai_t);
                    M_e<<cos(fai_t)*rou,sin(fai_t)*rou,B_e_0(2)+0.01*(traj_cal_index+1)*vec_des(2);
                    B_e_t<< LR_dir*(M_e(0) - R_b(0) - vec_des(1)/omega_des),
                            LR_dir*(M_e(1) - R_b(1) + vec_des(0)/omega_des),
                            M_e(2);
                    if(!FeasiCheck(B_e_t)) break;
                    B_e_traj.push_back(B_e_t);             
                    traj_cal_index+=1;
                }
            }
            else{
                double vec_norm=vec_des.norm();
                while(traj_cal_index<1000){
                    step_len_cur=vec_norm*0.01*(traj_cal_index+1);
                    // cout<<"step_len_cur="<<step_len_cur<<" vec_norm="<<vec_norm<<" traj_cal_index="<<traj_cal_index<<endl;
                    if(this->state==SWING) if(step_len_cur>=step_len/2.0) break;
                    else if(step_len_cur>=step_len) break;
                    B_e_t<< LR_dir*(B_e_0(0)+state_dir*vec_des(0)*(traj_cal_index+1)*0.01),
                            LR_dir*(B_e_0(1)+state_dir*vec_des(1)*(traj_cal_index+1)*0.01),
                            B_e_0(2)+vec_des(2)*(traj_cal_index+1)*0.01;
                    if(!FeasiCheck(B_e_t))break;
                    B_e_traj.push_back(B_e_t);
                    traj_cal_index+=1;
                    // cout<<"B_e_t="<<B_e_t.transpose()<<endl;
                    // cout<<"step_len_cur="<<step_len_cur<<endl;
                } 
            }
            B_e_0(0)=B_e_0(0)*LR_dir;
            B_e_0(1)=B_e_0(1)*LR_dir;
        }

        else if(state == RETRACE){
            //RETRACE 的终点是固定的，计算从当前点到终点的路经是否可行
            Eigen::Vector3d retrace_norm;
            double total_len;
            if(RUNNING_MODE==SUPPORT_MODE){
                B_e_t<<RETRACE_LEN,0,RETRACE_H;
                total_len=(B_e_t-B_e_0).norm();
                traj_cal_index=ceil(total_len/(VEC_VERTICAL*0.01));
                // total_len=(sita_retrace-sita_cur).norm();
                // traj_cal_index=ceil(total_len/(0.06*VEC_VERTICAL));
                B_e_traj.push_back(B_e_t);
                // while(true){
                //     step_len_cur=0.01*(traj_cal_index+1)*VEC;
                //     B_e_t=B_e_0+retrace_norm*step_len_cur;
                //     // if(!FeasiCheck(B_e_t)) {
                //     //     RED("Error: retrace traj error, point: "<<B_e_t.transpose()<<" will out of range");
                //     //     break;
                //     // }
                //     B_e_traj.push_back(B_e_t);
                //     traj_cal_index+=1;
                //     //距离小于等于一个步长，B_e_t赋值终点此时结束循环
                //     if(total_len-step_len_cur<=0.01*VEC){
                //         B_e_traj.push_back(Eigen::Vector3d({RETRACE_LEN,B_e_0(1),RETRACE_H}));
                //         traj_cal_index+=1;
                //         break;
                //     }
                // }                 
            }
            else if(RUNNING_MODE==CLAMP_MODE){
                B_e_t<<CLAMP_RETRACE_LEN,0,CLAMP_H;
                total_len=(B_e_t-B_e_0).norm();
                traj_cal_index=ceil(total_len/(0.01*VEC_VERTICAL));
                B_e_traj.push_back(B_e_t);
            }
        }

        else if(state == ATTACH){
            //计算到超出可行范围为止
            B_e_t=B_e_0;
            if(RUNNING_MODE==SUPPORT_MODE){
                while(true){
                    B_e_t(2)=B_e_0(2)-0.01*(traj_cal_index+1)*VEC_VERTICAL;
                    if(!FeasiCheck(B_e_t)) break;
                    B_e_traj.push_back(B_e_t);
                    traj_cal_index+=1;
                }
            }
            else if(RUNNING_MODE==CLAMP_MODE){
                while(true){
                    B_e_t(0)=B_e_0(0)-0.01*(traj_cal_index+1)*VEC_VERTICAL;
                    if(!FeasiCheck(B_e_t)) break;
                    B_e_traj.push_back(B_e_t);
                    traj_cal_index+=1;
                }
            }
        }

        B_e_t=B_e_traj.back();
        if(!fai_traj.empty()) fai_t=fai_traj.back();

        // if(this->state==SWING){
        //     cout<<this->name<<" B_e_0"<<this->B_e_0<<" B_e_t="<<this->B_e_t<<endl;
        // }
        // GREEN(this->name<<" "<<this->state<<": B_e_0="<<B_e_0.transpose()<<" B_e_t="<<B_e_t.transpose());
        // YELLOW((B_e_t-B_e_0).norm()/vec_des.norm());
    }
    void E_TrajCal(Eigen::Vector3d& vec_des, double& omega_des){
        B_e_t_traj.clear();
        dot_e_t_traj.clear();
        // sita_retrace_t_traj.clear();
        traj_execute_done=false;
        traj_pub_done=false;
        traj_pub_index=0;
        //通过计算，B_e_traj最后一位为可行，索引为e_traj_index,B_e_t_traj开始时间为0.01
        //可行时间为0,0.01,0.02,...,(traj_cal_index+1)*0.01
        vector<double> ts;
        for(int i=0;i<traj_cal_index+2;++i) ts.push_back(i*0.01);
        // RED(this->name<<":"<<this->state<<"  ts.size()="<<ts.size());
        vector<vector<double>> B_e_t_traj_axes(3);  //[x1,x2,x3,....] [y1,y2,y3,....] [z1,z2,z3,....]
        vector<vector<double>> dot_e_t_traj_axes(3);//[x1dot,x2dot,x3dot,....] [y1dot,y2dot,y3dot,....] [z1dot,z2dot,z3dot,....]
        vector<vector<double>> sita_t_traj_axes(3);//[x1dot,x2dot,x3dot,....] [y1dot,y2dot,y3dot,....] [z1dot,z2dot,z3dot,....]
        vector<double> dot_fai_t_traj;
        Eigen::Vector3d B_e, M_e, dot_e;
        if((state == STANCE || state==SWING)&&omega_des!=0){
            TRAJ_PTR->CalCoffs(fai_0,fai_t,ts.back());
            TRAJ_PTR->CalPoly(ts,fai_t_traj);
            TRAJ_PTR->CalPolyDot(ts,dot_fai_t_traj);
            for(int i=0; i<ts.size(); ++i){
                M_e<<cos(fai_t_traj[i])*rou,sin(fai_t_traj[i])*rou,0;
                B_e<< LR_dir*(M_e(0) - R_b(0) - vec_des(1)/omega_des),
                      LR_dir*(M_e(1) - R_b(1) + vec_des(0)/omega_des),
                      0;
                dot_e<<-LR_dir*sin(fai_t_traj[i])*rou*dot_fai_t_traj[i],
                       LR_dir*cos(fai_t_traj[i])*rou*dot_fai_t_traj[i],
                       0;
                B_e_t_traj.push_back(B_e);
                dot_e_t_traj.push_back(dot_e);
            }
            TRAJ_PTR->CalCoffs(B_e_0(2),B_e_t(2),ts.back());
            TRAJ_PTR->CalPoly(ts,B_e_t_traj_axes[2]);
            TRAJ_PTR->CalPolyDot(ts,dot_e_t_traj_axes[2]);
            for(int i=0;i<ts.size();i++){
                B_e_t_traj[i](2)=B_e_t_traj_axes[2][i];
                dot_e_t_traj[i](2)=dot_e_t_traj_axes[2][i];
                //omega_des!=0时，B_e 轨迹的x y位置分量和速度分量通过极坐标计算，这里为了轨迹的完整性，将通过极坐标计算得到的放入其中
                B_e_t_traj_axes[0].push_back(B_e_t_traj[i](0));
                B_e_t_traj_axes[1].push_back(B_e_t_traj[i](1));
                dot_e_t_traj_axes[0].push_back(dot_e_t_traj[i](0));
                dot_e_t_traj_axes[1].push_back(dot_e_t_traj[i](1));
            }
        }
        else{       
            for(int i=0;i<3;++i){
                TRAJ_PTR->CalCoffs(B_e_0(i),B_e_t(i),ts.back());
                // cout<<"E_traj_cal, CalCoffs ="<<TRAJ_PTR->c0<<","<<TRAJ_PTR->c3<<","<<TRAJ_PTR->c4<<","<<TRAJ_PTR->c5<<endl;
                TRAJ_PTR->CalPoly(ts,B_e_t_traj_axes[i]);
                TRAJ_PTR->CalPolyDot(ts,dot_e_t_traj_axes[i]);
            }
            for(int i=0; i<ts.size(); ++i){
                B_e_t_traj.push_back(
                    Eigen::Vector3d({B_e_t_traj_axes[0][i],B_e_t_traj_axes[1][i],B_e_t_traj_axes[2][i]})
                );
                dot_e_t_traj.push_back(
                    Eigen::Vector3d({dot_e_t_traj_axes[0][i],dot_e_t_traj_axes[1][i],dot_e_t_traj_axes[2][i]})
                );
            }
        }
        // if(this->state==RETRACE){
            // B_e_t_traj.clear();
            //使用角度计算
            // for(int i=0;i<3;++i){
            //     TRAJ_PTR->CalCoffs(sita_cur(i),sita_retrace(i),ts.back());
            //     TRAJ_PTR->CalPoly(ts,sita_t_traj_axes[i]);
            // }
            // for(int i=0; i<ts.size(); ++i){
            //     sita_retrace_t_traj.push_back(
            //         Eigen::Vector3d({sita_t_traj_axes[0][i],sita_t_traj_axes[1][i],sita_t_traj_axes[2][i]})
            //     );
            // }
            // for(auto sita:sita_retrace_t_traj){
            //     KIN_PTR->FwdKin(sita,this->B_e_t);
            //     B_e_t_traj.push_back(this->B_e_t);
            // }
        // }
        //绘制B_e_t_traj 的x y z随时间变化曲线
        //向文件中写入B_e_t_traj_axes
        // if(this->name==LM){
        //     ofstream ofs("/home/val/BIH_ws/hexapod_sim/src/control/data/B_e_t_traj_axes.txt");
        //     for(int i=0;i<ts.size();++i){
        //         ofs<<ts[i]<<" "<<B_e_t_traj_axes[0][i]<<" "<<B_e_t_traj_axes[1][i]<<" "<<B_e_t_traj_axes[2][i]<<endl;
        //     }
        //     RED("end");
        //     ofs.close();
        // }
        // YELLOW(this->name<<": B_e_t_traj.back=:"<<B_e_t_traj.back().transpose());
        // for(auto B_e_t:B_e_t_traj){
        //     YELLOW(B_e_t.transpose());
        // }
        if(name!=RM) return;
        cout<<this->name<<" "<<this->state<<endl;
        cout<<"B_e_t_traj\n";
        for(auto pos:B_e_t_traj){
            cout<<pos.transpose()<<endl;
        }
        cout<<endl;
    }
    void Sita_TrajCal(const ros::TimerEvent& e){
        // double KP=85;
        Eigen::Matrix3d damp_inv_jac;
        Eigen::Vector3d sita_dot_des, sita_des, B_e_set, dot_e_set, dot_B_e_set;
        // if(this->state==RETRACE){
        //     if(traj_pub_index<sita_retrace_t_traj.size()){
        //         sita_des=sita_retrace_t_traj[traj_pub_index];
        //         B_e_set=B_e_t_traj[traj_pub_index];
        //         sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),GetFootAngle(sita_des)});
        //         traj_pub_index+=1;
        //     }
        //     else{
        //         traj_pub_done=true;
        //         B_e_set=B_e_t;
        //     }
        //     sita_des_pub.publish(sita_des_msg);
        // }
        if(!B_e_t_traj.empty()){
            // if(this->name==RF && (this->state==ATTACH_DONE||this->state==WAIT_ADSORB)){
            //     cout<<"traj_pub_index111="<<traj_pub_index<<" B_e_t_traj.size()="<<B_e_t_traj.size()<<" <"<<(traj_pub_index<B_e_t_traj.size())<<endl;
            // }
            if(traj_pub_index<B_e_t_traj.size()){
                B_e_set=B_e_t_traj[traj_pub_index];
                dot_B_e_set=dot_e_t_traj[traj_pub_index];
                traj_pub_index+=1;
                    //用于发布可视化的topic
                    // VIS_PTR->VisPoint(B_e_set,this->R_T_B);        
                KIN_PTR->FwdKin(this->sita_cur,this->B_e_cur);
                KIN_PTR->DampInvJac(damp_inv_jac,this->sita_cur);
                sita_dot_des = damp_inv_jac*(KP*(B_e_set-B_e_cur)+dot_B_e_set);
                sita_des = sita_dot_des*0.01+sita_cur;
                // sita_dot_des=damp_inv_jac*dot_B_e_set;

                // KIN_PTR->InvKin(B_e_set,this->sita_cur,sita_des);
                // if(traj_pub_index>=10) return;
                //足部关节保持与身体平行，根据第二三关节角度计算的出
                // sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),sita_dot_des(0),sita_dot_des(1),sita_dot_des(2),GetFootAngle(sita_des)});
                if(this->state==SWING&&traj_pub_index>B_e_t_traj.size()/3){
                    sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),0,0,0,GetFootAngle(sita_des)});
                }
                else if(this->state == ATTACH || this->state == SWING_DONE ){
                    sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),0,0,0,GetFootAngle(sita_des)});
                }
                else{
                    sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),0,0,0,-999});
                }
                sita_des_pub.publish(sita_des_msg);                             
            }
            else{
                traj_pub_done=true;
                B_e_set=B_e_t;
                dot_B_e_set<<0,0,0;
                // GREEN("-------------------------->Sita Pub Done<------------------------");
            }


            debug_msgs.header.stamp=ros::Time::now();
            // debug_msgs.pose.position.x=this->B_e_cur(0);
            // debug_msgs.pose.position.y=this->B_e_cur(1);
            // debug_msgs.pose.position.z=this->B_e_cur(2);
            // debug_msgs.pose.orientation.x=dot_B_e_set(0);
            // debug_msgs.pose.orientation.y=dot_B_e_set(1);
            // debug_msgs.pose.orientation.z=dot_B_e_set(2);
            debug_msgs.pose.orientation.x=B_e_set(0);
            debug_msgs.pose.orientation.y=B_e_set(1);
            debug_msgs.pose.orientation.z=B_e_set(2); 
            debug_msgs_pub.publish(debug_msgs);

        }
        else {
            B_e_set<<0,0,0;
            dot_B_e_set<<0,0,0;
        }
   

        // cout<<this->name<<" sita_des="<<sita_des.transpose()<<" sita_cur="<<sita_cur.transpose()<<endl;        
        // if(this->name==RF){
            // debug_file.open("/home/val/BIH_ws/hexapod_sim/bag/"+this->name+".txt",ios::app);
            // debug_file<<this->state<<" "<<B_e_set.transpose()<<dot_B_e_set.transpose()<<" "<<std::fixed<<ros::Time::now().toSec()<<endl;
            // debug_file.close();
        // }
        //用于发布测试的topic
        // vcc=(B_e_set-B_e_last)/0.01;
        // acc=(vcc-vcc_last)/0.01;
        // vcc_last=vcc;
        // B_e_last=B_e_set;



        // debug_msgs.pose.orientation.x=acc(0);
        // debug_msgs.pose.orientation.y=acc(1);
        // debug_msgs.pose.orientation.z=acc(2);
        // debug_msgs.pose.position.x=180*(sita_cur(0)/M_PI);
        // debug_msgs.pose.position.y=180*(sita_cur(1)/M_PI);
        // // debug_msgs.pose.position.z=180*(sita_cur(2)/M_PI);
        // debug_msgs.pose.orientation.x=180*(sita_des(0)/M_PI);    
        // debug_msgs.pose.orientation.y=180*(sita_des(1)/M_PI);
        // debug_msgs.pose.orientation.z=180*(GetFootAngle(sita_des)/M_PI);

        // if(this->state==SWING||this->state==SWING_DONE) debug_msgs.pose.orientation.w=50;
        // else if(this->state==ATTACH) debug_msgs.pose.orientation.w=75;
        // else if(this->state==ATTACH_DONE||this->state==WAIT_ADSORB||this->state==ADJUST||
        //         this->state==ATTACH_DONE||this->state==STANCE||this->state==STANCE_DONE)
        // {
        //     debug_msgs.pose.orientation.w=100;
        // }
        // else if(this->state==WAIT_RELEASE||this->state==RETRACE||this->state==RETRACE_DONE) debug_msgs.pose.orientation.w=25;
        // debug_msgs.pose.orientation.w/=400.0;
        // debug_msgs.pose.position.x=contact_force(0);
        // debug_msgs.pose.position.y=contact_force(1);
        // debug_msgs.pose.position.z=contact_force(2);

        // debug_msgs.pose.position.x=torque_cur(0);
        // debug_msgs.pose.position.y=torque_cur(1);
        // debug_msgs.pose.position.z=torque_cur(2);
    }
    bool ContactDetection(){
        // return abs(this->B_e_cur(2)-SUPPORT_H)<=0.01;

 
        if (MODE=="real"){
            bool contact_flag=false;
            if(this->name[1]=='M'){
                contact_flag=(abs(this->torque_cur(1))+abs(this->torque_cur(2)))>=5.5;
            }
            else{
                contact_flag=(abs(this->torque_cur(1))+abs(this->torque_cur(2)))>=2.5;
            }
            if(contact_flag) count++;
            cout<<this->name<<" torque_cur="<<this->torque_cur.transpose()<<" contact_flag="<<contact_flag<<" count="<<count<<endl;
            if(count>=3){
                count=0;
                return true;
            }
        }
        else if(MODE=="sim"){
            if(contact_number>4){
                count++;
            }
            if(count>3){
                count=0;
                return true;
            }
        }
        return false;

        // if(traj_pub_done) return true;
        // return false;
            //Calculate contace force
        // Eigen::Matrix3d damp_tran_inv_jac;
        // KIN_PTR->DampTranInvJac(damp_tran_inv_jac,this->sita_cur);
        // this->contact_force = damp_tran_inv_jac*this->torque_cur;
        // cout<<this->contact_force(2)<<endl;
        // cout<<this->torque_cur.transpose()<<endl;
        // return abs(this->contact_force(2))>=30;

    }
    bool FeasiCheck(Eigen::Vector3d& B_e){
        return HEDRA_PTR->InPolyHedra(B_e);       
    }
    void JointUpdate(const std_msgs::Float64MultiArrayConstPtr& joint_msgs){
        //simulations joint state
        if(joint_msgs->data.size() == 8){
            debug_msgs.pose.position.z=180.0*(joint_msgs->data[3]/M_PI);
            sita_cur<<joint_msgs->data[0],joint_msgs->data[1],joint_msgs->data[2];
            torque_cur<<joint_msgs->data[4],joint_msgs->data[5],joint_msgs->data[6];


        }
        //real joint state
        else if(joint_msgs->data.size() == 9){
            sita_cur<<joint_msgs->data[0],joint_msgs->data[1],joint_msgs->data[2];
            torque_cur<<joint_msgs->data[3],joint_msgs->data[4],joint_msgs->data[5];
            // cout<<this->name<<" tor="<<torque_cur.transpose()<<endl;
        }
        else{
            YELLOW("joint_msgs->data.size()="<<joint_msgs->data.size());
            ROS_WARN("Received error number of joint_msgs");
        }
    }
    void ContactState(const gazebo_msgs::ContactsStateConstPtr& contact_msgs){
        if(contact_msgs->states.empty()){
            contact_number=0;
        }
        else{
            contact_number=contact_msgs->states[0].depths.size();
        }
    }
    double GetFootAngle(Eigen::Vector3d& sita_des){
        if(RUNNING_MODE==SUPPORT_MODE){
            return -(sita_des(1)+sita_des(2))-M_PI_2;
            // return -abs(sita_des(1))+abs(sita_des(2))-M_PI_2;
        }
        else if(RUNNING_MODE==CLAMP_MODE){
            return abs(sita_des(1))+abs(sita_des(2))-M_PI;
        }
        else {
            ROS_ERROR("MODE ERROR!");
            return 0;
        }
    }    
};


class HEXAPOD{
public:
    ros::NodeHandle n;
    vector<shared_ptr<LEG>>legs,A_legs,B_legs;
    vector<Eigen::Vector3d> R_COM_traj;
    vector<int> pv_command, pv_command_last;
    shared_ptr<LEG>rf,rm,rb,lf,lm,lb;
    ros::Subscriber command_sub;
    ros::Subscriber negative_pressure_sub;
    ros::Subscriber robot_state_sub;
    ros::Subscriber robot_state_imu_sub;
    ros::Publisher pv_command_pub;
    ros::Publisher robot_state_pub;
    ros::Publisher fitten_plane_pub; 
    ros::Timer gait_planning_timer;
    ros::Time tic_attach, tic_release;
    Eigen::Vector3d vec_des;
    Eigen::Vector3d euler_angle;
    // Eigen::Matrix3d W_rot_R,Rini_rot_W; //gazebo中机器人的旋转矩阵 Rini为机器人运动初始化后的旋转矩阵 R为机器人运动过程中的旋转矩阵
    Eigen::Quaterniond W_Q_R; //gazebo中机器人的四元数,用于记录机器人的位姿

    int use_neg=1;


    bool loop_done;         //表示半个步态周期循环结束，腿部轨迹执行完毕，同时腿部达到了期望轨迹
    bool set_move_init;     //表示对运动状态初始化进行了设置
    bool init_done;         //表示设置了所有初始状态，并且机器人腿部均达到了预定位置
    bool set_move;          //表示用户设置了期望运动速度
    double omega_des;
    fstream log_file;
public:
    HEXAPOD(const ros::NodeHandle& nh):n(nh){
        command_sub = n.subscribe<interface::joy_command>("/usr/command",1,&HEXAPOD::Rec_usr_command,this);
        negative_pressure_sub = n.subscribe<std_msgs::Int32MultiArray>("/negative_pressure",1,&HEXAPOD::NegPre,this);        
        gait_planning_timer = n.createTimer(ros::Duration(0.03),&HEXAPOD::GaitPlanning,this);
        pv_command_pub = n.advertise<std_msgs::Int16MultiArray>("/usr/pv_command",1);
        fitten_plane_pub = n.advertise<geometry_msgs::PoseStamped>("/fitten_plane",1);
        loop_done=false;
        set_move_init=false;
        init_done=false;
        set_move=false;
        pv_command.resize(12);
        rf = make_shared<LEG>(n,Eigen::Vector3d({R_B_X,R_B_Y,0}),"RF");
        rm = make_shared<LEG>(n,Eigen::Vector3d({R_B_X,0,0}),"RM");
        rb = make_shared<LEG>(n,Eigen::Vector3d({R_B_X,-R_B_Y,0}),"RB");
        lf = make_shared<LEG>(n,Eigen::Vector3d({-R_B_X,R_B_Y,0}),"LF");
        lm = make_shared<LEG>(n,Eigen::Vector3d({-R_B_X,0,0}),"LM");
        lb = make_shared<LEG>(n,Eigen::Vector3d({-R_B_X,-R_B_Y,0}),"LB");
        legs.assign({rf,rm,rb,lf,lm,lb});

        euler_angle<<0,0,0;
        //可视化相关初始化
        robot_state_pub = n.advertise<geometry_msgs::PoseStamped>("/robot_des_state",1);
        //为了记录期望身体位姿
        robot_state_sub = n.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states",1,&HEXAPOD::RobotState,this);
        // robot_state_imu_sub=n.subscribe<geometry_msgs::PoseStamped>("/imu/model_states",1,&HEXAPOD::ImuState,this);
        // log_file.open("/home/val/BIH_ws/hexapod_sim/bag/body_pos_des.txt",std::fstream::out);
        // log_file.open("/home/nv/BIH_ws/hexapod_sim/bag/body_pos_des.txt",std::fstream::out);
        // log_file.close();
        // ros::Duration(1.0).sleep();
        // Init();

        n.getParam("use_neg",use_neg);


    }
    void Init(){
        vec_des<<0.3,0.4,0;
        omega_des=1;
        //计时开始
        //测试程序
        //1.摆动、支撑轨迹计算正确性
        auto start_time = std::chrono::high_resolution_clock::now();
        for(auto &leg:legs){
            leg->B_e_0<<0.16,0.02,-0.07;
            if(leg->name==LM) leg->B_e_0<<0.18,0.02,-0.07;
            leg->state=SWING;
            // if(leg->name==RF||leg->name==RB||leg->name==LM){
            //     leg->state=SWING;
            // }
            cout<<leg->name<<"\n";
            leg->Leg_DesPosCal(vec_des,omega_des);
            cout<<"B_e_t_end="<<leg->B_e_t.transpose()<<endl;
            YELLOW("(B_e_t-B_e_0).norm()="<<(leg->B_e_t-leg->B_e_0).norm());
            leg->E_TrajCal(vec_des,omega_des);
            
            // cout<<"B_e_t_traj\n";
            // for(auto B_e:leg->B_e_t_traj){
            //     cout<<B_e.transpose()<<endl;
            // }
            // cout<<endl;
            // GREEN(leg->FeasiCheck(leg->B_e_0));
        }
        //2.调整身体目标姿态计算正确性
        //3.调整身体姿态轨迹计算正确性
        for(auto& leg:legs){
            if(leg->name==RM){
                leg->state=ADJUST;
                // leg->B_e_cur<<0.18,0,-0.07;
                leg->B_e_cur<<0.17,0.03,-0.1;
            }
            if(leg->name==LF){
                leg->state=ADJUST;
                // leg->B_e_cur<<0.18,0,-0.12;
                leg->B_e_cur<<0.16,0.02,-0.09;
            }
            if(leg->name==LB){
                leg->state=ADJUST;
                // leg->B_e_cur<<0.18,0,-0.12;
                leg->B_e_cur<<0.18,-0.030,-0.12;
            }                        
        }
        Adj_TrajsCal(0);

        //4.步态规划正确性与整体流程
        //完成

        //5.补充加持运动的代码

        
        //计时结束
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // std::cout << "程序运行时间为 " << duration.count()/1000.0 << " 毫秒" << std::endl;

    }
    void SetInit(bool move){
        Eigen::Vector3d sita_initial(0,1.1,-2.0),sita_des,B_e_initial_stance,B_e_initial_retrace;
        if(RUNNING_MODE==SUPPORT_MODE){
            B_e_initial_stance<<SUPPORT_LEN,0,SUPPORT_H;
            B_e_initial_retrace<<RETRACE_LEN,0,RETRACE_H;
        }
        else if(RUNNING_MODE==CLAMP_MODE){
            B_e_initial_stance<<CLAMP_SUPPORT_LEN,0,CLAMP_H;
            B_e_initial_retrace<<CLAMP_RETRACE_LEN,0,CLAMP_H;
        }
        for(auto &leg:legs){
            leg->B_e_t_traj.clear();
            if(!move){
                if(RUNNING_MODE==SUPPORT_MODE){
                    leg->state=SWING;
                }
                else if(RUNNING_MODE==CLAMP_MODE){
                    leg->state=SWING_DONE;
                    // leg->state=SWING;
                    leg->traj_pub_done=true;
                }
                leg->des_state=SWING_DONE;
                leg->B_e_0=B_e_initial_retrace;
                leg->B_e_t=B_e_initial_retrace;
                set_move_init=false;
            }
            else{
                if(leg->name==RM||leg->name==LB||leg->name==LF){
                    leg->state=STANCE;
                    leg->des_state=STANCE_DONE;
                    leg->B_e_0<<B_e_initial_stance;
                    leg->B_e_t<<B_e_initial_stance;
                }
                set_move_init=true;
            }
            if(KIN_PTR->InvKin(leg->B_e_t,sita_initial,sita_des)){
                leg->sita_des_msg.data.assign({TRAJ_FELLOW,sita_des(0),sita_des(1),sita_des(2),0.5,0.5,0.5,leg->GetFootAngle(sita_des)});
                leg->sita_des_pub.publish(leg->sita_des_msg);
                // if(leg->state==SWING||leg->state==SWING_DONE){
                //     leg->sita_retrace=sita_des;
                // }
            }
        }

        PV_Pub(); 
    }
    void Adj_TrajsCal(int failed_num){
        auto start=std::chrono::high_resolution_clock::now();        
        vector<Eigen::Vector3d> contact_points;
        vector<shared_ptr<LEG>> adj_legs,failed_legs;     
        Eigen::Matrix3d R_rot_R2;    //R系与R2系之间的旋转矩阵
        Eigen::Matrix4d R2_T_R, R_T_R2=Eigen::Matrix4d::Identity();
        bool feasi_flag=true;
        Eigen::Vector4d temp_e_normalize;
        Eigen::Vector3d temp_e;
        int feasi_interp_num=1;
        vector<double> ts;
        vector<double> pasi_t_traj,d_t_traj;
        double pasi_0=0, d_0=0;
        double pasi_t,d_t;
        Eigen::Quaterniond R_R2_Q;  //R2系在R中的姿态，四元数表示
        Eigen::Vector3d R_R2_origin;//R2原点在R系中的位置 

        Eigen::Vector3d plane_norm;
        double plane_d;
        double pasi;        //R系z轴与平面法向量的夹角
        double R_plane_len; //R系原点到平面的距离

        if(failed_num==0){
            for(auto& leg:legs){
                if(leg->state==ADJUST){
                    // contact_points.push_back( (leg->R_T_B*Eigen::Vector4d(leg->B_e_cur(0),leg->B_e_cur(1),leg->B_e_cur(2),1)).block<3,1>(0,0) );  //使用实际值计算
                    contact_points.push_back( (leg->R_T_B*Eigen::Vector4d(leg->B_e_t(0),leg->B_e_t(1),leg->B_e_t(2),1)).block<3,1>(0,0) );  //使用期望值计算
                    adj_legs.push_back(leg);
                    // cout<<(leg->R_T_B*Eigen::Vector4d(leg->B_e_cur(0),leg->B_e_cur(1),leg->B_e_cur(2),1)).block<3,1>(0,0).transpose()<<endl; 
                }
            }
            for(auto &leg:legs){
                if(leg->state!=ADJUST){
                    contact_points.push_back( (leg->R_T_B*Eigen::Vector4d(leg->B_e_contact(0),leg->B_e_contact(1),leg->B_e_contact(2),1)).block<3,1>(0,0) );  //使用期望值计算
                }
            }
            //根据y值大小，对contact_points进行排序
            // std::sort(contact_points.begin(),contact_points.end(),[](const Eigen::Vector3d& a,const Eigen::Vector3d& b){
            //     return a(1)<b(1);
            // });

            double support_h=abs(SUPPORT_H);            
            if(RUNNING_MODE==SUPPORT_MODE){
                //根据三点p0 p1 p2拟合平面, ax+by+cz+plane_d=0; [a,b,c]=plane_norm, plane_norm的方向与R系Z轴的夹角pasi<pi/2
                // plane_norm=( (contact_points[0]-contact_points[1]).cross(contact_points[0]-contact_points[2]) ).normalized();
                // plane_norm=plane_norm.dot(Eigen::Vector3d(0,0,1))>0?plane_norm:-plane_norm;
                // plane_d=-plane_norm.dot(contact_points[0]);
                // R_plane_len=abs(plane_d)/plane_norm.norm();
                // pasi=acos(plane_norm.dot(Eigen::Vector3d(0,0,1)));\

                //使用contact_points6点计算伪逆矩阵，拟合平面 ax+by+cz=1
                Eigen::MatrixXd A(6,3);
                Eigen::VectorXd b=Eigen::VectorXd::Ones(6);
                for(int i=0;i<6;++i){
                    A.row(i)=contact_points[i].transpose();
                }
                plane_norm=( ( ( (A.transpose()*A).inverse() )*A.transpose() )*b ).normalized();
                plane_norm=plane_norm.dot(Eigen::Vector3d(0,0,1))>0?plane_norm:-plane_norm;
                plane_d=-plane_norm.dot(contact_points[0]);
                R_plane_len=abs(plane_d)/plane_norm.norm();
                pasi=acos(plane_norm.dot(Eigen::Vector3d(0,0,1)));
                // YELLOW("plane norm="<<plane_norm.transpose());

                // GREEN("suport_h="<<support_h<<" plane="<<plane_norm.transpose()<<" "<<plane_d<<" R_plane_len="<<R_plane_len<<" pasi="<<pasi);
                if(plane_norm[2]==0){
                    ROS_ERROR("Failed to adjust posture because of singularity");
                    return;
                }
                pasi_t=pasi;
                d_t=support_h-R_plane_len;
                // if(abs(support_h-R_plane_len)<=0.001&&pasi<0.001){
                //     cout<<"don not need to adjust";
                //     for(auto& leg:adj_legs){
                //         leg->B_e_t_traj.clear();
                //         leg->dot_e_t_traj.clear();
                //         leg->traj_pub_index=0;
                //         leg->traj_execute_done=false;
                //         leg->traj_pub_done=false;
                //         leg->B_e_t_traj.push_back(leg->B_e_t);
                //         leg->dot_e_t_traj.push_back(Eigen::Vector3d::Zero());
                //     }
                //     return;
                // }
                //Y2轴与R系ZY轴平行计算旋转矩阵,这样不会改变yaw角，该变量始终由用户控制
                Eigen::Vector3d R_y2(0,1,-plane_norm[1]/plane_norm[2]);
                Eigen::Vector3d R_x2 = R_y2.cross(plane_norm);
                R_y2.normalize(); 
                R_x2.normalize();
                R_rot_R2<<R_x2,R_y2,plane_norm;
                //计算距离和姿态的插值，按照平均速度和角速度计算，并将时间较长的作为插值个数依据
                vector<double> d_interp({0});
                vector<double> pasi_interp({0});
                int pasi_interp_num=ceil(pasi/(OMEGA_ADJ*0.01));
                int d_interp_num=ceil(abs(support_h-R_plane_len)/(VEC_ADJ*0.01));
                int interp_num=pasi_interp_num>d_interp_num? pasi_interp_num:d_interp_num;
                for(int i=0;i<interp_num;++i){
                    pasi_interp.push_back((i+1)*pasi/interp_num);
                    d_interp.push_back((i+1)*(support_h-R_plane_len)/interp_num);
                }
                // YELLOW("interp_num="<<interp_num);
                //根据插值计算腿部坐标系的位置，是否可行，将最可行的索引长度放入feasi_interp_num中
                Eigen::Quaterniond Q(Eigen::Matrix3d::Identity()), Q2(R_rot_R2), Q_interp;
                for(int i=1; i<interp_num+1; ++i){
                    Q_interp=Q.slerp(pasi_interp[i]/pasi,Q2);
                    R_T_R2.block<3,3>(0,0)=Q_interp.toRotationMatrix();
                    R_T_R2.block<3,1>(0,3)=plane_norm*d_interp[i];
                    // R_T_R2(2,3)=d_interp[i];
                    R2_T_R=R_T_R2.inverse();
                    int j=0;
                    for(auto&leg:adj_legs){
                        temp_e_normalize<<contact_points[j].x(),contact_points[j].y(),contact_points[j].z(),1;
                        temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                        if(!leg->FeasiCheck(temp_e)){
                            feasi_flag=false;
                            break;
                        }
                        j++;
                    }
                    if(!feasi_flag) break;
                    feasi_interp_num=i+1;
                }
                //将可行的结果记录下来，用于向其他节点发布
                R_R2_Q = pasi_t==0? Q:Q.slerp(pasi_interp[feasi_interp_num-1]/pasi,Q2);
                R_R2_origin = plane_norm*d_interp[feasi_interp_num-1];

                if(feasi_interp_num==1) return;
                if(feasi_interp_num==interp_num+1) GREEN("--------------->Adjust Within Feasible Reigon<--------------");
                else YELLOW("--------------->Adjust Outside Feasible Reigon by "<<(interp_num+1-feasi_interp_num)<<" steps<--------------");
                
                for(int i=0; i<feasi_interp_num; ++i) ts.push_back(i*0.01);
                //计算pasi_t_traj和d_t_traj
                TRAJ_PTR->CalCoffs(pasi_0,pasi_interp[feasi_interp_num-1],ts.back());
                TRAJ_PTR->CalPoly(ts,pasi_t_traj);
                TRAJ_PTR->CalCoffs(d_0,d_interp[feasi_interp_num-1],ts.back());
                TRAJ_PTR->CalPoly(ts,d_t_traj);
                //根据pasi_t_traj和d_t_traj计算R_T_R2，并计算腿部在B中的位置

                // log_file.open("/home/val/BIH_ws/hexapod_sim/bag/body_pos_des.txt",std::fstream::app);
                // log_file.open("/home/nv/BIH_ws/hexapod_sim/bag/body_pos_des.txt",std::fstream::app);
                double time_start = ros::Time::now().toSec();
                int j=0;
                GREEN_WHITE("write body_pos_des.txt");
                for(auto&leg:adj_legs){
                    leg->B_e_t_traj.clear();
                    leg->dot_e_t_traj.clear();
                    leg->traj_pub_index=0;
                    leg->traj_execute_done=false;
                    leg->traj_pub_done=false;
                    for(int i=0;i<ts.size();++i){
                        //pasi=0会导致奇异
                        Q_interp= pasi_t ==0? Q:Q.slerp(pasi_t_traj[i]/pasi_interp[feasi_interp_num-1],Q2);
                        R_T_R2.block<3,3>(0,0)=Q_interp.toRotationMatrix();
                        R_T_R2.block<3,1>(0,3)=plane_norm*d_t_traj[i];
                        // R_T_R2(2,3)=d_t_traj[i];
                        R2_T_R=R_T_R2.inverse();
                        temp_e_normalize<<contact_points[j].x(),contact_points[j].y(),contact_points[j].z(),1;
                        temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                        leg->B_e_t_traj.push_back(temp_e);
                        //向文件中写入旋转矩阵的变化，使用欧拉角
                        if (leg==adj_legs[0]){
                            Eigen::Quaterniond q_temp = W_Q_R*Q_interp;
                            // Eigen::Matrix3d rot=W_rot_R*(R_T_R2.block<3,3>(0,0));
                            // Eigen::Matrix3d rot=R_T_R2.block<3,3>(0,0);
                            // Eigen::Vector3d eulerAngle_mine=rot.eulerAngles(2,1,0).array()*180.0/M_PI;
                            // eulerAngle_mine+=euler_angle;
                            // cout<<std::fixed<<time_start<<":euler_angle="<<euler_angle.transpose()<<endl;
                            //roll
                            // eulerAngle_mine(2) = 180.0*(std::atan2(rot(2, 1), rot(2, 2)))/M_PI+euler_angle(0);
                            // //pitch
                            // eulerAngle_mine(1) = 180.0*(std::atan2(-rot(2, 0), std::sqrt(rot(2, 1) * rot(2, 1) + rot(2, 2) * rot(2, 2))))/M_PI+euler_angle(1);
                            // //yaw
                            // eulerAngle_mine(0) = 180.0*(std::atan2(rot(1, 0), rot(0, 0)))/M_PI+euler_angle(2);
                            // log_file<<std::fixed<<time_start+ts[i]*0.8<<" "<<eulerAngle_mine.transpose()<<endl; 
                            // log_file<<std::fixed<<time_start+ts[i]*0.8<<" "<<q_temp.coeffs().transpose()<<endl; 
                            // log_file<<std::fixed<<time_start+ts[i]*0.8<<" "<<(q_temp.toRotationMatrix().eulerAngles(2,1,0).array()*180.0/M_PI).transpose()<<endl; 
                        }
                    }
                    j++;
                }
                // log_file.close();
            }
            else if(RUNNING_MODE==CLAMP_MODE){
                std::sort(contact_points.begin(),contact_points.end(),[](const Eigen::Vector3d& a,const Eigen::Vector3d& b){
                    return a(1)<b(1);
                });
                //将接触点投影至R系的XY平面，通过直线插值连接三点
                //x=k1*y+b1 x=k2*y+b2
                double k1,k2,k;
                double pasi;
                k1=(contact_points[1].x()-contact_points[0].x())/(contact_points[1].y()-contact_points[0].y());
                k2=(contact_points[2].x()-contact_points[1].x())/(contact_points[2].y()-contact_points[1].y());
                k=(k1+k2)/2.0;
                if(!isnan(k)){
                    pasi=-atan(k);
                }
                else{
                    pasi=M_PI_2;
                }
                R_rot_R2<<  cos(pasi),-sin(pasi),0,
                            sin(pasi),cos(pasi),0,
                            0,0,1;
                int interp_num=ceil(abs(pasi)/(OMEGA_ADJ*0.01));
                vector<double> pasi_interp({0});
                for(int i=0;i<interp_num;++i){
                    pasi_interp.push_back((i+1)*pasi/interp_num);
                }
                Eigen::Quaterniond Q(Eigen::Matrix3d::Identity()), Q2(R_rot_R2), Q_interp;
                for(int i=1; i<interp_num+1;++i){
                    Q_interp=Q.slerp(pasi_interp[i]/pasi,Q2);
                    R_T_R2.block<3,3>(0,0)=Q_interp.toRotationMatrix();
                    R2_T_R=R_T_R2.inverse();
                    for(auto&leg:adj_legs){
                        temp_e_normalize=leg->R_T_B*Eigen::Vector4d(leg->B_e_cur(0),leg->B_e_cur(1),leg->B_e_cur(2),1);
                        temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                        if(!leg->FeasiCheck(temp_e)){
                            feasi_flag=false;
                            break;
                        }
                    }
                    if(!feasi_flag) break;
                    feasi_interp_num=i+1;
                }
                cout<<"feasi_interp_num="<<feasi_interp_num<<" interp_num="<<interp_num<<endl;
                cout<<"R_rot_R2\n"<<R_rot_R2<<endl;
                if(feasi_interp_num==1) return;
                if(feasi_interp_num==interp_num+1) GREEN("--------------->Adjust Within Feasible Reigon<--------------");
                else YELLOW("--------------->Adjust Outside Feasible Reigon by "<<(interp_num+1-feasi_interp_num)<<" steps<--------------");
                //将可行的结果记录下来，用于向其他节点发布
                R_R2_Q = Q.slerp(pasi_interp[feasi_interp_num-1]/pasi,Q2);
                R_R2_origin = Eigen::Vector3d::Zero();
                for(int i=0; i<feasi_interp_num; ++i) ts.push_back(i*0.01);
                //计算pasi_t_traj和d_t_traj
                TRAJ_PTR->CalCoffs(pasi_0,pasi_interp[feasi_interp_num-1],ts.back());
                TRAJ_PTR->CalPoly(ts,pasi_t_traj);
                //根据pasi_t_traj计算R_T_R2，并计算腿部在B中的位置
                for(auto&leg:adj_legs){
                    leg->B_e_t_traj.clear();
                    leg->dot_e_t_traj.clear();
                    leg->traj_pub_index=0;
                    leg->traj_execute_done=false;
                    leg->traj_pub_done=false;
                    for(int i=0;i<ts.size();++i){
                        Q_interp=Q.slerp(pasi_t_traj[i]/pasi_interp[feasi_interp_num-1],Q2);
                        R_T_R2.block<3,3>(0,0)=Q_interp.toRotationMatrix();
                        R2_T_R=R_T_R2.inverse();
                        temp_e_normalize=leg->R_T_B*Eigen::Vector4d(leg->B_e_cur(0),leg->B_e_cur(1),leg->B_e_cur(2),1);
                        temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                        leg->B_e_t_traj.push_back(temp_e);
                    }
                }
            }
            //根据B_e_t_traj计算足端速度dot_e_t_traj
            for(auto&leg:adj_legs){
                leg->dot_e_t_traj.push_back(Eigen::Vector3d::Zero());
                for(int i=1; i<ts.size();++i){
                    leg->dot_e_t_traj.push_back( (leg->B_e_t_traj[i]-leg->B_e_t_traj[i-1])/0.01 );
                }

                // if(leg->name==LB){
                //     ofstream ofs("/home/val/BIH_ws/hexapod_sim/src/control/data/B_e_t_traj_axes.txt");
                //     for(int i=0;i<ts.size();++i){
                //         ofs<<ts[i]<<" "<<leg->B_e_t_traj[i][0]<<" "<<leg->B_e_t_traj[i][1]<<" "<<leg->B_e_t_traj[i][2]<<endl;
                //     }
                //     // RED("end");
                //     ofs.close();
                // }  
            }         
        }
        else if(failed_num==1){
            int rotate_dir;//身体姿态调整的旋转方向，右侧角failed为正，左侧角failed为负
            //此时腿部接触失败，另外三条腿的状态为STANCE_DONE，另外的腿为ATTACH_DONE或者FAILED，需要STANCE_DONE和ATTACH_DONE的腿计算并调整位置
            for(auto& leg:legs){
                if(leg->state==STANCE_DONE||leg->state==ATTACH_DONE){
                    adj_legs.push_back(leg);
                    contact_points.push_back( 
                        (leg->R_T_B*Eigen::Vector4d(leg->B_e_cur(0),leg->B_e_cur(1),leg->B_e_cur(2),1)).block<3,1>(0,0) 
                    );  //使用实际值计算
                }
                if(leg->state==FAILED){
                    rotate_dir=leg->name[0]=='R'?1:-1;
                }
            }
            double pasi=0,pasi_feasi=0;
            double d=0, d_feasi=0;
            int j=0;
            ts.push_back(0);
            while(true){
                pasi+=rotate_dir*OMEGA_ADJ*0.01;
                d-=VEC_ADJ*0.01;
                R_rot_R2<<cos(pasi),0,-sin(pasi),
                        0,1,0,
                        sin(pasi),0,cos(pasi);
                R_T_R2.block<3,3>(0,0)=R_rot_R2;
                R_T_R2(2,3)=d;
                R2_T_R=R_T_R2.inverse();
                int i=0;
                for(auto&leg:adj_legs){
                    temp_e_normalize<<contact_points[i].x(),contact_points[i].y(),contact_points[i].z(),1;
                    temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                    if(!leg->FeasiCheck(temp_e)){
                        feasi_flag=false;
                        break;
                    }
                    i++;
                }
                if(!feasi_flag) break;
                pasi_feasi=pasi;
                d_feasi=d;
                ts.push_back(j*0.01);
                j++;
            }
            TRAJ_PTR->CalCoffs(pasi_0,pasi_feasi,ts.back());
            TRAJ_PTR->CalPoly(ts,pasi_t_traj);
            TRAJ_PTR->CalCoffs(d_0,d_feasi,ts.back());
            TRAJ_PTR->CalPoly(ts,d_t_traj);
            int i=0;
            for(auto&leg:adj_legs){
                leg->B_e_t_traj.clear();
                leg->dot_e_t_traj.clear();
                leg->traj_pub_index=0;
                leg->traj_execute_done=false;
                for(int j=0;j<ts.size();++j){
                    R_T_R2.block<3,3>(0,0)=R_rot_R2;
                    R_T_R2(2,3)=d_t_traj[j];
                    R2_T_R=R_T_R2.inverse();
                    temp_e_normalize<<contact_points[i].x(),contact_points[i].y(),contact_points[i].z(),1;
                    temp_e=(leg->B_T_R*R2_T_R*temp_e_normalize).block<3,1>(0,0);
                    leg->B_e_t_traj.push_back(temp_e);
                }
                i++;
            }
        }
        else if(failed_num==2){
            //处于身体同一侧与failed_num==1处理方法相同
        }
        else if(failed_num==3){
            //降低身体的高度
        }
        // cout<<"pasi="<<pasi_t<<" d_t="<<d_t<<" Q_interp="<<R_R2_Q.x()<<" "<<R_R2_Q.y()<<" "<<R_R2_Q.z()<<" "<<R_R2_Q.w()<<endl;
        for(auto&leg:adj_legs){
            if(!leg->B_e_t_traj.empty()){
                if(leg->B_e_t_traj.back().hasNaN()){
                    // cout<<"!!!!!!!!\n";
                    leg->B_e_t_traj.clear();
                    leg->B_e_t_traj.push_back(leg->B_e_t);
                }
                else{
                    leg->B_e_t=leg->B_e_t_traj.back();
                }

                //可视化腿部轨迹
                // VIS_PTR->VisTraj(leg->B_e_t_traj,leg->R_T_B,leg->name);
            }
        }

        //向/robot_des_state发布机器人期望姿态
        geometry_msgs::PoseStamped des_state_msgs,fitten_plane_msg;
        des_state_msgs.header.frame_id="body";
        des_state_msgs.header.stamp=ros::Time::now();
        des_state_msgs.pose.position.x=R_R2_origin[0];
        des_state_msgs.pose.position.y=R_R2_origin[1];
        des_state_msgs.pose.position.z=R_R2_origin[2];
        des_state_msgs.pose.orientation.x=R_R2_Q.x();
        des_state_msgs.pose.orientation.y=R_R2_Q.y();
        des_state_msgs.pose.orientation.z=R_R2_Q.z();
        des_state_msgs.pose.orientation.w=R_R2_Q.w();
        robot_state_pub.publish(des_state_msgs);

        fitten_plane_msg.header.stamp=ros::Time::now();
        fitten_plane_msg.pose.orientation.x=plane_norm[0];
        fitten_plane_msg.pose.orientation.y=plane_norm[1];
        fitten_plane_msg.pose.orientation.z=plane_norm[2];
        fitten_plane_msg.pose.orientation.w=plane_d;
        fitten_plane_pub.publish(fitten_plane_msg);
        
        auto end=std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        // cout<<"adj_Traj_Cal time cost="<<elapsed.count()<<"ms"<<endl;
    }
    void Move_TrajsCal(){
        auto start=std::chrono::high_resolution_clock::now();
        //计算运动的轨迹
        //计算六条腿目标点，对指定步长进行分段可行性检查，得到每条腿最终目标点
        if(vec_des.norm()<=0.01&&abs(omega_des)<=0.01){
            ROS_WARN("Leg_DesPosCal: vec_des or omega_des is too small, please check");
            GREEN("--------------->vec_des="<<vec_des.transpose()<<" omega_des="<<omega_des<<"<---------------");
            return;
        }
        else{
            GREEN("--------------->vec_des="<<vec_des.transpose()<<" omega_des="<<omega_des<<"<---------------");
        }
        vector<int> min_feasi_indexes(2,INT_MAX);
        for(auto &leg:legs){
            if(leg->state == STANCE || leg->state == SWING || leg->state == RETRACE || leg->state == ATTACH){
                leg->Leg_DesPosCal(vec_des,omega_des);
                if(leg->state == STANCE){
                    if(leg->traj_cal_index<min_feasi_indexes[0]) min_feasi_indexes[0]=leg->traj_cal_index;
                }
                else if(leg->state == SWING){
                    if(leg->traj_cal_index<min_feasi_indexes[1]) min_feasi_indexes[1]=leg->traj_cal_index;
                }
            }
        }

        //根据最小的index调整腿的末端位置想通，调整traj_cal_index相同,只有STANCE和SWING需要
        int index;
        for(auto &leg:legs){
            if(leg->state == STANCE) index=0;
            else if(leg->state == SWING) index=1;
            else continue;        
            if(omega_des!=0) leg->fai_t = leg->fai_traj[min_feasi_indexes[index]];
            leg->B_e_t = leg->B_e_traj[min_feasi_indexes[index]];
            leg->traj_cal_index=min_feasi_indexes[index];
        }
        //根据index计算质心轨迹，主要用于可视化等工作
        

        //计算每条腿的轨迹
        for(auto &leg:legs) {
            if(leg->state == STANCE || leg->state == SWING|| leg->state == RETRACE || leg->state == ATTACH){
                // RED(leg->name<<"E_traj_cal: "<<leg->state);
                leg->E_TrajCal(vec_des,omega_des);

                //可视化腿部轨迹
                // VIS_PTR->VisTraj(leg->B_e_t_traj,leg->R_T_B,leg->name);
            }
        }
        auto end=std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        cout<<"Move_TrajsCal time cost="<<elapsed.count()<<"ms"<<endl;
    }
    void Rec_usr_command(const interface::joy_commandConstPtr& joy_command){
        //接受用户初始化指令
        if(joy_command->set_init){
            init_done=false;
            set_move=false;
            SetInit(joy_command->moving);
            cout<<"set_init"<<endl;
        }
        //接受用户切换模式指令
        if(joy_command->change_mode){
            RUNNING_MODE=RUNNING_MODE==SUPPORT_MODE? CLAMP_MODE:SUPPORT_MODE;
            GREEN("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>Robot now running under "<<RUNNING_MODE<<" <<<<<<<<<<<<<<<<<<<<<<<<<<");
        }
        //接受用户运动控制指令
        if(init_done){
            if(loop_done&&!STOP_FLAG){
                if(RUNNING_MODE==SUPPORT_MODE){
                    vec_des<<joy_command->x_vec,joy_command->y_vec,joy_command->z_vec;
                    omega_des=joy_command->w_twist;
                }
                else if(RUNNING_MODE==CLAMP_MODE){
                    //对于加持状态的机器人，只控制y方向的速度,由于加持运动每一步的步长较短，运动速度减慢
                    vec_des<<0,joy_command->y_vec/3.0,0;
                    omega_des=0;
                }
                //只有上一次指令控制机器人执行一个周期完毕后，才计算新的指令
                if(vec_des.norm()>=0.01||abs(omega_des)>=0.01){
                    //计算六条腿目标点，对指定步长进行分段可行性检查，得到每条腿最终目标
                    YELLOW("------------------>Rec vec_des and omega_des<------------------");
                    GREEN("Vec_des="<<vec_des.transpose()<<", omega_des="<<omega_des<<endl);
                    Move_TrajsCal();
                    set_move=true;
                }
            }
            //户发出停止指令，停止运动，机器人保持当前状态
            else if(joy_command->stop){
                STOP_FLAG=true;
                loop_done=true;
                set_move=false;
                YELLOW("------------------>Stop<------------------");
            }
            else{
                // YELLOW("------------------>Last loop is not done<------------------");
            }          
        }
        //接受用户紧急停止指令
        if(joy_command->action_valve|| joy_command->disable_pump || joy_command->disable_torque){
            Disable(joy_command);
            init_done=false;
        }

    }
    void GaitPlanning(const ros::TimerEvent& e){
        //用户未设置停止命令，机器人开始计算正运动学并执行步态规划阶段
        if(!STOP_FLAG){
            for(auto& leg:legs){
                KIN_PTR->FwdKin(leg->sita_cur,leg->B_e_cur);
                leg->traj_execute_done=(leg->B_e_cur-leg->B_e_t).norm()<0.05;
                // if(leg->traj_execute_done || leg->traj_pub_done){
                if(leg->traj_pub_done){
                    leg->B_e_0=leg->B_e_t;
                }
                // if(leg->traj_execute_done){
                //     leg->B_e_0=leg->B_e_cur;
                // }
            }
        }
        //当用户设置停止命令时，机器人立刻停止运动，不进入步态规划阶段，直接返回
        else{
            for(auto& leg:legs){
                cout<<"traj_pub_index="<<leg->traj_pub_index<<endl;
                leg->traj_pub_index=leg->traj_pub_index==0? 1:leg->traj_pub_index;
                leg->B_e_0=leg->B_e_t_traj[leg->traj_pub_index-1];
                leg->B_e_t_traj.clear();
                if(leg->state==ADJUST)leg->traj_pub_done=true;
            }
            STOP_FLAG=false;
            GREEN("------------------->Loop Done by Stop<-------------------");
            return;
        }
        if(!init_done){
            if(set_move_init){
                loop_done=true;
                if(RUNNING_MODE==SUPPORT_MODE){
                    for(auto &leg:legs){
                        if(!leg->traj_execute_done){
                            loop_done=false;
                            break;
                        }
                    }
                }
                else if(RUNNING_MODE==CLAMP_MODE){
                    for(auto& leg:legs){
                        if(leg->state==STANCE){
                            if(leg->ContactDetection()){
                                leg->sita_des_msg.data.assign({TRAPEZOID,leg->sita_cur(0),leg->sita_cur(1),leg->sita_cur(2),leg->GetFootAngle(leg->sita_cur)});
                                leg->sita_des_pub.publish(leg->sita_des_msg);
                                leg->B_e_t=leg->B_e_cur;
                            }
                            else{
                                loop_done=false;
                            }
                        }
                    }
                }
                if(loop_done){
                    init_done=true;
                    GREEN("------------------->Init Done<-------------------");
                    // //当完成初始化再设置运动,更改电机模式,为保证安全,设置3s延时在继续
                    // cout<<"change motor mode\n";

                    // for(int i=0;i<3;i++){
                    //     for(auto& leg:legs){
                    //         leg->sita_des_msg.data[4]=0;
                    //         leg->sita_des_msg.data[5]=0;
                    //         leg->sita_des_msg.data[6]=0;                            
                    //         leg->sita_des_pub.publish(leg->sita_des_msg);
                    //     }
                    //     ros::Duration(0.1).sleep();
                    // }
                    // // ros::Duration(1.0).sleep();
                    // cout<<"change motor mode wait done\n";
                }
            }
        }
        else{
            PV_Pub();           
            if(!set_move) return;
            YELLOW("before GaitPlanning");
            for(auto &leg:legs) {
            //     // if(leg->name==LM) cout<<leg->name<<": "<<leg->state<<"; traj_pub_done="<<leg->traj_pub_done<<endl;
                cout<<leg->name<<": "<<leg->state<<"; traj_pub_done="<<leg->traj_pub_done<<"; traj_index="<<leg->traj_pub_index<<"; leg->B_e_t_traj.size()="<<leg->B_e_t_traj.size()<<endl;
            }
            cout<<endl;
            //Decide state for each leg
            int adjust_num=0, adjust_done_num=0, stance_done_num=0,retrace_done_num=0,attach_done_num=0,wait_adsorb_num=0,wait_release_num=0,failed_num=0,retrace_num=0;
            for(auto &leg:legs){
                if(leg->state == STANCE){
                    if(leg->traj_pub_done) {
                        leg->state=STANCE_DONE;
                        leg->B_e_contact=leg->B_e_t_traj.back();
                    }
                }
                else if(leg->state == SWING){
                    if(leg->traj_pub_done) {
                        leg->state=SWING_DONE;
                    }
                }
                else if(leg->state == RETRACE){
                    if(leg->traj_pub_done) {
                        leg->state=RETRACE_DONE;
                    }
                }
                else if(leg->state == ATTACH){
                    //对于ATTACH状态，轨迹都是可行性范围内最大距离，判断到完成接触，则进入ATTACH_DONE状态,轨迹执行完毕
                    // RED("executing ContactDetection");
                    if(leg->ContactDetection()){
                        leg->state=ATTACH_DONE;
                        leg->traj_execute_done=true;
                        leg->traj_pub_done=true;
                        //为了使Sita_TrajCal停止发布命令，先记录当前足端位置作为B_e_t
                        if (leg->traj_pub_index<=leg->B_e_t_traj.size()){
                            leg->B_e_t=leg->B_e_t_traj[leg->traj_pub_index-1];
                        }
                        else leg->B_e_t=leg->B_e_t_traj.back();
                        // leg->B_e_t=leg->B_e_cur;
                        //leg->B_e_0=leg->B_e_t;
                        //然后设置traj_pub_index=B_e_t_traj.size()
                        leg->traj_pub_index=leg->B_e_t_traj.size();
                        // if(leg->name==RF){
                        //     ROS_ERROR("set traj_pub_index");
                        //     cout<<"traj_pub_index="<<leg->traj_pub_index<<endl;
                        // }
                    }
                    //未检测到接触，此时可行性轨迹已经执行完毕，说明该腿踩空了，则进入FAILED状态
                    else if(leg->traj_execute_done) {
                        leg->state=FAILED;
                        leg->traj_execute_done=false;
                        // failed_num++;
                    }
                }
                else if(leg->state == FAILED){
                    //对于FAILED状态，其余腿在调整身体姿态帮助该腿尽量靠近曲面，此时该腿要不断检测接触状态
                    if(leg->ContactDetection()){
                        leg->state=ATTACH_DONE;
                        leg->traj_execute_done=true;
                    }
                    //未接触到检测，此时身体已经完成调整，说明机器人调整失败
                    else if(leg->traj_execute_done){
                        RED("------------------->Adjust Failed<-------------------");
                    }
                }
                else if(leg->state == ADJUST){
                    if(leg->traj_pub_done) leg->state=ADJUST_DONE;
                    else adjust_num++;
                }

                if(leg->state == RETRACE_DONE){
                    retrace_done_num++;
                }
                else if(leg->state == ADJUST_DONE){
                    adjust_done_num++;
                }
                else if(leg->state == STANCE_DONE){
                    stance_done_num++;
                    //skip adjust by following 
                    // adjust_done_num++;
                }
                else if(leg->state == ATTACH_DONE){
                    attach_done_num++;
                }
                else if(leg->state == WAIT_ADSORB){
                    wait_adsorb_num++;
                }
                else if(leg->state == WAIT_RELEASE){
                    wait_release_num++;
                }
            }
            // YELLOW("In GaitPlanning");
            // for(auto &leg:legs) {
            //     if(leg->name==LM) cout<<leg->name<<": "<<leg->state<<"; traj_pub_done="<<leg->traj_pub_done<<endl;
            //     // cout<<leg->name<<": "<<leg->state<<"; traj_pub_done="<<leg->traj_pub_done<<"leg->B_e_t_traj.size()="<<leg->B_e_t_traj.size()<<endl;
            // }            
            //此时已经完成腿部规矩是否执行完的判断，此时判断腿部是否达到目标状态
            loop_done=true;
            for(auto &leg:legs){
                //腿没有达到目标状态,或者轨迹未执行完，循环就还未停止
                if(leg->state!=leg->des_state||!leg->traj_pub_done){
                    loop_done=false;
                }
            }
            if(loop_done){
                loop_done=true;
                set_move=false;
                for(auto& leg:legs){ 
                    // leg->des_state=leg->des_state==STANCE_DONE?SWING_DONE:STANCE_DONE;
                }
                GREEN("------------------->Loop Done<-------------------");
            }

            if(stance_done_num==3){
                for(auto &leg:legs){
                    if(leg->state == SWING_DONE){
                        leg->state=ATTACH;
                    }
                }
            }
            if(adjust_done_num==3){
                for(auto &leg:legs){
                    if(leg->state == ADJUST_DONE){
                        leg->state = STANCE;
                    }
                    if(leg->state == RETRACE_DONE){
                        if(RUNNING_MODE==SUPPORT_MODE){
                            leg->state = SWING;
                        }
                        else if(RUNNING_MODE==CLAMP_MODE){
                            leg->state = SWING_DONE;
                            // leg->state = SWING;
                        }
                    }
                }
            }
            if(attach_done_num==3){
                tic_attach=ros::Time::now();
                for(auto &leg:legs){
                    if(leg->state == ATTACH_DONE){
                        leg->state = WAIT_ADSORB;
                    }
                }
            }
            //完成吸附，STANCE_DONE开始释放
            //采用延时判断吸附状态
            // if(wait_adsorb_num==3&&stance_done_num==3&&(ros::Time::now()-tic_attach).toSec()>0.5){
            //采用负压传感器判断吸附状态
            // if(wait_adsorb_num==3&&stance_done_num==3){
            bool stable_adsorb=false;
            if(wait_adsorb_num==3&&stance_done_num==3){
                tic_release=ros::Time::now();
                double negative_pressure=0;
                for(auto &leg:legs){
                    if(leg->state == WAIT_ADSORB){
                        negative_pressure=leg->negative_pressure;
                        //实物判据，采用负压传感器数据
                        if(leg->negative_pressure<-30){
                            stable_adsorb=true;
                        }
                        //仿真判据，采用吸附时间加接触数量判断，实物实验可以注释或者不注释掉
                        if(!use_neg){
                            if((ros::Time::now()-tic_attach).toSec()>0.01){
                                stable_adsorb=true;
                            }
                        }
                    }

                }
                if(stable_adsorb){
                    for(auto &leg:legs){
                        if(leg->state == STANCE_DONE){
                            leg->state = WAIT_RELEASE;
                        }
                    }
                }

                //cout<<"negative_pressure="<<negative_pressure<<endl;
                if((ros::Time::now()-tic_attach).toSec()>0.5&&negative_pressure>=-45){
                    ROS_WARN("ADSORB timeout, negative_pressure=%f",negative_pressure);
                }
            }
            //A 组腿完成吸附，使用延时控制B组腿释放
            if(wait_release_num==3 && (ros::Time::now()-tic_release).toSec()>0.01){
                double negative_pressure=-30;
                for(auto &leg:legs){
                    if(leg->state == WAIT_RELEASE){
                        negative_pressure=leg->negative_pressure;
                        if(leg->negative_pressure>-2){
                            leg->state = RETRACE;
                        }
                        //仿真判据，采用吸附时间加接触数量判断，实物需要注释
                        if(!use_neg){
                            if((ros::Time::now()-tic_release).toSec()>0.01){
                                leg->state = RETRACE;
                            }
                        }
                    }
                }
                if((ros::Time::now()-tic_attach).toSec()>0.5&&negative_pressure<=0){
                    ROS_WARN("RELEASE timeout, negative_pressure=%f",negative_pressure);
                }
            }
            if(retrace_done_num==3){
                for(auto &leg:legs){
                    if(leg->state == WAIT_ADSORB){
                        leg->state = ADJUST;
                        adjust_num++;
                        //skip adjust by following
                        // leg->state = STANCE;
                    }
                }
            }
            YELLOW("After GaitPlanning");
            for(auto &leg:legs){
                cout<<leg->name<<": "<<leg->state<<"traj_index="<<leg->traj_pub_index<<"; traj_pub_done="<<leg->traj_pub_done<<endl;
                // if(leg->state == ADJUST){
                    // cout<<leg->name<<": "<<leg->state<<"; traj_pub_done="<<leg->traj_pub_done<<"; leg->des_state="<<leg->des_state<<endl;
                // }
            }
            cout<<endl;

            bool trajs_done=true;
            if(adjust_num==3){
                for(auto &leg:legs){
                    if(leg->state == ADJUST){
                        //若当前轨迹还未执行完，说明调整姿态的轨迹已经计算完毕，还在执行中
                        if(!leg->traj_pub_done){
                            trajs_done=false;
                            break;
                        }
                    }
                }
                if(trajs_done){
                    Adj_TrajsCal(failed_num);
                }
            }
            bool traj_cal=false;
            for(auto &leg:legs){
                if(leg->traj_pub_done&&!loop_done){
                    if(leg->state == STANCE|| leg->state == SWING || leg->state == RETRACE || leg->state == ATTACH){
                        traj_cal=true;
                    }
                }
            }
            if(traj_cal){
                Move_TrajsCal();
            }

            // //计算质心期望姿态，位置，如果omega_des!=0,用于可视化
            // if(omega_des!=0){
            // }
        }
    }
    void Disable(const interface::joy_commandConstPtr& joy_command){
        if(joy_command->disable_pump){
            ROS_WARN("DISABLE_PUMP");
            for(int i=0; i<12; ++i){
                pv_command[i]=0;
            }
            std_msgs::Int16MultiArray pv_command_msg;
            pv_command_msg.data.assign(pv_command.begin(),pv_command.end());
            this->pv_command_pub.publish(pv_command_msg);
        }
        else if(joy_command->disable_torque){
            ROS_WARN("DISABLE_TORQUE");
            for(auto leg:legs){
                leg->sita_des_msg.data.assign({DISABLE,0,0,0,0,0,0,-999});
                leg->sita_des_pub.publish(leg->sita_des_msg);
            }
        }
        else if(joy_command->action_valve){
            ROS_WARN("ACTION_VALVE");
            pv_command[0]=-1;
            std_msgs::Int16MultiArray action_valve_msg;
            action_valve_msg.data.assign(pv_command.begin(),pv_command.end());
            this->pv_command_pub.publish(action_valve_msg);
        }
        init_done=false;
        set_move_init=false;
        set_move=false;
        loop_done=false;
        for(auto& leg:legs) leg->B_e_t_traj.clear();
    }
    void PV_Pub(){
        std_msgs::Int16MultiArray pv_command_msg;
        pv_command_last=pv_command;
        int i=0;
        for(auto leg:legs){
            if(leg->state == STANCE || leg->state == ATTACH_DONE || leg->state == STANCE_DONE || leg->state == WAIT_ADSORB || leg->state == ADJUST || leg->state == ADJUST_DONE){
                leg->p_state =1;
                leg->v_state =0;
            }
            else if(leg->state == WAITING){
                leg->p_state = 0;
                leg->v_state = 0;
            }
            else {
                leg->p_state = 0;
                leg->v_state = 1;
            }
            this->pv_command[i]=leg->p_state;
            this->pv_command[i+1]=leg->v_state;
            i += 2;
        }
        if(pv_command_last!=pv_command) {
            pv_command_msg.data.assign(pv_command.begin(),pv_command.end());
            this->pv_command_pub.publish(pv_command_msg);
        }
    }  
    void NegPre(const std_msgs::Int32MultiArray::ConstPtr& msg){
        //0对应A组的腿(RF RB LM)负压，1对应B组的腿负压
        if(msg->data.size()!=2){
            ROS_ERROR("NegPre msg.data.size() !=2");
            return;
        }
        for(auto& leg:legs){
            if(leg->name==RF||leg->name==RB||leg->name==LM){
                leg->negative_pressure=-100+(100/2700.0)*msg->data[1];
            }
            else{
                leg->negative_pressure=-100+(100/2700.0)*msg->data[0];
            }
            // cout<<leg->name<<": neg_pre:"<<leg->negative_pressure<<endl;
        }
    }
    void RobotState(const gazebo_msgs::ModelStates::ConstPtr& msg){
        int tag=0;
        for(int j=0; j< msg->name.size(); j++){
            if (msg->name[j] == ROBOT_NAME){
                tag = j;
                break;
            }
        }
        W_Q_R=Eigen::Quaterniond(msg->pose[tag].orientation.w,msg->pose[tag].orientation.x,msg->pose[tag].orientation.y,msg->pose[tag].orientation.z);

        // Eigen::Quaterniond q;
        // if(!init_done){
            // q=Eigen::Quaterniond(msg->pose[tag].orientation.w,msg->pose[tag].orientation.x,msg->pose[tag].orientation.y,msg->pose[tag].orientation.z);
            // Rini_rot_W=q.toRotationMatrix();
            // W_rot_R=Rini_rot_W;
            // Rini_rot_W.inverse();
        // }
        // else{
            // q=Eigen::Quaterniond(msg->pose[tag].orientation.w,msg->pose[tag].orientation.x,msg->pose[tag].orientation.y,msg->pose[tag].orientation.z);
            // W_rot_R=q.toRotationMatrix();
            // W_rot_R=Rini_rot_W.inverse()*W_rot_R;
        // }
        // Eigen::Quaterniond q(msg->pose[tag].orientation.w,msg->pose[tag].orientation.x,msg->pose[tag].orientation.y,msg->pose[tag].orientation.z);
        // W_rot_R=q.toRotationMatrix();
        // euler_angle=W_rot_R.eulerAngles(2,1,0).array()/M_PI*180;
        // cout<<"Robot euler="<<euler_angle.transpose()<<endl;
    }
    void ImuState(const geometry_msgs::PoseStamped::ConstPtr& msg){
        // Eigen::Quaterniond q(msg->pose.orientation.w,msg->pose.orientation.x,msg->pose.orientation.y,msg->pose.orientation.z);
        // W_rot_R=q.toRotationMatrix();
        // euler_angle(0)=msg->pose.position.x;
        // euler_angle(1)=msg->pose.position.y;
        // euler_angle(2)=msg->pose.position.z;
    }
};


#endif