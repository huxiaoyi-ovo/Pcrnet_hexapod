//计算机器人腿部三自由度正运动学，雅可比矩阵为逆，迭代求解逆运动学
//五次多项式系数计算，值计算
#pragma once
#ifndef KINEMATIC_HPP
#define KINEMATIC_HPP
#include<ros/ros.h>
#include<Eigen/Dense>
#include<vector>
#include"ColorCout.h"
using namespace std;
class KINEMATIC{
private:
    double L1,L2,L3;
    ros::NodeHandle nh;
public:
    KINEMATIC(){}
    KINEMATIC(const ros::NodeHandle& n):nh(n){
        nh.getParam("L_1",L1);
        nh.getParam("L_2",L2);
        nh.getParam("L_3",L3);
    }
    void FwdKin(const Eigen::Vector3d& sita, Eigen::Vector3d& pos){
        pos<<
            L1*cos(sita(0))+L2*cos(sita(0))*cos(sita(1))+L3*cos(sita(0))*cos(sita(1)+sita(2)),
            L1*sin(sita(0))+L2*sin(sita(0))*cos(sita(1))+L3*sin(sita(0))*cos(sita(1)+sita(2)),
            L2*sin(sita(1))+L3*sin(sita(1)+sita(2));
    }
    bool InvKin(const Eigen::Vector3d& pos, Eigen::Vector3d& sita_cur, Eigen::Vector3d& sita_des, double sigma=0.01){
        Eigen::Vector3d pos_cur;
        Eigen::Matrix3d damp_inv_jac;
        FwdKin(sita_cur,pos_cur);
        for(int i=0;i<1000;i++){
            if((pos_cur-pos).norm()<=sigma) {
                sita_des = sita_cur;
                // cout<<"Iterative times: "<<i<<endl;
                return true;
            }            
            DampInvJac(damp_inv_jac,sita_cur);
            sita_cur = sita_cur + 0.05*damp_inv_jac*(pos-pos_cur);
            FwdKin(sita_cur,pos_cur);
        }
        cout<<"Iterative times: "<<1000<<endl;
        if((pos_cur-pos).norm()>sigma) {
            ROS_ERROR("Failed to iterate inverse kinematic!!!");
            return false;
        }
        else{
            return true;
        }
    }
    void DampInvJac(Eigen::Matrix3d& damp_inv_jac, const Eigen::Vector3d sita){
        Eigen::Vector3d J1, J2, J3;
        Eigen::Matrix3d jac_block;
        double sita1 = sita(0), sita2 = sita(1), sita3= sita(2);
        J1 << -L1*sin(sita1)-(L2*cos(sita2)+L3*cos(sita2+sita3))*sin(sita1), 
               L1*cos(sita1)+(L2*cos(sita2)+L3*cos(sita2+sita3))*cos(sita1),
               0;
        J2 << -(L2*sin(sita2)+L3*sin(sita2+sita3))*cos(sita1), 
              -(L2*sin(sita2)+L3*sin(sita2+sita3))*sin(sita1), 
                L2*cos(sita2)+L3*cos(sita2+sita3);
        J3 << -L3*cos(sita1)*sin(sita2+sita3), 
              -L3*sin(sita1)*sin(sita2+sita3), 
               L3*cos(sita2+sita3); 
        jac_block<<J1,J2,J3;
        cout<<"Jac_block:\n"<<jac_block<<endl;
        damp_inv_jac = jac_block.transpose()*((jac_block*jac_block.transpose()+0.0001*Eigen::Matrix3d::Identity()).inverse());
    }
    void DampTranInvJac(Eigen::Matrix3d& damp_tran_inv_jac, const Eigen::Vector3d sita){
        Eigen::Vector3d J1, J2, J3;
        Eigen::Matrix3d jac_block;
        double sita1 = sita(0), sita2 = sita(1), sita3= sita(2);
        J1 << -L1*sin(sita1)-(L2*cos(sita2)+L3*cos(sita2+sita3))*sin(sita1), 
               L1*cos(sita1)+(L2*cos(sita2)+L3*cos(sita2+sita3))*cos(sita1),
               0;
        J2 << -(L2*sin(sita2)+L3*sin(sita2+sita3))*cos(sita1), 
              -(L2*sin(sita2)+L3*sin(sita2+sita3))*sin(sita1), 
                L2*cos(sita2)+L3*cos(sita2+sita3);
        J3 << -L3*cos(sita1)*sin(sita2+sita3), 
              -L3*sin(sita1)*sin(sita2+sita3), 
               L3*cos(sita2+sita3); 
        jac_block<<J1,J2,J3;
        jac_block.transposeInPlace();
        damp_tran_inv_jac = jac_block.transpose()*((jac_block*jac_block.transpose()+pow(0.001,2)*Eigen::Matrix3d::Identity()).inverse());
    }
};

class POLYNOIMAL{
//根据初末状态计算五次多项式系数，计算多项式值
// private:
public:
    //c0=p0 v0=vt=0 a0=at=0
    //f(0) = p0
    //f_dot(t) = vt = 0
    //f_ddot(t) = at = 0
    //f(t)= c0+ c3*t^3 + c4*t^4 + c5*t^5
    double c0,c3,c4,c5,t;
    //A*coffs = b
    Eigen::Matrix3d A;
    Eigen::Vector3d B,coffs;
    //coffs = [c3,c4,c5] b=[p1-p0,0,0]

    //RETRACE 状态下的xy平面曲线，为了避免轨迹超出相邻凸包
    //x=a*y^2+b*y+c
    double a,b,c;
    
public:
    //计算五次多项式系数
    void CalCoffs(double& p0, double& pt, double& t){
        if(t==0&&p0==pt){
            coffs<<0,0,0;
        }
        else if(t==0&&p0!=pt){
            RED("Error: p0!=pt, but t=0");
            coffs<<0,0,0;
        }
        else if(t!=0&&p0==pt){
            // RED("Error: p0==pt, but t!=0");
            coffs<<0,0,0;
        }
        else{
            B << pt-p0,0,0;
            A << pow(t,3),    pow(t,4),     pow(t,5),
                3.0*pow(t,2),4.0*pow(t,3), 5.0*pow(t,4),
                6.0*t,       12.0*pow(t,2),20*pow(t,3);
            coffs = (A.inverse())*B;
        }
        c0=p0;
        c3=coffs[0];
        c4=coffs[1];
        c5=coffs[2];
    }
    void CalPoly(vector<double>& t,vector<double>& f_ts){
        f_ts.resize(t.size());
        for(int i=0;i<t.size();++i){
            //f(t)= c0+ t^3*( c3+t*(c4+c5*t) )
            f_ts[i]=c0+pow(t[i],3)*( c3+t[i]*(c4+c5*t[i]) );
        }
    }
    void CalPolyDot(vector<double>&t,vector<double>& f_dot_ts){        
        f_dot_ts.resize(t.size());
        for(int i=0;i<t.size();++i){
            //f_dot(t)=3c3*t^2 + 4c4*t^3 + 5c5*t^4=t^2*(3c3 + t*(4c4+5c5*t) )
            f_dot_ts[i]=pow(t[i],2)*(3*c3 + t[i]*(4*c4+5*c5*t[i]) );
        }
    }
    //计算二次多项式系数
    void CalCoffs2(vector<Eigen::Vector3d>& points){
        A<<pow(points[0](1),2),points[0](1),1,
           pow(points[1](1),2),points[1](1),1,
           pow(points[2](1),2),points[2](1),1;
        B<<points[0](0),points[1](0),points[2](0);
        coffs = (A.inverse())*B;
        a=coffs[0];
        b=coffs[1];
        c=coffs[2];
    }
    void CalPoly2(vector<double>& t,vector<double>& f_ts){
        f_ts.resize(t.size());
        for(int i=0;i<t.size();++i){
            //f(t)= a*y^2+b*y+c
            f_ts[i]=a*pow(t[i],2)+b*t[i]+c;
        }
    }
};
#endif
