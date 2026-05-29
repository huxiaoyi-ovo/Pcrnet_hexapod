#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
// #include "control/traj_seg_cal.hpp"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "utils/kinematic.hpp"

using namespace std;
using namespace Eigen;

// bool InPolyHedron(Eigen::Vector3d& p, Eigen::Matrix<double,-1,5>& hPoly){
//     int dot_res;
//     for(int i=0; i<hPoly.rows();++i){
//         dot_res = hPoly.row(i).block<3,1>(i,0).dot(p)>0? 1:-1;
//         if(dot_res != hPoly.row(i)(4)){
//             return false;
//         }
//     }
//     return true;
// }
// void cb1(const geometry_msgs::PoseStampedConstPtr& msg,KINEMATIC &Kin,ros::Publisher& pub){
//     geometry_msgs::PoseStamped pose_msg;
//     pose_msg.header.stamp=ros::Time::now();
    
// }
int main(int argc, char** argv)
{
   

    ros::init(argc, argv, "temp_test");
    ros::NodeHandle nh;
    KINEMATIC Kin(nh);
    Matrix3d damp_inv_jac;
    Vector3d sita_cur(0.2,-1.0,1.0);
    Vector3d pos;
    // Kin.FwdKin(sita_cur,pos);
    // cout<<"pos: "<<pos.transpose()<<endl;
    Kin.DampInvJac(damp_inv_jac,sita_cur);
    cout<<"damp_inv_jac:\n "<<damp_inv_jac<<endl;
    return 0;
}