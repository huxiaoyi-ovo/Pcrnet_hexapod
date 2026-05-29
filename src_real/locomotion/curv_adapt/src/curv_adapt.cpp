//calculate traj segment every gait
#include "curv_adapt/curv_adapt.hpp"
int main(int argc, char**argv){

    MODE=argv[1];
    GREEN_WHITE(">>>>>>>>>>>>>>>>robot running in "<<MODE<<" mode<<<<<<<<<<<<<<<<<");

    double p0=0,pt=2,t=3;
    // POLYNOIMAL poly1(p0,pt,t);
    ros::init(argc,argv,"traj_seg_cal");
    ros::NodeHandle nh;
    ros_time_start=ros::Time::now().toSec();
    cout<<"ros_time_start=========="<<std::fixed<<ros_time_start<<endl;
    TRAJ_PTR = make_shared<POLYNOIMAL>();
    KIN_PTR = make_shared<KINEMATIC>(nh);
    HEDRA_PTR = make_shared<POLYHEDRA3>(nh); 
    HEDRA2_PTR = make_shared<ROBOT_POLYHEDRA2>("normal");
    // VIS_PTR = make_shared<VISUALIZER>(nh); 
    // VIS_PTR->TrajPointVisInit();
    if(ConfigParam(nh)){
        HEXAPOD hexapod(nh);
        // hexapod.legs[0]->sita_cur<<0,0,0;
        // hexapod.legs[0]->KIN_PTR->FwdKin(hexapod.legs[0]->sita_cur,hexapod.legs[0]->B_e_cur);
        // RED(hexapod.legs[0]->B_e_cur);
        ros::spin();
    }
    return 0;
}
