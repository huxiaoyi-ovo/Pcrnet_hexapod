#include"utils/visualizer.hpp"
using namespace std;

int main(int argc, char** argv){
    //visualize joints move in rviz 
    ros::init(argc,argv,"vis_robot_state");
    ros::NodeHandle nh;
    VISUALIZER vis(nh);
    vis.StateVisInit();
    ros::spin();
    return 0;
}