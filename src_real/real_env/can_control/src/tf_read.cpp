#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
// #include<tf/tf_transform.h>
int main(int argc, char **argv)
{
  ros::init(argc, argv, "tf_read");
  ros::NodeHandle n;
  tf::TransformListener listener;

  tf::StampedTransform transformStamped;
  ros::Rate rate(100);
  while (ros::ok())
  {
    try
    {
        listener.waitForTransform("robot", "wall_link", ros::Time(), ros::Duration(1.0));
        listener.lookupTransform("robot", "wall_link", ros::Time(), transformStamped);
        ROS_INFO("Translation: (%f, %f, %f)", transformStamped.getOrigin().x(), transformStamped.getOrigin().y(), transformStamped.getOrigin().z());
        // ROS_INFO("Rotation: (%f, %f, %f, %f)", transformStamped.getRotation().x(), transformStamped.getRotation().y(), transformStamped.getRotation().z(), transformStamped.getRotation().w());
    // ROS_INFO("Translation:%f", transformStamped.getOrigin().x());
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        return 1;
    }
    rate.sleep();
  }
    
  return 0;
}