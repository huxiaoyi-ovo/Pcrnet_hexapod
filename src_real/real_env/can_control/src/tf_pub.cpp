#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tf_pub");
  ros::NodeHandle n;
  ros::Rate r(50);
  tf::TransformBroadcaster broadcaster;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.frame_id = "lidar";
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.child_frame_id = "robot";
  transformStamped.transform.translation.x = 0.0;
  transformStamped.transform.translation.y = 0.0;
  transformStamped.transform.translation.z = 0.0;
  transformStamped.transform.rotation.x = 0.0;
  transformStamped.transform.rotation.y = 0.0;
  transformStamped.transform.rotation.z = 0.0;
  transformStamped.transform.rotation.w = 1.0;

  while (ros::ok())
  {
    static double x = 0.0;
    x+=0.01;
    printf("x=%f\n",x);
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.transform.translation.x = x;
    broadcaster.sendTransform(transformStamped);
    r.sleep();
  }

  return 0;
}