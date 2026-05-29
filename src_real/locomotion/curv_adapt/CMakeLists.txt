cmake_minimum_required(VERSION 3.5)

project(curv_adapt)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  roscpp
  rospy
)
catkin_package()
include_directories(
  include
  "/usr/include/eigen3"
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../utils/include
  ${PROJECT_SOURCE_DIR}/../../../devel/include
)
add_executable(curv_adapt src/curv_adapt.cpp)
add_executable(temp_test src/temp_test.cpp)

target_link_libraries(curv_adapt ${catkin_LIBRARIES})
target_link_libraries(temp_test ${catkin_LIBRARIES})



catkin_install_python(PROGRAMS
 scripts/for_test.py scripts/plot_traj.py scripts/curv_adapt.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
