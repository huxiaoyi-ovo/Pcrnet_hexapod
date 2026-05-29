cmake_minimum_required(VERSION 3.0.2)
project(servo)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES servo
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
include/servo
  ${catkin_INCLUDE_DIRS}
)


# link_directories(include/servo)
# link_libraries(libSCServo.a)


# add_executable(WritePos src/WritePos.cpp ${hdrs} ${srs})


# target_link_libraries(WritePos
#   ${catkin_LIBRARIES} libSCServo.a
# )
