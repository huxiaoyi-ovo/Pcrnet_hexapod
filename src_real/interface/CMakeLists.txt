cmake_minimum_required(VERSION 3.0.2)
project(interface)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  message_generation
  message_runtime
)
add_message_files(
  FILES
  joy_command.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES interface
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  "/usr/include/eigen3"
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../utils/include
)

add_executable(joy_ctrl src/joy_ctrl.cpp)
add_executable(motor_interface src/motor_interface.cpp)
add_executable(pv_interface src/pv_interface.cpp)
add_executable(vis_robot_state src/vis_robot_state.cpp)

add_dependencies(joy_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(motor_interface ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(pv_interface ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(vis_robot_state ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(joy_ctrl
  ${catkin_LIBRARIES}
)
target_link_libraries(motor_interface
    ${catkin_LIBRARIES}
)
target_link_libraries(pv_interface
  ${catkin_LIBRARIES}
)
target_link_libraries(vis_robot_state
  ${catkin_LIBRARIES}
)
file(GLOB py_file1 "scripts/BC_learning/*.py")
file(GLOB py_file2 "scripts/physical_robot/*.py")
file(GLOB py_file3 "scripts/*.py")
catkin_install_python(PROGRAMS
  ${py_file1}
  ${py_file2}
  ${py_file3}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
