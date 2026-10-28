#!/bin/bash

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Project-specific ROS settings
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=5
export ROS_LOCALHOST_ONLY=0

# Launch RViz with the project configuration
rviz2 -d /home/maxwell-guico/rb2301/src/rb2301_fp/rviz/fp_robot_thinking.rviz