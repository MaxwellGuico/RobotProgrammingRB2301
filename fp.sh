#!/bin/bash
# Runs obstacle_course with real_robot.yaml always loaded (a bare ./fp.sh used to
# run on the code defaults). Extra arguments still work, e.g. ./fp.sh --calibrate
cd "$(dirname "$0")"
source install/setup.bash
ros2 run rb2301_fp obstacle_course "$@" --ros-args --params-file src/rb2301_fp/params/real_robot.yaml
