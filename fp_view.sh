#!/bin/bash
set -e
cd "$(dirname "$0")"
source install/setup.bash
ros2 run rb2301_fp terminal_view "$@"
