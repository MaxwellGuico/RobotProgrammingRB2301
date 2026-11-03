#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v rviz2 >/dev/null 2>&1; then
    for distro in "${ROS_DISTRO:-foxy}" foxy jazzy; do
        if [[ -f "/opt/ros/$distro/setup.bash" ]]; then
            source "/opt/ros/$distro/setup.bash"
            break
        fi
    done
fi
# Keep the caller's ROS_DOMAIN_ID and RMW settings; do not force a robot number.
exec rviz2 -d "$script_dir/rviz/fp_robot_thinking.rviz" "$@"
