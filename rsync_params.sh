#!/usr/bin/env bash
# Copy only the real-robot tuning file to the robot. No rebuild needed:
# fp.sh reads it straight from src/ via --params-file.
set -euo pipefail

robot="${1:-}"

if [[ ! "$robot" =~ ^[0-9]{1,2}$ ]]; then
    echo "Usage: $0 <robot number, 0–99>"
    echo "Example: $0 3"
    exit 1
fi

printf -v nn '%02d' "$((10#$robot))"
target="bingda@192.168.1.2${nn}"
remote_dir="Downloads/rb2301_fp_max/src/rb2301_fp/params"

ssh "$target" "mkdir -p ~/${remote_dir}"
rsync -avx \
    /home/maxwell-guico/rb2301/src/rb2301_fp/params/real_robot.yaml \
    "${target}:~/${remote_dir}/"

echo
echo "On the robot, from ~/Downloads/rb2301_fp_max:"
echo "  ./fp.sh --ros-args --params-file src/rb2301_fp/params/real_robot.yaml"
