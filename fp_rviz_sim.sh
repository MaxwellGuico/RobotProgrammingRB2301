#!/bin/bash
set -e
cd "$(dirname "$0")"
no_gui=0
replay=''
while (($#)); do
    case "$1" in
        --domain)
            if [[ ! ${2:-} =~ ^[0-9]+$ ]] || ((10#${2} > 232)); then
                echo 'Usage: ./fp_rviz_sim.sh [--domain 0-232] [--no-gui]' >&2
                exit 2
            fi
            export ROS_DOMAIN_ID="$2"
            shift 2
            ;;
        --no-gui) no_gui=1; shift ;;
        --replay)
            if [[ -z ${2:-} ]]; then echo '--replay requires a run name or trajectory path' >&2; exit 2; fi
            replay="$2"
            shift 2
            ;;
        --help|-h)
            echo 'Usage: ./fp_rviz_sim.sh [--domain 0-232] [--no-gui] [--replay RUN_NAME_OR_TRAJECTORY]'
            echo 'Attach to an existing simulation started with --terminal-debug.'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
# Keep the editor's Snap GTK libraries out of the system Qt/ROS process.
for task_env in $(compgen -e); do
    case "$task_env" in
        SNAP*|GTK*|GDK*|GIO*|GSETTINGS*) unset "$task_env" ;;
    esac
done
export XDG_DATA_DIRS="${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-/usr/local/share:/usr/share}"
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$PWD/log/fp_rviz_sim}"
mkdir -p "$ROS_LOG_DIR"
viewer_args=()
sim_time=true
if [[ -n $replay ]]; then
    if [[ ! -f $replay ]]; then replay="log/fp_sim_checks/$replay/trajectory.jsonl"; fi
    if [[ ! -f $replay ]]; then echo "Replay file not found: $replay" >&2; exit 2; fi
    viewer_args=(--replay "$replay")
    sim_time=false
fi
if ((no_gui)); then
    exec ros2 run rb2301_fp rviz_sim_view "${viewer_args[@]}" --ros-args -p "use_sim_time:=$sim_time"
fi
ros2 run rb2301_fp rviz_sim_view "${viewer_args[@]}" --ros-args -p "use_sim_time:=$sim_time" &
viewer_pid=$!
trap 'kill "$viewer_pid" 2>/dev/null || true; wait "$viewer_pid" 2>/dev/null || true' EXIT INT TERM
LIBGL_ALWAYS_SOFTWARE=1 rviz2 -d src/rb2301_fp/rviz/fp_simulation.rviz --ros-args -p "use_sim_time:=$sim_time"
