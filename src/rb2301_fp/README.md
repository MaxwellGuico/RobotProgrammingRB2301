# Obstacle-course navigation

The controller uses LiDAR and pose to build an observed grid, choose frontiers,
plan with A*, and drive the route. Blocked routes are replanned from the observed map; no breadcrumb trail is stored. It publishes **forward speed and yaw**;
`Vy` is always zero. The controller never reads the Gazebo world file.

## Run

Source your installed ROS distribution and build the workspace as usual. Gazebo
Harmonic simulation uses Jazzy here; the robot's RViz setup targets Foxy too.

```bash
# Terminal 1: simulation
./gz_fp.sh

# Terminal 2: full-course controller in simulation
./fp.sh --ros-args -p is_simulation:=true

# RViz, using the caller's ROS version and domain
./src/rb2301_fp/rviz_fp.sh
```

`is_simulation` defaults to `false`, selecting the physical robot and its
Optitrack topic. A plain `./fp.sh` runs full-course navigation once fresh pose
and LiDAR data arrive. On the physical robot:

```bash
./fp.sh
```

Use your robot's `ROS_DOMAIN_ID` and measured LiDAR mount. `laser_x`, `laser_y`
(metres) and `laser_yaw` (radians) are startup ROS parameters. Simulation defaults
match `nanocar_description.urdf`: `(0.11423, 0, 3.1416)`. Hardware defaults are
`(0, 0, 0)`, so pass the measured values for that robot. Optitrack frame calibration
lives in `rb2301_fp/optitrack_variables.config`. Translation on hardware is capped
at 0.3 m/s. `--calibrate` prints poses without driving.

Ctrl+C commands zero velocity and shuts down. Run one controller at a time.

## Physical robot on Foxy

With the robot's usual base/LiDAR drivers and Optitrack bridge running, use the
workspace on the robot:

```bash
cd ~/rb2301
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=7  # Example: replace 7 with your robot number.
colcon build --symlink-install --packages-select rb2301_fp
./fp.sh --calibrate
```

Calibration mode prints poses without driving. Ctrl+C exits it; `./fp.sh` then
starts full navigation. The existing physical Optitrack Y flip is applied to
both the displayed pose and the timestamped mapping pose. Set the measured
`laser_x`, `laser_y` and `laser_yaw` startup parameters for the real LiDAR mount.
In another terminal, source Foxy, use the same domain and workspace, then run
`./src/rb2301_fp/rviz_fp.sh`.

## Passage check

The passage test uses **robot width only**. A free route cell must be at least
`footprint_half_width` from an occupied cell. The default half-width is 0.096 m,
so a 0.20 m gap fits a 0.192 m robot and a 0.15 m gap does not. There is no extra
turning radius, drift allowance, or larger clearance threshold for goals.

A* prefers the centre of wider space when possible. The follower turns to face the look-ahead point before driving forward. Before driving forward, the controller checks that the next centre-line segment
fits the robot width. There is no rectangular-body sweep, braking-distance
prediction, or acceleration smoothing. Missing or stale sensor data still
commands zero velocity.

Settings are in `ObstacleCourseNode.__init__()`. Full-course navigation is
the default; there are no stage or primitive selection switches.

## Moving-obstacle staging

The robot chooses the left lane in observed space relative to its crossing
heading. It reaches the staging point, stops, and aligns down the corridor before
observing. It measures two complete openings for the same side of the swing,
then uses the next suitable opening in the third cycle. Opposite-side openings
are timed separately because their durations can differ.

## Visible iterative attempts

```bash
source install/setup.bash
python3 tools/fp_attempt.py --name attempt_003 --world /path/to/frozen_course.sdf
```

This opens Gazebo, RViz and a controller terminal. A failed controller, closed
window, prolonged lack of progress, or time limit ends the attempt and closes
all three processes. Ctrl+C in the launcher does the same. Each attempt writes
its logs and `result.json` under `log/fp_attempts/<name>/`. Use a new name each time.
`RB2301_RANDOMIZE=0` prevents the launch file from changing the frozen layout.

## RViz on Foxy

`rviz_fp.sh` uses the existing ROS environment and preserves `ROS_DOMAIN_ID`.
If ROS is not sourced, it searches installed distributions, preferring Foxy.
It has no fixed username, workspace path, Jazzy dependency, or forced robot domain.

The view uses only the standard Grid and MarkerArray plugins. The marker display
uses the `Topic` property supported by both the
[Foxy plugin](https://github.com/ros2/rviz/blob/foxy/rviz_default_plugins/src/rviz_default_plugins/displays/marker_array/marker_array_display.cpp)
and [Jazzy plugin](https://github.com/ros2/rviz/blob/jazzy/rviz_default_plugins/src/rviz_default_plugins/displays/marker_array/marker_array_display.cpp).
The observed map is rendered as markers too, avoiding the indexed-texture Map
shader that failed on this workstation. All displays use `fp_world`, so RViz does
not require a separate display TF tree or newer plugins.

Grey cells are observed free space, red cells are obstacles, cyan points are
current LiDAR returns, purple points are frontiers, and yellow is the planned route.
The green arrow and rectangle show the robot and footprint. The red sphere is the
goal; the orange sphere and line labelled CARROT show the look-ahead point.

The launcher/config were checked against Foxy's source and tested locally with
Jazzy. A live Foxy installation is not available here; the robot test remains
necessary. Set `RB2301_DEBUG_VISUALS=0` to turn visualization publishing off.

## Checks

```bash
source install/setup.bash
python3 -m unittest discover -s src/rb2301_fp/test -v
```

These checks cover projection, width-only line checks and planning,
`Vy=0`, robot mode selection, LiDAR mounting, and the Foxy-compatible RViz entry
point. Full-course acceptance comes from the recorded simulation attempts, not
from these unit checks alone.
