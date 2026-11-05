# Terminal viewer

Uses Python's built-in `curses` and existing ROS packages. No pip installation,
RViz, graphical desktop, or TF setup is needed. Works in an interactive SSH terminal.

Build once after updating:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select rb2301_fp --symlink-install
```

The controller defaults to the real robot and reads VRPN poses:

```bash
./fp.sh
```

Simulation must be selected explicitly; it reads `/odom`:

```bash
./fp.sh --simulation --terminal-debug --ros-args -p use_sim_time:=true
```

Mode is never inferred from the LiDAR's number of readings. The scan subscription
uses sensor-data QoS (best effort), compatible with best-effort and reliable
LiDAR publishers. Simulation odometry also uses sensor-data QoS; real VRPN
uses best-effort reliability. The controller starts without waiting for a scan, and waits
for the selected pose source and scan before navigating.

For real-robot navigation with optional state telemetry:

```bash
./fp.sh --terminal-debug
```

Real mode chooses `/vrpn_mocap/bingda_NNN/pose` from `ROS_DOMAIN_ID` when it is
between 1 and 99; otherwise it defaults to `bingda_003`. Check the startup
`Reading pose from ...` line against this robot's actual rigid-body topic.
The computer's hostname does not select the rigid body. `/odom` is unused in
real mode, so zero controller subscriptions on that topic is expected.

For the physical course, the configured course +x follows CA2 maze -y, with a
rotation of -90 degrees. The course origin has not been measured. Place the
robot's reference point at the intended start and face it up the first corridor.
Capture that position and inspect the transform without driving first:

```bash
./fp.sh --calibrate --start-here --terminal-debug
```

The course pose should start near `(0, 0, 0 degrees)`. Moving the robot up the
first corridor should increase course x; moving it right in the sketch should
decrease course y. Heading is still determined by Optitrack calibration, not
zeroed to whichever way the robot happens to face.

After stopping calibration, return the robot to the same intended start before
launching navigation:

```bash
./fp.sh --start-here --terminal-debug
```

`--start-here` captures only the first valid position, never reanchors during
motion, and does not save coordinates to disk. Every restart with this option
requires placing the robot back at the intended start. Simulation already has
its own course origin and ignores this option. The diagram establishes axis
directions; physical course dimensions and fixed navigation thresholds still
need to agree with the controller.

In a second terminal, with the same ROS environment and `ROS_DOMAIN_ID`:

```bash
./fp_view.sh
# Optional initial map radius:
./fp_view.sh --range 4
```

For simulation, use `./fp_view.sh --simulation`. Controller mounting telemetry
takes priority; the viewer's flag sets its fallback when telemetry is unavailable.

The viewer only subscribes; it never publishes motion commands. Quitting the
viewer does not stop the controller. Stop the controller in its own terminal.
Without `--terminal-debug`, LiDAR and commanded velocity still display, while
controller state and course pose show as waiting.

Keys: `+`/`-` zoom, `m` switch robot/course axes, `q` quit viewer.
The robot stays centred. In robot axes, forward is up and left is left. In
course axes, +x is right and +y is up; course pose telemetry is required.
Maps use an approximate 2:1 terminal character aspect ratio.

Use a terminal at least 60 columns by 24 rows; taller terminals show more map detail.
Symbols: `#` LiDAR return, `R` robot, `+` heading, `S` first valid robot position,
`W` current action, `T` selected gap target, and `M` tracked panel.
Start, action, and gap coordinates stay visible in
the text header even when their map markers are outside the displayed range.
The robot marker takes priority if several markers share a map cell.

All position readouts are course coordinates in metres. `T` also shows its
robot-relative coordinates. There is no configured destination or `G` marker;
the finish is the detected end wall. `W` shows the selected gap during approach,
the sensed waiting point during positioning, the current position while holding, and the
panel exit during crossing. In `finish`, it estimates the stop position from
the LiDAR wall; without that observation it describes the end-wall objective.

The velocity readout includes body-frame `linear.x`, `linear.y`, planar speed
`sqrt(vx² + vy²)`, and `angular.z` in rad/s and deg/s. These are commanded
values from `cmd_vel`, rather than measured robot velocity. Tracks are the
controller's retained observations, not a guarantee that a panel remains there.
Topic ages are measured since local receipt; data older than one second is
marked STALE and remains visible.

Navigation, panel tracking, and the viewer use `scan_to_body()` inside
`obstacle_course.py`. There is no separate scan geometry module. The transform uses
the scan's own angle metadata plus a 180 degree mounting yaw, since the LiDAR's
zero-angle direction points backward. This also matches the Gazebo laser joint.
This rotates the rays, not the robot's Optitrack heading or course frame.
The existing sensor position is retained at 0.11423 m forward of the robot
reference point, with zero lateral offset; facing backward does not establish
where the sensor is physically mounted. Verify that position on the hardware.

Check without driving using calibration mode and the viewer. An object in front
of the robot must appear above `R`, and an object on its left must appear left.
Controller parameters `lidar_yaw_deg`, `lidar_x_offset`, and `lidar_y_offset`
can override the mounting geometry. With debug telemetry, the viewer follows
those settings. Without it, the viewer uses the mounting setting in
`obstacle_course.py`, or the Gazebo mount with `--simulation`.

To undo the hardware correction, change this one line near the top of
`obstacle_course.py`, then restart the controller and viewer:

```python
REAL_LIDAR_YAW_DEG = 0.0  # 180.0 = backward facing; 0.0 = already robot aligned
```

Gazebo retains its model's 180 degree mount independently. To try zero yaw for
one launch without editing the source, use:

```bash
./fp.sh --calibrate --start-here --terminal-debug --ros-args -p lidar_yaw_deg:=0.0
```

For a standalone viewer without telemetry, use `./fp_view.sh --lidar-yaw-deg 0`.

This removes RViz as a dependency but still requires compatible ROS/DDS
communication between machines. If no topics arrive, check robot bringup,
ROS_DOMAIN_ID, networking, and the ROS environment in both terminals.

## Repeatable simulation checks

From the workspace with ROS and `install/setup.bash` sourced:

```bash
python3 tools/fp_sim_check.py --name my_seed_11 --seed 11
```

This starts an isolated headless Gazebo run on a ROS domain between 120 and 219
(chosen from the run name), with a unique
Gazebo partition. The generated world is saved under `log/fp_sim_checks/`,
leaving the normal world file untouched. Names must be unique. Use `--gui` to
open Gazebo as well. The controller uses simulation time; the launch file
bridges `/clock` from Gazebo to ROS. The runner passes `--simulation` explicitly.

A pass requires observing both obstacle crossings, the final `done` state,
and zero velocity commands for two seconds. This verifies course completion;
it does not certify collision-free navigation or real-hardware behavior.
Each run retains controller and Gazebo logs, trajectory JSONL, source snapshot,
source hash, terminal snapshots, and `result.json`. To stop a run for inspection,
create a file named `STOP` in its log directory. The runner closes its owned
processes and records the interruption.

The static gap controller evaluates farther openings while committing to short
segments, with a two-degree steering deadband. Automatic reverse recovery has
been removed. Navigation can move forward, rotate when the swept footprint is
clear, or stop. Negative forward commands are rejected by the safety checker
and clamped to zero by the command publisher. There is no enable-reverse flag.

After updating the robot's source files and rebuilding the package, run normally:

```bash
./fp.sh --terminal-debug
```

After two seconds of commanded motion without 0.015 m of position progress or
five degrees of heading progress, the controller stops until progress is seen.
An interval already blocked by collision checks or deliberately paused for a
panel does not count against that motion timer. Gap selection and footprint
collision checks remain active. Diagnostic metadata includes the motion status, count and location of returns
inside the safety footprint, requested/safety-filtered commands, and wall-clock
ages since the latest pose and scan callbacks. Receipt ages alone cannot detect
a tracking system repeatedly publishing the same frozen coordinates.

Clearance is rechecked each tick. The previous reverse-escape exception for an
overlapping safety box has also been removed.
Free directions use a 0.15 m chassis radius centred 0.07 m forward of the robot
reference. A usable opening needs at least three consecutive one-degree samples.
Planning horizons are tried in order: 1.20, 0.80, 0.60, 0.35, 0.20, 0.12, then
0.06 m. A small path history prefers clear openings leading away from traversed
space, preventing repeated trips around the same corridor. If every opening is
visited, the original clearance-based choice remains available for backtracking.
The planner considers openings within 10% of the widest opening's angular sample
count and chooses the one requiring the least turning, reducing scan-driven
switching. It commits to at most 0.20 m along that opening; if only a shorter
planning horizon is clear, the commitment is shortened to match.
The committed segment stays fixed in the pose frame. Steering follows a point
0.20 m ahead along that line, avoiding endpoint chasing and the resulting yaw
oscillation. Another gap is selected after projected progress reaches within
0.02 m of the segment's end.
A safety or stall stop, missing scan, or panel takeover cancels the commitment.
Safety and steering still update every 0.05 s. LiDAR determines whether openings
are clear. Pose retains the selected segment and records traversed space; there
is no fixed destination or course waypoint guiding static gap selection.
History samples are retained every 0.10 m, excluding the most recent 0.60 m from
the revisit preference. A candidate's probe, up to 0.80 m ahead, is considered
revisited within 0.30 m of those older samples. This changes preference, not the
footprint clearance or collision checks.
The navigation, steering, safety, and stall-stop implementation is in
`gap_navigation.py`; `obstacle_course.py` supplies ROS data and course orchestration.
Panel identification uses the known first-panel x band (3.1 to 4.3 m) and the second
panel's 0.6 m spacing, with 0.15 m tolerance. This rejects apparent motion of
occluded end-wall fragments. After passing the gate, staging uses the lane the
robot actually traversed. The sensed panel sets a waiting x threshold; it does
not bias gap selection toward a waypoint.

To plot saved runs using the environment's existing Matplotlib installation:

```bash
python3 tools/fp_sim_report.py my_seed_11 my_seed_22 my_seed_33
```

The controller uses one LiDAR subscription for both navigation and panel
tracking. `panel_command()` returns a velocity pair, or `None` to hand over to
`gap_command()`. The timer publishes the selected command in one place.
`safe_command()` checks its swept footprint; `stop_if_stalled()` stops motion
when there is no progress. The panel sequence is `scout -> position -> observe
-> cross`, repeated twice, then `finish -> done`. Generic gap navigation takes
the robot through the gate to its waiting point, without a separate gate state.

There are no `gap_goal_x` or `gap_goal_y` parameters. The panel controller uses
observed panel positions for staging and the observed end wall for finishing.
Its course-coordinate bands, panel spacing, and two-panel sequence remain
course-specific assumptions. `steer()` uses heading
error, yaw-rate damping filtered with a 0.12 s time constant, and a 1.5 rad/s²
limit on changes in requested angular velocity. Collision checks can still
reduce a command immediately. `learn_cycle()` compares shifted histories
at 20 Hz; it does not assume constant panel speed. All functions have short
docstrings, and the LiDAR reversal switch remains at the top of
`obstacle_course.py`.
