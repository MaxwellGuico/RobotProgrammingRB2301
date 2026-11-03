# Read the navigation code

Start with `rb2301_fp/obstacle_course.py`. Display code is isolated in
`rb2301_fp/diagnostics.py` and does not affect navigation decisions.

The control loop is:

1. `remember_scan()` and `remember_pose()` retain timestamped sensor data.
2. `prepare_scan()` checks freshness and projects the scan using the measured
   LiDAR mount and a matching robot pose.
3. `update_memory()` ray-traces observed free cells and occupied endpoints.
4. `planning_space()` accepts cells that fit half the robot width.
5. `select_local_goal()` groups frontiers and chooses an observation point;
   `Astar()` finds a route through known free cells.
6. `follow_carrot()` follows a point along that route, using forward speed and yaw. It always publishes `Vy=0`.
7. `navigation_step()` checks the next forward line using the robot width. A
   blocked line stops forward motion; missing sensors command zero velocity.

The controller also uses `observe_motion()` and `moving_obstacle_command()` to observe
moving obstacles before crossing, and `completion_command()` to confirm the
closed end after crossing two observed bands. An exhausted frontier does not
count as course completion.

The passage model is deliberately simple: occupied cells are padded by the robot
half-width. Goals use the same rule as route cells. There is no separate turning
radius or turn-room goal adjustment. See the package README for run commands,
Foxy RViz details, and the actual simulation validation status.
