import argparse
import configparser
import math
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  


_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course

def visit_cell(point, size=0.25):
    """World-fixed cells keep their meaning when the robot turns around."""
    return tuple(np.floor(np.asarray(point) / size).astype(int))


def path_visit_cost(start, target, visits):
    """Penalize repeated traversal, including the path before its endpoint."""
    distance = np.linalg.norm(target - start)
    if distance < 1e-6:
        return 0.0
    # Ignore the first 20 cm: every outgoing path shares the robot's current cell.
    fractions = np.linspace(min(0.2 / distance, 1.0), 1.0, 8)
    cells = {visit_cell(start + t * (target - start)) for t in fractions}
    return float(np.mean([min(visits.get(cell, 0), 5) for cell in cells]))


class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360: 
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough 

        self.sub_scan = self.create_subscription(
            LaserScan, "scan", self.sub_scan_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription( 
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None
        # Max's variables
        '''Lidar Processing'''
        self.lidar_min_distance = 0.0
        self.lidar_max_distance = 5.0
        self.angle_min = 0.0
        self.angle_increment = 0.0
        # Robot body is ~22 cm wide (half-width 0.11 m). With the margin below,
        # 0.14 needs a ~34 cm gap; the old 0.17 needed over 40 cm.
        self.robot_safe_radius = float(self.declare_parameter(
            'robot_safe_radius', 0.14).value)
        self.clearance_margin = float(self.declare_parameter(
            'clearance_margin', 0.03).value)
        # Hysteresis: a path needs the full margin to be chosen, but once
        # committed it is only dropped if something gets within this margin.
        # Without it, 1-2 cm of drift in a tight gap flips "clear" to "blocked".
        self.commit_margin = float(self.declare_parameter(
            'commit_margin', 0.01).value)
        self.min_gap_width = 2 * self.robot_safe_radius
        self.valid_gaps = []
        self.points = None
        '''PID Variables'''
        # Local goals change often: avoid distance derivative kicks and use
        # gentler angular damping so a new goal does not cause a sharp spin.
        self.Kp,self.Kd,self.Kp_theta,self.Kd_theta = 3.0, 0.0, 2.0, 0.15
        self.error_distance_prev = None
        self.error_heading_prev = None
        self.PID_prev_time = self.get_clock().now()
        '''Callback Variables'''
        self.committed_target = None
        # The simulated scanner faces backwards and sits ahead of base_link.
        # Parameters let hardware use its own measured mounting transform.
        self.lidar_x_offset = float(self.declare_parameter(
            'lidar_x_offset', 0.11423 if is_simulation else 0.0).value)
        self.lidar_yaw = math.radians(float(self.declare_parameter(
            'lidar_yaw_deg', 180.0).value))
        self.navigation_speed = float(self.declare_parameter(
            'navigation_speed', 0.35 if is_simulation else 0.2).value)
        # Penalty for paths pointing back against the current corridor direction.
        self.reverse_penalty_weight = float(self.declare_parameter(
            'reverse_penalty_weight', 1.5).value)
        self.progress_bias_weight = float(self.declare_parameter(
            'progress_bias_weight', 2.0).value)
        if not math.isfinite(self.progress_bias_weight):
            raise ValueError('progress_bias_weight must be finite')
        '''Corridor state machine'''
        # The course is always an L-shape plus the gate leg; only lengths vary.
        # Each corridor fixes which maze direction counts as "forward", and which
        # side the next corridor opens on. The gap planner still does the avoiding.
        # corridor_heading_deg rotates the whole layout if a real-robot frame differs.
        base = math.radians(float(self.declare_parameter('corridor_heading_deg', 0.0).value))
        def heading(deg):
            return np.array([math.cos(base + math.radians(deg)),
                             math.sin(base + math.radians(deg))])
        # side_depth: how far to the side must be empty to count as an opening.
        # Corridor 2 widens partway (its left wall steps out ~0.6 m) before the
        # gate, so it needs a deeper check; the gate opens onto a wide area.
        self.side_depth = float(self.declare_parameter('opening_side_depth', 1.1).value)
        gate_depth = float(self.declare_parameter('gate_side_depth', 1.8).value)
        self.corridors = [
            dict(name='CORRIDOR_1', direction=heading(0), turn='right',
                 side_depth=self.side_depth),
            dict(name='CORRIDOR_2', direction=heading(-90), turn='left',
                 side_depth=gate_depth),
            dict(name='CORRIDOR_3_GATE', direction=heading(0), turn=None)]
        start = int(self.declare_parameter('start_corridor', 1).value)
        if not 1 <= start <= len(self.corridors):
            raise ValueError(f'start_corridor must be 1..{len(self.corridors)}')
        self.opening_min_width = float(self.declare_parameter('opening_min_width', 0.4).value)
        # Distance into a corridor before its turn may trigger; the shortest
        # leg is well over 1 m, and this ignores the corner we just turned at.
        self.min_corridor_travel = float(self.declare_parameter(
            'min_corridor_travel', 0.8).value)
        self.state_index = None
        self.enter_corridor(start - 1)
        self.last_scan_time = None
        self.scan_pose = None
        self.blocked_since = None
        self.progress_distance = None
        self.progress_time = time.monotonic()
        self.rejected_targets = []
        '''Distance based Recording of visted history'''
        self.visited_positions = []
        self.history_spacing = 0.2
        self.visit_radius = 0.5 # metres
        self.visited_cells = {}
        self.last_visit_cell = None
    def sub_scan_callback(self, msg):
        """Build short collision-free paths instead of treating depth jumps as doors."""
        self.last_scan_time = time.monotonic()
        # Keep scan geometry tied to the pose at reception. A later timer may
        # run after the robot has turned; those points are not in its new frame.
        self.scan_pose = self.pose.copy() if self.pose is not None else None
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.last_scan = np.asarray(msg.ranges, dtype=float)
        self.angles = msg.angle_min + np.arange(len(self.last_scan)) * msg.angle_increment
        self.valid_gaps = []
        self.points = np.empty((0, 2))
        if not len(self.last_scan) or msg.angle_increment <= 0:
            return

        # +inf means no return within range; NaN/too-short readings are unknown.
        # Keep unknown readings blocked rather than opening an imaginary passage.
        max_range = min(self.lidar_max_distance, msg.range_max)
        known = ((np.isfinite(self.last_scan) &
                  (self.last_scan >= max(msg.range_min, self.lidar_min_distance))) |
                 np.isposinf(self.last_scan))
        ranges = np.where(known, np.minimum(self.last_scan, max_range), 0.0)
        self.scan_ranges = ranges
        angles_robot = self.angles + self.lidar_yaw
        # Apply BOTH rotation and translation before checking robot clearance.
        hits = np.isfinite(self.last_scan) & known & (self.last_scan <= max_range)
        self.points = np.column_stack((
            self.lidar_x_offset + self.last_scan[hits] * np.cos(angles_robot[hits]),
            self.last_scan[hits] * np.sin(angles_robot[hits])))
        # Inflate each return by robot radius + margin. Sampling directions over
        # the full circle also handles openings across the scan's -pi/pi seam.
        radius = self.robot_safe_radius + self.clearance_margin
        for heading in np.deg2rad(np.arange(-180, 180, 5)):
            direction = np.array([np.cos(heading), np.sin(heading)])
            reach = 1.0
            if len(self.points):
                along = self.points @ direction
                perpendicular_sq = np.maximum(0.0,
                    np.sum(self.points ** 2, axis=1) - along ** 2)
                intersects = (perpendicular_sq < radius ** 2) & (along > 0)
                if np.any(intersects):
                    entry = along[intersects] - np.sqrt(
                        radius ** 2 - perpendicular_sq[intersects])
                    reach = min(reach, float(np.min(entry)) - 0.02)
            # A useful local target must advance at least 25 cm. It need not
            # cross a doorway or travel 50 cm beyond an occluding obstacle.
            if reach < 0.25:
                continue
            target = reach * direction
            sensor_target = target - np.array([self.lidar_x_offset, 0.0])
            sensor_angle = math.atan2(sensor_target[1], sensor_target[0]) - self.lidar_yaw
            sensor_angle = math.atan2(math.sin(sensor_angle), math.cos(sensor_angle))
            index = int(round((sensor_angle - msg.angle_min) / msg.angle_increment))
            # Check neighboring rays too: one long beam does not prove width.
            neighborhood = np.arange(index - 1, index + 2)
            full_circle = len(ranges) * msg.angle_increment >= 2 * np.pi - 0.05
            if full_circle:
                neighborhood %= len(ranges)
            elif np.any((neighborhood < 0) | (neighborhood >= len(ranges))):
                continue
            if np.min(ranges[neighborhood]) < np.linalg.norm(sensor_target) + radius:
                continue
            self.valid_gaps.append({
                'target': target, 'free_depth': reach,
                'gap_width': 2 * radius, 'heading': heading})

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose 
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)
    def current_xy_error(self,goal):
        '''Calculate Current Error''' 
        robot_point = (self.pose[0],self.pose[1])
        '''Calculate Current Error'''            
        e_x_current = goal[0]-robot_point[0]
        e_y_current = goal[1]-robot_point[1]
        return e_x_current, e_y_current
    def PID(self, goal:tuple, Kp, Kd, Kp_theta, Kd_theta):
        e_x_current , e_y_current = self.current_xy_error(goal)
        error_distance = math.sqrt(e_x_current*e_x_current + e_y_current*e_y_current)
        desired_heading = np.arctan2(e_y_current,e_x_current)
        current_heading = np.deg2rad(self.pose[2])
        error_heading =   desired_heading - current_heading
        error_heading = np.arctan2(
            np.sin(error_heading),
            np.cos(error_heading),

        )
        '''Clock for dt'''  
        current_time = self.get_clock().now() # use only one, this works in nanoseconds so if i call again it will be a different timestamp
        dt = current_time - self.PID_prev_time
        if self.error_distance_prev is None and self.error_heading_prev is None:
            self.error_distance_prev = error_distance
            self.error_heading_prev = error_heading
        Vx = Kp*(error_distance) + Kd * ((error_distance-self.error_distance_prev)/((dt.nanoseconds/1e9)))\
            if (dt.nanoseconds/1e9) > 1e-9 else Kp*(error_distance)
        Vx = np.clip(Vx, 0.0, self.max_translate_velocity)
        Vx *= max(0.0, 0.5 * np.cos(error_heading)) # Slows down during turning
        Vy = 0
        # Wrap the derivative too: crossing +/-pi is not a 2*pi heading jump.
        heading_delta = math.atan2(math.sin(error_heading-self.error_heading_prev),
                                   math.cos(error_heading-self.error_heading_prev))
        w =  Kp_theta * error_heading + Kd_theta* (heading_delta/((dt.nanoseconds/1e9))) \
                   if (dt.nanoseconds/1e9) > 1e-9 else Kp_theta *(error_heading)
        
        self.get_logger().debug(f"Error Distance: {error_distance} Turning Correction Applied: {max(0.0, np.cos(error_heading))} Vx: {Vx}")
        self.get_logger().debug(f"Desired Heading: {desired_heading} Current Heading: {current_heading} Error Heading: {error_heading} Angular z: {w}")
        '''Save prev errors and prev PID time'''

        self.error_distance_prev = error_distance
        self.error_heading_prev = error_heading
        self.PID_prev_time = current_time

        return Vx, Vy,w 
    def enter_corridor(self, index):
        """Switch corridor: new forward direction, fresh opening detection."""
        self.state_index = index
        corridor = self.corridors[index]
        self.progress_direction = corridor['direction']
        self.side_wall_seen = False
        self.transition_since = None
        self.state_start = None if self.pose is None else self.pose[:2].copy()
        self.get_logger().info(
            f"State -> {corridor['name']}: forward={corridor['direction'].round(2)}, "
            f"next opening on the {corridor['turn'] or 'n/a'}")

    def corridor_observation(self, world_points):
        """Return (side_wall_beside, side_opening, end_wall_ahead) for this corridor."""
        corridor = self.corridors[self.state_index]
        forward = corridor['direction']
        left = np.array([-forward[1], forward[0]])
        side = -left if corridor['turn'] == 'right' else left
        rel = world_points - self.pose[:2]
        along, lateral, across = rel @ forward, rel @ side, rel @ left
        strip = (lateral > 0) & (lateral < corridor['side_depth'])
        # A side wall runs alongside us: it must fill most slices over +/-0.3 m.
        # A wall crossing our path (e.g. the previous corridor's end) fills one.
        wall_beside = sum(bool(np.any(strip & (along >= a) & (along < a + 0.1)))
                          for a in np.arange(-0.3, 0.3, 0.1)) >= 4
        # Opening: a run of empty 10 cm slices beside/just ahead of the robot.
        # Slicing means a box near the wall cannot hide a gap beside it.
        run = best = 0
        for start in np.arange(-0.2, 0.6, 0.1):
            empty = not np.any(strip & (along >= start) & (along < start + 0.1))
            run = run + 1 if empty else 0
            best = max(best, run)
        opening = best * 0.1 >= self.opening_min_width - 1e-6
        # End wall: something within 0.6 m ahead in at least 9 of the 10 slices
        # across +/-0.5 m. A side wall fills 1-2 slices and a box about 6, so
        # only a wall across the corridor qualifies.
        ahead = (along > 0.05) & (along < 0.6)
        filled = sum(bool(np.any(ahead & (across >= c) & (across < c + 0.1)))
                     for c in np.arange(-0.5, 0.5, 0.1))
        end_wall = filled >= 9
        return wall_beside, opening, end_wall

    def update_corridor_state(self, world_points, now):
        """Advance to the next corridor once its side opening is confirmed."""
        if self.corridors[self.state_index]['turn'] is None or not len(world_points):
            return False
        if self.state_start is None:
            self.state_start = self.pose[:2].copy()
        wall_beside, opening, end_wall = self.corridor_observation(world_points)
        self.side_wall_seen |= wall_beside
        travelled = float((self.pose[:2] - self.state_start) @ self.progress_direction)
        # "Wall seen, then gone" is the turn cue. The end wall is a fallback for
        # an opening that was hidden (e.g. by a box) as we passed it.
        ready = travelled > self.min_corridor_travel and (
            (self.side_wall_seen and opening) or end_wall)
        if not ready:
            self.transition_since = None
            return False
        if self.transition_since is None:
            self.transition_since = now
        if now - self.transition_since < 0.3:  # must persist over a few scans
            return False
        reason = 'side opening' if opening and self.side_wall_seen else 'end wall'
        self.get_logger().info(f"Corridor turn ({reason}) at {self.pose[:2].round(2)}")
        self.enter_corridor(self.state_index + 1)
        return True

    def reset_target(self):
        """Reset derivative history whenever the local goal changes."""
        self.committed_target = None
        self.error_distance_prev = self.error_heading_prev = None
        self.PID_prev_time = self.get_clock().now()
        self.blocked_since = None
        self.progress_distance = None
        self.progress_time = time.monotonic()

    def timer_callback(self):
        """Select a local free path, follow it, and replan if it stops working."""
        if self.pose is None:
            return
        if self.calibrate:
            # Calibration must never drive, even if accidentally used in Gazebo.
            self.move_2D()
            self._ticks += 1
            if self._ticks % 10 == 0:
                self.get_logger().info(f"Calibration pose: {self.pose}")
            return
        now = time.monotonic()
        if (self.last_scan_time is None or self.scan_pose is None or
                now - self.last_scan_time > 0.5):
            self.move_2D()
            return
        if not self.visited_positions or np.linalg.norm(
                self.pose[:2] - self.visited_positions[-1]) >= self.history_spacing:
            self.visited_positions.append(self.pose[:2].copy())
            cell = visit_cell(self.pose[:2])
            # Count re-entry instead of elapsed time: pausing or rotating in
            # place must not make every possible exit increasingly expensive.
            if cell != self.last_visit_cell:
                self.visited_cells[cell] = self.visited_cells.get(cell, 0) + 1
                self.last_visit_cell = cell
        theta = np.deg2rad(self.pose[2])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        scan_theta = np.deg2rad(self.scan_pose[2])
        scan_rotation = np.array([[np.cos(scan_theta), -np.sin(scan_theta)],
                                  [np.sin(scan_theta), np.cos(scan_theta)]])
        # Re-express the cached scan in the CURRENT robot frame before checking
        # a path. Otherwise a wall rotates with the robot between scan callbacks.
        world_points = self.scan_pose[:2] + self.points @ scan_rotation.T
        points_now = (world_points - self.pose[:2]) @ rotation
        self.rejected_targets = [(p, expiry) for p, expiry in self.rejected_targets
                                 if expiry > now]
        if self.update_corridor_state(world_points, now):
            # Old target and rejections were chosen for the previous direction.
            self.rejected_targets = []
            self.reset_target()
        if self.committed_target is None:
            candidates = []
            travel = self.progress_direction
            for gap in self.valid_gaps:
                target_world = self.scan_pose[:2] + scan_rotation @ gap['target']
                target_now = rotation.T @ (target_world - self.pose[:2])
                candidate_heading = math.atan2(target_now[1], target_now[0])
                if any(np.linalg.norm(target_world - p) < 0.35
                       for p, _ in self.rejected_targets):
                    continue
                # The old nearest-point penalty saturated after one visit and
                # ignored revisits along the path. Repeated loops now cost more.
                visited = path_visit_cost(self.pose[:2], target_world, self.visited_cells)
                endpoint_visits = min(self.visited_cells.get(visit_cell(target_world), 0), 5)
                # Signed metres of progress in the fixed maze direction. A
                # side-step around an obstacle is allowed; collision checks
                # still decide which paths may become candidates at all.
                maze_progress = float((target_world - self.pose[:2]) @ self.progress_direction)
                # Discourage U-turns: only paths pointing back against the
                # corridor direction are penalised; sidesteps cost nothing.
                reverse = 0.0
                offset = target_world - self.pose[:2]
                if np.linalg.norm(offset) > 1e-6:
                    reverse = max(0.0, -float(offset @ travel) / np.linalg.norm(offset))
                score = (0.6 * gap['free_depth'] + 0.3 * np.cos(candidate_heading)
                         - self.reverse_penalty_weight * reverse
                         - 1.2 * visited - 0.5 * endpoint_visits
                         + self.progress_bias_weight * maze_progress)
                candidates.append((score, gap, target_world))
            # Score ALL directions together. Filtering to forward paths first
            # discarded fresh exits behind us, regardless of the visit penalty.
            choices = candidates
            if not choices:
                self.move_2D()
                return
            _, gap, self.committed_target = max(choices, key=lambda c: c[0])
            selected_relative = rotation.T @ (self.committed_target - self.pose[:2])
            selected_heading = math.atan2(selected_relative[1], selected_relative[0])
            self.get_logger().info(
                f"New local target: {self.committed_target.round(2)}, "
                f"turn={np.rad2deg(selected_heading):.0f} deg")

        target_robot = rotation.T @ (self.committed_target - self.pose[:2])
        distance = np.linalg.norm(target_robot)
        if distance < 0.15:
            self.reset_target()
            self.move_2D()
            return
        # Recheck the committed path on every scan, including moving obstacles.
        blocked = False
        if len(points_now):
            along = points_now @ target_robot
            t = np.clip(along / distance ** 2, 0, 1)
            clearance = np.linalg.norm(points_now - t[:, None] * target_robot, axis=1)
            # Points behind the start cannot obstruct translation away from them.
            # This matches candidate generation and lets us leave a tight corner.
            blocked = bool(np.any((along > 0) &
                (clearance < self.robot_safe_radius + self.commit_margin)))
        if self.progress_distance is None or distance < self.progress_distance - 0.03:
            self.progress_distance, self.progress_time = distance, now
        if blocked:
            self.blocked_since = now if self.blocked_since is None else self.blocked_since
            self.move_2D()
        else:
            self.blocked_since = None
        # Briefly wait for a moving obstacle; then reject this goal temporarily
        # so replanning cannot immediately select the same blocked position.
        if ((self.blocked_since is not None and now - self.blocked_since > 0.7) or
                now - self.progress_time > 5.0):
            why = ("blocked" if self.blocked_since is not None and now - self.blocked_since > 0.7
                   else "no progress for 5 s")
            self.get_logger().info(
                f"Replanning: {why} (target {self.committed_target.round(2)}, "
                f"pose {self.pose.round(2)})")
            self.rejected_targets.append((self.committed_target.copy(), now + 12.0))
            self.reset_target()
            self.move_2D()
            return
        if blocked:
            return
        # Use the actual goal distance: a fixed-length carrot hides approach
        # progress from the distance controller and prevents smooth slowing.
        vx, _, turn = self.PID(self.committed_target, self.Kp, self.Kd,
                               self.Kp_theta, self.Kd_theta)
        # Clearance was checked along a straight segment. Driving while making
        # a large turn follows an unchecked arc and can cut into an obstacle,
        # causing repeated stops and new targets around the same small area.
        # Align first, then translate along the path we actually validated.
        heading_error = math.atan2(target_robot[1], target_robot[0])
        if abs(heading_error) > math.radians(12):
            vx = 0.0
        self.move_2D(min(vx, self.navigation_speed, 0.8 * distance), 0, turn)


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        # Always publish a stop before releasing the controller's publisher.
        obstacle.move_2D()
        obstacle.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
