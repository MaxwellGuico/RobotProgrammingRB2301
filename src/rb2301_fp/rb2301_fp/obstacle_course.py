import argparse
from collections import deque
from types import SimpleNamespace
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

def to_start_frame(node, x, y, heading, source):
    '''Pose relative to where the robot started: that spot is (0, 0) and, with
    zero_start_heading, 0 deg points straight down corridor 1. That direction is
    the robot's start heading plus node.start_yaw_offset, the corridor-wall angle
    measured by the LiDAR at startup (see align_start).'''
    if node.start_pose is None:
        node.start_pose = np.array((x, y, heading))
        node.get_logger().info(
            f"Start pose ({source}) {node.start_pose.round(3)} is now (0, 0"
            f"{', 0 deg' if node.zero_start_heading else ''})")
    x0, y0, h0 = node.start_pose
    dx, dy = x - x0, y - y0
    if node.zero_start_heading:
        h0 = h0 + getattr(node, 'start_yaw_offset', 0.0)
        c, s_ = math.cos(math.radians(-h0)), math.sin(math.radians(-h0))
        dx, dy = c * dx - s_ * dy, s_ * dx + c * dy
        heading = wrap_deg(heading - h0)
    return np.array((dx, dy, heading))


def wall_angle(points, max_range=1.5):
    '''Angle (deg, in [-45, 45)) of the nearby walls relative to the frame of
    `points`, folded onto 90-degree steps, and the total wall length used (m).
    Short segments between neighbouring scan points show which way a wall runs.'''
    if points is None or len(points) < 3:
        return 0.0, 0.0
    near = np.linalg.norm(points, axis=1) < max_range
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = (lengths > 0.02) & (lengths < 0.15) & near[:-1] & near[1:]
    total = float(lengths[valid].sum())
    if total == 0.0:
        return 0.0, 0.0
    angles = np.arctan2(steps[valid, 1], steps[valid, 0])
    weights = lengths[valid]
    mean4 = math.atan2(np.sum(weights * np.sin(4 * angles)),
                       np.sum(weights * np.cos(4 * angles)))
    return math.degrees(mean4 / 4), total


def rotation(heading_deg):
    """2D rotation matrix for a heading in degrees (robot frame -> world)."""
    theta = math.radians(heading_deg)
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


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
        # Best effort matches both reliable and best-effort LiDAR publishers.
        self.sub_scan = self.create_subscription(
            LaserScan, "scan", self.sub_scan_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
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
        # Real robot: treat the pose at startup as the origin (see optitrack_callback).
        self.zero_at_start = bool(self.declare_parameter('zero_at_start', True).value)
        self.zero_start_heading = bool(self.declare_parameter('zero_start_heading', True).value)
        self.start_pose = None
        # At startup, measure the corridor walls with the LiDAR and make 0 deg
        # point straight down corridor 1, even if the robot is placed skewed.
        self.align_start = bool(self.declare_parameter('align_start', True).value)
        self.start_align_max_deg = float(self.declare_parameter(
            'start_align_max_deg', 30.0).value)
        self.start_yaw_offset = 0.0
        self.start_aligned = False
        self.align_scan_time = None
        self.frame = self.course = None
        self._ticks = 0

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.4 # Real-robot limit 0.4 m/s (TA-approved, 2026-11-10)

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
        # True if the base moves toward the robot's rear (the side the LiDAR
        # calls 180 deg) when sent +x. Turning is unaffected.
        self.invert_linear = bool(self.declare_parameter('invert_linear', False).value)
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None
        # Max's variables
        '''Lidar Processing'''
        self.lidar_min_distance = 0.0
        self.lidar_max_distance = 5.0
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
        # How far the nose (0.18 m ahead of the rear axle) sticks out past the
        # safety circle, which is centred on the rear axle.
        self.nose_overhang = max(0.0, 0.18 - self.robot_safe_radius)
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
        # Furthest a local target may be placed. Longer = fewer stop/replan cycles.
        self.lookahead_distance = float(self.declare_parameter(
            'lookahead_distance', 1.0 if is_simulation else 1.6).value)
        # Pick the next target this far before reaching the current one, without
        # stopping in between (gate waypoints still stop precisely).
        self.retarget_distance = float(self.declare_parameter(
            'retarget_distance', 0.15 if is_simulation else 0.35).value)
        # Keep targets this far short of the nearest obstacle, so a few cm of
        # pose/scan noise does not flip the path to "blocked" while driving.
        self.target_buffer = float(self.declare_parameter(
            'target_buffer', 0.0 if is_simulation else 0.1).value)
        # How long to wait on a blocked path before choosing another target.
        self.blocked_wait = float(self.declare_parameter('blocked_wait', 0.7).value)
        # Heading error above which the robot turns on the spot before driving.
        self.align_angle = math.radians(float(self.declare_parameter(
            'align_angle_deg', 12.0).value))
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
                 side_depth=gate_depth, cue='gate'),
            dict(name='CORRIDOR_3_GATE', direction=heading(0), turn=None,
                 side_depth=self.side_depth)]
        start = int(self.declare_parameter('start_corridor', 1).value)
        if not 1 <= start <= len(self.corridors):
            raise ValueError(f'start_corridor must be 1..{len(self.corridors)}')
        self.opening_min_width = float(self.declare_parameter('opening_min_width', 0.4).value)
        # Distance into a corridor before its turn may trigger; the shortest
        # leg is well over 1 m, and this ignores the corner we just turned at.
        self.min_corridor_travel = float(self.declare_parameter(
            'min_corridor_travel', 0.6).value)
        '''Gate detection (corridor 2)'''
        # Recent scans in world terms, for looking through the gate past
        # obstacles that briefly cross the view (max range over ~1 s).
        self.scan_history = deque()
        self.gate_scan_history_s = 1.0
        self.gate_min_width = float(self.declare_parameter('gate_min_width', 0.38).value)
        self.gate_max_width = float(self.declare_parameter('gate_max_width', 0.80).value)
        self.gate_standoff = float(self.declare_parameter('gate_standoff', 0.45).value)
        self.gate_exit = float(self.declare_parameter('gate_exit', 0.12).value)
        # Narrow gates: how precisely to reach the line-up point (m), the
        # body radius used while crossing (m), and the max heading error (deg)
        # while driving through.
        self.gate_arrive = float(self.declare_parameter(
            'gate_arrive', 0.08 if is_simulation else 0.03).value)
        self.gate_cross_radius = float(self.declare_parameter(
            'gate_cross_radius', 0.15 if is_simulation else 0.13).value)
        self.gate_align_deg = float(self.declare_parameter(
            'gate_align_deg', 20.0 if is_simulation else 5.0).value)
        # How far to the side (m) the gate's wall may be from the robot.
        self.gate_max_lateral = float(self.declare_parameter('gate_max_lateral', 2.2).value)
        # Points needed in a 5 cm band to count as a wall line.
        self.gate_line_points = int(self.declare_parameter('gate_line_points', 12).value)
        self.gate_debug = {}
        self.gate_candidates_history = deque(maxlen=5)  # one list per scan
        self.gate_last_detect_scan = None
        self.gate = None          # confirmed gate: centre, normal, width
        self.gate_phase = None    # None, 'approach' or 'cross'
        '''Moving obstacles (corridor 3): stop, observe, cross'''
        self.dyn_phase = None     # look / advance / observe / cross / finished
        self.cross_speed = float(self.declare_parameter(
            'cross_speed', 0.35 if is_simulation else 0.3).value)  # real limit 0.4 m/s
        self.look_time = float(self.declare_parameter('look_time', 1.5).value)
        self.wall_hug_gap = float(self.declare_parameter('wall_hug_gap', 0.08).value)
        self.strafe_speed = float(self.declare_parameter('strafe_speed', 0.15).value)
        self.advance_step = float(self.declare_parameter('advance_step', 0.4).value)
        # Extra sideways clearance kept from a moving obstacle while crossing.
        self.cross_margin = float(self.declare_parameter('cross_margin', 0.06).value)
        # Crossing decision: the obstacle must be in our path at least
        # lane_block_time (s) to count, our path clear lane_clear_time (s), and
        # (if it never reached us) its edge lane_clear_gap (m) away and moving off.
        self.lane_block_time = float(self.declare_parameter('lane_block_time', 0.2).value)
        self.lane_clear_time = float(self.declare_parameter('lane_clear_time', 0.2).value)
        self.lane_clear_gap = float(self.declare_parameter('lane_clear_gap', 0.15).value)
        # A crossing ends when the rear is this far past the lane (m).
        self.cross_clear = float(self.declare_parameter(
            'cross_clear', 0.15 if is_simulation else 0.05).value)
        # Reach the right wall by sliding sideways (needs a base that can strafe)
        # instead of turn-drive-turn. Sliding ends wall_hug_gap from the wall.
        self.use_strafe = bool(self.declare_parameter('use_strafe', False).value)
        # Turn-drive-turn: stop when the nose is this close to the wall.
        # Negative = work it out from the simulated body's turning circle.
        self.wall_nose_gap = float(self.declare_parameter('wall_nose_gap', -1.0).value)
        # Wait with the nose this far from a moving obstacle's path
        # (negative = old behaviour: middle of the free space).
        self.lane_nose_gap = float(self.declare_parameter('lane_nose_gap', -1.0).value)
        # Distance from the LiDAR centre to the front bumper (m). Used for every
        # "nose" distance; the default fits the simulated robot.
        self.BODY_FRONT = float(self.declare_parameter('body_front', 0.18).value)
        # Distance from the LiDAR centre to the rear bumper (m), for backing up.
        self.BODY_REAR = float(self.declare_parameter('body_rear', 0.034).value)
        # A lane must show motion in this many look scans and sweep this far
        # sideways (m); flicker on a static box does neither.
        self.lane_min_scans = int(self.declare_parameter('lane_min_scans', 3).value)
        self.lane_min_sweep = float(self.declare_parameter('lane_min_sweep', 0.2).value)
        # Slowest speed (m/s) for short positioning moves near a lane.
        self.min_creep_speed = float(self.declare_parameter(
            'min_creep_speed', 0.0 if is_simulation else 0.06).value)
        # Moving-obstacle lanes that must be crossed before the run may finish.
        self.lanes_required = int(self.declare_parameter('lanes_required', 2).value)
        # End of the course: nothing moving and a wall this close to the nose.
        self.end_stop_gap = float(self.declare_parameter('end_stop_gap', 0.3).value)
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
        self.visited_cells = {}
        self.last_visit_cell = None
    def sub_scan_callback(self, msg):
        """Turn each scan into robot-frame points and straight collision-free paths."""
        self.last_scan_time = time.monotonic()
        # Keep scan geometry tied to the pose at reception. A later timer may
        # run after the robot has turned; those points are not in its new frame.
        self.scan_pose = self.pose.copy() if self.pose is not None else None
        self.last_scan = np.asarray(msg.ranges, dtype=float)
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
        angles_robot = (msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
                        + self.lidar_yaw)
        full_circle = len(ranges) * msg.angle_increment >= 2 * np.pi - 0.05
        # Apply BOTH rotation and translation before checking robot clearance.
        hits = np.isfinite(self.last_scan) & known & (self.last_scan <= max_range)
        self.points = np.column_stack((
            self.lidar_x_offset + self.last_scan[hits] * np.cos(angles_robot[hits]),
            self.last_scan[hits] * np.sin(angles_robot[hits])))
        if self.scan_pose is not None:
            yaw = math.radians(self.scan_pose[2])
            sensor_xy = self.scan_pose[:2] + self.lidar_x_offset * np.array(
                [math.cos(yaw), math.sin(yaw)])
            # For the gate see-through check only: some real LiDARs report 0 or
            # NaN when a beam hits nothing (e.g. through the gate into the open
            # area). Count those as open there, not as a wall at 0 m.
            no_return = np.isnan(self.last_scan) | (self.last_scan == 0.0)
            self.scan_history.append(dict(
                time=self.last_scan_time, xy=sensor_xy, yaw=yaw + self.lidar_yaw,
                angle_min=msg.angle_min, increment=msg.angle_increment,
                ranges=np.where(no_return, max_range, ranges), full=full_circle,
                points=self.scan_pose[:2] + self.points @ rotation(self.scan_pose[2]).T))
            while (self.scan_history and
                   self.last_scan_time - self.scan_history[0]['time'] > self.gate_scan_history_s):
                self.scan_history.popleft()
        # Inflate each return by robot radius + margin. Sampling directions over
        # the full circle also handles openings across the scan's -pi/pi seam.
        # Try the full margin first. If that leaves no path (e.g. we committed
        # into a slot that only passes the smaller commit margin), retry with
        # the commit margin rather than freezing in place.
        for margin in (self.clearance_margin, self.commit_margin):
            radius = self.robot_safe_radius + margin
            for heading in np.deg2rad(np.arange(-180, 180, 5)):
                direction = np.array([np.cos(heading), np.sin(heading)])
                reach = self.lookahead_distance
                if len(self.points):
                    along = self.points @ direction
                    perpendicular_sq = np.maximum(0.0,
                        np.sum(self.points ** 2, axis=1) - along ** 2)
                    intersects = (perpendicular_sq < radius ** 2) & (along > 0)
                    if np.any(intersects):
                        entry = along[intersects] - np.sqrt(
                            radius ** 2 - perpendicular_sq[intersects])
                        # The nose sticks out past the safety circle (front
                        # 0.18 m vs radius 0.14 m): stop short by that much too.
                        reach = min(reach, float(np.min(entry)) - 0.02 - self.nose_overhang)
                # A useful local target must advance at least 25 cm. It need not
                # cross a doorway or travel 50 cm beyond an occluding obstacle.
                # Back off from the obstacle, unless that leaves too short a target.
                buffered = reach - getattr(self, 'target_buffer', 0.0)
                reach = buffered if buffered >= 0.25 else reach
                if reach < 0.25:
                    continue
                target = reach * direction
                sensor_target = target - np.array([self.lidar_x_offset, 0.0])
                sensor_angle = math.atan2(sensor_target[1], sensor_target[0]) - self.lidar_yaw
                sensor_angle = math.atan2(math.sin(sensor_angle), math.cos(sensor_angle))
                index = int(round((sensor_angle - msg.angle_min) / msg.angle_increment))
                # Check neighboring rays too: one long beam does not prove width.
                neighborhood = np.arange(index - 1, index + 2)
                if full_circle:
                    neighborhood %= len(ranges)
                elif np.any((neighborhood < 0) | (neighborhood >= len(ranges))):
                    continue
                if np.min(ranges[neighborhood]) < np.linalg.norm(sensor_target) + radius:
                    continue
                self.valid_gaps.append({
                    'target': target, 'free_depth': reach, 'heading': heading})
            if self.valid_gaps:
                break

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
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.maze_pose = np.array((x, y, heading))  # before start-zeroing
        if not self.zero_at_start:
            self.pose = np.array((x,y,heading))
            return self.pose
        # Course frame: start spot is (0, 0), corridor 1 is 0 deg.
        self.pose = to_start_frame(self, x, y, heading, 'OptiTrack maze frame')
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
        x, y = latest_pose_msg.position.x, latest_pose_msg.position.y
        if self.zero_at_start:
            # Same start frame as the real robot: start spot (0, 0), corridor 1 = 0 deg.
            self.pose = to_start_frame(self, x, y, heading, 'odom')
        else:
            self.pose = np.array((x, y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        if getattr(self, 'invert_linear', False):
            # Base drives toward the robot's rear for +x: flip forward/sideways.
            x, y = -x, -y
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

    def update_wall_direction(self, world_points):
        """Measure how far the walls near the robot are rotated from the
        assumed square layout (real walls may be off by up to ~10 deg).

        Short segments between neighbouring scan points show which way a wall
        surface runs. Angles are folded into 90-degree steps, so walls along
        AND across the corridor both count. The length-weighted mean, kept
        within +/-12 deg and smoothed, becomes self.wall_offset (radians).
        """
        if (self.state_index == len(self.corridors) - 1 and
                getattr(self, 'dyn_phase', None) is not None):
            # Moving-obstacle section: the swinging obstacles have angled faces
            # that skewed this estimate by up to 12 deg, so the robot turned
            # toward the wall it was hugging. Keep the estimate from before.
            return
        rel = world_points - self.pose[:2]
        near = np.linalg.norm(rel, axis=1) < 1.5
        steps = np.diff(world_points, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        # Same surface: neighbouring rays 2-15 cm apart, both points nearby.
        valid = (lengths > 0.02) & (lengths < 0.15) & near[:-1] & near[1:]
        if lengths[valid].sum() < 1.0:
            return
        nominal = math.atan2(*self.corridors[self.state_index]['direction'][::-1])
        angles = np.arctan2(steps[valid, 1], steps[valid, 0]) - nominal
        # x4 maps every 90 degrees onto the same direction; average there.
        weights = lengths[valid]
        mean4 = math.atan2(np.sum(weights * np.sin(4 * angles)),
                           np.sum(weights * np.cos(4 * angles)))
        offset = mean4 / 4
        if abs(offset) > math.radians(12):
            return  # not a slightly skewed square layout: keep the last estimate
        previous = getattr(self, 'wall_offset', 0.0)
        self.wall_offset = 0.8 * previous + 0.2 * offset
        self.progress_direction = self.corridor_axes()[1]

    def corridor_axes(self):
        """(corridor, forward, left, side): side = the way the next turn goes.
        Forward is the corridor's nominal direction rotated by the measured
        wall skew."""
        corridor = self.corridors[self.state_index]
        skew = getattr(self, 'wall_offset', 0.0)
        c, s_ = math.cos(skew), math.sin(skew)
        nominal = corridor['direction']
        forward = np.array([c * nominal[0] - s_ * nominal[1], s_ * nominal[0] + c * nominal[1]])
        left = np.array([-forward[1], forward[0]])
        side = -left if corridor['turn'] == 'right' else left
        return corridor, forward, left, side

    def in_gate_corridor(self):
        return self.corridors[self.state_index].get('cue') == 'gate'

    def corridor_observation(self, world_points):
        """Return (side_wall_beside, side_opening, end_wall_ahead) for this corridor."""
        corridor, forward, left, side = self.corridor_axes()
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
        # The nearest point in each slice must also line up (within 15 cm):
        # an end wall is one straight line, not a wall corner plus a box.
        ahead = (along > 0.05) & (along < 0.6)
        nearest = []
        for c in np.arange(-0.5, 0.5, 0.1):
            in_slice = ahead & (across >= c) & (across < c + 0.1)
            if np.any(in_slice):
                nearest.append(float(np.min(along[in_slice])))
        end_wall = len(nearest) >= 9 and np.ptp(nearest) < 0.15
        return wall_beside, opening, end_wall

    def update_corridor_state(self, world_points, now):
        """Advance to the next corridor once its side opening is confirmed."""
        if self.corridors[self.state_index]['turn'] is None or not len(world_points):
            return False
        if self.state_start is None:
            self.state_start = self.pose[:2].copy()
        if self.gate_phase is not None:
            return False  # gate approach/crossing drives the next transition
        wall_beside, opening, end_wall = self.corridor_observation(world_points)
        self.side_wall_seen |= wall_beside
        travelled = float((self.pose[:2] - self.state_start) @ self.progress_direction)
        if self.in_gate_corridor():
            # Corridor 2: the gate detector decides; ignore plain side openings
            # (the corridor widens before the gate). End wall stays a fallback.
            opening = False
        # "Wall seen, then gone" is the turn cue. The end wall is a fallback for
        # an opening that was hidden (e.g. by a box) as we passed it.
        ready = travelled > self.min_corridor_travel and (
            (self.side_wall_seen and opening) or end_wall)
        if now - getattr(self, 'last_corridor_log', 0.0) > 1.0:
            self.last_corridor_log = now
            corridor = self.corridors[self.state_index]
            self.get_logger().info(
                f"{corridor['name']}: travelled {travelled:.2f} m "
                f"(need > {self.min_corridor_travel:.2f}), "
                f"{corridor['turn']} wall seen={self.side_wall_seen}, "
                f"opening={opening}, end wall={end_wall}, pose={self.pose.round(2)}")
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

    def ray_range_recent(self, point):
        """Longest range seen toward a world point over the recent scans.

        Each scan is looked up from its own sensor position, so it stays valid
        while the robot moves. Returns (best_range, distance from that sensor).
        """
        best = (0.0, 1.0)
        for scan in self.scan_history:
            offset = np.asarray(point) - scan['xy']
            distance = float(np.linalg.norm(offset))
            angle = math.atan2(offset[1], offset[0]) - scan['yaw']
            angle = math.atan2(math.sin(angle), math.cos(angle))
            index = int(round((angle - scan['angle_min']) / scan['increment']))
            n = len(scan['ranges'])
            if scan['full']:
                index %= n
            elif not 0 <= index < n:
                continue
            reach = float(scan['ranges'][index]) - distance
            if reach > best[0] - best[1]:
                best = (float(scan['ranges'][index]), distance)
        return best

    def detect_gate_candidates(self, world_points):
        """Find gaps in wall lines on the gate side of corridor 2.

        The corridor direction is known, so walls on that side are lines at a
        fixed sideways offset. Look for a 0.38-0.8 m break in such a line that
        has wall on both ends and can be seen through (not a shadow).
        """
        _, forward, _, side = self.corridor_axes()
        rel = world_points - self.pose[:2]
        along, lateral = rel @ forward, rel @ side
        near = ((lateral > 0.25) & (lateral < self.gate_max_lateral) &
                (along > -1.0) & (along < 3.0))
        along, lateral = along[near], lateral[near]
        # Why no gate was found, for the 1 Hz log in update_gate_detection.
        debug = self.gate_debug = dict(points=len(lateral), lines=[], outer=None,
                                       gaps=[], rejected={})
        if len(lateral) < 10:
            return []
        # Wall lines: 5 cm lateral bins holding many points (local maxima).
        bins = np.arange(0.25, self.gate_max_lateral + 0.05, 0.05)
        counts, _ = np.histogram(lateral, bins)
        lines = []
        for i in range(len(counts)):
            window = counts[max(0, i - 1):i + 2]
            if counts[i] < self.gate_line_points or counts[i] < window.max():
                continue
            rough = bins[i] + 0.025
            on_line = np.abs(lateral - rough) < 0.06
            if np.count_nonzero(on_line) < self.gate_line_points:
                continue
            line = float(np.median(lateral[on_line]))
            positions = np.sort(along[np.abs(lateral - line) < 0.05])
            coverage = 0.05 * len(np.unique(np.floor(positions / 0.05)))
            lines.append((line, positions, coverage))
        # The gate is in the corridor's OUTER wall: the furthest long line on
        # this side. Box and bin faces sit inside the corridor, nearer than it,
        # and can otherwise pair up into a fake "gate".
        debug['lines'] = [(round(line, 2), round(coverage, 2)) for line, _, coverage in lines]
        long_lines = [line for line, _, coverage in lines if coverage >= 0.8]
        if not long_lines:
            return []
        outer = max(long_lines)
        debug['outer'] = round(outer, 2)
        rejected = debug['rejected']
        candidates = []
        for line, positions, _ in lines:
            if line < outer - 0.1:
                continue
            for a, b in zip(positions[:-1], positions[1:]):
                width = b - a
                if width >= 0.2:
                    debug['gaps'].append(round(width, 2))
                if not self.gate_min_width <= width <= self.gate_max_width:
                    if width >= 0.2:
                        rejected['width'] = rejected.get('width', 0) + 1
                    continue
                # Wall on both ends: a real gate is a break between two walls,
                # not the end of a wall (the corridor's widening step).
                if (np.count_nonzero((positions > a - 0.3) & (positions <= a)) < 3 or
                        np.count_nonzero((positions >= b) & (positions < b + 0.3)) < 3):
                    rejected['wall_ends'] = rejected.get('wall_ends', 0) + 1
                    continue
                # See-through check: rays aimed at the gap must pass well beyond
                # the wall line, else the gap is a shadow behind a nearer object.
                see_through = True
                for fraction in (0.3, 0.5, 0.7):
                    sample = (self.pose[:2] + forward * (a + fraction * width)
                              + side * line)
                    best, distance = self.ray_range_recent(sample)
                    if best < distance + 0.3:
                        see_through = False
                        break
                if not see_through:
                    rejected['see_through'] = rejected.get('see_through', 0) + 1
                    continue
                centre = self.pose[:2] + forward * (0.5 * (a + b)) + side * line
                candidates.append(dict(centre=centre, normal=side.copy(), width=width))
        return candidates

    def update_gate_detection(self, world_points):
        """Confirm a gate seen in at least 3 of the last 5 scans within 10 cm."""
        if self.gate is not None or self.last_scan_time == self.gate_last_detect_scan:
            return False
        self.gate_last_detect_scan = self.last_scan_time
        self.gate_candidates_history.append(self.detect_gate_candidates(world_points))
        now = time.monotonic()
        if now - getattr(self, 'last_gate_log', 0.0) > 1.0:
            self.last_gate_log = now
            d = self.gate_debug
            self.get_logger().info(
                f"Gate search: {d.get('points', 0)} pts on gate side, "
                f"lines(lateral m, length m)={d.get('lines')}, outer={d.get('outer')}, "
                f"gaps={d.get('gaps')}, rejected={d.get('rejected')}, "
                f"candidates now={len(self.gate_candidates_history[-1])}")
        for candidate in self.gate_candidates_history[-1]:
            matches = [candidate]
            for older in list(self.gate_candidates_history)[:-1]:
                close = [c for c in older
                         if np.linalg.norm(c['centre'] - candidate['centre']) < 0.10]
                if close:
                    matches.append(close[0])
            if len(matches) >= 3:
                self.gate = dict(
                    centre=np.mean([m['centre'] for m in matches], axis=0),
                    normal=candidate['normal'],
                    width=float(np.mean([m['width'] for m in matches])))
                self.get_logger().info(
                    f"Gate confirmed: centre={self.gate['centre'].round(2)}, "
                    f"width={self.gate['width']:.2f} m ({len(matches)}/5 scans)")
                return True
        return False

    def gate_waypoint(self):
        """Current gate target: the standoff point, or just through when crossing.

        The first moving obstacle runs only ~0.4 m past the gate's wall line,
        so the crossing stops in that pocket (rear axle 12 cm past the line,
        robot rear ~8 cm past the 1 cm wall, front ~10 cm short of the lane).
        """
        if self.gate_phase == 'approach':
            standoff = getattr(self, 'gate_standoff_used', None) or self.gate_standoff
            return self.gate['centre'] - self.gate['normal'] * standoff
        return self.gate['centre'] + self.gate['normal'] * self.gate_exit

    def reachable_standoff(self, world_points):
        '''Largest line-up distance (<= gate_standoff) whose point is clear of
        walls by the robot radius + clearance margin. A narrow corridor cannot
        fit the full standoff: the point would sit against the opposite wall,
        so the robot drove at it, found it blocked and turned away.'''
        need = self.robot_safe_radius + self.clearance_margin
        for standoff in np.arange(self.gate_standoff, 0.249, -0.025):
            point = self.gate['centre'] - self.gate['normal'] * standoff
            if not len(world_points) or np.min(
                    np.linalg.norm(world_points - point, axis=1)) >= need:
                if standoff < self.gate_standoff - 1e-6:
                    self.get_logger().info(
                        f"Gate: corridor too narrow for gate_standoff={self.gate_standoff:.2f}; "
                        f"lining up {standoff:.2f} m from the gate")
                return float(standoff)
        self.get_logger().warn("Gate: no clear line-up point found; using 0.25 m")
        return 0.25

    def path_is_clear(self, points_now, target_robot, margin, radius=None):
        """Swept-circle check of a straight segment in the current robot frame.
        Points behind the start are ignored, so the robot can back out of a
        tight corner (matches how candidate paths are generated)."""
        distance = float(np.linalg.norm(target_robot))
        if not len(points_now) or distance < 1e-6:
            return True
        # Check to where the nose ends up, not just the rear axle.
        target_robot = target_robot * (1 + self.nose_overhang / distance)
        distance += self.nose_overhang
        along = points_now @ target_robot
        t = np.clip(along / distance ** 2, 0, 1)
        clearance = np.linalg.norm(points_now - t[:, None] * target_robot, axis=1)
        radius = self.robot_safe_radius if radius is None else radius
        return not bool(np.any((along > 0) & (clearance < radius + margin)))

    def gate_cross_check(self):
        '''(margin, radius) for checking the gate crossing: the real body
        half-width plus gate_cross_margin, not the larger safety circle.'''
        return 0.0, getattr(self, 'gate_cross_radius', self.robot_safe_radius)

    def reset_target(self):
        """Reset derivative history whenever the local goal changes."""
        self.committed_target = None
        self.error_distance_prev = self.error_heading_prev = None
        self.PID_prev_time = self.get_clock().now()
        self.blocked_since = None
        self.progress_distance = None
        self.progress_time = time.monotonic()

    def timer_callback(self):
        """Main loop (20 Hz). Each behaviour is one line, so any of them can be
        commented out to test the others on their own (e.g. on the real robot).
        """
        if self.pose is None:
            now = time.monotonic()
            if now - getattr(self, 'last_no_pose_log', 0.0) > 2.0:
                self.last_no_pose_log = now
                source = ('odom' if self.is_simulation else
                          f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose')
                self.get_logger().warn(f"Waiting for first pose on {source}")
            return
        if self.calibrate:
            # Calibration must never drive, even if accidentally used in Gazebo.
            self.move_2D()
            self._ticks += 1
            if self._ticks % 10 == 0:
                # pose = what the controller uses (0,0,0 where the robot was at
                # startup); maze = OptiTrack maze frame, for reference.
                self.get_logger().info(
                    f"Calibration pose: {self.pose} (maze frame: "
                    f"{getattr(self, 'maze_pose', self.pose)})")
                # LiDAR check: the nearest object's WORLD position must stay put
                # while the robot is turned by hand. If it moves, lidar_yaw_deg
                # is wrong or the scan is mirrored.
                if (self.points is not None and len(self.points) and
                        self.scan_pose is not None):
                    i = int(np.argmin(np.linalg.norm(self.points, axis=1)))
                    p = self.points[i]
                    world = self.scan_pose[:2] + rotation(self.scan_pose[2]) @ p
                    self.get_logger().info(
                        f"Nearest object: world=({world[0]:.2f}, {world[1]:.2f}) | "
                        f"robot frame: {np.linalg.norm(p):.2f} m at "
                        f"{math.degrees(math.atan2(p[1], p[0])):.0f} deg "
                        f"(0 = straight ahead, +90 = left)")
            return
        now = time.monotonic()
        if (self.last_scan_time is None or self.scan_pose is None or
                now - self.last_scan_time > 0.5):
            # Stop without LiDAR (or without a pose at scan time). Say so: a
            # silent stop is hard to tell apart from "waiting" on the robot.
            self.move_2D()
            if now - getattr(self, 'last_stale_log', 0.0) > 2.0:
                self.last_stale_log = now
                age = 'never' if self.last_scan_time is None else f'{now - self.last_scan_time:.1f} s ago'
                self.get_logger().warn(f"Stopped: no fresh LiDAR scan (last {age})")
            return
        if not self.start_aligned:
            self.align_start_frame(now)
            return
        if self.last_scan_time == self.align_scan_time:
            self.move_2D()  # wait for a scan taken in the aligned frame
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
        pose_rotation, scan_rotation = rotation(self.pose[2]), rotation(self.scan_pose[2])
        # Re-express the cached scan in the CURRENT robot frame before checking
        # a path. Otherwise a wall rotates with the robot between scan callbacks.
        world_points = self.scan_pose[:2] + self.points @ scan_rotation.T
        points_now = (world_points - self.pose[:2]) @ pose_rotation
        frame = SimpleNamespace(now=now, rotation=pose_rotation, scan_rotation=scan_rotation,
                                world_points=world_points, points_now=points_now)

        self.update_wall_direction(world_points)  # follow walls up to ~10 deg off square
        self.step_corridor_state(frame)    # Corridor turns (state machine)
        self.step_gate_detection(frame)    # 2. Find + confirm the gate (logs only)
        self.step_gate_approach(frame)     # 2b. Drive to and through the confirmed gate
        if self.step_dynamic_obstacles(frame): return  # 3. Stop, observe, cross
        self.step_drive(frame)             # 1. Obstacle-avoiding local controller

    def align_start_frame(self, now):
        '''Once, before driving: set 0 deg = straight down corridor 1, using the
        walls the LiDAR sees (the robot is standing still at the start).'''
        self.move_2D()
        if not (self.align_start and self.zero_at_start and self.zero_start_heading):
            self.start_aligned = True
            return
        if not hasattr(self, 'align_since'):
            self.align_since = now
        angle, length = wall_angle(self.points)
        if length < 1.0:
            if now - self.align_since > 2.0:
                self.start_aligned = True
                self.get_logger().warn(
                    f"Start alignment skipped: only {length:.2f} m of wall seen; "
                    "corridor 1 = the way the robot faces")
            return
        if abs(angle) > self.start_align_max_deg:
            self.start_aligned = True
            self.get_logger().warn(
                f"Start alignment skipped: walls {angle:.1f} deg off the robot heading "
                f"(more than start_align_max_deg={self.start_align_max_deg:.0f}); "
                "corridor 1 = the way the robot faces")
            return
        # Walls run at `angle` from the robot's nose, so corridor 1 is there.
        self.start_yaw_offset = angle
        self.start_aligned = True
        self.align_scan_time = self.last_scan_time  # older scans use the old frame
        self.scan_history.clear()
        self.get_logger().info(
            f"Start alignment: corridor 1 is {angle:+.1f} deg from the robot heading "
            f"({length:.1f} m of wall seen); robot now reads {-angle:+.1f} deg")

    def step_corridor_state(self, frame):
        """Switch corridor when its turn cue is confirmed."""
        if self.update_corridor_state(frame.world_points, frame.now):
            # Old target and rejections were chosen for the previous direction.
            self.rejected_targets = []
            self.reset_target()

    def step_gate_detection(self, frame):
        """In corridor 2, look for the gate each scan; confirm over several scans."""
        if self.in_gate_corridor() and self.gate_phase is None:
            self.update_gate_detection(frame.world_points)

    def step_gate_approach(self, frame):
        """Once a gate is confirmed, steer the controller to it and through it."""
        if self.gate is not None and self.gate_phase is None and self.in_gate_corridor():
            self.gate_standoff_used = self.reachable_standoff(frame.world_points)
            self.gate_phase = 'approach'
            self.rejected_targets = []
            self.reset_target()

    # ---------------- 3. Moving obstacles: stop, observe, cross ----------------
    # Robot body relative to base_link (rear axle), from the URDF meshes.
    BODY_REAR, BODY_FRONT, BODY_HALF_WIDTH = 0.034, 0.18, 0.11
    # Turning on the spot (about the rear axle in Gazebo) swings the front
    # corners out to this radius.
    TURN_RADIUS = math.hypot(0.18, 0.11)

    def dynamic_points(self, since, lag=0.4):
        """World points in the newest scan that were not there ~lag s earlier.
        Only scans taken after `since` (when the robot stopped) are compared:
        a moving robot re-samples walls, which would look like motion."""
        if len(self.scan_history) < 2:
            return np.empty((0, 2))
        newest = self.scan_history[-1]
        older = [h for h in self.scan_history
                 if since <= h['time'] <= newest['time'] - lag]
        if not older or not len(newest['points']) or not len(older[-1]['points']):
            return np.empty((0, 2))
        current, reference = newest['points'], older[-1]['points']
        nearest = np.min(np.linalg.norm(
            current[:, None, :] - reference[None, :, :], axis=2), axis=1)
        return current[nearest > 0.07]

    def lane_ahead(self, dyn, scan_ids=None):
        """Nearest moving obstacle ahead as dict(s=progress of its centre line,
        thick=half its thickness along the corridor), or None.

        scan_ids gives the look-scan each point came from. A real moving
        obstacle shows up in several scans AND sweeps sideways across the
        corridor; LiDAR flicker on a static box edge (or a box being
        uncovered) stays in one spot, so it is not taken for a lane."""
        _, forward, left, _ = self.corridor_axes()
        if scan_ids is None:
            scan_ids = np.zeros(len(dyn), dtype=int)
            min_scans = 1
        else:
            scan_ids = np.asarray(scan_ids)
            min_scans = getattr(self, 'lane_min_scans', 3)
        min_sweep = getattr(self, 'lane_min_sweep', 0.2)
        rel = dyn - self.pose[:2]
        along, lateral = rel @ forward, rel @ left
        ahead = (along > self.BODY_FRONT) & (along < 1.6) & (np.abs(lateral) < 2.5)
        along, lateral, points, ids = along[ahead], lateral[ahead], dyn[ahead], scan_ids[ahead]
        # Nearest 15 cm-deep band of moving points that passes both checks:
        # stray points closer in must not hide the real lane behind them.
        for start in np.sort(along):
            band = (along >= start) & (along < start + 0.15)
            if np.count_nonzero(band) < 3:
                continue
            if len(np.unique(ids[band])) < min_scans:
                continue
            if np.ptp(lateral[band]) < min_sweep:
                continue
            s_ = points[band] @ forward
            return dict(s=float(np.median(s_)),
                        thick=float(np.clip(np.ptp(s_) / 2, 0.02, 0.2)))
        return None

    def creep_speed(self, error):
        """Speed for short positioning moves: proportional, but never below
        min_creep_speed (slower than that the real motors barely move)."""
        speed = min(0.2, max(getattr(self, 'min_creep_speed', 0.0), 1.5 * abs(error)))
        return float(math.copysign(speed, error))

    def right_clearance(self, points_now):
        """Gap between the robot's right side and the nearest point beside the body."""
        beside = ((points_now[:, 0] > -self.BODY_REAR + 0.01) &
                  (points_now[:, 0] < self.BODY_FRONT + 0.03) & (points_now[:, 1] < 0))
        if not np.any(beside):
            return float('inf')
        return float(-np.max(points_now[beside, 1])) - self.BODY_HALF_WIDTH

    def nose_gap(self, points_now):
        """Gap between the robot's nose and the nearest point straight ahead."""
        ahead = ((points_now[:, 0] > self.BODY_FRONT - 0.02) &
                 (np.abs(points_now[:, 1]) < self.BODY_HALF_WIDTH + 0.02))
        if not np.any(ahead):
            return float('inf')
        return float(np.min(points_now[ahead, 0])) - self.BODY_FRONT

    def strip_blocked(self, points_now, strip_end):
        """Anything in the robot-width strip straight ahead, up to strip_end?"""
        p = points_now
        return np.count_nonzero(
            (p[:, 0] > self.BODY_FRONT + 0.02) & (p[:, 0] < strip_end) &
            (p[:, 1] > -(self.BODY_HALF_WIDTH + 0.03)) &
            (p[:, 1] < self.BODY_HALF_WIDTH + self.cross_margin)) >= 2

    def heading_error_to(self, direction):
        """Signed angle (rad) to turn to face a world direction."""
        error = math.atan2(direction[1], direction[0]) - math.radians(self.pose[2])
        return math.atan2(math.sin(error), math.cos(error))

    def set_dyn_phase(self, phase, now, message=None):
        self.dyn_phase, self.dyn_since = phase, now
        if message:
            self.get_logger().info(message)

    def step_dynamic_obstacles(self, frame):
        """Corridor 3: look for moving obstacles, line up beside the right wall,
        wait for each to leave our path, cross. Returns True while it drives.

        Phases: align -> look -> centre -> turn_right -> to_wall -> align
                -> observe -> cross (repeat per lane), or look -> advance /
                finished. A pocket too shallow to turn in uses 'shift'
                (slide sideways) instead of turn_right + to_wall.
        """
        if self.state_index != len(self.corridors) - 1 or self.gate_phase is not None:
            return False
        now, points_now = frame.now, frame.points_now
        _, forward, _, _ = self.corridor_axes()
        progress = float(self.pose[:2] @ forward)
        turn = self.Kp_theta * self.heading_error_to(forward)  # hold corridor heading
        if self.dyn_phase is None:
            # Just behind us: the gate wall (1 cm thick), else our own rear.
            self.behind_s = (float(self.gate['centre'] @ forward) + 0.01
                             if self.gate is not None else progress - self.BODY_REAR)
            self.lane = None
            self.lanes_crossed = 0
            self.at_wall = False  # set once we have reached the right wall
            self.set_dyn_phase('align', now, "Moving obstacles: stop, observe, cross")
        phase = self.dyn_phase

        if phase == 'finished':
            self.move_2D()

        elif phase == 'align':
            # Face down the corridor: looking needs the robot still and square.
            if abs(self.heading_error_to(forward)) > math.radians(5):
                self.move_2D(0, 0, turn)
            elif self.lane is not None:
                self.lane['was_blocked'], self.lane['clear_since'] = False, None
                self.set_dyn_phase('observe', now)
            else:
                self.look_dyn, self.look_scan = [], None
                self.set_dyn_phase('look', now)

        elif phase == 'look':
            self.move_2D()
            # Collect motion from every scan pair during the look: one pair
            # can easily miss an obstacle.
            if self.last_scan_time != self.look_scan:
                self.look_scan = self.last_scan_time
                self.look_dyn.append(self.dynamic_points(since=self.dyn_since))
            if now - self.dyn_since < self.look_time:
                return True
            ids = np.concatenate([np.full(len(d), i) for i, d in enumerate(self.look_dyn)])
            lane = self.lane_ahead(np.vstack(self.look_dyn), ids.astype(int))
            if lane is not None:
                self.lane = lane
                self.set_dyn_phase('centre', now,
                                   f"Moving obstacle {lane['s'] - progress:.2f} m ahead")
            elif (self.nose_gap(points_now) < self.end_stop_gap and
                  self.lanes_crossed < getattr(self, 'lanes_required', 0)):
                # Wall ahead but not all lanes crossed: not the end. Keep
                # watching for the next moving obstacle.
                self.move_2D()
                self.set_dyn_phase('align', now,
                                   f"Wall ahead but only {self.lanes_crossed}/"
                                   f"{self.lanes_required} lanes crossed: looking again")
            elif self.nose_gap(points_now) < self.end_stop_gap:
                # Nothing moving and a wall straight in front: the end. Only
                # points in front of the body count, not the wall we hug.
                self.move_2D()
                self.set_dyn_phase('finished', now,
                                   f"Finished at {self.pose[:2].round(2)} "
                                   f"(wall {self.nose_gap(points_now):.2f} m ahead of the nose)")
            else:
                # Never step past end_stop_gap from a wall ahead, so the next
                # look ends the run instead of looping look -> blocked -> look.
                free = self.nose_gap(points_now)
                self.advance_to = progress + min(self.advance_step,
                                                 free - self.end_stop_gap + 0.01)
                self.set_dyn_phase('advance', now, "Nothing moving ahead: advancing")

        elif phase == 'advance':
            # Straight down the corridor, no turning (turning swings the front
            # corners out). Stop to look again after a step or if blocked.
            remaining = self.advance_to - progress
            step = frame.rotation.T @ (forward * max(remaining, 0.05))
            # Straight-line check of the body-wide strip ahead only: a swept
            # circle also hit the side wall we hug and stopped us for nothing.
            if (remaining < 0.02 or now - self.dyn_since > 6.0 or
                    self.strip_blocked(points_now,
                                       self.BODY_FRONT + max(remaining, 0.05) + 0.05)):
                self.move_2D()
                self.set_dyn_phase('align', now)
            else:
                self.move_2D(min(self.navigation_speed, 1.5 * remaining + 0.05), 0, turn)

        elif phase == 'centre':
            # Get to the wall like a car: turn right, drive, turn back. That
            # needs the turning circle to fit in the pocket between what is
            # behind us and the lane ahead; otherwise slide sideways.
            face = self.lane['s'] - self.lane['thick']
            if self.at_wall:
                # Already beside the wall (we cross straight along it): no need
                # to turn to it again. Only back up if the nose is too close to
                # this lane's sweep, then wait and cross.
                rear_gap = progress - self.BODY_REAR - self.behind_s
                gap = getattr(self, 'lane_nose_gap', -1.0)
                if gap >= 0:
                    # Move along the wall until the nose is `gap` from the lane.
                    target = max(face - gap - self.BODY_FRONT,
                                 self.behind_s + self.BODY_REAR + 0.02)
                    error = target - progress
                    if abs(error) > 0.015 and now - self.dyn_since < 8.0:
                        self.move_2D(self.creep_speed(error), 0, turn)
                    else:
                        self.set_dyn_phase('align', now,
                                           f"Waiting {face - progress - self.BODY_FRONT:.2f} m from the lane")
                    return True
                if face - (progress + self.BODY_FRONT) >= 0.08 or rear_gap < 0.05:
                    self.set_dyn_phase('align', now)
                else:
                    self.move_2D(-0.1, 0, turn)
                return True
            depth = face - self.behind_s
            self.car_style = (not getattr(self, 'use_strafe', False) and
                              depth >= 2 * self.TURN_RADIUS + 0.06)
            gap = getattr(self, 'lane_nose_gap', -1.0)
            if self.car_style:  # rear axle (turning point) mid-pocket
                target = 0.5 * (self.behind_s + face)
            elif gap >= 0:      # nose `gap` from the lane, then slide
                target = max(face - gap - self.BODY_FRONT,
                             self.behind_s + self.BODY_REAR + 0.02)
            else:               # body mid-pocket, for the sideways slide
                target = 0.5 * (self.behind_s + face) - 0.5 * (self.BODY_FRONT - self.BODY_REAR)
            error = target - progress
            if abs(error) > 0.02 and now - self.dyn_since < 8.0:
                self.move_2D(self.creep_speed(error), 0, turn)
            elif self.car_style:
                self.set_dyn_phase('turn_right', now)
            else:
                self.set_dyn_phase('shift', now,
                                   "Sliding sideways to the right wall (use_strafe)"
                                   if getattr(self, 'use_strafe', False) else
                                   f"Pocket {depth:.2f} m too shallow to turn: sliding sideways")

        elif phase == 'turn_right':
            right = np.array([forward[1], -forward[0]])
            error = self.heading_error_to(right)
            if abs(error) > math.radians(4):
                self.move_2D(0, 0, self.Kp_theta * error)
            else:
                self.set_dyn_phase('to_wall', now)

        elif phase == 'to_wall':
            # Drive at the wall until the rear axle is just far enough from it
            # to turn back (nose ~5 cm from the wall).
            right = np.array([forward[1], -forward[0]])
            gap = self.nose_gap(points_now)
            stop_gap = (self.wall_nose_gap if getattr(self, 'wall_nose_gap', -1.0) >= 0
                        else self.TURN_RADIUS + 0.03 - self.BODY_FRONT)
            if gap > stop_gap and now - self.dyn_since < 12.0:
                self.move_2D(min(self.navigation_speed, 1.5 * (gap - stop_gap) + 0.03), 0,
                             self.Kp_theta * self.heading_error_to(right))
            elif gap <= stop_gap + 0.05:
                self.move_2D()
                self.at_wall = True
                self.set_dyn_phase('align', now,
                                   f"At the right wall (nose {gap:.2f} m away); turning back")
            else:
                self.move_2D()
                self.lane = None
                self.set_dyn_phase('align', now,
                                   f"Could not reach the right wall ({gap:.2f} m); retrying")

        elif phase == 'shift':
            # Hug the right wall: obstacles turn round there, so right after one
            # leaves our path it has its whole trip out and back to go.
            # Slide sideways (mecanum) so we never move toward the lane.
            clearance = self.right_clearance(points_now)
            if clearance > self.wall_hug_gap + 0.02 and now - self.dyn_since < 15.0:
                speed = min(self.strafe_speed, 1.5 * (clearance - self.wall_hug_gap) + 0.03)
                self.move_2D(0, -speed, turn)
            elif clearance < 0.25:
                self.move_2D()
                self.at_wall = True
                self.set_dyn_phase('align', now, f"Hugging right wall ({clearance:.2f} m gap)")
            else:
                # Without the wall the "just left our path" rule is not safe:
                # stop and try the look + slide again rather than cross.
                self.move_2D()
                self.lane = None
                self.set_dyn_phase('align', now,
                                   f"Could not reach the right wall ({clearance:.2f} m); retrying")

        elif phase == 'observe':
            # Against the wall an obstacle cannot pass through our strip: it
            # comes in, turns at the wall and leaves the same way. Cross only
            # when it is MOVING AWAY with a gap: "clear" alone also happens
            # while it is still coming (its edge flickers in and out).
            self.move_2D()
            lane = self.lane
            far_face = lane['s'] + lane['thick']
            face = lane['s'] - lane['thick']
            strip_end = far_face + 0.05 - progress
            left_edge = self.BODY_HALF_WIDTH + self.cross_margin
            block_min = getattr(self, 'lane_block_time', 0.2)
            clear_hold = getattr(self, 'lane_clear_time', 0.2)
            if self.last_scan_time != lane.get('scan'):
                lane['scan'] = self.last_scan_time
                p = points_now
                # Nearest edge of the obstacle to the left of our strip, in the lane band.
                band = ((p[:, 0] > face - progress - 0.05) & (p[:, 0] < strip_end) &
                        (p[:, 1] > left_edge) & (p[:, 1] < 2.0))
                edge = float(np.min(p[band, 1])) if np.any(band) else float('inf')
                lane.setdefault('edges', deque(maxlen=30)).append((now, edge))
                if self.strip_blocked(p, strip_end):
                    lane['blocked_since'] = lane.get('blocked_since') or now
                    # In our path for a while, not one flicker of its edge.
                    if now - lane['blocked_since'] >= block_min:
                        lane['was_blocked'] = True
                    lane['clear_since'] = None
                else:
                    lane['blocked_since'] = None
                    if lane['clear_since'] is None:
                        lane['clear_since'] = now
            edges = lane.get('edges', [])
            edge_now = edges[-1][1] if edges else float('inf')
            older = [e for t, e in edges if now - t >= 0.3]
            if math.isinf(edge_now):
                away = True   # swung out of the lane band (out of view)
            elif older and math.isfinite(older[-1]):
                away = edge_now - older[-1] > 0.02   # edge moving left = away from us
            else:
                away = False
            gap = edge_now - left_edge
            clear_for = now - lane['clear_since'] if lane['clear_since'] is not None else 0.0
            just_left = lane['was_blocked'] and clear_for >= clear_hold and away
            # Never reaches the wall: go when it heads away and is far enough off.
            going_away = (not lane['was_blocked'] and clear_for >= clear_hold and away and
                          math.isfinite(edge_now) and
                          gap >= getattr(self, 'lane_clear_gap', 0.15))
            never_came = (not lane['was_blocked'] and math.isinf(edge_now) and
                          clear_for > 12.0)
            if now - getattr(self, 'last_observe_log', 0.0) > 1.0:
                self.last_observe_log = now
                self.get_logger().info(
                    f"Watching lane {self.lanes_crossed + 1}: in my path={lane['clear_since'] is None}, "
                    f"seen in path={lane['was_blocked']}, obstacle edge "
                    f"{gap if math.isfinite(gap) else float('nan'):.2f} m left of my path, "
                    f"moving away={away}, nose "
                    f"{face - progress - self.BODY_FRONT:.2f} m from the lane")
            if just_left or going_away or never_came:
                # Drive until the rear is cross_clear past the obstacle's far face.
                self.cross_to = far_face + self.BODY_REAR + getattr(self, 'cross_clear', 0.15)
                why = ("obstacle left our path and is moving away" if just_left else
                       f"obstacle moving away, {gap:.2f} m clear" if going_away else
                       "nothing seen for 12 s")
                self.set_dyn_phase('cross', now, f"Crossing: {why}")

        elif phase == 'cross':
            remaining = self.cross_to - progress
            if remaining < 0.03:
                self.move_2D()
                self.behind_s, self.lane = self.lane['s'] + self.lane['thick'], None
                self.lanes_crossed += 1
                self.set_dyn_phase('align', now, f"Lane crossed ({self.lanes_crossed}/"
                                   f"{getattr(self, 'lanes_required', 0)})")
                return True
            front = progress + self.BODY_FRONT
            far_face = self.lane['s'] + self.lane['thick']
            if (front > far_face + 0.02 and
                    self.nose_gap(points_now) < max(getattr(self, 'lane_nose_gap', -1.0), 0.05)):
                # Nose is through this lane and something (the next lane's
                # obstacle, a box) is right in front: stop here rather than
                # drive into it. This counts as crossed; the next look finds it.
                self.move_2D()
                self.behind_s, self.lane = far_face, None
                self.lanes_crossed += 1
                self.set_dyn_phase('align', now,
                                   f"Lane crossed ({self.lanes_crossed}/"
                                   f"{getattr(self, 'lanes_required', 0)}), stopped early: "
                                   f"something {self.nose_gap(points_now):.2f} m ahead of the nose")
                return True
            # Abort only for something in the SAME strip the observe phase
            # watches. A longer, wider check also saw things past the lane, so
            # the robot flipped start/abort many times a second and lost its
            # window (the obstacle swung back at it).
            if (front < self.lane['s'] - self.lane['thick'] - 0.05 and
                    self.strip_blocked(points_now, far_face + 0.05 - progress)):
                # Not in the lane yet and something is in the way: back off.
                self.move_2D()
                self.lane['was_blocked'], self.lane['clear_since'] = True, None
                self.set_dyn_phase('observe', now, "Crossing aborted before the lane")
            else:
                # Once in the lane keep going: stopping there is the worst option.
                self.move_2D(min(self.cross_speed, 3.0 * remaining + 0.03), 0, turn)
        return True

    def step_drive(self, frame):
        """Select a local free path, follow it, and replan if it stops working.
        Aims along the corridor, or at the gate waypoint during a gate phase."""
        now, rotation, scan_rotation = frame.now, frame.rotation, frame.scan_rotation
        world_points, points_now = frame.world_points, frame.points_now
        self.rejected_targets = [(p, expiry) for p, expiry in self.rejected_targets
                                 if expiry > now]
        goal = self.gate_waypoint() if self.gate_phase is not None else None
        if self.committed_target is None and goal is not None:
            # Go straight to the gate waypoint whenever that line is clear;
            # otherwise fall through to the gap planner, scored toward it.
            goal_robot = rotation.T @ (goal - self.pose[:2])
            if self.gate_phase == 'cross':
                margin, radius = self.gate_cross_check()
            else:
                margin, radius = self.clearance_margin, None
            if self.path_is_clear(points_now, goal_robot, margin, radius):
                self.committed_target = goal.copy()
                self.target_length = float(np.linalg.norm(goal_robot))
                self.get_logger().info(
                    f"Gate {self.gate_phase}: heading to {goal.round(2)}")
        if self.committed_target is None:
            candidates = []
            travel = self.progress_direction
            if goal is not None:
                to_goal = goal - self.pose[:2]
                travel = to_goal / max(np.linalg.norm(to_goal), 1e-6)
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
                if goal is not None:
                    maze_progress = float(np.linalg.norm(goal - self.pose[:2]) -
                                          np.linalg.norm(goal - target_world))
                else:
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
            if not candidates:
                self.move_2D()
                if self.rejected_targets:
                    # Every free path was rejected recently; retry them now
                    # rather than standing still until the rejections expire.
                    self.rejected_targets = []
                if now - getattr(self, 'last_no_path_log', 0.0) > 2.0:
                    self.last_no_path_log = now
                    self.get_logger().info(
                        f"No free path from {self.pose[:2].round(2)}; waiting")
                return
            _, gap, self.committed_target = max(candidates, key=lambda c: c[0])
            selected_relative = rotation.T @ (self.committed_target - self.pose[:2])
            self.target_length = float(np.linalg.norm(selected_relative))
            selected_heading = math.atan2(selected_relative[1], selected_relative[0])
            self.get_logger().info(
                f"New local target: {self.committed_target.round(2)}, "
                f"turn={np.rad2deg(selected_heading):.0f} deg")

        target_robot = rotation.T @ (self.committed_target - self.pose[:2])
        distance = np.linalg.norm(target_robot)
        at_gate_waypoint = goal is not None and np.linalg.norm(self.committed_target - goal) < 1e-6
        # Gate waypoints must be reached precisely: the standoff so the crossing
        # lines up, the exit so the robot is clear of the wall but short of the
        # first moving obstacle.
        # Never count a target as reached as soon as it is chosen: a short
        # target (down to 0.25 m) is driven at least halfway.
        arrive = min(self.retarget_distance,
                     0.5 * getattr(self, 'target_length', float('inf')))
        if at_gate_waypoint:
            arrive = (getattr(self, 'gate_arrive', 0.08) if self.gate_phase == 'approach'
                      else 0.05)
        if distance < arrive:
            self.reset_target()
            if not at_gate_waypoint:
                # Keep the last command for one tick; the next tick picks a new
                # target. Stopping here caused stop-and-go between short targets.
                return
            self.move_2D()
            if at_gate_waypoint and self.gate_phase == 'approach':
                self.gate_phase = 'cross'
                self.get_logger().info("Gate: lined up, crossing")
            elif at_gate_waypoint and self.gate_phase == 'cross':
                self.gate_phase = None
                self.get_logger().info("Gate: crossed")
                self.enter_corridor(self.state_index + 1)
            return
        # Recheck the committed path on every scan, including moving obstacles.
        # Only once facing the target: turning on the spot cannot hit anything
        # ahead, and scans taken mid-turn are smeared by a few cm, which made
        # clear paths look blocked on the real robot (turn -> blocked -> replan).
        crossing_gate = at_gate_waypoint and self.gate_phase == 'cross'
        # Through a gate only drive when nearly straight: a skewed body swings
        # its corners into the gate edges.
        align_limit = (min(self.align_angle, math.radians(getattr(self, 'gate_align_deg', 20.0)))
                       if crossing_gate else self.align_angle)
        facing = abs(math.atan2(target_robot[1], target_robot[0])) <= align_limit
        margin, radius = (self.gate_cross_check() if crossing_gate
                          else (self.commit_margin, None))
        blocked = facing and not self.path_is_clear(points_now, target_robot, margin, radius)
        if self.progress_distance is None or distance < self.progress_distance - 0.03:
            self.progress_distance, self.progress_time = distance, now
        if blocked:
            if self.blocked_since is None:
                self.blocked_since = now
                self.log_stop('blocked', self.blocked_message(points_now, target_robot))
            self.move_2D()
        else:
            self.blocked_since = None
        if at_gate_waypoint and self.gate_phase == 'cross':
            # Something in the gate (e.g. a moving obstacle): wait, don't
            # abandon the crossing. After a long block, line up again.
            if blocked and now - self.blocked_since > 8.0:
                self.get_logger().info("Gate: crossing blocked for 8 s, lining up again")
                self.gate_phase = 'approach'
                self.reset_target()
                self.move_2D()
                return
            if blocked:
                return
            self.progress_time = now  # turning to face the gate is progress
        # Briefly wait for a moving obstacle; then reject this goal temporarily
        # so replanning cannot immediately select the same blocked position.
        wait = getattr(self, 'blocked_wait', 0.7)
        if ((self.blocked_since is not None and now - self.blocked_since >= wait) or
                now - self.progress_time > 5.0):
            why = ("blocked" if self.blocked_since is not None and now - self.blocked_since >= wait
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
        if abs(heading_error) > align_limit:
            vx = 0.0
            self.log_stop('turning', f"Stopped to turn on the spot: target "
                          f"{np.rad2deg(heading_error):.0f} deg off (align_angle_deg)")
        else:
            self.log_stop(None)
        self.move_2D(min(vx, self.navigation_speed, 0.8 * distance), 0, turn)

    def log_stop(self, reason, message=None):
        """Log why the robot stopped, once per stop (not every tick)."""
        if reason != getattr(self, 'stop_reason', None) and message:
            self.get_logger().info(message)
        self.stop_reason = reason

    def blocked_message(self, points_now, target_robot):
        """Describe the scan point that blocks the committed path."""
        distance = float(np.linalg.norm(target_robot))
        if not len(points_now) or distance < 1e-6:
            return "Paused: path blocked"
        target = target_robot * (1 + self.nose_overhang / distance)
        along = points_now @ target
        t = np.clip(along / float(target @ target), 0, 1)
        clearance = np.linalg.norm(points_now - t[:, None] * target, axis=1)
        hit = (along > 0) & (clearance < self.robot_safe_radius + self.commit_margin)
        if not np.any(hit):
            return "Paused: path blocked"
        p = points_now[hit][np.argmin(np.linalg.norm(points_now[hit], axis=1))]
        return (f"Paused: path blocked by a point at x={p[0]:.2f}, y={p[1]:.2f} m "
                f"(robot frame, {np.linalg.norm(p):.2f} m away; "
                f"{np.count_nonzero(hit)} points in the way)")


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course", flush=True)
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    started = last_log = time.monotonic()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar, timeout_sec=0.5)
        if time.monotonic() - last_log > 2.0:
            last_log = time.monotonic()
            print(f"Waiting for first LiDAR scan on /scan "
                  f"({last_log - started:.0f} s)...", flush=True)
    is_simulation = lidar.is_simulation
    print(f"First scan after {time.monotonic() - started:.1f} s: "
          f"{'simulation' if is_simulation else 'real robot'}", flush=True)
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
