import argparse
import configparser
import math
import os
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

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

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

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 

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

    def init_navigation(self):
        '''Keep tuning in one place; initialize only after the calibration block.'''
        from collections import deque
        from time import monotonic
        from tf2_ros import Buffer, TransformListener

        self.nav_settings = {
            "scan_timeout": 0.5, # seconds without a new, advancing scan
            "pose_timeout": 0.5, # seconds without a new, advancing pose
            "pose_scan_tolerance": 0.2, # maximum timestamp mismatch, seconds
            "debug_visuals": True, # optional RViz output; control never reads it
            "debug_interval": 0.2, # seconds between RViz updates
            "status_interval": 1.0, # seconds between terminal printouts
            # Rounded outwards from the simulation collision meshes, in metres.
            "footprint_front": 0.182,
            "footprint_rear": 0.034,
            "footprint_half_width": 0.096,
            # Simulation starting values; physical braking still needs measurement.
            "linear_speed": 0.10,
            "angular_speed": 0.30,
            "minimum_turn_speed": 0.14, # simulation yaw deadband; remeasure on hardware
            "linear_acceleration": 0.20,
            "linear_braking": 0.40,
            "angular_acceleration": 0.60,
            "angular_braking": 0.80,
            "reaction_delay": 0.20,
            "clearance_margin": 0.06,
            "turn_drift_margin": 0.04, # extra room for observed base drift during turns
            "resume_margin": 0.02,
            "resume_scans": 4,
            "forward_test_distance": 0.45,
            "turn_test_angle": 45.0, # degrees
            "position_tolerance": 0.015,
            "heading_tolerance": 2.0, # degrees
            "map_resolution": 0.05, # metres per observed cell
            "mapping_range": 2.5, # cap observation distance, not the course dimensions
            "breadcrumb_spacing": 0.15, # metres along the travelled route
            "frontier_standoff": 0.25, # keep the approach inside observed space
            "approach_search_distance": 0.75, # search farther inward for room to turn
            "minimum_frontier_size": 5, # ignore isolated gaps after diagonal grouping
            "minimum_goal_distance": 0.40,
            "frontier_revisit_distance": 0.20, # skip observation goals beside the travelled trail
            "prefer_motion_frontiers": True, # forward preference from observed motion
            "preferred_clearance": 0.30, # prefer room to turn on planned routes
            "planning_interval": 2.0, # seconds between retries while stopped
            "look_ahead_distance": 0.18, # metres along the planned route
            "goal_tolerance": 0.08,
            "heading_gain": 2.0,
            "heading_derivative_gain": 0.05,
            "distance_gain": 1.0,
            "turn_in_place_angle": 8.0, # align before translating through narrow gaps
            "progress_distance": 0.04,
            "progress_timeout": 25.0, # stop and report; recovery is Stage 6
            "obstacle_wait_timeout": 15.0, # reconsider a persistent blockage
            "goal_cooldown": 30.0, # temporarily skip a failed approach region
            "backtrack_distance": 0.45, # metres of recorded trail to retrace
            "motion_observation_interval": 0.5,
            "motion_range_change": 0.12, # metres; reject small range noise
            "motion_confirmation_samples": 3,
            "motion_confirmation_wait": 24.0, # stopped observation before abandoning a motion hint
            "motion_minimum_span": 0.30, # metres of repeatedly changing returns
            "crossing_speed": 0.30, # simulation only; hardware remains disabled
            "crossing_acceleration": 0.40, # simulation crossing ramp; measure on hardware
            "crossing_padding": 0.05, # allowance beyond observed moving returns
            "crossing_time_margin": 0.25, # seconds beyond reaction/acceleration time
            "crossing_observed_windows": 2,
            "completion_required_bands": 2,
            "completion_wall_distance": 0.65, # metres from pose origin to front wall
            "completion_minimum_wall_span": 0.20,
            "completion_hold_time": 6.0, # seconds of stable closed-end observations
            "completion_observations": 30, # different fresh scans, while stopped
            "completion_gap_factor": 1.5, # allowance over angular ray spacing
        }
        self.sensor_time = monotonic
        self.scan_msg = None
        self.scan_received = self.pose_received = None
        self.scan_stamp = self.pose_stamp = None
        self.pose_history = deque(maxlen=200)
        self.scan_points = np.empty((0, 2))
        self.scan_pose = None
        self.projected_stamp = None
        self.sensor_state = None
        self.last_status = -math.inf
        self.debug_publisher = None
        self.last_debug_time = -math.inf
        self.laser_transform = None
        self.laser_frame = None
        self.last_command = np.zeros(2) # forward m/s, yaw rad/s; lateral is always zero
        self.measured_motion = np.zeros(2)
        self.last_motion_time = monotonic()
        self.waiting_for_clearance = False
        self.clear_scan_count = 0
        self.safety_scan_stamp = None
        self.clearance_history = None
        self.current_clearance = math.inf
        self.minimum_clearance = math.inf
        self.motion_start = None
        self.motion_done = False
        # Explicit simulation-only test selection; hardware always starts stopped.
        self.test_primitive = os.environ.get('RB2301_TEST_PRIMITIVE', 'stopped') if self.is_simulation else 'stopped'
        if self.test_primitive not in ('stopped', 'forward', 'backward', 'turn_left', 'turn_right', 'navigate'):
            raise ValueError('Unknown RB2301_TEST_PRIMITIVE')
        self.nav_settings['debug_visuals'] = os.environ.get('RB2301_DEBUG_VISUALS', '1') != '0'
        self.active_stage = int(os.environ.get('RB2301_STAGE', '5'))
        if self.active_stage not in (2, 3, 4, 5, 6, 7):
            raise ValueError('Only implemented stages 2 through 7 are available')
        if self.test_primitive == 'navigate' and self.active_stage < 5:
            raise ValueError('The route follower is enabled from Stage 5')
        self.observation_grid = {} # (world cell x, y) -> 0 free or 100 occupied; absent is unknown
        self.occupied_evidence = {}
        self.breadcrumbs = []
        self.memory_stamp = None
        self.frontier_groups = []
        self.goal = None
        self.goal_anchor = None
        self.route = []
        self.free_cells = set()
        self.goal_cells = set()
        self.cell_clearance = {}
        self.last_plan_time = -math.inf
        self.goals_selected = 0
        self.goals_reached = 0
        self.goal_checked_stamp = None
        self.failed_goals = {}
        self.blocked_since = None
        self.goal_blocked = False
        self.goal_clear_scans = 0
        self.goal_map_size = 0
        self.observation_start = None
        self.recoveries = 0
        self.backtracking = False
        self.backtrack_reverse = False
        self.backtrack_index = None
        self.motion_baseline = None
        self.motion_evidence = {}
        self.motion_zones = []
        self.motion_checked_stamp = None
        self.crossing_zone = None
        self.crossing_phase = None
        self.course_complete = False
        self.completion_since = self.completion_anchor = self.completion_stamp = None
        self.completion_count = 0
        self.path_progress = 0.0
        self.current_segment_idx = 0
        self.carrot = None
        self.previous_heading_error = None
        self.previous_control_time = None
        self.progress_time = monotonic()
        self.progress_mark = 0.0
        self.navigation_stalled = False
        self.stop_requested = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.full_scan_sub = self.create_subscription(LaserScan, 'scan', self.remember_scan, qos)
        if self.is_simulation:
            self.full_pose_sub = self.create_subscription(Odometry, 'odom', self.remember_pose, qos)
        else:
            self.full_pose_sub = self.create_subscription(
                PoseStamped, f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose', self.remember_pose, qos)
        import signal
        signal.signal(signal.SIGINT, self.stop_navigation)
        signal.signal(signal.SIGTERM, self.stop_navigation)
        self.get_logger().info(f'Stage {self.active_stage}: {self.test_primitive}; hardware motion disabled for these tests.')

    def stop_navigation(self, signum=None, frame=None):
        '''Let the timer stop cleanly instead of raising inside a ROS C callback.'''
        self.stop_requested = True

    def remember_scan(self, msg):
        '''Retain the original message, including angles, frame, ranges and time.'''
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.scan_stamp is None or stamp > self.scan_stamp:
            self.scan_msg = msg
            self.scan_stamp = stamp
            self.scan_received = self.sensor_time()
        elif stamp < self.scan_stamp: # A simulation reset must not reuse the old projection.
            self.scan_stamp = None
            self.scan_msg = None
            self.projected_stamp = None
            self.pose_history.clear()
            self.observation_grid.clear()
            self.occupied_evidence.clear()
            self.breadcrumbs.clear()
            self.memory_stamp = None
            self.goal, self.route = None, []

    def remember_pose(self, msg):
        '''Keep timestamped poses in the same world frame as self.pose.'''
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.pose_stamp is not None and stamp < self.pose_stamp:
            self.pose_history.clear()
            self.pose_stamp = None
        if self.pose_stamp is None or stamp > self.pose_stamp:
            p = msg.pose.pose if self.is_simulation else msg.pose
            heading = math.degrees(self.yaw_from_quaternion(p.orientation))
            pose = (p.position.x, p.position.y, heading)
            if not self.is_simulation:
                pose = optitrack_to_maze(*pose, self.frame)
            if not np.all(np.isfinite(pose)):
                return
            if self.pose_history:
                previous_stamp, previous_pose = self.pose_history[-1]
                dt = stamp - previous_stamp
                if 0 < dt < 0.5:
                    heading = math.radians(pose[2])
                    speed = np.dot(np.asarray(pose[:2])-previous_pose[:2], (math.cos(heading), math.sin(heading))) / dt
                    turn = math.radians(wrap_deg(pose[2] - previous_pose[2])) / dt
                    self.measured_motion = np.array((speed, turn))
            self.pose_history.append((stamp, np.array(pose)))
            self.pose_stamp = stamp
            self.pose_received = self.sensor_time()

    def project_scan(self, msg, pose, laser_transform):
        '''Project finite obstacle returns from the LiDAR through base into world.'''
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid] + laser_transform[2]
        base_x = laser_transform[0] + ranges[valid] * np.cos(angles)
        base_y = laser_transform[1] + ranges[valid] * np.sin(angles)
        yaw = math.radians(pose[2])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.column_stack((pose[0] + c * base_x - s * base_y,
                                pose[1] + s * base_x + c * base_y))

    def check_sensors(self):
        '''Fail closed until fresh, timestamp-aligned data and the mount TF exist.'''
        now = self.sensor_time()
        if self.scan_msg is None or not self.pose_history:
            return 'WAITING_FOR_SENSORS'
        if now - self.scan_received > self.nav_settings['scan_timeout']:
            return 'STALE_SCAN'
        if now - self.pose_received > self.nav_settings['pose_timeout']:
            return 'STALE_POSE'
        msg = self.scan_msg
        if (not msg.ranges or not np.isfinite(msg.angle_increment) or msg.angle_increment == 0
                or not np.isfinite(msg.angle_min) or not np.isfinite(msg.range_min)
                or not np.isfinite(msg.range_max) or not 0 <= msg.range_min < msg.range_max):
            return 'INVALID_SCAN_METADATA'
        ranges = np.asarray(msg.ranges)
        if not np.any(np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges < msg.range_max)):
            # Positive infinity is a valid no-return, but an all-no-return scan cannot
            # validate this enclosed course. Stay stopped instead of inventing walls.
            return 'NO_OBSTACLE_RETURNS'
        frame = msg.header.frame_id.lstrip('/')
        if not frame:
            return 'MISSING_SCAN_FRAME'
        if self.laser_transform is None or frame != self.laser_frame:
            from rclpy.time import Time
            from tf2_ros import TransformException
            try:
                t = self.tf_buffer.lookup_transform('base_footprint', frame, Time()).transform
            except TransformException:
                return 'WAITING_FOR_LASER_TF'
            self.laser_transform = (t.translation.x, t.translation.y,
                                    float(self.yaw_from_quaternion(t.rotation)))
            self.laser_frame = frame
            self.get_logger().info(f'LiDAR mount from TF: {self.laser_transform} (m, m, rad)')
        stamp, pose = min(self.pose_history, key=lambda item: abs(item[0] - self.scan_stamp))
        if abs(stamp - self.scan_stamp) > self.nav_settings['pose_scan_tolerance']:
            return 'POSE_SCAN_TIME_MISMATCH'
        if self.projected_stamp != self.scan_stamp:
            self.scan_pose = pose
            self.scan_points = self.project_scan(msg, pose, self.laser_transform)
            self.projected_stamp = self.scan_stamp
        return 'READY_STOPPED'

    def show_diagnostics(self, state):
        '''Optional RViz output. Remove this helper and its call to remove RViz.'''
        if not self.nav_settings['debug_visuals']:
            return
        now = self.sensor_time()
        if now-self.last_debug_time < self.nav_settings['debug_interval']:
            return
        self.last_debug_time = now
        from copy import deepcopy
        from geometry_msgs.msg import Point, TransformStamped
        from rclpy.duration import Duration
        from tf2_msgs.msg import TFMessage
        from visualization_msgs.msg import Marker, MarkerArray
        display_lifetime = Duration(seconds=max(1.0,3*self.nav_settings['debug_interval'])).to_msg()

        if self.debug_publisher is None:
            self.debug_publisher = self.create_publisher(MarkerArray, '/fp_debug/markers', 2)
            self.debug_publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            self.debug_scan_publisher = self.create_publisher(LaserScan, '/fp_debug/scan', 2)
            self.debug_tf = self.create_publisher(TFMessage, '/fp_debug/tf', 10)
        if self.sensor_ready:
            # Dedicated diagnostic frames avoid changing the robot's existing TF.
            world_robot, robot_laser = TransformStamped(), TransformStamped()
            world_robot.header.stamp = robot_laser.header.stamp = self.get_clock().now().to_msg()
            world_robot.header.frame_id, world_robot.child_frame_id = 'fp_world', 'fp_robot'
            robot_laser.header.frame_id, robot_laser.child_frame_id = 'fp_robot', 'fp_lidar'
            world_robot.transform.translation.x = float(self.scan_pose[0])
            world_robot.transform.translation.y = float(self.scan_pose[1])
            yaw = math.radians(self.scan_pose[2])
            world_robot.transform.rotation.z, world_robot.transform.rotation.w = math.sin(yaw/2), math.cos(yaw/2)
            robot_laser.transform.translation.x, robot_laser.transform.translation.y = self.laser_transform[:2]
            robot_laser.transform.rotation.z = math.sin(self.laser_transform[2]/2)
            robot_laser.transform.rotation.w = math.cos(self.laser_transform[2]/2)
            self.debug_tf.publish(TFMessage(transforms=[world_robot, robot_laser]))
            scan = deepcopy(self.scan_msg)
            scan.header.frame_id = 'fp_lidar'
            scan.header.stamp = world_robot.header.stamp
            self.debug_scan_publisher.publish(scan)
        markers = MarkerArray()
        for number, kind in enumerate((Marker.POINTS, Marker.ARROW, Marker.TEXT_VIEW_FACING, Marker.LINE_STRIP)):
            marker = Marker()
            marker.header.frame_id = 'fp_world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id, marker.type = 'sensors', number, kind
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            markers.markers.append(marker)
        points, robot, label, footprint = markers.markers
        points.scale.x = points.scale.y = 0.025
        points.color.g = points.color.b = 1.0
        if self.sensor_ready:
            points.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in self.scan_points]
        pose = self.pose
        robot.pose.position.x, robot.pose.position.y = float(pose[0]), float(pose[1])
        yaw = math.radians(pose[2])
        robot.pose.orientation.z, robot.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        robot.scale.x, robot.scale.y, robot.scale.z = 0.25, 0.07, 0.07
        robot.color.g = 1.0
        label.pose.position.x, label.pose.position.y, label.pose.position.z = float(pose[0]), float(pose[1]), 0.4
        label.scale.z = 0.12
        label.color.r = label.color.g = 1.0
        label.text = f'Stage {self.active_stage}: {state}'
        footprint.scale.x = 0.015
        footprint.color.r = 1.0 if self.waiting_for_clearance else 0.0
        footprint.color.g = 0.0 if self.waiting_for_clearance else 1.0
        p = self.nav_settings
        for x, y in ((-p['footprint_rear'], -p['footprint_half_width']),
                     (p['footprint_front'], -p['footprint_half_width']),
                     (p['footprint_front'], p['footprint_half_width']),
                     (-p['footprint_rear'], p['footprint_half_width']),
                     (-p['footprint_rear'], -p['footprint_half_width'])):
            footprint.points.append(Point(x=float(pose[0] + math.cos(yaw)*x - math.sin(yaw)*y),
                                          y=float(pose[1] + math.sin(yaw)*x + math.cos(yaw)*y), z=0.06))
        self.debug_publisher.publish(markers)
        if self.active_stage >= 3 and self.observation_grid:
            from nav_msgs.msg import OccupancyGrid
            if not hasattr(self, 'debug_map_publisher'):
                self.debug_map_publisher = self.create_publisher(OccupancyGrid, '/fp_debug/map', 1)
            grid = OccupancyGrid()
            grid.header.frame_id = 'fp_world'
            grid.header.stamp = self.get_clock().now().to_msg()
            grid.info.resolution = self.nav_settings['map_resolution']
            xs, ys = zip(*self.observation_grid)
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
            grid.info.width, grid.info.height = xmax-xmin+1, ymax-ymin+1
            grid.info.origin.position.x = xmin*grid.info.resolution
            grid.info.origin.position.y = ymin*grid.info.resolution
            grid.info.origin.orientation.w = 1.0
            data = np.full((grid.info.height, grid.info.width), -1, dtype=np.int8)
            for (x, y), value in self.observation_grid.items():
                data[y-ymin, x-xmin] = value
            grid.data = data.ravel().tolist()
            self.debug_map_publisher.publish(grid)
            trail = Marker()
            trail.header, trail.ns, trail.id = grid.header, 'breadcrumbs', 0
            trail.type, trail.action = Marker.LINE_STRIP, Marker.ADD
            trail.pose.orientation.w, trail.scale.x = 1.0, 0.02
            trail.color.r = trail.color.b = trail.color.a = 1.0
            trail.points = [Point(x=float(x), y=float(y), z=0.07) for x, y in self.breadcrumbs]
            self.debug_publisher.publish(MarkerArray(markers=[trail]))
            if self.active_stage >= 4:
                route, frontiers, goal = Marker(), Marker(), Marker()
                for number, marker in enumerate((route, frontiers, goal)):
                    marker.header, marker.ns, marker.id = grid.header, 'planning', number
                    marker.action, marker.pose.orientation.w = Marker.ADD, 1.0
                    marker.color.a = 1.0
                route.type, route.scale.x = Marker.LINE_STRIP, 0.025
                route.color.r = route.color.g = 1.0
                route.points = [Point(x=float(x), y=float(y), z=0.09) for x, y in self.route]
                frontiers.type = Marker.POINTS
                frontiers.scale.x = frontiers.scale.y = 0.045
                frontiers.color.b, frontiers.color.r = 1.0, 0.5
                frontiers.points = [Point(x=float(x), y=float(y), z=0.08)
                                    for group in self.frontier_groups for cell in group
                                    for x, y in [self.cell_world(cell)]]
                goal.type = Marker.SPHERE
                goal.scale.x = goal.scale.y = goal.scale.z = 0.12
                goal.color.r = 1.0
                if self.goal is not None:
                    goal.pose.position.x, goal.pose.position.y = self.goal
                    goal.pose.position.z = 0.12
                else:
                    goal.action = Marker.DELETE
                self.debug_publisher.publish(MarkerArray(markers=[route, frontiers, goal]))
                # Make the goal's extra turning-room test visible separately
                # from the body footprint at the robot's current position.
                room = Marker()
                room.header, room.ns, room.id = grid.header, 'planning', 3
                room.type = Marker.LINE_STRIP
                show_room = self.goal is not None and self.crossing_zone is None and not self.backtracking
                room.action = Marker.ADD if show_room else Marker.DELETE
                room.pose.orientation.w, room.scale.x = 1.0, 0.012
                room.color.r, room.color.g, room.color.a = 1.0, 0.5, 0.7
                if show_room:
                    radius = math.hypot(max(p['footprint_front'],p['footprint_rear']),p['footprint_half_width'])+p['clearance_margin']+p['turn_drift_margin']
                    room.points = [Point(x=float(self.goal[0]+radius*math.cos(t)),
                                         y=float(self.goal[1]+radius*math.sin(t)),z=0.10)
                                   for t in np.linspace(0,2*math.pi,49)]
                self.debug_publisher.publish(MarkerArray(markers=[room]))
                carrot = Marker()
                carrot.header, carrot.ns, carrot.id = grid.header, 'carrot', 0
                carrot.type, carrot.action = Marker.SPHERE, Marker.ADD if self.carrot is not None else Marker.DELETE
                carrot.pose.orientation.w = 1.0
                carrot.scale.x = carrot.scale.y = carrot.scale.z = 0.08
                carrot.color.r, carrot.color.g, carrot.color.a = 1.0, 0.5, 1.0
                if self.carrot is not None:
                    carrot.pose.position.x, carrot.pose.position.y = map(float, self.carrot)
                    carrot.pose.position.z = 0.14
                self.debug_publisher.publish(MarkerArray(markers=[carrot]))
                if self.active_stage >= 6:
                    direction = Marker()
                    direction.lifetime = display_lifetime
                    direction.header, direction.ns, direction.id = grid.header, 'motion_preference', 0
                    direction.type, direction.pose.orientation.w = Marker.ARROW, 1.0
                    direction.scale.x, direction.scale.y, direction.scale.z = 0.025, 0.06, 0.08
                    direction.color.g = direction.color.b = direction.color.a = 1.0
                    target = self.motion_target()
                    direction.action = Marker.DELETE if target is None else Marker.ADD
                    if target is not None:
                        delta = target-self.pose[:2]
                        end = self.pose[:2]+delta*min(1.0,0.6/max(np.linalg.norm(delta),1e-6))
                        direction.points = [Point(x=float(x),y=float(y),z=0.18) for x,y in (self.pose[:2],end)]
                    self.debug_publisher.publish(MarkerArray(markers=[direction]))
                    for index, zone in enumerate(self.motion_zones):
                        band = Marker()
                        band.lifetime = display_lifetime
                        band.header, band.ns, band.id = grid.header, 'observed_motion', index
                        band.type, band.action, band.pose.orientation.w = Marker.LINE_STRIP, Marker.ADD, 1.0
                        band.scale.x, band.color.a = 0.02, 0.8
                        band.color.r, band.color.g = (0.0,1.0) if zone['passed'] else (1.0,0.3)
                        base = zone['centre']-np.dot(zone['centre'],zone['axis'])*zone['axis']
                        pad = p['crossing_padding']
                        for lateral,offset in ((zone['low'],-pad),(zone['high'],-pad),(zone['high'],pad),(zone['low'],pad),(zone['low'],-pad)):
                            x,y = base+lateral*zone['axis']+offset*zone['normal']
                            band.points.append(Point(x=float(x),y=float(y),z=0.12))
                        self.debug_publisher.publish(MarkerArray(markers=[band]))

    def neighbours(self, cell, diagonals=False):
        '''Paths use four edges; frontier groups also join at touching corners.'''
        x, y = cell
        if diagonals:
            return ((x+1,y), (x-1,y), (x,y+1), (x,y-1),
                    (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1))
        return ((x+1,y), (x-1,y), (x,y+1), (x,y-1))

    def planning_space(self):
        '''Known-free centre cells with obstacle clearance; unknown is not a route.'''
        p = self.nav_settings
        resolution = p['map_resolution']
        free = {cell for cell, value in self.observation_grid.items() if value == 0}
        turn_clearance = math.hypot(max(p['footprint_front'], p['footprint_rear']), p['footprint_half_width'])+p['clearance_margin']+p['turn_drift_margin']+resolution/math.sqrt(2)
        clearance_cap = max(p['preferred_clearance'], turn_clearance)
        clearance = dict.fromkeys(free, clearance_cap)
        radius = math.ceil(clearance_cap/resolution)
        offsets = [(dx,dy,math.hypot(dx,dy)*resolution)
                   for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)
                   if math.hypot(dx,dy)*resolution <= clearance_cap]
        for (x,y), value in self.observation_grid.items():
            if value != 100:
                continue
            for dx, dy, distance in offsets:
                cell = (x+dx,y+dy)
                if cell in clearance and distance < clearance[cell]:
                    clearance[cell] = distance
        padding = p['footprint_half_width']+p['clearance_margin']+resolution/2
        self.cell_clearance = clearance
        self.free_cells = {cell for cell in free if clearance[cell] >= padding}
        self.goal_cells = {cell for cell in self.free_cells if clearance[cell] >= turn_clearance}
        if len(getattr(self, 'scan_points', [])):
            required = turn_clearance-resolution/math.sqrt(2)
            self.goal_cells = {cell for cell in self.goal_cells
                               if np.min(np.linalg.norm(self.scan_points-np.asarray(self.cell_world(cell)), axis=1)) >= required}
        now = self.sensor_time()
        self.failed_goals = {cell:expiry for cell,expiry in getattr(self, 'failed_goals', {}).items() if expiry > now}
        skip_radius = 2*p['footprint_half_width']+p['clearance_margin']
        cells = math.ceil(skip_radius/resolution)
        for x,y in self.failed_goals:
            self.goal_cells.difference_update((x+dx,y+dy) for dx in range(-cells,cells+1) for dy in range(-cells,cells+1)
                                              if math.hypot(dx,dy)*resolution <= skip_radius)
        if getattr(self, 'active_stage', 5) >= 6:
            radius = math.ceil(p['frontier_revisit_distance']/resolution)
            offsets = [(dx,dy) for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)
                       if math.hypot(dx,dy)*resolution <= p['frontier_revisit_distance']]
            for point in self.breadcrumbs:
                x,y = self.world_cell(*point)
                self.goal_cells.difference_update((x+dx,y+dy) for dx,dy in offsets)

    def reconstruct_path(self, parents, current):
        '''Adapted from CA2: walk parent links, then reverse into travel order.'''
        path = [current]
        while current in parents:
            current = parents[current]
            path.append(current)
        return path[::-1]

    def Astar(self, start, goal):
        '''CA2 four-neighbour A*, adapted to observed cells and a heap queue.'''
        import heapq
        heuristic = lambda cell: abs(cell[0]-goal[0])+abs(cell[1]-goal[1])
        queue, costs, parents = [(heuristic(start), 0.0, start)], {start:0.0}, {}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost != costs.get(current):
                continue
            if current == goal:
                return self.reconstruct_path(parents, current)
            for cell in self.neighbours(current):
                if cell not in self.free_cells:
                    continue
                # Clearance affects the route, not the frontier information score.
                step = 1+max(0.0, self.nav_settings['preferred_clearance']-self.cell_clearance[cell])/self.nav_settings['map_resolution']
                candidate = cost+step
                if candidate < costs.get(cell, math.inf):
                    costs[cell], parents[cell] = candidate, current
                    heapq.heappush(queue, (candidate+heuristic(cell), candidate, cell))
        return []

    def simplify_path(self, path):
        '''CA2 collinear simplification; never shortcut a corner through a wall.'''
        if len(path) < 3:
            return path
        result = [path[0]]
        for i in range(1, len(path)-1):
            incoming = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
            outgoing = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            if incoming != outgoing:
                result.append(path[i])
        return result+[path[-1]]

    def select_local_goal(self):
        '''Largest reachable frontier group, then shortest available approach.'''
        from collections import deque
        self.planning_space()
        free = {cell for cell, value in self.observation_grid.items() if value == 0}
        frontier = {cell for cell in free if any(n not in self.observation_grid for n in self.neighbours(cell))}
        groups = []
        while frontier:
            seed = min(frontier)
            frontier.remove(seed)
            group, queue = [seed], deque([seed])
            while queue:
                for cell in self.neighbours(queue.popleft(), diagonals=True):
                    if cell in frontier:
                        frontier.remove(cell)
                        group.append(cell)
                        queue.append(cell)
            if len(group) >= self.nav_settings['minimum_frontier_size']:
                groups.append(group)
        self.frontier_groups = groups
        start = self.world_cell(*self.pose[:2])
        if start not in self.free_cells:
            return False
        distances, queue = {start:0}, deque([start])
        while queue:
            current = queue.popleft()
            for cell in self.neighbours(current):
                if cell in self.free_cells and cell not in distances:
                    distances[cell] = distances[current]+1
                    queue.append(cell)
        p, candidates = self.nav_settings, []
        target = self.motion_target() if getattr(self, 'active_stage', 5) >= 6 and p['prefer_motion_frontiers'] else None
        search_distance = max(p['frontier_standoff'], p['approach_search_distance'])
        radius = math.ceil(search_distance/p['map_resolution'])
        offsets = [(dx,dy,math.hypot(dx,dy)*p['map_resolution'])
                   for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)
                   if p['frontier_standoff'] <= math.hypot(dx,dy)*p['map_resolution'] <= search_distance]
        for group in groups:
            # Local goals must leave room to turn toward the following route.
            # The route itself can still pass through a narrower doorway.
            approaches = [(distance, distances[(x+dx,y+dy)], (x+dx,y+dy))
                          for x,y in group for dx,dy,distance in offsets
                          if (x+dx,y+dy) in distances
                          and (x+dx,y+dy) in self.goal_cells
                          and distances[(x+dx,y+dy)]*p['map_resolution'] >= p['minimum_goal_distance']
                          # A nearby approach across a wall cannot reveal this frontier.
                          and all(self.observation_grid.get(c) == 0
                                  for c in self.ray_cells((x+dx,y+dy), (x,y)))]
            if approaches:
                _, _, approach = min(approaches)
                centre = np.mean([self.cell_world(c) for c in group],axis=0)
                toward_motion = target is None or np.dot(centre-self.pose[:2],target-self.pose[:2]) > 0
                candidates.append((not toward_motion, -len(group), distances[approach], approach))
        for _, _, _, approach in sorted(candidates):
            cells = self.Astar(start, approach)
            if cells:
                self.begin_route([tuple(self.pose[:2])]+[self.cell_world(cell) for cell in self.simplify_path(cells)[1:]],self.sensor_time())
                self.goals_selected += 1
                self.get_logger().info(f'GOAL_SELECTED {self.goal} | route_points={len(self.route)} | motion_preference={target is not None} | groups={[len(g) for g in groups]}')
                return True
        return False

    def begin_route(self, route, now):
        '''One reset for frontier routes, moving-band approaches and breadcrumb retreats.'''
        self.route, self.goal = route, route[-1]
        self.goal_anchor, self.carrot, self.goal_checked_stamp = self.goal, None, None
        self.goal_blocked, self.goal_clear_scans, self.blocked_since = False, 0, None
        self.goal_map_size = len(self.observation_grid)
        self.path_progress, self.current_segment_idx = 0.0, 0
        self.progress_time, self.progress_mark = now, 0.0
        self.previous_heading_error = self.previous_control_time = None
        self.observation_start, self.navigation_stalled = None, False

    def route_stalled(self, now):
        '''Only travelled route progress resets the timer; safety waits pause it.'''
        if self.waiting_for_clearance:
            self.progress_time = now
        if self.path_progress >= self.progress_mark+self.nav_settings['progress_distance']:
            self.progress_mark, self.progress_time = self.path_progress, now
            self.blocked_since = None
        return now-self.progress_time > self.nav_settings['progress_timeout']

    def get_carrot(self, robot_position, path, look_ahead_dist):
        '''CA2 arc-length carrot, using world points and monotonic path progress.'''
        if len(path) < 2:
            return None
        points = np.asarray(path)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        while True:
            index = self.current_segment_idx
            a, b = points[index], points[index+1]
            ab = b-a
            t = float(np.clip(np.dot(np.asarray(robot_position)-a, ab)/max(np.dot(ab,ab), 1e-12), 0, 1))
            if index < len(lengths)-1:
                next_ab = points[index+2]-b
                next_t = float(np.clip(np.dot(np.asarray(robot_position)-b, next_ab)/max(np.dot(next_ab,next_ab), 1e-12), 0, 1))
                current_distance = np.linalg.norm(np.asarray(robot_position)-(a+t*ab))
                next_distance = np.linalg.norm(np.asarray(robot_position)-(b+next_t*next_ab))
                # A rounded corner can miss the waypoint yet already lie along
                # the next segment. Only consider the adjacent segment, so a
                # nearby later branch cannot steal progress through a wall.
                passed_corner = next_t > 0 and next_distance+0.01 < current_distance
                if t >= 0.95 or math.dist(robot_position, b) < 0.05 or passed_corner:
                    self.current_segment_idx += 1
                    continue
            break
        self.path_progress = max(self.path_progress, cumulative[index]+t*lengths[index])
        distance = min(self.path_progress+look_ahead_dist, cumulative[-1])
        segment = min(len(lengths)-1, int(np.searchsorted(cumulative, distance, side='right')-1))
        alpha = (distance-cumulative[segment])/max(lengths[segment], 1e-12)
        return points[segment]+alpha*(points[segment+1]-points[segment])

    def segment_clear(self, start, end):
        '''Check the body along a carrot shortcut and the initial turn toward it.'''
        delta = np.asarray(end)-start
        length = float(np.linalg.norm(delta))
        yaw = math.atan2(delta[1], delta[0]) if length > 1e-6 else math.radians(self.pose[2])
        if getattr(self, 'backtrack_reverse', False):
            yaw += math.pi
        # The local goal is satisfied within its tolerance. Checking beyond that
        # stopping point can reject a safe approach beside a wall unnecessarily.
        goal = getattr(self, 'goal', None)
        if goal is not None and math.dist(end, goal) < 1e-6 and length > 1e-6:
            delta *= max(0.0, length-self.nav_settings['goal_tolerance'])/length
            length = float(np.linalg.norm(delta))
        clearance = self.nav_settings['clearance_margin']
        current_yaw = math.radians(self.pose[2])
        turn = math.atan2(math.sin(yaw-current_yaw), math.cos(yaw-current_yaw))
        for angle in np.linspace(current_yaw, current_yaw+turn, max(2, math.ceil(abs(turn)/0.1)+1)):
            if self.footprint_clearance(self.scan_points, start[0], start[1], angle) < clearance:
                return False
        for alpha in np.linspace(0, 1, max(2, math.ceil(length/0.025)+1)):
            point = np.asarray(start)+alpha*delta
            if self.footprint_clearance(self.scan_points, point[0], point[1], yaw) < clearance:
                return False
        return True

    def follow_carrot(self, now):
        '''CA2 distance/heading PD control, with feasible forward/turn commands.'''
        p = self.nav_settings
        lookahead = p['look_ahead_distance']
        self.carrot = self.get_carrot(self.pose[:2], self.route, lookahead)
        while self.carrot is not None and not self.segment_clear(self.pose[:2], self.carrot):
            lookahead /= 2
            if lookahead < 0.025:
                self.previous_heading_error = self.previous_control_time = None
                return (0.0, 0.0), 'ROUTE_BLOCKED'
            self.carrot = self.get_carrot(self.pose[:2], self.route, lookahead)
        if self.carrot is None:
            return (0.0, 0.0), 'NO_ROUTE'
        error = self.carrot-self.pose[:2]
        desired = math.atan2(error[1], error[0])
        if getattr(self, 'backtrack_reverse', False):
            desired += math.pi
        heading_error = math.atan2(math.sin(desired-math.radians(self.pose[2])), math.cos(desired-math.radians(self.pose[2])))
        turn = self.heading_command(heading_error,now)
        speed = min(p['linear_speed'], p['distance_gain']*float(np.linalg.norm(error)), math.dist(self.pose[:2], self.goal))
        speed *= max(0.0, math.cos(heading_error))
        if abs(math.degrees(heading_error)) > p['turn_in_place_angle']:
            speed = 0.0
        if getattr(self, 'backtrack_reverse', False):
            speed = -speed
        return (speed, turn), 'FOLLOWING'

    def heading_command(self, error, now=None):
        '''Shared radian P/PD controller, deadband, turn cap and minimum turn speed.'''
        p, derivative = self.nav_settings, 0.0
        if now is not None:
            if self.previous_heading_error is not None and self.previous_control_time is not None:
                dt = now-self.previous_control_time
                if 0.001 < dt < 0.5:
                    change = math.atan2(math.sin(error-self.previous_heading_error),math.cos(error-self.previous_heading_error))
                    derivative = change/dt
            self.previous_heading_error, self.previous_control_time = error, now
        turn = float(np.clip(p['heading_gain']*error+p['heading_derivative_gain']*derivative,-p['angular_speed'],p['angular_speed']))
        if abs(math.degrees(error)) < p['heading_tolerance']:
            return 0.0
        return math.copysign(p['minimum_turn_speed'],error) if abs(turn) < p['minimum_turn_speed'] else turn

    def refine_goal(self):
        '''Keep the same observation goal when a nearby safe stopping cell exists.'''
        if not len(self.scan_points) or len(self.route) < 2:
            return False
        p = self.nav_settings
        anchor = self.goal_anchor if self.goal_anchor is not None else self.goal
        centre = self.world_cell(*anchor)
        radius = math.ceil(p['goal_tolerance']/p['map_resolution'])
        required = math.hypot(max(p['footprint_front'], p['footprint_rear']), p['footprint_half_width'])+p['clearance_margin']+p['turn_drift_margin']
        candidates = []
        for x in range(centre[0]-radius, centre[0]+radius+1):
            for y in range(centre[1]-radius, centre[1]+radius+1):
                point = self.cell_world((x,y))
                if self.observation_grid.get((x,y)) != 0 or math.dist(point, anchor) > p['goal_tolerance']:
                    continue
                if np.min(np.linalg.norm(self.scan_points-np.asarray(point), axis=1)) < required:
                    continue
                if all(self.observation_grid.get(c) == 0 for c in self.ray_cells(self.world_cell(*self.route[-2]), (x,y))):
                    candidates.append((math.dist(point, anchor), point))
        if not candidates:
            return False
        _, self.goal = min(candidates)
        self.route[-1] = self.goal
        self.previous_heading_error = self.previous_control_time = None
        self.get_logger().info(f'GOAL_ADJUSTED {anchor} -> {self.goal}')
        return True

    def discard_goal(self, reason, now):
        '''Skip this approach briefly, then let the ordinary planner try another.'''
        if self.goal_anchor is not None:
            self.failed_goals[self.world_cell(*self.goal_anchor)] = now+self.nav_settings['goal_cooldown']
        self.get_logger().info(f'RECOVERY_REPLAN: {reason}')
        self.goal, self.goal_anchor, self.route, self.carrot = None, None, [], None
        self.navigation_stalled = self.goal_blocked = False
        self.blocked_since, self.goal_clear_scans = None, 0
        # A navigation goal change must not erase safety hysteresis or motion evidence.
        self.previous_heading_error = self.previous_control_time = None
        self.progress_time = now
        self.recoveries += 1

    def start_backtracking(self, now):
        '''Retrace recorded positions in reverse; current LiDAR still has final say.'''
        if self.backtracking or len(self.breadcrumbs) < 3:
            return False
        route, distance = [tuple(self.pose[:2])], 0.0
        for index in range(len(self.breadcrumbs)-1, -1, -1):
            point = self.breadcrumbs[index]
            distance += math.dist(route[-1], point)
            route.append(point)
            if distance >= self.nav_settings['backtrack_distance']:
                self.begin_route(route,now)
                self.backtracking, self.backtrack_index = True, index
                first = next(np.asarray(p)-self.pose[:2] for p in route[1:] if math.dist(p, self.pose[:2]) > 0.02)
                heading = math.radians(self.pose[2])
                self.backtrack_reverse = np.dot(first, (math.cos(heading), math.sin(heading))) < 0
                self.get_logger().info(f'BACKTRACK_STARTED: {distance:.2f}m along {len(route)} recorded points')
                return True
        return False

    def observe_motion(self):
        '''Confirm moving bands from changing ranges while the robot is stopped.'''
        if self.motion_checked_stamp == self.scan_stamp:
            return
        self.motion_checked_stamp = self.scan_stamp
        p = self.nav_settings
        current = (self.scan_stamp, self.scan_pose.copy(), np.asarray(self.scan_msg.ranges).copy())
        if self.motion_baseline is None:
            self.motion_baseline = current
            return
        stamp, pose, ranges = self.motion_baseline
        if self.scan_stamp-stamp < p['motion_observation_interval']:
            return
        self.motion_baseline = current
        # World projection alone does not distinguish motion from changing visibility.
        # Compare only nearly identical viewpoints, after the robot has settled.
        if (math.dist(pose[:2], self.scan_pose[:2]) > p['map_resolution']/10
                or abs(wrap_deg(pose[2]-self.scan_pose[2])) > p['heading_tolerance']/4
                or len(ranges) != len(current[2])):
            return
        new = current[2]
        valid = (np.isfinite(ranges) & np.isfinite(new) & (ranges >= self.scan_msg.range_min)
                 & (new >= self.scan_msg.range_min) & (np.minimum(ranges,new) < p['mapping_range'])
                 & (np.abs(ranges-new) > p['motion_range_change']))
        # The nearer return belongs to the occluder, rather than the wall revealed behind it.
        from copy import copy
        msg = copy(self.scan_msg)
        msg.ranges = np.where(valid, np.minimum(ranges,new), np.nan).tolist()
        points = self.project_scan(msg, self.scan_pose, self.laser_transform)
        for cell in {self.world_cell(x,y) for x,y in points}:
            count, first, _ = self.motion_evidence.get(cell, (0,self.scan_stamp,self.scan_stamp))
            self.motion_evidence[cell] = count+1, first, self.scan_stamp
        remaining = {c for c,(count,first,last) in self.motion_evidence.items()
                     if count >= p['motion_confirmation_samples'] and last-first >= 2*p['motion_observation_interval']}
        while remaining:
            group, queue = [remaining.pop()], []
            queue.extend(group)
            while queue:
                x,y = queue.pop()
                for dx in range(-2,3):
                    for dy in range(-2,3):
                        cell = x+dx,y+dy
                        if cell in remaining:
                            remaining.remove(cell)
                            group.append(cell)
                            queue.append(cell)
            if len(group) < p['motion_confirmation_samples']:
                continue
            cloud = np.array([self.cell_world(c) for c in group])
            values, vectors = np.linalg.eigh(np.cov(cloud.T))
            axis = vectors[:,-1]
            normal = np.array((axis[1], -axis[0]))
            along, across = cloud@axis, cloud@normal
            if np.ptp(along) < p['motion_minimum_span'] or np.ptp(across) > 3*p['map_resolution']:
                continue
            centre = cloud.mean(axis=0)
            match = next((z for z in self.motion_zones if abs(np.dot(centre-z['centre'], z['normal'])) < 3*p['map_resolution']
                          and abs(np.dot(axis,z['axis'])) > 0.95), None)
            if match is None:
                if np.dot(centre-self.pose[:2],normal) < 0:
                    normal = -normal
                match = dict(centre=centre,axis=axis,normal=normal,passed=False,lateral=None,
                             clear_since=None,windows=[],was_clear=None)
                self.motion_zones.append(match)
                self.get_logger().info(f'MOVING_BAND_CONFIRMED: centre={centre.round(2)} span={np.ptp(along):.2f}m')
            # Keep the initial direction fixed. Extend only the observed sweep bounds.
            match['low'] = min(match.get('low',math.inf),float(np.min(cloud@match['axis'])))
            match['high'] = max(match.get('high',-math.inf),float(np.max(cloud@match['axis'])))

    def motion_target(self):
        '''Direction hint from sensor evidence; only confirmed bands can count as passed.'''
        remaining = [z for z in self.motion_zones if not z['passed']]
        if remaining:
            return min(remaining,key=lambda z:math.dist(self.pose[:2],z['centre']))['centre']
        recent = [self.cell_world(c) for c,(_,_,last) in self.motion_evidence.items()
                  if self.scan_stamp-last <= self.nav_settings['motion_confirmation_wait']
                  and not any(z['passed'] and abs(np.dot(np.asarray(self.cell_world(c))-z['centre'],z['normal'])) <= self.nav_settings['crossing_padding']
                              for z in self.motion_zones)]
        return np.mean(recent,axis=0) if len(recent) >= self.nav_settings['motion_confirmation_samples'] else None

    def moving_band_points(self, zone):
        '''Current visible returns on a confirmed band; an occluded band is unknown.'''
        p = self.nav_settings
        normal, along = (self.scan_points-zone['centre'])@zone['normal'], self.scan_points@zone['axis']
        mask = ((np.abs(normal) <= p['crossing_padding'])
                & (along >= zone['low']-p['map_resolution']) & (along <= zone['high']+p['map_resolution']))
        return self.scan_points[mask]

    def moving_obstacle_command(self, now):
        '''Approach outside the sweep, observe clear cycles, then cross straight.'''
        p = self.nav_settings
        if getattr(self, 'backtracking', False):
            return None
        zone = self.crossing_zone
        if zone is None:
            choices = [z for z in self.motion_zones if not z['passed']
                       and now >= z.get('retry_after', -math.inf)
                       and 0 < np.dot(z['centre']-self.pose[:2],z['normal']) < p['mapping_range']
                       and z['low'] <= np.dot(self.pose[:2],z['axis']) <= z['high']]
            if not choices:
                return None
            zone = min(choices,key=lambda z:math.dist(self.pose[:2],z['centre']))
            self.planning_space()
            # Use the edges of known space with room to align. A newly detected sweep
            # may be incomplete; its first visible extent is not a corridor edge.
            stand_off = p['footprint_front']+p['clearance_margin']+p['crossing_padding']+p['turn_drift_margin']
            if np.dot(zone['centre']-self.pose[:2],zone['normal']) < stand_off:
                self.discard_goal('retreat before selecting a moving-band approach',now)
                self.start_backtracking(now)
                return (0.0,0.0), 'SWING_RETREAT'
            yaw = math.atan2(zone['normal'][1],zone['normal'][0])
            candidates = [c for c in self.free_cells
                          if stand_off <= np.dot(zone['centre']-self.cell_world(c),zone['normal']) <= stand_off+p['map_resolution']
                          and zone['low']-p['motion_minimum_span'] <= np.dot(self.cell_world(c),zone['axis']) <= zone['high']+p['motion_minimum_span']
                          and all(self.footprint_clearance(self.scan_points,*self.cell_world(c),angle) >= p['clearance_margin']+p['turn_drift_margin']
                                  for angle in np.linspace(yaw-math.pi/2,yaw+math.pi/2,19))]
            if not candidates:
                return None # Continue frontier exploration until a staging route is known.
            edges = [min(candidates,key=lambda c:np.dot(self.cell_world(c),zone['axis'])),
                     max(candidates,key=lambda c:np.dot(self.cell_world(c),zone['axis']))]
            for cell in sorted(edges,key=lambda c:math.dist(self.cell_world(c),self.pose[:2])):
                cells = self.Astar(self.world_cell(*self.pose[:2]),cell)
                if cells:
                    self.begin_route([tuple(self.pose[:2])]+[self.cell_world(c) for c in self.simplify_path(cells)[1:]],now)
                    zone['lateral'] = float(np.dot(self.goal,zone['axis']))
                    self.crossing_zone, self.crossing_phase = zone, 'APPROACH'
                    self.backtracking = self.backtrack_reverse = False
                    self.get_logger().info(f'SWING_APPROACH: stable staging goal {self.goal}')
                    break
            if self.crossing_zone is None:
                return None
        if self.crossing_phase == 'APPROACH':
            if (self.blocked_since is not None and now-self.blocked_since >= p['obstacle_wait_timeout']) or self.route_stalled(now):
                zone['retry_after'] = now+p['goal_cooldown']
                self.crossing_zone, self.crossing_phase = None, None
                self.discard_goal('moving-band approach remained blocked',now)
                self.start_backtracking(now)
                return (0.0,0.0), 'SWING_RETREAT'
            if math.dist(self.pose[:2],self.goal) > 2*p['position_tolerance']:
                command, state = self.follow_carrot(now)
                return command, 'SWING_APPROACH' if state == 'FOLLOWING' else state
            self.crossing_phase = 'OBSERVE'
            zone['clear_since'], zone['was_clear'], zone['windows'] = None, None, []
            zone['aligned_lateral'] = False
        normal = zone['normal']
        error = math.atan2(normal[1],normal[0])-math.radians(self.pose[2])
        error = math.atan2(math.sin(error),math.cos(error))
        aligned = abs(math.degrees(error)) <= p['heading_tolerance']
        turn = 0.0 if aligned else self.heading_command(error)
        finish = p['footprint_rear']+p['clearance_margin']+p['crossing_padding']
        remaining = finish-float(np.dot(self.pose[:2]-zone['centre'],normal))
        if self.crossing_phase == 'CROSS':
            if remaining <= 0:
                zone['passed'] = True
                self.crossing_phase = 'SETTLE'
                self.blocked_since = None
                self.get_logger().info(f'MOVING_BAND_PASSED: {sum(z["passed"] for z in self.motion_zones)} confirmed bands')
            else:
                return (p['crossing_speed'] if aligned else 0.0,turn), 'SWING_CROSSING'
        if self.crossing_phase == 'SETTLE':
            if abs(self.measured_motion[0]) < 0.01 and np.max(np.abs(self.last_command)) < 1e-4:
                self.crossing_zone, self.crossing_phase = None, None
                self.goal, self.goal_anchor, self.route, self.carrot = None, None, [], None
                self.blocked_since, self.progress_time = None, now
            return (0.0,0.0), 'SWING_STOPPING'
        if self.crossing_phase == 'SENSOR_HOLD':
            return (0.0,0.0), 'SWING_SENSOR_HOLD'
        if not aligned or abs(self.measured_motion[0]) > 0.01:
            return (0.0,turn), 'SWING_ALIGNING'
        if not zone.get('aligned_lateral', False):
            # Turning can move the pose origin slightly. Time the actual crossing line.
            zone['lateral'] = float(np.dot(self.pose[:2],zone['axis']))
            zone['aligned_lateral'] = True
        if zone.get('window_stamp') != self.scan_stamp:
            zone['window_stamp'] = self.scan_stamp
            points = self.moving_band_points(zone)
            clear = bool(len(points) >= 2 and np.min(np.abs(points@zone['axis']-zone['lateral'])) >= p['footprint_half_width']+p['clearance_margin']+p['map_resolution']/2)
            if clear and zone['was_clear'] is False:
                zone['clear_since'] = self.scan_stamp
            elif not clear and zone['was_clear'] and zone['clear_since'] is not None:
                duration = self.scan_stamp-zone['clear_since']
                if len(points) >= 2:
                    zone['windows'].append(duration)
                    self.get_logger().info(f'SWING_CLEAR_WINDOW: {duration:.2f}s observed')
                zone['clear_since'] = None
            zone['was_clear'] = clear
        if len(zone['windows']) >= p['crossing_observed_windows'] and zone['clear_since'] is not None:
            available = min(zone['windows'][-p['crossing_observed_windows']:])-(self.scan_stamp-zone['clear_since'])
            required = remaining/p['crossing_speed']+p['crossing_speed']/(2*p['crossing_acceleration'])+p['reaction_delay']+p['crossing_time_margin']
            if available >= required and not self.waiting_for_clearance:
                self.crossing_phase = 'CROSS'
                self.blocked_since = None
                self.get_logger().info(f'SWING_CROSSING: window remaining={available:.2f}s required={required:.2f}s')
                return (p['crossing_speed'],0.0), 'SWING_CROSSING'
        return (0.0,0.0), 'SWING_OBSERVING'

    def terminal_walls(self, normal, axis):
        '''Fit front/left/right rectilinear walls from the current projected scan.'''
        p = self.nav_settings
        delta = self.scan_points-self.pose[:2]
        forward, lateral = delta@normal, delta@axis
        resolution = p['map_resolution']
        front_cells = sorted(set(np.rint(forward[(forward > p['footprint_front']+p['clearance_margin'])
                                                    & (forward <= p['completion_wall_distance'])]/resolution).astype(int)))
        for cell in front_cells:
            mask = np.abs(forward-cell*resolution) <= resolution
            if np.count_nonzero(mask) < 6:
                continue
            face = float(np.median(forward[mask]))
            span = lateral[mask]
            if not (np.min(span) < -p['footprint_half_width']-p['clearance_margin']
                    and np.max(span) > p['footprint_half_width']+p['clearance_margin']):
                continue
            side_points = (forward >= -p['footprint_rear']) & (forward <= face+resolution)
            sides = []
            for sign in (-1,1):
                candidates = sorted(set(np.rint(sign*lateral[side_points & (sign*lateral > p['footprint_half_width']+p['clearance_margin'])]/resolution).astype(int)))
                wall = None
                for side_cell in candidates:
                    side_mask = side_points & (np.abs(sign*lateral-side_cell*resolution) <= resolution/2)
                    if np.count_nonzero(side_mask) >= 3 and np.ptp(forward[side_mask]) >= p['completion_minimum_wall_span']:
                        wall = float(np.median(lateral[side_mask]))
                        break
                if wall is None:
                    break
                sides.append(wall)
            if len(sides) == 2 and np.min(span) <= sides[0]+2*resolution and np.max(span) >= sides[1]-2*resolution:
                covered = np.sort(span[(span >= sides[0]-resolution) & (span <= sides[1]+resolution)])
                if len(covered) < 6:
                    continue
                yaw = math.radians(self.scan_pose[2])
                lx, ly = self.laser_transform[:2]
                laser = self.scan_pose[:2]+np.array((math.cos(yaw)*lx-math.sin(yaw)*ly,
                                                    math.sin(yaw)*lx+math.cos(yaw)*ly))-self.pose[:2]
                depth = face-float(np.dot(laser,normal))
                if depth <= resolution:
                    continue
                lateral_range = np.maximum(np.abs(covered[:-1]-np.dot(laser,axis)),
                                           np.abs(covered[1:]-np.dot(laser,axis)))
                # On a plane y=d*tan(theta), dy/dtheta=(d*d+y*y)/d.
                spacing = (depth*depth+lateral_range*lateral_range)/depth*abs(self.scan_msg.angle_increment)
                allowed = np.minimum(2*(p['footprint_half_width']+p['clearance_margin']),
                                     np.maximum(2*resolution,p['completion_gap_factor']*spacing))
                if np.any(np.diff(covered) > allowed) or np.any(np.diff(covered) >= 2*(p['footprint_half_width']+p['clearance_margin'])):
                    continue
                return np.array((np.dot(self.pose[:2],normal)+face,
                                 np.dot(self.pose[:2],axis)+sides[0],np.dot(self.pose[:2],axis)+sides[1]))
        return None

    def completion_command(self, now):
        '''Only the closed end after two actually crossed moving bands can finish.'''
        p = self.nav_settings
        if self.course_complete:
            return (0.0,0.0), 'COURSE_COMPLETE'
        passed = [z for z in self.motion_zones if z['passed']]
        if len(passed) < p['completion_required_bands']:
            return None
        if self.crossing_phase in ('CROSS','SETTLE'):
            return None
        normal, axis = passed[-1]['normal'], passed[-1]['axis']
        if any(np.dot(z['normal'],normal) < 0.95 for z in passed):
            return (0.0,0.0), 'COURSE_UNCONFIRMED'
        walls = self.terminal_walls(normal,axis)
        if walls is None:
            self.completion_since, self.completion_count, self.completion_anchor = None, 0, None
            target = self.pose[:2]+normal*p['look_ahead_distance']
            if not self.segment_clear(self.pose[:2],target):
                return (0.0,0.0), 'COURSE_UNCONFIRMED'
            error = math.atan2(normal[1],normal[0])-math.radians(self.pose[2])
            error = math.atan2(math.sin(error),math.cos(error))
            turn = 0.0 if abs(math.degrees(error)) <= p['heading_tolerance'] else self.heading_command(error)
            return (p['linear_speed'] if abs(math.degrees(error)) <= p['turn_in_place_angle'] else 0.0,turn), 'FINAL_APPROACH'
        if (np.max(np.abs(self.last_command)) > 1e-4 or np.max(np.abs(self.measured_motion)) > 0.01
                or self.footprint_clearance(self.scan_points,*self.pose[:2],math.radians(self.pose[2])) < p['clearance_margin']):
            self.completion_since, self.completion_count = None, 0
            return (0.0,0.0), 'COMPLETION_STOPPING'
        if self.completion_stamp != self.scan_stamp:
            self.completion_stamp = self.scan_stamp
            if self.completion_anchor is None or np.max(np.abs(walls-self.completion_anchor)) > p['map_resolution']:
                self.completion_anchor = walls
                self.completion_since, self.completion_count = self.scan_stamp, 1
            else:
                if self.completion_since is None:
                    self.completion_since = self.scan_stamp
                self.completion_count += 1
            if (self.completion_count >= p['completion_observations']
                    and self.scan_stamp-self.completion_since >= p['completion_hold_time']):
                self.course_complete = True
                self.get_logger().info('COURSE_COMPLETE: crossed moving bands and repeatedly observed a stable closed end')
                return (0.0,0.0), 'COURSE_COMPLETE'
        return (0.0,0.0), 'COMPLETION_OBSERVING'

    def navigation_command(self, now):
        '''Follow local goals; Stage 6 adds waiting and failed-goal cooldowns.'''
        p = self.nav_settings
        recover = getattr(self, 'active_stage', 5) >= 6
        if getattr(self, 'active_stage', 5) >= 7:
            completed = self.completion_command(now)
            if completed is not None:
                return completed
        if recover:
            moving = self.moving_obstacle_command(now)
            if moving is not None:
                return moving
        if recover and self.blocked_since is not None and now-self.blocked_since >= p['obstacle_wait_timeout']:
            was_backtracking = self.backtracking
            self.discard_goal('blockage persisted beyond wait timeout', now)
            self.backtracking = False
            self.backtrack_reverse = False
            if not was_backtracking:
                self.start_backtracking(now)
            return (0.0, 0.0), 'REPLAN'
        if self.navigation_stalled:
            return (0.0, 0.0), 'NO_PROGRESS'
        turn_radius = math.hypot(max(p['footprint_front'], p['footprint_rear']), p['footprint_half_width'])
        if self.goal is not None and not getattr(self, 'backtracking', False) and not self.waiting_for_clearance and self.goal_checked_stamp != self.scan_stamp:
            self.goal_checked_stamp = self.scan_stamp
            nearest = float(np.min(np.linalg.norm(self.scan_points-np.asarray(self.goal), axis=1))) if len(self.scan_points) else -math.inf
            goal_clearance = nearest-turn_radius
            required = turn_radius+p['clearance_margin']+p['turn_drift_margin']
            if nearest < required and not self.refine_goal():
                if recover:
                    moving_goal = any(not z['passed']
                                      and abs(np.dot(np.asarray(self.goal)-z['centre'],z['normal'])) <= required+p['crossing_padding']
                                      and z['low']-required <= np.dot(self.goal,z['axis']) <= z['high']+required
                                      for z in self.motion_zones)
                    if not moving_goal:
                        self.discard_goal('new observation invalidated the goal; replan from current pose',now)
                        return (0.0,0.0), 'REPLAN'
                    if not self.goal_blocked:
                        self.get_logger().info(f'GOAL_TURN_ROOM: nearest return {nearest:.3f}m from goal centre; required {required:.3f}m')
                    self.goal_blocked, self.goal_clear_scans = True, 0
                    if self.blocked_since is None:
                        self.blocked_since = now
                    self.progress_time = now
                    return (0.0, 0.0), 'GOAL_BLOCKED'
                self.get_logger().info(f'GOAL_INVALIDATED {self.goal}: fresh scan leaves {goal_clearance:.3f}m turning clearance')
                self.goal, self.route, self.carrot = None, [], None
                self.goal_anchor = None
                self.previous_heading_error = self.previous_control_time = None
                return (0.0, 0.0), 'GOAL_INVALIDATED'
            if recover and self.goal_blocked:
                self.goal_clear_scans += 1
                if self.goal_clear_scans >= p['resume_scans']:
                    self.goal_blocked, self.blocked_since = False, None
                    self.previous_heading_error = self.previous_control_time = None
                    self.progress_time = now
        if recover and self.goal_blocked:
            self.progress_time = now
            return (0.0, 0.0), 'GOAL_BLOCKED'
        turn_room = float(np.min(np.linalg.norm(self.scan_points-self.pose[:2], axis=1)))-turn_radius if len(self.scan_points) else -math.inf
        anchor = (getattr(self, 'goal_anchor', None) or self.goal)
        goal_distance = math.dist(self.pose[:2], self.goal) if self.goal is not None else math.inf
        anchor_distance = math.dist(self.pose[:2], anchor) if anchor is not None else math.inf
        if (goal_distance <= p['goal_tolerance'] and anchor_distance <= p['goal_tolerance']
                and (getattr(self, 'backtracking', False) or turn_room >= p['clearance_margin']+p['turn_drift_margin'])):
            if getattr(self, 'backtracking', False):
                self.breadcrumbs = self.breadcrumbs[:self.backtrack_index+1]
                self.backtracking = False
                self.backtrack_reverse = False
                self.goal, self.goal_anchor, self.route, self.carrot = None, None, [], None
                self.blocked_since = None
                self.get_logger().info('BACKTRACK_COMPLETE: stopped before replanning')
                return (0.0, 0.0), 'BACKTRACK_COMPLETE'
            self.get_logger().info(f'LOCAL_GOAL_REACHED {self.goal}')
            self.goals_reached += 1
            if recover and len(self.observation_grid) <= self.goal_map_size and self.goal_anchor is not None:
                self.failed_goals[self.world_cell(*self.goal_anchor)] = now+p['goal_cooldown']
                self.get_logger().info('GOAL_NO_NEW_SPACE: cool down this observation point')
            self.goal, self.route, self.carrot = None, [], None
            self.goal_anchor = None
            self.observation_start, self.last_plan_time = now, now
            return (0.0, 0.0), 'LOCAL_GOAL_REACHED'
        if self.goal is None:
            # Finish the ordinary deceleration before doing a potentially slower search.
            if np.max(np.abs(self.last_command)) > 1e-4 or abs(self.measured_motion[0]) > 0.01:
                return (0.0, 0.0), 'STOPPING_TO_PLAN'
            if recover and self.observation_start is not None:
                wait = p['motion_confirmation_wait'] if self.motion_target() is not None else p['planning_interval']
                if now-self.observation_start < wait:
                    return (0.0,0.0), 'OBSERVING_FOR_MOTION'
                self.observation_start = None
            if now-self.last_plan_time >= p['planning_interval']:
                self.move_2D()
                self.select_local_goal()
                self.last_plan_time = self.sensor_time()
                if recover and self.goal is None:
                    self.start_backtracking(now)
            return (0.0, 0.0), 'PLAN_READY' if self.goal is not None else 'NO_REACHABLE_FRONTIER'
        command, state = self.follow_carrot(now)
        if self.route_stalled(now):
            if recover:
                self.discard_goal('no route progress', now)
                self.start_backtracking(now)
                return (0.0, 0.0), 'REPLAN'
            self.navigation_stalled = True
            return (0.0, 0.0), 'NO_PROGRESS'
        return command, state

    def world_cell(self, x, y):
        '''CA2 world_to_grid convention, with an unbounded world origin.'''
        resolution = self.nav_settings['map_resolution']
        return math.floor(x/resolution), math.floor(y/resolution)

    def cell_world(self, cell):
        '''CA2 grid_to_world convention: positions are cell centres.'''
        resolution = self.nav_settings['map_resolution']
        return ((cell[0]+0.5)*resolution, (cell[1]+0.5)*resolution)

    def ray_cells(self, start, end):
        '''Integer ray traversal; do not create a wall at a no-return endpoint.'''
        x, y = start
        dx, dy = abs(end[0]-x), abs(end[1]-y)
        sx, sy = (1 if end[0] > x else -1), (1 if end[1] > y else -1)
        error = dx-dy
        while True:
            yield x, y
            if (x, y) == end:
                break
            twice = 2*error
            if twice > -dy:
                error -= dy
                x += sx
            if twice < dx:
                error += dx
                y += sy

    def update_memory(self):
        '''One evidence update per fresh scan; remember observed space and travel.'''
        if self.memory_stamp == self.scan_stamp:
            return
        p, pose = self.nav_settings, self.scan_pose
        yaw = math.radians(pose[2])
        lx, ly, mounting_yaw = self.laser_transform
        origin = (pose[0]+math.cos(yaw)*lx-math.sin(yaw)*ly,
                  pose[1]+math.sin(yaw)*lx+math.cos(yaw)*ly)
        start = self.world_cell(*origin)
        free, occupied = set(), set()
        msg = self.scan_msg
        for index, value in enumerate(msg.ranges):
            if math.isnan(value) or value < msg.range_min or value > msg.range_max and not math.isinf(value):
                continue
            distance = min(value, p['mapping_range'], msg.range_max)
            angle = yaw+mounting_yaw+msg.angle_min+index*msg.angle_increment
            end = self.world_cell(origin[0]+distance*math.cos(angle), origin[1]+distance*math.sin(angle))
            cells = list(self.ray_cells(start, end))
            hit = math.isfinite(value) and value < msg.range_max and value <= p['mapping_range']
            free.update(cells[:-1] if hit else cells)
            if hit:
                occupied.add(end)
        # The space physically occupied by the robot is already known, even in
        # the LiDAR's shadow. Do not mark an arbitrary surrounding disk as free.
        radius = p['footprint_front']+p['footprint_half_width']
        low, high = self.world_cell(pose[0]-radius, pose[1]-radius), self.world_cell(pose[0]+radius, pose[1]+radius)
        for x in range(low[0], high[0]+1):
            for y in range(low[1], high[1]+1):
                wx, wy = self.cell_world((x,y))
                bx = math.cos(yaw)*(wx-pose[0])+math.sin(yaw)*(wy-pose[1])
                by = -math.sin(yaw)*(wx-pose[0])+math.cos(yaw)*(wy-pose[1])
                if -p['footprint_rear'] <= bx <= p['footprint_front'] and abs(by) <= p['footprint_half_width']:
                    free.add((x,y))
        for cell in free-occupied:
            evidence = max(0, self.occupied_evidence.get(cell, 0)-1)
            if evidence:
                self.occupied_evidence[cell] = evidence
            else:
                self.occupied_evidence.pop(cell, None)
            self.observation_grid[cell] = 100 if evidence else 0
        for cell in occupied:
            self.occupied_evidence[cell] = 2
            self.observation_grid[cell] = 100
        if not getattr(self, 'backtracking', False) and (not self.breadcrumbs or math.dist(self.breadcrumbs[-1], self.pose[:2]) >= p['breadcrumb_spacing']):
            self.breadcrumbs.append(tuple(self.pose[:2]))
        self.memory_stamp = self.scan_stamp

    def footprint_clearance(self, points, x=0.0, y=0.0, yaw=0.0):
        '''Signed distance from obstacle points to the full rectangular footprint.'''
        if len(points) == 0:
            return math.inf
        p = self.nav_settings
        dx, dy = points[:, 0] - x, points[:, 1] - y
        bx = math.cos(yaw)*dx + math.sin(yaw)*dy
        by = -math.sin(yaw)*dx + math.cos(yaw)*dy
        cx = (p['footprint_front'] - p['footprint_rear']) / 2
        hx = (p['footprint_front'] + p['footprint_rear']) / 2
        qx, qy = np.abs(bx-cx)-hx, np.abs(by)-p['footprint_half_width']
        distances = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
        return float(np.min(distances))

    def braking_clearance(self, points, forward, turn):
        '''Sweep the footprint through reaction delay and a conservative full stop.'''
        p = self.nav_settings
        horizon = p['reaction_delay'] + max(abs(forward)/p['linear_braking'], abs(turn)/p['angular_braking'])
        steps = max(1, math.ceil(horizon / 0.025))
        dt = horizon / steps
        x = y = yaw = 0.0
        clearance = self.footprint_clearance(points)
        for i in range(steps):
            elapsed = (i + 0.5)*dt
            brake_time = max(0.0, elapsed-p['reaction_delay'])
            v = math.copysign(max(0.0, abs(forward)-p['linear_braking']*brake_time), forward)
            w = math.copysign(max(0.0, abs(turn)-p['angular_braking']*brake_time), turn)
            x += v*math.cos(yaw+w*dt/2)*dt
            y += v*math.sin(yaw+w*dt/2)*dt
            yaw += w*dt
            clearance = min(clearance, self.footprint_clearance(points, x, y, yaw))
        return clearance

    def safe_motion(self, forward, turn, dt):
        '''Safety overrides smoothing; resume needs several different clear scans.'''
        p = self.nav_settings
        yaw = math.radians(self.pose[2])
        delta = self.scan_points - self.pose[:2]
        points = np.column_stack((math.cos(yaw)*delta[:,0] + math.sin(yaw)*delta[:,1],
                                  -math.sin(yaw)*delta[:,0] + math.cos(yaw)*delta[:,1]))
        self.current_clearance = self.footprint_clearance(points)
        self.minimum_clearance = min(self.minimum_clearance, self.current_clearance)
        # Check both the proposed motion and residual measured/commanded motion.
        speed = max(abs(self.measured_motion[0]), abs(self.last_command[0]))
        residual_forward = self.measured_motion[0] if abs(self.measured_motion[0]) > abs(self.last_command[0]) else self.last_command[0]
        residual_turn = self.measured_motion[1] if abs(self.measured_motion[1]) > abs(self.last_command[1]) else self.last_command[1]
        clearance = min(self.braking_clearance(points, forward, turn),
                        self.braking_clearance(points, residual_forward, residual_turn))
        closing = 0.0
        fresh_scan = self.safety_scan_stamp != self.scan_stamp
        if fresh_scan:
            if self.clearance_history is not None:
                stamp, previous = self.clearance_history
                if 0 < self.scan_stamp-stamp < 0.5:
                    closing = max(0.0, (previous-self.current_clearance)/(self.scan_stamp-stamp)-speed)
            self.clearance_history = (self.scan_stamp, self.current_clearance)
            self.safety_scan_stamp = self.scan_stamp
        extra = closing*(p['reaction_delay'] + speed/p['linear_braking'])
        if clearance < p['clearance_margin'] + extra:
            self.waiting_for_clearance = True
            self.clear_scan_count = 0
        elif self.waiting_for_clearance and fresh_scan:
            if clearance >= p['clearance_margin'] + p['resume_margin'] + extra:
                self.clear_scan_count += 1
            else:
                self.clear_scan_count = 0
            if self.clear_scan_count >= p['resume_scans']:
                self.waiting_for_clearance = False
        if self.waiting_for_clearance:
            return np.zeros(2), 'BRAKE_WAIT'
        target = np.array((forward, turn))
        accelerating = np.abs(target) > np.abs(self.last_command)
        acceleration = p['crossing_acceleration'] if getattr(self, 'crossing_phase', None) == 'CROSS' else p['linear_acceleration']
        limits = np.where(accelerating, (acceleration, p['angular_acceleration']),
                          (p['linear_braking'], p['angular_braking']))
        command = self.last_command + np.clip(target-self.last_command, -limits*dt, limits*dt)
        return command, 'MOVING' if np.any(np.abs(command) > 1e-6) else 'STOPPED'

    def primitive_command(self):
        '''Small repeatable motion tests; deliberately no course navigation yet.'''
        p = self.nav_settings
        if self.test_primitive == 'stopped' or self.motion_done:
            return 0.0, 0.0
        if self.motion_start is None:
            self.motion_start = self.pose.copy()
        if self.test_primitive in ('forward', 'backward'):
            remaining = p['forward_test_distance'] - np.linalg.norm(self.pose[:2]-self.motion_start[:2])
            self.motion_done = remaining <= p['position_tolerance']
            heading_error = math.radians(wrap_deg(self.motion_start[2]-self.pose[2]))
            direction = 1 if self.test_primitive == 'forward' else -1
            return (0.0, 0.0) if self.motion_done else (direction*min(p['linear_speed'], remaining), float(np.clip(2*heading_error, -p['angular_speed'], p['angular_speed'])))
        sign = 1 if self.test_primitive == 'turn_left' else -1
        error = wrap_deg(self.motion_start[2] + sign*p['turn_test_angle'] - self.pose[2])
        self.motion_done = abs(error) <= p['heading_tolerance']
        turn = math.copysign(min(p['angular_speed'], max(p['minimum_turn_speed'], 2*abs(math.radians(error)))), error)
        return (0.0, 0.0) if self.motion_done else (0.0, turn)

    def motion_stage(self):
        '''Check sensors and safety on every tick, then publish through move_2D.'''
        if not hasattr(self, 'nav_settings'):
            self.init_navigation()
        if self.stop_requested:
            self.move_2D()
            raise SystemExit
        state = self.check_sensors()
        now = self.sensor_time()
        dt = min(0.1, max(0.0, now-self.last_motion_time))
        self.last_motion_time = now
        self.sensor_ready = state == 'READY_STOPPED'
        if self.sensor_ready and self.active_stage >= 3:
            self.update_memory()
        if self.sensor_ready and self.active_stage >= 6:
            self.observe_motion()
        if self.sensor_ready and self.active_stage >= 4 and self.test_primitive != 'navigate' and self.goal is None and now-self.last_plan_time >= self.nav_settings['planning_interval']:
            self.move_2D() # Planning runs while stopped, before enabling any follower.
            self.select_local_goal()
            self.last_plan_time = self.sensor_time()
        if self.sensor_ready and not self.calibrate:
            if self.active_stage >= 5 and self.test_primitive == 'navigate':
                desired, navigation_state = self.navigation_command(now)
            else:
                desired, navigation_state = self.primitive_command(), None
            command, state = self.safe_motion(*desired, dt)
            if navigation_state is not None and state != 'BRAKE_WAIT':
                state = navigation_state
            if self.motion_done and state == 'STOPPED':
                state = 'PRIMITIVE_DONE'
            if self.active_stage >= 4 and self.test_primitive != 'navigate':
                command = np.zeros(2)
                state = 'PLAN_READY_STOPPED' if self.goal is not None else 'NO_REACHABLE_FRONTIER'
        else:
            command = np.zeros(2)
            self.completion_since, self.completion_count, self.completion_anchor = None, 0, None
            if self.crossing_zone is not None:
                self.crossing_zone['clear_since'], self.crossing_zone['was_clear'], self.crossing_zone['windows'] = None, None, []
                if self.crossing_phase == 'CROSS':
                    self.crossing_phase = 'SENSOR_HOLD'
        if self.stop_requested:
            self.move_2D()
            raise SystemExit
        self.last_command = command
        self.move_2D(command[0], 0.0, command[1])
        if self.active_stage >= 6 and state in ('BRAKE_WAIT', 'ROUTE_BLOCKED', 'NO_ROUTE') and self.blocked_since is None:
            self.blocked_since = now
        elif self.active_stage >= 6 and state in ('FOLLOWING','SWING_APPROACH','SWING_ALIGNING','SWING_CROSSING') and np.any(np.abs(command) > 1e-4):
            self.blocked_since = None
        if state != self.sensor_state or now - self.last_status >= self.nav_settings['status_interval']:
            scan_age = now - self.scan_received if self.scan_received is not None else math.inf
            pose_age = now - self.pose_received if self.pose_received is not None else math.inf
            self.get_logger().info(
                f'{state} | pose={self.pose} | scan_age={scan_age:.3f}s '
                f'pose_age={pose_age:.3f}s | clearance={self.current_clearance:.3f}m '
                f'min={self.minimum_clearance:.3f}m | cells={len(self.observation_grid)} '
                f'breadcrumbs={len(self.breadcrumbs)} goals={self.goals_reached}/{self.goals_selected} '
                f'progress={self.path_progress:.2f}m | cmd=({command[0]:.3f}, 0, {command[1]:.3f})')
            self.sensor_state, self.last_status = state, now
        self.show_diagnostics(state)

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######
        self.motion_stage()
        ###### INSERT CODE HERE ######
                

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
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
