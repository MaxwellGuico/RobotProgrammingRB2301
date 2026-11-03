import argparse
import configparser
import math
import os
import signal
from collections import deque
from time import monotonic
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from rb2301_fp.diagnostics import publish_diagnostics

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  


_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    """Keep angles between -180 and 180 degrees so the robot takes the shorter turn."""
    return (angle + 180.0) % 360.0 - 180.0

def update_clear_window(zone, clear, side, stamp):
    """Time a complete blocked-clear-blocked opening and remember which side was clear.
    The first partial opening is ignored because we did not see when it started.
    """
    # Scan comparisons may return NumPy booleans; use a normal bool for the state transitions.
    clear = bool(clear)
    duration = None
    if clear and zone['was_clear'] is False:
        zone['clear_since'], zone['clear_side'] = stamp, side
    elif not clear and zone['was_clear']:
        if zone['clear_since'] is not None:
            duration = stamp-zone['clear_since']
            zone['windows'].append((zone['clear_side'], duration))
        zone['clear_since'] = None
    zone['was_clear'] = clear
    return duration


def clear_window_remaining(zone, stamp, observations):
    """Estimate how much of this opening is left using the last two matching openings.
    We keep the two sides separate because one side can stay clear much longer.
    """
    if zone['clear_since'] is None:
        return 0.0
    durations = [duration for side, duration in zone['windows'] if side == zone.get('clear_side')]
    if len(durations) < observations:
        return 0.0
    # Use the shorter measured opening, then subtract the time already spent in this one.
    return min(durations[-observations:])-(stamp-zone['clear_since'])


def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    """Subtract the measured origin and rotate Optitrack into the calibrated maze axes.
    The physical Y flip is applied by both pose callbacks after this conversion.
    """
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    """Optionally express a maze pose relative to the final-course origin and heading."""
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    """Read the measured Optitrack alignment and any heading correction for this robot."""
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

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=False, calibrate:bool=False):
        """Set up parameters, sensor subscriptions and navigation memory before starting the timer."""
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        # Choose the sensor source once at startup. Switching it later would need new subscriptions.
        self.is_simulation = self.declare_parameter(
            'is_simulation', is_simulation,
            ParameterDescriptor(
                read_only=True,
                description='Use Gazebo odometry instead of physical robot Optitrack; set at startup.'),
        ).value
        self.get_logger().info(f'Robot mode: {"simulation" if self.is_simulation else "hardware"}')
        # Gazebo's laser_joint is forward of base_footprint and faces backwards.
        # Physical mounting calibration can be supplied through the same parameters.
        mount = (0.11423, 0.0, 3.1416) if self.is_simulation else (0.0, 0.0, 0.0)
        self.laser_transform = tuple(
            self.declare_parameter(name, value, ParameterDescriptor(
                read_only=True, description=f'LiDAR mounting {axis} relative to the pose reference; set at startup.'
            )).value
            for name, value, axis in zip(
                ('laser_x', 'laser_y', 'laser_yaw'), mount, ('x [m]', 'y [m]', 'yaw [rad]')))
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        if self.is_simulation:
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

        self.pose = None

        # Tuning lives here; the driving functions below use these same values in both modes.
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
            "reaction_delay": 0.20,
            "clearance_margin": 0.0, # passage acceptance uses the physical width only
            "heading_tolerance": 2.0, # degrees
            "map_resolution": 0.05, # metres per observed cell
            "mapping_range": 2.5, # cap observation distance, not the course dimensions
            "frontier_standoff": 0.25, # keep the approach inside observed space
            "approach_search_distance": 0.75, # search farther inward for room to turn
            "minimum_frontier_size": 5, # ignore isolated gaps after diagonal grouping
            "minimum_goal_distance": 0.40,
            "crossing_approach_spacing": 0.20, # separate crossing lines after an unsuccessful attempt
            "prefer_motion_frontiers": True, # forward preference from observed motion
            "preferred_clearance": 0.30, # prefer room to turn on planned routes
            "planning_interval": 2.0, # seconds between retries while stopped
            "look_ahead_distance": 0.25, # metres along the planned route
            "goal_tolerance": 0.08,
            "heading_gain": 2.0,
            "heading_derivative_gain": 0.05,
            "distance_gain": 1.0,
            "turn_in_place_angle": 8.0, # align before translating through narrow gaps
            "progress_distance": 0.04,
            "progress_timeout": 25.0, # replan when a route stops progressing
            "obstacle_wait_timeout": 15.0, # reconsider a persistent blockage
            "goal_cooldown": 30.0, # temporarily skip a failed approach region
            "motion_observation_interval": 0.5,
            "motion_range_change": 0.12, # metres; reject small range noise
            "motion_confirmation_samples": 3,
            "motion_confirmation_wait": 24.0, # stopped observation before abandoning a motion hint
            "motion_minimum_span": 0.30, # metres of repeatedly changing returns
            "crossing_speed": 0.30, # commanded crossing speed; hardware retains its 0.3 m/s cap
            "crossing_acceleration": 0.80, # crossing ramp; measure on hardware
            "crossing_padding": 0.05, # allowance beyond observed moving returns
            "crossing_time_margin": 0.25, # seconds beyond reaction/acceleration time
            "crossing_observed_windows": 2,
            "crossing_observation_timeout": 50.0, # allow two complete same-side cycles
            "crossing_maximum_approaches": 2, # try another observed staging line, then report failure
            "completion_required_bands": 2,
            "completion_wall_distance": 0.65, # metres from pose origin to front wall
            "completion_minimum_wall_span": 0.20,
            "completion_hold_time": 6.0, # seconds of stable closed-end observations
            "completion_observations": 30, # different fresh scans, while stopped
            "completion_gap_factor": 1.5, # allowance over angular ray spacing
        }
        # Message timestamps align the sensors. Local receipt times tell us if data has stopped arriving.
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
        # LiDAR mount relative to robot pose reference: x [m], y [m], yaw [rad].
        # Measure these on the physical robot before enabling movement.
        self.last_command = np.zeros(2) # forward m/s, yaw rad/s; lateral is always zero
        self.measured_motion = np.zeros(2)
        self.nav_settings['debug_visuals'] = os.environ.get('RB2301_DEBUG_VISUALS', '1') != '0'
        # Missing cells are unknown, so the planner cannot invent a route through unseen space.
        self.observation_grid = {} # (world cell x, y) -> 0 free or 100 occupied; absent is unknown
        self.occupied_evidence = {}
        self.memory_stamp = None
        self.frontier_groups = []
        self.goal = None
        self.route = []
        self.free_cells = set()
        self.goal_cells = set()
        self.cell_clearance = {}
        self.last_plan_time = -math.inf
        self.goals_selected = 0
        self.goals_reached = 0
        self.failed_goals = {}
        self.blocked_since = None
        self.goal_map_size = 0
        self.observation_start = None
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
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.full_scan_sub = self.create_subscription(LaserScan, 'scan', self.remember_scan, qos)
        if self.is_simulation:
            self.full_pose_sub = self.create_subscription(Odometry, 'odom', self.remember_pose, qos)
        else:
            self.full_pose_sub = self.create_subscription(
                PoseStamped, f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose', self.remember_pose, qos)
        self.get_logger().info(f'Full-course navigation; Vy=0; mode={"simulation" if self.is_simulation else "hardware"}.')

        # Start the 20 Hz loop only after every callback has the state it needs.
        self.timer = self.create_timer(0.05, self.timer_callback)

    def sub_scan_callback(self, msg):
        """Keep the latest ranges for the original robot interface; mapping uses the full scan."""
        if self.is_simulation:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2]

    def yaw_from_quaternion(self, q):
        """Extract the robot heading in radians from the ROS orientation quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        """Update the physical robot pose, including its calibrated Y-axis convention."""
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        # Preserve the physical Optitrack-to-world Y convention in both callbacks.
        self.pose = np.array((x,-y,heading))
        return self.pose

    def to_course(self, pose=None):
        """Return the pose in the optional course frame without changing the stored pose."""
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        """Update the simulated robot position and heading from Gazebo odometry."""
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        """Publish forward speed and yaw rate, capped at the robot limits. Sideways speed is always zero."""
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        # Even if another caller supplies y, the robot still receives Vy=0.
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), 0.0, 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def stop_navigation(self, signum=None, frame=None):
        """Ask the next timer update to publish zero velocity and end the controller."""
        self.stop_requested = True

    def remember_scan(self, msg):
        """Keep each new stamped scan once; clear old map data if the simulation clock resets."""
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
            self.memory_stamp = None
            self.goal, self.route = None, []

    def remember_pose(self, msg):
        """Keep recent stamped poses so a scan can use the pose from when it was measured.
        Also estimate actual forward and turning speed so we can tell when the robot has stopped.
        """
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.pose_stamp is not None and stamp < self.pose_stamp:
            self.pose_history.clear()
            self.pose_stamp = None
        if self.pose_stamp is None or stamp > self.pose_stamp:
            p = msg.pose.pose if self.is_simulation else msg.pose
            heading = math.degrees(self.yaw_from_quaternion(p.orientation))
            pose = (p.position.x, p.position.y, heading)
            if not self.is_simulation:
                x, y, heading = optitrack_to_maze(*pose, self.frame)
                # Keep this identical to optitrack_callback(): the physical frame needs the Y flip.
                pose = (x, -y, heading)
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
        """Convert valid LiDAR hits into world points: apply the mount first, then the robot pose.
        Infinity and invalid ranges are skipped because they do not describe an obstacle hit.
        """
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid] + laser_transform[2]
        # The LiDAR can be offset and face backwards; applying the mount first puts hits in robot coordinates.
        base_x = laser_transform[0] + ranges[valid] * np.cos(angles)
        base_y = laser_transform[1] + ranges[valid] * np.sin(angles)
        yaw = math.radians(pose[2])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.column_stack((pose[0] + c * base_x - s * base_y,
                                pose[1] + s * base_x + c * base_y))

    def prepare_scan(self):
        """Prepare a scan only when both sensors are fresh and their timestamps agree.
        The fixed mounting parameters are used directly; this does not look up a TF tree.
        """
        msg, now = self.scan_msg, self.sensor_time()
        if msg is None or not self.pose_history:
            return 'WAITING_FOR_SENSORS'
        if now-self.scan_received > self.nav_settings['scan_timeout']:
            return 'STALE_SCAN'
        if now-self.pose_received > self.nav_settings['pose_timeout']:
            return 'STALE_POSE'
        if (not msg.ranges or msg.angle_increment == 0
                or not np.all(np.isfinite((msg.angle_min,msg.angle_increment,msg.range_min,msg.range_max)))
                or not 0 <= msg.range_min < msg.range_max):
            return 'INVALID_SCAN_METADATA'
        # Use the pose nearest the scan time, rather than the robot's later position.
        stamp, pose = min(self.pose_history, key=lambda item:abs(item[0]-self.scan_stamp))
        if abs(stamp-self.scan_stamp) > self.nav_settings['pose_scan_tolerance']:
            return 'POSE_SCAN_TIME_MISMATCH'
        if self.projected_stamp != self.scan_stamp:
            points = self.project_scan(msg,pose,self.laser_transform)
            if not len(points):
                return 'NO_OBSTACLE_RETURNS'
            self.scan_pose, self.scan_points, self.projected_stamp = pose, points, self.scan_stamp
        return 'READY_STOPPED'

    def show_diagnostics(self, state):
        """Send the current state to RViz; these display messages never feed back into driving."""
        publish_diagnostics(self, state)

    def neighbours(self, cell, diagonals=False):
        """Use four neighbours for paths, or eight when joining touching frontier cells."""
        x, y = cell
        if diagonals:
            return ((x+1,y), (x-1,y), (x,y+1), (x,y-1),
                    (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1))
        return ((x+1,y), (x-1,y), (x,y+1), (x,y-1))

    def planning_space(self):
        """Find observed free cells that fit half the robot width.
        Keep their obstacle distances too, so A* can prefer the middle of wider passages.
        """
        p = self.nav_settings
        resolution = p['map_resolution']
        free = {cell for cell, value in self.observation_grid.items() if value == 0}
        clearance_cap = p['preferred_clearance']
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
        self.cell_clearance = clearance
        # This is the passage test: the centre must fit half the robot width on either side.
        self.free_cells = {cell for cell in free if clearance[cell] >= p['footprint_half_width']}
        self.goal_cells = self.free_cells.copy()
        now = self.sensor_time()
        self.failed_goals = {cell:expiry for cell,expiry in getattr(self, 'failed_goals', {}).items() if expiry > now}
        skip_radius = 2*p['footprint_half_width']+p['clearance_margin']
        cells = math.ceil(skip_radius/resolution)
        for x,y in self.failed_goals:
            self.goal_cells.difference_update((x+dx,y+dy) for dx in range(-cells,cells+1) for dy in range(-cells,cells+1)
                                              if math.hypot(dx,dy)*resolution <= skip_radius)

    def reconstruct_path(self, parents, current):
        """Follow A* parent links backwards from the goal, then put the path in travel order."""
        path = [current]
        while current in parents:
            current = parents[current]
            path.append(current)
        return path[::-1]

    def Astar(self, start, goal):
        """Find a route through known free cells, with a small cost for travelling close to obstacles."""
        import heapq
        if start not in self.free_cells or self.observation_grid.get(start) != 0:
            return []
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
                # Narrow cells stay usable; they just cost more than the middle of a wide passage.
                step = 1+max(0.0, self.nav_settings['preferred_clearance']-self.cell_clearance[cell])/self.nav_settings['map_resolution']
                candidate = cost+step
                if candidate < costs.get(cell, math.inf):
                    costs[cell], parents[cell] = candidate, current
                    heapq.heappush(queue, (candidate+heuristic(cell), candidate, cell))
        return []

    def simplify_path(self, path):
        """Remove points on straight runs while keeping every corner of the grid path."""
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
        """Choose an observation point beside a reachable frontier.
        Prefer the direction of observed motion, then the largest frontier group, then the shorter approach.
        """
        self.planning_space()
        free = {cell for cell, value in self.observation_grid.items() if value == 0}
        # A frontier is free space touching something we have not observed yet.
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
        # Flood the usable map first so unreachable frontier approaches are never scored as goals.
        distances, queue = {start:0}, deque([start])
        while queue:
            current = queue.popleft()
            for cell in self.neighbours(current):
                if cell in self.free_cells and cell not in distances:
                    distances[cell] = distances[current]+1
                    queue.append(cell)
        p, candidates = self.nav_settings, []
        target = self.motion_target() if p['prefer_motion_frontiers'] else None
        search_distance = max(p['frontier_standoff'], p['approach_search_distance'])
        radius = math.ceil(search_distance/p['map_resolution'])
        offsets = [(dx,dy,math.hypot(dx,dy)*p['map_resolution'])
                   for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)
                   if p['frontier_standoff'] <= math.hypot(dx,dy)*p['map_resolution'] <= search_distance]
        for group in groups:
            # A passage is usable when its free half-width fits the robot.
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
                # Sorting this tuple applies the priorities in order: motion direction, size, distance.
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
        """Start following a new route and reset its progress counters, keeping the map and motion observations."""
        self.route, self.goal = route, route[-1]
        self.carrot = None
        self.blocked_since = None
        self.goal_map_size = len(self.observation_grid)
        self.path_progress, self.current_segment_idx = 0.0, 0
        self.progress_time, self.progress_mark = now, 0.0
        self.previous_heading_error = self.previous_control_time = None
        self.observation_start, self.navigation_stalled = None, False

    def route_stalled(self, now):
        """Report a route that has not gained enough distance along the path within the timeout."""
        if self.path_progress >= self.progress_mark+self.nav_settings['progress_distance']:
            self.progress_mark, self.progress_time = self.path_progress, now
            self.blocked_since = None
        return now-self.progress_time > self.nav_settings['progress_timeout']

    def get_carrot(self, robot_position, path, look_ahead_dist):
        """Project the robot onto its current route segment and look a little farther along the path.
        Progress only moves forwards, so a nearby parallel leg cannot steal the carrot.
        """
        if len(path) < 2:
            return None
        points = np.asarray(path)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        while True:
            index = self.current_segment_idx
            a, b = points[index], points[index+1]
            ab = b-a
            # Project onto this segment; clamping keeps the projection between its endpoints.
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
        # Walk the look-ahead distance along the route, including any corners ahead.
        distance = min(self.path_progress+look_ahead_dist, cumulative[-1])
        segment = min(len(lengths)-1, int(np.searchsorted(cumulative, distance, side='right')-1))
        alpha = (distance-cumulative[segment])/max(lengths[segment], 1e-12)
        return points[segment]+alpha*(points[segment+1]-points[segment])

    def segment_clear(self, start, end):
        """Sample a centre line and require half the robot width between it and every visible obstacle."""
        if not len(self.scan_points):
            return False
        count = max(2, math.ceil(math.dist(start, end)/self.nav_settings['map_resolution'])+1)
        return all(np.min(np.linalg.norm(self.scan_points-point, axis=1)) >= self.nav_settings['footprint_half_width']
                   for point in np.linspace(start, end, count))

    def follow_carrot(self, now):
        """Turn towards the carrot, then drive forwards once the heading is close enough."""
        p = self.nav_settings
        lookahead = p['look_ahead_distance']
        self.carrot = self.get_carrot(self.pose[:2], self.route, lookahead)
        # Keep the carrot in observed space; shorten the look-ahead near corners.
        while self.carrot is not None and any(
                self.observation_grid.get(cell) != 0
                for cell in self.ray_cells(self.world_cell(*self.pose[:2]),self.world_cell(*self.carrot))):
            lookahead /= 2
            if lookahead < 0.025:
                self.previous_heading_error = self.previous_control_time = None
                return (0.0, 0.0), 'ROUTE_BLOCKED'
            self.carrot = self.get_carrot(self.pose[:2], self.route, lookahead)
        if self.carrot is None:
            return (0.0, 0.0), 'NO_ROUTE'
        error = self.carrot-self.pose[:2]
        desired = math.atan2(error[1], error[0])
        heading_error = math.atan2(math.sin(desired-math.radians(self.pose[2])), math.cos(desired-math.radians(self.pose[2])))
        turn = self.heading_command(heading_error,now)
        speed = min(p['linear_speed'], p['distance_gain']*float(np.linalg.norm(error)), math.dist(self.pose[:2], self.goal))
        speed *= max(0.0, math.cos(heading_error))
        # Face the route before translating. The follower never selects reverse motion.
        if abs(math.degrees(heading_error)) > p['turn_in_place_angle']:
            speed = 0.0
        return (speed, turn), 'FOLLOWING'

    def heading_command(self, error, now=None):
        """Turn in proportion to the heading error, with a small damping term and a speed limit.
        The minimum turn rate helps the simulated robot overcome its low-speed turning deadband.
        """
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

    def discard_goal(self, reason, now):
        """Temporarily exclude an unsuccessful goal and let the map planner choose another one."""
        if self.goal is not None:
            self.failed_goals[self.world_cell(*self.goal)] = now+self.nav_settings['goal_cooldown']
        self.get_logger().info(f'RECOVERY_REPLAN: {reason}')
        self.goal, self.route, self.carrot = None, [], None
        self.navigation_stalled = False
        self.blocked_since = None
        # A navigation goal change must not erase safety hysteresis or motion evidence.
        self.previous_heading_error = self.previous_control_time = None
        self.progress_time = now

    def observe_motion(self):
        """Look for repeated range changes while the robot is nearly stationary.
        Group those changes into a thin swept band instead of tracking an individual obstacle velocity.
        """
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
            # The direction with the greatest spread is the obstacle's observed sweep direction.
            _, vectors = np.linalg.eigh(np.cov(cloud.T))
            axis = vectors[:,-1]
            normal = np.array((axis[1], -axis[0])) # perpendicular to the sweep: our crossing direction
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
            match.setdefault('cells',set()).update(group)
            cloud = np.array([self.cell_world(c) for c in match['cells']])
            # Refine from all observed motion before an approach starts. Freeze the
            # crossing frame while following it, so its goal and timing stay stable.
            if match is not getattr(self,'crossing_zone',None) and not match['passed']:
                _,vectors = np.linalg.eigh(np.cov(cloud.T))
                axis = vectors[:,-1]
                if np.dot(axis,match['axis']) < 0:
                    axis = -axis
                normal = np.array((axis[1],-axis[0]))
                if np.dot(normal,match['normal']) < 0:
                    normal = -normal
                match['centre'],match['axis'],match['normal'] = cloud.mean(axis=0),axis,normal
            match['low'],match['high'] = float(np.min(cloud@match['axis'])),float(np.max(cloud@match['axis']))

    def motion_target(self):
        """Point exploration towards the nearest uncrossed moving band, or recent unconfirmed changes."""
        remaining = [z for z in self.motion_zones if not z['passed']]
        if remaining:
            return min(remaining,key=lambda z:math.dist(self.pose[:2],z['centre']))['centre']
        recent = [self.cell_world(c) for c,(_,_,last) in self.motion_evidence.items()
                  if self.scan_stamp-last <= self.nav_settings['motion_confirmation_wait']
                  and not any(z['passed'] and abs(np.dot(np.asarray(self.cell_world(c))-z['centre'],z['normal'])) <= self.nav_settings['crossing_padding']
                              for z in self.motion_zones)]
        return np.mean(recent,axis=0) if len(recent) >= self.nav_settings['motion_confirmation_samples'] else None

    def moving_band_points(self, zone):
        """Pick current scan hits inside a learned moving band; an empty result means it is not visible."""
        p = self.nav_settings
        normal, along = (self.scan_points-zone['centre'])@zone['normal'], self.scan_points@zone['axis']
        mask = ((np.abs(normal) <= p['crossing_padding'])
                & (along >= zone['low']-p['map_resolution']) & (along <= zone['high']+p['map_resolution']))
        return self.scan_points[mask]

    def moving_obstacle_command(self, now):
        """Approach a left-side staging point, align, observe two cycles, cross, then stop.
        This takes priority over frontier exploration while a crossing is in progress.
        """
        p = self.nav_settings
        if getattr(self,'navigation_stalled',False):
            return (0.0,0.0), 'NO_PROGRESS'
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
            stand_off = p['footprint_front']+p['clearance_margin']+p['crossing_padding']
            candidates = [c for c in self.free_cells
                          if stand_off <= np.dot(zone['centre']-self.cell_world(c),zone['normal']) <= stand_off+p['map_resolution']
                          and zone['low']-p['motion_minimum_span'] <= np.dot(self.cell_world(c),zone['axis']) <= zone['high']+p['motion_minimum_span']
                          and all(abs(np.dot(self.cell_world(c),zone['axis'])-lateral) >= p['crossing_approach_spacing']
                                  for lateral in zone.get('failed_laterals',[]))
                          and self.cell_clearance[c] >= p['footprint_half_width']]
            if not candidates:
                return None # Continue frontier exploration until a staging route is known.
            # Stage on the left, inset by one half-width from the outermost
            # usable centre cell. This follows observed space, not world axes.
            left = np.array((-zone['normal'][1], zone['normal'][0]))
            lane = max(np.dot(self.cell_world(c),left) for c in candidates)-p['footprint_half_width']
            for cell in sorted(candidates,key=lambda c:(abs(np.dot(self.cell_world(c),left)-lane),math.dist(self.cell_world(c),self.pose[:2]))):
                cells = self.Astar(self.world_cell(*self.pose[:2]),cell)
                if cells:
                    self.begin_route([tuple(self.pose[:2])]+[self.cell_world(c) for c in self.simplify_path(cells)[1:]],now)
                    zone['lateral'] = float(np.dot(self.goal,zone['axis']))
                    self.crossing_zone, self.crossing_phase = zone, 'APPROACH'
                    self.get_logger().info(f'SWING_APPROACH: left-lane staging goal {self.goal}; observe two cycles')
                    break
            if self.crossing_zone is None:
                return None
        if self.crossing_phase == 'APPROACH':
            if (self.blocked_since is not None and now-self.blocked_since >= p['obstacle_wait_timeout']) or self.route_stalled(now):
                zone['retry_after'] = now+p['goal_cooldown']
                self.crossing_zone, self.crossing_phase = None, None
                self.discard_goal('moving-band approach remained blocked',now)
                return (0.0,0.0), 'SWING_RETREAT'
            if math.dist(self.pose[:2],self.goal) > p['goal_tolerance']:
                command, state = self.follow_carrot(now)
                return command, 'SWING_APPROACH' if state == 'FOLLOWING' else state
            self.crossing_phase = 'OBSERVE'
            zone['clear_since'], zone['was_clear'], zone['windows'] = None, None, []
            zone['aligned_lateral'] = False
            zone['observe_started'] = None
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
                self.goal, self.route, self.carrot = None, [], None
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
            zone['observe_started'] = now
        if zone.get('window_stamp') != self.scan_stamp:
            zone['window_stamp'] = self.scan_stamp
            points = self.moving_band_points(zone)
            offsets = points@zone['axis']-zone['lateral']
            clear = len(points) >= 2 and np.min(np.abs(offsets)) >= p['footprint_half_width']
            side = int(np.sign(offsets[np.argmin(np.abs(offsets))])) if len(points) else 0
            duration = update_clear_window(zone, clear, side, self.scan_stamp)
            if duration is not None:
                cycles = sum(s == zone['clear_side'] for s, _ in zone['windows'])
                self.get_logger().info(f'SWING_CLEAR_WINDOW: side={zone["clear_side"]} cycles={cycles}/2 duration={duration:.2f}s')
        # Travel time to clear the band, plus an acceleration estimate, response delay and time allowance.
        required = remaining/p['crossing_speed']+p['crossing_speed']/(2*p['crossing_acceleration'])+p['reaction_delay']+p['crossing_time_margin']
        available = clear_window_remaining(zone, self.scan_stamp, p['crossing_observed_windows'])
        if available >= required:
            self.crossing_phase = 'CROSS'
            self.blocked_since = None
            self.get_logger().info(f'SWING_CROSSING: window remaining={available:.2f}s required={required:.2f}s')
            return (p['crossing_speed'],0.0), 'SWING_CROSSING'
        insufficient = (len(zone['windows']) >= 2*p['crossing_observed_windows']
                        and max(duration for _, duration in zone['windows']) < required)
        if zone.get('observe_started') is None:
            zone['observe_started'] = now
        timed_out = now-zone['observe_started'] >= p['crossing_observation_timeout']
        if insufficient or timed_out:
            zone.setdefault('failed_laterals',[]).append(zone['lateral'])
            self.get_logger().warning(f'SWING_REPOSITION: observed windows {zone["windows"][-4:]}; required {required:.2f}s')
            if len(zone['failed_laterals']) >= p['crossing_maximum_approaches']:
                self.navigation_stalled = True
                self.get_logger().error('NO_SAFE_CROSSING: approach attempts exhausted; course completion remains unconfirmed')
                return (0.0,0.0), 'NO_PROGRESS'
            self.crossing_zone,self.crossing_phase = None,None
            self.discard_goal('learn another crossing line after insufficient clear intervals',now)
            return (0.0,0.0), 'SWING_REPLAN'
        return (0.0,0.0), 'SWING_OBSERVING'

    def terminal_walls(self, normal, axis):
        """Look for a front wall that joins the two corridor sides without a robot-sized opening.
        This checks the end-wall shape from the current scan, not a stored finish coordinate.
        """
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
                # Rays are farther apart near the side of the view; allow their expected sampling gaps.
                spacing = (depth*depth+lateral_range*lateral_range)/depth*abs(self.scan_msg.angle_increment)
                allowed = np.minimum(2*(p['footprint_half_width']+p['clearance_margin']),
                                     np.maximum(2*resolution,p['completion_gap_factor']*spacing))
                if np.any(np.diff(covered) > allowed) or np.any(np.diff(covered) >= 2*(p['footprint_half_width']+p['clearance_margin'])):
                    continue
                return np.array((np.dot(self.pose[:2],normal)+face,
                                 np.dot(self.pose[:2],axis)+sides[0],np.dot(self.pose[:2],axis)+sides[1]))
        return None

    def completion_command(self, now):
        """Confirm the course-specific finish: two crossed bands, then a stable closed end.
        The robot must be stopped while collecting the final observations.
        """
        p = self.nav_settings
        if self.course_complete:
            return (0.0,0.0), 'COURSE_COMPLETE'
        passed = [z for z in self.motion_zones if z['passed']]
        # This course is expected to end after two moving bands. An empty frontier alone is not a finish.
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
        if np.max(np.abs(self.last_command)) > 1e-4 or np.max(np.abs(self.measured_motion)) > 0.01:
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
        """Choose what to do next: finish confirmation, a moving-obstacle crossing, or a frontier route."""
        p = self.nav_settings
        # Completion and a timed crossing take priority over choosing another frontier.
        completed = self.completion_command(now)
        if completed is not None:
            return completed
        moving = self.moving_obstacle_command(now)
        if moving is not None:
            return moving
        if self.navigation_stalled:
            return (0.0, 0.0), 'NO_PROGRESS'
        if self.blocked_since is not None and now-self.blocked_since >= p['obstacle_wait_timeout']:
            self.discard_goal('blockage persisted; choose another map route', now)
            return (0.0, 0.0), 'REPLAN'
        if self.goal is not None:
            nearest = np.min(np.linalg.norm(self.scan_points-np.asarray(self.goal), axis=1))
            if nearest < p['footprint_half_width']:
                self.discard_goal('observation goal no longer fits the robot width', now)
                return (0.0, 0.0), 'REPLAN'
            if math.dist(self.pose[:2], self.goal) <= p['goal_tolerance']:
                self.get_logger().info(f'LOCAL_GOAL_REACHED {self.goal}')
                self.goals_reached += 1
                if len(self.observation_grid) <= self.goal_map_size:
                    self.failed_goals[self.world_cell(*self.goal)] = now+p['goal_cooldown']
                self.goal, self.route, self.carrot = None, [], None
                self.observation_start, self.last_plan_time = now, now
                return (0.0, 0.0), 'LOCAL_GOAL_REACHED'
        if self.goal is None:
            if np.max(np.abs(self.last_command)) > 1e-4 or abs(self.measured_motion[0]) > 0.01:
                return (0.0, 0.0), 'STOPPING_TO_PLAN'
            if self.observation_start is not None:
                wait = p['motion_confirmation_wait'] if self.motion_target() is not None else p['planning_interval']
                if now-self.observation_start < wait:
                    return (0.0, 0.0), 'OBSERVING_FOR_MOTION'
                self.observation_start = None
            if now-self.last_plan_time >= p['planning_interval']:
                self.select_local_goal()
                self.last_plan_time = self.sensor_time()
            return (0.0, 0.0), 'PLAN_READY' if self.goal is not None else 'NO_REACHABLE_FRONTIER'
        if self.route_stalled(now):
            self.discard_goal('no route progress; choose another map route', now)
            return (0.0, 0.0), 'REPLAN'
        return self.follow_carrot(now)

    def world_cell(self, x, y):
        """Convert metres to a grid cell. Floor also gives the correct cell for negative coordinates."""
        resolution = self.nav_settings['map_resolution']
        return math.floor(x/resolution), math.floor(y/resolution)

    def cell_world(self, cell):
        """Convert a grid cell back to its centre position in metres."""
        resolution = self.nav_settings['map_resolution']
        return ((cell[0]+0.5)*resolution, (cell[1]+0.5)*resolution)

    def ray_cells(self, start, end):
        """Walk the grid cells crossed by a laser ray using integer steps."""
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
        """Use each fresh scan to clear cells along rays and mark their obstacle endpoints.
        Occupied cells need two clear observations to disappear, so moving objects can leave the map.
        """
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
            # A no-return ray clears space, but must not invent an obstacle at its range limit.
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
        # Two clear observations remove an old hit, allowing a moving obstacle to leave the map.
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
        self.memory_stamp = self.scan_stamp

    def navigation_step(self):
        """Prepare the sensors, update the map, choose a command and apply the simple width check.
        Missing or stale data sends zero velocity; no stopping-distance prediction is used.
        """
        if self.stop_requested:
            self.move_2D()
            raise SystemExit
        state = self.prepare_scan()
        now = self.sensor_time()
        self.sensor_ready = state == 'READY_STOPPED'
        if self.sensor_ready:
            self.update_memory()
            self.observe_motion()
        if self.sensor_ready and not self.calibrate:
            desired, state = self.navigation_command(now)
            command = np.asarray(desired, dtype=float)
            # Check only the robot's width along the next forward segment.
            # A blocked segment stops translation, but still lets the robot turn.
            if command[0] > 0:
                yaw = math.radians(self.pose[2])
                ahead = self.pose[:2]+self.nav_settings['look_ahead_distance']*np.array((math.cos(yaw),math.sin(yaw)))
                if not self.segment_clear(self.pose[:2],ahead):
                    command[0], state = 0.0, 'ROUTE_BLOCKED'
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
        if state in ('ROUTE_BLOCKED', 'NO_ROUTE') and self.blocked_since is None:
            self.blocked_since = now
        elif state in ('FOLLOWING','SWING_APPROACH','SWING_ALIGNING','SWING_CROSSING') and np.any(np.abs(command) > 1e-4):
            self.blocked_since = None
        if state != self.sensor_state or now - self.last_status >= self.nav_settings['status_interval']:
            scan_age = now - self.scan_received if self.scan_received is not None else math.inf
            pose_age = now - self.pose_received if self.pose_received is not None else math.inf
            self.get_logger().info(
                f'{state} | pose={self.pose} | scan_age={scan_age:.3f}s '
                f'pose_age={pose_age:.3f}s | cells={len(self.observation_grid)} '
                f'goals={self.goals_reached}/{self.goals_selected} '
                f'progress={self.path_progress:.2f}m | cmd=({command[0]:.3f}, 0, {command[1]:.3f})')
            self.sensor_state, self.last_status = state, now
        self.show_diagnostics(state)

    def timer_callback(self):
        """Run one navigation update, or print physical calibration poses without driving."""
        if self.stop_requested:
            self.move_2D()
            raise SystemExit
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        self.navigation_step()
                

def main(args=None):
    """Read the calibration option and ROS parameters, then run the controller until stopped."""
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(calibrate=cli_args.calibrate)
    # Let the timer send zero velocity before the ROS node is destroyed.
    signal.signal(signal.SIGINT, obstacle.stop_navigation)
    signal.signal(signal.SIGTERM, obstacle.stop_navigation)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
