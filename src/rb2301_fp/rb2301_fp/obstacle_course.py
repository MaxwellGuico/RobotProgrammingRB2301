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
        # Max's variables
        '''Lidar Processing'''
        self.lidar_min_distance = 0.0
        self.lidar_max_distance = 5.0
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.robot_safe_radius = 0.17
        self.min_gap_width = 2 * self.robot_safe_radius
        self.valid_gaps = []
        self.points = None
        '''PID Variables'''
        self.Kp,self.Kd,self.Kp_theta,self.Kd_theta = 3.0, 0.8,3.0,0.8
        self.error_distance_prev = None
        self.error_heading_prev = None
        self.PID_prev_time = self.get_clock().now()
        '''Callback Variables'''
        self.committed_target = None
        '''Distance based Recording of visted history'''
        self.visited_positions = []
        self.history_spacing = 0.2
        self.visit_radius = 0.5 # metres

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 
            self.angle_increment *= 2
        # Boolean masking to get valid measurements, better than np.clip because if np.inf and we change it to lets say 5.0, it thinks theres an obstacle there.
        valid = (
            np.isfinite(self.last_scan)
            & (self.last_scan >= max(self.lidar_min_distance, msg.range_min))
            & (self.last_scan <= min(self.lidar_max_distance, msg.range_max))
        )
        #theta i = self.angle_min + index x self.angle_increment
        self.angles = (
            self.angle_min
            + np.arange(len(self.last_scan)) * self.angle_increment
        )
        # Preserve original indices
        # Lidar points give (r,theta)
        # Convert to cartesian coordinates
        x = self.last_scan[valid] * np.cos(self.angles[valid])
        y = self.last_scan[valid] * np.sin(self.angles[valid])

        self.points = np.column_stack((x, y)) # Combine into points (N,2) given N is number of lidar scans
        if len(self.points) < 2: # edge case: if every measurement is invalid, clustering will give empty cluster
            self.valid_gaps = []
            return
        # Cluster LiDar Points
        differences = self.points[1:] - self.points[:-1] # Example, P1 - P0, P2 - p1, P3 - P2 pattern
        distances = np.linalg.norm(differences, axis=1) # get the euclidean distance for each row
        valid_indices = np.where(valid)[0] # theres a gap if you dont filter with valid
        index_differences = valid_indices[1:] - valid_indices[:-1]
        boundaries = np.where((distances > 0.2) | (index_differences > 1))[0] # get points that are far apart, because we are looking for cluster boundaries to tell us where a cluster starts and ends
        # Splitting into clusters
        # np.split() splits BEFORE the specified index
        split_indices = boundaries + 1
        clusters = np.split(self.points, split_indices) # list of numpy arrays
        # clusters = [c for c in clusters if len(c) >= 2] # filter out for lidar noise, if there is less than 2 valid points, remove it
        # Detect Gaps between adjacent clusters
        '''
        CLUSTER A                  CLUSTER B

        •──•──•──A                 B──•──•──•
                ↑                 ↑
            Last point        First point
        
        '''
        valid_gaps = list()
        for i in range(len(clusters) - 1):
            point_A = clusters[i][-1]  # Last point of Cluster 0
            point_B = clusters[i+1][0]   # First point of Cluster 1
            gap_width = np.linalg.norm((point_B - point_A)) # Eucldiean distance

            if gap_width >= self.min_gap_width:
                gap_midpoint = (point_A + point_B) / 2
                '''Estimate: free-space depth'''
                free_depth = 0.0
                theta_midpoint = np.arctan2(gap_midpoint[1],gap_midpoint[0])
                index = int(round((theta_midpoint - self.angle_min)/ self.angle_increment))
                if 0 <= index < len(self.last_scan): # checking that the index is valid
                    lidar_distance = self.last_scan[index]
                    max_range = min(self.lidar_max_distance, msg.range_max)
                    if(np.isfinite(lidar_distance) and lidar_distance >= self.lidar_min_distance and lidar_distance <= self.lidar_max_distance): # normal reading
                        free_depth = max(0,lidar_distance - np.linalg.norm(gap_midpoint))
                    elif np.isposinf(lidar_distance): # Edge case, if lidar distance is inf
                        free_depth = max(0,max_range - np.linalg.norm(gap_midpoint))
                    else: # if np.nan
                        free_depth = 0.0
                '''Collision Checking'''
                # Calculate crossing direction
                gap_vector = point_B - point_A
                normal = np.array([-gap_vector[1], gap_vector[0]])

                # Ensure normal points away from robot
                if np.dot(normal, gap_midpoint) < 0:
                    normal = -normal

                # Normalize
                normal = normal / np.linalg.norm(normal)

                # Calculate crossing target, its a point after the midpoint
                crossing_distance = 0.5
                crossing_target = gap_midpoint + crossing_distance * normal
                # Use projections to see if the points will hit the robot
                # t = P dot M / M dot M
                denominator = np.dot(crossing_target, crossing_target)
                if denominator < 1e-6:
                    continue
                numerator = self.points @ crossing_target
                t = np.clip(numerator / denominator,0,1) # size (N,), clip to ensure it fits 0 and 1
                # Q = R + t(M - R) but since its relative to robot now, R is (0,0)
                Q = t[:, None] * crossing_target # (N,1) * N,2
                D = np.linalg.norm((self.points - Q), axis=1)


                '''Valid Candidate Gaps'''
                if np.min(D) > self.robot_safe_radius:
                    valid_gaps.append({
                        'gap_width': gap_width,
                        'free_depth' : free_depth,
                        'target': crossing_target,
                        'A': point_A,
                        'B': point_B
                    })
        self.valid_gaps = valid_gaps
                
        
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
        w =  Kp_theta * error_heading + Kd_theta* ((error_heading-self.error_heading_prev)/((dt.nanoseconds/1e9))) \
                   if (dt.nanoseconds/1e9) > 1e-9 else Kp_theta *(error_heading)
        
        self.get_logger().debug(f"Error Distance: {error_distance} Turning Correction Applied: {max(0.0, np.cos(error_heading))} Vx: {Vx}")
        self.get_logger().debug(f"Desired Heading: {desired_heading} Current Heading: {current_heading} Error Heading: {error_heading} Angular z: {w}")
        '''Save prev errors and prev PID time'''

        self.error_distance_prev = error_distance
        self.error_heading_prev = error_heading
        self.PID_prev_time = current_time

        return Vx, Vy,w 
    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack
        '''Calibration'''
        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######
        self.get_logger().info(f"Pose: {self.pose}")       
        '''Position History'''
        if not self.visited_positions:
            self.visited_positions.append(self.pose[:2].copy())

        elif np.linalg.norm(
            self.pose[:2] - self.visited_positions[-1]
        ) >= self.history_spacing:
            self.visited_positions.append(self.pose[:2].copy())
        # Convert carrot to real world coordinates Cw = R + RotationR*Carrot
        theta = np.deg2rad(self.pose[2])
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        '''
        Target Selection
        gap['target'] is w.r.t Robot
        self.committed_target is w.r.t World
        target_robot_frame is w.r.t Robot
        '''
        # Only select a new target if none is committed
        if self.committed_target is None:
            if not self.valid_gaps:
                self.move_2D(0.0, 0.0, 0.0)
                return
            '''
            Gap Scoring
            Importance of width * Normalised gap width + Importance of depth * Normalised depth
            '''
            # For now its based on width and depth
            max_width = max(gap['gap_width'] for gap in self.valid_gaps)
            max_depth = max(gap['free_depth'] for gap in self.valid_gaps)
            # Calculate score for each gap
            for gap in self.valid_gaps:
                normalised_gap = gap['gap_width'] / max_width if max_width > 0 else 0.0
                normalised_depth = gap['free_depth'] / max_depth if max_depth > 0 else 0.0

                # Visted-Location Penalty 
                target_world = self.pose[:2] + rotation @ gap['target']
                history = np.array(self.visited_positions)
                differences = history - target_world
                distances = np.linalg.norm(differences,axis=1)
                nearest_visited = np.min(distances)
                '''Implementing penalty V = max(0,1 - nereast_visited/Radius visited)'''
                visited_penalty = max(0.0,1.0-nearest_visited/self.visit_radius)
                '''Backward penalty'''
                backward_penalty = 1.0 if gap['target'][0] < 0 else 0.0
                gap['score'] = 0.5 * normalised_gap + 0.5 * normalised_depth - 0.3 * visited_penalty - 0.2 * backward_penalty
            best_gap = max(self.valid_gaps,key=lambda gap : gap['score'])

            target = best_gap['target']
            # Robot frame -> World frame
            self.committed_target = self.pose[:2] + rotation @ target

        # Target selection: World frame -> Robot frame
        target_robot_frame = rotation.T @ (
            self.committed_target - self.pose[:2]
        )
        '''Target commmitment reset check'''
        distance = np.linalg.norm(target_robot_frame)
        if distance < 0.15:
            self.committed_target = None
            self.move_2D(0.0, 0.0, 0.0)
            return
        '''Path commitment check'''
        # Same idea as the collision checking along midpoint path
        denominator = np.dot(target_robot_frame, target_robot_frame)
        if denominator < 1e-6:
            self.move_2D(0.0, 0.0, 0.0)
            return
        if self.points is None or len(self.points) == 0:
            self.move_2D(0.0, 0.0, 0.0)
            return
        numerator = self.points @ target_robot_frame
        t = np.clip(numerator / denominator , 0 , 1)
        Q = t[:, None] * target_robot_frame
        D = np.linalg.norm((self.points - Q ), axis=1)
        if np.min(D) < self.robot_safe_radius:
            self.move_2D(0.0, 0.0, 0.0)
            return
        '''Carrot Follower'''
        # Calculate direction
        distance = np.linalg.norm(target_robot_frame)
        if distance < 1e-6:
            self.move_2D(0.0, 0.0, 0.0)
            return
        # unit vector
        direction = target_robot_frame / distance
        # Calculate carrot
        # carrot = lookahead distance * direction
        lookahead_distance = 0.5
        carrot = min(lookahead_distance, distance) * direction

        carrot_world = np.array(self.pose[:2]) + rotation @ carrot
        v_final, vy , w = self.PID(carrot_world, self.Kp,self.Kd,self.Kp_theta,self.Kd_theta)

        self.move_2D(v_final,0,w)
        
                

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