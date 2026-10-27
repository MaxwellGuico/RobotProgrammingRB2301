import argparse
import configparser
import json
import math
import os
from pathlib import Path
import sys
import time
import ast

import numpy as np
import heapq
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
import yaml

from rclpy.qos import (
    ReliabilityPolicy,
    QoSProfile,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image
from geometry_msgs.msg import Twist


np.set_printoptions(
    2, suppress=True, threshold=np.inf
)  # Print numpy arrays to specified d.p., suppress scientific notation (e.g. 1e-5), and do not truncate

set_logger_level("waypoint", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

occupancy_grid_resolution = 0.2 # Sim grid resolution, in metres per cell (the real maze uses the same 0.2m cells, see ca2_irl_layout.json)
max_translate_velocity = 1.4 # Overwritten in main() based on sim vs real-life; 0.4m/s cap for real life, please keep that in place

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


# --- Coordinate conversion --------------------------------------------------
# A grid index (i, j) represents a CELL, not a point. That cell's world
# coordinate is its CENTER, e.g. cell [0, 0] is centred half a resolution-step
# away from the grid's origin corner, not exactly on it. This matches how the
# Gazebo world and the real maze are physically laid out (goal tape/markers
# sit in the middle of a cell, not on its boundary line).
def grid_to_world(i:int, j:int, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert grid index (i, j) to the world (x, y) coordinate of that cell's centre.'''
    return (origin[0] + (i + 0.5) * resolution, origin[1] + (j + 0.5) * resolution)

def world_to_grid(x:float, y:float, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert a world (x, y) coordinate to the grid index (i, j) of the cell containing it.'''
    return (int(np.floor((x - origin[0]) / resolution)), int(np.floor((y - origin[1]) / resolution)))


# --- Sim / real-life profiles -----------------------------------------------
# Run without arguments for the simulation. For the real maze pass
# --run test1 | test2 | full  (see README). Everything that differs between
# sim and real -- which map to load, the cell resolution, where the grid's
# [0, 0] corner sits in the frame your pose is reported in, the goal points,
# the speed cap -- lives in one of these profiles.
sim_config = {
    "map_file": "ca2_sim_map.npy",
    "origin": (-1.0, -5.0),
    "resolution": occupancy_grid_resolution,
    "goal_list": [(3.5, -3.5), (3.3, 0.3), (2.5, -3.5), (-0.3, -3.7)],
    "max_translate_velocity": 1.4,
}

IRL_RUNS = ("test1", "test2", "full")

RUNTIME_PARAMETER_DEFAULTS = {
    "kp": 3.0,
    "kd": 0.8,
    "kp_theta": 3.0,
    "kd_theta": 0.8,
    "look_ahead_dist": 0.15,
    "goal_tolerance": 0.05,
    "generate_image": True,
}


# --- Real maze <-> Optitrack frame -------------------------------------------
# The real maze is a fixed layout (ca2_irl_map.npy + ca2_irl_layout.json, in
# "maze-local" metres: (0, 0) is the outer corner of array cell [0, 0], +x runs
# along array axis 0, +y along axis 1 -- exactly like the sim grid). Optitrack
# reports poses in ITS OWN frame, wherever the maze happens to be taped down.
# optitrack_variables.config holds the 3 numbers that relate the two:
#   origin_x, origin_y : Optitrack coordinates of the maze's (0, 0) corner
#   rotation_deg       : angle of the maze's +x axis in the Optitrack frame
#                        (counter-clockwise positive, degrees)
# The pose from Optitrack is converted into the maze frame before your code
# sees it, so self.origin is (0, 0) and everything else (world_to_grid,
# goal_list, ...) works exactly like in simulation.
def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> maze-frame pose.'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame.get("heading_offset_deg", 0.0))) # offset: the Motive rigid body's "forward" vs the robot's front

def maze_to_optitrack(mx:float, my:float, heading_deg:float, frame:dict) -> tuple:
    '''Inverse of optitrack_to_maze (handy for working out where to place the robot).'''
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (frame["origin_x"] + c * mx - s * my, frame["origin_y"] + s * mx + c * my,
            wrap_deg(heading_deg + frame["rotation_deg"] + frame.get("heading_offset_deg", 0.0)))

def robot_frame(config:dict, robot_number:int) -> dict:
    '''The frame for one robot: the shared origin/rotation plus that robot's heading offset (per-robot value from the config if listed).'''
    frame = dict(config["frame"])
    frame["heading_offset_deg"] = config["heading_offsets"].get(robot_number, frame["heading_offset_deg"])
    return frame

def load_irl_config(run:str) -> dict:
    '''Build the real-maze profile for run "test1", "test2" or "full".'''
    if run not in IRL_RUNS:
        raise ValueError(f"--run must be one of {IRL_RUNS}, got {run!r}")
    with open(os.path.join(_PACKAGE_DIR, "ca2_irl_layout.json")) as f:
        layout = json.load(f)
    parser = configparser.ConfigParser()
    config_path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(config_path):
        raise FileNotFoundError(f"Could not read {config_path}")
    frame = {
        "origin_x": parser.getfloat("frame", "origin_x"),
        "origin_y": parser.getfloat("frame", "origin_y"),
        "rotation_deg": parser.getfloat("frame", "rotation_deg"),
    }
    # Rigid bodies are created in Motive with the robot facing the maze's +y direction, so the rigid body's zero heading IS maze +y
    # (maze heading 90 deg): heading_maze = reported + 90 = reported - rotation - offset  =>  offset = -(rotation + 90). "auto" means that.
    raw_offset = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw_offset == "auto" else float(raw_offset)
    heading_offsets = {int(k): float(v) for k, v in parser.items("heading_offsets")} if parser.has_section("heading_offsets") else {}
    return {
        "map_file": "ca2_irl_map.npy",
        "origin": (0.0, 0.0), # pose is converted into the maze frame, so the grid corner is (0, 0)
        "resolution": float(layout["resolution"]),
        "goal_list": [tuple(g) for g in layout["runs"][run]["goals"]],
        "start": tuple(layout["runs"][run]["start"]),
        "frame": frame,
        "robot_number": parser.getint("robot", "number"),
        "heading_offsets": heading_offsets, # per-robot override of frame["heading_offset_deg"]
        "max_translate_velocity": 0.4, # Please keep this in place
    }
class SearchNode:
    '''SearchNode is a helper class for Astar Function'''
    def __init__(self,cell,g,h,parent=None,order=None):

        self.cell = cell # Cell is (i,j) in Grid Coordinates
        self.g = g # Cost to go from start to the current cell
        self.h = h # Heuristic cost from the goal to the current cell
        self.f = g + h
        self.parent = parent # This is for path reconstruction
        self.order = order # Sorting
    def cost(self):
        return self.f

    def __lt__(self, other): # Less than inbuilt function, useful for heapq
        return (self.f, self.order) < (other.f, other.order) # this helps for tie breaker situation
    
    def __eq__(self, value): # Equal to inbuilt function, useful for heapq
        return self.f == value.f

    
class WaypointNode(Node):
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True, origin:tuple=(0.0, 0.0), resolution:float=occupancy_grid_resolution,
                 frame:dict=None, robot_number:int=5, expected_start:tuple=None, calibrate:bool=False,
                 run:str=None, params_output_path:str=None):
        self._images_disabled = run in ("test1", "test2")
        image_override = [Parameter("generate_image", value=False)] if self._images_disabled else None
        super().__init__('waypoint', parameter_overrides=image_override)
        self.get_logger().info("Starting WaypointNode")

        self.is_simulation = is_simulation
        self.params_output_path = Path(params_output_path) if params_output_path else None
        self.add_on_set_parameters_callback(self._validate_runtime_parameters)
        for name, default in RUNTIME_PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        self.add_post_set_parameters_callback(self._runtime_parameters_changed)
        self.save_runtime_parameters()
        if self._images_disabled:
            self.get_logger().info(f"Image creation disabled for --run {run}")

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.goal_list = goal_list
        self.map_array = map_array
        self.origin = origin # World (x, y) coordinate of the grid's [0, 0] corner. Use with grid_to_world()/world_to_grid()
        self.resolution = resolution # Metres per grid cell (0.2). Use with grid_to_world()/world_to_grid()

        self.frame = frame # Optitrack->maze frame (real robot only), see optitrack_to_maze()
        self.expected_start = expected_start # Where this run says the robot should be placed (maze frame), real robot only
        self.calibrate = calibrate # Real robot: only print poses, never publish cmd_vel
        self.raw_pose = None # Latest pose exactly as Optitrack reported it (real robot only)
        self._start_checked = False
        self._calibrate_ticks = 0

        self.pose = None
        self.path = [] # Set this to your planned route (a list of grid-index tuples, in travel order) once you've computed it -- it'll automatically show up in the terminal map print
        self.simplified_points = [] # Reduced points
        self.cumulative_dist_segments = [0.0]

        '''Tracking the indices of goal_index and waypoint_index from self.path'''
        self.goal_idx = -1
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.goal_reached = True
        self._last_printed_path = None
        '''PID Variables'''
        self.error_distance_prev = None
        self.error_heading_prev = None
        self.PID_prev_time = self.get_clock().now()
        '''Carrot Variables'''
        self.path_progress = 0.0 # Initialise as 0
        self.current_segment_idx = 0


    def _validate_runtime_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name not in RUNTIME_PARAMETER_DEFAULTS:
                continue
            value = parameter.value
            if parameter.name == "generate_image":
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="generate_image must be a boolean")
                if self._images_disabled and value:
                    return SetParametersResult(successful=False, reason="Image creation is disabled for test1 and test2")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return SetParametersResult(successful=False, reason=f"{parameter.name} must be a finite number")
            positive = parameter.name in ("look_ahead_dist", "goal_tolerance")
            if value < 0.0 or (positive and value == 0.0):
                bound = "positive" if positive else "non-negative"
                return SetParametersResult(successful=False, reason=f"{parameter.name} must be {bound}")
        return SetParametersResult(successful=True)

    def save_runtime_parameters(self):
        '''Save the effective settings as a reusable ROS parameter YAML file.'''
        if self.params_output_path is None:
            return
        names = list(RUNTIME_PARAMETER_DEFAULTS) + ["use_sim_time"]
        values = {parameter.name: parameter.value for parameter in self.get_parameters(names)}
        self.params_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.params_output_path.with_suffix(".yaml.tmp")
        with temporary_path.open("w") as stream:
            yaml.safe_dump({self.get_fully_qualified_name(): {"ros__parameters": values}}, stream, sort_keys=False)
        os.replace(temporary_path, self.params_output_path)
        self.get_logger().info(f"Runtime parameters saved to {self.params_output_path}")

    def _runtime_parameters_changed(self, parameters):
        if any(parameter.name in RUNTIME_PARAMETER_DEFAULTS or parameter.name == "use_sim_time" for parameter in parameters):
            try:
                self.save_runtime_parameters()
            except OSError as error:
                self.get_logger().error(f"Could not save runtime parameters: {error}")


    def _print_calibration(self):
        '''--calibrate: twice a second, print the raw Optitrack pose next to the maze-frame pose and grid cell.'''
        self._calibrate_ticks += 1
        if self._calibrate_ticks % 10:
            return
        ox, oy, oh = self.raw_pose
        mx, my, mh = self.pose
        i, j = world_to_grid(mx, my, self.origin, self.resolution)
        self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({mx:+.3f}, {my:+.3f}, {mh:+7.1f} deg)  cell [{i}, {j}]")

    def _check_start_once(self):
        '''Once, on the first pose: warn if the robot isn't near this run's start point (usually a placement or config mistake).'''
        if self._start_checked or self.expected_start is None:
            return
        self._start_checked = True
        dist = math.hypot(self.pose[0] - self.expected_start[0], self.pose[1] - self.expected_start[1])
        msg = f"Robot is at maze ({self.pose[0]:.2f}, {self.pose[1]:.2f}); this run starts at {self.expected_start} -- {dist:.2f} m away"
        if dist > 0.4:
            self.get_logger().warn(msg + ". Check where the robot is placed; if it is on the start point, ask a TA.")
        else:
            self.get_logger().info(msg + " (ok)")

    def print_map(self):
        '''Prints the occupancy grid to the terminal: walls, your current position ('S'), all goal points ('W'/'G'),
        and your planned route (self.path) if you've set one ('*'). Safe to call anytime pose is known; does nothing
        useful before then. Called automatically from timer_callback() whenever self.path changes.'''
        if self.pose is None:
            return
        shape = self.map_array.shape
        clip = lambda cell: (int(np.clip(cell[0], 0, shape[0]-1)), int(np.clip(cell[1], 0, shape[1]-1)))
        current_cell = clip(world_to_grid(self.pose[0], self.pose[1], self.origin, self.resolution))
        goal_cells = [clip(world_to_grid(gx, gy, self.origin, self.resolution)) for gx, gy in self.goal_list]
        grid = Grid(self.map_array, starting_position=current_cell, goal_position=goal_cells[-1])
        grid.print_grid_map(waypoints=goal_cells, path=self.path)
        if self.get_parameter("generate_image").value:
            grid.draw_grid_map(simplified_points=self.simplified_points,waypoints=goal_cells,path=self.path,save_path='path_result.png')

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
        if self.frame is not None:
            x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # Your code works in the maze frame, like in simulation
        self.pose = np.array((x,y,heading))
        return self.pose

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
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def set_waypoints(self, waypoints:list):
        '''Set new waypoints when a goal has been reached'''
        self.goal_reached = False
        self.waypoints = waypoints
        self.current_waypoint_idx = 1 if len(waypoints) > 1 else 0

        '''Reset Carrot'''
        self.path_progress = 0.0 
        self.current_segment_idx = 0
        '''Corner Aware Speed Regulator'''
        self.is_cornering = False
        self.corner_idx = None
        
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
        Vx = np.clip(Vx, 0.0, max_translate_velocity)
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
    def reset_pid(self):
        self.error_heading_prev = None
        self.error_distance_prev = None

    def reconstruct_path(self,current:SearchNode):
        path = []
        while current is not None:
            path.append(current.cell)
            current = current.parent
        return path[::-1] # The first node is the goal node, the last node is the next point to go to

    def simplify_path(self,path:list):
        if not path: # Edge case: Empty list recieved
            return []
        if len(path) == 1: # Edge case: 1 Point only
            return path
        path_idx = 1
        new_path = [path[path_idx-1]]
        
        while path_idx < len(path) - 1:
            prev_node = np.array(path[path_idx-1])
            current_node = np.array(path[path_idx])
            next_node = np.array(path[path_idx+1])
            din = prev_node-current_node
            dout = current_node-next_node
            if not np.array_equal(din,dout): 
            # Compare the change, 
            # if there is a difference in the direction of increment
            # din changes in x, dout changes in y, means its an actual waypoint to go to.
                new_path.append(path[path_idx])
            path_idx += 1
        new_path.append(path[-1])
        return new_path # I also edited the draw_grid and _colors function to reflect the path more accurately
    
    def Astar(self,start:tuple, goal:tuple):
        def heuristic(start:tuple,goal:tuple):
            '''Chosen heuristic is manhattan distance'''
            return abs(goal[0]-start[0]) + abs(goal[1]-start[1])
        def cost(node:SearchNode):
            return node.cost()
        start_node = SearchNode(cell=start,g=0,h=heuristic(start,goal),parent=None)
        
        
        openList = [start_node] # List of SearchNode
        closedList = [] # List of SearchNode

        grid_size_x = self.map_array.shape[0]
        grid_size_y = self.map_array.shape[1]

        while openList:
            '''Get node with loweset f value'''
            current = openList[0] # Revist this, right now it assumes that the first one is the best f value

            # Check if we have reached the goal
            if current.cell == goal:
                return self.reconstruct_path(current)

            # Move current node to the closed list
            openList.remove(current)
            closedList.append(current)

            # Check all neighbouring nodes
            '''Check the 4 directions of cell, reject index that are negative or outside of grid size'''
            
            neighbour_up = (current.cell[0] + 1, current.cell[1]) if grid_size_x > (current.cell[0] + 1) >= 0 else None
            neighbour_down = (current.cell[0] - 1, current.cell[1]) if grid_size_x > (current.cell[0] - 1) >= 0 else None
            neighbour_left = (current.cell[0], current.cell[1] + 1) if grid_size_y > (current.cell[1] + 1) >= 0 else None
            neighbour_right = (current.cell[0], current.cell[1] - 1) if grid_size_y > (current.cell[1] - 1) >= 0 else None
            neighbours = [neighbour_up,neighbour_down,neighbour_left,neighbour_right]

            for neighbour in neighbours:
                if neighbour is None:
                    continue # Skip invalid index nodes
                if any(neighbour == node.cell for node in closedList):
                    continue # Skip evaluated node

                # calculate tentative g score
                tentative_g = current.g + 1 # Plus one since we are only looking at 4 directions one cell difference

                for node in openList:
                    if node.cell == neighbour:
                        if tentative_g <= node.g:
                            node.g = tentative_g
                            node.f = node.g + node.h
                            node.parent = current

                if not any(neighbour == node.cell for node in openList) and self.map_array[neighbour] == 0: 
                    '''check if neighbour is in NOT in open list and is a free cell'''
                    neighbourNode = SearchNode(parent=current,cell=neighbour,g=tentative_g,h=heuristic(neighbour, goal))
                    openList.append(neighbourNode)

            openList.sort(key=cost)
        return [] # Failure case
    def get_carrot(self,robot_curr_position:tuple,path,look_ahead_dist):
        
        '''
        ### robot_curr_position
            - (x,y) position of the robot, do not give heading
        ### look_ahead_dist: distance of carrot
        ### path: list of points to the goal
        For N waypoints,
            N Path World Coord
            N - 1 Length segments
            N Cumulative Distance Segments
            0 <= self.current_segment_idx <= N - 2
            To adv to next segment, 0 <= self.current_segment_idx < N - 2
        '''
        if len(path) < 2:
            return False

        path_world_coord = [grid_to_world(grid_coord[0],grid_coord[1],origin=self.origin,resolution=self.resolution) for grid_coord in path]
        length_segments = []
        t = 0.0 
        
        for i in range(len(path_world_coord)):
            if 0 < i:
                length_segments.append(math.dist(path_world_coord[i-1],path_world_coord[i]))
        if sum(length_segments) <= 1e-9:
                return path_world_coord[-1]
        self.cumulative_dist_segments = [0.0]
        for length in length_segments:
            self.cumulative_dist_segments.append(length + self.cumulative_dist_segments[-1])
        
        while True:
        # Calculate AB, AR and t
            A = np.array(path_world_coord[self.current_segment_idx])
            B = np.array(path_world_coord[self.current_segment_idx + 1])
            R = np.array(robot_curr_position)
            AB = B - A
            if np.linalg.norm(AB) <= 1e-9: # If the length segment is 0
                if self.current_segment_idx == len(path_world_coord) - 2 : # If its the maximum segment idx to check
                    t = 0.0 # Ensure t resets to 0
                    break
                self.current_segment_idx += 1 # Move to the next length segment since this one is 0
                continue
            AR = R - A
            t = max(0, min(1,np.dot(AR,AB) / np.dot(AB,AB))) # This ensures that the t given is on the line segment selected

        # If t more than 95% switch
            # When im on line segment 1, it is from point 1 to point 2
            if t >= 0.95 and self.current_segment_idx < len(path_world_coord) - 2: # Maximum segment_idx = N - 2
                self.current_segment_idx += 1 
                continue
            break
        P_projected = np.array(path_world_coord[self.current_segment_idx]) + t * (AB) 
        distance_to_projected = np.linalg.norm(P_projected-np.array(robot_curr_position)) # TO be used for debugging
        s_projected = self.cumulative_dist_segments[self.current_segment_idx] + t *length_segments[self.current_segment_idx] # length projected
        self.path_progress = max(self.path_progress, s_projected)  # S[i] + t * Li # Cumulative progress s of projection


        s_carrot = min(self.path_progress + look_ahead_dist, self.cumulative_dist_segments[-1])
        segment = 0 # this would be the segment that the carrot should be in 
        while segment < len(length_segments) and length_segments[segment] <= 1e-9: # find the initial segment index that doesnt have a zero segment length
            segment += 1

        for idx in range(len(self.cumulative_dist_segments) - 1):
            if length_segments[idx] <= 1e-9:
                continue
            if self.cumulative_dist_segments[idx] <= s_carrot <= self.cumulative_dist_segments[idx + 1]:
                segment = idx
                break

        d_in_segment = s_carrot - self.cumulative_dist_segments[segment]
        alpha = d_in_segment / length_segments[segment]
        carrot_A = np.array(path_world_coord[segment])
        carrot_B = np.array(path_world_coord[segment + 1])
        carrot = carrot_A + alpha * (carrot_B - carrot_A)

        self.get_logger().debug(
            f"Robot Segment: {self.current_segment_idx}, "
            f"Carrot Segment: {segment}, "
            f"Progress: {self.path_progress:.3f}, "
            f"Carrot: {tuple(carrot)}"
            f"t value: {t*100}%"
        )

        return tuple(carrot)

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            return # Does not run if no pose received from Odom or Optitrack
        now = time.time()
        if now - getattr(self, "_last_pose_log", 0.0) >= 1.0: # at most once a second, so it does not bury the map / warnings below
            self._last_pose_log = now
            self.get_logger().debug(f"Pose: {self.pose}")

        if self.calibrate:
            self._print_calibration()
            return # Calibration mode: look, don't drive

        self._check_start_once()
        

        if self.path != self._last_printed_path: # Prints once immediately (map + start + goals), then again each time self.path changes
            self.print_map()
            self._last_printed_path = list(self.path)

        self.get_logger().debug(f"Pose: {self.pose}")
        
        '''Path Planner Logic'''
        # New goal
            # Make new path
            # self.goal_reached = False
        # Still same goal
            # Get Carrot
            # Check if pose is goal
                #if it is set goal_reached=True
        if not self.goal_reached and self.path: # If same goal and available path
            carrot = self.get_carrot(
                robot_curr_position=self.pose[:2],
                path = self.simplified_points,
                look_ahead_dist=self.get_parameter("look_ahead_dist").value
            )

            ''' PD Parameters to tune'''
            Kp = self.get_parameter("kp").value
            Kd = self.get_parameter("kd").value
            Kp_theta = self.get_parameter("kp_theta").value
            Kd_theta = self.get_parameter("kd_theta").value

            if carrot is False:
                self.get_logger().error("Carrot generation failed!")
                self.move_2D(0, 0, 0)
                return
            
            #self.get_logger().debug(f"Robot Pose: {self.pose[:2]}")
            self.get_logger().debug(f"Path generated(Simplified) : {self.simplified_points}")
            #self.get_logger().debug(f"Segment the robot is following: {self.current_segment_idx}")
            #self.get_logger().debug(f"Carrot Point: {carrot}")
           
            self.get_logger().debug(
                f"Path Progress: {self.path_progress:.3f} m"
            )
            v_final, vy , w = self.PID(carrot, Kp,Kd,Kp_theta,Kd_theta)


            self.move_2D(v_final, vy, w)


            self.get_logger().debug(f"Velocities Given (Vx={v_final},Vy={vy}, w = {w})")


            goal_error = self.current_xy_error(self.goal_list[self.goal_idx])
            goal_tolerance = self.get_parameter("goal_tolerance").value
            if abs(goal_error[0]) < goal_tolerance and abs(goal_error[1]) < goal_tolerance:
                self.goal_reached = True
                self.move_2D(0,0)
        else: # If new goal
            if self.goal_idx < len(self.goal_list) - 1:
                self.goal_idx += 1 
                curr_grid = world_to_grid(self.pose[0],self.pose[1],origin=self.origin,resolution=self.resolution)
                goal_grid = world_to_grid(self.goal_list[self.goal_idx][0],self.goal_list[self.goal_idx][1],origin=self.origin,resolution=self.resolution)
                self.path = self.Astar(curr_grid,goal_grid)
                self.simplified_points = self.simplify_path(self.path)
                self.set_waypoints(self.simplified_points)
                self.goal_reached = False
                '''Reset PID Variables'''
                self.reset_pid()
            else:
                self.move_2D(0,0)
            
        if self.path != self._last_printed_path: # Prints once immediately (map + start + goals), then again each time self.path changes
            self.print_map()
            self._last_printed_path = list(self.path)
            

    
class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given. Can show it and/or save it to a .png file
        print_grid_map : Prints an ASCII-art version of the same map (grid, waypoints, path) to the terminal
        animate_path : Prints the ASCII-art map frame by frame in the terminal, animating the robot moving along a given path

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array:np.array=np.array([]), starting_position:tuple=(0,0), goal_position:tuple=(-1,-1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0]+1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0]-1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]+1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]-1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1: # Means the flood is able to reach starting position from the ending position
            return True
        else:
            return False

    def _colour_grid(self, simplified_points:list=(), waypoints:list=(), path:list=(), obstacle_threshold:float=50) -> np.array:
        '''Builds the (H, W, 3) colour image array shared by draw_grid_map. Maze walls in blue, empty space in white, path taken in green and waypoints in red'''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)
        for x, y in simplified_points:
                        image_grid[x][y] = (125,0,125)
        for point in waypoints:
            image_grid[point] = (255,0,0)

        return image_grid

    def draw_grid_map(self,simplified_points:list = (), waypoints:list=(), path:list=(), obstacle_threshold:float=50, save_path:str=None, show:bool=True):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints. Will be represented in red, takes precedence over path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            save_path : Optional file path (e.g. "path_result.png") to save the image to, in addition to/instead of showing it
            show : Whether to pop up the image in a viewer (default True). Set to False if you only want to save it
        '''
        image_grid = self._colour_grid(simplified_points, waypoints, path, obstacle_threshold)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), getattr(Image, "Resampling", Image).NEAREST)

        if save_path:
            img.save(save_path)
            print(f"Saved map image to {save_path}")

        if show:
            img.show()

    def print_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50):
        '''Prints an ASCII-art version of the maze to the terminal: '#' wall, '.' free, '*' solution path, 'W' waypoint, 'S' start, 'G' goal

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
        '''
        chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')

        for x, y in path:
            chars[x, y] = '*'
        for point in waypoints:
            chars[point] = 'W'
        chars[self.starting_position] = 'S'
        chars[self.goal_position] = 'G'

        display = np.flip(chars, axis=1)[::-1]
        print('\n'.join(''.join(row) for row in display))

    def animate_path(self, path:list, waypoints:list=(), obstacle_threshold:float=50, delay:float=0.2):
        '''Animates the robot moving along path, one cell at a time, by reprinting the ASCII-art map to the terminal.
        'R' marks the robot's current cell, '*' cells it has already passed through.

        Args:
            path : list of tuple grid indices forming the solution path, in travel order
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            delay : Seconds to pause between animation frames
        '''
        path = list(path)
        base_chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')
        for point in waypoints:
            base_chars[point] = 'W'
        base_chars[self.goal_position] = 'G'

        for step in range(len(path)):
            frame = base_chars.copy()
            for x, y in path[:step]:
                frame[x, y] = '*'
            frame[path[step]] = 'R'

            display = np.flip(frame, axis=1)[::-1]
            print("\033c", end="")  # Clear terminal between frames
            print('\n'.join(''.join(row) for row in display))
            time.sleep(delay)


def main(args=None):
    global max_translate_velocity

    arg_parser = argparse.ArgumentParser(description="RB2301 CA2 path planning")
    arg_parser.add_argument(
        '--run', choices=IRL_RUNS, default=None,
        help="Run on the real maze: test1, test2 or full (starts/goals in README). Omit to run in Gazebo simulation."
    )
    arg_parser.add_argument(
        '--robot', type=int, default=None,
        help="Number of your Bingda robot (3 -> /vrpn_mocap/bingda_003/pose). Default: your ROS_DOMAIN_ID (set to the robot number on the robots), else [robot] number in optitrack_variables.config."
    )
    arg_parser.add_argument(
        '--calibrate', action='store_true',
        help="Real robot: only print Optitrack and maze-frame poses (twice a second) and never publish cmd_vel. Use it to set origin/rotation."
    )
    arg_parser.add_argument(
        '--save-params', default=None,
        help="Save the effective runtime parameters to this YAML file at startup and whenever they change."
    )
    process_args = sys.argv if args is None else [sys.argv[0], *args]
    cli_args = arg_parser.parse_args(remove_ros_args(args=process_args)[1:])
    run = cli_args.run or ("test1" if cli_args.calibrate else None)
    is_simulation = run is None # Remember: pass --run test1|test2|full to ca2.sh when testing on the real lab setup

    print("Starting path planning")
    rclpy.init(args=process_args)

    robot_number = 3
    frame = expected_start = None
    if is_simulation:
        config = sim_config
    else:
        config = load_irl_config(run)
        frame, expected_start = config["frame"], config["start"]
        domain = os.environ.get("ROS_DOMAIN_ID", "")
        if cli_args.robot is not None:
            robot_number = cli_args.robot
        elif domain.isdigit() and 1 <= int(domain) <= 99: # on the Bingda robots ROS_DOMAIN_ID is the robot number
            robot_number = int(domain)
        else:
            robot_number = config["robot_number"]
        frame = robot_frame(config, robot_number)
        expected_start = config["start"]
        print(f"Running on the real maze, run={run}, robot=bingda_{robot_number:03d}, resolution={config['resolution']}")
        print(f"  maze origin in Optitrack frame = ({frame['origin_x']}, {frame['origin_y']}), rotation = {frame['rotation_deg']} deg, robot heading offset = {frame['heading_offset_deg']} deg")
        print(f"  start={expected_start}, goals={config['goal_list']}")
        if cli_args.calibrate:
            print("  CALIBRATE MODE: printing poses only, the robot will not be commanded")

    max_translate_velocity = config["max_translate_velocity"]

    map_array = np.load(os.path.join(_PACKAGE_DIR, config["map_file"]), allow_pickle=True)
    waypoint = WaypointNode(map_array, config["goal_list"], is_simulation, config["origin"], config["resolution"],
                            frame=frame, robot_number=robot_number, expected_start=expected_start, calibrate=cli_args.calibrate,
                            run=run, params_output_path=cli_args.save_params)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):
        print("Shutting down")
    finally:
        waypoint.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
