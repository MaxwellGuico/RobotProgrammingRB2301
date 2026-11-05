"""Navigate LiDAR gaps, learn two moving panels, and stop before the end wall."""
import argparse
import configparser
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rb2301_fp.gap_navigation import (
    GAP_LOOKAHEAD, GAP_REACHED_TOLERANCE, GapNavigation, body_overlap,
    rotate_points, wrap_deg, wrap_rad)

# Set to 0.0 if the real LiDAR already faces forward. Gazebo's mount stays 180.
REAL_LIDAR_YAW_DEG = 180.0


def scan_clusters(points, indices):
    """Split at missing beams or jumps over 12 cm; retain scan order."""
    if not len(points):
        return []
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.12
    return np.split(points, np.flatnonzero(jumps | (np.diff(indices) != 1)) + 1)


def scan_to_body(msg, yaw_deg=0., x_offset=.11423, y_offset=0.,
                 max_range=math.inf, front_only=False):
    """Use scan angles and the mount offset to produce robot +x forward, +y left."""
    ranges = np.asarray(msg.ranges)
    angles = wrap_rad(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
                      + np.deg2rad(yaw_deg))
    beams = np.arange(len(ranges))
    if front_only:
        beams = beams[np.abs(angles) <= np.pi / 2 + 1e-9]
        beams = beams[np.argsort(angles[beams], kind='stable')]
    indices = np.arange(len(beams)) if front_only else beams
    r, a = ranges[beams], angles[beams]
    valid = (np.isfinite(r) & (r >= max(.05, msg.range_min))
             & (r <= msg.range_max) & (r < max_range))
    points = np.column_stack((r[valid] * np.cos(a[valid]) + x_offset,
                              r[valid] * np.sin(a[valid]) + y_offset))
    return [tuple(point) for point in points.tolist()], indices[valid].tolist()


def maze_to_course(x, y, heading, frame):
    """Translate to a frame's origin, then rotate into its axes."""
    xy = rotate_points(np.array([x - frame['origin_x'], y - frame['origin_y']]),
                       -frame['rotation_deg'])
    return xy[0], xy[1], wrap_deg(heading - frame['rotation_deg'])


def optitrack_to_maze(x, y, heading, frame):
    """Apply the measured maze frame and the rigid body's heading calibration."""
    x, y, heading = maze_to_course(x, y, heading, frame)
    return x, y, wrap_deg(heading - frame['heading_offset_deg'])


def load_frame(robot_number):
    """Load frame positions and this robot's Optitrack heading correction."""
    config = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'optitrack_variables.config')
    if not config.read(path):
        raise FileNotFoundError(path)
    keys = ('origin_x', 'origin_y', 'rotation_deg')
    frame = {k: config.getfloat('frame', k) for k in keys}
    offset = config.get('frame', 'heading_offset_deg', fallback='auto').strip().lower()
    automatic = wrap_deg(-frame['rotation_deg'] - 90)
    offset = automatic if offset == 'auto' else float(offset)
    frame['heading_offset_deg'] = config.getfloat(
        'heading_offsets', str(robot_number), fallback=offset)
    course = {k: config.getfloat('course', k, fallback=0.) for k in keys}
    return frame, course


class ObstacleCourseNode(GapNavigation, Node):
    """Use one scan stream and publish one chosen velocity per 20 Hz tick."""

    def __init__(self, is_simulation=False, calibrate=False,
                 terminal_debug=False, start_here=False):
        """Connect pose, scan, commands, and optional terminal telemetry."""
        super().__init__('obstaclecourse')
        self.get_logger().info('Mode: simulation (odom)' if is_simulation else 'Mode: real (VRPN)')
        self.is_simulation, self.calibrate = is_simulation, calibrate
        self.pose = self.raw_pose = self.last_scan = None
        self.start = None
        self.frame = self.course = None
        self.capture_start = start_here and not is_simulation
        self.state, self.panel_number = 'approach', 1
        self.tracks, self.target = [], None
        self._ticks = 0
        # The real robot must remain capped at 0.3 m/s.
        self.speed_limit = 1.4 if is_simulation else .3
        yaw = 180. if is_simulation else REAL_LIDAR_YAW_DEG
        defaults = dict(yaw_deg=yaw, x_offset=.11423, y_offset=0.)
        self.lidar = {
            k: self.declare_parameter('lidar_' + k, v).value
            for k, v in defaults.items()}
        self.create_subscription(LaserScan, 'scan', self.sub_scan_callback,
                                 qos_profile_sensor_data)
        if is_simulation:
            self.create_subscription(Odometry, 'odom', self.pose_callback,
                                     qos_profile_sensor_data)
        else:
            domain = os.environ.get('ROS_DOMAIN_ID', '')
            number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(number)
            topic = f'/vrpn_mocap/bingda_{number:03d}/pose'
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(PoseStamped, topic, self.pose_callback, qos)
            self.get_logger().info('Reading pose from ' + topic)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(.05, self.timer_callback)
        if terminal_debug:
            self.debug_pub = self.create_publisher(String, 'fp/debug', 2)
            self.create_timer(.2, self.publish_debug)

    def sub_scan_callback(self, msg):
        """Keep the latest timestamped scan for navigation and panel tracking."""
        self.last_scan = msg
        self.scan_received = time.monotonic()

    def pose_callback(self, msg):
        """Read yaw and position; calibrate real poses and capture an optional start."""
        pose = msg.pose.pose if self.is_simulation else msg.pose
        q = pose.orientation
        yaw = np.rad2deg(np.arctan2(2 * (q.w*q.z + q.x*q.y),
                                    1 - 2 * (q.y*q.y + q.z*q.z)))
        values = (pose.position.x, pose.position.y, yaw)
        if not self.is_simulation:
            self.raw_pose = np.array(values)
            values = optitrack_to_maze(*values, self.frame)
        if not np.all(np.isfinite(values)):
            return
        if self.capture_start:
            self.course.update(origin_x=float(values[0]), origin_y=float(values[1]))
            self.capture_start = False
            self.get_logger().info(f'Course origin captured: {values[:2]}')
        self.pose = np.array(values)
        self.pose_received = time.monotonic()
        if self.start is None:
            self.start = self.to_course()[:2].tolist()

    def to_course(self, pose=None):
        """Convert maze poses to course coordinates; simulation is already aligned."""
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(*pose, self.course) if self.course else pose)


    def move_2D(self, x=0., y=0., turn=0.):
        """Publish a Twist with the hardware speed cap applied; other axes stay zero."""
        msg, limit = Twist(), self.speed_limit
        msg.linear.x = float(np.clip(x, 0., limit))
        msg.linear.y = float(np.clip(y, -limit, limit))
        msg.angular.z = float(np.clip(turn, -2*limit, 2*limit))
        self.publisher_.publish(msg)


    def observe_panels(self):
        """Cluster fresh scans into short faces and match each face by its course x."""
        msg = self.last_scan
        if msg is None:
            return None, np.empty((0, 2))
        now = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if now == getattr(self, 'scan_time', None):
            return now, self.world_points
        self.scan_time = now
        local, indices = scan_to_body(msg, **self.lidar, max_range=3.5)
        pose = self.to_course()
        points = rotate_points(np.asarray(local).reshape(-1, 2), pose[2]) + pose[:2]
        self.world_points = points
        used = []
        for cluster in sorted(scan_clusters(points, indices), key=len, reverse=True):
            span = np.ptp(cluster, axis=0)
            x, y = (cluster.min(axis=0) + cluster.max(axis=0)) / 2
            if (len(cluster) < 2 or span[0] > .12 or not .06 < span[1] < .42
                    or not (3.1 < x < 5.5 and -4.7 < y < -1.4)
                    or any(abs(t['x'] - x) < .10 for t in used)):
                continue
            track = min(self.tracks, key=lambda t: abs(t['x'] - x), default=None)
            if track is None or abs(track['x'] - x) >= .10:
                track = dict(x=float(x), samples=[], fit_time=-100., model=None)
                self.tracks.append(track)
            used.append(track)
            track['x'] = .9*track['x'] + .1*x
            track['samples'] = [(t, y) for t, y in track['samples'] if t >= now-32]
            track['samples'].append((now, float(y)))
            if now - track['fit_time'] >= 1:
                track['model'] = self.learn_cycle(track) or track['model']
                track['fit_time'] = now
        return now, points

    def learn_cycle(self, track):
        """Compare a 20 Hz history against shifts of 3-13 s to learn its cycle."""
        samples = np.asarray(track['samples'])
        if (len(samples) < 30 or samples[-1, 0]-samples[0, 0] < 8
                or np.ptp(samples[:, 1]) < .5):
            return None
        t, y = samples.T
        values = np.interp(np.arange(t[0], t[-1], .05), t, y)
        lags = np.arange(60, min(261, len(values)-79))
        errors = np.array([np.abs(values[n::2]-values[:-n:2]).mean() for n in lags])
        if len(errors) < 3 or errors.min() > .08:
            return None
        valid = ((errors[1:-1] <= errors.min()+.01)
                 & (errors[1:-1] <= errors[:-2]) & (errors[1:-1] <= errors[2:]))
        periods = lags[1:-1][valid] * .05
        return dict(period=periods[0], t=t, y=y) if len(periods) else None

    def enter_state(self, state, now):
        """Enter a panel state and reset steering history before its next command."""
        self.state, self.state_since = state, now
        self.steer(None)
        print(f'Moving obstacle {self.panel_number}: {state}', flush=True)

    def can_cross(self, track, now, pose):
        """Predict panel clearance over the entire time the chassis occupies its sweep."""
        model = track['model']
        if model is None or now-track['samples'][-1][0] > .25:
            return False
        plane, speed = track['x'] + .025, min(.4, self.speed_limit)
        enter = max(0., (plane-.22-pose[0])/speed - .08)
        leave = max(enter, (plane+.10-pose[0])/speed + .08)
        times = now + np.linspace(enter, leave, 30) - model['period']
        predicted = np.interp(times, model['t'], model['y'])
        current = np.interp(now-model['period'], model['t'], model['y'])
        return (times[0] >= model['t'][0] and abs(current-track['samples'][-1][1]) < .08
                and np.all(np.abs(predicted-self.lane) > .33))

    def panel_candidates(self, pose):
        """Reject wall fragments using the first-panel band and 0.6 m panel spacing."""
        candidates = []
        for track in self.tracks:
            x, samples = track['x'], track['samples']
            in_band = (3.1 < x < 4.3 if self.panel_number == 1 else
                       abs(x-self.first_panel_x-.6) <= .15)
            if (in_band and x > pose[0]+.15 and len(samples) > 10
                    and np.ptp(np.asarray(samples)[:, 1]) > .14):
                candidates.append(track)
        return candidates

    def panel_command(self):
        """Stage before each panel, wait for a predicted window, cross, then finish."""
        now, points = self.observe_panels()
        pose = self.to_course()
        x, y, heading = pose
        yaw, state = np.deg2rad(heading), self.state
        self.body_points = rotate_points(points-pose[:2], -heading)
        if state == 'done' or now is None:
            return (0., 0.) if state == 'done' else None
        if state == 'approach':
            if x < 3 or y > -2.1:
                return None
            self.tracks = []
            self.enter_state('scout', now)
            state = 'scout'
        if state == 'scout':
            candidates = self.panel_candidates(pose)
            minimum = 6 if self.panel_number == 1 else 3
            if now-self.state_since < minimum or not candidates:
                return 0., 0.
            self.panel = min(candidates, key=lambda t: t['x'])
            plane = self.panel['x'] + .025
            if self.panel_number == 1:
                self.first_panel_x = self.panel['x']
            self.wait_x, self.lane = max(x, plane-.32), float(y)
            self.enter_state('position', now)
        elif state == 'position':
            if x < self.wait_x-.06:
                return None
            # The gate is behind us: cross along the lane actually reached.
            self.lane = float(y)
            if abs(yaw) > np.deg2rad(4):
                return self.safe_command(self.body_points, 0.,
                                              float(np.clip(-3*yaw, -.8, .8)))
            self.enter_state('observe', now)
        elif state == 'observe':
            minimum = 18 if self.panel_number == 1 else 3
            if (now-self.state_since >= minimum
                    and self.can_cross(self.panel, now, pose)):
                self.enter_state('cross', now)
        elif state == 'cross':
            if x <= self.panel['x']+.175:
                w = np.clip(2.5*(self.lane-y) - 3*yaw, -.6, .6)
                return min(.4, self.speed_limit), float(w)
            print(f'Obstacle {self.panel_number} cleared', flush=True)
            next_state = 'scout' if self.panel_number == 1 else 'finish'
            self.panel_number = 2
            self.enter_state(next_state, now)
        elif state == 'finish':
            wall = points[(points[:, 0] > x+.15)
                          & (np.abs(points[:, 1]-self.lane) < .15)]
            if len(wall):
                distance = float(wall[:, 0].min()-x)
                if distance > .36:
                    v = min(.12, self.speed_limit, max(.03, distance-.35))
                    return v, float(np.clip(-2*yaw, -.4, .4))
                self.state = 'done'
                print('completed', flush=True)
        return 0., 0.


    def gap_command(self, scan):
        """Give transformed LiDAR to the separate commitment controller."""
        points, _ = scan_to_body(scan, **self.lidar, max_range=1.5, front_only=True)
        return GapNavigation.follow_gap(self, np.asarray(points).reshape(-1, 2))

    def timer_callback(self):
        """Wait for sensors, optionally print calibration, then publish one command."""
        if self.pose is None:
            return
        if self.calibrate:
            self._ticks += 1
            if self._ticks % 10 == 0:
                self.get_logger().info(
                    f'Raw {self.raw_pose}; maze {self.pose.round(3)}; '
                    f'course {self.to_course().round(3)}')
            return
        if self.last_scan is None or len(self.last_scan.ranges) < 4:
            self.steer(None)
            self.progress = None
            self.move_2D()
            return
        command = self.panel_command()
        if command is None:
            command = self.gap_command(self.last_scan)
        else:
            self.steer(None)
            self.progress = None
        self.move_2D(x=command[0], turn=command[1])

    def publish_debug(self):
        """Publish pose, sensed staging point, gap target, and learned panel periods."""
        tracks = [dict(x=t['x'], y=t['samples'][-1][1],
                       period=float(t['model']['period']) if t['model'] else None)
                  for t in self.tracks if t['samples']]
        payload = dict(pose=self.to_course().tolist() if self.pose is not None else None,
                       target=self.target,
                       navigation=getattr(self, 'gap_debug', {}),
                       state='calibrate' if self.calibrate else self.state,
                       obstacle=self.panel_number, simulation=self.is_simulation,
                       tracks=tracks, lidar=self.lidar, start=self.start,
                       wait_point=[self.wait_x, self.lane] if hasattr(self, 'wait_x') else None,
                       lane=getattr(self, 'lane', None),
                       panel_x=float(self.panel['x']) if hasattr(self, 'panel') else None)
        self.debug_pub.publish(String(data=json.dumps(payload)))


def main(args=None):
    """Default to real VRPN; use --simulation for odometry and clean up on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation', action='store_true',
                        help='Use simulation odometry instead of real VRPN')
    parser.add_argument('--calibrate', action='store_true', help='Print poses without driving')
    parser.add_argument('--terminal-debug', action='store_true', help='Enable terminal telemetry')
    parser.add_argument('--start-here', action='store_true',
                        help='Real robot: capture start position')
    options, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    obstacle = None
    try:
        obstacle = ObstacleCourseNode(options.simulation, options.calibrate,
                                      options.terminal_debug, options.start_here)
        rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        print('Shutting down')
    finally:
        if obstacle is not None:
            if rclpy.ok():
                obstacle.move_2D()
            obstacle.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
