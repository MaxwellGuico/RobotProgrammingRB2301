"""Read-only ROS terminal viewer using only Python's standard library and ROS."""
import argparse
import curses
import json
import math
import sys
import time
from rb2301_fp.obstacle_course import REAL_LIDAR_YAW_DEG, scan_to_body


def scan_points(msg, **mount):
    """Use the same robot-frame conversion as the controller."""
    return scan_to_body(msg, **mount)[0]


def course_point(point, pose):
    """Convert a robot-relative point into course coordinates, in metres."""
    yaw = math.radians(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    return (pose[0] + c*point[0] - s*point[1],
            pose[1] + s*point[0] + c*point[1])


def navigation_targets(debug, points):
    """Describe the current aim without changing the controller's commands."""
    pose, state = debug.get('pose'), debug.get('state')
    gap = course_point(debug['target'], pose) if pose and debug.get('target') else None
    waypoint, action = gap, 'follow gaps' if state == 'approach' else state or 'waiting'
    if state == 'position':
        waypoint, action = debug.get('wait_point'), 'panel staging'
    if pose and state in ('scout', 'observe', 'done', 'calibrate'):
        waypoint, action = pose[:2], 'hold'
    elif state == 'cross' and debug.get('panel_x') is not None:
        waypoint = (debug['panel_x'] + .175, debug['lane'])
    elif state == 'finish':
        waypoint, action = None, 'end wall; stop at 0.36 m'
        if pose and debug.get('lane') is not None:
            world = [course_point(p, pose) for p in points]
            wall = [x for x, y in world if x > pose[0]+.15
                    and abs(y-debug['lane']) < .15]
            if wall:
                waypoint, action = (min(wall)-.36, debug['lane']), 'finish estimate'
    return waypoint, gap, action


def format_point(point):
    """Format a coordinate pair without inventing missing positions."""
    return 'waiting' if point is None else '({:+.2f},{:+.2f})'.format(*point[:2])


def render_frame(width, height, points, debug, command, status, radius=2.0, course=False):
    """Render a robot-centred map without requiring a terminal or ROS runtime."""
    pose = debug.get('pose')
    start = debug.get('start')
    waypoint, gap, action = navigation_targets(debug, points)
    mode = 'COURSE AXES (+x right, +y up)' if course else 'ROBOT AXES (front up, left left)'
    if command is None:
        velocity = ['cmd_vel: waiting for commands', 'cmd_vel: wz=waiting']
    else:
        vx, vy, wz = (command[0], 0., command[1]) if len(command) == 2 else command
        velocity = [
            'cmd_vel: vx={:+.2f} vy={:+.2f} | speed={:.2f} m/s'.format(
                vx, vy, math.hypot(vx, vy)),
            'cmd_vel: wz={:+.2f} rad/s ({:+.1f} deg/s) [commanded]'.format(
                wz, math.degrees(wz))]
    lines = ['RB2301 FP | READ ONLY | ' + mode,
             status,
             'State: {} | obstacle: {} | hardware: {}'.format(
                 debug.get('state', '?'), debug.get('obstacle', '?'),
                 ('simulation' if debug['simulation'] else 'real') if 'simulation' in debug else '?'),
             ('Course pose: x={:+.2f} y={:+.2f} heading={:+.1f} deg'.format(*pose)
              if pose else 'Course pose: waiting for controller telemetry'),
             *velocity,
             'Start S={} [course m] | Finish: detected wall'.format(format_point(start)),
             'Action W={} [{}; course m]'.format(format_point(waypoint), action),
             'Gap T={} [course m]; local={}'.format(
                 format_point(gap), format_point(debug.get('target'))),
             'Radius {:.2f}m | # scan R robot S start W action T gap M panel'.format(radius)]
    if width < 60 or height < 24:
        return ['Enlarge terminal to at least 60 columns x 24 rows.']
    cols = min(width - 3, 101)
    rows = min(height - len(lines) - 6, cols // 2)
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    scale = min(cx, cy * 2) / radius
    yaw = math.radians(pose[2]) if pose else 0.0
    c, s = math.cos(yaw), math.sin(yaw)

    def draw(x, y, mark, world=False):
        if world:
            if not pose:
                return
            x, y = x - pose[0], y - pose[1]
            if not course:
                x, y = c * x + s * y, -s * x + c * y
        elif course:
            if not pose:
                return
            x, y = c * x - s * y, s * x + c * y
        horizontal, vertical = (x, y) if course else (-y, x)
        col, row = round(cx + horizontal * scale), round(cy - vertical * scale / 2)
        if 0 <= col < cols and 0 <= row < rows:
            grid[row][col] = mark

    for x, y in points:
        draw(x, y, '#')
    for track in debug.get('tracks', []):
        draw(track['x'], track['y'], 'M', world=True)
    if start is not None:
        draw(*start[:2], 'S', world=True)
    if waypoint is not None:
        draw(*waypoint[:2], 'W', world=True)
    draw(min(0.45, radius * 0.7), 0, '+')
    if gap is not None:
        draw(*gap, 'T', world=True)
    grid[round(cy)][round(cx)] = 'R'
    lines += ['+' + '-' * cols + '+']
    lines += ['|' + ''.join(row) + '|' for row in grid]
    lines += ['+' + '-' * cols + '+']
    tracks = debug.get('tracks', [])
    cycles = ['x={:.2f}: {}'.format(t['x'], '{:.2f}s'.format(t['period'])
                                   if t.get('period') is not None else 'learning') for t in tracks]
    lines += ['Panel cycles: ' + (', '.join(cycles) or 'none tracked'),
              'q quit viewer | +/- zoom | m switch axes | robot stays centred',
              './fp.sh --terminal-debug | Targets outside map hidden.']
    return [line[:width - 1] for line in lines[:height - 1]]


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation', action='store_true',
                        help='Use the Gazebo LiDAR mount when telemetry is unavailable')
    parser.add_argument('--range', type=float, default=2.0, dest='radius', help='Map radius in metres')
    parser.add_argument('--lidar-yaw-deg', type=float, default=None,
                        help='Override LiDAR mounting yaw (default: controller telemetry, otherwise obstacle_course.py mounting setting)')
    options, ros_args = parser.parse_known_args(args)
    if not math.isfinite(options.radius) or options.radius <= 0:
        parser.error('--range must be a positive finite number')
    if options.lidar_yaw_deg is not None and not math.isfinite(options.lidar_yaw_deg):
        parser.error('--lidar-yaw-deg must be finite')
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error('Run this viewer in an interactive terminal')

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    class Viewer(Node):
        def __init__(self):
            super().__init__('fp_terminal_view')
            self.points, self.debug, self.command = [], {}, None
            self.last_scan = None
            self.received = {}
            self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
            self.create_subscription(Twist, 'cmd_vel', self.on_command, qos_profile_sensor_data)
            self.create_subscription(String, 'fp/debug', self.on_debug, qos_profile_sensor_data)

        def on_scan(self, msg):
            self.last_scan = msg
            self.update_points()
            self.received['scan'] = time.monotonic()

        def update_points(self):
            if self.last_scan is None:
                return
            mount = dict(self.debug.get('lidar', {}))
            mount.setdefault('yaw_deg', 180.0 if options.simulation else REAL_LIDAR_YAW_DEG)
            if options.lidar_yaw_deg is not None:
                mount['yaw_deg'] = options.lidar_yaw_deg
            self.points = scan_points(self.last_scan, **mount)

        def on_command(self, msg):
            self.command = (msg.linear.x, msg.linear.y, msg.angular.z)
            self.received['cmd'] = time.monotonic()

        def on_debug(self, msg):
            try:
                payload = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            if isinstance(payload, dict):
                self.debug = payload
                self.update_points()
                self.received['state'] = time.monotonic()

    rclpy.init(args=ros_args)
    node = Viewer()

    def display(screen):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.nodelay(True)
        radius, course = options.radius, False
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            key = screen.getch()
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('+'), ord('=')):
                radius = max(0.25, radius / 1.25)
            elif key == ord('-'):
                radius = min(20.0, radius * 1.25)
            elif key in (ord('m'), ord('M')):
                course = not course
            now = time.monotonic()
            status = ' | '.join('{}: {}'.format(name, 'waiting' if name not in node.received else
                                '{:.1f}s ago{}'.format(now - node.received[name],
                                ' STALE' if now - node.received[name] > 1.0 else ''))
                                for name in ('scan', 'cmd', 'state'))
            height, width = screen.getmaxyx()
            screen.erase()
            for row, line in enumerate(render_frame(width, height, node.points, node.debug,
                                                    node.command, status, radius, course)):
                try:
                    screen.addstr(row, 0, line)
                except curses.error:
                    pass
            screen.refresh()

    try:
        curses.wrapper(display)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
