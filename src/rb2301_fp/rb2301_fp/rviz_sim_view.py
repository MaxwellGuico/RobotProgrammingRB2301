"""Read-only simulation diagnostics; keep all RViz code outside the controller."""
import argparse
import ast
from bisect import bisect_right
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from rb2301_fp.obstacle_course import GAP_LOOKAHEAD, scan_to_body
from rb2301_fp.gap_navigation import PLANNING_HORIZONS, choose_gap


COLORS = dict(robot=(.35, .65, 1., 1.), lidar=(.8, .8, .8, .8),
              tested=(1., .3, .3, .22), clear=(.15, .9, .4, .25),
              preview=(.15, .9, .9, .9), target=(1., .85, .1, 1.),
              motion=(.95, .3, .9, 1.), track=(.55, .65, .75, .8),
              selected=(1., .45, .1, 1.), text=(1., 1., 1., 1.))


def gap_geometry(points, lookahead=GAP_LOOKAHEAD):
    """Reconstruct current scan eligibility; actual commitment comes from telemetry."""
    choice = choose_gap(points, horizons=(lookahead,), commitment=lookahead)
    return choice.directions, choice.groups


def recorded_setting(trajectory, name):
    """Read a literal setting from either navigation or legacy controller snapshots."""
    for filename in ('gap_navigation.py', 'obstacle_course.py'):
        snapshot = Path(trajectory).with_name(filename)
        if snapshot.is_file():
            for node in ast.parse(snapshot.read_text()).body:
                if (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
                    return ast.literal_eval(node.value)
    return None


def recorded_lookahead(trajectory):
    """Read a saved numeric setting without executing the controller snapshot."""
    value = recorded_setting(trajectory, 'GAP_LOOKAHEAD')
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def build_markers(debug, scan, command, stamp, stale='', trails=None, body_points=None,
                  lookahead=GAP_LOOKAHEAD, horizons=PLANNING_HORIZONS):
    """Create odom-frame diagnostics without publishing or changing robot commands."""
    result = MarkerArray()
    result.markers.append(Marker(action=Marker.DELETEALL))
    pose = debug.get('pose') or [0., 0., 0.]
    yaw = math.radians(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)

    def point(x, y, z=.04, world=False):
        if not world:
            x, y = pose[0]+c*x-s*y, pose[1]+s*x+c*y
        return Point(x=float(x), y=float(y), z=float(z))

    def marker(namespace, kind, color, width=.015):
        msg = Marker()
        msg.header.frame_id, msg.header.stamp = 'odom', stamp
        msg.ns, msg.id, msg.type, msg.action = namespace, len(result.markers), kind, Marker.ADD
        msg.pose.orientation.w = 1.
        msg.scale.x = float(width)
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = COLORS[color]
        msg.lifetime.sec, msg.lifetime.nanosec = 0, 800_000_000
        result.markers.append(msg)
        return msg

    def polyline(namespace, pts, color, width=.015, world=False, z=.04):
        msg = marker(namespace, Marker.LINE_STRIP, color, width)
        msg.points = [point(x, y, z, world) for x, y in pts]
        return msg

    def text(namespace, label, x, y, color='text', world=False, z=.35, size=.10):
        msg = marker(namespace, Marker.TEXT_VIEW_FACING, color)
        msg.pose.position = point(x, y, z, world)
        msg.scale.z, msg.text = float(size), label
        msg.color.a = 1.
        return msg

    def arrow(namespace, start, end, color, z=.09):
        msg = marker(namespace, Marker.ARROW, color, .025)
        msg.scale.y, msg.scale.z = .06, .08
        msg.points = [point(*start, z=z), point(*end, z=z)]
        return msg

    def dot(namespace, x, y, color, size=.08, world=False):
        msg = marker(namespace, Marker.SPHERE, color)
        msg.pose.position = point(x, y, .1, world)
        msg.scale.x = msg.scale.y = msg.scale.z = float(size)
        return msg

    if debug.get('simulation') is not True or debug.get('pose') is None:
        text('status', 'Waiting for simulation telemetry\n./fp.sh --simulation --terminal-debug',
             0., 0., z=.5)
        return result

    circle = np.linspace(0, 2*math.pi, 81)
    polyline('robot_clearance_circle', np.c_[.07+.15*np.cos(circle), .15*np.sin(circle)], 'robot')
    polyline('robot_safety_box', [(-.05,-.13),(.2,-.13),(.2,.13),(-.05,.13),(-.05,-.13)], 'robot')
    arrow('robot_heading', (0., 0.), (.28, 0.), 'robot')
    dot('robot_reference', 0., 0., 'robot', .035)

    mount = debug.get('lidar', dict(yaw_deg=180., x_offset=.11423, y_offset=0.))
    horizon = None
    if scan is not None or body_points is not None:
        points = np.asarray(body_points if body_points is not None else
                            scan_to_body(scan, **mount, max_range=3.5)[0]).reshape(-1, 2)
        cloud = marker('lidar', Marker.POINTS, 'lidar', .018)
        cloud.scale.y = .018
        cloud.points = [point(x, y) for x, y in points]
        if body_points is None:
            front = np.asarray(scan_to_body(scan, **mount, max_range=1.5,
                                            front_only=True)[0]).reshape(-1, 2)
        else:
            relative = points-[mount.get('x_offset', .11423), mount.get('y_offset', 0.)]
            front = points[(relative[:, 0] >= -1e-9)
                           & (np.linalg.norm(relative, axis=1) < 1.5)]
        history = debug.get('navigation', {}).get('history')
        choice = choose_gap(front, horizons=horizons, commitment=lookahead,
                            pose=pose, history=history)
        if history:
            visited = marker('visited_space', Marker.POINTS, 'track', .035)
            visited.scale.y = .035
            visited.points = [point(x, y, world=True) for x, y in history]
        directions, groups, horizon = choice.directions, choice.groups, choice.horizon
        rays = marker('tested_directions', Marker.LINE_LIST, 'tested', .004)
        for direction in directions[::5]:
            rays.points.extend([point(.07, 0.), point(*([.07, 0.]+horizon*direction))])
        for group in groups:
            fan = marker('steerable_gap_geometry', Marker.TRIANGLE_LIST, 'clear', 1.)
            fan.scale.y = fan.scale.z = 1.
            for a, b in zip(group[:-1], group[1:]):
                fan.points.extend([point(.07, 0., .025),
                                   point(*([.07, 0.]+horizon*directions[a]), z=.025),
                                   point(*([.07, 0.]+horizon*directions[b]), z=.025)])
            centre = directions[group[len(group)//2]]*horizon+[.07, 0.]
            text('gap_labels', f'{len(group)} samples', *centre, color='clear', z=.14, size=.055)
        if choice.target is not None:
            arrow('next_gap_preview', (0., 0.), choice.target, 'preview', z=.11)

    # This is the actual retained waypoint, rather than a fresh scan's preference.
    target = debug.get('target')
    if target is not None:
        arrow('committed_waypoint', (0., 0.), target, 'target', z=.13)
        dot('committed_waypoint', *target, 'target')
        text('committed_waypoint_label', f'Committed: {np.linalg.norm(target):.2f} m remaining',
             *target, color='target', z=.25, size=.075)

    arc_angles = np.linspace(math.radians(-85), math.radians(85), 90)
    polyline('lookahead_distance', np.c_[.07+lookahead*np.cos(arc_angles),
                                        lookahead*np.sin(arc_angles)], 'preview', .009)
    v, w = command if command is not None else (0., 0.)
    times = np.linspace(0, .5, 21)
    theta = w*times
    xs = v*times if abs(w) < 1e-6 else v*np.sin(theta)/w
    ys = np.zeros_like(times) if abs(w) < 1e-6 else v*(1-np.cos(theta))/w
    polyline('commanded_motion_0_5s', zip(xs, ys), 'motion', .025, z=.07)
    for index in (8, 20):
        box = np.array([[-.05,-.13],[.2,-.13],[.2,.13],[-.05,.13],[-.05,-.13]])
        co, si = math.cos(theta[index]), math.sin(theta[index])
        box = box@np.array([[co,si],[-si,co]])+[xs[index],ys[index]]
        polyline('predicted_safety_footprint', box, 'motion', .009, z=.06)

    active = debug.get('state') in ('position', 'observe', 'cross')
    panel_x = debug.get('panel_x') if active else None
    for i, track in enumerate(debug.get('tracks', [])):
        selected = panel_x is not None and abs(track['x']-panel_x) < .12
        color = 'selected' if selected else 'track'
        namespace = 'selected_swinging_obstacle' if selected else 'observed_swinging_obstacles'
        dot(namespace, track['x'], track['y'], color, .15 if selected else .09, world=True)
        period = track.get('period')
        label = ('SELECTED panel' if selected else 'Observed track') + f' {i+1}'
        label += f'\nperiod {period:.2f} s' if period else '\nlearning period'
        text('panel_labels', label, track['x'], track['y'], color, world=True, z=.4, size=.085)
        history = (trails or {}).get(i, [])
        if len(history) > 1:
            polyline('observed_swing_history', history, color, .012, world=True)
    if active and panel_x is not None:
        lane = debug.get('lane', pose[1])
        polyline('selected_panel_plane', [(panel_x, lane-.5), (panel_x, lane+.5)],
                 'selected', .02, world=True)
    wait_point = debug.get('wait_point')
    if wait_point is not None and debug.get('state') in ('position', 'observe'):
        dot('panel_wait_point', *wait_point, 'selected', .09, world=True)

    state = debug.get('state', 'waiting')
    horizon_text = f'{horizon:.2f} m' if horizon is not None else 'waiting'
    status = (f"{state} | panel {debug.get('obstacle', '?')} | commitment ≤ {lookahead:.2f} m\n"
              f'Next-gap planning horizon: {horizon_text}\n'
              'Green: clear gap geometry | cyan: next-gap preview\n'
              'Yellow: committed waypoint | magenta: commanded 0.5 s motion\n'
              'Grey: observed tracks | orange: selected panel')
    if stale:
        status = stale+'\n'+status
    text('status', status, -.15, .85, z=.55, size=.09)
    return result


class SimulationViewer(Node):
    """Subscribe to diagnostics and publish markers; never own cmd_vel."""

    def __init__(self, replay=None):
        super().__init__('fp_rviz_sim_view')
        self.debug, self.scan, self.command = {}, None, None
        self.received, self.trails = {}, {}
        self.publisher = self.create_publisher(MarkerArray, 'fp/simulation_markers', 1)
        self.replay = None
        if replay is not None:
            self.replay = [json.loads(line) for line in Path(replay).read_text().splitlines()]
            self.replay = [e for e in self.replay if e.get('debug', {}).get('pose') is not None]
            if not self.replay:
                raise ValueError('Replay contains no controller poses')
            self.replay_times = [entry['elapsed'] for entry in self.replay]
            self.replay_lookahead = recorded_lookahead(replay)
            self.replay_horizons = (recorded_setting(replay, 'PLANNING_HORIZONS')
                                    or (self.replay_lookahead or GAP_LOOKAHEAD,))
            self.replay_started = time.monotonic()
            self.get_logger().info('Replaying recorded diagnostics; no robot commands are published')
        else:
            self.create_subscription(String, 'fp/debug', self.on_debug, qos_profile_sensor_data)
            self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
            self.create_subscription(Twist, 'cmd_vel', self.on_command, qos_profile_sensor_data)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.2, self.publish_markers, clock=self.steady_clock)

    def on_debug(self, msg):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        if not self.debug and data.get('simulation') is True:
            self.get_logger().info('Live simulation telemetry connected')
        self.debug, self.received['debug'] = data, time.monotonic()
        tracks = data.get('tracks', [])
        self.trails = {i: history for i, history in self.trails.items() if i < len(tracks)}
        for i, track in enumerate(tracks):
            history = self.trails.setdefault(i, [])
            position = [track['x'], track['y']]
            if history and abs(history[-1][0]-position[0]) > .2:
                history.clear()
            if not history or position != history[-1]:
                history.append(position)
                del history[:-160]

    def on_scan(self, msg):
        self.scan, self.received['scan'] = msg, time.monotonic()

    def on_command(self, msg):
        self.command = msg.linear.x, msg.angular.z
        self.received['command'] = time.monotonic()

    def publish_markers(self):
        now = time.monotonic()
        if self.replay is not None:
            elapsed = self.replay_times[0] + now-self.replay_started
            index = max(0, min(len(self.replay)-1, bisect_right(self.replay_times, elapsed)-1))
            entry = self.replay[index]
            command = entry.get('command', [0., 0.])
            command = (command[0], command[2]) if len(command) == 3 else command
            status = f"REPLAY {entry['elapsed']:.1f} / {self.replay_times[-1]:.1f} s"
            if elapsed >= self.replay_times[-1]:
                status += ' (ended)'
            if self.replay_lookahead is None:
                status += '\nPreview uses current config; saved lookahead unavailable'
            self.publisher.publish(build_markers(entry['debug'], None, command,
                self.get_clock().now().to_msg(), status,
                body_points=entry.get('lidar_body', []),
                lookahead=self.replay_lookahead or GAP_LOOKAHEAD,
                horizons=self.replay_horizons))
            return
        stale = [name for name in ('debug', 'scan', 'command')
                 if now-self.received.get(name, -math.inf) > 1.]
        warning = 'STALE / missing: '+', '.join(stale) if stale else ''
        self.publisher.publish(build_markers(self.debug, self.scan, self.command,
                                             self.get_clock().now().to_msg(), warning, self.trails))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, help='Replay a recorded trajectory.jsonl without Gazebo')
    options, ros_args = parser.parse_known_args(args)
    if options.replay is not None and not options.replay.is_file():
        parser.error('Replay file does not exist: '+str(options.replay))
    # Let Python stop spinning before shutting down the ROS context.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = SimulationViewer(options.replay)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous_handlers = {sig: signal.signal(sig, interrupt)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
