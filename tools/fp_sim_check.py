#!/usr/bin/env python3
"""Run an isolated, seeded Gazebo course and retain controller/trajectory output."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stop(child):
    if child.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
            child.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--stall', type=float, default=100.)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--rviz', action='store_true',
                        help='Open the separate RViz decision viewer in this run environment')
    args = parser.parse_args()
    directory = ROOT / 'log' / 'fp_sim_checks' / args.name
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    domain = str(120 + int.from_bytes(hashlib.sha256(args.name.encode()).digest()[:2], 'big') % 100)
    env.update(ROS_DOMAIN_ID=domain, ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               GZ_PARTITION='rb2301_check_' + args.name, RB2301_RANDOMIZE='0',
               ROS_LOG_DIR=str(directory / 'ros'), GZ_HOMEDIR=str(directory / 'gz'),
               XDG_CACHE_HOME=str(directory / 'cache'), PYTHONUNBUFFERED='1')
    for key in tuple(env):
        if key.startswith(('SNAP', 'GTK', 'GDK', 'GIO', 'GSETTINGS')):
            env.pop(key, None)
    env['XDG_DATA_DIRS'] = env.get('XDG_DATA_DIRS_VSCODE_SNAP_ORIG', '/usr/local/share:/usr/share')
    os.environ.update({k: env[k] for k in ('ROS_DOMAIN_ID', 'ROS_AUTOMATIC_DISCOVERY_RANGE', 'ROS_LOG_DIR')})
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    from rb2301_fp import obstacle_modifier
    from rb2301_fp.terminal_view import render_frame, scan_points

    np.random.seed(args.seed)
    obstacle_modifier.overwrite_file = str(directory / 'world.sdf')
    obstacle_modifier.modify_sdf_file()
    source = ROOT / 'src/rb2301_fp/rb2301_fp/obstacle_course.py'
    sources = {}
    for filename in ('obstacle_course.py', 'gap_navigation.py'):
        path = source.with_name(filename)
        if path.exists():
            contents = path.read_bytes()
            (directory / filename).write_bytes(contents)
            sources[filename] = hashlib.sha256(contents).hexdigest()
    combined = ''.join(name+sources[name] for name in sorted(sources))
    result = {'name': args.name, 'seed': args.seed, 'result': 'failed',
              'reason': 'interrupted', 'source_sha256': sources['obstacle_course.py'],
              'controller_sources': sources,
              'controller_sha256': hashlib.sha256(combined.encode()).hexdigest()}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('fp_sim_check')
    data = {'pose': None, 'scan': None, 'debug': {}, 'command': (0., 0.)}
    transitions = []

    def pose(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        data['pose'] = [p.x, p.y, yaw]

    def scan(msg):
        data['scan'] = msg
        data['scan_received'] = time.monotonic()

    def debug(msg):
        data['debug'] = json.loads(msg.data)

    def command(msg):
        data['command'] = (msg.linear.x, msg.angular.z)

    node.create_subscription(Odometry, 'odom', pose, qos_profile_sensor_data)
    node.create_subscription(LaserScan, 'scan', scan, qos_profile_sensor_data)
    node.create_subscription(String, 'fp/debug', debug, qos_profile_sensor_data)
    node.create_subscription(Twist, 'cmd_vel', command, qos_profile_sensor_data)
    children, files = [], []

    def interrupted(signum, _frame):
        result['reason'] = 'Interrupted by signal ' + str(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(argv, name):
        log = (directory / (name + '.log')).open('w')
        files.append(log)
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    started = time.monotonic()
    last_sample = last_print = last_progress = started
    progress_pose, previous_state, controller, done_since = None, None, None, None
    try:
        gz = launch(['ros2', 'launch', 'rb2301_gz', 'fp_gazebo.launch.py',
                     'x:=0.0', 'y:=0.0', 'world:=' + str(directory / 'world.sdf'),
                     'gz_args:= -r -v1' if args.gui else 'gz_args:= -r -s --headless-rendering -v1'], 'gazebo')
        print('RUN:', args.name, 'seed:', args.seed, 'ROS domain:', domain, 'logs:', directory, flush=True)
        with (directory / 'trajectory.jsonl').open('w') as history, (directory / 'terminal.txt').open('w') as terminal:
            while time.monotonic() - started < args.timeout:
                rclpy.spin_once(node, timeout_sec=.05)
                now = time.monotonic()
                if (directory / 'STOP').exists():
                    result['reason'] = 'Stopped for investigation; see STOP'
                    break
                if gz.poll() is not None:
                    result['reason'] = 'Gazebo launch exited; see gazebo.log'
                    break
                if controller is None and data['scan'] is not None and data['pose'] is not None:
                    controller = launch(['bash', str(ROOT / 'fp.sh'), '--simulation', '--terminal-debug',
                                         '--ros-args', '-p', 'use_sim_time:=true'], 'controller')
                    print('Sensors ready; controller started.', flush=True)
                    if args.rviz:
                        launch(['bash', str(ROOT / 'fp_rviz_sim.sh'), '--domain', domain], 'rviz')
                        print('RViz decision viewer started on ROS domain ' + domain, flush=True)
                if controller is not None and controller.poll() is not None:
                    result['reason'] = 'Controller exited; see controller.log'
                    break
                if controller is not None and now-data.get('scan_received', now) > 10:
                    result['reason'] = 'Simulation sensors stopped updating; inspect gazebo.log'
                    break
                if controller is not None and not data['debug'] and now-started > 30:
                    result['reason'] = 'No controller telemetry after startup; check clock and controller log'
                    break
                state = (data['debug'].get('obstacle'), data['debug'].get('state'))
                if state != previous_state and state[1] is not None:
                    transitions.append({'elapsed': round(now-started, 2), 'obstacle': state[0], 'state': state[1]})
                    previous_state = state
                    last_progress = now
                    print('STATE:', state, flush=True)
                if data['pose'] is not None:
                    if progress_pose is None or np.linalg.norm(np.array(data['pose'][:2])-progress_pose[:2]) > .03 or abs(data['pose'][2]-progress_pose[2]) > 3:
                        progress_pose, last_progress = data['pose'][:], now
                if now - last_sample >= .5:
                    entry = {'elapsed': round(now-started, 2), 'pose': data['pose'],
                             'command': data['command'], 'debug': data['debug']}
                    if data['scan'] is not None:
                        msg = data['scan']
                        entry['sim_time'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                        entry['lidar_body'] = scan_points(msg, **data['debug'].get('lidar', {'yaw_deg': 180.}))
                    history.write(json.dumps(entry) + '\n')
                    history.flush()
                    last_sample = now
                if now - last_print >= 5:
                    print('t={:.0f}s pose={} cmd={} state={}'.format(now-started,
                          [round(p, 3) for p in data['pose']] if data['pose'] else None,
                          [round(c, 3) for c in data['command']], state), flush=True)
                    if data['scan'] is not None:
                        points = scan_points(data['scan'], **data['debug'].get('lidar', {'yaw_deg': 180.}))
                        frame = render_frame(100, 36, points, data['debug'], data['command'], 'run ' + args.name)
                        terminal.write('\nELAPSED {:.1f}s\n'.format(now-started) + '\n'.join(frame) + '\n')
                        terminal.flush()
                    last_print = now
                if state[1] == 'done':
                    if done_since is None:
                        done_since = now
                    if now - done_since > 2 and data['command'] == (0., 0.):
                        crossed = {t['obstacle'] for t in transitions if t['state'] == 'cross'}
                        if crossed == {1, 2}:
                            result.update(result='complete', reason='both crossings, done state and stopped command observed')
                        else:
                            result['reason'] = 'Done without observing both crossings'
                        break
                if now - last_progress > args.stall:
                    result['reason'] = 'No pose or state progress for {:.0f}s'.format(args.stall)
                    break
            else:
                result['reason'] = 'Time limit reached'
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            stop(child)
        for log in files:
            log.close()
        result.update(elapsed=round(time.monotonic()-started, 2), final_pose=data['pose'],
                      transitions=transitions, final_command=data['command'])
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        node.destroy_node()
        rclpy.shutdown()
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['result'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
