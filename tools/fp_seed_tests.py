#!/usr/bin/env python3
"""Run fp in three reproducible, isolated Gazebo worlds without editing the active world.

Usage (from the workspace):
  source install/setup.bash
  python3 tools/fp_seed_tests.py --seconds 120

Seeds 0, 2, and 3 avoid the known seed-1 spawn overlap.
Each seed records its exact world, controller source, telemetry, ROS/Gazebo logs,
and summary. Progress and circling are reported separately from completion.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/rb2301_fp/rb2301_fp/obstacle_course.py'


def stop_owned(child):
    """Signal only this test's process group, never all Gazebo/ruby processes."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            continue
        # The group can outlive its launch parent; terminate any remaining members.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        break


def run_seed(args):
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    from rclpy.qos import qos_profile_sensor_data

    directory = Path(args.output) / f'seed_{args.seed}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'obstacle_course.py').write_bytes(SOURCE.read_bytes())
    spec = importlib.util.spec_from_file_location(
        'seed_world_modifier', ROOT / 'src/rb2301_fp/rb2301_fp/obstacle_modifier.py')
    modifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modifier)
    modifier.overwrite_file = str(directory / 'world.sdf')
    modifier.original_file = str(ROOT / 'src/rb2301_gz/worlds/obstacle_course_world_fp_original.sdf')
    np.random.seed(args.seed)
    modifier.modify_sdf_file()
    world = ET.parse(modifier.overwrite_file).getroot().find('world')
    gate_poses = [np.fromstring(e.findtext('pose'), sep=' ') for e in world.findall('include')
                  if 'configurable' in e.findtext('name', '')]
    # The 1.2 m wall mesh has its centre 0.6 m along local y; yaw=pi/2.
    gate_centre = [float(gate_poses[0][0] - .6),
                   float((gate_poses[0][1] + gate_poses[1][1]) / 2)]
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(('SNAP', 'GTK', 'GDK', 'GIO', 'GSETTINGS')):
            env.pop(key, None)
    env.update(RB2301_RANDOMIZE='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               XDG_DATA_DIRS='/usr/local/share:/usr/share', ROS_LOG_DIR=str(directory / 'ros'))
    children, handles, samples = [], [], []
    result = dict(seed=args.seed, gate_centre=gate_centre,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  requested_observation_seconds=args.seconds, reason='interrupted')
    latest = dict(pose=None, cmd=None, scan_count=0, scan_age=None)
    last_scan = None
    node = None
    telemetry = (directory / 'telemetry.jsonl').open('w')
    try:
        rclpy.init()
        node = rclpy.create_node('fp_seed_observer')
        def odom(msg):
            p = msg.pose.pose
            q = p.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            latest['pose'] = [p.position.x, p.position.y, math.degrees(yaw)]
        def scan(msg):
            nonlocal last_scan
            last_scan = time.monotonic()
            latest['scan_count'] += 1
        def command(msg):
            latest['cmd'] = [msg.linear.x, msg.angular.z]
        node.create_subscription(Odometry, 'odom', odom, qos_profile_sensor_data)
        node.create_subscription(LaserScan, 'scan', scan, qos_profile_sensor_data)
        node.create_subscription(Twist, 'cmd_vel', command, 20)
        for name, command_args in [
            ('gazebo', ['ros2', 'launch', 'rb2301_gz', 'fp_gazebo.launch.py',
                        'x:=0.0', 'y:=0.0', 'world:=' + modifier.overwrite_file]),
            ('controller', ['bash', 'fp.sh'])]:
            handle = (directory / (name + '.log')).open('w')
            handles.append(handle)
            command_text = 'source install/setup.bash; exec ' + shlex.join(command_args)
            children.append(subprocess.Popen(['bash', '-c', command_text], cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        startup = time.monotonic()
        ready = None
        next_sample = next_report = startup
        while True:
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if latest['pose'] is not None and last_scan is not None and ready is None:
                ready = now
            if any(c.poll() is not None for c in children):
                result['reason'] = 'process exited; inspect logs'
                break
            if ready is None and now-startup > 35:
                result['reason'] = 'no pose/scan within startup limit'
                break
            if ready is not None and now-ready >= args.seconds:
                result['reason'] = 'observation window complete'
                break
            if ready is not None and now >= next_sample:
                sample = dict(elapsed=now-ready, pose=latest['pose'], cmd=latest['cmd'],
                              scan_count=latest['scan_count'], scan_age=now-last_scan)
                samples.append(sample)
                telemetry.write(json.dumps(sample) + '\n')
                telemetry.flush()
                next_sample = now + .5
            if ready is not None and now >= next_report:
                print(json.dumps(dict(seed=args.seed, elapsed=round(now-ready),
                                      pose=latest['pose'])), flush=True)
                next_report = now + 20
        result['observed_seconds'] = 0 if ready is None else time.monotonic()-ready
    finally:
        # Stop the controller first so its final zero command can reach Gazebo.
        for child in reversed(children):
            stop_owned(child)
        for handle in handles:
            handle.close()
        telemetry.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if samples:
            xy = np.array([s['pose'][:2] for s in samples])
            result.update(final_pose=samples[-1]['pose'],
                          path_length=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                          max_distance_from_start=float(np.linalg.norm(xy-xy[0], axis=1).max()),
                          unique_25cm_cells=len({tuple(np.floor(p/.25).astype(int)) for p in xy}),
                          closest_gate_distance=float(np.linalg.norm(xy-np.array(gate_centre), axis=1).min()))
            # Repeated visits require leaving a cell and returning, not just waiting.
            cells = [tuple(np.floor(p/.25).astype(int)) for p in xy]
            entries = [c for i,c in enumerate(cells) if i == 0 or c != cells[i-1]]
            result['cell_reentries'] = len(entries)-len(set(entries))
            intervals = []
            for end in range(len(samples)):
                begin = next((i for i,s in enumerate(samples)
                              if s['elapsed'] >= samples[end]['elapsed']-30), 0)
                if samples[end]['elapsed']-samples[begin]['elapsed'] < 29:
                    continue
                segment = xy[begin:end+1]
                length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
                displacement = float(np.linalg.norm(segment[-1]-segment[0]))
                if length > 1.5 and displacement < .5:
                    intervals.append(round(samples[end]['elapsed'], 1))
            result['loop_window_end_times'] = intervals
        controller_log = (directory/'controller.log').read_text()
        result['replans'] = controller_log.count('Replanning:')
        result['target_changes'] = controller_log.count('New local target:')
        (directory/'summary.json').write_text(json.dumps(result, indent=2))
        print('RESULT ' + json.dumps(result), flush=True)
    return 0 if result['reason'] == 'observation window complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 2, 3])
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seed is not None:
        return run_seed(args)
    args.output = (args.output or ROOT/'log/fp_seed_tests'/time.strftime('%Y%m%d_%H%M%S')).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    status = 0
    for index, seed in enumerate(args.seeds):
        env = os.environ.copy()
        env.update(ROS_DOMAIN_ID=str(200+index), GZ_PARTITION='fp_seed_'+uuid.uuid4().hex,
                   ROS_LOG_DIR=str(args.output/'observer_ros'), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(seed),
                   '--seconds', str(args.seconds), '--output', str(args.output)]
        status |= subprocess.run(command, cwd=ROOT, env=env).returncode
    summaries = [json.loads(p.read_text()) for p in sorted(args.output.glob('seed_*/summary.json'))]
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))
    print('BATCH RESULTS: ' + str(args.output/'summary.json'), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
