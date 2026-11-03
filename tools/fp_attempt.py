#!/usr/bin/env python3
"""One visible Gazebo/RViz/controller attempt. Always close all owned processes."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stop_group(pid):
    """Stop only the processes owned by this attempt, giving ROS time to send its final zero command."""
    for sig, delay in ((signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 0)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        until = time.monotonic()+delay
        while time.monotonic() < until:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)


def controller_terminal(directory):
    """Run the controller in the visible terminal and copy its output to the attempt log."""
    child = subprocess.Popen(['bash', str(ROOT/'fp.sh'), '--ros-args', '-p', 'is_simulation:=true'],
                             cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, start_new_session=True)
    (directory/'controller.pid').write_text(str(child.pid))
    def stop(*_):
        """Forward a terminal interrupt to the controller process group."""
        stop_group(child.pid)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        with (directory/'controller.log').open('w') as log:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        code = child.wait()
    finally:
        stop_group(child.pid)
    (directory/'controller.exit').write_text(str(code))
    return code


def desktop_environment():
    """Remove inherited Snap GUI settings while preserving the caller's ROS domain and distribution."""
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(('SNAP', 'GTK', 'GDK', 'GIO', 'GSETTINGS')):
            env.pop(key, None)
    env['XDG_DATA_DIRS'] = env.get('XDG_DATA_DIRS_VSCODE_SNAP_ORIG', '/usr/local/share:/usr/share')
    env.update(PYTHONUNBUFFERED='1', RB2301_RANDOMIZE='0', RB2301_DEBUG_VISUALS='1')
    return env


def run_attempt(args):
    """Open the three visible applications, watch the attempt, and close every owned process on exit."""
    directory = ROOT/'log'/'fp_attempts'/args.name
    directory.mkdir(parents=True, exist_ok=False)
    sources = ('src/rb2301_fp/rb2301_fp/obstacle_course.py',
               'src/rb2301_fp/rb2301_fp/diagnostics.py',
               'src/rb2301_fp/rviz/fp_robot_thinking.rviz')
    snapshot = directory/'source'
    snapshot.mkdir()
    hashes = {}
    for name in sources:
        data = (ROOT/name).read_bytes()
        (snapshot/Path(name).name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (directory/'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    roles, files = [], []
    env = desktop_environment()
    result = dict(result='failed', reason='interrupted', directory=str(directory))
    def launch(argv, name):
        """Start one application in its own process group and save its output for diagnosis."""
        log = (directory/(name+'.log')).open('w')
        files.append(log)
        child = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        roles.append(child)
        return child
    def interrupted(*_):
        """Leave the watch loop through its cleanup block when the user stops the attempt."""
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        gz = launch(['ros2', 'launch', 'rb2301_gz', 'fp_gazebo.launch.py', 'x:=0.0', 'y:=0.0',
                     'world:='+str(Path(args.world).resolve())], 'gazebo')
        rviz = launch(['bash', str(ROOT/'src/rb2301_fp/rviz_fp.sh')], 'rviz')
        command = 'source '+shlex.quote(str(ROOT/'install/setup.bash'))+'; exec '+shlex.join(
            ['python3', str(Path(__file__).resolve()), '--controller', str(directory)])
        terminal = launch(['gnome-terminal', '--wait', '--title=RB2301 '+args.name,
                           '--', 'bash', '-c', command], 'terminal')
        print('VISIBLE ATTEMPT:', directory, flush=True)
        started = progress = time.monotonic()
        previous_pose = None
        observations = 0
        resumed = False
        while time.monotonic()-started < args.timeout:
            time.sleep(1)
            log = directory/'controller.log'
            content = log.read_text() if log.exists() else ''
            if not resumed and 'Entity creation successful' in (directory/'gazebo.log').read_text():
                subprocess.run(['gz', 'service', '-s', '/world/empty/control', '--reqtype',
                                'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean',
                                '--timeout', '3000', '--req', 'pause: false'],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                resumed = True
            if 'COURSE_COMPLETE:' in content:
                result.update(result='complete', reason='controller confirmed course completion')
                print('COURSE COMPLETE', flush=True)
                time.sleep(5)
                break
            if any(word in content for word in ('Traceback (most recent', 'NO_SAFE_CROSSING:', 'NO_PROGRESS |')):
                result['reason'] = 'controller failure; see controller.log'
                break
            if any(role.poll() is not None for role in (gz, rviz, terminal)):
                result['reason'] = 'a visible process exited'
                break
            count = content.count('SWING_CLEAR_WINDOW:')
            if count > observations:
                observations, progress = count, time.monotonic()
            matches = re.findall(r'\| pose=\[([^]]+)\]', content)
            if matches:
                pose = [float(value) for value in matches[-1].split()]
                if previous_pose is None or (abs(pose[0]-previous_pose[0])+abs(pose[1]-previous_pose[1]) > .03
                                            or abs(pose[2]-previous_pose[2]) > 3):
                    progress, previous_pose = time.monotonic(), pose
            if time.monotonic()-progress > 55:
                result['reason'] = 'no pose progress for 55 seconds'
                break
        else:
            result['reason'] = 'attempt time limit reached'
    except KeyboardInterrupt:
        pass
    finally:
        # The controller is launched by the terminal server, so stop its recorded
        # group explicitly before closing that terminal and the two GUI processes.
        pidfile = directory/'controller.pid'
        if pidfile.exists():
            stop_group(int(pidfile.read_text()))
        for child in reversed(roles):
            stop_group(child.pid)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
        for file in files:
            file.close()
        (directory/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return 0 if result['result'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--name', default=time.strftime('attempt_%Y%m%d_%H%M%S'))
    parser.add_argument('--world', default=str(ROOT/'src/rb2301_gz/worlds/obstacle_course_world_fp.sdf'))
    parser.add_argument('--timeout', type=float, default=900)
    options = parser.parse_args()
    sys.exit(controller_terminal(options.controller) if options.controller else run_attempt(options))
