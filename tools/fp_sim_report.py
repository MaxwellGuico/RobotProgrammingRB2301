#!/usr/bin/env python3
"""Plot the measured paths and LiDAR maps retained by fp_sim_check.py."""
import argparse
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rb2301_fp_plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output-prefix', help='Use a distinct report filename prefix')
    options = parser.parse_args()
    count = len(options.runs)
    cols, rows = min(count, 3), math.ceil(count / 3)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    for unused in axes.flat[count:]:
        unused.set_visible(False)
    summaries = []
    output = []
    for axis, name in zip(axes.flat, options.runs):
        directory = ROOT / 'log/fp_sim_checks' / name
        result = json.loads((directory / 'result.json').read_text())
        entries = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]
        entries = [entry for entry in entries if entry['pose'] is not None]
        for entry in entries[::8]:
            pts = np.asarray(entry.get('lidar_body', []))
            if not len(pts):
                continue
            x, y, heading = entry['pose']
            yaw = math.radians(heading)
            c, s = math.cos(yaw), math.sin(yaw)
            world = pts @ np.array([[c, s], [-s, c]]) + [x, y]
            axis.scatter(world[:, 0], world[:, 1], color='#8b949e', s=1, alpha=.25, rasterized=True)
        poses = np.asarray([entry['pose'] for entry in entries])
        last = entries[-1]
        x, y, heading = last['pose']
        yaw = math.radians(heading)
        c, s = math.cos(yaw), math.sin(yaw)
        final_points = np.asarray(last['lidar_body']) @ np.array([[c, s], [-s, c]]) + [x, y]
        wall = final_points[(final_points[:, 0] > x + .15) & (np.abs(final_points[:, 1] - y) < .15)]
        wall_distance = float(wall[:, 0].min() - x) if len(wall) else None
        actors = ET.parse(directory / 'world.sdf').getroot().findall('world/actor')
        panel_x = [float(actor.find('script/trajectory/waypoint/pose').text.split()[0]) for actor in actors]
        result['observed_end_wall_distance_m'] = wall_distance
        result['beyond_actual_panel_planes'] = bool(x > max(panel_x) + .15)
        end_wall = next(item for item in ET.parse(directory / 'world.sdf').getroot()
                        .findall('world/include')
                        if item.findtext('name') == 'nist_maze_wall_120_17')
        # This fixed mesh extends 0.6 m back from its placement x coordinate.
        end_x = float(end_wall.findtext('pose').split()[0]) - .6
        result['world_end_wall_distance_m'] = end_x - x
        result['independent_finish_checks_passed'] = bool(
            result['result'] == 'complete' and result['final_command'] == [0., 0.]
            and result['beyond_actual_panel_planes'] and wall_distance is not None
            and .30 < wall_distance <= .37 and .30 < end_x - x < .40)
        output.append('Seed {}: {} | {:.2f}s | final pose {} | wall {:.3f}m | final cmd {}'.format(
            result['seed'], result['result'], result['elapsed'],
            [round(v, 3) for v in result['final_pose']], wall_distance, result['final_command']))
        for line in (directory / 'controller.log').read_text().splitlines():
            if line.startswith(('Moving obstacle ', 'Obstacle 1 cleared', 'Obstacle 2 cleared', 'completed')):
                output.append('  ' + line)
        axis.plot(poses[:, 0], poses[:, 1], color='#1674b8', linewidth=2, label='Measured robot path')
        axis.plot(*poses[0, :2], 'go', label='Start')
        axis.plot(*poses[-1, :2], 'r*', markersize=13, label='Stopped')
        for transition in result['transitions']:
            if transition['state'] == 'cross':
                entry = min(entries, key=lambda e: abs(e['elapsed']-transition['elapsed']))
                axis.annotate('Cross ' + str(transition['obstacle']), entry['pose'][:2],
                              xytext=(8, 16 if transition['obstacle'] == 1 else -28),
                              textcoords='offset points', fontsize=9,
                              arrowprops={'arrowstyle': '->', 'color': '#30363d'})
        axis.set(xlim=(-.7, 6.5), ylim=(-4.8, .8), aspect='equal',
                 xlabel='Course x (m)', ylabel='Course y (m)',
                 title='Seed {}: {}\n{:.1f}s wall time'.format(result['seed'], result['result'], result['elapsed']))
        axis.grid(alpha=.2)
        axis.legend(loc='upper right', fontsize=8)
        summaries.append(result)
    fig.suptitle('RB2301 FP — actual Gazebo runs\nGrey: observed LiDAR surfaces; blue: odometry trajectory', fontsize=14)
    fig.tight_layout()
    stem = options.output_prefix or ('five_runs' if count == 5 else 'three_runs')
    target = ROOT / 'log/fp_sim_checks' / (stem + '.png')
    fig.savefig(target, dpi=160)
    (target.parent / (stem + '.json')).write_text(json.dumps(summaries, indent=2))
    output.append('Identical controller sources: ' + str(len({
        s.get('controller_sha256', s['source_sha256']) for s in summaries}) == 1))
    summary_name = stem + '_summary.txt' if options.output_prefix else 'summary.txt'
    (target.parent / summary_name).write_text('\n'.join(output) + '\n')
    print('\n'.join(output))
    print(target)


if __name__ == '__main__':
    main()
