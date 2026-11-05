"""Gap selection, short commitments, steering, and swept-footprint safety."""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

GAP_LOOKAHEAD = .20
GAP_REACHED_TOLERANCE = .02
PLANNING_HORIZONS = (1.2, .8, .6, .35, .20, .12, .06)


def wrap_rad(angle):
    """Return the shortest signed angle, in radians."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_deg(angle):
    """Wrap degrees into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def rotate_points(points, heading_deg):
    """Rotate 2-D row-vector points counterclockwise by the given degrees."""
    yaw = np.deg2rad(heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return points @ np.array([[c, s], [-s, c]])


def body_overlap(x, y):
    """Depth inside the padded chassis box; zero means outside."""
    return np.maximum(0, np.minimum.reduce((x + .05, .20 - x, .13 - np.abs(y))))


@dataclass
class GapChoice:
    """A fresh scan's openings and its selected short movement target."""
    target: Optional[np.ndarray]
    horizon: float
    directions: np.ndarray
    groups: list


def _centre_group(groups, angles):
    """Prefer less turning among similarly wide openings."""
    widest = max(map(len, groups))
    eligible = [group for group in groups if len(group) >= .9*widest]
    return min(eligible, key=lambda g: (abs(angles[g[len(g)//2]]), -len(g)))


def _stall_diagnostics(node):
    """Summarize why motion stopped without dumping the path history."""
    detail = getattr(node, 'gap_debug', {})
    result = {key: detail.get(key) for key in
              ('motion_status', 'inside_points', 'closest_inside', 'requested', 'safe')}
    result['ages_s'] = {key: round(time.monotonic()-getattr(node, key+'_received'), 2)
                        if hasattr(node, key+'_received') else None for key in ('pose', 'scan')}
    return result


def choose_gap(points, horizons=PLANNING_HORIZONS, commitment=GAP_LOOKAHEAD,
               pose=None, history=None):
    """Prefer distant clear openings while committing only a short safe step."""
    angles = np.deg2rad(np.arange(-85, 86))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    centred = np.asarray(points).reshape(-1, 2) - [.07, 0.]
    along = centred @ directions.T
    across = np.abs(centred[:, 0, None]*directions[:, 1]
                    - centred[:, 1, None]*directions[:, 0])
    entry = along - np.sqrt(np.maximum(.15**2-across**2, 0.))
    clearance = np.min(np.where((along > 0) & (across < .15), entry, np.inf),
                       axis=0, initial=np.inf)
    fallback = None
    for horizon in horizons:
        free = np.flatnonzero(clearance > horizon)
        groups = np.split(free, np.flatnonzero(np.diff(free) > 1)+1)
        groups = [group for group in groups if len(group) >= 3]
        if groups:
            group = _centre_group(groups, angles)
            target = min(commitment, horizon)*directions[group[len(group)//2]]
            fallback = fallback or GapChoice(target, horizon, directions, groups)
            if pose is not None and history is not None and len(history):
                old = np.asarray(history)
                novel = []
                for group in groups:
                    probe = min(.8, horizon)*directions[group[len(group)//2]]
                    world = np.asarray(pose[:2])+rotate_points(probe, pose[2])
                    if np.linalg.norm(old-world, axis=1).min() > .3:
                        novel.append(group)
                if not novel:
                    continue
                group = _centre_group(novel, angles)
                target = min(commitment, horizon)*directions[group[len(group)//2]]
            return GapChoice(target, horizon, directions, groups)
    return fallback or GapChoice(None, horizons[-1], directions, [])


class GapNavigation:
    """ROS-independent navigation methods mixed into the course node."""

    def gap_target(self, points):
        """Select using distant clearance; keep movement commitments short."""
        history = getattr(self, 'gap_history', [])
        old = ([list(p[1:]) for p in history if p[0] < history[-1][0]-.6]
               if history else [])
        choice = choose_gap(points, pose=getattr(self, 'pose', None), history=old)
        self.gap_horizon = choice.horizon
        self.gap_debug = dict(horizon=choice.horizon, history=old)
        self.gap_speed = .06 if choice.horizon < .12 else .18
        return choice.target


    def steer(self, target):
        """Track the waypoint with filtered yaw damping and limited turn acceleration."""
        self.target = None if target is None else list(map(float, target))
        if target is None:
            self.gap_waypoint = None
            self.steering_previous = None
            self.steering_rate = self.steering_turn = 0.
            return 0., 0.
        now, yaw = self.get_clock().now().nanoseconds / 1e9, np.deg2rad(self.pose[2])
        previous, rate = getattr(self, 'steering_previous', None), 0.
        dt = .05 if previous is None else max(0., now-previous[0])
        if previous is not None and dt > 0:
            rate = wrap_rad(yaw - previous[1]) / dt
        filtered = getattr(self, 'steering_rate', 0.)
        self.steering_rate = filtered + dt/(.12+dt)*(rate-filtered)
        self.steering_previous = (now, yaw)
        heading = np.arctan2(target[1], target[0])
        speed = min(self.speed_limit, getattr(self, 'gap_speed', .18),
                    1.5*np.linalg.norm(target))
        forward = speed * max(0., np.cos(heading))**3
        turn = (0. if abs(heading) <= np.deg2rad(2) else
                2*heading - .35*self.steering_rate)
        limit = min(.8, 2*self.speed_limit)
        turn = float(np.clip(turn, -limit, limit))
        previous_turn = getattr(self, 'steering_turn', 0.)
        max_change = 1.5*min(dt, .1)
        self.steering_turn = previous_turn + np.clip(turn-previous_turn, -max_change, max_change)
        return float(forward), float(self.steering_turn)


    def safe_command(self, points, forward, turn):
        """Check the next half-second of forward motion or rotation; reject reverse."""
        if forward < 0:
            return 0., 0.
        times = np.linspace(.05, .5, 8)
        for scale in (1., .5, .2, 0.):
            v, w = forward*scale, turn*scale
            yaw = w*times
            x = v*times if abs(w) < 1e-6 else v*np.sin(yaw)/w
            y = np.zeros_like(times) if abs(w) < 1e-6 else v*(1-np.cos(yaw))/w
            dx, dy = points[:, 0, None] - x, points[:, 1, None] - y
            depth = body_overlap(dx*np.cos(yaw) + dy*np.sin(yaw),
                                 -dx*np.sin(yaw) + dy*np.cos(yaw))
            if not np.any(depth):
                return float(v), float(w)
        return 0., 0.


    def stop_if_stalled(self, forward, turn):
        """Stop on missing motion progress; never replace a command with backing."""
        if forward < 0:
            return 0., 0.
        if not np.any((forward, turn)):
            self.progress = None
            return 0., 0.
        now = self.get_clock().now().nanoseconds / 1e9
        previous = getattr(self, 'progress', None)
        if (previous is None or np.linalg.norm(self.pose[:2] - previous[1][:2]) > .015
                or abs(wrap_deg(self.pose[2] - previous[1][2])) > 5):
            self.progress = now, self.pose.copy()
            self.stall_reported = False
        elif now - previous[0] > 2:
            if not getattr(self, 'stall_reported', False):
                print('Navigation stalled: stopping;', _stall_diagnostics(self), flush=True)
            self.stall_reported = True
            return 0., 0.
        return forward, turn


    def follow_gap(self, points):
        """Track a committed segment, retaining its heading through the short step."""
        history = getattr(self, 'gap_history', [])
        distance = (0. if not history else
                    float(np.linalg.norm(self.pose[:2]-history[-1][1:])))
        if not history or distance >= .1:
            travelled = history[-1][0]+distance if history else 0.
            history.append((travelled, float(self.pose[0]), float(self.pose[1])))
            self.gap_history = history[-300:]
        waypoint = getattr(self, 'gap_waypoint', None)
        target = (None if waypoint is None else
                  rotate_points(waypoint-self.pose[:2], -self.pose[2]))
        origin = getattr(self, 'gap_origin', self.pose[:2])
        length = 0. if waypoint is None else np.linalg.norm(waypoint-origin)
        progress = (0. if length < 1e-9 else
                    np.dot(self.pose[:2]-origin, waypoint-origin)/length)
        if target is None or progress >= length-GAP_REACHED_TOLERANCE:
            target = self.gap_target(points)
            self.gap_origin = self.pose[:2].copy()
            self.gap_waypoint = (None if target is None else
                                 self.pose[:2] + rotate_points(target, self.pose[2]))
        tracking = target
        if self.gap_waypoint is not None:
            segment = self.gap_waypoint-self.gap_origin
            direction = segment/np.linalg.norm(segment)
            along = np.dot(self.pose[:2]-self.gap_origin, direction)
            carrot = self.gap_origin+(max(0., along)+GAP_LOOKAHEAD)*direction
            tracking = rotate_points(carrot-self.pose[:2], -self.pose[2])
        requested = self.steer(tracking)
        self.target = None if target is None else list(map(float, target))
        command = self.safe_command(self.body_points, *requested)
        body = np.asarray(self.body_points).reshape(-1, 2)
        inside = body[body_overlap(body[:, 0], body[:, 1]) > 0]
        closest = inside[np.argmin(np.linalg.norm(inside, axis=1))].tolist() if len(inside) else None
        blocked = np.any(requested) and not np.any(command)
        status = ('no_clear_gap' if target is None else
                  'return_inside_footprint' if blocked and len(inside) else
                  'swept_path_blocked' if blocked else
                  'motion_commanded')
        self.gap_debug = dict(getattr(self, 'gap_debug', {}), motion_status=status,
                              inside_points=len(inside), closest_inside=closest,
                              requested=list(map(float, requested)), safe=list(map(float, command)))
        command = self.stop_if_stalled(*command)
        if not np.any(command):
            self.gap_waypoint = None
        return command


