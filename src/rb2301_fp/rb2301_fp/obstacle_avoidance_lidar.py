import argparse
from collections import deque
import math
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

# LiDAR-only obstacle_course.py: the pose (odom / OptiTrack) is only logged. Everything is in the robot frame (x ahead, y left,
# origin at the LiDAR); heading is held square to the walls and distances/turns are measured by scan matching (lidar_odometry).
# FSM: CORRIDOR_1 -> TURN right -> CORRIDOR_2 (gate on left) -> GATE_APPROACH -> TURN left -> GATE_CROSS -> DYNAMIC x2 -> finished

def wall_angle(points, max_range=1.5):
    '''Angle (deg, in [-45, 45)) of the nearby walls relative to the robot's nose, folded onto 90-degree steps, and the
    total wall length used (m). Segments between neighbouring scan points show which way a wall runs; walls along AND across vote.'''
    if points is None or len(points) < 3:
        return 0.0, 0.0
    near = np.linalg.norm(points, axis=1) < max_range
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = (lengths > 0.02) & (lengths < 0.15) & near[:-1] & near[1:]
    total = float(lengths[valid].sum())
    if total == 0.0:
        return 0.0, 0.0
    angles, weights = np.arctan2(steps[valid, 1], steps[valid, 0]), lengths[valid]
    mean4 = math.atan2(np.sum(weights * np.sin(4 * angles)), np.sum(weights * np.cos(4 * angles)))
    return math.degrees(mean4 / 4), total

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.is_simulation = len(msg.ranges) <= 360

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course using only the lidar scan data (pose is logged, not used)'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode (LiDAR only)")
        self.is_simulation, self.calibrate = is_simulation, calibrate # calibrate: only print poses and scan info, never drive
        self.max_translate_velocity = 1.4 if is_simulation else 0.3 # Please keep this in place; 0.3m/s is more than fast enough
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)) # Subscribe to LiDAR scan data
        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data (logging only)
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")
            self.map_sub = self.create_subscription(PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose', self.optitrack_callback, qos_profile)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.
        self.pose = None
        # ---- Parameters (see params/real_robot.yaml for what each does; unused ones there are ignored) ----
        f = lambda name, default: float(self.declare_parameter(name, default).value)
        self.lidar_yaw = math.radians(f('lidar_yaw_deg', 180.0))
        self.body_front, self.body_rear, self.half_width = f('body_front', 0.07), f('body_rear', 0.15), 0.11
        self.robot_safe_radius, self.clearance_margin = f('robot_safe_radius', 0.14), f('clearance_margin', 0.03)
        self.target_buffer, self.navigation_speed = f('target_buffer', 0.15), f('navigation_speed', 0.2)
        self.lookahead, self.min_corridor_travel = f('lookahead_distance', 2.0), f('min_corridor_travel', 0.6)
        self.side_depth, self.opening_min_width = f('opening_side_depth', 1.1), f('opening_min_width', 0.4)
        self.gate_min_width, self.gate_max_width = f('gate_min_width', 0.38), f('gate_max_width', 0.80)
        self.gate_max_lateral, self.gate_line_points = f('gate_max_lateral', 2.2), int(f('gate_line_points', 12))
        self.gate_standoff, self.gate_exit, self.gate_arrive = f('gate_standoff', 0.45), f('gate_exit', 0.18), f('gate_arrive', 0.03)
        self.look_time, self.lane_min_scans = f('look_time', 3.0), int(f('lane_min_scans', 3))
        self.lane_min_sweep, self.lane_nose_gap = f('lane_min_sweep', 0.2), f('lane_nose_gap', 0.07)
        self.min_creep_speed, self.cross_speed = f('min_creep_speed', 0.06), f('cross_speed', 0.4)
        self.cross_margin, self.lane_block_time = f('cross_margin', 0.06), f('lane_block_time', 0.2)
        self.lane_clear_time, self.lane_clear_gap = f('lane_clear_time', 0.2), f('lane_clear_gap', 0.15)
        self.cross_clear, self.strafe_speed = f('cross_clear', 0.10), f('strafe_speed', 0.15)
        self.wall_hug_gap, self.advance_step = f('wall_hug_gap', 0.08), f('advance_step', 0.4)
        self.lanes_required, self.end_stop_gap = int(f('lanes_required', 2)), f('end_stop_gap', 0.3)
        self.turn_rate, self.min_turn_rate, self.Kp_theta = f('turn_rate', 0.6), f('min_turn_rate', 0.2), 2.0  # [rad/s], gain
        self.points, self.scan, self.scan_time, self.history = None, None, None, deque()
        self.travel, self.turned, self.dodge = 0.0, 0.0, None     # scan-matched since state entry; gap being dodged to
        self.state, self.since, self.wall_seen, self.last_log = None, time.monotonic(), False, 0.0
        self.gate, self.gate_hist, self.gate_seen_scan = None, deque(maxlen=5), None
        self.dyn_phase, self.lane, self.lanes_crossed, self.at_wall, self.behind_x = None, None, 0, False, None
        self.set_state('CORRIDOR_1')

    def sub_scan_callback(self, msg):
        """Scan subscriber: robot-frame points plus a lookup table of ranges for see-through checks"""
        r = np.asarray(msg.ranges, dtype=float)
        if not len(r) or msg.angle_increment <= 0:
            return
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment + self.lidar_yaw
        ok = np.isfinite(r) & (r >= max(msg.range_min, 0.05)) & (r <= min(msg.range_max, 5.0))
        self.points = np.column_stack((r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])))
        # No return (inf / NaN / 0 on some real LiDARs) = open as far as we care; too short = unknown (0)
        ranges = np.where(ok, r, np.where(np.isposinf(r) | np.isnan(r) | (r == 0.0), 5.0, 0.0))
        self.scan_time = time.monotonic()
        if self.scan is not None:
            self.lidar_odometry(self.scan['points'], self.points)
        self.scan = dict(t=self.scan_time, a0=msg.angle_min + self.lidar_yaw, inc=msg.angle_increment,
                         ranges=ranges, points=self.points)
        self.history.append(self.scan)
        while self.scan_time - self.history[0]['t'] > 1.5:
            self.history.popleft()

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def optitrack_callback(self, msg:PoseStamped):
        '''2D pose from the Optitrack node, raw Optitrack frame (logged only; navigation is LiDAR-only)'''
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.pose = np.array((msg.pose.position.x, msg.pose.position.y, heading))
        return self.pose

    def odometer_callback(self, msg):
        '''2D pose from the Gazebo odometer (logged only; navigation is LiDAR-only)'''
        return self.optitrack_callback(msg.pose)  # same x, y, heading extraction

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def log(self, text, every=1.0):
        '''Rate-limited info log (the FSM runs at 20 Hz).'''
        if time.monotonic() - self.last_log > every:
            self.last_log = time.monotonic()
            self.get_logger().info(text)

    def set_state(self, state, message=None):
        '''Enter an FSM state: reset the scan-matched travel/turn counters and cue timers.'''
        self.state, self.since, self.travel, self.turned, self.cue_since = state, time.monotonic(), 0.0, 0.0, 0.0
        self.get_logger().info(message or f"State -> {state}")

    def scan_match(self, prev, cur, iters=10):
        '''Point-to-line ICP: (R, t) with R @ p_cur + t = p_prev (robot motion between two scans). Pairs > 3x the median apart
        are dropped (moving obstacles); along a featureless corridor the unobservable direction stays zero, not guessed.'''
        prev, cur = prev[::max(1, len(prev) // 360)], cur[::max(1, len(cur) // 360)]
        tang = np.roll(prev, -1, axis=0) - np.roll(prev, 1, axis=0)
        length = np.linalg.norm(tang, axis=1)
        ok = (length > 1e-4) & (length < 0.2)  # neighbours on the same surface
        prev, normals = prev[ok], np.column_stack((-tang[ok, 1], tang[ok, 0])) / length[ok, None]
        R, t = np.eye(2), np.zeros(2)
        for _ in range(iters):
            if len(prev) < 30 or len(cur) < 30:
                return None
            moved = cur @ R.T + t
            d = np.linalg.norm(moved[:, None, :] - prev[None, :, :], axis=2)
            nn = d.argmin(axis=1); dist = d[np.arange(len(moved)), nn]
            keep = dist < min(0.3, max(0.05, 3 * np.median(dist)))
            if np.count_nonzero(keep) < 30:
                return None
            A, B, N = moved[keep], prev[nn[keep]], normals[nn[keep]]
            J = np.column_stack((N[:, 1] * A[:, 0] - N[:, 0] * A[:, 1], N))  # d(residual)/d(theta, tx, ty)
            x = np.linalg.lstsq(J, -np.sum(N * (A - B), axis=1), rcond=1e-2)[0]
            dR = np.array(((math.cos(x[0]), -math.sin(x[0])), (math.sin(x[0]), math.cos(x[0]))))
            R, t = dR @ R, dR @ t + x[1:]
            if np.max(np.abs(x)) < 1e-4:
                break
        return R, t

    def lidar_odometry(self, prev, cur):
        '''Measure the motion since the last scan by scan matching (the sim and the motors do not move exactly as
        commanded), add it to the travel/turn counters and shift tracked robot-frame features into the new frame.'''
        match = self.scan_match(prev, cur)
        if match is None or np.linalg.norm(match[1]) > 0.15:
            return  # too few matched points, or an implausible jump for 0.1 s: skip this pair
        R, t = match
        self.travel += float(t[0])
        self.turned += math.atan2(R[1, 0], R[0, 0])  # signed, + = left
        shift = lambda q: R.T @ (np.asarray(q, dtype=float) - t)
        if self.gate is not None:
            self.gate['centre'] = shift(self.gate['centre'])
        if self.lane is not None:
            self.lane['x'] = float(shift((self.lane['x'], 0.0))[0])
        self.behind_x = None if self.behind_x is None else float(shift((self.behind_x, 0.0))[0])
        self.dodge = None if self.dodge is None else float(shift((0.0, self.dodge))[1])

    def hold(self, min_rate=0.0):
        '''Turn rate that keeps the robot square to the walls (0 if few seen); at least min_rate when > 1 deg off (motor dead band).'''
        angle, length = wall_angle(self.points)
        w = self.Kp_theta * math.radians(angle) if length > 0.5 else 0.0
        return math.copysign(max(abs(w), min_rate), w) if w and abs(angle) > 1.0 else w
    def creep_speed(self, error):
        '''Speed for short positioning moves: proportional, never below min_creep_speed (motors stall).'''
        return float(math.copysign(min(0.2, max(self.min_creep_speed, 1.5 * abs(error))), error))
    def free_depth(self, p, y_off=0.0, half=None):
        '''Gap between the nose and the nearest point in a body-wide strip ahead, offset sideways by y_off.'''
        m = (np.abs(p[:, 1] - y_off) < (self.half_width + 0.02 if half is None else half)) & (p[:, 0] > self.body_front - 0.02)
        return float(np.min(p[m, 0])) - self.body_front if np.any(m) else float('inf')
    def side_gap(self, p, sign):
        '''Gap between the robot's side (+1 left, -1 right) and the nearest point beside the body.'''
        beside = (p[:, 0] > -self.body_rear + 0.01) & (p[:, 0] < self.body_front + 0.03) & (sign * p[:, 1] > 0)
        return float(np.min(sign * p[beside, 1])) - self.half_width if np.any(beside) else float('inf')
    def lateral_clear(self, p, y_off):
        '''Can the body slide sideways to y_off without touching anything alongside it?'''
        lo, hi = min(0.0, y_off) - self.half_width - 0.02, max(0.0, y_off) + self.half_width + 0.02
        return not np.any((p[:, 0] > -self.body_rear) & (p[:, 0] < self.body_front + 0.05) & (p[:, 1] > lo) & (p[:, 1] < hi))
    def strip_blocked(self, p, strip_end):
        '''Anything in the robot-width strip straight ahead, up to strip_end (robot x)?'''
        return np.count_nonzero((p[:, 0] > self.body_front + 0.02) & (p[:, 0] < strip_end) &
                                (p[:, 1] > -(self.half_width + 0.03)) & (p[:, 1] < self.half_width + self.cross_margin)) >= 2
    def ray_range(self, point):
        '''Range the newest scan measured along the ray through a robot-frame point.'''
        angle = math.atan2(point[1], point[0]) - self.scan['a0']
        return float(self.scan['ranges'][int(round(angle / self.scan['inc'])) % len(self.scan['ranges'])])
    def side_view(self, p, sign, depth):
        '''(wall_beside, opening, opening_centre_x) on one side (+1 left, -1 right): a side wall fills most
        10 cm slices over +/-0.3 m; an opening is a run of empty slices in [-0.2, 0.6] (a box cannot hide it).'''
        along, lat = p[:, 0], sign * p[:, 1]
        strip = (lat > 0) & (lat < depth)
        wall = sum(bool(np.any(strip & (along >= a) & (along < a + 0.1))) for a in np.arange(-0.3, 0.3, 0.1)) >= 4
        run = best = 0; end = 0.0
        for a in np.arange(-0.2, 0.6, 0.1):
            run = run + 1 if not np.any(strip & (along >= a) & (along < a + 0.1)) else 0
            if run > best:
                best, end = run, a + 0.1
        return wall, best * 0.1 >= self.opening_min_width - 1e-6, end - 0.05 * best

    def drive_corridor(self, p):
        '''Local avoidance without a map: pick the sideways offset whose body-wide strip is free the furthest,
        preferring the middle between the side walls, then drive forward along it while sliding into it.'''
        half = self.robot_safe_radius + self.clearance_margin
        left, right = self.side_gap(p, 1), self.side_gap(p, -1)
        centre = float(np.clip(0.5 * (left - right), -0.3, 0.3)) if np.isfinite(left) and np.isfinite(right) else 0.0
        best, cands = None, {}
        for y_off in np.arange(-0.6, 0.61, 0.05):
            if self.lateral_clear(p, y_off):
                cands[float(y_off)] = depth = min(self.free_depth(p, y_off, half), self.lookahead)
                best = max(best or (-9, 0, 0), (depth - 0.8 * abs(y_off - centre), float(y_off), depth))
        if best is None:
            self.move_2D(0, 0, self.hold())
            return self.log("No free strip ahead; waiting")
        ahead = self.free_depth(p, 0.0, half)  # keep sliding to the gap already chosen while it stays open (no dithering)
        keep = min(cands, key=lambda y: abs(y - self.dodge)) if self.dodge is not None and cands else None
        y_off = keep if keep is not None and abs(keep - self.dodge) < 0.08 and cands[keep] > min(ahead + 0.15, 0.5) else best[1]
        self.dodge = y_off if abs(y_off) > 0.05 and ahead < 0.5 else None
        vx = 0.0 if abs(y_off) > 0.15 and ahead < 0.3 else float(np.clip(1.5 * (ahead - self.target_buffer), 0.0, self.navigation_speed))  # sidestep first
        vy = math.copysign(min(self.strafe_speed, max(0.1, 1.5 * abs(y_off))), y_off) if abs(y_off) > 0.02 else 0.0
        self.move_2D(vx, vy, self.hold())  # (slower slides than 0.1 m/s barely move)

    # ------------------------------ helpers: corridor turns ------------------------------
    def corridor_cue(self, p, now):
        '''Corridor 1: wall on the right seen, then an opening there (or the end wall) after min_corridor_travel.'''
        wall, opening, centre_x = self.side_view(p, -1, self.side_depth)
        self.wall_seen |= wall
        end_wall = self.free_depth(p, 0.0, 0.3) < 0.35
        self.log(f"CORRIDOR_1: travelled {self.travel:.2f} m (need > {self.min_corridor_travel:.2f}), "
                 f"right wall seen={self.wall_seen}, opening={opening}, end wall={end_wall}")
        if not (self.travel > self.min_corridor_travel and ((self.wall_seen and opening) or end_wall)):
            self.cue_since = 0.0
            return False
        self.cue_since = self.cue_since or now
        if now - self.cue_since < 0.3:  # must persist over a few scans
            return False
        self.turn_sign, self.turn_next, self.creep = -1, 'CORRIDOR_2', max(0.0, centre_x) if opening else 0.0  # centre level with the opening
        self.set_state('TURN', f"Corridor turn ({'side opening' if opening else 'end wall'})")
        return True

    def step_turn(self, p, now):
        '''Turn 90 deg (turn_sign +1 left, -1 right) by the scan-matched rotation, after creeping `creep` m forward,
        finished by squaring up to the walls (wall_angle -> 0).'''
        if self.travel < self.creep - 0.01 and now - self.since < 6.0:
            return self.move_2D(self.creep_speed(self.creep - self.travel), 0, self.hold())
        (angle, length), turned = wall_angle(self.points), self.turn_sign * self.turned
        if turned < math.radians(75) and now - self.since < 30.0:
            self.move_2D(0, 0, self.turn_sign * self.turn_rate)
        elif (length > 0.5 and abs(angle) < 3.0) or turned > math.radians(110) or now - self.since > 30.0:
            self.move_2D()
            self.wall_seen = False
            self.set_state(self.turn_next, f"Turned {'left' if self.turn_sign > 0 else 'right'} "
                           f"({math.degrees(turned):.0f} deg by scan matching, walls {angle:+.1f} deg off)")
        else:
            self.move_2D(0, 0, self.turn_sign * self.turn_rate if length < 0.5 else self.hold(self.min_turn_rate))

    # ------------------------------ helpers: gate (corridor 2, left wall) ------------------------------
    def gate_candidates(self, p):
        '''Gaps of gate width in wall lines on the left: lines are 5 cm lateral bins holding many points; the
        gate is in the OUTER (furthest long) wall, has wall on both ends and can be seen through (not a shadow).'''
        near = (p[:, 1] > 0.25) & (p[:, 1] < self.gate_max_lateral) & (p[:, 0] > -1.0) & (p[:, 0] < 3.0)
        along, lateral = p[near, 0], p[near, 1]
        self.gate_info = f"{len(lateral)} pts on the left"
        if len(lateral) < 10:
            return []
        bins = np.arange(0.25, self.gate_max_lateral + 0.05, 0.05)
        counts, lines = np.histogram(lateral, bins)[0], []
        for i in range(len(counts)):
            on_line = np.abs(lateral - (bins[i] + 0.025)) < 0.06
            if counts[i] < self.gate_line_points or counts[i] < counts[max(0, i - 1):i + 2].max() or np.count_nonzero(on_line) < self.gate_line_points:
                continue
            line = float(np.median(lateral[on_line]))
            positions = np.sort(along[np.abs(lateral - line) < 0.05])
            lines.append((line, positions, 0.05 * len(np.unique(np.floor(positions / 0.05)))))
        long_lines = [l for l, _, cov in lines if cov >= 0.8]
        self.gate_info += f", lines(lateral m, length m)={[(round(l, 2), round(c, 2)) for l, _, c in lines]}"
        if not long_lines:
            return []
        outer, candidates, gaps = max(long_lines), [], []
        for line, positions, _ in [l for l in lines if l[0] >= outer - 0.1]:
            for a, b in zip(positions[:-1], positions[1:]):
                width = b - a
                gaps += [round(width, 2)] if width >= 0.2 else []
                if not self.gate_min_width <= width <= self.gate_max_width:
                    continue
                if (np.count_nonzero((positions > a - 0.3) & (positions <= a)) < 3 or
                        np.count_nonzero((positions >= b) & (positions < b + 0.3)) < 3):
                    continue  # a real gate is a break between two walls, not the end of a wall
                samples = [np.array((a + f * width, line)) for f in (0.3, 0.5, 0.7)]
                if any(self.ray_range(s) < np.linalg.norm(s) + 0.3 for s in samples):
                    continue  # cannot see through it: a shadow behind a nearer object
                candidates.append(dict(centre=np.array((0.5 * (a + b), line)), width=width))
        self.gate_info += f", outer={outer:.2f}, gaps={gaps}"
        return candidates

    def update_gate(self, p):
        '''Confirm a gate seen in at least 3 of the last 5 scans within 15 cm (the robot moves ~3 cm per scan).'''
        if self.gate is not None or self.scan_time == self.gate_seen_scan:
            return False
        self.gate_seen_scan = self.scan_time
        self.gate_hist.append(self.gate_candidates(p))
        self.log(f"Gate search: {self.gate_info}, candidates={len(self.gate_hist[-1])}")
        for cand in self.gate_hist[-1]:
            matches = [cand] + [c[0] for older in list(self.gate_hist)[:-1]
                                for c in [[o for o in older if np.linalg.norm(o['centre'] - cand['centre']) < 0.15]] if c]
            if len(matches) >= 3:
                self.gate = dict(centre=np.mean([m['centre'] for m in matches], axis=0),
                                 width=float(np.mean([m['width'] for m in matches])))
                self.get_logger().info(f"Gate confirmed: {self.gate['centre'].round(2)} (robot frame), "
                                       f"width={self.gate['width']:.2f} m ({len(matches)}/5 scans)")
                return True
        return False

    def track_gate(self, p):
        '''While lining up, re-measure the gate each scan (nearest fresh candidate to the scan-matched one).'''
        if self.scan_time == self.gate_seen_scan:
            return
        self.gate_seen_scan = self.scan_time
        close = [c for c in self.gate_candidates(p) if np.linalg.norm(c['centre'] - self.gate['centre']) < 0.25]
        if close:  # average with the nearest fresh measurement
            self.gate['centre'] = 0.5 * (self.gate['centre'] + min(close, key=lambda c: np.linalg.norm(c['centre'] - self.gate['centre']))['centre'])

    def step_gate_approach(self, p, now):
        '''Slide to the line-up point gate_standoff in front of the gate (shrunk if the right wall is close), then turn left.'''
        self.track_gate(p)
        gx, gy = self.gate['centre']
        room = self.side_gap(p, -1) + self.half_width  # centre to the right wall
        standoff = max(0.25, min(self.gate_standoff, gy + room - self.robot_safe_radius - self.clearance_margin))
        ex, ey = gx, gy - standoff
        self.log(f"Gate approach: gate at ({gx:+.2f}, {gy:+.2f}), line-up error ({ex:+.2f}, {ey:+.2f}) m")
        if math.hypot(ex, ey) < self.gate_arrive or now - self.since > 15.0:
            self.move_2D()
            self.cross_dist = standoff + self.gate_exit
            self.turn_sign, self.turn_next, self.creep = 1, 'GATE_CROSS', 0.0
            return self.set_state('TURN', f"Gate: lined up {standoff:.2f} m from it, turning left")
        vx = self.creep_speed(ex) if abs(ex) > 0.01 and (ex < 0 or self.free_depth(p) >= ex) else 0.0  # blocked: slide first
        vy = float(np.clip(self.creep_speed(ey) if abs(ey) > 0.01 else 0.0, -self.strafe_speed, self.strafe_speed))
        self.move_2D(vx, vy, self.hold())

    def step_gate_cross(self, p, now):
        '''Commit: drive straight through, centred between the two gate posts, until the LiDAR is gate_exit past the line.'''
        posts = (p[:, 0] > -0.05) & (p[:, 0] < self.cross_dist - self.travel + 0.2) & (np.abs(p[:, 1]) < 0.45)
        left, right = p[posts & (p[:, 1] > 0), 1], p[posts & (p[:, 1] < 0), 1]
        centre = 0.5 * (float(np.min(left)) + float(np.max(right))) if len(left) and len(right) else 0.0
        remaining, nose = self.cross_dist - self.travel, self.free_depth(p)
        if remaining < 0.02 or now - self.since > 12.0 or (self.travel > 0.5 * self.cross_dist and nose < self.lane_nose_gap):
            self.move_2D()
            self.behind_x = -(self.gate_exit + 0.01)  # the gate wall is just behind us
            return self.set_state('DYNAMIC', f"Gate crossed (nose {nose:.2f} m from the next thing ahead)")
        self.move_2D(min(self.navigation_speed, 1.5 * remaining + 0.05, 1.5 * nose), float(np.clip(1.5 * centre, -0.1, 0.1)), self.hold())

    # ------------------------------ helpers: moving obstacles ------------------------------
    def dynamic_points(self, since, lag=0.4):
        '''Points in the newest scan that were not there ~lag s earlier. Only scans taken after `since`
        (when the robot stopped) are compared: a moving robot re-samples walls, which looks like motion.'''
        newest = self.history[-1]
        older = [h for h in self.history if since <= h['t'] <= newest['t'] - lag]
        if not older or not len(newest['points']) or not len(older[-1]['points']):
            return np.empty((0, 2))
        nearest = np.min(np.linalg.norm(newest['points'][:, None, :] - older[-1]['points'][None, :, :], axis=2), axis=1)
        return newest['points'][nearest > 0.07]

    def lane_ahead(self, dyn, ids):
        '''Nearest moving obstacle ahead as dict(x=its centre line, thick=half its thickness), or None. A real
        one shows up in several look scans AND sweeps sideways; flicker on a static box edge does neither.'''
        ahead = (dyn[:, 0] > self.body_front) & (dyn[:, 0] < 1.6) & (np.abs(dyn[:, 1]) < 2.5)
        along, lateral, ids = dyn[ahead, 0], dyn[ahead, 1], ids[ahead]
        for start in np.sort(along):
            band = (along >= start) & (along < start + 0.15)
            if (np.count_nonzero(band) >= 3 and len(np.unique(ids[band])) >= self.lane_min_scans and
                    np.ptp(lateral[band]) >= self.lane_min_sweep):
                return dict(x=float(np.median(along[band])), thick=float(np.clip(np.ptp(along[band]) / 2, 0.02, 0.2)))
        return None
    def set_dyn(self, phase, now, message=None):
        self.dyn_phase, self.dyn_since = phase, now
        message and self.get_logger().info(message)

    def observe(self, p, now):
        '''Against the wall an obstacle cannot pass through our strip: it comes in, turns and leaves the same
        way. Cross only when it is MOVING AWAY with a gap (clear alone also happens while it is still coming).'''
        lane = self.lane
        face, far_face = lane['x'] - lane['thick'], lane['x'] + lane['thick']
        strip_end, left_edge = far_face + 0.05, self.half_width + self.cross_margin
        if self.scan_time != lane.get('scan'):
            lane['scan'] = self.scan_time
            band = (p[:, 0] > face - 0.05) & (p[:, 0] < strip_end) & (p[:, 1] > left_edge) & (p[:, 1] < 2.0)
            lane.setdefault('edges', deque(maxlen=30)).append((now, float(np.min(p[band, 1])) if np.any(band) else float('inf')))
            if self.strip_blocked(p, strip_end):
                lane['blocked_since'] = lane.get('blocked_since') or now
                lane['was_blocked'] |= now - lane['blocked_since'] >= self.lane_block_time
                lane['clear_since'] = None
            else:
                lane['blocked_since'], lane['clear_since'] = None, lane['clear_since'] or now
        edges, edge_now = lane['edges'], lane['edges'][-1][1]
        older = [e for t, e in edges if now - t >= 0.3]
        away = (True if math.isinf(edge_now) else
                edge_now - older[-1] > 0.02 if older and math.isfinite(older[-1]) else False)
        gap = edge_now - left_edge  # turned: its edge came closer, then turned back (just now) -> a whole swing to go
        recent = [(t, e) for t, e in edges if now - t < 1.2 and math.isfinite(e)]
        t_min, e_min = min(recent, key=lambda s: s[1]) if recent else (now, math.inf)
        turned = any(t < t_min and e > e_min + 0.05 for t, e in recent) and edge_now > e_min + 0.03
        clear_for = now - lane['clear_since'] if lane['clear_since'] else 0.0
        just_left = lane['was_blocked'] and clear_for >= self.lane_clear_time and away
        going_away = not lane['was_blocked'] and clear_for >= self.lane_clear_time and away and turned and self.lane_clear_gap <= gap < math.inf
        never_came = not lane['was_blocked'] and math.isinf(edge_now) and clear_for > 12.0
        self.log(f"Watching lane {self.lanes_crossed + 1}: in my path={lane['clear_since'] is None}, seen in path="
                 f"{lane['was_blocked']}, edge {gap:.2f} m left of my path, moving away={away}, nose {face - self.body_front:.2f} m from the lane")
        if just_left or going_away or never_came:
            self.cross_to, self.travel = far_face + self.body_rear + self.cross_clear, 0.0  # rear cross_clear past the far face, from here
            why = ("obstacle left our path and is moving away" if just_left else
                   f"obstacle turned back, {gap:.2f} m clear" if going_away else "nothing seen for 12 s")
            self.set_dyn('cross', now, f"Crossing lane {self.lanes_crossed + 1}: {why}")

    def lane_done(self, now, note=""):
        '''Book a crossed lane; finish after lanes_required.'''
        self.move_2D()
        self.behind_x, self.lane, self.lanes_crossed = self.lane['x'] + self.lane['thick'], None, self.lanes_crossed + 1
        if self.lanes_crossed >= self.lanes_required:
            self.set_dyn('finished', now, f"Finished: {self.lanes_crossed} lanes crossed, rear ~{self.cross_clear:.2f} m past the last{note}")
        else:
            self.set_dyn('align', now, f"Lane crossed ({self.lanes_crossed}/{self.lanes_required}){note}")

    def step_dynamic(self, p, now):
        '''Corridor 3: align -> look -> approach (nose lane_nose_gap from the lane) -> shift (slide to the right
        wall) -> align -> observe -> cross, per lane; look -> advance when nothing moves; finished after the last.'''
        if self.dyn_phase is None:
            self.set_dyn('align', now, "Moving obstacles: stop, observe, cross")
        phase, hold = self.dyn_phase, self.hold()
        if phase == 'finished':
            self.move_2D()
        elif phase == 'align':  # square to the corridor: looking needs the robot still and straight
            angle, length = wall_angle(self.points)
            if length > 0.5 and abs(angle) > 3.0 and now - self.dyn_since < 4.0:
                self.move_2D(0, 0, self.hold(self.min_turn_rate))
            elif self.lane is not None:
                self.lane['was_blocked'], self.lane['clear_since'] = False, None
                self.set_dyn('observe', now)
            else:
                self.look_dyn, self.look_scan = [], None
                self.set_dyn('look', now)
        elif phase == 'look':  # stand still; collect motion from every scan pair during the look
            self.move_2D()
            if self.scan_time != self.look_scan:
                self.look_scan = self.scan_time
                self.look_dyn.append(self.dynamic_points(since=self.dyn_since))
            if now - self.dyn_since < self.look_time:
                return
            ids = np.concatenate([np.full(len(d), i) for i, d in enumerate(self.look_dyn)]).astype(int)
            lane, nose = self.lane_ahead(np.vstack(self.look_dyn), ids), self.free_depth(p)
            if lane is not None:
                self.lane = lane
                self.set_dyn('approach', now, f"Moving obstacle {lane['x']:.2f} m ahead ({lane['thick'] * 2:.2f} m thick)")
            elif nose < self.end_stop_gap:
                self.set_dyn('align', now, f"Wall {nose:.2f} m ahead but only {self.lanes_crossed}/{self.lanes_required} lanes crossed: looking again")
            else:  # never step past end_stop_gap from whatever is ahead
                self.advance_to, self.travel = min(self.advance_step, nose - self.end_stop_gap + 0.01), 0.0
                self.set_dyn('advance', now, "Nothing moving ahead: advancing")
        elif phase == 'advance':  # straight ahead, no turning (turning swings the corners out)
            remaining = self.advance_to - self.travel
            if remaining < 0.02 or now - self.dyn_since > 6.0 or self.strip_blocked(p, self.body_front + max(remaining, 0.05) + 0.05):
                self.move_2D()
                self.set_dyn('align', now)
            else:
                self.move_2D(min(self.navigation_speed, 1.5 * remaining + 0.05), 0, hold)
        elif phase == 'approach':  # creep along until the nose is lane_nose_gap from the obstacle's path
            face = self.lane['x'] - self.lane['thick']
            behind = -9.0 if self.behind_x is None else self.behind_x + self.body_rear + 0.02  # never back into what is behind
            target = max(face - self.lane_nose_gap - self.body_front, behind)
            if abs(target) > 0.015 and now - self.dyn_since < 8.0:
                self.move_2D(self.creep_speed(target), 0, hold)
            elif self.at_wall and self.side_gap(p, -1) < self.wall_hug_gap + 0.06:
                self.set_dyn('align', now, f"Waiting {face - self.body_front:.2f} m from the lane, already at the right wall")
            else:
                self.set_dyn('shift', now, "Sliding sideways to the right wall")
        elif phase == 'shift':  # hug the right wall: obstacles turn round there, so after one leaves it has its whole trip to go
            clearance = self.side_gap(p, -1)
            if clearance > self.wall_hug_gap + 0.02 and now - self.dyn_since < 15.0:
                self.move_2D(0, -min(self.strafe_speed, 1.5 * (clearance - self.wall_hug_gap) + 0.03), hold)
            elif clearance < 0.25:
                self.move_2D()
                self.at_wall = True
                self.set_dyn('align', now, f"Hugging the right wall ({clearance:.2f} m gap)")
            else:
                self.move_2D()
                self.lane = None
                self.set_dyn('align', now, f"Could not reach the right wall ({clearance:.2f} m); looking again")
        elif phase == 'observe':
            self.move_2D()
            self.observe(p, now)
        elif phase == 'cross':
            remaining, far_face, nose = self.cross_to - self.travel, self.lane['x'] + self.lane['thick'], self.free_depth(p)
            if remaining < 0.03:
                self.lane_done(now)
            elif self.body_front > far_face + 0.02 and nose < max(self.lane_nose_gap, 0.05):
                self.lane_done(now, f", stopped early: something {nose:.2f} m ahead of the nose")
            elif self.body_front < self.lane['x'] - self.lane['thick'] - 0.05 and self.strip_blocked(p, far_face + 0.05):
                self.move_2D()  # not in the lane yet and something is in the way: back off
                self.lane['was_blocked'], self.lane['clear_since'] = True, None
                self.set_dyn('observe', now, "Crossing aborted before the lane")
            else:  # once in the lane keep going: stopping there is the worst option
                self.move_2D(min(self.cross_speed, 3.0 * remaining + 0.03), 0, hold)

    def timer_callback(self):
        """Controller loop (20 Hz): LiDAR-only finite state machine"""
        now = time.monotonic()
        if self.scan is None or now - self.scan_time > 0.5 or len(self.points) < 30:  # blind (tipped?) is not clear
            self.move_2D()
            return self.log("Stopped: no fresh LiDAR scan, or it sees almost nothing", 2.0)
        if self.calibrate:
            self.move_2D()
            angle, length = wall_angle(self.points)
            return self.log(f"pose={self.pose} walls {angle:+.1f} deg off ({length:.1f} m seen), left gap {self.side_gap(self.points, 1):.2f}, "
                            f"right gap {self.side_gap(self.points, -1):.2f}, nose gap {self.free_depth(self.points):.2f} m", 0.5) # Look, don't drive

        ###### INSERT CODE HERE ######
        p = self.points
        if self.state == 'CORRIDOR_1':
            if not self.corridor_cue(p, now):
                self.drive_corridor(p)
        elif self.state == 'TURN':
            self.step_turn(p, now)
        elif self.state == 'CORRIDOR_2':
            if self.update_gate(p):
                self.set_state('GATE_APPROACH')
            else:
                self.drive_corridor(p)
        elif self.state == 'GATE_APPROACH':
            self.step_gate_approach(p, now)
        elif self.state == 'GATE_CROSS':
            self.step_gate_cross(p, now)
        elif self.state == 'DYNAMIC':
            self.step_dynamic(p, now)
        ###### INSERT CODE HERE ######

def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project (LiDAR only)")
    arg_parser.add_argument('--calibrate', action='store_true', help="Only print the pose and what the LiDAR sees twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course (LiDAR only)", flush=True)
    rclpy.init(args=ros_args)
    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar, timeout_sec=0.5)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()
    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except BaseException as error:  # Ctrl-C or a timeout shuts ROS down first: that is a normal exit, re-raise anything else
        if rclpy.ok() and not isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if rclpy.ok():
            obstacle.move_2D() # Always publish a stop before releasing the publisher
        obstacle.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
