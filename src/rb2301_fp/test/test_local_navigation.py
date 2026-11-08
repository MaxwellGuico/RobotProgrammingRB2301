"""Regression checks for local targets, scan frames, and unknown measurements."""
import math
from collections import deque
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch
import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode, path_visit_cost, visit_cell


class LocalNavigationTests(unittest.TestCase):
    def analyze(self, ranges, offset=0.11423):
        node = SimpleNamespace(lidar_min_distance=0.0, lidar_max_distance=5.0,
                               robot_safe_radius=0.14, clearance_margin=.03, commit_margin=.01, nose_overhang=.04,
                               lidar_x_offset=offset,
                               lidar_yaw=math.pi, pose=None)
        scan = SimpleNamespace(ranges=ranges, angle_min=-math.pi,
                               angle_increment=2 * math.pi / len(ranges),
                               range_min=0.05, range_max=10.0)
        ObstacleCourseNode.sub_scan_callback(node, scan)
        return node

    def test_open_space_needs_no_cluster_edges(self):
        node = self.analyze([float('inf')] * 360)
        self.assertTrue(any(g['target'][0] > .9 for g in node.valid_gaps))

    def test_unknown_scan_does_not_allow_motion(self):
        self.assertEqual(self.analyze([float('nan')] * 360).valid_gaps, [])

    def test_enclosed_robot_has_no_route(self):
        self.assertEqual(self.analyze([.1] * 360, offset=0).valid_gaps, [])

    def test_mount_translation_and_rotation(self):
        ranges = [float('inf')] * 360
        ranges[180] = 1.0  # Scanner +x faces robot -x.
        np.testing.assert_allclose(self.analyze(ranges).points[0],
                                   [-.88577, 0], atol=1e-6)

    def test_obstacle_blocks_forward_but_allows_detour(self):
        ranges = [float('inf')] * 360
        ranges[:12] = [.3] * 12
        ranges[-12:] = [.3] * 12
        node = self.analyze(ranges)
        self.assertFalse(any(abs(g['heading']) < math.radians(10)
                             for g in node.valid_gaps))
        self.assertTrue(any(abs(g['heading']) > math.radians(45)
                            for g in node.valid_gaps))
        # Every accepted straight segment clears the inflated robot footprint.
        for gap in node.valid_gaps:
            target = gap['target']
            t = np.clip((node.points @ target) / (target @ target), 0, 1)
            clearance = np.linalg.norm(node.points - t[:, None] * target, axis=1)
            self.assertGreater(float(np.min(clearance)), .17)

    def controller(self, points):
        commands = []
        node = SimpleNamespace(
            pose=np.array([0., 0., 0.]), scan_pose=np.zeros(3),
            calibrate=False, last_scan_time=11.,
            visited_positions=[np.zeros(2)], history_spacing=.2,
            visited_cells={}, last_visit_cell=None,
            progress_direction=np.array([1., -1.]) / np.sqrt(2), progress_bias_weight=2.,
            rejected_targets=[], committed_target=np.array([1., 0.]),
            points=np.asarray(points), robot_safe_radius=.14, clearance_margin=.03, commit_margin=.01,
            nose_overhang=.04, reverse_penalty_weight=1.5,
            progress_distance=1., progress_time=11., blocked_since=None,
            navigation_speed=.35, Kp=3., Kd=0., Kp_theta=2., Kd_theta=.15,
            PID=lambda *args: (.1, 0., 0.),
            move_2D=lambda *args: commands.append(args),
            get_logger=lambda: SimpleNamespace(info=lambda msg: None, warn=lambda msg: None),
            get_clock=lambda: SimpleNamespace(now=lambda: 0))
        node.corridors = [dict(name='C1', direction=np.array([1., 0.]), turn='right'),
                          dict(name='C3', direction=np.array([1., 0.]), turn=None)]
        node.state_index, node.gate, node.gate_phase = 0, None, None
        for name in ('reset_target', 'step_corridor_state', 'step_gate_detection',
                     'step_gate_approach', 'step_drive', 'gate_waypoint', 'path_is_clear',
                     'in_gate_corridor', 'step_dynamic_obstacles'):
            setattr(node, name, MethodType(getattr(ObstacleCourseNode, name), node))
        node.update_corridor_state = lambda *args: False
        node.update_wall_direction = lambda *args: None  # tests set directions themselves
        return node, commands

    def test_blocked_target_is_released_after_wait(self):
        node, commands = self.controller([[.3, 0.]])
        node.blocked_since = 10.
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertIsNone(node.committed_target)
        self.assertEqual(len(node.rejected_targets), 1)
        self.assertEqual(commands[-1], ())  # default command is zero

    def test_can_move_away_from_obstacle_behind_start(self):
        node, commands = self.controller([[-.18, 0.]])
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertGreater(commands[-1][0], 0.)

    def test_stale_scan_stops_robot(self):
        node, commands = self.controller([])
        node.last_scan_time = 9.
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertEqual(commands[-1], ())

    def test_repeated_path_cost_grows_with_revisits(self):
        start, target = np.array([0., 0.]), np.array([1., 0.])
        cells = {visit_cell([x, 0.]) for x in np.linspace(.2, 1., 20)}
        once = {c: 1 for c in cells}
        repeated = {c: 3 for c in cells}
        self.assertGreater(path_visit_cost(start, target, repeated),
                           path_visit_cost(start, target, once))

    def test_path_penalty_sees_visited_corridor_before_fresh_endpoint(self):
        visits = {visit_cell([.4, 0.]): 3, visit_cell([.6, 0.]): 3}
        self.assertGreater(path_visit_cost(np.zeros(2), np.array([1., 0.]), visits), 0.)
        self.assertEqual(visits.get(visit_cell([1., 0.]), 0), 0)

    def test_fresh_rear_exit_can_beat_repeated_forward_loop(self):
        node, _ = self.controller(np.empty((0, 2)))
        node.committed_target = None
        node.valid_gaps = [dict(target=np.array([.8, 0.]), heading=0., free_depth=.8),
                           dict(target=np.array([-.8, 0.]), heading=math.pi, free_depth=.8)]
        node.visited_cells = {visit_cell([x, 0.]): 3 for x in [.25, .5, .75]}
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertLess(node.committed_target[0], 0.)

    def test_large_turn_aligns_before_translation(self):
        node, commands = self.controller(np.empty((0, 2)))
        node.committed_target = np.array([.1, 1.])
        node.PID = lambda *args: (.3, 0., .8)
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertEqual(commands[-1][0], 0.)
        self.assertGreater(commands[-1][2], 0.)

    def test_old_scan_is_rotated_into_current_robot_frame(self):
        node, commands = self.controller([[.3, 0.]])
        node.pose[2] = 90.  # Robot turned since the obstacle was measured.
        node.committed_target = np.array([0., 1.])
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertGreater(commands[-1][0], 0.)

    def test_world_bias_stays_toward_exit_when_robot_faces_backwards(self):
        node, _ = self.controller(np.empty((0, 2)))
        node.pose[2] = node.scan_pose[2] = 180.
        node.committed_target = None
        node.valid_gaps = [dict(target=np.array([.8, 0.]), heading=0., free_depth=.8),
                           dict(target=np.array([-.8, 0.]), heading=math.pi, free_depth=.8)]
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertGreater(node.committed_target[0], 0.)

    def test_world_bias_prefers_negative_y_over_positive_y(self):
        node, _ = self.controller(np.empty((0, 2)))
        node.committed_target = None
        node.valid_gaps = [dict(target=np.array([0., .8]), heading=math.pi/2, free_depth=.8),
                           dict(target=np.array([0., -.8]), heading=-math.pi/2, free_depth=.8)]
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertLess(node.committed_target[1], 0.)

    def test_does_not_turn_back_against_corridor(self):
        # Corridor runs -y. The forward option is a short side-step; the long
        # free path behind should still lose.
        node, _ = self.controller(np.empty((0, 2)))
        node.committed_target = None
        node.progress_bias_weight = 0.
        node.progress_direction = np.array([0., -1.])
        node.valid_gaps = [dict(target=np.array([.4, -.3]), heading=-.6, free_depth=.5),
                           dict(target=np.array([0., 1.]), heading=math.pi/2, free_depth=1.)]
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertLess(node.committed_target[1], 0.)

    def corridor_node(self, index=0, pose=(0., 0., 0.)):
        node = SimpleNamespace(pose=np.array(pose), side_depth=1.1, opening_min_width=.4,
            min_corridor_travel=.8, gate_phase=None,
            state_start=np.zeros(2), side_wall_seen=False, transition_since=None,
            get_logger=lambda: SimpleNamespace(info=lambda msg: None))
        node.corridors = [dict(name='C1', direction=np.array([1., 0.]), turn='right',
                               side_depth=1.1),
                          dict(name='C2', direction=np.array([0., -1.]), turn='left',
                               side_depth=1.8),
                          dict(name='C3', direction=np.array([1., 0.]), turn=None)]
        node.enter_corridor = lambda i: ObstacleCourseNode.enter_corridor(node, i)
        node.corridor_axes = lambda: ObstacleCourseNode.corridor_axes(node)
        node.in_gate_corridor = lambda: ObstacleCourseNode.in_gate_corridor(node)
        node.corridor_observation = lambda pts: ObstacleCourseNode.corridor_observation(node, pts)
        node.enter_corridor(index)
        node.state_start = np.zeros(2)
        return node

    @staticmethod
    def wall(x0, x1, y):
        return np.column_stack((np.arange(x0, x1, .02), np.full(len(np.arange(x0, x1, .02)), y)))

    def test_corridor_1_straight_while_right_wall_present(self):
        node = self.corridor_node(pose=(1., 0., 0.))
        points = np.vstack((self.wall(-1, 3, .6), self.wall(-1, 3, -.6)))
        beside, opening, end = ObstacleCourseNode.corridor_observation(node, points)
        self.assertTrue(beside)
        self.assertFalse(opening)
        self.assertFalse(end)

    def test_corridor_1_turns_right_when_right_wall_ends(self):
        node = self.corridor_node(pose=(1., 0., 0.))
        points = np.vstack((self.wall(-1, 3, .6), self.wall(-1, 1.05, -.6)))
        for t in (10., 10.2, 10.4):
            ObstacleCourseNode.update_corridor_state(node, points, t)
        self.assertEqual(node.state_index, 1)
        np.testing.assert_allclose(node.progress_direction, [0., -1.])

    def test_box_beside_wall_does_not_trigger_turn_or_hide_gap_forever(self):
        # 30 cm box against the right wall: wall is continuous, so no turn.
        node = self.corridor_node(pose=(1., 0., 0.))
        box = np.column_stack((np.full(15, 1.1), np.linspace(-.6, -.3, 15)))
        points = np.vstack((self.wall(-1, 3, .6), self.wall(-1, 3, -.6), box))
        _, opening, _ = ObstacleCourseNode.corridor_observation(node, points)
        self.assertFalse(opening)

    def test_end_wall_is_fallback_turn_cue(self):
        node = self.corridor_node(pose=(1.9, 0., 0.))
        # Right side hidden the whole way (box), but corridor ends at x=2.3.
        points = np.vstack((self.wall(-1, 2.3, .6), self.wall(-1, 2.3, -.6),
                            np.column_stack((np.full(60, 2.3), np.linspace(-.6, .6, 60)))))
        _, _, end = ObstacleCourseNode.corridor_observation(node, points)
        self.assertTrue(end)

    def test_small_box_ahead_is_not_end_wall(self):
        node = self.corridor_node(pose=(1., 0., 0.))
        box = np.column_stack((np.full(20, 1.4), np.linspace(-.3, .2, 20)))
        points = np.vstack((self.wall(-1, 3, .6), self.wall(-1, 3, -.6), box))
        _, _, end = ObstacleCourseNode.corridor_observation(node, points)
        self.assertFalse(end)

    def test_corridor_2_turns_left(self):
        node = self.corridor_node(index=1, pose=(1.7, -2.6, -90.))
        node.state_start = np.array([1.7, -.3])
        node.side_wall_seen = True
        # Right wall (x=1.1) continues, left wall (x=2.3) ended at y=-2.4.
        left = np.column_stack((np.full(60, 2.3), np.linspace(-2.4, 0, 60)))
        right = np.column_stack((np.full(150, 1.1), np.linspace(-5, 0, 150)))
        for t in (0., .2, .4):
            ObstacleCourseNode.update_corridor_state(node, np.vstack((left, right)), t)
        self.assertEqual(node.state_index, 2)

    def test_40cm_gap_is_passable(self):
        # Two wall posts 40 cm apart, 0.6 m ahead (robot frame +x).
        offset = 0.11423
        ranges = [float('inf')] * 360
        for wall_y in (-.2, .2):
            for extra in np.linspace(0, .6, 30):
                y = wall_y + math.copysign(extra, wall_y)
                x = .6 - offset
                angle = math.atan2(y, x) - math.pi  # scanner faces backwards
                i = int(round((math.atan2(math.sin(angle), math.cos(angle)) + math.pi)
                              / (2 * math.pi / 360))) % 360
                ranges[i] = min(ranges[i], math.hypot(x, y))
        node = self.analyze(ranges)
        self.assertTrue(any(abs(g['heading']) < math.radians(3) and g['target'][0] > .6
                            for g in node.valid_gaps))

    def test_crossing_wall_after_turn_is_not_a_side_wall(self):
        # Just turned into corridor 2 (-y). Corridor 1's far wall (y=0.6) runs
        # across our path; the left side (+x) is the open end of corridor 1.
        node = self.corridor_node(index=1, pose=(1.1, 0., -90.))
        node.state_start = np.array([1., .4])
        points = np.vstack((self.wall(-1, 2.3, .6),
                            np.column_stack((np.full(150, 1.1 - .5), np.linspace(-5, -.6, 150)))))
        beside, _, _ = ObstacleCourseNode.corridor_observation(node, points)
        self.assertFalse(beside)
        for t in (0., .2, .4):
            ObstacleCourseNode.update_corridor_state(node, points, t)
        self.assertEqual(node.state_index, 1)

    def test_corridor_2_widening_is_not_the_gate(self):
        # Left wall steps out from x=2.3 to x=2.9 at y=-1.8; gate is further down.
        node = self.corridor_node(index=1, pose=(1.3, -1.9, -90.))
        node.state_start = np.array([1.3, -.3])
        node.side_wall_seen = True
        points = np.vstack((
            np.column_stack((np.full(60, 2.3), np.linspace(-1.8, 0, 60))),
            np.column_stack((np.full(40, 2.9), np.linspace(-2.9, -1.8, 40))),
            np.column_stack((np.full(150, 1.1), np.linspace(-5, 0, 150)))))
        for t in (0., .2, .4):
            ObstacleCourseNode.update_corridor_state(node, points, t)
        self.assertEqual(node.state_index, 1)

    def test_committed_path_survives_small_drift_in_tight_gap(self):
        # Obstacle 16 cm from the path: below the 17 cm needed to choose it,
        # but a committed path should not be dropped for it.
        node, commands = self.controller([[.5, .16]])
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertIsNone(node.blocked_since)
        self.assertGreater(commands[-1][0], 0.)

    def test_committed_path_still_stops_for_real_obstruction(self):
        node, commands = self.controller([[.5, .1]])
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertIsNotNone(node.blocked_since)
        self.assertEqual(commands[-1], ())

    def test_tight_slot_falls_back_to_commit_margin(self):
        # Walls 16 cm either side: blocked at the 17 cm margin, open at 15 cm.
        offset = 0.11423
        ranges = [float('inf')] * 360
        for wall_y in (-.16, .16):
            for x in np.linspace(-.5, 1.5, 200):
                sx = x - offset
                angle = math.atan2(wall_y, sx) - math.pi
                i = int(round((math.atan2(math.sin(angle), math.cos(angle)) + math.pi)
                              / (2 * math.pi / 360))) % 360
                ranges[i] = min(ranges[i], math.hypot(sx, wall_y))
        node = self.analyze(ranges)
        self.assertTrue(any(abs(g['heading']) < math.radians(3) for g in node.valid_gaps))

    def gate_node(self, pose=(1.8, -2.0, -90.), long_rays=True):
        """Corridor 2 node with a scan history whose rays are all long (or short)."""
        node = self.corridor_node(index=1, pose=pose)
        node.gate_min_width, node.gate_max_width = .38, .8
        node.gate_candidates_history = deque(maxlen=5)
        node.gate_last_detect_scan, node.gate = None, None
        node.ray_range_recent = MethodType(ObstacleCourseNode.ray_range_recent, node)
        node.detect_gate_candidates = MethodType(ObstacleCourseNode.detect_gate_candidates, node)
        node.scan_history = deque([dict(time=0., xy=np.array(pose[:2]), yaw=0., angle_min=-math.pi,
            increment=2 * math.pi / 360, full=True,
            ranges=np.full(360, 5.0 if long_rays else 0.5))])
        return node

    @staticmethod
    def gate_world():
        # Corridor 2: right wall x=1.1; left wall x=2.3 down to y=-1.8, steps out
        # to x=2.9; gate in the x=2.9 wall between y=-2.8 and y=-3.3.
        def col(x, y0, y1):
            ys = np.arange(y0, y1, .02)
            return np.column_stack((np.full(len(ys), x), ys))
        return np.vstack((col(1.1, -4.0, 0.), col(2.3, -1.8, 0.),
                          np.column_stack((np.linspace(2.3, 2.88, 30), np.full(30, -1.8))),
                          col(2.9, -2.8, -1.8), col(2.9, -4.0, -3.3)))

    def test_gate_found_in_wall_line(self):
        node = self.gate_node()
        gates = ObstacleCourseNode.detect_gate_candidates(node, self.gate_world())
        self.assertEqual(len(gates), 1)
        np.testing.assert_allclose(gates[0]['centre'], [2.9, -3.05], atol=.03)
        np.testing.assert_allclose(gates[0]['normal'], [1., 0.])

    def test_corridor_widening_is_not_a_gate(self):
        # Same world without the gate: the step at y=-1.8 must not count.
        node = self.gate_node()
        points = self.gate_world()
        points = points[~((points[:, 0] > 2.85) & (points[:, 1] < -1.8))]
        points = np.vstack((points, np.column_stack((np.full(110, 2.9),
                                                     np.linspace(-4.0, -1.82, 110)))))
        self.assertEqual(ObstacleCourseNode.detect_gate_candidates(node, points), [])

    def test_shadow_gap_is_not_a_gate(self):
        # Rays toward the gap stop short (an object in front): reject.
        node = self.gate_node(long_rays=False)
        self.assertEqual(ObstacleCourseNode.detect_gate_candidates(node, self.gate_world()), [])

    def test_gate_needs_three_of_five_scans(self):
        node = self.gate_node()
        world = self.gate_world()
        for stamp in (1., 2.):
            node.last_scan_time = stamp
            self.assertFalse(ObstacleCourseNode.update_gate_detection(node, world))
        node.last_scan_time = 3.
        self.assertTrue(ObstacleCourseNode.update_gate_detection(node, world))
        np.testing.assert_allclose(node.gate['centre'], [2.9, -3.05], atol=.03)

    def test_same_scan_is_not_counted_twice(self):
        node = self.gate_node()
        node.last_scan_time = 1.
        for _ in range(5):
            ObstacleCourseNode.update_gate_detection(node, self.gate_world())
        self.assertIsNone(node.gate)

    def test_recent_long_ray_beats_passing_obstacle(self):
        # Newest scan is blocked by a moving obstacle; one 0.5 s earlier saw through.
        node = self.gate_node(long_rays=False)
        node.scan_history.appendleft(dict(time=-.5, xy=np.array([1.8, -2.0]), yaw=0.,
            angle_min=-math.pi, increment=2 * math.pi / 360, full=True,
            ranges=np.full(360, 5.0)))
        best, distance = ObstacleCourseNode.ray_range_recent(node, np.array([2.9, -3.05]))
        self.assertGreater(best, distance + .3)

    def test_box_and_bin_inside_corridor_are_not_a_gate(self):
        # Box face at x=1.7 (y -0.45..-0.95) and bin at x=1.7 (y -1.6..-2.0)
        # leave a see-through 0.65 m "gap", but they are inside the corridor.
        node = self.gate_node(pose=(1.4, -0.3, -90.))
        ys1, ys2 = np.linspace(-.95, -.45, 25), np.linspace(-2.0, -1.6, 20)
        points = np.vstack((self.gate_world(),
                            np.column_stack((np.full(25, 1.7), ys1)),
                            np.column_stack((np.full(20, 1.7), ys2))))
        gates = ObstacleCourseNode.detect_gate_candidates(node, points)
        self.assertFalse(any(abs(g['centre'][0] - 1.7) < .1 for g in gates))

    def test_wall_corner_plus_box_is_not_end_wall(self):
        # Just turned into corridor 2 (-y), hugging the inner corner: corridor 1's
        # wall end (y=-0.6) and a box face (y=-0.9) together fill the slices.
        node = self.corridor_node(index=1, pose=(1.49, -.41, -90.))
        points = np.vstack((
            np.column_stack((np.linspace(.9, 1.42, 27), np.full(27, -.6))),
            np.column_stack((np.linspace(1.5, 2.0, 26), np.full(26, -.9)))))
        _, _, end = ObstacleCourseNode.corridor_observation(node, points)
        self.assertFalse(end)

    def optitrack_node(self, **kw):
        logs = []
        node = SimpleNamespace(calibrate=False, zero_at_start=True, zero_start_heading=True,
            start_pose=None, frame=None, get_logger=lambda: SimpleNamespace(info=logs.append))
        node.__dict__.update(kw)
        node.yaw_from_quaternion = lambda q: q
        return node

    @staticmethod
    def pose_msg(x, y, yaw_deg):
        return SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y),
                                                    orientation=math.radians(yaw_deg)))

    def test_start_position_becomes_origin(self):
        node = self.optitrack_node()
        with patch('rb2301_fp.obstacle_course.optitrack_to_maze', lambda x, y, h, f: (x, y, h)):
            ObstacleCourseNode.optitrack_callback(node, self.pose_msg(2., 3., 90.))
            np.testing.assert_allclose(node.pose, [0., 0., 0.], atol=1e-9)
            # Drive 1 m in the direction it started facing (world +y): that is +x.
            ObstacleCourseNode.optitrack_callback(node, self.pose_msg(2., 4., 90.))
            np.testing.assert_allclose(node.pose, [1., 0., 0.], atol=1e-9)

    def test_start_heading_kept_when_disabled(self):
        node = self.optitrack_node(zero_start_heading=False)
        with patch('rb2301_fp.obstacle_course.optitrack_to_maze', lambda x, y, h, f: (x, y, h)):
            ObstacleCourseNode.optitrack_callback(node, self.pose_msg(2., 3., 90.))
            ObstacleCourseNode.optitrack_callback(node, self.pose_msg(2., 4., 90.))
        np.testing.assert_allclose(node.pose, [0., 1., 90.], atol=1e-9)

    def test_calibration_sees_start_as_origin(self):
        node = self.optitrack_node(calibrate=True)
        with patch('rb2301_fp.obstacle_course.optitrack_to_maze', lambda x, y, h, f: (x, y, h)):
            ObstacleCourseNode.optitrack_callback(node, self.pose_msg(2., 3., 90.))
        np.testing.assert_allclose(node.pose, [0., 0., 0.], atol=1e-9)
        np.testing.assert_allclose(node.maze_pose, [2., 3., 90.], atol=1e-9)
    def dyn_node(self):
        node = SimpleNamespace(cross_margin=.06, BODY_HALF_WIDTH=.11, BODY_FRONT=.18,
                               BODY_REAR=.034, pose=np.zeros(3))
        node.corridors = [dict(name='C3', direction=np.array([1., 0.]), turn=None)]
        node.state_index = 0
        node.corridor_axes = lambda: ObstacleCourseNode.corridor_axes(node)
        return node

    def test_dynamic_points_only_reports_what_moved(self):
        wall = np.column_stack((np.linspace(0, 2, 50), np.full(50, 1.)))
        node = SimpleNamespace(scan_history=deque([
            dict(time=0., points=np.vstack((wall, [[1., 0.2]]))),
            dict(time=.5, points=np.vstack((wall, [[1., -0.2]])))]))
        moved = ObstacleCourseNode.dynamic_points(node, since=0.)
        np.testing.assert_allclose(moved, [[1., -0.2]])

    def test_lane_ahead_finds_nearest_moving_obstacle(self):
        node = self.dyn_node()
        near = np.column_stack((np.full(8, .8), np.linspace(-.15, .15, 8)))
        far = np.column_stack((np.full(8, 1.4), np.linspace(.3, .6, 8)))
        lane = ObstacleCourseNode.lane_ahead(node, np.vstack((near, far)))
        self.assertAlmostEqual(lane['s'], .8)

    def test_stray_points_do_not_hide_lane(self):
        node = self.dyn_node()
        stray = np.array([[.3, 1.0], [.32, -1.2]])
        lane = np.column_stack((np.full(8, .8), np.linspace(-.15, .15, 8)))
        found = ObstacleCourseNode.lane_ahead(node, np.vstack((stray, lane)))
        self.assertAlmostEqual(found['s'], .8)

    def test_nose_keeps_margin_from_wall_ahead(self):
        # Wall straight ahead at x=1.0 (robot frame): the target must leave the
        # nose (0.18 m ahead of base_link) at least the margin away.
        offset = 0.11423
        ranges = [float('inf')] * 360
        for y in np.linspace(-1, 1, 400):
            sx = 1.0 - offset
            angle = math.atan2(y, sx) - math.pi
            i = int(round((math.atan2(math.sin(angle), math.cos(angle)) + math.pi)
                          / (2 * math.pi / 360))) % 360
            ranges[i] = min(ranges[i], math.hypot(sx, y))
        node = self.analyze(ranges)
        ahead = [g for g in node.valid_gaps if abs(g['heading']) < 1e-6]
        self.assertTrue(ahead)
        self.assertLessEqual(ahead[0]['target'][0] + .18, 1.0 - .03 + 1e-6)

    def test_strip_ignores_wall_beside_but_sees_obstacle_ahead(self):
        node = self.dyn_node()
        wall_beside = np.column_stack((np.linspace(-.5, 1.5, 50), np.full(50, -.2)))
        self.assertFalse(ObstacleCourseNode.strip_blocked(node, wall_beside, .8))
        board = np.column_stack((np.full(10, .5), np.linspace(-.1, .2, 10)))
        self.assertTrue(ObstacleCourseNode.strip_blocked(node, np.vstack((wall_beside, board)), .8))
        # Beyond the strip end (next lane) does not count.
        self.assertFalse(ObstacleCourseNode.strip_blocked(node, board + [.5, 0.], .8))

    def test_right_clearance_ignores_corner_behind(self):
        node = self.dyn_node()
        behind = np.array([[-.1, -.15]])           # gate wall end behind the rear
        beside = np.array([[.05, -.30], [.1, -.31]])  # wall alongside
        self.assertAlmostEqual(
            ObstacleCourseNode.right_clearance(node, np.vstack((behind, beside))), .30 - .11)

    def test_nose_gap_sees_wall_ahead_only(self):
        node = self.dyn_node()
        wall_ahead = np.column_stack((np.full(20, .5), np.linspace(-.3, .3, 20)))
        side = np.array([[.3, .5]])
        self.assertAlmostEqual(ObstacleCourseNode.nose_gap(node, np.vstack((wall_ahead, side))),
                               .5 - .18)

    def test_wall_direction_follows_skewed_walls(self):
        # Corridor nominally +x, but both walls are rotated by 7 degrees.
        skew = math.radians(7)
        t = np.linspace(-1.4, 1.4, 120)
        d, n = np.array([math.cos(skew), math.sin(skew)]), np.array([-math.sin(skew), math.cos(skew)])
        points = np.vstack((t[:, None] * d + .6 * n, t[:, None] * d - .6 * n))
        node = self.corridor_node()
        node.corridor_axes = lambda: ObstacleCourseNode.corridor_axes(node)
        for _ in range(30):  # smoothed: converges over several scans
            ObstacleCourseNode.update_wall_direction(node, points)
        self.assertAlmostEqual(node.wall_offset, skew, places=3)
        np.testing.assert_allclose(ObstacleCourseNode.corridor_axes(node)[1], d, atol=1e-3)

    def test_wall_direction_ignores_large_angles(self):
        # 30 degrees off is not "slightly skewed": keep the square assumption.
        skew = math.radians(30)
        t = np.linspace(-1.4, 1.4, 120)
        d = np.array([math.cos(skew), math.sin(skew)])
        node = self.corridor_node()
        node.corridor_axes = lambda: ObstacleCourseNode.corridor_axes(node)
        ObstacleCourseNode.update_wall_direction(node, t[:, None] * d + [0., .6])
        self.assertEqual(getattr(node, 'wall_offset', 0.0), 0.0)

    def test_gate_found_in_skewed_wall(self):
        # Rotate the whole corridor-2 test world by 6 degrees about the robot.
        skew = math.radians(6)
        node = self.gate_node()
        node.wall_offset = skew
        R = np.array([[math.cos(skew), -math.sin(skew)], [math.sin(skew), math.cos(skew)]])
        pivot = node.pose[:2]
        world = (self.gate_world() - pivot) @ R.T + pivot
        gates = ObstacleCourseNode.detect_gate_candidates(node, world)
        self.assertEqual(len(gates), 1)
        np.testing.assert_allclose(gates[0]['centre'], (np.array([2.9, -3.05]) - pivot) @ R.T + pivot,
                                   atol=.04)


if __name__ == '__main__':
    unittest.main()
