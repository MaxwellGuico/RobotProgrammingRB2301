"""Regression checks for local targets, scan frames, and unknown measurements."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode, path_visit_cost, visit_cell


class LocalNavigationTests(unittest.TestCase):
    def analyze(self, ranges, offset=0.11423):
        node = SimpleNamespace(lidar_min_distance=0.0, lidar_max_distance=5.0,
                               robot_safe_radius=0.14, clearance_margin=.03,
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
            points=np.asarray(points), robot_safe_radius=.14, clearance_margin=.03,
            reverse_penalty_weight=1.5,
            progress_distance=1., progress_time=11., blocked_since=None,
            navigation_speed=.35, Kp=3., Kd=0., Kp_theta=2., Kd_theta=.15,
            PID=lambda *args: (.1, 0., 0.),
            move_2D=lambda *args: commands.append(args),
            get_logger=lambda: SimpleNamespace(info=lambda msg: None),
            get_clock=lambda: SimpleNamespace(now=lambda: 0))
        node.reset_target = lambda: ObstacleCourseNode.reset_target(node)
        node.travel_direction = lambda: ObstacleCourseNode.travel_direction(node)
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

    def test_does_not_turn_back_toward_where_it_came_from(self):
        # Robot drove -y (toward +x/-y bias would also favour -y). The only
        # forward option is a short sideways gap; the long free path is behind.
        node, _ = self.controller(np.empty((0, 2)))
        node.committed_target = None
        node.progress_bias_weight = 0.
        node.visited_positions = [np.array([0., .4]), np.array([0., .2]), np.zeros(2)]
        node.valid_gaps = [dict(target=np.array([.4, -.3]), heading=-.6, free_depth=.5),
                           dict(target=np.array([0., 1.]), heading=math.pi/2, free_depth=1.)]
        with patch('rb2301_fp.obstacle_course.time.monotonic', return_value=11.):
            ObstacleCourseNode.timer_callback(node)
        self.assertLess(node.committed_target[1], 0.)

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


if __name__ == '__main__':
    unittest.main()
