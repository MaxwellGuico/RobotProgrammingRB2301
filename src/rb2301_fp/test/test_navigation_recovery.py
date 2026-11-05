"""Stalls stop the robot; navigation safety must never allow reversing."""
import math
from types import SimpleNamespace, MethodType
import unittest
import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class NavigationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.time = SimpleNamespace(nanoseconds=0)
        self.node = SimpleNamespace(
            pose=np.array([0., 0., 0.]),
            get_clock=lambda: SimpleNamespace(now=lambda: self.time))
        self.node.safe_command = MethodType(ObstacleCourseNode.safe_command, self.node)

    def command(self, forward=.1, turn=0.):
        return ObstacleCourseNode.stop_if_stalled(self.node, forward, turn)

    def test_stall_stops_and_remains_stopped_until_pose_progress(self):
        self.assertEqual(self.command(), (.1, 0.))
        for seconds in (3, 4, 8):
            self.time.nanoseconds = seconds*1_000_000_000
            self.assertEqual(self.command(), (0., 0.))
        self.node.pose[0] = .02
        self.assertEqual(self.command(), (.1, 0.))

    def test_position_progress_prevents_a_stall_stop(self):
        self.command()
        self.time.nanoseconds = 3_000_000_000
        self.node.pose[0] = .1
        self.assertEqual(self.command(), (.1, 0.))

    def test_old_recovery_settings_cannot_reenable_reverse(self):
        self.node.enable_reverse_recovery = True
        self.node.recovery_until, self.node.turn_at = 20., 10.
        self.command()
        self.time.nanoseconds = 3_000_000_000
        self.assertEqual(self.command(), (0., 0.))

    def test_blocked_interval_does_not_count_as_failed_motion(self):
        self.command()
        self.time.nanoseconds = 3_000_000_000
        self.assertEqual(self.command(0., 0.), (0., 0.))
        self.time.nanoseconds = 10_000_000_000
        self.assertEqual(self.command(), (.1, 0.))

    def test_negative_motion_is_rejected_by_progress_guard(self):
        self.assertEqual(self.command(-.08, .6), (0., 0.))

    def test_reverse_is_rejected_even_with_empty_space(self):
        for turn in (0., .6, -.6):
            self.assertEqual(self.node.safe_command(np.empty((0, 2)), -.08, turn), (0., 0.))

    def test_reverse_escape_is_removed_and_forward_collision_check_remains(self):
        points = np.array([[.19, 0.]])
        self.assertEqual(self.node.safe_command(points, -.08, 0.), (0., 0.))
        self.assertEqual(self.node.safe_command(points, .08, 0.), (0., 0.))

    def test_rotation_still_checks_the_swept_footprint(self):
        points = np.array([[.19, .14]])
        self.assertEqual(self.node.safe_command(points, 0., .6), (0., 0.))
        self.assertEqual(self.node.safe_command(points, 0., -.6), (0., -.6))

    def test_narrow_bottle_wall_gap_is_rejected(self):
        # Ray-cast a bottle near the lower corridor wall. Its lower opening is
        # about 29 cm in the sampled scan; the safe route is straight/above it.
        points, indices = [], []
        for i, angle in enumerate(np.linspace(-math.pi / 2, math.pi / 2, 181)):
            dx, dy = math.cos(angle), math.sin(angle)
            distance = .6 / abs(dy) if abs(dy) > 1e-6 else 10.
            cx, cy = .59 - .11423, -.29
            along = cx * dx + cy * dy
            discriminant = .04 ** 2 - (cx * cx + cy * cy - along * along)
            if discriminant >= 0 and along > 0:
                distance = min(distance, along - math.sqrt(discriminant))
            if distance < 1.5:
                points.append([distance * dx + .11423, distance * dy])
                indices.append(i)
        node = self.node
        target = ObstacleCourseNode.gap_target(node, np.asarray(points))
        self.assertIsNotNone(target)
        self.assertGreater(np.arctan2(target[1], target[0]), np.deg2rad(-11))

    def test_occluded_end_wall_is_not_selected_as_first_panel(self):
        wall = {'x': 5.27, 'samples': [(i, -2. + i * .03) for i in range(11)]}
        node = SimpleNamespace(panel_number=1, tracks=[wall])
        self.assertEqual(ObstacleCourseNode.panel_candidates(node, np.array([3., -2.8, 0.])), [])

    def test_second_panel_must_match_panel_spacing(self):
        def track(x):
            return {'x': x, 'samples': [(i, -2. + i * .03) for i in range(11)]}
        panel, wall = track(4.65), track(5.27)
        node = SimpleNamespace(panel_number=2, first_panel_x=4.05, tracks=[panel, wall])
        self.assertEqual(ObstacleCourseNode.panel_candidates(node, np.array([4.25, -2.5, 0.])), [panel])

    def test_staging_uses_the_lane_actually_reached(self):
        node = SimpleNamespace(
            state='position', lane=-2.44, wait_x=3.74,
            to_course=lambda: np.array([3.69, -2.53, 0.]),
            observe_panels=lambda: (10., np.empty((0, 2))))
        node.enter_state = lambda state, now: setattr(node, 'state', state)
        self.assertEqual(ObstacleCourseNode.panel_command(node), (0., 0.))
        self.assertEqual(node.state, 'observe')
        self.assertEqual(node.lane, -2.53)

    def test_gap_selection_needs_no_pose_or_goal(self):
        node = SimpleNamespace()
        target = ObstacleCourseNode.gap_target(node, np.empty((0, 2)))
        np.testing.assert_allclose(target, [.2, 0.], atol=1e-12)

    def test_gap_selection_responds_to_obstacle_side(self):
        points = np.array([[.30, -.14]])
        left = ObstacleCourseNode.gap_target(SimpleNamespace(), points)
        right = ObstacleCourseNode.gap_target(SimpleNamespace(), points * [1., -1.])
        self.assertGreater(left[1], 0.)
        self.assertLess(right[1], 0.)
        np.testing.assert_allclose(left, right * [1., -1.], atol=.006)

    def test_course_pose_does_not_change_gap_selection_for_the_same_scan(self):
        points = np.array([[.30, -.14]])
        targets = [ObstacleCourseNode.gap_target(SimpleNamespace(pose=np.array(p)), points)
                   for p in ([0., 0., 0.], [9., -8., 175.])]
        np.testing.assert_allclose(*targets)

    def test_finish_waits_without_an_observed_wall(self):
        node = SimpleNamespace(
            state='finish', lane=-2.5,
            to_course=lambda: np.array([4., -2.5, 0.]),
            observe_panels=lambda: (10., np.empty((0, 2))))
        self.assertEqual(ObstacleCourseNode.panel_command(node), (0., 0.))
        self.assertEqual(node.state, 'finish')


if __name__ == '__main__':
    unittest.main()
