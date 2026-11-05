"""Check that 20 cm gap commitments survive motion but yield to safety."""
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from rb2301_fp.obstacle_course import ObstacleCourseNode


class GapCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.node = SimpleNamespace(
            pose=np.array([1., 2., 90.]),
            lidar={}, body_points=np.empty((0, 2)),
            gap_target=Mock(return_value=np.array([.2, 0.])),
            steer=Mock(return_value=(.18, 0.)),
            safe_command=Mock(side_effect=lambda points, v, w: (v, w)),
            stop_if_stalled=Mock(side_effect=lambda v, w: (v, w)),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)))
        self.scan = SimpleNamespace(ranges=[float('inf')]*181,
                                    angle_min=-np.pi/2, angle_increment=np.pi/180,
                                    range_min=.05, range_max=10.)

    def command(self):
        return ObstacleCourseNode.gap_command(self.node, self.scan)

    def test_commitment_stays_fixed_as_robot_translates_and_turns(self):
        self.command()
        np.testing.assert_allclose(self.node.gap_waypoint, [1., 2.2])
        self.node.pose = np.array([1., 2.08, 0.])
        self.scan.ranges[90] = .4
        self.command()
        self.node.gap_target.assert_called_once()
        np.testing.assert_allclose(self.node.steer.call_args.args[0], [0., .2], atol=1e-12)
        np.testing.assert_allclose(self.node.target, [0., .12], atol=1e-12)
        self.assertEqual(self.node.safe_command.call_count, 2)

    def test_reaching_waypoint_selects_next_gap(self):
        self.command()
        self.node.pose[1] = 2.19
        self.command()
        self.assertEqual(self.node.gap_target.call_count, 2)
        np.testing.assert_allclose(self.node.gap_waypoint, [1., 2.39])

    def test_passing_waypoint_selects_next_gap_instead_of_turning_back(self):
        self.command()
        self.node.pose[1] = 2.24
        self.command()
        self.assertEqual(self.node.gap_target.call_count, 2)
        np.testing.assert_allclose(self.node.gap_waypoint, [1., 2.44])

    def test_safety_stop_cancels_commitment_and_replans_on_next_tick(self):
        self.command()
        self.node.safe_command.side_effect = None
        self.node.safe_command.return_value = (0., 0.)
        self.assertEqual(self.command(), (0., 0.))
        self.assertIsNone(self.node.gap_waypoint)
        self.command()
        self.assertEqual(self.node.gap_target.call_count, 2)

    def test_safety_slowdown_keeps_commitment(self):
        self.node.safe_command.side_effect = None
        self.node.safe_command.return_value = (.09, 0.)
        self.command()
        self.command()
        self.node.gap_target.assert_called_once()
        self.assertIsNotNone(self.node.gap_waypoint)

    def test_large_open_gap_reports_a_return_inside_the_safety_footprint(self):
        self.node.body_points = np.array([[.19, 0.]])
        self.node.safe_command = MethodType(ObstacleCourseNode.safe_command, self.node)
        self.assertEqual(self.command(), (0., 0.))
        self.assertEqual(self.node.gap_debug['motion_status'], 'return_inside_footprint')
        self.assertEqual(self.node.gap_debug['inside_points'], 1)
        self.assertEqual(self.node.gap_debug['closest_inside'], [.19, 0.])

    def test_unblocked_command_is_distinguishable_from_a_safety_stop(self):
        self.command()
        self.assertEqual(self.node.gap_debug['motion_status'], 'motion_commanded')
        self.assertEqual(self.node.gap_debug['inside_points'], 0)
        self.assertEqual(self.node.gap_debug['safe'], [.18, 0.])

    def test_stall_stop_cancels_commitment(self):
        self.node.stop_if_stalled.side_effect = None
        self.node.stop_if_stalled.return_value = (0., 0.)
        self.assertEqual(self.command(), (0., 0.))
        self.assertIsNone(self.node.gap_waypoint)

    def test_stop_clears_waypoint(self):
        self.command()
        self.node.steer = MethodType(ObstacleCourseNode.steer, self.node)
        self.node.steer(None)
        self.assertIsNone(self.node.gap_waypoint)

    def test_tight_openings_shorten_the_commitment(self):
        angles = np.deg2rad(np.arange(-85, 86))
        points = np.column_stack((np.cos(angles), np.sin(angles)))*.34 + [.07, 0.]
        node = SimpleNamespace()
        target = ObstacleCourseNode.gap_target(node, points)
        self.assertAlmostEqual(np.linalg.norm(target), .12)
        self.assertEqual(node.gap_horizon, .12)

    def test_distant_obstacle_changes_gap_without_lengthening_commitment(self):
        target = ObstacleCourseNode.gap_target(SimpleNamespace(), np.array([[.45, 0.]]))
        self.assertAlmostEqual(np.linalg.norm(target), .2)
        self.assertGreater(abs(target[1]), .01)

    def test_rotating_past_target_heading_does_not_drop_the_world_waypoint(self):
        self.command()
        self.node.pose[2] = -90.
        self.command()
        self.node.gap_target.assert_called_once()
        np.testing.assert_allclose(self.node.gap_waypoint, [1., 2.2])

    def test_short_segment_finishes_with_heading_aligned_instead_of_orbiting_endpoint(self):
        self.node.pose = np.array([0., .02, 20.])
        self.node.speed_limit = .3
        self.node.gap_origin = np.array([0., 0.])
        self.node.gap_waypoint = np.array([.2, 0.])
        self.node.steer = MethodType(ObstacleCourseNode.steer, self.node)
        elapsed = [0]
        self.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=elapsed[0]))
        for _ in range(200):
            elapsed[0] += 50_000_000
            v, w = self.command()
            yaw = np.deg2rad(self.node.pose[2])
            self.node.pose += [.05*v*np.cos(yaw), .05*v*np.sin(yaw), np.rad2deg(.05*w)]
            if self.node.pose[0] >= .18:
                break
        self.assertGreaterEqual(self.node.pose[0], .18)
        self.assertLess(abs(self.node.pose[1]), .05)
        self.assertLess(abs(self.node.pose[2]), 15.)
        self.node.gap_target.assert_not_called()


if __name__ == '__main__':
    unittest.main()
