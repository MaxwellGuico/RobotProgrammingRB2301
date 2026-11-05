"""Check smooth turning under waypoint changes and noisy heading measurements."""
from types import SimpleNamespace
import unittest

import numpy as np

from rb2301_fp.obstacle_course import ObstacleCourseNode


class SteeringStabilityTests(unittest.TestCase):
    def setUp(self):
        self.time = SimpleNamespace(nanoseconds=0)
        self.node = SimpleNamespace(
            pose=np.array([0., 0., 0.]), speed_limit=.3,
            get_clock=lambda: SimpleNamespace(now=lambda: self.time))

    def steer(self, angle):
        self.time.nanoseconds += 50_000_000
        return ObstacleCourseNode.steer(
            self.node, .4*np.array([np.cos(angle), np.sin(angle)]))

    def test_switching_target_cannot_reverse_turn_abruptly(self):
        previous = 0.
        for angle in [1.]*10 + [-1.]*20:
            _, turn = self.steer(angle)
            self.assertLessEqual(abs(turn-previous), .075+1e-12)
            self.assertLessEqual(abs(turn), .6+1e-12)
            previous = turn
        self.assertLess(previous, 0.)

    def test_small_alternating_target_errors_do_not_chatter(self):
        for angle in np.deg2rad([1.5, -1.5]*20):
            _, turn = self.steer(angle)
            self.assertEqual(turn, 0.)

    def test_yaw_measurement_spike_is_filtered(self):
        self.steer(.2)
        self.node.pose[2] = 5.
        self.steer(.2)
        raw_rate = np.deg2rad(5)/.05
        self.assertGreater(self.node.steering_rate, 0.)
        self.assertLess(self.node.steering_rate, raw_rate/2)

    def test_stop_resets_turn_history(self):
        for _ in range(10):
            self.steer(1.)
        self.assertEqual(ObstacleCourseNode.steer(self.node, None), (0., 0.))
        _, turn = self.steer(-1.)
        self.assertAlmostEqual(turn, -.075)

    def test_forward_speed_reduces_near_a_short_waypoint(self):
        far, _ = ObstacleCourseNode.steer(self.node, np.array([.2, 0.]))
        near, _ = ObstacleCourseNode.steer(self.node, np.array([.04, 0.]))
        self.assertAlmostEqual(far, .18)
        self.assertAlmostEqual(near, .06)
        self.assertLess(near, far)

    def test_closed_loop_turn_converges_without_repeated_overshoot(self):
        yaw, target_yaw, turns = 0., .8, []
        for _ in range(200):
            self.node.pose[2] = np.rad2deg(yaw)
            _, turn = self.steer(target_yaw-yaw)
            yaw += turn*.05
            turns.append(turn)
        self.assertLess(abs(target_yaw-yaw), np.deg2rad(2))
        self.assertGreaterEqual(min(turns), -1e-12)


if __name__ == '__main__':
    unittest.main()
