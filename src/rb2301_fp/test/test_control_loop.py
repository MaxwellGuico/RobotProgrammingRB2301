"""Check command ownership and the controller's stop/calibration behavior."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class ControlLoopTests(unittest.TestCase):
    def setUp(self):
        self.node = SimpleNamespace(
            pose=np.array([0., 0., 0.]), raw_pose=None, calibrate=False, _ticks=0,
            last_scan=SimpleNamespace(ranges=[float('inf')] * 360),
            panel_command=Mock(return_value=(0., 0.)),
            gap_command=Mock(return_value=(.18, .2)),
            steer=Mock(), move_2D=Mock())

    def tick(self):
        ObstacleCourseNode.timer_callback(self.node)

    def test_panel_pause_publishes_zero_without_starting_gap_navigation(self):
        self.node.progress = (0., self.node.pose.copy())
        self.tick()
        self.node.move_2D.assert_called_once_with(x=0., turn=0.)
        self.node.gap_command.assert_not_called()
        self.assertIsNone(self.node.progress)

    def test_gap_navigation_runs_when_panel_controller_hands_over(self):
        self.node.panel_command.return_value = None
        self.tick()
        self.node.gap_command.assert_called_once_with(self.node.last_scan)
        self.node.move_2D.assert_called_once_with(x=.18, turn=.2)

    def test_crossing_command_is_published_once(self):
        self.node.panel_command.return_value = (.4, -.1)
        self.tick()
        self.node.move_2D.assert_called_once_with(x=.4, turn=-.1)
        self.node.gap_command.assert_not_called()

    def test_calibration_never_enters_navigation(self):
        self.node.calibrate = True
        self.node.raw_pose = np.array([2., 3., 0.])
        self.tick()
        self.node.move_2D.assert_not_called()
        self.node.panel_command.assert_not_called()
        self.node.gap_command.assert_not_called()

    def test_missing_scan_stops_instead_of_reusing_a_command(self):
        self.node.last_scan = None
        self.tick()
        self.node.move_2D.assert_called_once_with()
        self.node.panel_command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
