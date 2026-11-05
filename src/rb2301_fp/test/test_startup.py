"""Check explicit mode selection and forwarding of normal ROS arguments."""
import unittest
from unittest.mock import Mock, patch

from rb2301_fp import obstacle_course


class StartupTests(unittest.TestCase):
    def launch(self, args, expected, ros_args=None, interrupted=False):
        """Run the CLI without sensors or motion, checking startup and cleanup."""
        node = Mock()
        with patch.object(obstacle_course, 'rclpy') as ros, \
                patch.object(obstacle_course, 'ObstacleCourseNode', return_value=node) as factory:
            ros.ok.return_value = True
            if interrupted:
                ros.spin.side_effect = KeyboardInterrupt
            obstacle_course.main(args)
            factory.assert_called_once_with(*expected)
            ros.init.assert_called_once_with(args=ros_args or [])
            ros.spin.assert_called_once_with(node)
            ros.spin_once.assert_not_called()
            node.move_2D.assert_called_once_with()
            node.destroy_node.assert_called_once_with()
            ros.shutdown.assert_called_once_with()

    def test_default_is_real_without_waiting_for_a_scan(self):
        self.launch([], (False, False, False, False))

    def test_simulation_must_be_explicit_and_ros_parameters_are_forwarded(self):
        ros_args = ['--ros-args', '-p', 'use_sim_time:=true']
        self.launch(['--simulation', '--terminal-debug', *ros_args],
                    (True, False, True, False), ros_args)

    def test_real_start_and_calibration_flags_still_work(self):
        self.launch(['--start-here', '--calibrate', '--terminal-debug'],
                    (False, True, True, True))

    def test_interrupt_stops_and_cleans_up_the_node(self):
        self.launch(['--simulation'], (True, False, False, False), interrupted=True)


if __name__ == '__main__':
    unittest.main()
