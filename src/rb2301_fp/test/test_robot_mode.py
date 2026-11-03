"""Check startup mode selection without spinning or commanding the robot."""
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import rclpy
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np

from rb2301_fp.obstacle_course import ObstacleCourseNode


class RobotModeTests(unittest.TestCase):
    def make_node(self, args):
        rclpy.init(args=args)
        self.addCleanup(rclpy.shutdown)
        node = ObstacleCourseNode()
        self.addCleanup(node.destroy_node)
        return node

    def test_default_is_hardware_even_with_a_short_scan(self):
        node = self.make_node([])
        node.sub_scan_callback(SimpleNamespace(ranges=[1.0]*360))
        self.assertFalse(node.get_parameter('is_simulation').value)
        self.assertFalse(node.is_simulation)
        self.assertEqual(node.max_translate_velocity, 0.3)
        topics = {sub.topic_name for sub in node.subscriptions}
        self.assertNotIn('/odom', topics)
        self.assertTrue(any(topic.startswith('/vrpn_mocap/') for topic in topics))

    def test_ros_override_selects_simulation_even_with_a_long_scan(self):
        node = self.make_node(['--ros-args', '-p', 'is_simulation:=true'])
        node.sub_scan_callback(SimpleNamespace(ranges=[1.0]*720))
        self.assertTrue(node.get_parameter('is_simulation').value)
        self.assertTrue(node.is_simulation)
        self.assertEqual(node.max_translate_velocity, 1.4)
        self.assertIn('/odom', {sub.topic_name for sub in node.subscriptions})

    def test_mode_change_requires_a_restart(self):
        node = self.make_node([])
        result = node.set_parameters([Parameter('is_simulation', value=True)])[0]
        self.assertFalse(result.successful)
        self.assertFalse(node.is_simulation)

    def test_simulation_laser_mount_matches_robot_description(self):
        node = self.make_node(['--ros-args', '-p', 'is_simulation:=true'])
        robot = ET.parse(Path(__file__).resolve().parents[2] / 'rb2301_gz/urdf/nanocar_description.urdf')
        origin = robot.find("joint[@name='laser_joint']/origin")
        x, y, _ = map(float, origin.get('xyz').split())
        _, _, yaw = map(float, origin.get('rpy').split())
        self.assertEqual(node.laser_transform, (x, y, yaw))

    def test_laser_mount_can_be_calibrated_for_hardware(self):
        node = self.make_node(['--ros-args', '-p', 'laser_x:=0.2',
                               '-p', 'laser_y:=-0.01', '-p', 'laser_yaw:=0.5'])
        self.assertFalse(node.is_simulation)
        self.assertEqual(node.laser_transform, (0.2, -0.01, 0.5))

    def test_sensor_callbacks_work_immediately_after_construction(self):
        node = self.make_node(['--ros-args', '-p', 'is_simulation:=true'])
        self.assertEqual(node.prepare_scan(), 'WAITING_FOR_SENSORS')
        scan = LaserScan()
        scan.header.stamp.sec = 1
        scan.angle_increment = 0.1
        scan.range_min, scan.range_max = 0.05, 10.0
        scan.ranges = [1.0]
        pose = Odometry()
        pose.header.stamp.sec = 1
        pose.pose.pose.orientation.w = 1.0
        node.remember_scan(scan)
        node.remember_pose(pose)
        self.assertEqual(node.prepare_scan(), 'READY_STOPPED')
        np.testing.assert_allclose(node.scan_points[0],
            [0.11423+np.cos(3.1416), np.sin(3.1416)], atol=1e-12)

    def test_stop_before_first_pose_sends_zero_and_exits(self):
        node = self.make_node([])
        with patch.object(node, 'move_2D') as publish:
            node.stop_navigation()
            with self.assertRaises(SystemExit):
                node.timer_callback()
            publish.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
