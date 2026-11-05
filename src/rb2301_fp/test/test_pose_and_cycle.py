"""Check hardware frame/start behavior and the shortened cycle learner."""
import math
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rb2301_fp.obstacle_course import ObstacleCourseNode, load_frame


class PoseAndCycleTests(unittest.TestCase):
    def real_node(self):
        frame, course = load_frame(3)
        node = SimpleNamespace(is_simulation=False, frame=frame, course=course,
                               capture_start=True, pose=None, start=None, get_logger=Mock())
        node.to_course = MethodType(ObstacleCourseNode.to_course, node)
        return node

    def observation(self, node, x, y):
        """Build raw Optitrack data for a known maze pose facing maze -y."""
        frame = node.frame
        r = math.radians(frame['rotation_deg'])
        msg = PoseStamped()
        msg.pose.position.x = frame['origin_x'] + math.cos(r)*x - math.sin(r)*y
        msg.pose.position.y = frame['origin_y'] + math.sin(r)*x + math.cos(r)*y
        yaw = math.radians(-90 + frame['rotation_deg'] + frame['heading_offset_deg'])
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        return msg

    def test_start_is_captured_once_and_course_axes_match_the_drawing(self):
        node = self.real_node()
        for maze, expected in (((-2, 3), (0, 0, 0)),
                               ((-2, 2), (1, 0, 0)),
                               ((-3, 3), (0, -1, 0))):
            ObstacleCourseNode.pose_callback(node, self.observation(node, *maze))
            np.testing.assert_allclose(node.to_course(), expected, atol=1e-9)
        self.assertFalse(node.capture_start)
        np.testing.assert_allclose(node.start, [0., 0.], atol=1e-9)

    def test_invalid_pose_does_not_capture_the_start(self):
        node = self.real_node()
        msg = self.observation(node, -2, 3)
        msg.pose.position.x = float('nan')
        ObstacleCourseNode.pose_callback(node, msg)
        self.assertTrue(node.capture_start)
        self.assertIsNone(node.pose)

    def test_simulation_pose_is_not_rotated_into_the_hardware_frame(self):
        node = SimpleNamespace(is_simulation=True, capture_start=False,
                               course=None, start=None)
        node.to_course = MethodType(ObstacleCourseNode.to_course, node)
        msg = Odometry()
        msg.pose.pose.position.x, msg.pose.pose.position.y = 1.2, -.4
        msg.pose.pose.orientation.w = 1.
        ObstacleCourseNode.pose_callback(node, msg)
        np.testing.assert_allclose(node.pose, [1.2, -.4, 0.])

    def test_hardware_commands_keep_the_speed_cap(self):
        node = SimpleNamespace(speed_limit=.3, publisher_=SimpleNamespace(publish=Mock()))
        ObstacleCourseNode.move_2D(node, x=9., y=-9., turn=9.)
        msg = node.publisher_.publish.call_args.args[0]
        self.assertEqual((msg.linear.x, msg.linear.y, msg.angular.z), (.3, -.3, .6))

    def test_publisher_never_sends_negative_forward_velocity(self):
        node = SimpleNamespace(speed_limit=.3, publisher_=SimpleNamespace(publish=Mock()))
        for requested in (-.001, -.08, -1.):
            ObstacleCourseNode.move_2D(node, x=requested, turn=-.2)
            msg = node.publisher_.publish.call_args.args[0]
            self.assertEqual(msg.linear.x, 0.)
            self.assertAlmostEqual(msg.angular.z, -.2)

    def test_cycle_is_learned_from_a_repeating_waveform(self):
        t = np.arange(12., 44., .1)
        y = -2.6 + .6*np.cos(2*np.pi*t/5.35)
        learner = MethodType(ObstacleCourseNode.learn_cycle, SimpleNamespace())
        model = learner({'samples': list(zip(t, y))})
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model['period'], 5.35, delta=.051)

    def test_nonrepeating_motion_is_not_given_a_cycle(self):
        t = np.arange(0., 32., .1)
        learner = MethodType(ObstacleCourseNode.learn_cycle, SimpleNamespace())
        self.assertIsNone(learner({'samples': list(zip(t, .1*t))}))


if __name__ == '__main__':
    unittest.main()
