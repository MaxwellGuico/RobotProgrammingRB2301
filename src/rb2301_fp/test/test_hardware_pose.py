"""The physical robot pose and timestamped mapping pose share one frame."""
from collections import deque
import math
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from geometry_msgs.msg import PoseStamped
from rb2301_fp.obstacle_course import ObstacleCourseNode, optitrack_to_maze


class HardwarePoseTests(unittest.TestCase):
    def test_displayed_pose_matches_mapping_pose_for_nonzero_y(self):
        for y in (-2.,2.):
            with self.subTest(y=y):
                node=SimpleNamespace(is_simulation=False,pose_stamp=None,pose_history=deque(),
                    frame=dict(origin_x=.3,origin_y=-.7,rotation_deg=20.,heading_offset_deg=-100.),
                    sensor_time=lambda:5.)
                node.yaw_from_quaternion=MethodType(ObstacleCourseNode.yaw_from_quaternion,node)
                message=PoseStamped()
                message.header.stamp.sec=1
                message.pose.position.x,message.pose.position.y=1.,y
                message.pose.orientation.z=math.sin(math.radians(45.)/2)
                message.pose.orientation.w=math.cos(math.radians(45.)/2)
                ObstacleCourseNode.optitrack_callback(node,message)
                ObstacleCourseNode.remember_pose(node,message)
                x,maze_y,heading=optitrack_to_maze(1.,y,45.,node.frame)
                expected=(x,-maze_y,heading)
                np.testing.assert_allclose(node.pose,expected)
                np.testing.assert_allclose(node.pose_history[-1][1],expected)


if __name__=='__main__':unittest.main()
