"""Scan preparation uses configured extrinsics, with no TF-tree dependency."""
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class SensorPreparationTests(unittest.TestCase):
    def setUp(self):
        # No frame_id or TF buffer is supplied: this path uses the configured mount.
        scan=SimpleNamespace(ranges=[1.,float('inf')],angle_min=0.,angle_increment=.1,
                             range_min=.05,range_max=10.)
        self.node=SimpleNamespace(scan_msg=scan,scan_stamp=1.,scan_received=10.,pose_received=10.,
            pose_history=[(1.,np.array((2.,3.,90.)))],sensor_time=lambda:10.1,
            nav_settings=dict(scan_timeout=.5,pose_timeout=.5,pose_scan_tolerance=.2),
            projected_stamp=None,laser_transform=(0.,0.,0.))
        self.node.project_scan=Mock(wraps=MethodType(ObstacleCourseNode.project_scan,self.node))

    def test_projects_scan_without_a_tf_tree(self):
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'READY_STOPPED')
        np.testing.assert_allclose(self.node.scan_points,[[2.,4.]],atol=1e-12)
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'READY_STOPPED')
        self.node.project_scan.assert_called_once()

    def test_stale_scan_and_pose_are_rejected(self):
        self.node.scan_received=9.
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'STALE_SCAN')
        self.node.scan_received=10.
        self.node.pose_received=9.
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'STALE_POSE')

    def test_missing_sensor_data_waits(self):
        self.node.scan_msg=None
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'WAITING_FOR_SENSORS')

    def test_mismatched_timestamps_are_rejected(self):
        self.node.pose_history=[(0.,np.array((2.,3.,90.)))]
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'POSE_SCAN_TIME_MISMATCH')

    def test_empty_returns_do_not_authorize_navigation(self):
        self.node.scan_msg.ranges=[float('inf'),float('nan')]
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'NO_OBSTACLE_RETURNS')

    def test_invalid_scan_angles_are_rejected(self):
        self.node.scan_msg.angle_increment=float('nan')
        self.assertEqual(ObstacleCourseNode.prepare_scan(self.node),'INVALID_SCAN_METADATA')


if __name__=='__main__':unittest.main()
