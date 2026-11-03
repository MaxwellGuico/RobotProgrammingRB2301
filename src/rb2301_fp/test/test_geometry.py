"""Projection and width-only line checks without a running ROS graph."""
import math
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class GeometryTests(unittest.TestCase):
    def test_scan_applies_mount_then_robot_rotation_and_ignores_no_returns(self):
        scan = SimpleNamespace(ranges=[1.0, math.inf, math.nan, 5.0, 0.01],
                               angle_min=0.0, angle_increment=0.1,
                               range_min=0.1, range_max=5.0)
        points = ObstacleCourseNode.project_scan(None, scan, (2.0, 3.0, 90.0), (0.5, 0.0, math.pi/2))
        np.testing.assert_allclose(points, [[1.0, 3.5]], atol=1e-12)

    def line(self, gap):
        points=np.array([(x,y) for x in np.linspace(0.,1.,41) for y in (-gap/2,gap/2)])
        node=SimpleNamespace(scan_points=points,nav_settings=dict(map_resolution=.05,footprint_half_width=.096))
        return ObstacleCourseNode.segment_clear(node,np.array((0.,0.)),np.array((1.,0.)))

    def test_width_check_accepts_twenty_centimetre_gap(self):
        self.assertTrue(self.line(.20))

    def test_width_check_rejects_fifteen_centimetre_gap(self):
        self.assertFalse(self.line(.15))

    def test_no_scan_cannot_authorize_a_forward_line(self):
        node=SimpleNamespace(scan_points=np.empty((0,2)))
        self.assertFalse(ObstacleCourseNode.segment_clear(node,np.zeros(2),np.ones(2)))


if __name__ == '__main__': unittest.main()
