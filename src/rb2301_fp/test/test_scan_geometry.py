"""Check frame directions without starting ROS nodes or using hardware."""
import math
from types import SimpleNamespace
import unittest

from rb2301_fp.obstacle_course import scan_to_body


def scan(ranges, start=0.0, step=math.pi / 2):
    return SimpleNamespace(ranges=ranges, angle_min=start, angle_increment=step,
                           range_min=0.05, range_max=10.0)


class ScanGeometryTests(unittest.TestCase):
    def test_zero_yaw_disables_mount_rotation(self):
        points, _ = scan_to_body(scan([1.]), yaw_deg=0.0, x_offset=0.0)
        self.assertEqual(points, [(1.0, 0.0)])

    def test_backward_mount_all_four_directions(self):
        points, indices = scan_to_body(scan([1.0] * 4), yaw_deg=180.0, x_offset=0.0)
        expected = [(-1., 0.), (0., -1.), (1., 0.), (0., 1.)]
        for actual, wanted in zip(points, expected):
            self.assertAlmostEqual(actual[0], wanted[0])
            self.assertAlmostEqual(actual[1], wanted[1])
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_front_selection_uses_robot_axes(self):
        points, _ = scan_to_body(scan([1.] * 8, step=math.pi / 4), yaw_deg=180.,
                                 x_offset=0., front_only=True)
        self.assertEqual(len(points), 5)
        self.assertTrue(all(x >= -1e-9 for x, _ in points))
        self.assertLess(points[0][1], 0.)
        self.assertGreater(points[-1][1], 0.)

    def test_front_scan_seam_and_invalid_beam_preserve_cluster_gap(self):
        msg = scan([1., float('inf'), 1., 1., 1., 1., 1., 1.], step=math.pi / 4)
        _, indices = scan_to_body(msg, front_only=True)
        self.assertEqual(indices, [0, 1, 2, 4])

    def test_clockwise_scan(self):
        points, _ = scan_to_body(scan([1.] * 8, step=-math.pi / 4), yaw_deg=180.,
                                 x_offset=0., front_only=True)
        self.assertLess(points[0][1], points[-1][1])

    def test_mount_translation_is_not_rotated(self):
        points, _ = scan_to_body(scan([1.]), yaw_deg=180., x_offset=.1, y_offset=.2)
        self.assertAlmostEqual(points[0][0], -.9)
        self.assertAlmostEqual(points[0][1], .2)

    def test_invalid_ranges_and_detection_limits(self):
        msg = scan([float('nan'), float('inf'), .01, .5, 4., 11.])
        points, indices = scan_to_body(msg, max_range=3.5)
        self.assertEqual(indices, [3])
        self.assertEqual(len(points), 1)
        self.assertEqual(scan_to_body(scan([])), ([], []))


if __name__ == '__main__':
    unittest.main()
