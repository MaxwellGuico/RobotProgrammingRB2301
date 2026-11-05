"""Clear new openings should win over repeated trips through the same space."""
import unittest

import numpy as np

from rb2301_fp.gap_navigation import choose_gap, rotate_points


class GapSelectionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[.45, 0.]])
        self.history = np.array([[.45, -.65], [.7, -.4], [.2, -.8]])

    def test_previously_visited_side_yields_to_a_clear_new_side(self):
        ordinary = choose_gap(self.points)
        remembered = choose_gap(self.points, pose=[0., 0., 0.], history=self.history)
        self.assertLess(ordinary.target[1], 0.)
        self.assertGreater(remembered.target[1], 0.)
        self.assertAlmostEqual(np.linalg.norm(remembered.target), .2)
        self.assertEqual(remembered.horizon, ordinary.horizon)

    def test_translation_and_rotation_of_history_preserve_the_robot_frame_choice(self):
        reference = choose_gap(self.points, pose=[0., 0., 0.], history=self.history)
        pose = np.array([3., 4., 90.])
        history = rotate_points(self.history, pose[2])+pose[:2]
        transformed = choose_gap(self.points, pose=pose, history=history)
        np.testing.assert_allclose(reference.target, transformed.target)

    def test_backtracking_remains_available_when_all_openings_are_visited(self):
        grid = np.array([(x, y) for x in np.arange(-1., 1.01, .1)
                         for y in np.arange(-1., 1.01, .1)])
        original = choose_gap(self.points)
        fallback = choose_gap(self.points, pose=[0., 0., 0.], history=grid)
        np.testing.assert_allclose(fallback.target, original.target)

    def test_history_cannot_make_a_blocked_scan_traversable(self):
        angles = np.deg2rad(np.arange(-85, 86))
        points = np.c_[np.cos(angles), np.sin(angles)]*.19+[.07, 0.]
        self.assertIsNone(choose_gap(points, pose=[0., 0., 0.], history=self.history).target)


if __name__ == '__main__':
    unittest.main()
