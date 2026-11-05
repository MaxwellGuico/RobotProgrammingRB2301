"""Check simulation markers against real gap selection and retained telemetry."""
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
from visualization_msgs.msg import Marker

from rb2301_fp.obstacle_course import ObstacleCourseNode
from rb2301_fp.rviz_sim_view import build_markers, gap_geometry, recorded_lookahead


class RvizSimulationTests(unittest.TestCase):
    def setUp(self):
        self.debug = dict(simulation=True, pose=[1., 2., 90.], state='approach',
                          target=[.3, 0.], tracks=[], obstacle=1)

    def markers(self, **overrides):
        return build_markers(dict(self.debug, **overrides), None, (.18, .2), Time())

    def test_retained_waypoint_is_transformed_into_odom_frame(self):
        messages = self.markers().markers
        arrow = next(m for m in messages if m.ns == 'committed_waypoint' and m.type == Marker.ARROW)
        self.assertAlmostEqual(arrow.points[1].x, 1.)
        self.assertAlmostEqual(arrow.points[1].y, 2.3)
        self.assertTrue(all(m.header.frame_id == 'odom' for m in messages[1:]))

    def test_observed_tracks_are_separate_from_the_selected_panel(self):
        tracks = [dict(x=4., y=-2.5, period=6.), dict(x=4.6, y=-2.4, period=None)]
        messages = self.markers(state='observe', panel_x=4., lane=-2.5, tracks=tracks).markers
        selected = [m for m in messages if m.ns == 'selected_swinging_obstacle']
        observed = [m for m in messages if m.ns == 'observed_swinging_obstacles']
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(observed), 1)
        self.assertAlmostEqual(selected[0].pose.position.x, 4.)
        self.assertTrue(any('period 6.00 s' in m.text for m in messages))

    def test_previous_panel_is_not_highlighted_as_selected_during_scout(self):
        messages = self.markers(state='scout', panel_x=4.,
                                tracks=[dict(x=4., y=-2.5, period=6.)]).markers
        self.assertFalse(any(m.ns == 'selected_swinging_obstacle' for m in messages))

    def test_gap_geometry_matches_controller_eligibility(self):
        for points in (np.empty((0, 2)), np.array([[.3, -.14]]),
                       np.c_[np.cos(np.deg2rad(np.arange(-85, 86))),
                             np.sin(np.deg2rad(np.arange(-85, 86)))]*.54+[.07, 0.]):
            directions, groups = gap_geometry(points)
            target = ObstacleCourseNode.gap_target(SimpleNamespace(), points)
            self.assertEqual(bool(groups), target is not None)
            if target is not None:
                index = np.argmax(directions@target)
                self.assertTrue(any(index in group for group in groups))

    def test_missing_pose_and_hardware_telemetry_do_not_use_simulation_coordinates(self):
        for debug in ({}, dict(simulation=True, pose=None), dict(self.debug, simulation=False)):
            messages = build_markers(debug, None, None, Time()).markers
            self.assertEqual(len(messages), 2)
            self.assertIn('Waiting for simulation', messages[-1].text)

    def test_old_markers_are_cleared_and_new_markers_serialize(self):
        messages = self.markers()
        self.assertEqual(messages.markers[0].action, Marker.DELETEALL)
        self.assertGreater(len(serialize_message(messages)), 0)
        self.assertTrue(all(m.lifetime.nanosec for m in messages.markers[1:]))

    def test_no_waypoint_marker_is_invented_when_telemetry_has_no_target(self):
        self.assertFalse(any(m.ns == 'committed_waypoint' for m in self.markers(target=None).markers))

    def test_replay_uses_its_saved_lookahead_instead_of_the_current_setting(self):
        messages = build_markers(self.debug, None, (0., 0.), Time(),
                                 body_points=[], lookahead=.4).markers
        arrow = next(m for m in messages if m.ns == 'next_gap_preview')
        self.assertAlmostEqual(arrow.points[1].y, 2.4)
        self.assertTrue(any('commitment ≤ 0.40 m' in m.text for m in messages))

    def test_snapshot_settings_are_read_without_executing_the_snapshot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'trajectory.jsonl'
            path.with_name('obstacle_course.py').write_text(
                'GAP_LOOKAHEAD = .40\nraise RuntimeError("must not execute")\n')
            self.assertEqual(recorded_lookahead(path), .4)

    def test_preview_matches_the_controller_for_the_same_points(self):
        points = np.array([[.30, -.14]])
        target = ObstacleCourseNode.gap_target(SimpleNamespace(), points)
        messages = build_markers(self.debug, None, (0., 0.), Time(), body_points=points).markers
        arrow = next(m for m in messages if m.ns == 'next_gap_preview')
        self.assertAlmostEqual(arrow.points[1].x, 1.-target[1])
        self.assertAlmostEqual(arrow.points[1].y, 2.+target[0])

    def test_preview_uses_the_controllers_recorded_path_history(self):
        from rb2301_fp.gap_navigation import choose_gap, rotate_points
        points = np.array([[.45, 0.]])
        old = rotate_points(np.array([[.45, -.65], [.7, -.4], [.2, -.8]]), 90.)+[1., 2.]
        debug = dict(self.debug, navigation=dict(history=old.tolist()))
        expected = choose_gap(points, pose=debug['pose'], history=old).target
        messages = build_markers(debug, None, (0., 0.), Time(), body_points=points).markers
        arrow = next(m for m in messages if m.ns == 'next_gap_preview')
        self.assertAlmostEqual(arrow.points[1].x, 1.-expected[1])
        self.assertAlmostEqual(arrow.points[1].y, 2.+expected[0])
        self.assertTrue(any(m.ns == 'visited_space' for m in messages))


if __name__ == '__main__':
    unittest.main()
