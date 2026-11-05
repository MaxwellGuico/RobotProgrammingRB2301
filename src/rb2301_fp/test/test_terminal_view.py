"""Check terminal coordinates, phase-specific aims, and velocity labels."""
import unittest
from rb2301_fp.terminal_view import navigation_targets, render_frame


class TerminalViewTests(unittest.TestCase):
    def setUp(self):
        self.debug = dict(pose=[1., 2., 90.], start=[0., 0.],
                          wait_point=[1.5, 2.5],
                          target=[.3, .1], state='position', obstacle=1,
                          simulation=True, tracks=[])

    def test_gap_target_is_rotated_and_translated_to_course_coordinates(self):
        waypoint, gap, _ = navigation_targets(self.debug, [])
        self.assertEqual(waypoint, [1.5, 2.5])
        self.assertAlmostEqual(gap[0], .9)
        self.assertAlmostEqual(gap[1], 2.3)

    def test_crossing_aim_is_ahead_instead_of_the_old_waiting_point(self):
        self.debug.update(state='cross', panel_x=4., lane=-2.5)
        waypoint, _, action = navigation_targets(self.debug, [])
        self.assertEqual(waypoint, (4.175, -2.5))
        self.assertEqual(action, 'cross')

    def test_approach_displays_the_sensed_gap_without_a_fixed_destination(self):
        self.debug['state'] = 'approach'
        waypoint, gap, action = navigation_targets(self.debug, [])
        self.assertEqual(waypoint, gap)
        self.assertEqual(action, 'follow gaps')
        text = '\n'.join(render_frame(100, 36, [], self.debug, (.1, .2), 'test'))
        self.assertNotIn('Goal G=', text)

    def test_observing_holds_the_current_position(self):
        self.debug['state'] = 'observe'
        waypoint, _, action = navigation_targets(self.debug, [])
        self.assertEqual(waypoint, [1., 2.])
        self.assertEqual(action, 'hold')

    def test_finish_estimate_uses_the_detected_wall(self):
        self.debug.update(pose=[4., -2.5, 0.], state='finish', lane=-2.5)
        waypoint, _, action = navigation_targets(self.debug, [(1., 0.)])
        self.assertEqual(waypoint, (4.64, -2.5))
        self.assertEqual(action, 'finish estimate')

    def test_planar_speed_includes_sideways_command(self):
        text = '\n'.join(render_frame(
            100, 36, [], self.debug, (.3, .4, -.2), 'test data'))
        for label in ('speed=0.50 m/s', 'vy=+0.40', 'wz=-0.20 rad/s',
                      'Start S=(+0.00,+0.00)', 'Finish: detected wall',
                      'Action W=(+1.50,+2.50)', 'Gap T=(+0.90,+2.30)'):
            self.assertIn(label, text)

    def test_small_terminal_preserves_footer_and_fits_every_line(self):
        for width, height in ((60, 24), (80, 24), (100, 36)):
            frame = render_frame(width, height, [], self.debug, (.1, .2), 'test')
            self.assertLess(len(frame), height)
            self.assertTrue(all(len(line) < width for line in frame))
            self.assertIn('--terminal-debug', frame[-1])

    def test_old_recorded_commands_and_missing_telemetry_still_render(self):
        text = '\n'.join(render_frame(80, 24, [], {}, (.1, .2), 'test'))
        self.assertIn('speed=0.10 m/s', text)
        self.assertIn('Finish: detected wall', text)
        text = '\n'.join(render_frame(80, 24, [], {}, None, 'test'))
        self.assertIn('waiting for commands', text)


if __name__ == '__main__':
    unittest.main()
