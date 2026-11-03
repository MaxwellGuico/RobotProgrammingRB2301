"""The navigation loop uses one width check and still stops on stale data."""
from types import SimpleNamespace
import unittest

import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class NavigationDefaultsTests(unittest.TestCase):
    def step(self, sensor_state, width_clear=True, desired=(.1,.2)):
        events=[]
        commands=[]
        node=SimpleNamespace(
            nav_settings=dict(status_interval=1.0,look_ahead_distance=.18),stop_requested=False,calibrate=False,
            prepare_scan=lambda:sensor_state,sensor_time=lambda:.05,
            update_memory=lambda:events.append('mapping'),observe_motion=lambda:events.append('motion observation'),
            navigation_command=lambda now:(events.append('navigation') or (desired,'FOLLOWING')),
            segment_clear=lambda start,end:(events.append('width check') or width_clear),
            pose=np.zeros(3),scan_received=0.,pose_received=0.,observation_grid={},
            goals_reached=0,goals_selected=0,path_progress=0.,
            get_logger=lambda:SimpleNamespace(info=lambda text:None),
            completion_since=None,completion_count=0,completion_anchor=None,crossing_zone=None,
            move_2D=lambda *args:commands.append(args),blocked_since=None,
            sensor_state='FOLLOWING',last_status=0.,show_diagnostics=lambda state:events.append('display'))
        ObstacleCourseNode.navigation_step(node)
        return events,commands

    def test_ready_sensors_run_full_navigation_without_mode_switches(self):
        events,commands=self.step('READY_STOPPED')
        self.assertEqual(events,['mapping','motion observation','navigation','width check','display'])
        self.assertEqual(commands,[(.1,0.,.2)])

    def test_stale_sensors_still_command_zero(self):
        events,commands=self.step('STALE_SCAN')
        self.assertEqual(events,['display'])
        self.assertEqual(commands,[(0.,0.,0.)])

    def test_blocked_width_stops_translation_but_allows_turning(self):
        events,commands=self.step('READY_STOPPED',width_clear=False)
        self.assertIn('width check',events)
        self.assertEqual(commands,[(0.,0.,.2)])

    def test_turning_does_not_require_a_forward_line(self):
        events,commands=self.step('READY_STOPPED',width_clear=False,desired=(0.,.2))
        self.assertNotIn('width check',events)
        self.assertEqual(commands,[(0.,0.,.2)])

    def test_completion_check_does_not_require_a_stage_attribute(self):
        node=SimpleNamespace(nav_settings={},completion_command=lambda now:((0.,0.),'COURSE_COMPLETE'))
        self.assertEqual(ObstacleCourseNode.navigation_command(node,0.),((0.,0.),'COURSE_COMPLETE'))


if __name__=='__main__': unittest.main()
