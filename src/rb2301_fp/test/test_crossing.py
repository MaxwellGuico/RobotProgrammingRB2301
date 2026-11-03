"""A swinging obstacle can have different opening durations on its two sides."""
import unittest
from rb2301_fp.obstacle_course import update_clear_window, clear_window_remaining


class CrossingTests(unittest.TestCase):
    def test_short_opening_does_not_replace_long_phase_measurements(self):
        zone=dict(was_clear=False, clear_since=None, windows=[])
        stamp=0.0
        for side,duration in ((1,1.4),(-1,3.1),(1,1.5),(-1,3.2)):
            update_clear_window(zone,True,side,stamp)
            stamp+=duration
            update_clear_window(zone,False,0,stamp)
            stamp+=1.0
        update_clear_window(zone,True,1,stamp)
        self.assertAlmostEqual(clear_window_remaining(zone,stamp+.1,2),1.3)
        update_clear_window(zone,False,0,stamp+1.4)
        stamp+=2.4
        update_clear_window(zone,True,-1,stamp)
        self.assertAlmostEqual(clear_window_remaining(zone,stamp+.1,2),3.0)

    def test_one_matching_observation_cannot_authorize_crossing(self):
        zone=dict(was_clear=True, clear_since=10.0, clear_side=-1, windows=[(-1,3.1),(1,1.4)])
        self.assertEqual(clear_window_remaining(zone,10.1,2),0.0)

    def test_initial_partial_opening_is_not_a_measured_cycle(self):
        zone=dict(was_clear=None,clear_since=None,windows=[])
        update_clear_window(zone,True,1,0.0)
        self.assertIsNone(update_clear_window(zone,False,0,1.0))
        self.assertEqual(zone['windows'],[])


class StagingTests(unittest.TestCase):
    def test_staging_is_left_of_travel_for_both_corridor_directions(self):
        import numpy as np
        from types import SimpleNamespace
        from rb2301_fp.obstacle_course import ObstacleCourseNode
        for normal,axis in ((np.array((1.,0.)),np.array((0.,1.))),
                            (np.array((0.,1.)),np.array((1.,0.)))):
            with self.subTest(normal=normal):
                candidates={tuple(.75*normal+t*axis) for t in np.arange(-.4,.41,.05)}
                zone=dict(passed=False,centre=normal,normal=normal,axis=axis,low=-.5,high=.5)
                node=SimpleNamespace(pose=np.array((0.,0.,0.)),motion_zones=[zone],crossing_zone=None,
                    nav_settings=dict(mapping_range=2.5,footprint_front=.182,footprint_half_width=.096,
                        clearance_margin=0.,crossing_padding=.05,map_resolution=.05,motion_minimum_span=.3,
                        crossing_approach_spacing=.2,goal_cooldown=30.,obstacle_wait_timeout=15.,goal_tolerance=.08),
                    free_cells=candidates,cell_clearance=dict.fromkeys(candidates,.1),blocked_since=None,
                    planning_space=lambda:None,cell_world=lambda c:c,world_cell=lambda *p:tuple(p),
                    Astar=lambda a,b:[a,b],simplify_path=lambda p:p,route_stalled=lambda now:False,
                    follow_carrot=lambda now:((.1,0.),'FOLLOWING'),get_logger=lambda:SimpleNamespace(info=lambda text:None))
                def begin(route,now):node.goal=route[-1]
                node.begin_route=begin
                ObstacleCourseNode.moving_obstacle_command(node,0.)
                left=np.array((-normal[1],normal[0]))
                self.assertAlmostEqual(float(np.dot(node.goal,left)),.3)

    def test_numpy_boolean_scan_result_starts_a_complete_window(self):
        import numpy as np
        zone=dict(was_clear=None,clear_since=None,windows=[])
        update_clear_window(zone,np.bool_(False),0,0.)
        update_clear_window(zone,np.bool_(True),1,1.)
        self.assertEqual(zone['clear_since'],1.)


if __name__=='__main__': unittest.main()
