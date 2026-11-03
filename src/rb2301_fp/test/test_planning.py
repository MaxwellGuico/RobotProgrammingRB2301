"""Width-only route acceptance: no extra turning radius or length requirement."""
import unittest
from types import MethodType, SimpleNamespace

import numpy as np
from rb2301_fp.obstacle_course import ObstacleCourseNode


class PlanningTests(unittest.TestCase):
    def planner(self, gap):
        planner = SimpleNamespace(nav_settings=dict(map_resolution=.025,
            footprint_half_width=.096, preferred_clearance=.30,
            clearance_margin=0.0, frontier_revisit_distance=0.0),
            failed_goals={}, sensor_time=lambda: 0.0)
        for name in ('neighbours','planning_space','reconstruct_path','Astar'):
            setattr(planner,name,MethodType(getattr(ObstacleCourseNode,name),planner))
        wall = round(gap/(2*.025))
        planner.observation_grid = {(x,y):100 if abs(y)==wall else 0
                                    for x in range(-20,21) for y in range(-wall,wall+1)}
        planner.planning_space()
        return planner

    def test_twenty_centimetre_gap_accepts_nineteen_point_two_centimetre_robot(self):
        planner = self.planner(.20)
        self.assertTrue(planner.Astar((-10,0),(10,0)))
        self.assertGreater(len(planner.cell_clearance),0)

    def test_fifteen_centimetre_gap_is_too_narrow(self):
        planner = self.planner(.15)
        self.assertFalse(planner.Astar((-10,0),(10,0)))

    def test_unknown_start_is_not_connected(self):
        planner = self.planner(.20)
        del planner.observation_grid[(-10,0)]
        self.assertFalse(planner.Astar((-10,0),(10,0)))

    def test_no_sideways_velocity_can_be_published(self):
        messages=[]
        publisher=SimpleNamespace(publish=messages.append)
        node=SimpleNamespace(publisher_=publisher,max_translate_velocity=.3)
        ObstacleCourseNode.move_2D(node,.1,.2,.1)
        self.assertEqual(messages[0].linear.y,0.0)
        self.assertAlmostEqual(messages[0].linear.x,.1)


class ForwardFollowerTests(unittest.TestCase):
    def command(self, point):
        from collections import defaultdict
        node = SimpleNamespace(pose=np.array([0.,0.,0.]),route=[(0.,0.),(-1.,0.)],goal=(-1.,0.),
            nav_settings=dict(look_ahead_distance=.18,linear_speed=.1,distance_gain=1.,
                              turn_in_place_angle=8.,map_resolution=.05),
            get_carrot=lambda *args:np.array(point),
            heading_command=lambda error,now:float(np.clip(2*error,-.3,.3)),
            observation_grid=SimpleNamespace(get=lambda cell:0))
        for name in ('world_cell','ray_cells'):
            setattr(node,name,MethodType(getattr(ObstacleCourseNode,name),node))
        return ObstacleCourseNode.follow_carrot(node,0.)[0]

    def test_point_behind_causes_turn_instead_of_reverse(self):
        speed, turn=self.command((-.18,0.))
        self.assertEqual(speed,0.)
        self.assertNotEqual(turn,0.)

    def test_point_ahead_drives_forward(self):
        speed, turn=self.command((.18,0.))
        self.assertGreater(speed,0.)
        self.assertEqual(turn,0.)


if __name__=='__main__': unittest.main()
