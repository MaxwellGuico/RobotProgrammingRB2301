"""The simple RViz message contains both the footprint and labelled carrot."""
from types import SimpleNamespace
import unittest

import numpy as np
from builtin_interfaces.msg import Time
from rb2301_fp.diagnostics import publish_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.messages=[]
        self.node=SimpleNamespace(
            nav_settings=dict(debug_visuals=True,debug_interval=.2,map_resolution=.05,
                              footprint_front=.182,footprint_rear=.034,footprint_half_width=.096),
            sensor_time=lambda:10.,last_debug_time=0.,debug_publisher=None,
            create_publisher=lambda *args:SimpleNamespace(publish=self.messages.append),
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:Time(sec=1))),
            sensor_ready=True,pose=np.array([1.,2.,90.]),scan_points=np.array([[3.,2.]]),
            route=[(1.,2.),(1.2,2.)],frontier_groups=[],goal=(1.4,2.),carrot=(1.2,2.),
            observation_grid={(0,0):0,(1,0):100},cell_world=lambda c:((c[0]+.5)*.05,(c[1]+.5)*.05))

    def test_footprint_and_carrot_have_visible_geometry(self):
        publish_diagnostics(self.node,'FOLLOWING')
        markers={marker.ns:marker for marker in self.messages[0].markers}
        self.assertEqual(len(markers['footprint'].points),5)
        self.assertAlmostEqual(markers['footprint'].points[0].x,1.096)
        self.assertAlmostEqual(markers['footprint'].points[0].y,1.966)
        self.assertEqual(markers['carrot_label'].text,'CARROT')
        self.assertEqual(len(markers['carrot_line'].points),2)
        self.assertTrue(all(marker.header.frame_id=='fp_world' for marker in markers.values()))
        self.assertLess(markers['free_space'].points[0].z,markers['route'].points[0].z)

    def test_disabled_diagnostics_do_not_publish(self):
        self.node.nav_settings['debug_visuals']=False
        publish_diagnostics(self.node,'FOLLOWING')
        self.assertEqual(self.messages,[])


if __name__=='__main__':unittest.main()
