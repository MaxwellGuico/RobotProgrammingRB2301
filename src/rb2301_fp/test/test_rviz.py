"""The RViz entry point preserves the caller's ROS setup and uses standard markers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

PACKAGE = Path(__file__).resolve().parents[1]


class RvizTests(unittest.TestCase):
    def test_config_uses_foxy_and_jazzy_marker_topic_property(self):
        config = yaml.safe_load((PACKAGE/'rviz/fp_robot_thinking.rviz').read_text())
        displays = config['Visualization Manager']['Displays']
        markers = next(item for item in displays if item['Class']=='rviz_default_plugins/MarkerArray')
        self.assertEqual(markers['Topic']['Value'], '/fp_debug/markers')
        self.assertNotIn('Marker Topic',markers)
        self.assertEqual(config['Visualization Manager']['Global Options']['Fixed Frame'],'fp_world')
        self.assertFalse(any(item['Class']=='rviz_default_plugins/Map' for item in displays))

    def test_launcher_keeps_foxy_and_domain_from_caller(self):
        with tempfile.TemporaryDirectory() as folder:
            fake=Path(folder)/'rviz2'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$ROS_DISTRO" "$ROS_DOMAIN_ID" "$@"\n')
            fake.chmod(0o755)
            env=os.environ.copy()
            env.update(PATH=folder+':'+env['PATH'],ROS_DISTRO='foxy',ROS_DOMAIN_ID='23')
            result=subprocess.run(['bash',str(PACKAGE/'rviz_fp.sh')],env=env,
                                  stdout=subprocess.PIPE,text=True,check=True)
            lines=result.stdout.splitlines()
            self.assertEqual(lines[:2],['foxy','23'])
            self.assertEqual(lines[2],'-d')
            self.assertTrue(Path(lines[3]).is_file())


if __name__=='__main__': unittest.main()
