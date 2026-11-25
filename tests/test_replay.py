import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from urban_drone.replay import Recording,PlaybackClock,prepare_world,GazeboViewer


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'recording';self.source.mkdir()
        (self.source/'estimated_trajectory.csv').write_text('elapsed,x,y,z,strategy,phase\n'
                   '0,0,0,0,TRACK,WAIT\n1,1,2,3,PASS,FLY\n2,2,4,0,TRACK,DONE\n')
        (self.source/'global_smooth.csv').write_text('# x,y,z\n0,0,3\n2,4,3\n')
        (self.source/'urban_delivery.sdf').write_text('<sdf version="1.9"><world name="urban_delivery">'
             '<plugin name="gz::sim::systems::Physics" filename="physics"/>'
             '<plugin name="gz::sim::systems::UserCommands" filename="commands"/>'
             '<plugin name="gz::sim::systems::Sensors" filename="sensors"/>'
             '<model name="wall"><static>true</static></model>'
             '<include><uri>model://urban_x500</uri><name>urban_x500_0</name><pose>0 0 0 0 0 0</pose></include>'
             '</world></sdf>')
        self.recording=Recording.load(self.source)

    def test_interpolation_and_status_use_original_mission_clock(self):
        t,xyz,strategy,phase=self.recording.sample(.5)
        np.testing.assert_allclose(xyz,[.5,1,1.5])
        self.assertEqual((strategy,phase),('TRACK','WAIT'))
        self.assertEqual(self.recording.sample(1.1)[2:],('PASS','FLY'))
        self.assertEqual(self.recording.sample(99)[0],2.)

    def test_rewind_removes_future_blue_trace(self):
        self.assertGreater(len(self.recording.flown(2)),len(self.recording.flown(.5)))
        early=self.recording.flown(.5)
        self.assertTrue(np.all(early[:,0]<=.5))
        np.testing.assert_allclose(early[-1],[.5,1,1.5])

    def test_play_pause_speed_seek_and_end(self):
        clock=PlaybackClock(0,10,now=0)
        self.assertEqual(clock.update(1),0)
        clock.toggle(1);self.assertEqual(clock.update(3),2)
        clock.set_rate(2,3);self.assertEqual(clock.update(4),4)
        clock.toggle(4);self.assertEqual(clock.update(6),4)
        clock.seek(9,6);clock.toggle(6);self.assertEqual(clock.update(8),10)
        self.assertFalse(clock.playing)
        clock.toggle(8);self.assertEqual(clock.position,0)
        for rate in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):clock.set_rate(rate,8)

    def test_world_copy_freezes_only_viewer_vehicle_and_preserves_original(self):
        hashes={p.name:hashlib.sha256(p.read_bytes()).digest() for p in self.source.iterdir()}
        world=prepare_world(self.recording,Path(self.temp.name)/'viewer','replay_test')
        root=ET.parse(world).getroot().find('world')
        self.assertEqual(root.get('name'),'replay_test')
        self.assertEqual(root.find('include/static').text,'true')
        self.assertEqual(root.find('include/name').text,'urban_x500_0')
        self.assertIsNotNone(root.find("model[@name='wall']"))
        self.assertFalse(any('Sensors' in p.get('name') for p in root.findall('plugin')))
        self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).digest() for p in self.source.iterdir()})
        with self.assertRaises(FileExistsError):prepare_world(self.recording,world.parent,'again')

    def test_invalid_recording_is_rejected(self):
        path=self.source/'estimated_trajectory.csv'
        for body in ['0,0,0,nan,TRACK,WAIT\n1,1,0,0,TRACK,DONE\n',
                     '2,0,0,0,TRACK,WAIT\n1,1,0,0,TRACK,DONE\n']:
            path.write_text('elapsed,x,y,z,strategy,phase\n'+body)
            with self.assertRaises(ValueError):Recording.load(self.source)

    def test_transport_updates_only_isolated_viewer_and_draws_no_future(self):
        from gz.msgs10.pose_pb2 import Pose
        calls=[];markers=threading.Event();moved=threading.Event()
        class FakeNode:
            def advertise(self,*args):return SimpleNamespace(publish=lambda m:None)
            def request(self,service,message,*args):
                if isinstance(message,Pose):
                    calls.append((service,message.name,message.position.x,message.position.z))
                    if message.position.x==.5:moved.set()
                else:
                    calls.append((service,message.SerializeToString()))
                    markers.set()
                return True,SimpleNamespace(data=True)
        viewer=GazeboViewer(self.recording,'isolated_world',SimpleNamespace(poll=lambda:None))
        with patch('gz.transport13.Node',FakeNode):
            viewer.start()
            try:
                self.assertTrue(viewer.ready.wait(2))
                self.assertTrue(markers.wait(2),viewer.error)
                viewer.frame(.5);self.assertTrue(moved.wait(2))
            finally:viewer.close()
        self.assertIsNone(viewer.error)
        poses=[r for r in calls if r[0]!='/marker_array']
        self.assertTrue(poses)
        self.assertTrue(all(r[0]=='/world/isolated_world/set_pose' and r[1]=='urban_x500_0' for r in poses))
        self.assertIn(('/world/isolated_world/set_pose','urban_x500_0',.5,1.5),poses)


if __name__=='__main__':unittest.main()
