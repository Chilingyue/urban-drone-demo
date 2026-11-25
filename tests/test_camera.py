import unittest
from types import SimpleNamespace
from gz.msgs10.cameratrack_pb2 import CameraTrack
from urban_drone.camera import configure_camera,TARGET,RELEASE_TARGET

OFFSETS={'follow':(-2.,0.,2.),'view':(-5.,-5.,8.)}


class FakeGui:
    def __init__(self,connected=True,wrong_target=False,clear_first=0,escape=False):
        self.connected=connected;self.wrong_target=wrong_target;self.sent=[];self.callback=None
        self.clear_first=clear_first;self.escape=escape
    def subscribe(self,kind,topic,callback):self.callback=callback
    def unsubscribe(self,topic):self.callback=None
    def advertise(self,topic,kind):
        return SimpleNamespace(has_connections=lambda:self.connected,publish=self.publish)
    def publish(self,message):
        copy=CameraTrack();copy.CopyFrom(message);self.sent.append(copy)
        if self.wrong_target:copy.follow_target.name='another_model'
        if len(self.sent)<=self.clear_first:
            copy.follow_target.Clear();copy.track_target.Clear()
        if self.callback:self.callback(copy)
        if self.escape and self.callback:
            self.callback(CameraTrack(track_mode=CameraTrack.NONE))
        # Native Python publisher returns None, not True.


class CameraTests(unittest.TestCase):
    def test_gui_cleared_target_is_rebound_during_startup(self):
        gui=FakeGui(clear_first=2)
        result=configure_camera('follow',OFFSETS['follow'],timeout=2,node=gui)
        self.assertTrue(result['confirmed'])
        self.assertEqual(result['attempts'],3)
        self.assertIsNone(gui.callback)

    def test_escape_during_startup_stops_rebinding(self):
        gui=FakeGui(clear_first=1,escape=True)
        result=configure_camera('follow',OFFSETS['follow'],node=gui)
        self.assertFalse(result['confirmed']);self.assertTrue(result['manual_release'])
        self.assertEqual(len(gui.sent),1)

    def test_follow_and_view_use_requested_target_and_offsets(self):
        for mode in ('follow','view'):
            gui=FakeGui();result=configure_camera(mode,OFFSETS[mode],node=gui)
            self.assertTrue(result['confirmed'])
            msg=gui.sent[0]
            self.assertEqual(msg.follow_target.name,TARGET)
            self.assertEqual(msg.track_mode,CameraTrack.FOLLOW_LOOK_AT)
            self.assertEqual((msg.follow_offset.x,msg.follow_offset.y,msg.follow_offset.z),OFFSETS[mode])

    def test_no_subscriber_does_not_publish_or_claim_success(self):
        gui=FakeGui(connected=False)
        self.assertFalse(configure_camera('follow',OFFSETS['follow'],timeout=0,node=gui)['confirmed'])
        self.assertFalse(gui.sent)

    def test_wrong_target_does_not_count_as_confirmation(self):
        self.assertFalse(configure_camera('follow',OFFSETS['follow'],timeout=.01,node=FakeGui(wrong_target=True))['confirmed'])

    def test_free_clears_tracking_and_follow_is_not_repeated(self):
        gui=FakeGui();result=configure_camera('free',node=gui)
        self.assertTrue(result['confirmed']);self.assertEqual(gui.sent[0].track_mode,CameraTrack.NONE)
        self.assertEqual(gui.sent[0].follow_target.name,RELEASE_TARGET)
        gui=FakeGui();configure_camera('follow',OFFSETS['follow'],node=gui)
        self.assertIsNone(gui.callback)
        self.assertEqual(len(gui.sent),1)


if __name__=='__main__':unittest.main()
