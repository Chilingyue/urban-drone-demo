"""Bounded camera initialization while the GUI loads its render target."""
import threading
import time

TARGET='urban_x500_0'
RELEASE_TARGET='__urban_camera_release__'


def configure_camera(mode,offset=None,timeout=8.,node=None):
    """Wait for the rendered target, then confirm mode, target and offset.

    Retry only during this bounded startup call. Gazebo clears target names
    when a model is not rendered yet. Stop after success or manual release;
    never run a background loop that would undo Esc.
    """
    from gz.msgs10.cameratrack_pb2 import CameraTrack
    if mode not in ('follow','view','free'):raise ValueError(f'Unknown camera mode: {mode}')
    if mode!='free' and (offset is None or len(offset)!=3):
        raise ValueError('A follow camera needs an (x, y, z) offset.')
    if node is None:
        from gz.transport13 import Node
        node=Node()
    requested=CameraTrack()
    requested.track_mode=CameraTrack.NONE if mode=='free' else CameraTrack.FOLLOW_LOOK_AT
    if mode=='free':
        # gz-gui8's NONE mode alone leaves the old rendering FollowTarget
        # attached. A deliberately absent target makes OnRender detach both
        # camera pointers; an empty name would preserve the old selection.
        requested.follow_target.name=requested.track_target.name=RELEASE_TARGET
    else:
        requested.follow_target.name=requested.track_target.name=TARGET
        requested.follow_offset.x,requested.follow_offset.y,requested.follow_offset.z=offset
        requested.follow_pgain=requested.track_pgain=1.
    sent=threading.Event();lock=threading.Lock()
    state=dict(seen_ours=False,manual_release=False,match_since=None,last_received=0.,
               retry_allowed=True,last_targets=None)
    def status(message):
        if not sent.is_set():return
        now=time.monotonic()
        actual=message.follow_offset
        correct_offset=mode=='free' or all(abs(a-b)<1e-4 for a,b in zip((actual.x,actual.y,actual.z),offset))
        targets=(message.follow_target.name,message.track_target.name)
        with lock:
            state['last_received']=now;state['last_targets']=targets
            if mode!='free' and state['seen_ours'] and message.track_mode!=requested.track_mode:
                state['manual_release']=True
            ours=message.track_mode==requested.track_mode and correct_offset
            if ours:state['seen_ours']=True
            match=ours and (mode=='free' or targets==(TARGET,TARGET))
            if match:
                if state['match_since'] is None:state['match_since']=now
                state['retry_allowed']=False
            else:
                state['match_since']=None
                # Retry a cleared startup target, not another user-selected
                # nonempty target. NONE before first acceptance is initial state.
                state['retry_allowed']=not any(targets)
    node.subscribe(CameraTrack,'/gui/currently_tracked',status)
    publisher=node.advertise('/gui/track',CameraTrack)
    deadline=time.monotonic()+timeout
    attempts=0
    try:
        while not publisher.has_connections():
            if time.monotonic()>=deadline:
                return dict(confirmed=False,mode=mode,reason='GUI camera subscriber not ready')
            time.sleep(.05)
        last_sent=-float('inf')
        while time.monotonic()<deadline:
            now=time.monotonic()
            with lock:current=state.copy()
            if current['manual_release']:
                return dict(confirmed=False,mode=mode,manual_release=True,attempts=attempts,
                            reason='camera released or mode changed during initialization; not relocking')
            if (current['match_since'] is not None and now-current['match_since']>=.15 and
                    now-current['last_received']<.5):
                return dict(confirmed=True,mode=mode,offset=offset if mode!='free' else None,
                            attempts=attempts,reason='render target and GUI tracking state confirmed')
            if attempts==0 or (current['retry_allowed'] and now-last_sent>=.3):
                sent.set()
                # transport13 Python publish returns None on success.
                if publisher.publish(requested) is False:
                    return dict(confirmed=False,mode=mode,attempts=attempts,reason='camera message publish failed')
                last_sent=now;attempts+=1
            time.sleep(.02)
        return dict(confirmed=False,mode=mode,attempts=attempts,last_targets=state['last_targets'],
                    reason='GUI render target or tracking acknowledgement did not become ready')
    finally:
        node.unsubscribe('/gui/currently_tracked')
