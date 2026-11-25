"""Offline Gazebo trajectory viewer. Never launches ROS, PX4 or a mission."""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'urban_x500_0'


@dataclass
class Recording:
    directory: Path
    times: np.ndarray
    positions: np.ndarray
    strategies: list
    phases: list
    plan: np.ndarray

    @classmethod
    def load(cls,directory):
        directory=Path(directory).expanduser().resolve()
        with (directory/'estimated_trajectory.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        if len(rows)<2:raise ValueError('The recording needs at least two estimated pose samples.')
        values=np.array([[float(r[k]) for k in ('elapsed','x','y','z')] for r in rows])
        if not np.isfinite(values).all() or np.any(np.diff(values[:,0])<0):
            raise ValueError('Trajectory must have finite values and ordered timestamps.')
        # Equal timestamps use the last recorded sample, without reordering.
        keep=np.r_[np.diff(values[:,0])>0,True]
        values=values[keep];rows=[r for r,k in zip(rows,keep) if k]
        if len(values)<2:raise ValueError('The recording has no positive time interval.')
        plan=np.atleast_2d(np.loadtxt(directory/'global_smooth.csv',delimiter=','))
        if plan.shape[1]!=3 or not np.isfinite(plan).all():raise ValueError('Invalid recorded global plan.')
        if not (directory/'urban_delivery.sdf').is_file():raise ValueError('Recorded world is missing.')
        return cls(directory,values[:,0],values[:,1:],
                   [r['strategy'] for r in rows],[r['phase'] for r in rows],plan)

    def sample(self,seconds):
        t=float(np.clip(seconds,self.times[0],self.times[-1]))
        xyz=np.array([np.interp(t,self.times,self.positions[:,i]) for i in range(3)])
        index=max(0,int(np.searchsorted(self.times,t,side='right')-1))
        return t,xyz,self.strategies[index],self.phases[index]

    def flown(self,seconds):
        t,xyz,_,_=self.sample(seconds)
        count=int(np.searchsorted(self.times,t,side='right'))
        # Rebuild from time on every seek; future points never survive rewind.
        stride=max(1,math.ceil(count/2500))
        return np.vstack((self.positions[:count:stride],xyz))


class PlaybackClock:
    def __init__(self,start,end,rate=1.,now=None):
        self.start,self.end=float(start),float(end)
        if not math.isfinite(rate) or rate<=0:raise ValueError('Playback speed must be positive and finite.')
        self.rate=float(rate);self.position=self.start;self.playing=False
        self.last=time.monotonic() if now is None else now

    def update(self,now):
        if self.playing:
            self.position=min(self.end,self.position+max(0,now-self.last)*self.rate)
            if self.position>=self.end:self.playing=False
        self.last=now
        return self.position

    def seek(self,seconds,now):
        self.position=float(np.clip(seconds,self.start,self.end));self.last=now

    def toggle(self,now):
        self.update(now)
        if not self.playing and self.position>=self.end:self.position=self.start
        self.playing=not self.playing

    def set_rate(self,rate,now):
        if not math.isfinite(rate) or rate<=0:raise ValueError('Playback speed must be positive and finite.')
        self.update(now);self.rate=float(rate)


def prepare_world(recording,output,world_name):
    """Copy the recorded world; freeze the display model and disable sensors."""
    tree=ET.parse(recording.directory/'urban_delivery.sdf')
    world=tree.getroot().find('world')
    if world is None:raise ValueError('Recorded SDF has no world.')
    world.set('name',world_name)
    keep={'Physics','UserCommands','SceneBroadcaster'}
    for plugin in list(world.findall('plugin')):
        if plugin.get('name','').split('::')[-1] not in keep:world.remove(plugin)
    include=next((i for i in world.findall('include') if i.findtext('name')==MODEL),None)
    if include is None:raise ValueError('Recorded world does not contain the expected vehicle.')
    static=include.find('static')
    if static is None:static=ET.SubElement(include,'static')
    static.text='true'
    xyz=recording.positions[0]
    include.find('pose').text=f'{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0'
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    destination=output/'replay.sdf'
    ET.indent(tree);tree.write(destination,encoding='unicode',xml_declaration=True)
    return destination


class GazeboViewer:
    """Latest-frame worker; transport waits cannot block the playback panel."""
    def __init__(self,recording,world,server,markers=True):
        self.recording,self.world,self.server=recording,world,server
        self.markers=markers;self.lock=threading.Lock();self.stop=threading.Event()
        self.ready=threading.Event();self.error=None;self.seconds=float(recording.times[0])
        self.camera_command=None
        self.worker=threading.Thread(target=self.work,daemon=True)

    def start(self):self.worker.start()

    def frame(self,seconds):
        with self.lock:self.seconds=seconds

    def camera(self,mode):
        with self.lock:self.camera_command=mode

    def work(self):
        try:
            from gz.transport13 import Node
            from gz.msgs10.pose_pb2 import Pose
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.cameratrack_pb2 import CameraTrack
            from gz.msgs10.marker_v_pb2 import Marker_V
            from gz.msgs10.marker_pb2 import Marker
            from .path_visuals import path_marker,PLAN_RED,FLIGHT_BLUE
            node=Node();camera=node.advertise('/gui/track',CameraTrack)
            pose=Pose();pose.name=MODEL;pose.orientation.w=1.
            begun=time.monotonic();last_frame=None;last_markers=0.;failures=0
            while not self.stop.is_set():
                if self.server.poll() is not None:raise RuntimeError('Gazebo server exited; inspect gazebo.log.')
                with self.lock:
                    seconds=self.seconds;mode=self.camera_command;self.camera_command=None
                if mode is not None:
                    track=CameraTrack()
                    track.track_mode=CameraTrack.NONE if mode=='free' else CameraTrack.FOLLOW_LOOK_AT
                    track.follow_target.name=track.track_target.name=MODEL
                    if mode=='free':
                        from .camera import RELEASE_TARGET
                        track.follow_target.name=track.track_target.name=RELEASE_TARGET
                    offset=(-10,-12,8) if mode=='follow' else (-60,-80,100)
                    track.follow_offset.x,track.follow_offset.y,track.follow_offset.z=offset
                    track.follow_pgain=track.track_pgain=1.
                    camera.publish(track)
                t,xyz,strategy,phase=self.recording.sample(seconds)
                if t!=last_frame or not self.ready.is_set():
                    # Estimated CSV and global plan use the original MAVROS
                    # frame. Model origin is .24 m below its body centre.
                    pose.position.x,pose.position.y,pose.position.z=map(float,xyz)
                    ok,response=node.request(f'/world/{self.world}/set_pose',pose,Pose,Boolean,200)
                    if ok and response.data:
                        self.ready.set();last_frame=t;failures=0
                    else:
                        failures+=1
                        if (self.ready.is_set() and failures>=15) or time.monotonic()-begun>45:
                            raise RuntimeError('Replay pose service unavailable; inspect gazebo.log.')
                now=time.monotonic()
                if self.markers and self.ready.is_set() and now-last_markers>.35:
                    batch=Marker_V()
                    batch.marker.append(path_marker(1,self.recording.plan,PLAN_RED))
                    batch.marker.append(path_marker(2,self.recording.flown(t),FLIGHT_BLUE))
                    label=Marker();label.ns='urban_replay';label.id=1
                    label.action=Marker.ADD_MODIFY;label.type=Marker.TEXT;label.visibility=Marker.GUI
                    label.pose.orientation.w=1.
                    label.pose.position.x=float(xyz[0]);label.pose.position.y=float(xyz[1]);label.pose.position.z=float(xyz[2]+1.5)
                    label.scale.z=.6;label.material.emissive.r=label.material.emissive.g=label.material.emissive.b=1.
                    label.material.emissive.a=1.;label.material.diffuse.CopyFrom(label.material.emissive)
                    label.text=f'REPLAY {t:.1f}s | {phase} | {strategy}'
                    batch.marker.append(label)
                    node.request('/marker_array',batch,Marker_V,Boolean,100)
                    last_markers=now
                self.stop.wait(.04)
        except Exception as error:
            self.error=str(error);self.ready.set()

    def close(self):
        self.stop.set();self.worker.join(timeout=3)


def configure_panel_font(root):
    """Use an installed CJK family explicitly; never rely on X11 fallback."""
    from tkinter import font,ttk
    available={name.casefold():name for name in font.families(root)}
    preferred=('Microsoft YaHei','Noto Sans CJK SC','WenQuanYi Micro Hei',
               'WenQuanYi Zen Hei','Source Han Sans SC','SimHei','SimSun')
    family=next((available[name.casefold()] for name in preferred if name.casefold() in available),None)
    if family:
        for name in ('TkDefaultFont','TkTextFont','TkMenuFont','TkHeadingFont','TkCaptionFont','TkTooltipFont'):
            font.nametofont(name,root=root).configure(family=family,size=10)
        ttk.Style(root).configure('.',font=(family,10))
        root.option_add('*TCombobox*Listbox.font',(family,10))
    return family


def panel(recording,clock,viewer,gui,camera_mode):
    import tkinter as tk
    from tkinter import ttk
    root=tk.Tk();root.title('Trajectory replay - '+recording.directory.name)
    family=configure_panel_font(root)
    def text(chinese,english):return chinese if family else english
    root.geometry('760x260');root.minsize(720,260)
    status=tk.StringVar(value=text('正在加载 Gazebo…','Loading Gazebo...'));progress=tk.DoubleVar(value=clock.position)
    dragging=False;camera_sent=False
    ttk.Label(root,text=text('历史轨迹回放（不运行 PX4）','Recorded trajectory replay (PX4 is not running)')).pack(pady=(10,2))
    ttk.Label(root,textvariable=status).pack()
    slider=ttk.Scale(root,from_=clock.start,to=clock.end,variable=progress)
    slider.pack(fill='x',padx=18,pady=12)
    def begin(_):
        nonlocal dragging
        dragging=True
    def finish(_):
        nonlocal dragging
        clock.seek(progress.get(),time.monotonic());viewer.frame(clock.position);dragging=False
    slider.bind('<ButtonPress-1>',begin);slider.bind('<ButtonRelease-1>',finish)
    row=ttk.Frame(root);row.pack()
    play=ttk.Button(row,text=text('播放','Play'))
    def toggle():
        nonlocal camera_sent
        if not viewer.ready.is_set() or viewer.error:return
        if not camera_sent:
            viewer.camera(camera_mode);camera_sent=True
        clock.toggle(time.monotonic())
    play.configure(command=toggle);play.pack(side='left',padx=4)
    ttk.Button(row,text=text('从头','Restart'),command=lambda:clock.seek(clock.start,time.monotonic())).pack(side='left',padx=4)
    speed=ttk.Combobox(row,values=['0.25','0.5','1','2','4'],width=5,state='readonly')
    speed.set(str(clock.rate));speed.pack(side='left',padx=4)
    speed.bind('<<ComboboxSelected>>',lambda _:clock.set_rate(float(speed.get()),time.monotonic()))
    ttk.Label(row,text=text('倍速','x')).pack(side='left')
    def set_camera(mode):
        nonlocal camera_sent
        viewer.camera(mode);camera_sent=True
    for title,mode in [(text('跟随镜头','Follow'),'follow'),(text('俯视镜头','Overview'),'overview'),(text('自由镜头','Free view'),'free')]:
        ttk.Button(row,text=title,command=lambda mode=mode:set_camera(mode)).pack(side='left',padx=3)
    hint=text('先等 Gazebo 画面加载，再点播放。拖动进度可回看；Gazebo 中按 Esc 可自由观察。',
              'Wait for Gazebo, then click Play. Drag the slider to seek; press Esc in Gazebo for a free camera.')
    ttk.Label(root,text=hint,wraplength=700,justify='center').pack(pady=12)
    def update():
        if gui.poll() is not None:root.destroy();return
        if viewer.error:
            status.set(text('回放错误：','Replay error: ')+viewer.error);play.configure(state='disabled')
        elif not viewer.ready.is_set():
            status.set(text('正在加载 Gazebo…','Loading Gazebo...'));play.configure(state='disabled')
        else:
            play.configure(state='normal',text=text('暂停','Pause') if clock.playing else text('播放','Play'))
            if not dragging:
                clock.update(time.monotonic());progress.set(clock.position)
            else:clock.last=time.monotonic()
            t,_,strategy,phase=recording.sample(progress.get())
            status.set(f'{t:.1f} / {clock.end:.1f} s | {phase} | {strategy}')
            viewer.frame(progress.get())
        root.after(70,update)
    root.after(100,update)
    try:root.mainloop()
    finally:
        try:root.destroy()
        except tk.TclError:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path,help='Original run directory; read only')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--start',type=float,default=0.,help='Recorded mission seconds')
    parser.add_argument('--camera',choices=['follow','overview','free'],default='follow')
    parser.add_argument('--prepare-only',action='store_true',help='Validate recording and build replay world; launch nothing')
    args=parser.parse_args()
    try:
        recording=Recording.load(args.recording)
        clock=PlaybackClock(recording.times[0],recording.times[-1],args.speed)
        if not math.isfinite(args.start):raise ValueError('Start time must be finite.')
        clock.seek(args.start,time.monotonic())
        session=ROOT/'replays'/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')+'_'+recording.directory.name)
        world_name='urban_replay'
        world=prepare_world(recording,session,world_name)
        (session/'replay.json').write_text(json.dumps(dict(source=str(recording.directory),
                trajectory='estimated_trajectory.csv',recorded_seconds=float(recording.times[-1]),
                speed=args.speed,start=clock.position,flight_control=False),indent=2)+'\n')
        print(f'Recording: {recording.directory}\nReplay session: {session}\n'
              f'{len(recording.times)} samples, {clock.end:.1f} s. Original files remain read-only.',flush=True)
        if args.prepare_only:return
        # A unique transport partition keeps replay set_pose and GUI traffic
        # isolated from every active simulation. No flight processes launched.
        partition=f'urban_replay_{os.getpid()}'
        os.environ.update(GZ_PARTITION=partition,GZ_IP='127.0.0.1')
        manifest=recording.directory/'run_manifest.json'
        px4=Path(json.loads(manifest.read_text())['arguments']['px4']) if manifest.exists() else Path.home()/'PX4-Autopilot'
        models=recording.directory/'source/models'
        if not models.is_dir():models=ROOT/'models'
        env=os.environ.copy()
        env['GZ_SIM_RESOURCE_PATH']=os.pathsep.join([str(models),str(px4/'Tools/simulation/gz/models'),env.get('GZ_SIM_RESOURCE_PATH','')])
        env['GZ_SIM_SYSTEM_PLUGIN_PATH']=str(px4/'build/px4_sitl_default/src/modules/simulation/gz_plugins')+os.pathsep+env.get('GZ_SIM_SYSTEM_PLUGIN_PATH','')
        processes=[];logs=[];viewer=None
        def interrupted(*_):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        try:
            def launch(name,cmd):
                log=(session/f'{name}.log').open('w');logs.append(log)
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(proc);return proc
            server=launch('gazebo',['gz','sim','-r','-s','-v','2',str(world)])
            gui=launch('gui',['gz','sim','-g'])
            viewer=GazeboViewer(recording,world_name,server);viewer.frame(clock.position);viewer.start()
            print('Replay only: wait for Gazebo, then click Play in the control panel. Closing either window ends replay.',flush=True)
            panel(recording,clock,viewer,gui,args.camera)
        finally:
            if viewer is not None:viewer.close()
            for proc in reversed(processes):
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for proc in reversed(processes):
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait(timeout=3)
            for log in logs:log.close()
    except KeyboardInterrupt:
        print('Replay closed.')
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'Replay could not start: {error}\n')


if __name__=='__main__':main()
