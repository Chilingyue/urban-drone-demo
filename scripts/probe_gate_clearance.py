#!/usr/bin/env python3
"""Offline four-sided aperture probe; synthetic rays, no PX4/Gazebo flight."""

import argparse
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from urban_drone.clearance import VEHICLE_RADIUS
from urban_drone.mission import Mission
from urban_drone.perception import RangeMap,Scan
from urban_drone.planning import PriorMap,sample_path
from urban_drone.safety import check_motion,braking_speed,PositionBrake
from urban_drone.trajectory import approach_speed,VelocityGuidance


def gate_boxes(left):
    right=left-4.5
    # Finite 1.5 m thick frame, 4.5 x 4 m opening at z=10.5..14.5.
    return [(np.array(lo,float),np.array(hi,float)) for lo,hi in (
        ([0,left,0],[1.5,8,18]),([0,-8,0],[1.5,right,18]),
        ([0,right,14.5],[1.5,left,18]),([0,right,0],[1.5,left,10.5]))]


def gate_scan(boxes,position,received):
    az=np.linspace(-np.pi,np.pi,181);el=np.linspace(-np.pi/2,np.pi/2,37)
    a,e=np.meshgrid(az,el)
    dirs=np.stack((np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)),axis=-1).reshape(-1,3)
    origin=np.asarray(position)+[0,0,.3]
    ranges=np.full(len(dirs),np.inf)
    parallel=abs(dirs)<1e-10;divisor=np.where(parallel,1.,dirs)
    for lo,hi in boxes:
        t1,t2=(lo-origin)/divisor,(hi-origin)/divisor
        near=np.where(parallel,-np.inf,np.minimum(t1,t2)).max(axis=1)
        far=np.where(parallel,np.inf,np.maximum(t1,t2)).min(axis=1)
        outside=np.any(parallel & ((origin<lo)|(origin>hi)),axis=1)
        hit=(far>=np.maximum(near,0)) & (far>0) & ~outside
        ranges=np.minimum(ranges,np.where(hit,np.where(near>0,near,far),np.inf))
    return Scan(ranges.reshape(a.shape),az[0],az[1]-az[0],el[0],el[1]-el[0],
                12.,received,np.asarray(position).copy(),np.array([0.,0.,0.,1.]),0.)


def surface_points(boxes,left):
    faces=[]
    for lo,hi in boxes:
        for fixed in range(3):
            moving=[i for i in range(3) if i!=fixed]
            axes=[np.linspace(lo[i],hi[i],int(np.ceil((hi[i]-lo[i])/.1))+1) for i in moving]
            aa,bb=np.meshgrid(*axes)
            for value in (lo[fixed],hi[fixed]):
                p=np.empty((aa.size,3));p[:,fixed]=value
                p[:,moving[0]]=aa.ravel();p[:,moving[1]]=bb.ravel();faces.append(p)
    # Exact cross-section samples make the two distance definitions explicit.
    faces.append(np.array([[.75,left,13],[.75,left-4.5,13],[.75,0,14.5],[.75,0,10.5]]))
    return np.vstack(faces)


def setup(left,integrated=False,position=None,received=None):
    prior=PriorMap(dict(bounds=[-20,30,-20,20],grid_resolution=.5,vehicle_radius=.35,
        safety_margin=.15,tracking_margin=.7,known_buildings=[],cruise_height=13.,
        max_height=25.,start=[-2,0],goal=[20,0]))
    observation=RangeMap(prior.bounds,window=[24,24,30]);boxes=gate_boxes(left)
    p=np.array([.75,0,13.]) if position is None else np.asarray(position)
    now=time.monotonic() if received is None else received
    observation.recenter(p)
    if integrated:
        for i in range(3):observation.integrate(gate_scan(boxes,p,now-(2-i)*.2))
    else:
        observation.points=surface_points(boxes,left);observation.tree=cKDTree(observation.points)
        observation.last_scan=gate_scan(boxes,p,now);observation.last_update=now;observation.updates=3
    return observation,prior,boxes


def continuous_pass(left,integrated):
    start=time.monotonic();position=np.array([-2.,0.,13.])
    observation,prior,boxes=setup(left,integrated,position,start)
    path=sample_path([position,[6,0,13]])
    events=[];aborts=[]
    fake=SimpleNamespace(complete=False,position=position,last_control=start-.05,
        phase='FLY',started=start,args=SimpleNamespace(timeout=1000),prior=prior,
        prior_data=prior.prior,state=SimpleNamespace(armed=True,mode='OFFBOARD'),pose_time=start,
        map=observation,sensor_hold_since=None,lock=threading.RLock(),active_path=path,path_index=0,
        profile_path=None,profile_limits=None,strategy='PASS',rejoin=path[-1],planning_hold=True,
        velocity=np.zeros(3),guidance=VelocityGuidance(),maneuver_phase='APPROACH',execution_state=None,
        global_route=path.copy(),initial_route=path.copy(),stopper=PositionBrake(),
        planning_wait_speed=None,recovery_reverse_authorized=False,
        escape_override_active=False,hold_target=None,hold_since=None,
        event=lambda name,**kw:events.append(dict(event=name,**kw)),setpoint=lambda p:None,
        command_file=StringIO(),progress_anchor=position.copy(),last_progress=start,
        abort=lambda reason:aborts.append(reason),arrival_since=None)
    held=0;crossed=False;trace=[];dt=.05
    for step in range(300):
        now=start+(step+1)*dt
        fake.pose_time=now
        # Match runtime's 5 Hz scans and 20 Hz control, rather than refreshing
        # sensor evidence on every control tick.
        with patch('urban_drone.mission.time.monotonic',return_value=now):
            if step%4==0:
                scan=gate_scan(boxes,fake.position,now)
                if integrated:
                    fake.map.recenter(fake.position);fake.map.integrate(scan)
                else:
                    fake.map.last_scan=scan;fake.map.last_update=now;fake.map.updates+=1
            Mission.control(fake)
        held+=int(fake.hold_target is not None)
        trace.append([float((step+1)*dt),*map(float,fake.position),int(fake.hold_target is not None)])
        fake.velocity=fake.target-fake.position
        fake.position=fake.position+fake.velocity*dt
        if fake.position[0]>1.5:crossed=True
        if fake.position[0]>=3.5 or aborts:break
    stages=list(dict.fromkeys(e['stage'] for e in events if e['event']=='execution_state'))
    return dict(passed=bool(crossed and fake.position[0]>=3.5 and held==0 and not aborts),
        map_mode='real RangeMap.integrate, 0.30 m voxels' if integrated else 'surface samples at 0.10 m',
        forward_m=float(fake.position[0]+2.),final_position=fake.position.tolist(),holds=held,
        virtual_seconds=(step+1)*dt,steps=step+1,stages=stages,aborts=aborts,
        planning_hold_flag=True,plant='ideal velocity = position lead / 1 s; no PX4 dynamics',
        state_events=[e for e in events if e['event'] in ('execution_state','execution_hold')],trace=trace)


def probe(left,label):
    observation,prior,_=setup(left);p=np.array([.75,0.,13.])
    cap=float(braking_speed(p[None],[[0,1,0]],observation)[0])
    drift=cap+.02
    cases=[('straight_through',[.4,0,0],[1,0,0],True),
           ('correct_right_while_forward',[.4,-.2,0],[1,0,0],True),
           ('left_drift_while_aiming_forward',[.4,0,0],[0,drift,0],False),
           ('correct_right_against_left_drift',[.2,-.2,0],[0,drift,0],True),
           ('cross_right_wall',[0,-(4.5-left-.3),0],[0,0,0],False),
           ('cross_lintel',[0,0,1.2],[0,0,0],False),
           ('cross_sill',[0,0,-2.2],[0,0,0],False)]
    results=[]
    for name,offset,velocity,expected in cases:
        decision=check_motion(p,p+offset,velocity,observation,prior)
        results.append(dict(case=name,expected_allowed=expected,passed=decision.allowed==expected,
                            velocity_m_s=velocity,**asdict(decision)))
    forward_cap=float(approach_speed(p[None],[[1,0,0]],observation)[0])
    controls=[continuous_pass(left,False),continuous_pass(left,True)]
    return dict(distance_definition=label,center_to_left_m=left,sphere_shell_to_left_m=left-VEHICLE_RADIUS,
        center_to_right_m=4.5-left,center_to_lintel_m=1.5,center_to_sill_m=2.5,
        forward_speed_cap_m_s=forward_cap,leftward_speed_cap_m_s=cap,
        motion_cases=results,continuous_pass=controls,
        passed=bool(abs(forward_cap-1.8)<1e-8 and all(r['passed'] for r in results) and all(c['passed'] for c in controls)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/gate_left_1p1_probe.json')
    args=parser.parse_args()
    results=[probe(1.1,'reference_point_1.1_m'),probe(1.1+VEHICLE_RADIUS,'sphere_shell_1.1_m')]
    report=dict(kind='offline finite four-sided aperture probe',flight_launched=False,
        geometry=dict(opening_width_m=4.5,opening_height_m=4.,wall_depth_m=1.5),
        sensing='analytic box ray intersections; real visibility and optional real voxel integration',
        results=results,passed=all(r['passed'] for r in results))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for result in results:
        print(result['distance_definition'],'PASS' if result['passed'] else 'FAIL')
        for case in result['motion_cases']:
            print(' ',case['case'],case['allowed'],case['reason'],'PASS' if case['passed'] else 'FAIL')
        for run in result['continuous_pass']:
            print(' ',run['map_mode'],'x=',round(run['final_position'][0],3),
                  'holds=',run['holds'],'stages=',run['stages'],'PASS' if run['passed'] else 'FAIL')
    print('Report:',args.output)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
