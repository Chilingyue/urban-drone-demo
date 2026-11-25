#!/usr/bin/env python3
"""Offline 1.1 m wall probe: synthetic scan + actual control/planning code.

No ROS node, PX4, Gazebo or real flight. The multi-step check uses an ideal
one-second position response, not a simulated flight-controller plant.
Run with the project environment sourced. Distances use the current sphere.
"""

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

from urban_drone.clearance import VEHICLE_RADIUS,SEARCH_CLEARANCE
from urban_drone.local import forward_connector
from urban_drone.mission import Mission
from urban_drone.perception import RangeMap,Scan
from urban_drone.planning import PriorMap,sample_path
from urban_drone.safety import check_motion,braking_speed,PositionBrake
from urban_drone.trajectory import approach_speed,VelocityGuidance


def wall_scan(wall_y,position,received):
    az=np.linspace(-np.pi,np.pi,181);el=np.linspace(-np.pi/2,np.pi/2,37)
    a,e=np.meshgrid(az,el);dy=np.cos(e)*np.sin(a)
    ranges=np.full(dy.shape,np.inf)
    np.divide(wall_y-position[1],dy,out=ranges,where=dy>1e-8)
    return Scan(ranges,az[0],az[1]-az[0],el[0],el[1]-el[0],12.,received,
                np.asarray(position).copy(),np.array([0.,0.,0.,1.]),0.)


def geometry(distance):
    prior=PriorMap(dict(bounds=[-20,30,-20,20],grid_resolution=.5,
        vehicle_radius=.35,safety_margin=.15,tracking_margin=.7,known_buildings=[],
        cruise_height=3.,max_height=10.,start=[0,0],goal=[20,0]))
    observation=RangeMap(prior.bounds,window=[24,24,12])
    x,z=np.meshgrid(np.arange(-4,18.01,.1),np.arange(0,6.01,.1))
    observation.points=np.column_stack((x.ravel(),np.full(x.size,distance),z.ravel()))
    observation.tree=cKDTree(observation.points)
    observation.last_update=time.monotonic();observation.updates=3
    observation.last_scan=wall_scan(distance,[0,0,3],observation.last_update)
    return observation,prior


def control_progress(distance):
    observation,prior=geometry(distance);start=time.monotonic()
    position=np.array([0.,0.,3.]);path=sample_path([position,[10,0,3]])
    events=[]
    fake=SimpleNamespace(complete=False,position=position,last_control=start-.05,
        phase='FLY',started=start,args=SimpleNamespace(timeout=1000),prior=prior,
        prior_data=prior.prior,state=SimpleNamespace(armed=True,mode='OFFBOARD'),
        pose_time=start,map=observation,sensor_hold_since=None,lock=threading.RLock(),
        active_path=path,path_index=0,profile_path=None,profile_limits=None,strategy='PASS',
        rejoin=path[-1],planning_hold=True,velocity=np.zeros(3),guidance=VelocityGuidance(),
        global_route=path.copy(),initial_route=path.copy(),stopper=PositionBrake(),
        planning_wait_speed=None,recovery_reverse_authorized=False,
        maneuver_phase='APPROACH',execution_state=None,escape_override_active=False,
        hold_target=None,hold_since=None,event=lambda name,**kw:events.append(dict(event=name,**kw)),
        setpoint=lambda target:None,command_file=StringIO(),progress_anchor=position.copy(),
        last_progress=start,abort=lambda reason:(_ for _ in ()).throw(AssertionError(reason)),
        arrival_since=None)
    held=0;dt=.05
    for step in range(300):
        now=start+(step+1)*dt
        fake.pose_time=now;fake.map.last_update=now
        fake.map.last_scan=wall_scan(distance,fake.position,now)
        # Time is synthetic only inside this single-threaded offline call.
        with patch('urban_drone.mission.time.monotonic',return_value=now):
            Mission.control(fake)
        held+=int(fake.hold_target is not None)
        fake.velocity=fake.target-fake.position
        fake.position=fake.position+fake.velocity*dt
        if fake.position[0]>=3.:break
    assert held==0, f'{distance}: unexpected holds: {held}'
    assert fake.position[0]>=3., f'{distance}: no finite forward progress'
    return dict(steps=step+1,virtual_seconds=(step+1)*dt,forward_m=float(fake.position[0]),
                holds=held,planning_hold_flag=True,
                model='ideal velocity = position lead / 1 s; no PX4 dynamics')


def probe(distance,label):
    observation,prior=geometry(distance);p=np.array([0.,0.,3.])
    cap=float(braking_speed(p[None],[[0,1,0]],observation)[0])
    cases=[('along_wall',[.4,0,0],[1.,0,0],True),
           ('away_from_wall',[0,-.4,0],[0,-1.,0],True),
           ('toward_wall_slow',[0,.3,0],[0,.4,0],True),
           ('toward_wall_too_fast',[0,.3,0],[0,cap+.05,0],False),
           ('sideways_drift_while_aiming_forward',[.4,0,0],[1.,cap+.05,0],False)]
    results=[]
    for name,offset,velocity,expected in cases:
        decision=check_motion(p,p+offset,velocity,observation,prior)
        assert decision.allowed==expected, f'{label}/{name}: {decision}'
        results.append(dict(case=name,velocity_m_s=velocity,**asdict(decision)))
    speed_caps=approach_speed(np.repeat(p[None],3,axis=0),[[1,0,0],[0,-1,0],[0,1,0]],observation)
    np.testing.assert_allclose(speed_caps[:2],[1.8,1.8])
    assert float(speed_caps[2])<1.8
    measured=float(observation.distances(p[None])[0])
    assert abs(measured-distance)<1e-8
    anchor,connector=forward_connector(p,[1,0],observation,prior)
    assert anchor is not None
    assert observation.distances(anchor[None])[0]>SEARCH_CLEARANCE
    if distance<SEARCH_CLEARANCE:
        assert connector is not None
        assert connector[-1,1]<p[1], 'connector must move away from wall'
    else:
        assert connector is None
    return dict(distance_definition=label,center_distance_m=distance,
        sphere_shell_clearance_m=distance-VEHICLE_RADIUS,
        speed_caps_m_s=dict(along=float(speed_caps[0]),away=float(speed_caps[1]),toward=float(speed_caps[2])),
        motion_cases=results,new_planning_connector_used=connector is not None,
        planning_anchor_clearance_m=float(observation.distances(anchor[None])[0]),
        control_progress=control_progress(distance))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/wall_1p1_probe.json')
    args=parser.parse_args()
    results=[probe(1.1,'reference_point_1.1_m'),probe(1.1+VEHICLE_RADIUS,'sphere_shell_1.1_m')]
    report=dict(kind='offline synthetic wall probe',flight_launched=False,
        sensing='analytic planar scan, real RangeMap.observed_free; occupied points sampled at 0.1 m, not live voxel integration',
        control='actual Mission.control with supplied pose/velocity and ideal tracking',
        results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for r in results:
        print(r['distance_definition'],json.dumps(r['speed_caps_m_s']),
              'progress',round(r['control_progress']['forward_m'],3),
              'holds',r['control_progress']['holds'])
    print('PASS:',args.output)


if __name__=='__main__':main()
