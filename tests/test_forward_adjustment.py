"""Adjust future motion from the live pose, rather than chasing passed points."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.global_policy import GlobalReplanPolicy,eligible_candidate
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap,sample_path
from urban_drone.recovery import clearance_recovery
from urban_drone.safety import forward_remainder,forward_target,PositionBrake
from urban_drone.trajectory import VelocityGuidance

DATA=Path(__file__).parent/'data'


class ForwardAdjustmentTests(unittest.TestCase):
    def geometry(self,points=()):
        prior=PriorMap(dict(bounds=[-20,100,-30,30],grid_resolution=.5,vehicle_radius=.35,
            safety_margin=.15,tracking_margin=.7,known_buildings=[],cruise_height=13.,
            max_height=30.,start=[0,0],goal=[20,0]))
        obs=RangeMap(prior.bounds,window=[24,24,30])
        obs.points=np.asarray(points,float).reshape(-1,3)
        obs.tree=cKDTree(obs.points) if len(obs.points) else None
        obs.observed_free=lambda p,margin=.4:np.ones(len(p),bool)
        obs.snapshot=lambda **kw:obs
        obs.updates=3;obs.last_update=100.
        return obs,prior

    def test_captured_wall_reconnect_climbs_from_live_pose_instead_of_returning(self):
        path=np.loadtxt(DATA/'forward_join_wall.csv',delimiter=',',skiprows=1)
        live=path[0];self.assertLess(path[1,0],live[0])
        connected=forward_remainder(live,path,0,[1,0])
        np.testing.assert_allclose(connected[0],live)
        self.assertGreaterEqual(connected[1,0],live[0])
        self.assertGreater(connected[1,2]-live[2],.3)
        np.testing.assert_allclose(connected[-1],path[-1])

    def test_captured_pillar_reconnect_moves_forward_outward_and_stays_clear(self):
        path=np.loadtxt(DATA/'forward_join_pillar.csv',delimiter=',',skiprows=1)
        live=path[0];self.assertLess(path[1,0],live[0]-.1)
        connected=forward_remainder(live,path,0,[1,0])
        self.assertGreater(connected[1,0],live[0])
        self.assertLess(connected[1,1],live[1]-.3)
        theta,z=np.meshgrid(np.linspace(0,2*np.pi,361),np.arange(12.5,13.51,.1))
        points=np.column_stack((76.968244642+2.5*np.cos(theta.ravel()),
                                -7.912623281+2.5*np.sin(theta.ravel()),z.ravel()))
        obs,prior=self.geometry(points)
        self.assertTrue(eligible_candidate(connected,obs,prior))

    def test_remove_backward_component_keeps_outward_and_upward_correction(self):
        p=np.array([3.,0.,13.]);target=forward_target(p,[2.8,-.5,13.6],[1,0])
        np.testing.assert_allclose(target,[3.,-.5,13.6])

    def test_braking_with_inertia_does_not_command_return_to_old_anchor(self):
        brake=PositionBrake();p=np.array([0.,0.,13.]);v=1.5;history=[]
        for i in range(160):
            target=brake.update(p,[v,0,0],[1,0],i*.05)
            self.assertGreaterEqual(target[0],p[0]-1e-9)
            wanted=target[0]-p[0]
            v+=float(np.clip(wanted-v,-.8*.05,.8*.05));p[0]+=v*.05
            history.append(p[0])
        self.assertTrue(np.all(np.diff(history)>=-1e-9))
        self.assertLess(abs(v),.01)
        self.assertGreater(p[0],1.)

    def control_fixture(self):
        obs,prior=self.geometry();path=sample_path([[0,0,13],[20,0,13]])
        guidance=VelocityGuidance();guidance.velocity=np.array([1.5,0,0])
        return SimpleNamespace(complete=False,position=np.array([0.,0.,13.]),velocity=np.array([1.5,0,0]),
            phase='FLY',started=100.,last_control=100.,args=SimpleNamespace(timeout=1000),
            prior=prior,prior_data=prior.prior,state=SimpleNamespace(armed=True,mode='OFFBOARD'),
            pose_time=100.,map=obs,sensor_hold_since=None,lock=threading.RLock(),
            active_path=path,global_route=path.copy(),initial_route=path.copy(),global_index=0,path_index=0,
            profile_path=None,profile_limits=None,strategy='TRACK',rejoin=None,planning_hold=True,
            guidance=guidance,stopper=PositionBrake(),planning_wait_speed=None,recovery_reverse_authorized=False,
            maneuver_phase='TRACK',execution_state=None,escape_override_active=False,
            hold_target=None,hold_since=None,event=Mock(),setpoint=Mock(),command_file=StringIO(),
            progress_anchor=np.array([0.,0.,13.]),last_progress=100.,arrival_since=None,abort=Mock(),
            global_policy=GlobalReplanPolicy())

    def test_actual_control_slows_forward_then_stops_during_planning_wait(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture();positions=[];modes=[]
        for i in range(100):
            now=100.+(i+1)*.05;fake.pose_time=now;fake.map.last_update=now
            with patch('urban_drone.mission.time.monotonic',return_value=now):Mission.control(fake)
            self.assertGreaterEqual(fake.target[0],fake.position[0]-1e-9)
            positions.append(fake.position[0]);modes.append(fake.execution_state[2])
            fake.velocity=fake.target-fake.position;fake.position+=fake.velocity*.05
        self.assertIn('DECELERATE',modes);self.assertIn('BRAKE',modes)
        self.assertGreater(fake.position[0],.5)
        self.assertTrue(np.all(np.diff(positions)>=-1e-9))
        self.assertIsNotNone(fake.hold_target)

    def test_actual_control_ignores_cached_old_hold_point_after_overshoot(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture();fake.planning_hold=False
        fake.map.points=np.array([[1.,0.,13.]]);fake.map.tree=cKDTree(fake.map.points)
        fake.hold_target=np.array([-1.,0.,13.]);fake.stopper.anchor=fake.hold_target.copy()
        fake.pose_time=100.05;fake.map.last_update=100.05
        with patch('urban_drone.mission.time.monotonic',return_value=100.05):Mission.control(fake)
        self.assertIsNotNone(fake.hold_target)
        self.assertGreaterEqual(fake.target[0],fake.position[0])

    def test_actual_control_keeps_sideways_motion_without_chasing_behind_turn_point(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture();fake.strategy='SIDE_RIGHT';fake.planning_hold=False
        fake.velocity=np.zeros(3);fake.guidance.reset()
        fake.active_path=sample_path([[0,0,13],[-.1,-.6,13],[.2,-1,13],[20,0,13]])
        fake.rejoin=fake.active_path[-1];fake.pose_time=100.05;fake.map.last_update=100.05
        with patch('urban_drone.mission.time.monotonic',return_value=100.05):Mission.control(fake)
        self.assertGreaterEqual(fake.target[0],fake.position[0]-1e-9)
        self.assertLess(fake.target[1],fake.position[1])

    def plane(self):
        obs,prior=self.geometry([[.8,0,13]])
        obs.distances=lambda p:.8-np.asarray(p)[:,0]
        return obs,prior

    def test_recovery_cannot_reverse_without_explicit_blockage_authorization(self):
        obs,prior=self.plane();p=np.array([0.,0.,13.])
        self.assertIsNone(clearance_recovery(p,obs,prior,forward=[1,0],allow_up=True,allow_down=True))
        back=clearance_recovery(p,obs,prior,forward=[1,0],allow_up=True,allow_down=True,allow_backward=True)
        self.assertIsNotNone(back);self.assertLess(back[-1,0],p[0])

    def test_partial_forward_escape_precedes_full_reverse_even_when_reverse_allowed(self):
        obs,prior=self.geometry([[0,0,12.1]])
        def distances(points):
            x=np.asarray(points)[:,0];d=np.sqrt(.9**2+x**2)
            return np.where(x>=0,np.minimum(d,1.25),d)
        obs.distances=distances
        route=clearance_recovery([0,0,13],obs,prior,forward=[1,0],allow_backward=True)
        self.assertIsNotNone(route)
        self.assertGreater(route[-1,0],0.)
        self.assertLess(obs.distances(route[-1:])[0],1.8)

    def test_small_backward_component_is_not_misclassified_as_sideways(self):
        normal=np.array([-.02,1.]);normal/=np.linalg.norm(normal)
        obs,prior=self.geometry([[-.8*normal[0],-.8*normal[1],13.]])
        obs.distances=lambda p:.8+np.asarray(p)[:,:2]@normal
        obs.observed_free=lambda p,margin=.4:abs(np.asarray(p)[:,0]+.02*np.asarray(p)[:,1])<.001
        start=np.array([0.,0.,13.])
        self.assertIsNone(clearance_recovery(start,obs,prior,forward=[1,0]))
        route=clearance_recovery(start,obs,prior,forward=[1,0],allow_backward=True)
        self.assertIsNotNone(route)
        self.assertLess(route[-1,0],0.)

    def test_actual_recovery_authorizes_reverse_only_for_positive_route_collision(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as directory:
            for blocked in (False,True):
                obs,prior=self.plane();route=sample_path([[0,0,13],[2,0,13]] if blocked else [[0,0,13],[0,2,13]])
                fake=SimpleNamespace(prior=prior,active_path=route,path_index=0,global_route=route,global_index=0,
                    strategy='TRACK',lock=threading.RLock(),rejoin=None,recovery_context=None,recoveries=0,
                    reverse_gate=GlobalReplanPolicy(),
                    output=Path(directory),event=Mock(),maneuver_selector=SimpleNamespace(side=None))
                for i in range(3):
                    obs.updates=3+i
                    with patch('urban_drone.mission.time.monotonic',return_value=100.+i):
                        found=Mission.start_recovery(fake,np.array([0.,0.,13.]),obs,np.array([1.,0.]))
                    self.assertEqual(found,blocked and i==2)
                if blocked:self.assertTrue(fake.recovery_reverse_authorized)

    def test_explicitly_authorized_recovery_is_not_clipped_to_hover(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture();fake.strategy='RECOVER';fake.planning_hold=False
        fake.recovery_reverse_authorized=True;fake.velocity=np.zeros(3);fake.guidance.reset()
        fake.active_path=sample_path([[0,0,13],[-1,0,13]]);fake.rejoin=fake.active_path[-1]
        fake.pose_time=100.05;fake.map.last_update=100.05
        with patch('urban_drone.mission.time.monotonic',return_value=100.05):Mission.control(fake)
        self.assertLess(fake.target[0],fake.position[0])


if __name__=='__main__':unittest.main()
