"""Sensed escape geometry and directional speed; no PX4 flight is simulated."""

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.clearance import EXECUTION_CLEARANCE,CONTACT_CLEARANCE,RECOVERY_TARGET
from urban_drone.perception import RangeMap,Scan
from urban_drone.planning import PriorMap,sample_path
from urban_drone.recovery import clearance_recovery,separating_segment
from urban_drone.trajectory import approach_speed,VelocityGuidance
from urban_drone.global_policy import GlobalReplanPolicy


class EscapePolicyTests(unittest.TestCase):
    def setup_wall(self,start=.45):
        prior=PriorMap(dict(bounds=[-10,10,-10,10],grid_resolution=.5,
            vehicle_radius=.35,safety_margin=.15,tracking_margin=.7,
            known_buildings=[],max_height=10.,cruise_height=2.))
        observation=RangeMap(prior.bounds,window=[12,12,12])
        y,z=np.meshgrid(np.arange(-3,3.01,.2),np.arange(0,6.01,.2))
        observation.points=np.column_stack((np.zeros(y.size),y.ravel(),z.ravel()))
        observation.tree=cKDTree(observation.points)
        # A synthetic scan of the same wall; outward rays see open space.
        az=np.linspace(-np.pi,np.pi,181);el=np.linspace(-np.pi/2,np.pi/2,37)
        a,e=np.meshgrid(az,el);dx=np.cos(e)*np.cos(a)
        ranges=np.full(dx.shape,np.inf)
        np.divide(-start,dx,out=ranges,where=dx < -1e-6)
        observation.last_scan=Scan(ranges,az[0],az[1]-az[0],el[0],el[1]-el[0],
            12.,time.monotonic(),np.array([start,0.,2.]),np.array([0.,0.,0.,1.]),0.)
        observation.updates=2
        return np.array([start,0.,2.]),observation,prior

    def test_wall_to_the_side_does_not_cap_tangential_or_outward_speed(self):
        start,observation,_=self.setup_wall(.65)
        positions=np.repeat(start[None],3,axis=0)
        caps=approach_speed(positions,[[0,1,0],[1,0,0],[-1,0,0]],observation)
        np.testing.assert_allclose(caps[:2],1.8)
        self.assertGreaterEqual(caps[2],.2)
        self.assertLess(caps[2],.2+1.6*(.3-.2)/.3,
                        'closing into the wall also has a dynamic braking cap')
        # One heading can be broadcast across more than one point batch.
        np.testing.assert_allclose(observation.forward_distances(np.repeat(start[None],65,axis=0),[1,0,0]),np.inf)

    def test_inside_stop_floor_can_escape_outward_with_real_visibility_check(self):
        start,observation,prior=self.setup_wall()
        self.assertFalse(observation.safe(start[None],EXECUTION_CLEARANCE))
        path=clearance_recovery(start,observation,prior,forward=[1,0])
        self.assertIsNotNone(path)
        self.assertTrue(separating_segment(path,observation,prior))
        self.assertGreaterEqual(observation.distances(path[-1:])[0],RECOVERY_TARGET)
        self.assertFalse(separating_segment(sample_path([start,start-[.02,0,0]]),observation,prior))
        self.assertFalse(separating_segment(sample_path([start,start+[0,1,0]]),observation,prior))

    def test_contact_unseen_and_altitude_violations_are_rejected(self):
        start,observation,prior=self.setup_wall(CONTACT_CLEARANCE-.01)
        self.assertIsNone(clearance_recovery(start,observation,prior,forward=[1,0],allow_up=True))
        start,observation,prior=self.setup_wall()
        observation.last_scan=None
        self.assertIsNone(clearance_recovery(start,observation,prior,forward=[1,0],allow_up=True))
        start,observation,prior=self.setup_wall()
        observation.observed_free=lambda p,margin=.4:np.ones(len(p),bool)
        self.assertFalse(separating_segment([[.45,0,1],[2,0,.8]],observation,prior))
        self.assertFalse(separating_segment([[.45,0,9],[2,0,10.1]],observation,prior))

    def test_outward_commands_reach_normal_clearance_in_a_simple_integrator(self):
        position,observation,prior=self.setup_wall()
        endpoint=clearance_recovery(position,observation,prior,forward=[1,0])[-1]
        guidance=VelocityGuidance();dt=.05;initial=position.copy()
        for _ in range(240):
            # Isolate the emergency low-speed branch with perfect tracking;
            # this is a control/geometry check, not a dynamics or SITL result.
            requested=(endpoint-position);requested*=min(1.,.2/np.linalg.norm(requested))
            velocity=guidance.update(requested,dt)
            segment=sample_path([position,position+velocity*dt],.02)
            self.assertTrue(separating_segment(segment,observation,prior))
            position=segment[-1]
            if observation.distances(position[None])[0]>1.2:break
        self.assertGreater(position[0]-initial[0],.7)
        self.assertGreater(observation.distances(position[None])[0],1.2)

    def test_recovery_recheck_does_not_include_a_sample_behind_live_pose(self):
        from urban_drone.mission import Mission
        position,observation,prior=self.setup_wall(.61)
        observation.snapshot=lambda:observation
        route=sample_path([[0,0,2],[5,0,2]])
        fake=SimpleNamespace(complete=False,position=position,phase='FLY',map=observation,
            publish_paths=lambda:None,initial_route=route,global_route=route,global_index=0,
            strategy='RECOVER',rejoin=np.array([2.,0.,2.]),prior=prior,
            lock=threading.RLock(),active_path=sample_path([[.55,0,2],[2,0,2]]),path_index=0,
            resume_after_recovery=Mock(),global_policy=GlobalReplanPolicy())
        Mission.plan(fake)
        fake.resume_after_recovery.assert_not_called()

    def test_captured_first_wall_uses_three_dimensional_separation(self):
        data=Path(__file__).parent/'data'
        prior=PriorMap(json.loads((data/'roof_recovery_prior.json').read_text()))
        fixture=np.load(data/'first_wall_recovery_128.npz')
        observation=RangeMap(prior.bounds,window=prior.prior['local_window'])
        observation.points=fixture['points'];observation.tree=cKDTree(observation.points)
        # Recorded occupancy lacks raw visibility; geometry-only regression.
        observation.observed_free=lambda p,margin=.4:np.ones(len(p),bool)
        start=fixture['position']
        path=clearance_recovery(start,observation,prior,forward=[1,0],allow_up=True)
        self.assertIsNotNone(path)
        self.assertGreater(path[-1,0],start[0])
        self.assertGreater(path[-1,2],start[2]+.5)
        self.assertTrue(separating_segment(path,observation,prior))
        self.assertGreaterEqual(approach_speed(start[None],path[-1:]-start,observation)[0],.2)


if __name__=='__main__':unittest.main()
