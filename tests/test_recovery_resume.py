import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap,sample_path,reference_forward
from urban_drone.recovery import clearance_recovery
from urban_drone.global_policy import GlobalReplanPolicy

DATA=Path(__file__).parent/'data'


class RecoveryResumeTests(unittest.TestCase):
    def captured(self,version):
        prior=PriorMap(json.loads((DATA/'roof_recovery_prior.json').read_text()))
        data=np.load(DATA/f'roof_recovery_{version}.npz')
        observation=RangeMap(prior.bounds,window=prior.prior['local_window'])
        observation.points=data['points'];observation.tree=cKDTree(observation.points)
        # Stored occupancy has no raw visibility history. These are geometry
        # regressions; real execution must still verify recent free evidence.
        observation.observed_free=lambda p,margin=.35:np.ones(len(p),bool)
        return data['position'],observation,prior,data['rejoin']

    def test_captured_roof_can_rise_instead_of_retreat_when_observed(self):
        for version in (855,924):
            position,observation,prior,target=self.captured(version)
            before=observation.points.copy()
            path=clearance_recovery(position,observation,prior,forward=target-position,allow_up=True)
            self.assertIsNotNone(path)
            self.assertGreater(path[-1,2]-position[2],.5)
            forward=(target-position)[:2];forward/=np.linalg.norm(forward)
            self.assertGreater((path[-1,:2]-position[:2])@forward,.05)
            self.assertGreater(observation.distances(path).min(),.75)
            self.assertGreaterEqual(observation.distances(path[-1:])[0],1.6)
            np.testing.assert_array_equal(before,observation.points)

    def test_unobserved_ascent_is_not_used(self):
        position,observation,prior,target=self.captured(855)
        observation.observed_free=lambda p,margin=.35:abs(p[:,2]-position[2])<.1
        path=clearance_recovery(position,observation,prior,forward=target-position,allow_up=True)
        # Partial outward adjustment within the observed layer is valid;
        # it must never climb outside that layer to regain preferred reserve.
        if path is not None:
            self.assertLess(np.max(abs(path[:,2]-position[2])),.1)
            self.assertTrue(observation.observed_free(path).all())

    def test_recovery_preserves_over_or_pass_context_and_brakes_before_replanning(self):
        from urban_drone.mission import Mission
        for strategy in ('OVER','PASS'):
            position=np.array([170.,-5.,15.1]);target=np.array([178.,-5.,13.])
            fake=SimpleNamespace(lock=threading.RLock(),recovery_context=dict(strategy=strategy,rejoin=target),
                                 reverse_gate=GlobalReplanPolicy(),
                                 strategy='RECOVER',rejoin=np.array([170.,-5.,15.]),event=lambda *a,**k:None)
            Mission.resume_after_recovery(fake,position)
            self.assertEqual(fake.strategy,strategy)
            np.testing.assert_allclose(fake.rejoin,target)
            np.testing.assert_allclose(fake.active_path,position[None])
            np.testing.assert_allclose(fake.hold_target,position)
            self.assertTrue(fake.planning_hold)

    def test_recovery_brake_point_is_not_selected_as_zero_cost_incumbent(self):
        from urban_drone.mission import Mission
        position=np.array([4.,0.,16.]);route=sample_path([[0,0,13],[30,0,13]])
        observation=SimpleNamespace(updates=10,points=np.empty((0,3)),safe=lambda *a,**k:True,
                                    distances=lambda p:np.full(len(p),5.),
                                    observed_free=lambda p,margin=.4:np.ones(len(p),bool),
                                    aperture_ahead=lambda *a:False)
        observation.snapshot=lambda:observation
        collected=[]
        def choose(options,*args):collected.extend(options);return None,'wait_for_observations'
        def no_candidates(*a,**kwargs):kwargs['diagnostics'].update(rejected=[]);return []
        with TemporaryDirectory() as directory:
            fake=SimpleNamespace(complete=False,position=position,phase='FLY',map=observation,
                 global_route=route,initial_route=route.copy(),global_index=0,strategy='OVER',rejoin=np.array([12.,0.,13.]),
                 prior=SimpleNamespace(safe=lambda *a,**k:True,execution_inflation=.75),
                 active_path=position[None],path_index=0,lock=threading.RLock(),velocity=np.zeros(3),
                 args=SimpleNamespace(baseline='full'),output=Path(directory),comparisons=0,
                 maneuver_selector=SimpleNamespace(side=None,choose=choose),
                 publish_paths=lambda:None,event=lambda *a,**k:None,global_policy=GlobalReplanPolicy(),
                 reverse_gate=GlobalReplanPolicy())
            with patch('urban_drone.mission.candidates',side_effect=no_candidates):Mission.plan(fake)
        self.assertEqual(collected,[],'the temporary brake point is not a complete incumbent route')


if __name__=='__main__':unittest.main()
