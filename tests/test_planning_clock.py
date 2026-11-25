"""Planning snapshots freeze evidence ages; execution never freezes freshness."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.global_policy import GlobalReplanPolicy,eligible_candidate
from urban_drone.local import candidates,ManeuverSelector
from urban_drone.perception import RangeMap,Scan
from urban_drone.planning import PriorMap,Candidate,sample_path
from urban_drone.trajectory import assess_path


class PlanningClockTests(unittest.TestCase):
    def observation(self,wall=False):
        prior=PriorMap(dict(bounds=[-10,20,-12,12],grid_resolution=.5,vehicle_radius=.35,
            safety_margin=.15,tracking_margin=.7,known_buildings=[],cruise_height=3.,
            max_height=10.,start=[0,0],goal=[12,0],local_lateral_limit=10.))
        obs=RangeMap(prior.bounds,window=[24,24,16])
        az=np.linspace(-np.pi,np.pi,181);el=np.linspace(-np.pi/2,np.pi/2,37)
        a,e=np.meshgrid(az,el);ranges=np.full(a.shape,np.inf)
        if wall:
            dirs=np.stack((np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)),axis=-1)
            origin=np.array([0.,0.,3.3]);safe=np.where(abs(dirs)<1e-10,1.,dirs)
            t1=(np.array([6.,-7.,0.])-origin)/safe
            t2=(np.array([7.2,7.,4.2])-origin)/safe
            parallel=abs(dirs)<1e-10
            near=np.where(parallel,-np.inf,np.minimum(t1,t2)).max(axis=-1)
            far=np.where(parallel,np.inf,np.maximum(t1,t2)).min(axis=-1)
            outside=np.any(parallel & ((origin<[6,-7,0])|(origin>[7.2,7,4.2])),axis=-1)
            ranges=np.where((far>=np.maximum(near,0)) & (far>0) & ~outside,near,np.inf)
            y,z=np.meshgrid(np.arange(-7,7.01,.2),np.arange(0,4.21,.2))
            obs.points=np.vstack([np.column_stack((np.full(y.size,x),y.ravel(),z.ravel())) for x in (6.,7.2)])
            obs.tree=cKDTree(obs.points)
        obs.last_scan=Scan(ranges,az[0],az[1]-az[0],el[0],el[1]-el[0],12.,100.,
                           np.array([0.,0.,3.]),np.array([0.,0.,0.,1.]),0.)
        obs.updates=3;obs.last_update=100.
        return obs,prior

    def test_same_planning_path_keeps_score_while_live_view_expires(self):
        obs,_=self.observation();path=np.array([[0.,0.,3.],[5.,0.,3.]])
        with patch('urban_drone.perception.time.monotonic',return_value=100.5):snapshot=obs.snapshot()
        scores=[]
        for now in (100.6,102.0):
            with patch('urban_drone.perception.time.monotonic',return_value=now):
                score,metrics=assess_path(path,snapshot,path[0],path[-1],np.zeros(3))
                scores.append(score);self.assertEqual(metrics['unknown_fraction'],0.)
        self.assertEqual(scores[0],scores[1])
        with patch('urban_drone.perception.time.monotonic',return_value=102.):
            self.assertFalse(obs.snapshot(copy_evidence=False).observed_free(path).any())
            self.assertEqual(snapshot.snapshot().evaluation_time,100.5)

    def test_scan_already_stale_at_capture_remains_unobserved(self):
        obs,_=self.observation()
        with patch('urban_drone.perception.time.monotonic',return_value=102.):snapshot=obs.snapshot()
        with patch('urban_drone.perception.time.monotonic',return_value=103.):
            self.assertFalse(snapshot.observed_free([[1,0,3]]).any())
            self.assertEqual(snapshot.visibility_detail([1,0,3])['reason'],'scan_stale')

    def test_recent_free_voxels_use_same_frozen_clock(self):
        obs,_=self.observation();obs.last_scan.ranges[:]=np.nan
        point=np.array([1.,0.,3.3]);idx=tuple(np.floor((point-obs.origin)/obs.resolution).astype(int))
        obs.odds[idx]=-1.;obs.free_seen_at[idx]=97.8
        with patch('urban_drone.perception.time.monotonic',return_value=100.5):snapshot=obs.snapshot()
        with patch('urban_drone.perception.time.monotonic',return_value=101.):
            self.assertTrue(snapshot.observed_free(point[None])[0])
            self.assertFalse(obs.snapshot(copy_evidence=False).observed_free(point[None])[0])

    def test_slow_scoring_does_not_remove_last_over_candidate_or_change_scores(self):
        obs,prior=self.observation(wall=True);clock=[100.5]
        with patch('urban_drone.perception.time.monotonic',side_effect=lambda:clock[0]):
            snapshot=obs.snapshot()
            fast=candidates([0,0,3],[12,0,3],snapshot,prior,velocity=np.zeros(3))
            def slow_score(*a,**kw):
                clock[0]+=.6
                return assess_path(*a,**kw)
            with patch('urban_drone.local.assess_path',side_effect=slow_score):
                slow=candidates([0,0,3],[12,0,3],snapshot,prior,velocity=np.zeros(3))
        self.assertGreater(clock[0],101.5)
        self.assertIn('OVER',[c.strategy for c in slow])
        self.assertEqual([c.strategy for c in slow],[c.strategy for c in fast])
        np.testing.assert_allclose([c.cost for c in slow],[c.cost for c in fast])

    def test_rejection_records_actual_unobserved_point_and_neighbor_rays(self):
        obs,prior=self.observation(wall=True);obs.last_scan.ranges[:]=np.nan
        with patch('urban_drone.perception.time.monotonic',return_value=100.5):
            snapshot=obs.snapshot();diagnostics={}
            candidates([0,0,3],[12,0,3],snapshot,prior,diagnostics=diagnostics)
            rejected=next(r for r in diagnostics['rejected'] if r['strategy']=='OVER' and r['reason']=='unobserved_prefix')
            detail=rejected['visibility']
            self.assertFalse(snapshot.observed_free([detail['point']],margin=.4)[0])
            self.assertEqual(len(detail['rays']),5)
            self.assertEqual(detail['scan_age_at_evaluation'],.5)
            self.assertEqual(detail['available_m'],0.)
            json.dumps(detail,allow_nan=False)

    def test_new_obstacle_is_rejected_by_live_submission_view(self):
        obs,prior=self.observation();path=sample_path([[0,0,3],[4,0,3]])
        with patch('urban_drone.perception.time.monotonic',return_value=100.5):
            snapshot=obs.snapshot()
            obs.points=np.array([[1.,0.,3.]]);obs.tree=cKDTree(obs.points)
            self.assertTrue(eligible_candidate(path,snapshot,prior))
            self.assertFalse(eligible_candidate(path,obs.snapshot(copy_evidence=False),prior))

    def test_actual_plan_does_not_submit_candidate_after_live_scan_expires(self):
        from urban_drone.mission import Mission
        obs,prior=self.observation(wall=True);route=sample_path([[0,0,3],[12,0,3]])
        path=sample_path([[0,0,3],[3,0,5.5],[9,0,5.5],[12,0,3]])
        clock=[100.5]
        def slow_candidates(*a,**kw):
            kw['diagnostics'].update(rejected=[],planning_origin=[0,0,3]);clock[0]=102.
            return [Candidate('OVER',path,10.)]
        with TemporaryDirectory() as d:
            fake=SimpleNamespace(complete=False,position=np.array([0.,0.,3.]),phase='FLY',
                map=obs,prior=prior,global_route=route,initial_route=route.copy(),global_index=0,
                strategy='TRACK',rejoin=None,active_path=route,path_index=0,lock=threading.RLock(),
                velocity=np.zeros(3),args=SimpleNamespace(baseline='full'),comparisons=0,
                output=Path(d),publish_paths=lambda:None,event=Mock(),planning_hold=False,
                maneuver_selector=ManeuverSelector(),
                global_policy=GlobalReplanPolicy(),global_replan=Mock(return_value=False))
            with patch('urban_drone.perception.time.monotonic',side_effect=lambda:clock[0]),\
                 patch('urban_drone.mission.candidates',side_effect=slow_candidates):
                Mission.plan(fake)
            self.assertTrue(fake.planning_hold)
            self.assertIs(fake.active_path,route)
            self.assertIsNone(fake.maneuver_selector.key,'a rejected proposal must not commit its maneuver key')
            self.assertTrue(any(a.args[0]=='candidate_submission_rejected' for a in fake.event.call_args_list))
            fake.global_replan.assert_not_called()


if __name__=='__main__':unittest.main()
