"""Global changes need sustained infeasibility or material whole-route gain."""

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.global_policy import (GlobalReplanPolicy,working_remainder,obstruction,
    eligible_candidate,join_paths,evaluate_remaining)
from urban_drone.local import ManeuverSelector
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap,sample_path,length


class GlobalPolicyTests(unittest.TestCase):
    def test_gain_must_pay_switch_cost_and_exceed_both_thresholds(self):
        p=GlobalReplanPolicy()
        self.assertFalse(p.compare(100,84,1,0).replace)  # 16 raw, 14 net < 15%
        self.assertFalse(p.compare(10,4,2,1).replace)    # 60%, but only 4 net < 5
        self.assertIsNone(p.better_since)

    def test_improvement_needs_distinct_maps_and_elapsed_confirmation(self):
        p=GlobalReplanPolicy()
        self.assertFalse(p.compare(100,70,1,0).replace)
        self.assertFalse(p.compare(100,70,1,3).replace)
        self.assertFalse(p.compare(100,70,2,3.1).replace)
        verdict=p.compare(100,70,3,3.2)
        self.assertTrue(verdict.replace);self.assertEqual(verdict.reason,'confirmed_improvement')

    def test_lost_benefit_resets_confirmation(self):
        p=GlobalReplanPolicy();p.compare(100,70,1,0);p.compare(100,70,2,1)
        p.compare(100,99,3,2)
        self.assertFalse(p.compare(100,70,4,3).replace)
        self.assertFalse(p.compare(100,70,5,4).replace)
        self.assertTrue(p.compare(100,70,6,5).replace)

    def test_unknown_or_available_local_route_never_proves_infeasibility(self):
        p=GlobalReplanPolicy()
        for i in range(4):
            self.assertFalse(p.observe_obstruction(None,True,i,i))
            self.assertFalse(p.observe_obstruction('observed_collision',False,i,i))
        self.assertFalse(p.blocked_versions)

    def test_blockage_needs_new_maps_time_and_local_failure(self):
        p=GlobalReplanPolicy()
        self.assertFalse(p.observe_obstruction('observed_collision',True,1,0))
        self.assertFalse(p.observe_obstruction('observed_collision',True,1,10))
        self.assertFalse(p.observe_obstruction('observed_collision',True,2,11))
        self.assertTrue(p.observe_obstruction('observed_collision',True,3,12))
        self.assertFalse(p.observe_obstruction(None,True,4,13))
        self.assertFalse(p.observe_obstruction('observed_collision',True,5,14))

    def test_infeasible_can_accept_longer_route_but_nonfinite_candidate_never_passes(self):
        p=GlobalReplanPolicy()
        self.assertTrue(p.compare(10,100,1,0,blocked_confirmed=True).replace)
        self.assertFalse(p.compare(10,float('inf'),2,1,blocked_confirmed=True).replace)
        self.assertFalse(p.compare(float('inf'),10,3,2).replace)

    def geometry(self,points=()):
        prior=PriorMap(dict(bounds=[-10,60,-30,30],grid_resolution=.5,vehicle_radius=.35,
            safety_margin=.15,tracking_margin=.7,known_buildings=[],max_height=10.,
            cruise_height=3.,start=[0,0],goal=[40,0]))
        observation=RangeMap(prior.bounds,window=[24,24,12])
        observation.points=np.asarray(points,dtype=float).reshape(-1,3)
        observation.tree=cKDTree(observation.points) if len(observation.points) else None
        observation.observed_free=lambda p,margin=.4:np.ones(len(p),bool)
        observation.snapshot=lambda **kw:observation
        observation.updates=1
        return observation,prior

    def test_search_margin_and_missing_visibility_are_not_collision_evidence(self):
        observation,prior=self.geometry([[0,1.1,3],[1,1.1,3]])
        path=sample_path([[0,0,3],[1,0,3]])
        self.assertFalse(observation.safe(path,1.2))
        observation.observed_free=lambda p,margin=.4:np.zeros(len(p),bool)
        self.assertIsNone(obstruction(path,observation,prior))
        self.assertFalse(eligible_candidate(path,observation,prior))

    def test_comparison_includes_local_detour_and_global_tail_to_same_goal(self):
        route=sample_path([[0,0,3],[10,0,3],[20,0,3]])
        position=np.array([5.,2.,4.]);local=sample_path([position,[8,2,4],[10,0,3]])
        combined=working_remainder(position,route,30,local,0,np.array([10,0,3]))
        np.testing.assert_allclose(combined[0],position)
        np.testing.assert_allclose(combined[-1],route[-1])
        self.assertTrue(np.any(np.linalg.norm(combined-[8,2,4],axis=1)<1e-8))
        self.assertGreater(length(combined),length(local)+9.9)

    def test_duplicate_join_does_not_add_artificial_motion_cost(self):
        observation,_=self.geometry()
        a=np.array([[0.,0.,3.],[2.,0.,3.]])
        b=np.array([[2.,0.,3.],[4.,0.,3.]])
        combined=join_paths(a,b)
        self.assertTrue(np.all(np.linalg.norm(np.diff(combined,axis=0),axis=1)>0))
        cost,_=evaluate_remaining(combined,observation,combined[0],combined[-1],np.zeros(3))
        direct,_=evaluate_remaining([a[0],b[-1]],observation,a[0],b[-1],np.zeros(3))
        self.assertAlmostEqual(cost,direct,places=5)

    def mission(self,directory,blocked=False,detour=True):
        observation,prior=self.geometry([[10,0,3]] if blocked else ())
        old=sample_path([[0,0,3],[0,20,3],[40,20,3],[40,0,3]] if detour else [[0,0,3],[40,0,3]])
        events=[]
        def new_route(start,occupancy):
            # A large detour is intentionally costly for the infeasibility test.
            points=[start,[start[0],20,3],[40,20,3],[40,0,3]] if blocked else [start,[40,0,3]]
            route=sample_path(points)
            return route.copy(),route
        prior.route=Mock(side_effect=new_route)
        fake=SimpleNamespace(position=np.array([0.,0.,3.]),prior=prior,prior_data=prior.prior,
            map=observation,velocity=np.zeros(3),global_route=old,global_index=0,
            active_path=old.copy(),path_index=0,rejoin=None,strategy='TRACK',
            initial_route=old.copy(),last_global_replan=0.,global_policy=GlobalReplanPolicy(),
            reverse_gate=GlobalReplanPolicy(),
            lock=threading.RLock(),maneuver_selector=ManeuverSelector(),planning_hold=False,
            recovery_requested=False,replans=0,output=Path(directory),
            event=lambda name,**kw:events.append(dict(event=name,**kw)))
        return fake,observation,events

    def test_actual_mission_replaces_only_after_repeated_whole_route_benefit(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,events=self.mission(d);initial=fake.initial_route.copy()
            for version,now in ((1,100.),(2,105.),(3,110.)):
                observation.updates=version
                with patch('urban_drone.mission.time.monotonic',return_value=now):
                    changed=Mission.global_replan(fake,fake.position,observation)
                self.assertEqual(changed,version==3)
            self.assertEqual(fake.replans,1)
            self.assertLess(length(fake.global_route),length(initial))
            np.testing.assert_array_equal(fake.initial_route,initial)
            self.assertTrue((Path(d)/'global_replanned_01.csv').exists())
            self.assertEqual([e for e in events if e['event']=='global_replanned'][-1]['reason'],'confirmed_improvement')

    def test_actual_mission_confirms_collision_then_can_accept_costlier_route(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,events=self.mission(d,blocked=True,detour=False)
            for version,now in ((1,100.),(2,101.),(3,102.)):
                observation.updates=version
                with patch('urban_drone.mission.time.monotonic',return_value=now):
                    changed=Mission.global_replan(fake,fake.position,observation,local_failed=True)
                self.assertEqual(changed,version==3)
            self.assertEqual(fake.prior.route.call_count,1)
            comparison=[e for e in events if e['event']=='global_replan_comparison'][-1]
            self.assertEqual(comparison['reason'],'confirmed_infeasible')
            self.assertGreater(comparison['new_cost'],comparison['old_cost'])

    def test_actual_mission_does_not_replan_for_unknown_straight_route(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,_=self.mission(d,detour=False)
            observation.observed_free=lambda p,margin=.4:np.zeros(len(p),bool)
            for i in range(4):
                observation.updates=i+1
                with patch('urban_drone.mission.time.monotonic',return_value=100.+i*5):
                    self.assertFalse(Mission.global_replan(fake,fake.position,observation,local_failed=True))
            fake.prior.route.assert_not_called()
            self.assertEqual(fake.replans,0)

    def test_actual_mission_rejects_proposal_blocked_by_fresh_observation(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,events=self.mission(d)
            fresh,_=self.geometry([[1,0,3],[0,1,3]])
            observation.snapshot=lambda **kw:fresh
            with patch('urban_drone.mission.time.monotonic',return_value=100.):
                self.assertFalse(Mission.global_replan(fake,fake.position,observation))
            self.assertEqual(fake.replans,0)
            self.assertTrue(any(e.get('reason')=='candidate_not_executable' for e in events))

    def test_actual_mission_keeps_active_over_and_respects_height_gate(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,_=self.mission(d);fake.strategy='OVER'
            with patch('urban_drone.mission.time.monotonic',return_value=100.):
                self.assertFalse(Mission.global_replan(fake,fake.position,observation))
            fake.strategy='TRACK';fake.position[2]=4.
            with patch('urban_drone.mission.time.monotonic',return_value=105.):
                self.assertFalse(Mission.global_replan(fake,fake.position,observation))
            fake.prior.route.assert_not_called()

    def test_fresh_candidate_must_keep_search_reserve_not_just_execution_floor(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,events=self.mission(d)
            # New direct tail passes near (20, 1, 3): 0.8 m from this point is
            # executable, but below the 1.2 m required for a new search tail.
            fresh,_=self.geometry([[20,1.8,3]])
            observation.snapshot=lambda **kw:fresh
            with patch('urban_drone.mission.time.monotonic',return_value=100.):
                self.assertFalse(Mission.global_replan(fake,fake.position,observation))
            self.assertEqual(fake.replans,0)
            self.assertTrue(any(e.get('reason')=='candidate_lost_planning_reserve' for e in events))

    def test_real_astar_route_runs_through_confirmation_and_preserves_red_reference(self):
        from urban_drone.mission import Mission
        with TemporaryDirectory() as d:
            fake,observation,events=self.mission(d,blocked=True,detour=False)
            # Use the real global A*, connector and smoothing implementation,
            # rather than the proposal stub used to isolate comparison cases.
            fake.prior.route=PriorMap.route.__get__(fake.prior,PriorMap)
            original=fake.initial_route.copy()
            for version,now in ((1,100.),(2,101.),(3,102.)):
                observation.updates=version
                with patch('urban_drone.mission.time.monotonic',return_value=now):
                    changed=Mission.global_replan(fake,fake.position,observation,local_failed=True)
                self.assertEqual(changed,version==3)
            self.assertEqual(fake.replans,1)
            np.testing.assert_array_equal(fake.initial_route,original)
            self.assertGreater(observation.distances(sample_path(fake.global_route,.05)).min(),1.2)
            self.assertTrue(any(e.get('reason')=='confirmed_infeasible' for e in events))


if __name__=='__main__':unittest.main()
