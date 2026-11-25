"""Collision, direction and maneuver integration regressions; no flight."""

from io import StringIO
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.clearance import EXECUTION_CLEARANCE,REACTION_TIME,BRAKING_XY,BRAKING_Z
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap,sample_path
from urban_drone.safety import (braking_speed,directional_deceleration,stopping_path,
    check_motion,remaining_path,path_prefix,corridor_valid,maneuver_stage,PositionBrake)
from urban_drone.trajectory import VelocityGuidance,approach_speed
from urban_drone.recovery import clearance_recovery
from urban_drone.global_policy import GlobalReplanPolicy


class DirectionalSafetyTests(unittest.TestCase):
    def setup_geometry(self,points):
        prior=PriorMap(dict(bounds=[-20,20,-20,20],grid_resolution=.5,vehicle_radius=.35,
            safety_margin=.15,tracking_margin=.7,known_buildings=[],max_height=10.,cruise_height=3.))
        observation=RangeMap(prior.bounds,window=[12,12,12])
        observation.points=np.asarray(points,dtype=float).reshape(-1,3)
        observation.tree=cKDTree(observation.points) if len(observation.points) else None
        # These tests isolate geometry/control. The actual visibility function
        # and unseen escape refusal are covered in test_escape_policy.
        observation.observed_free=lambda p,margin=.4:np.ones(len(p),bool)
        observation.last_update=time.monotonic();observation.updates=2
        return observation,prior

    def gate(self,half_width=.65):
        x,z=np.meshgrid(np.arange(-2,5.01,.1),np.arange(1,5.01,.1))
        wall=lambda side:np.column_stack((x.ravel(),np.full(x.size,side),z.ravel()))
        return self.setup_geometry(np.vstack((wall(half_width),wall(-half_width))))

    def test_clear_roof_allows_horizontal_travel_but_not_descent(self):
        x,y=np.meshgrid(np.arange(-2,5.01,.1),np.arange(-2,2.01,.1))
        observation,prior=self.setup_geometry(np.column_stack((x.ravel(),y.ravel(),np.full(x.size,2.))))
        p=np.array([0.,0.,2.65])
        self.assertTrue(check_motion(p,p+[.5,0,0],[1,0,0],observation,prior).allowed)
        np.testing.assert_allclose(approach_speed(p[None],[[1,0,0]],observation),1.8)
        self.assertFalse(check_motion(p,p+[.5,0,-.25],[0,0,0],observation,prior).allowed)
        decision=check_motion(p,p+[.3,0,0],[0,0,-.5],observation,prior)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason,'braking_distance')

    def test_gate_side_walls_allow_forward_motion_but_lateral_drift_brakes(self):
        observation,prior=self.gate();p=np.array([0.,0.,3.])
        self.assertTrue(check_motion(p,p+[.4,0,0],[1,0,0],observation,prior).allowed)
        decision=check_motion(p,p+[.4,0,0],[1,.35,0],observation,prior)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason,'braking_distance')

    def test_forward_outward_correction_is_allowed_when_coasting_keeps_contact_guard(self):
        observation,prior=self.gate();p=np.array([0.,.1,3.])
        # Correct the lateral drift without forcing a full reversal in X.
        decision=check_motion(p,p+[.2,-.1,0],[1,.12,0],observation,prior)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.state,'SEPARATE')

    def test_tangential_sample_ripple_is_not_an_outward_escape_from_left_drift(self):
        # Along a planar wall, staggered samples create ~1 mm distance ripple.
        # That must not authorize a forward-only command while drifting left.
        x=np.arange(0,2.01,.1)
        points=np.column_stack((x,np.full(len(x),1.1),np.full(len(x),3.)))
        points=np.vstack((points,[.75,1.1,3.]))
        observation,prior=self.setup_geometry(points);p=np.array([.75,0.,3.])
        tangent=check_motion(p,p+[.4,0,0],[0,.82,0],observation,prior)
        self.assertFalse(tangent.allowed)
        self.assertEqual(tangent.reason,'braking_distance')
        outward=check_motion(p,p+[.2,-.2,0],[0,.82,0],observation,prior)
        self.assertTrue(outward.allowed)
        self.assertEqual(outward.state,'SEPARATE')

    def test_braking_model_has_distinct_axes_and_matching_speed_budget(self):
        np.testing.assert_allclose(directional_deceleration([[1,0,0],[0,0,1]]),[BRAKING_XY,BRAKING_Z])
        observation,_=self.setup_geometry([[2,0,3]])
        cap=braking_speed([[0,0,3]],[[1,0,0]],observation)[0]
        self.assertAlmostEqual(cap*REACTION_TIME+cap**2/(2*BRAKING_XY),2-EXECUTION_CLEARANCE)
        slow=stopping_path([0,0,3],[.5,0,0])[-1,0]
        fast=stopping_path([0,0,3],[1,0,0])[-1,0]
        self.assertGreater(fast,2*slow)
        observation,_=self.setup_geometry([[1,.49,3]])
        expected=1-np.sqrt(EXECUTION_CLEARANCE**2-.49**2)
        self.assertAlmostEqual(observation.collision_distances([[0,0,3]],[[1,0,0]])[0],expected)

    def test_predicted_stopping_path_must_be_observed(self):
        observation,prior=self.setup_geometry([])
        observation.observed_free=lambda p,margin=.4:np.asarray(p)[:,0]<.6
        decision=check_motion([0,0,3],[.2,0,3],[1,0,0],observation,prior)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason,'stopping_unobserved')

    def test_control_snapshot_keeps_geometry_and_origin_during_recenter(self):
        observation,_=self.setup_geometry([[1,0,3]])
        view=observation.snapshot(copy_evidence=False)
        origin=view.origin.copy();odds=view.odds
        self.assertIs(view.lock,observation.lock)
        observation.recenter([10,0,3])
        observation.points=np.array([[10.,0.,3.]])
        observation.tree=cKDTree(observation.points)
        np.testing.assert_array_equal(view.origin,origin)
        self.assertIs(view.odds,odds)
        self.assertIsNot(view.odds,observation.odds)
        self.assertAlmostEqual(view.distances([[0,0,3]])[0],1.)

    def test_existing_corridor_keeps_bend_and_executes_below_search_reserve(self):
        observation,prior=self.setup_geometry([[.48,.52,3]])
        path=sample_path([[0,0,3],[1,0,3],[1,1,3]])
        p=np.array([.9,0.,3.]);index=int(np.argmin(np.linalg.norm(path-p,axis=1)))
        old_skip=np.vstack((p,path[min(index+6,len(path)-1):]))
        self.assertFalse(observation.safe(sample_path(old_skip,.02),EXECUTION_CLEARANCE))
        remainder=remaining_path(p,path,index)
        self.assertTrue(corridor_valid(remainder,path[-1],observation,prior))
        self.assertTrue(np.any(np.linalg.norm(remainder-[1,0,3],axis=1)<1e-8))
        self.assertAlmostEqual(np.linalg.norm(np.diff(path_prefix([[0,0,3],[5,0,3]],2),axis=0)),2.)

    def test_corridor_rejects_new_intrusion_unknown_prefix_and_temporary_point(self):
        observation,prior=self.setup_geometry([]);path=sample_path([[0,0,3],[3,0,3]])
        self.assertTrue(corridor_valid(path,path[-1],observation,prior))
        self.assertFalse(corridor_valid(path[:1],path[-1],observation,prior))
        observation.observed_free=lambda p,margin=.4:np.zeros(len(p),bool)
        self.assertFalse(corridor_valid(path,path[-1],observation,prior))
        observation,prior=self.setup_geometry([[1,0,3]])
        self.assertFalse(corridor_valid(path,path[-1],observation,prior))
        self.assertFalse(check_motion([0,0,3],[1,0,3],[0,0,0],observation,prior).allowed)

    def test_known_buildings_never_get_a_maneuver_exemption(self):
        observation,_=self.setup_geometry([])
        prior=PriorMap(dict(bounds=[-20,20,-20,20],grid_resolution=.5,vehicle_radius=.35,
            safety_margin=.15,known_buildings=[dict(center=[1,0,3],size=[.2,2,6])],max_height=10.))
        self.assertFalse(check_motion([0,0,3],[2,0,3],[0,0,0],observation,prior,recovering=True).allowed)

    def test_maneuver_stages_follow_observed_bottleneck(self):
        observation,_=self.setup_geometry([[3,.7,3]])
        path=sample_path([[0,0,3],[6,0,3]])
        for strategy in ('OVER','PASS'):
            self.assertEqual(maneuver_stage(path,[0,0,3],observation,strategy),'APPROACH')
            self.assertEqual(maneuver_stage(path,[3,0,3],observation,strategy),'TRANSIT')
            self.assertEqual(maneuver_stage(path,[6,0,3],observation,strategy),'EXIT')

    def test_emergency_recovery_can_regain_execution_reserve_in_narrow_gate(self):
        observation,prior=self.gate();start=np.array([0.,.2,3.])
        self.assertLess(observation.distances(start[None])[0],EXECUTION_CLEARANCE)
        path=clearance_recovery(start,observation,prior,forward=[1,0],allow_up=True,allow_down=True)
        self.assertIsNotNone(path)
        measured=observation.distances(path)
        self.assertGreaterEqual(measured[-1],EXECUTION_CLEARANCE+.10)
        self.assertLess(measured[-1],1.8,'the preferred reserve does not fit in this corridor')
        self.assertTrue(np.all(np.diff(measured)>=-.002))

    def test_recovery_restores_old_corridor_only_after_fresh_validation(self):
        from urban_drone.mission import Mission
        observation,prior=self.gate();path=sample_path([[0,0,3],[4,0,3]])
        position=np.array([.1,.05,3.])
        for visible in (True,False):
            observation.observed_free=lambda p,margin=.4:np.full(len(p),visible)
            fake=SimpleNamespace(lock=threading.RLock(),prior=prior,event=Mock(),strategy='RECOVER',global_route=path,
                reverse_gate=GlobalReplanPolicy(),
                recovery_context=dict(strategy='PASS',rejoin=path[-1],path=path,index=0))
            Mission.resume_after_recovery(fake,position,observation)
            self.assertEqual(fake.strategy,'PASS')
            self.assertEqual(fake.planning_hold,not visible)
            if visible:np.testing.assert_allclose(fake.active_path[-1],path[-1])
            else:np.testing.assert_allclose(fake.active_path,position[None])

    def test_short_recovery_is_not_finished_while_start_is_still_inside_stop_floor(self):
        from urban_drone.mission import Mission
        observation,prior=self.gate();position=np.array([0.,.2,3.]);observation.snapshot=lambda:observation
        route=sample_path([[0,0,3],[4,0,3]])
        fake=SimpleNamespace(complete=False,position=position,phase='FLY',map=observation,
            publish_paths=lambda:None,initial_route=route,global_route=route,global_index=0,
            strategy='RECOVER',rejoin=np.array([0.,.1,3.]),prior=prior,
            lock=threading.RLock(),resume_after_recovery=Mock(),global_policy=GlobalReplanPolicy())
        Mission.plan(fake)
        fake.resume_after_recovery.assert_not_called()

    def control_fixture(self,velocity):
        observation,prior=self.gate(.55)
        observation.snapshot=lambda **kwargs:observation
        now=time.monotonic();position=np.array([0.,0.,3.]);path=sample_path([position,[3,0,3]])
        return SimpleNamespace(complete=False,position=position,last_control=now-.05,
            phase='FLY',started=now,args=SimpleNamespace(timeout=1000),prior=prior,
            prior_data=dict(cruise_height=3.,start=[0,0],goal=[10,0]),
            state=SimpleNamespace(armed=True,mode='OFFBOARD'),pose_time=now,map=observation,
            sensor_hold_since=None,lock=threading.RLock(),active_path=path,path_index=0,
            profile_path=None,profile_limits=None,strategy='PASS',rejoin=path[-1],
            global_route=path.copy(),initial_route=path.copy(),stopper=PositionBrake(),
            planning_wait_speed=None,recovery_reverse_authorized=False,
            planning_hold=True,velocity=np.array(velocity,dtype=float),guidance=VelocityGuidance(),
            maneuver_phase='APPROACH',execution_state=None,escape_override_active=False,
            hold_target=None,hold_since=None,event=Mock(),setpoint=Mock(),command_file=StringIO(),
            progress_anchor=position.copy(),last_progress=now,abort=Mock(),arrival_since=None,
            global_policy=GlobalReplanPolicy())

    def test_control_continues_valid_pass_while_planner_compares_alternatives(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture([.2,0,0]);Mission.control(fake)
        self.assertIsNone(fake.hold_target)
        self.assertGreater(fake.target[0],fake.position[0])
        self.assertFalse(fake.recovery_requested)

    def test_control_brakes_drift_and_planner_bypasses_switch_wait_to_recover(self):
        from urban_drone.mission import Mission
        fake=self.control_fixture([1,.35,0]);Mission.control(fake)
        self.assertTrue(fake.recovery_requested)
        np.testing.assert_allclose(fake.target,fake.position)
        fake.publish_paths=lambda:None
        fake.initial_route=fake.active_path;fake.global_route=fake.active_path;fake.global_index=0
        fake.args.baseline='full';fake.start_recovery=Mock(return_value=True)
        with patch('urban_drone.mission.candidates') as candidates:
            Mission.plan(fake)
        fake.start_recovery.assert_called_once()
        candidates.assert_not_called()
        self.assertFalse(fake.recovery_requested)


if __name__=='__main__':unittest.main()
