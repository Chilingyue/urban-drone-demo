import unittest
from types import SimpleNamespace
import numpy as np
from urban_drone.local import ManeuverSelector,candidates
from urban_drone.planning import Candidate,arc_smooth,sample_path
from urban_drone.trajectory import assess_path,speed_limits,clearance_speed,pursuit_target,VelocityGuidance
from urban_drone.clearance import VEHICLE_RADIUS,MIN_APPROACH_SPEED


class ClearObservation:
    updates=1
    def distances(self,p):return np.full(len(p),5.)
    def observed_free(self,p,margin=.35):return np.ones(len(p),dtype=bool)


class TrajectoryTests(unittest.TestCase):
    def test_proximity_cap_has_requested_floor(self):
        caps=clearance_speed(VEHICLE_RADIUS+np.array([0.,.1,.2,.35,.5,.7]))
        self.assertTrue(np.all(caps>=MIN_APPROACH_SPEED))
        self.assertAlmostEqual(caps[0],.2)
        self.assertGreater(caps[-1],caps[0])

    def test_body_clearance_bands_are_used_for_control_and_cost(self):
        limits=clearance_speed(VEHICLE_RADIUS+np.array([0.,.15,.2,.35,.5,.7,5.]))
        np.testing.assert_allclose(limits,[.2,.2,.2,1.,1.8,1.8,1.8])
        class Tight(ClearObservation):
            def distances(self,p):return np.full(len(p),.65)
        path=[[0,0,2],[10,0,2]]
        _,clear=assess_path(path,ClearObservation(),path[0],path[-1])
        _,tight=assess_path(path,Tight(),path[0],path[-1])
        self.assertGreater(tight['motion_seconds'],clear['motion_seconds'])

    def test_pursuit_shortens_bend_chord_without_losing_braking_horizon(self):
        path=arc_smooth([[0,0,0],[2,0,0],[2,2,0]],lambda p:True,radius=.5,shortcut_first=False)
        limits=speed_limits(path)
        nearest=int(np.argmin(np.linalg.norm(path-[1.5,0,0],axis=1)))
        target,limit=pursuit_target(path,nearest,limits)
        self.assertLess(np.linalg.norm(target-path[nearest]),1.)
        self.assertLess(limit,1.)
        np.testing.assert_allclose(pursuit_target(path,len(path)-1,limits)[0],path[-1])

    def test_single_terminal_waypoint_keeps_position_correction(self):
        self.assertGreater(speed_limits(np.array([[10.,0.,2.]]))[0],0.)
        guidance=VelocityGuidance();position=np.array([10.9,0.,2.]);goal=np.array([10.,0.,2.])
        for _ in range(240):
            position+=guidance.update(goal-position,.05)*.05
        self.assertLess(np.linalg.norm(position-goal),.05)

    def test_near_rejoin_keeps_diagnostics_complete(self):
        diagnostics={}
        self.assertEqual(candidates([0,0,2],[.5,0,2],ClearObservation(),SimpleNamespace(prior={'cruise_height':2}),diagnostics=diagnostics),[])
        self.assertEqual(diagnostics['rejected'],[])
        self.assertEqual(diagnostics['map_version_end'],1)

    def test_circular_fillet_has_constant_radius_and_tangent_entry(self):
        path=arc_smooth([[0,0,0],[5,0,0],[5,5,0]],lambda p:True,radius=1.,shortcut_first=False)
        arc=path[(path[:,0]>=4.-1e-6)&(path[:,1]<=1.+1e-6)]
        np.testing.assert_allclose(np.linalg.norm(arc-np.array([4,1,0]),axis=1),1.,atol=.002)
        np.testing.assert_allclose(path[0],[0,0,0]);np.testing.assert_allclose(path[-1],[5,5,0])
        distances=np.linalg.norm(np.diff(path,axis=0),axis=1)
        self.assertGreater(np.min(distances[:-1]),.14)
        self.assertLessEqual(np.max(distances),.151)
        self.assertGreater(np.linalg.norm(path[1]-path[0]),0)

    def test_circular_fillet_reduces_radius_to_keep_observed_clearance(self):
        safe=lambda p:bool(np.all(np.linalg.norm(np.asarray(p)[:,:2]-[4.6,.4],axis=1)>.25))
        path=arc_smooth([[0,0,0],[5,0,0],[5,5,0]],safe,radius=2.,shortcut_first=False)
        self.assertTrue(safe(sample_path(path,.01)))
        self.assertTrue(np.any((path[:,0]<5)&(path[:,1]>0)))

    def test_vertical_fillet_and_collinear_points_remain_finite(self):
        vertical=arc_smooth([[0,0,2],[5,0,2],[5,0,7]],lambda p:True,radius=1.,shortcut_first=False)
        np.testing.assert_allclose(vertical[:,1],0)
        straight=arc_smooth([[0,0,0],[1,0,0],[2,0,0]],lambda p:True,shortcut_first=False)
        self.assertTrue(np.isfinite(straight).all())
        np.testing.assert_allclose(straight[:,-2:],0)

    def test_turns_and_vertical_speed_have_measurable_time_cost(self):
        observation=ClearObservation();start=np.array([0.,0.,2.]);end=np.array([10.,0.,2.])
        straight,m=assess_path([start,end],observation,start,end,np.array([1.,0.,0.]))
        zigzag,z=assess_path([start,[3,2,2],[6,-2,2],end],observation,start,end,np.array([1.,0.,0.]))
        climb,c=assess_path([start,[3,0,5],[7,0,5],end],observation,start,end,np.array([1.,0.,0.]))
        self.assertLess(straight,zigzag)
        self.assertLess(straight,climb)
        self.assertGreater(z['motion_seconds'],m['motion_seconds'])
        self.assertGreater(c['motion_seconds'],m['motion_seconds'])
        self.assertAlmostEqual(straight,sum(m[k] for k in ['motion_seconds','entry_seconds','clearance_seconds','unknown_seconds','deviation_seconds']))

    def test_fast_corner_is_slowed_before_the_turn(self):
        path=np.array([[0,0,2],[1,0,2],[2,0,2],[2,.2,2],[2,1,2]],dtype=float)
        limits=speed_limits(path)
        self.assertLess(limits[2],1.)
        self.assertLess(limits[1],2.)

    def test_over_can_replace_a_feasible_pass_on_cost(self):
        selector=ManeuverSelector();path=np.array([[0.,0.,2.],[5.,0.,2.]])
        old=Candidate('PASS',path,20,side='SIDE_RIGHT')
        selector.choose([old],1,0)
        old.incumbent=True
        over=Candidate('OVER',path,8)
        self.assertIs(selector.choose([old,over],2,1)[0],old)
        chosen,reason=selector.choose([old,over],3,2.1)
        self.assertIs(chosen,over)
        self.assertEqual(reason,'cost_improvement_confirmed')

    def test_pass_can_replace_over_when_it_is_cheaper(self):
        selector=ManeuverSelector();path=np.array([[0.,0.,2.],[5.,0.,2.]])
        old=Candidate('OVER',path,20);selector.choose([old],1,0);old.incumbent=True
        passage=Candidate('PASS',path,8,side='SIDE_LEFT')
        selector.choose([old,passage],2,1)
        self.assertIs(selector.choose([old,passage],3,2.1)[0],passage)

    def test_missing_over_needs_observation_confirmation_before_switch(self):
        selector=ManeuverSelector();path=np.array([[0.,0.,2.],[5.,0.,2.]])
        selector.choose([Candidate('OVER',path,10)],1,0)
        side=Candidate('SIDE_LEFT',path,20,side='SIDE_LEFT')
        self.assertIsNone(selector.choose([side],2,1)[0])
        self.assertIsNone(selector.choose([side],3,2)[0])
        self.assertIs(selector.choose([side],4,2.6)[0],side)

    def test_velocity_acceleration_and_jerk_are_continuous_across_target_changes(self):
        guidance=VelocityGuidance();previous=np.zeros(3);previous_accel=np.zeros(3);dt=.05
        for target in [np.array([2.,0.,0.])]*60+[np.array([0.,-1.,.5])]*60:
            current=guidance.update(target,dt)
            acceleration=(current-previous)/dt
            self.assertLessEqual(np.linalg.norm(acceleration),1.+1e-8)
            self.assertLessEqual(np.linalg.norm(acceleration-previous_accel)/dt,2.5+1e-8)
            previous,previous_accel=current,acceleration


if __name__=='__main__':unittest.main()
