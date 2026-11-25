"""Online mission: public prior + lidar observations + estimated FCU pose only."""

import argparse
import copy
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Path as RosPath
from visualization_msgs.msg import Marker

from .planning import Candidate, PriorMap, sample_path, length, reference_forward
from .perception import LidarReceiver, RangeMap
from .local import ManeuverSelector, SEARCH_CLEARANCE, candidates
from .recovery import clearance_recovery, RECOVERY_TRIGGER,separating_segment
from .trajectory import assess_path, speed_limits, approach_speed, pursuit_target, VelocityGuidance
from .clearance import (EXECUTION_CLEARANCE, INCUMBENT_CLEARANCE, PREFERRED_CLEARANCE,
                        MIN_APPROACH_SPEED, VEHICLE_RADIUS, CONTACT_CLEARANCE,
                        REACTION_TIME,BRAKING_XY,BRAKING_Z)
from .safety import (check_motion,remaining_path,forward_remainder,forward_target,PositionBrake,
                     path_prefix,corridor_valid,maneuver_stage)
from .global_policy import (GlobalReplanPolicy,working_remainder,obstruction,eligible_candidate,
                            keeps_search_reserve,evaluate_remaining,could_improve,join_paths,
                            CHECK_INTERVAL,SWITCH_SECONDS)


class Mission(Node):
    def __init__(self, args):
        super().__init__('urban_delivery_mission')
        self.args = args
        # This is intentionally the only environment file opened by this node.
        self.prior_data = json.loads(Path(args.prior).read_text())
        self.prior = PriorMap(self.prior_data)
        self.raw_route, self.global_route = self.prior.route()
        self.initial_route = self.global_route.copy()
        self.map = RangeMap(self.prior_data['bounds'], window=self.prior_data.get('local_window'),
                            aperture_lateral_limit=self.prior_data.get('aperture_lateral_limit',5),
                            aperture_center_limit=self.prior_data.get('aperture_center_limit',2.5))
        self.lidar = LidarReceiver(self.map)
        self.state, self.extended = State(), ExtendedState()
        self.position = None
        self.velocity = np.zeros(3)
        self.guidance = VelocityGuidance()
        self.last_control = time.monotonic()
        self.profile_path = None
        self.profile_limits = None
        self.comparisons = 0
        self.pose_time = 0
        self.phase = 'WAIT'
        self.phase_start = time.monotonic()
        self.started = time.monotonic()
        self.flight_started = None
        self.last_request = 0
        self.active_path = self.global_route.copy()
        self.path_index = 0
        self.global_index = 0
        self.strategy = 'TRACK'
        self.maneuver_selector = ManeuverSelector()
        self.rejoin = None
        self.target = None
        self.complete = False
        self.goal_reached = False
        self.arrival_since = None
        self.failure = None
        self.replans = 0
        self.decisions = []
        self.hold_since = None
        self.hold_target = None
        self.planning_hold = False
        self.recoveries = 0
        self.recovery_context = None
        self.escape_override_active = False
        self.execution_state = None
        self.maneuver_phase = 'TRACK'
        self.recovery_requested = False
        self.recovery_reverse_authorized = False
        self.planning_wait_speed = None
        self.stopper = PositionBrake()
        self.sensor_hold_since = None
        self.last_plan = 0
        self.last_global_replan = 0
        self.global_policy = GlobalReplanPolicy()
        self.reverse_gate = GlobalReplanPolicy()
        self.last_progress = time.monotonic()
        self.progress_anchor = None
        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.event_file = (self.output / 'events.jsonl').open('w', buffering=1)
        self.pose_file = (self.output / 'estimated_trajectory.csv').open('w', buffering=1)
        self.pose_file.write('elapsed,x,y,z,strategy,phase\n')
        self.command_file = (self.output / 'command_trace.csv').open('w',buffering=1)
        self.command_file.write('elapsed,x,y,z,target_x,target_y,target_z,vx,vy,vz,strategy,held\n')
        self.pose_log_time = 0
        self.sp_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.global_pub = self.create_publisher(RosPath, '/urban/global_path', 10)
        self.local_pub = self.create_publisher(RosPath, '/urban/local_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/urban/observations', 10)
        self.create_subscription(State, '/mavros/state', self.on_state, 10)
        self.create_subscription(ExtendedState, '/mavros/extended_state', self.on_extended, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(TwistStamped,'/mavros/local_position/velocity_local',self.on_velocity,
                                 qos_profile_sensor_data)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.plan_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(0.05, self.control, callback_group=self.control_group)
        self.plan_timer = self.create_timer(0.8, self.plan, callback_group=self.plan_group)
        np.savetxt(self.output / 'global_raw.csv', self.raw_route, delimiter=',', header='x,y,z')
        np.savetxt(self.output / 'global_smooth.csv', self.global_route, delimiter=',', header='x,y,z')
        self.event('mission_created', raw_points=len(self.raw_route),
                   route_length=length(self.global_route), perception='gpu_lidar+MAVROS_pose',
                   baseline=args.baseline,
                   clearance_policy=dict(vehicle_radius_m=VEHICLE_RADIUS,
                                         body_stop_m=EXECUTION_CLEARANCE-VEHICLE_RADIUS,
                                         escape_contact_guard_m=CONTACT_CLEARANCE,
                                         execution_m=EXECUTION_CLEARANCE,planning_m=SEARCH_CLEARANCE,
                                         preferred_m=PREFERRED_CLEARANCE,min_approach_speed_m_s=MIN_APPROACH_SPEED))
        self.event('braking_model',reaction_seconds=REACTION_TIME,horizontal_m_s2=BRAKING_XY,
                   vertical_m_s2=BRAKING_Z,identified_from_flight=False)

    def event(self, name, **values):
        record = dict(event=name, **values)
        record.setdefault('elapsed', round(time.monotonic() - self.started, 3))
        with self.log_lock:
            self.event_file.write(json.dumps(record) + '\n')
        if name != 'candidate_comparison':
            self.get_logger().info(json.dumps(record, ensure_ascii=False))

    def on_velocity(self,message):
        v=message.twist.linear
        self.velocity=np.array([v.x,v.y,v.z])

    def on_state(self, message):
        self.state = message

    def on_extended(self, message):
        self.extended = message

    def on_pose(self, message):
        if self.complete:
            return
        p, q = message.pose.position, message.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.pose_time = time.monotonic()
        self.lidar.add_pose(self.position, [q.x, q.y, q.z, q.w])
        if self.pose_time - self.pose_log_time > 0.1:
            self.pose_log_time = self.pose_time
            self.pose_file.write(f'{self.pose_time-self.started:.3f},{p.x},{p.y},{p.z},'
                                 f'{self.strategy},{self.phase}\n')

    def change_phase(self, phase):
        self.phase, self.phase_start = phase, time.monotonic()
        self.event('phase', phase=phase)

    def request_mode(self, mode):
        if time.monotonic() - self.last_request < 2 or not self.mode_client.service_is_ready():
            return
        request = SetMode.Request()
        request.custom_mode = mode
        future = self.mode_client.call_async(request)
        self.last_request = time.monotonic()
        future.add_done_callback(lambda f: self.event('mode_response', requested=mode,
                                 sent=f.result().mode_sent) if not f.exception() else None)

    def setpoint(self, target):
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(float, target)
        # The scanner is omnidirectional; keep the nose east for a stable view.
        message.pose.orientation.w = 1.0
        self.sp_pub.publish(message)

    def control(self):
        now, position = time.monotonic(), self.position
        dt=now-self.last_control
        self.last_control=now
        if self.complete or position is None:
            return
        if self.phase not in ('LAND', 'LANDING', 'DONE') and now - self.started > self.args.timeout:
            self.abort('mission_timeout')
        cruise = self.prior_data['cruise_height']
        home = np.array([*self.prior_data['start'], cruise])
        if self.phase == 'WAIT':
            if self.state.connected and self.map.updates >= 3 and now - self.started > 10:
                self.change_phase('STREAM')
        if self.phase in ('STREAM', 'MODE', 'ARM', 'TAKEOFF'):
            self.setpoint(home)
        if self.phase == 'STREAM' and now - self.phase_start >= 5:
            self.change_phase('MODE')
        if self.phase == 'MODE':
            if self.state.mode == 'OFFBOARD':
                self.change_phase('ARM')
            else:
                self.request_mode('OFFBOARD')
        elif self.phase == 'ARM':
            if self.state.armed:
                self.flight_started = now
                self.change_phase('TAKEOFF')
            elif now - self.last_request >= 2 and self.arm_client.service_is_ready():
                request = CommandBool.Request()
                request.value = True
                future = self.arm_client.call_async(request)
                self.last_request = now
                future.add_done_callback(lambda f: self.event('arm_response', success=f.result().success,
                                         result=f.result().result) if not f.exception() else None)
        elif self.phase == 'TAKEOFF' and np.linalg.norm(position - home) < 0.3:
            self.map.reset()  # discard early EKF frame-alignment transients
            self.target = position.copy()
            self.progress_anchor = position.copy()
            self.last_progress = now
            self.guidance.reset()
            self.change_phase('FLY')
        elif self.phase == 'FLY':
            if not self.state.armed:
                self.abort('unexpected_disarm')
                return
            if self.state.mode != 'OFFBOARD':
                self.abort('unexpected_mode_' + self.state.mode)
                return
            stale = now - self.pose_time > 0.5 or now - self.map.last_update > 1.5
            if stale:
                self.arrival_since = None
                forward=reference_forward(self.global_route,position)
                self.hold_target=self.stopper.update(position,self.velocity,forward,now)
                self.setpoint(self.hold_target)
                self.guidance.reset()
                if self.sensor_hold_since is None:
                    self.sensor_hold_since = now
                    self.event('sensor_hold')
                if now - self.sensor_hold_since > 4:
                    self.abort('sensor_or_pose_stale')
                return
            self.sensor_hold_since = None
            observation=self.map.snapshot(copy_evidence=False)
            velocity=self.velocity.copy()
            with self.lock:
                path = self.active_path
                strategy=self.strategy
                planning_hold=self.planning_hold
                if len(path) == 0:
                    self.setpoint(position)
                    return
                # Arc sampling density must not strand the cursor behind the
                # aircraft; compare geometric distance over the remaining path.
                stop = len(path)
                nearest = self.path_index + int(np.argmin(np.linalg.norm(path[self.path_index:stop] - position,
                                                                         axis=1)))
                self.path_index = nearest
                if self.profile_path is not path:
                    self.profile_path=path
                    self.profile_limits=speed_limits(path)
                    self.maneuver_phase='APPROACH' if strategy in ('OVER','PASS') else strategy
                desired,limit=pursuit_target(path,nearest,self.profile_limits,strategy=='RECOVER')
                remaining=remaining_path(position,path,nearest)
                rejoin=None if self.rejoin is None else self.rejoin.copy()
            current_clearance = float(observation.distances(position[None])[0])
            forward=reference_forward(self.global_route,position)
            goal=np.array([*self.prior_data['goal'],cruise])
            terminal_correction=np.linalg.norm(position-goal)<1.
            reverse_allowed=(strategy=='RECOVER' and self.recovery_reverse_authorized)
            if not reverse_allowed and not terminal_correction:
                desired=forward_target(position,desired,forward)
            vector=desired-position
            # Intended motion sets its own speed budget. Actual velocity is
            # checked independently over the full predicted stopping path.
            caps=approach_speed(position[None],vector[None],observation)
            limit=min(limit,float(np.min(caps)))
            if strategy=='RECOVER' and current_clearance<=EXECUTION_CLEARANCE:
                limit=min(limit,MIN_APPROACH_SPEED)
            incumbent_floor=EXECUTION_CLEARANCE if strategy in ('OVER','PASS') else INCUMBENT_CLEARANCE
            continuing=(planning_hold and strategy!='RECOVER' and
                        corridor_valid(remaining,rejoin,observation,self.prior,incumbent_floor))
            waiting=planning_hold and not continuing
            if waiting:
                if self.planning_wait_speed is None:
                    self.planning_wait_speed=min(limit,float(np.linalg.norm(velocity)))
                self.planning_wait_speed=max(0.,self.planning_wait_speed-BRAKING_XY*float(np.clip(dt,.01,.15)))
                limit=min(limit,self.planning_wait_speed)
            else:
                self.planning_wait_speed=None
            requested=vector*min(1.,limit/max(float(np.linalg.norm(vector)),1e-9))
            requested[2]=float(np.clip(requested[2],-1.,1.))
            command_velocity=self.guidance.update(requested,dt)
            # Follow the validated arc direction immediately. Carrying a
            # filtered OLD direction across a route replacement can drive
            # outside the safe corridor. Smooth the speed envelope instead.
            wanted_speed=float(np.linalg.norm(requested))
            command_speed=min(float(np.linalg.norm(command_velocity)),wanted_speed)
            command_velocity=requested*(command_speed/max(wanted_speed,1e-9))
            # PX4's position loop tracks a continuously changing, bounded lead
            # point. Normal path changes no longer jump from a 3 m lead to .8 m.
            desired=position+command_velocity
            decision=check_motion(position,desired,velocity,observation,self.prior,
                                  recovering=strategy=='RECOVER')
            escape_override=decision.allowed and decision.state=='SEPARATE'
            if escape_override:
                if not self.escape_override_active:
                    self.event('emergency_separation',clearance=current_clearance,target=desired.tolist())
            self.escape_override_active=escape_override
            # Planning may be comparing alternatives while a valid maneuver
            # already exists. Keep it executable at the ordinary hard floor;
            # a temporary one-point recovery brake cannot qualify.
            held=not decision.allowed or (waiting and (self.planning_wait_speed<.03 or wanted_speed<.03))
            self.recovery_requested=(not decision.allowed and decision.reason in ('occupied','braking_distance'))
            stage=maneuver_stage(path,position,observation,strategy)
            order={'APPROACH':0,'TRANSIT':1,'EXIT':2}
            if stage not in order or order[stage]>=order.get(self.maneuver_phase,0):
                self.maneuver_phase=stage
            slowing=waiting or float(np.linalg.norm(velocity))>limit+.05
            state=(strategy,self.maneuver_phase,'BRAKE' if held else ('DECELERATE' if slowing else decision.state),
                   decision.reason if not decision.allowed else
                   ('replanning_stop' if held else ('replanning_slowdown' if waiting else decision.reason)))
            if state!=self.execution_state:
                self.execution_state=state
                self.event('execution_state',strategy=state[0],stage=state[1],state=state[2],reason=state[3],
                           stop_distance=decision.stop_distance,clearance=current_clearance,
                           continued_corridor=bool(continuing and not held))
            if held:
                self.hold_target=self.stopper.update(position,velocity,forward,now)
                stop_segment=sample_path([position,self.hold_target],.05)
                if (not self.prior.safe(stop_segment,self.prior.execution_inflation) or
                        not observation.safe(stop_segment,EXECUTION_CLEARANCE) or
                        not observation.observed_free(stop_segment,margin=.4).all()):
                    self.hold_target=position.copy()
                    self.stopper.anchor=position.copy()
                desired=self.hold_target.copy()
                self.guidance.reset()
                if self.hold_since is None:
                    self.hold_since = now
                    reason = decision.reason if not decision.allowed else 'replanning_stop'
                    self.event('execution_hold', reason=reason, clearance=current_clearance)
            else:
                self.hold_since = None
                self.hold_target = None
                self.stopper.reset()
            self.target = desired
            self.setpoint(desired)
            self.command_file.write(','.join(map(str,[now-self.started,*position,*desired,
                                   *(desired-position),self.strategy,int(self.hold_target is not None)]))+'\n')
            if np.linalg.norm(position - self.progress_anchor) > 0.5:
                self.last_progress, self.progress_anchor = now, position.copy()
            if now - self.last_progress > 25:
                self.abort('no_progress')
            goal = np.array([*self.prior_data['goal'], cruise])
            if np.linalg.norm(position - goal) < self.prior_data.get('arrival_tolerance',0.6):
                if self.arrival_since is None:
                    self.arrival_since = now
                    self.event('arrival_settling',position=position.tolist())
                if now-self.arrival_since >= self.prior_data.get('arrival_hold_seconds',0):
                    self.goal_reached = True
                    self.event('goal_reached', position=position.tolist())
                    self.change_phase('LAND')
            else:
                self.arrival_since = None
        elif self.phase == 'LAND':
            # Keep the Offboard stream until the FCU confirms AUTO.LAND.
            self.setpoint(position)
            if self.state.mode == 'AUTO.LAND':
                self.change_phase('LANDING')
            elif not self.state.armed:
                self.finish()
            else:
                self.request_mode('AUTO.LAND')
        elif self.phase == 'LANDING':
            if not self.state.armed and self.extended.landed_state == ExtendedState.LANDED_STATE_ON_GROUND:
                self.finish()
        if self.phase in ('LAND', 'LANDING') and now - self.phase_start > self.prior_data.get('landing_timeout', 45):
            self.failure = self.failure or 'landing_timeout'
            self.finish()

    def abort(self, reason):
        if self.phase in ('LAND', 'LANDING', 'DONE'):
            return
        self.failure = reason
        self.event('abort', reason=reason)
        self.change_phase('LAND')

    def publish_paths(self):
        for publisher, points in [(self.global_pub, self.initial_route), (self.local_pub, self.active_path)]:
            message = RosPath()
            message.header.frame_id = 'map'
            message.header.stamp = self.get_clock().now().to_msg()
            for point in points[::2]:
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, point)
                pose.pose.orientation.w = 1.0
                message.poses.append(pose)
            publisher.publish(message)
        with self.map.lock:
            cloud = self.map.points[::3].copy()
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = 'observed', 0, Marker.POINTS, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.15
        marker.color.r, marker.color.g, marker.color.a = 1.0, 0.5, 0.8
        from geometry_msgs.msg import Point
        for p in cloud:
            marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        self.marker_pub.publish(marker)

    def plan(self):
        if self.complete or self.position is None:
            return
        self.publish_paths()
        if self.phase != 'FLY' or self.map.updates < 2:
            return
        position = self.position.copy()
        forward_hint=reference_forward(self.initial_route,position)
        observation = self.map.snapshot()
        route = self.global_route
        stop = min(len(route), self.global_index + 90)
        self.global_index += int(np.argmin(np.linalg.norm(route[self.global_index:stop, :2] - position[:2], axis=1)))
        current_clearance = float(observation.distances(position[None])[0])
        if (getattr(self,'recovery_requested',False) and self.strategy!='RECOVER' and
                self.args.baseline!='global_only' and current_clearance<RECOVERY_TRIGGER):
            # A control-level threat must not wait for route-switch hysteresis.
            if self.start_recovery(position,observation,forward_hint):
                self.global_policy.reset()
                self.recovery_requested=False
                return
        if self.strategy == 'RECOVER':
            self.global_policy.reset()
            target_clearance=float(observation.distances(self.rejoin[None])[0])
            recovered_floor=min(RECOVERY_TRIGGER,max(EXECUTION_CLEARANCE+.05,target_clearance-.05))
            if np.linalg.norm(position-self.rejoin) < .15 and current_clearance>=recovered_floor:
                self.event('clearance_recovered' if current_clearance >= RECOVERY_TRIGGER else 'clearance_recovery_retry',
                           position=position.tolist(), clearance=current_clearance)
                self.resume_after_recovery(position,observation)
                # Keep maneuver_selector.side: recovering clearance is not a
                # reason to abandon a still-feasible committed corridor.
            else:
                with self.lock:
                    # Recovery connectors are straight. Recheck from the live
                    # pose to the endpoint; the nearest sampled point may be
                    # behind the aircraft and would falsely look like retreat.
                    upcoming = sample_path([position,self.rejoin],.08)
                if separating_segment(upcoming,observation,self.prior):
                    return
                self.resume_after_recovery(position,observation)
        maneuver_active = self.strategy != 'TRACK'
        if maneuver_active:
            if self.rejoin is not None and np.linalg.norm(position - self.rejoin) < 0.5:
                self.event('rejoin', strategy=self.strategy, position=position.tolist())
                with self.lock:
                    self.strategy, self.rejoin = 'TRACK', None
                    self.maneuver_selector.reset()
                    self.active_path, self.path_index = route[self.global_index:].copy(), 0
                maneuver_active=False
            else:
                pass  # Compare the remaining route against ALL new families.
        end_index, ahead = self.global_index, 0.0
        while end_index < len(route) - 1 and ahead < 10:
            ahead += np.linalg.norm(route[end_index + 1] - route[end_index])
            end_index += 1
        # A fixed look-ahead can end on an obstacle's visible face. Move the
        # rejoin target farther along the reference rather than planning to it.
        # A point inside a wide, sparsely sampled obstacle can be more than the
        # planning clearance away from every observed point while still being
        # solid; the visibility test rejects those occluded/unknown points.
        def visible_target(point):
            return (observation.safe(point[None], SEARCH_CLEARANCE) and
                    bool(observation.observed_free(point[None], margin=0.4)[0]))

        fixed_end = (maneuver_active and self.rejoin is not None and
                     observation.safe(self.rejoin[None],EXECUTION_CLEARANCE))
        if fixed_end:
            end_index=int(np.argmin(np.linalg.norm(route-self.rejoin,axis=1)))
        while not fixed_end and end_index < len(route) - 1 and not visible_target(route[end_index]):
            ahead += np.linalg.norm(route[end_index + 1] - route[end_index])
            end_index += 1
            if ahead > 16:
                break
        main = np.vstack((position, route[self.global_index:end_index + 1]))
        main_samples = sample_path(main)
        vector = main[-1, :2] - position[:2]
        norm = np.linalg.norm(vector)
        aperture = norm > 1 and observation.aperture_ahead(position, vector / norm)
        blocked = not observation.safe(main_samples, SEARCH_CLEARANCE)
        # Unknown/occluded space along the reference must trigger local sensing
        # based replanning; otherwise the control visibility gate stops the
        # vehicle without any candidate ever being generated.
        near = main_samples[np.linalg.norm(main_samples - position, axis=1) <= 10.0]
        if len(near):
            blocked = blocked or not observation.observed_free(near, margin=0.4).all()
        if not blocked and not aperture and not maneuver_active:
            with self.lock:
                self.active_path, self.path_index = route[self.global_index:].copy(), 0
                self.strategy = 'TRACK'
                self.rejoin = None
                self.maneuver_selector.reset()
                self.planning_hold = False
            self.reverse_gate.reset()
            if self.args.baseline!='global_only':
                self.global_replan(position,observation)
            return
        if self.args.baseline == 'global_only':
            return  # control's execution gate stops instead of colliding
        diagnostics = {}
        map_version = observation.updates
        # Preserve a short, currently executable prefix. Alternatives start at
        # its far end, so selecting OVER instead of PASS cannot turn the
        # immediate setpoint sideways while the aircraft is still committed.
        with self.lock:
            previous_path=self.active_path
            remaining=remaining_path(position,previous_path,self.path_index)
            prefix=path_prefix(remaining,2.)
        reference=route[end_index][:2]-position[:2]
        reference/=max(np.linalg.norm(reference),1e-9)
        if (np.linalg.norm(route[end_index]-position)<3. or not .5<length(prefix)<3. or
                np.min((prefix[:,:2]-position[:2])@reference)<-.05 or
                not observation.safe(prefix,SEARCH_CLEARANCE) or not self.prior.safe(prefix) or
                not observation.observed_free(prefix,margin=.4).all()):
            prefix=None
        comparison_velocity=self.velocity.copy()
        options = candidates(position, route[end_index], observation, self.prior,
                             allow_over=self.args.baseline != 'side_only', diagnostics=diagnostics,
                             velocity=comparison_velocity,
                             committed_prefix=prefix,forward_hint=forward_hint)
        if fixed_end:
            # Sparse lidar visibility at a farther point must not discard an
            # otherwise safe incumbent and induce OVER/SIDE oscillation.
            # The control loop checks the entire actual lead at every tick.
            # The one-point brake path after recovery is not a complete
            # incumbent route to the rejoin target and must not win at cost 0.
            incumbent_floor=EXECUTION_CLEARANCE if self.strategy in ('OVER','PASS') else INCUMBENT_CLEARANCE
            if corridor_valid(remaining,self.rejoin,observation,self.prior,clearance=incumbent_floor):
                cost,metrics=assess_path(remaining,observation,position,route[end_index],comparison_velocity)
                options.append(Candidate(self.strategy,remaining,cost,metrics['unknown_fraction'],
                                         self.maneuver_selector.side,metrics,True))
        diagnostics.update(position=position.tolist(), rejoin=route[end_index].tolist(),
                           previous_strategy=self.strategy,initial_route_forward=forward_hint.tolist())
        if not options or self.strategy != 'TRACK':
            self.event('planning_diagnostics', **diagnostics)
            np.savez_compressed(self.output / f'planning_map_{map_version:05d}.npz',
                                points=observation.points, position=position, rejoin=route[end_index])
        committed_side = self.maneuver_selector.side
        selector_before_submission=copy.deepcopy(self.maneuver_selector)
        chosen, selection_reason = self.maneuver_selector.choose(options, map_version, time.monotonic())
        self.comparisons+=1
        def details(c):
            return dict(strategy=c.strategy,side=c.side,cost=round(c.cost,3),
                        incumbent=c.incumbent,unknown_fraction=round(c.unknown_fraction,3),metrics=c.metrics)
        self.event('candidate_comparison',comparison=self.comparisons,map_version=map_version,
                   position=position.tolist(),rejoin=route[end_index].tolist(),reason=selection_reason,
                   selected=None if chosen is None else details(chosen),candidates=[details(c) for c in options],
                   rejected=diagnostics['rejected'])
        if chosen is not None and chosen.incumbent:
            self.planning_hold=False
            self.reverse_gate.reset()
            self.global_replan(position,observation)
            return
        if chosen is None:
            if selection_reason == 'wait_for_observations':
                self.global_policy.reset()
                self.reverse_gate.reset()
                self.event('maneuver_wait', side=committed_side, map_version=map_version)
                return
            self.event('no_local_candidate')
            if self.args.baseline != 'global_only':
                # Ordinary candidates already failed; only now consider a short
                # retreat, and only if the pose is inside the recovery band.
                forward = forward_hint
                if current_clearance < RECOVERY_TRIGGER and self.start_recovery(
                        position, observation, forward):
                    self.global_policy.reset()
                    return
                replanned = self.global_replan(position,observation,local_failed=True)
                if not replanned:
                    self.planning_hold = True
            return
        live=self.position.copy()
        fresh=self.map.snapshot(copy_evidence=False)
        forward=reference_forward(self.global_route,live)
        proposal=forward_remainder(live,chosen.points,0,forward)
        anchor=np.asarray(diagnostics['planning_origin'])
        join=int(np.argmin(np.linalg.norm(chosen.points-anchor,axis=1)))
        executable=eligible_candidate(proposal,fresh,self.prior)
        reserve=fresh.safe(sample_path(chosen.points[join:]),SEARCH_CLEARANCE)
        if not executable or not reserve:
            self.maneuver_selector=selector_before_submission
            self.planning_hold=True
            samples=sample_path(proposal,.10)
            near=sample_path(path_prefix(proposal,2.),.10)
            visible=fresh.observed_free(near,margin=.4)
            detail=None
            if len(visible) and not visible.all() and hasattr(fresh,'visibility_detail'):
                detail=fresh.visibility_detail(near[np.flatnonzero(~visible)[0]],.4)
            self.event('candidate_submission_rejected',strategy=chosen.strategy,
                       reason='latest_geometry_or_visibility',map_version=fresh.updates,
                       planning_map_version=observation.updates,
                       checks=dict(executable=executable,search_tail_reserve=reserve,
                                   prior_geometry=self.prior.safe(samples,self.prior.execution_inflation),
                                   observed_geometry=fresh.safe(samples,EXECUTION_CLEARANCE),
                                   near_visible=bool(visible.all()),forward_remainder_m=length(proposal)),
                       visibility=detail,
                       observation_clock=fresh.observation_clock() if hasattr(fresh,'observation_clock') else None)
            return
        with self.lock:
            self.active_path, self.path_index = proposal, 0
            self.strategy, self.rejoin = chosen.strategy, route[end_index].copy()
            self.recovery_reverse_authorized=False
            self.reverse_gate.reset()
            self.planning_hold = False
        decision = dict(strategy=chosen.strategy, position=position.tolist(), rejoin=self.rejoin.tolist(),
                        submission_position=live.tolist(),submission_map_version=fresh.updates,
                        diagnostics=diagnostics,
                        side=chosen.side, previous_side=committed_side, selection_reason=selection_reason,
                        candidates=[details(c) for c in options])
        self.decisions.append(decision)
        np.savetxt(self.output / f'local_{len(self.decisions):02d}_{chosen.strategy}.csv',
                   proposal, delimiter=',', header='x,y,z')
        self.event('decision', **decision)
        self.global_replan(position,observation)

    def resume_after_recovery(self,position,observation=None):
        """Revalidate or replan the interrupted maneuver before cruise."""
        with self.lock:
            context=self.recovery_context
            self.recovery_context=None
            self.recovery_reverse_authorized=False
            self.reverse_gate.reset()
            self.strategy=context['strategy'] if context else 'TRACK'
            self.rejoin=context['rejoin'].copy() if context else None
            restored=None
            if observation is not None and context and 'path' in context:
                forward=reference_forward(self.global_route,position)
                remainder=forward_remainder(position,context['path'],context['index'],forward)
                if (self.strategy in ('OVER','PASS') and
                        corridor_valid(remainder,self.rejoin,observation,self.prior)):
                    restored=remainder
            # A stored route is only a proposal. Restore it only after fresh
            # whole-route geometry and near-prefix visibility checks; otherwise
            # brake at this height while a new route is planned.
            self.active_path=np.asarray(position)[None].copy() if restored is None else restored
            self.path_index=0
            self.planning_hold=restored is None
            self.hold_target=np.asarray(position).copy() if restored is None else None
        self.event('recovery_resume',strategy=self.strategy,position=np.asarray(position).tolist(),
                   rejoin=None if self.rejoin is None else self.rejoin.tolist(),
                   awaiting_safe_route=restored is None,corridor_restored=restored is not None)

    def start_recovery(self, position, observation, forward):
        """Recover after a control threat or failure of ordinary candidates."""
        start_clearance=float(observation.distances(position[None])[0])
        emergency=start_clearance<=EXECUTION_CLEARANCE
        upcoming=remaining_path(position,self.active_path,self.path_index)
        if len(upcoming)<2:
            upcoming=remaining_path(position,self.global_route,self.global_index)
        evidence=obstruction(path_prefix(upcoming,2.),observation,self.prior)
        recovery = clearance_recovery(position, observation, self.prior, forward=forward,
                                      allow_up=True,allow_down=True,allow_backward=False)
        confirmed=self.reverse_gate.observe_obstruction(evidence,recovery is None,observation.updates,time.monotonic())
        if recovery is None and confirmed:
            recovery=clearance_recovery(position,observation,self.prior,forward=forward,
                                       allow_up=True,allow_down=True,allow_backward=True)
        if recovery is None:
            self.event('clearance_recovery_wait', clearance=float(observation.distances(position[None])[0]),
                       map_version=observation.updates,path_block_evidence=evidence,
                       backward_confirmation_maps=len(self.reverse_gate.blocked_versions))
            return False
        with self.lock:
            if self.recovery_context is None and self.rejoin is not None:
                self.recovery_context=dict(strategy=self.strategy,rejoin=self.rejoin.copy(),
                                           path=self.active_path.copy(),index=self.path_index)
            self.active_path, self.path_index = recovery, 0
            self.strategy, self.rejoin = 'RECOVER', recovery[-1].copy()
            self.recovery_reverse_authorized=bool(confirmed and
                (recovery[-1,:2]-position[:2])@np.asarray(forward[:2]) < -1e-6)
            self.planning_hold = False
            self.recoveries += 1
        self.event('clearance_recovery', position=position.tolist(), target=recovery[-1].tolist(),
                   start_clearance=float(observation.distances(position[None])[0]),
                   target_clearance=float(observation.distances(recovery[-1:])[0]),
                   retained_side=self.maneuver_selector.side, map_version=observation.updates,
                   initial_route_forward=np.asarray(forward[:2]).tolist(),emergency=emergency,
                   backward_authorized=self.recovery_reverse_authorized,path_block_evidence=evidence)
        np.savetxt(self.output / f'recovery_{self.recoveries:02d}.csv', recovery,
                   delimiter=',', header='x,y,z')
        return True

    def global_replan(self, position, observation=None,local_failed=False):
        """Propose, compare, confirm, then replace; never replace on A* success alone."""
        observation=self.map.snapshot() if observation is None else observation
        now=time.monotonic()
        current=working_remainder(position,self.global_route,self.global_index,
                                  self.active_path,self.path_index,self.rejoin)
        evidence=obstruction(current,observation,self.prior)
        blocked=self.global_policy.observe_obstruction(evidence,local_failed,observation.updates,now)
        # Positive occupancy must persist AND ordinary local/recovery choices
        # must fail. Unknown visibility and the larger search margin do not
        # establish infeasibility.
        if evidence and not blocked:
            self.global_policy.clear_improvement()
            return False
        if not blocked and self.strategy!='TRACK':
            self.global_policy.clear_improvement()
            return False
        if abs(position[2] - self.prior_data['cruise_height']) > 0.5:
            self.global_policy.clear_improvement()
            return False
        if now-self.last_global_replan<CHECK_INTERVAL:
            return False
        goal=np.array([*self.prior_data['goal'],self.prior_data['cruise_height']])
        if length(current)<.5:
            self.global_policy.clear_improvement()
            return False
        if not blocked:
            _,metrics=evaluate_remaining(current,observation,position,goal,self.velocity)
            if not could_improve(metrics,position,goal):
                self.global_policy.clear_improvement()
                return False
        self.last_global_replan=now
        # Preserve a sensed short prefix for normal route improvements. A
        # confirmed blocked path may need to depart immediately instead.
        prefix=path_prefix(current,2.)
        if not eligible_candidate(prefix,observation,self.prior):
            prefix=np.asarray(position)[None]
        anchor=prefix[-1]
        try:
            raw, tail = self.prior.route(start=anchor,occupancy=observation)
        except RuntimeError as error:
            self.global_policy.clear_improvement()
            self.event('global_replan_failed', reason=str(error))
            return False
        proposal=join_paths(prefix,tail)
        # Search may take time while control advances. Compare both remaining
        # paths from the SAME live pose, goal, velocity and NEW map snapshot.
        live=self.position.copy()
        fresh=self.map.snapshot()
        proposal=join_paths(remaining_path(live,proposal))
        current=working_remainder(live,self.global_route,self.global_index,
                                  self.active_path,self.path_index,self.rejoin)
        if (np.linalg.norm(proposal[-1]-goal)>.25 or
                not eligible_candidate(proposal,fresh,self.prior)):
            self.global_policy.clear_improvement()
            self.event('global_replan_rejected',reason='candidate_not_executable',map_version=fresh.updates)
            return False
        if not keeps_search_reserve(raw,tail,fresh,self.prior):
            self.global_policy.clear_improvement()
            self.event('global_replan_rejected',reason='candidate_lost_planning_reserve',map_version=fresh.updates)
            return False
        live_evidence=obstruction(current,fresh,self.prior)
        if live_evidence and not blocked:
            self.global_policy.clear_improvement()
            self.event('global_replan_rejected',reason='unconfirmed_old_route_collision',map_version=fresh.updates)
            return False
        blocked=blocked and live_evidence is not None
        comparison_velocity=self.velocity.copy()
        old_cost,old_metrics=evaluate_remaining(current,fresh,live,goal,comparison_velocity)
        new_cost,new_metrics=evaluate_remaining(proposal,fresh,live,goal,comparison_velocity)
        current_position=self.position.copy()
        current_map=self.map.snapshot(copy_evidence=False)
        proposal=join_paths(remaining_path(current_position,proposal))
        if (not eligible_candidate(proposal,current_map,self.prior) or
                not keeps_search_reserve(raw,tail,current_map,self.prior)):
            self.global_policy.clear_improvement()
            self.event('global_replan_rejected',reason='latest_submission_check',map_version=current_map.updates)
            return False
        verdict=self.global_policy.compare(old_cost,new_cost,fresh.updates,time.monotonic(),blocked)
        self.event('global_replan_comparison',reason=verdict.reason,accepted=verdict.replace,
                   old_cost=old_cost,new_cost=new_cost,switch_cost=SWITCH_SECONDS,
                   gain=verdict.gain,threshold=verdict.threshold,old_metrics=old_metrics,new_metrics=new_metrics,
                   map_version=fresh.updates,obstruction=live_evidence,
                   blocked_maps=len(self.global_policy.blocked_versions),
                   improvement_maps=len(self.global_policy.better_versions))
        if not verdict.replace:return False
        route=proposal
        with self.lock:
            self.global_route, self.global_index = route, 0
            self.active_path, self.path_index = route.copy(), 0
            self.strategy, self.rejoin = 'TRACK', None
            self.maneuver_selector.reset()
            self.planning_hold = False
            self.recovery_requested=False
            self.recovery_reverse_authorized=False
            self.reverse_gate.reset()
            self.replans += 1
            self.global_policy.reset()
        self.event('global_replanned', count=self.replans, length=length(route),reason=verdict.reason)
        np.savetxt(self.output / f'global_replanned_{self.replans:02d}.csv', route,
                   delimiter=',', header='x,y,z')
        return True

    def finish(self):
        self.phase = 'DONE'
        result = dict(success=self.goal_reached and self.failure is None and not self.state.armed,
                      goal_reached=self.goal_reached, failure=self.failure,
                      disarmed=not self.state.armed,
                      landed_state=int(self.extended.landed_state),
                      elapsed=round(time.monotonic() - self.started, 3),
                      strategies=[d['strategy'] for d in self.decisions],
                      replans=self.replans, map_updates=self.map.updates,
                      recoveries=self.recoveries,
                      baseline=self.args.baseline)
        (self.output / 'mission_result.json').write_text(json.dumps(result, indent=2) + '\n')
        with self.map.lock:
            np.save(self.output / 'observed_points.npy', self.map.points)
        self.event('mission_finished', **result)
        self.complete = True

    def close(self):
        self.lidar.close()
        self.event_file.close()
        self.pose_file.close()
        self.command_file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--baseline', choices=['full', 'side_only', 'global_only'], default='full')
    args = p.parse_args()
    rclpy.init()
    node = Mission(args)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.complete:
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        node.event('interrupted')
    finally:
        node.control_timer.cancel()
        node.plan_timer.cancel()
        executor.remove_node(node)
        # Drain work already submitted to the thread pool while its guard
        # condition is still alive (Jazzy's shutdown destroys it first).
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)
        executor.shutdown(timeout_sec=10, wait_for_threads=True)
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
