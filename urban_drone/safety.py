"""Directional braking and revalidated execution corridors from sensed data.

No obstacle identities, scenario names, simulator geometry or mode exemptions.
The braking model is an outer-loop estimate, not a PX4 dynamics guarantee.
"""

from dataclasses import dataclass
import numpy as np

from .clearance import (EXECUTION_CLEARANCE,CONTACT_CLEARANCE,REACTION_TIME,
                        BRAKING_XY,BRAKING_Z,PREFERRED_CLEARANCE)
from .planning import sample_path,length
from .recovery import separating_segment,altitude_safe


def directional_deceleration(directions):
    u=np.asarray(directions,dtype=float)
    u=u/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-9)
    # Harmonic blend keeps the declared horizontal/vertical tuning explicit.
    inverse=np.sum(u[...,:2]**2,axis=-1)/BRAKING_XY+u[...,2]**2/BRAKING_Z
    return 1./np.maximum(inverse,1e-9)


def braking_speed(points,directions,observation):
    if not hasattr(observation,'collision_distances'):
        return np.full(len(points),np.inf)
    room=observation.collision_distances(points,directions)
    a=directional_deceleration(directions)
    # room >= speed * reaction_time + speed**2 / (2*a)
    return np.maximum(np.sqrt((a*REACTION_TIME)**2+2*a*room)-a*REACTION_TIME,0.)


def stopping_path(position,velocity):
    v=np.asarray(velocity,dtype=float);speed=float(np.linalg.norm(v))
    if speed<.03:return np.asarray(position,dtype=float)[None]
    distance=speed*REACTION_TIME+speed**2/(2*float(directional_deceleration(v)))
    return sample_path([position,np.asarray(position)+v/speed*distance],.05)


def remaining_path(position,path,index=0):
    """Join the next vertex after the closest segment projection, never +6.

    Paths are densely sampled. Every new connector is checked by the caller,
    including drift outside a bend; projection is not permission to cut it.
    """
    p=np.asarray(path,dtype=float)
    if len(p)<2:return np.asarray(position)[None]
    begin=min(index,len(p)-2)
    edges=np.diff(p[begin:],axis=0)
    fraction=np.clip(np.sum((position-p[begin:-1])*edges,axis=1)/
                     np.maximum(np.sum(edges**2,axis=1),1e-12),0,1)
    projected=p[begin:-1]+fraction[:,None]*edges
    nearest=begin+int(np.argmin(np.linalg.norm(projected-position,axis=1)))
    return np.vstack((position,p[nearest+1:]))


def forward_remainder(position,path,index,forward):
    """Reconnect from LIVE pose to a future vertex, never a passed ramp/turn."""
    remainder=remaining_path(np.asarray(position),path,index)
    f=np.asarray(forward[:2],dtype=float);f=f/max(np.linalg.norm(f),1e-9)
    candidates=np.flatnonzero((remainder[1:,:2]-np.asarray(position)[:2])@f>=-1e-6)
    if not len(candidates):return np.asarray(position,dtype=float)[None]
    return np.vstack((position,remainder[int(candidates[0])+1:]))


def forward_target(position,target,forward):
    """Keep sideways/upward correction, remove only the backward component.

    The caller must recheck the resulting 3-D segment; clipping is not a safety
    exemption. Terminal arrival correction and authorized escape are separate.
    """
    p=np.asarray(position);delta=np.asarray(target)-p
    f=np.asarray(forward[:2],dtype=float);f=f/max(np.linalg.norm(f),1e-9)
    progress=float(delta[:2]@f)
    if progress<0:delta=delta.copy();delta[:2]-=f*progress
    return p+delta


class PositionBrake:
    """Brake at the live pose while coasting; latch only after settling.

    A fixed old position commands a return after overshoot. This anchor follows
    residual motion and never pulls backwards along the mission direction.
    """
    def __init__(self):self.reset()

    def reset(self):self.anchor=None;self.settling_since=None

    def update(self,position,velocity,forward,now):
        p=np.asarray(position,dtype=float)
        if self.anchor is None or np.linalg.norm(velocity)>.10:
            self.anchor=p.copy();self.settling_since=None
        elif self.settling_since is None:
            self.settling_since=now;self.anchor=p.copy()
        elif now-self.settling_since<.3:
            self.anchor=p.copy()
        self.anchor=forward_target(p,self.anchor,forward)
        return self.anchor.copy()


def path_prefix(path,distance=2.):
    p=np.asarray(path,dtype=float)
    if len(p)<2:return p.copy()
    sizes=np.linalg.norm(np.diff(p,axis=0),axis=1)
    cumulative=np.cumsum(sizes)
    hits=np.flatnonzero(cumulative>=distance)
    if not len(hits):return p.copy()
    i=int(hits[0]);before=cumulative[i-1] if i else 0.
    end=p[i]+(p[i+1]-p[i])*(distance-before)/max(sizes[i],1e-9)
    return np.vstack((p[:i+1],end))


def corridor_valid(path,rejoin,observation,prior,clearance=EXECUTION_CLEARANCE):
    """A complete old route may use execution reserve, never search inflation.

    Check geometry along the ENTIRE remainder and recent sensing on the
    immediate prefix. The control guard checks each actual lead and stop path.
    A one-point recovery brake is never a complete corridor.
    """
    if rejoin is None or length(path)<=.5 or np.linalg.norm(path[-1]-rejoin)>=.25:
        return False
    samples=sample_path(path,.08)
    prefix=sample_path(path_prefix(path,.5),.08)
    return bool(altitude_safe(samples,prior) and prior.safe(samples,prior.execution_inflation) and
                observation.safe(samples,clearance) and
                observation.observed_free(prefix,margin=.4).all())


@dataclass(frozen=True)
class MotionDecision:
    allowed: bool
    state: str
    reason: str
    stop_distance: float


def check_motion(position,target,velocity,observation,prior,recovering=False):
    """Separate geometric contact, visibility and velocity-dependent threat."""
    segment=sample_path([position,target],.05)
    stop=stopping_path(position,velocity)
    stop_distance=length(stop)
    if not altitude_safe(segment,prior) or not altitude_safe(stop,prior):
        return MotionDecision(False,'BRAKE','altitude',stop_distance)
    if not observation.observed_free(segment,margin=.4).all():
        return MotionDecision(False,'BRAKE','unobserved',stop_distance)
    ordinary=(prior.safe(segment,prior.execution_inflation) and
              observation.safe(segment,EXECUTION_CLEARANCE))
    escape=recovering and separating_segment(segment,observation,prior)
    if not ordinary and not escape:
        return MotionDecision(False,'BRAKE','occupied',stop_distance)
    if len(stop)>1:
        stop_clear=(prior.safe(stop,prior.execution_inflation) and
                    observation.safe(stop,EXECUTION_CLEARANCE))
        # In a too-close recovery, outward actual motion can have an initially
        # invalid ordinary floor too. It must satisfy the same escape rules.
        stop_escape=escape and separating_segment(stop,observation,prior)
        if not observation.observed_free(stop,margin=.4).all():
            return MotionDecision(False,'BRAKE','stopping_unobserved',stop_distance)
        if not stop_clear and not stop_escape:
            # If the ordinary stop reserve is already threatened, allow a
            # verified outward correction of the closing component,
            # provided predicted coasting still clears the contact guard.
            corrective=(separating_segment(segment,observation,prior) and
                        prior.safe(stop,prior.execution_inflation) and
                        observation.safe(stop,CONTACT_CLEARANCE))
            if not corrective:
                return MotionDecision(False,'BRAKE','braking_distance',stop_distance)
            return MotionDecision(True,'SEPARATE','outward_braking',stop_distance)
    if escape and not ordinary:
        return MotionDecision(True,'SEPARATE','outward_escape',stop_distance)
    return MotionDecision(True,'EXECUTE','clear',stop_distance)


def maneuver_stage(path,position,observation,strategy):
    """Diagnostic stages from measured 3-D proximity, not obstacle labels."""
    if strategy not in ('OVER','PASS'):return strategy
    p=np.asarray(path)
    near=np.flatnonzero(observation.distances(p)<PREFERRED_CLEARANCE)
    if not len(near):return 'APPROACH'
    nearest=int(np.argmin(np.linalg.norm(p-position,axis=1)))
    if nearest<near[0]:return 'APPROACH'
    if nearest<=near[-1]:return 'TRANSIT'
    return 'EXIT'
