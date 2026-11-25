"""Geometry-based motion estimates and continuous velocity guidance."""

import numpy as np
from .planning import length, sample_path
from .clearance import EXECUTION_CLEARANCE,PREFERRED_CLEARANCE,MIN_APPROACH_SPEED,VEHICLE_RADIUS
from .safety import braking_speed

SPEED_XY = 1.8
SPEED_Z = 1.0
ACCELERATION = 1.0
LATERAL_ACCELERATION = 0.8


def clearance_speed(clearances):
    """User bands in BODY-SURFACE clearance, not centre distance.

    >=.5 m: nominal speed; .2-.5 m: linear slowing; <=.2 m: .2 m/s.
    Collision/visibility gates and terminal settling can still stop motion.
    """
    outer=np.maximum(np.asarray(clearances)-VEHICLE_RADIUS,0.)
    fraction=np.clip((outer-.2)/.3,0,1)
    return MIN_APPROACH_SPEED+(SPEED_XY-MIN_APPROACH_SPEED)*fraction


def approach_speed(points,directions,occupancy):
    """Only obstacles in the swept forward corridor set approach speed."""
    if hasattr(occupancy,'forward_distances'):
        distances=occupancy.forward_distances(points,directions)
    else:
        distances=occupancy.distances(points)
    # Keep the three proximity bands, with an additional closing-direction
    # braking cap. Non-closing side/roof points do not set that cap.
    return np.minimum(clearance_speed(distances),
                      np.maximum(MIN_APPROACH_SPEED,braking_speed(points,directions,occupancy)))


def pursuit_target(path,nearest,limits,recover=False):
    """Anticipate braking over a wide horizon, follow tight bends closely."""
    def advance(distance):
        k=nearest; travelled=0.
        while k<len(path)-1 and travelled<distance:
            travelled+=np.linalg.norm(path[k+1]-path[k]);k+=1
        return k
    horizon=advance(.4 if recover else 1.8)
    limit=float(np.min(limits[nearest:horizon+1]))
    # With a one-second position lead, a small-radius bend needs a shorter
    # chord. Keep the wider speed horizon so braking still starts early.
    target=advance(.4 if recover else min(1.8,max(.5,limit)))
    return path[target].copy(),limit


def speed_limits(points):
    points = np.asarray(points,dtype=float)
    if len(points)<2:
        # A terminal waypoint still needs position correction after a small
        # overshoot; zero here would freeze the lead point at the current pose.
        return np.full(len(points),SPEED_XY)
    segments=np.diff(points,axis=0)
    ds=np.linalg.norm(segments,axis=1)
    unit=segments/np.maximum(ds[:,None],1e-9)
    xy=np.linalg.norm(unit[:,:2],axis=1)
    caps=np.minimum(SPEED_XY/np.maximum(xy,1e-6),SPEED_Z/np.maximum(abs(unit[:,2]),1e-6))
    speeds=np.r_[caps[0],np.minimum(caps[:-1],caps[1:]),caps[-1]]
    if len(points)>2:
        angles=np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1))
        curvature=angles/np.maximum((ds[:-1]+ds[1:])/2,1e-6)
        speeds[1:-1]=np.minimum(speeds[1:-1],np.sqrt(LATERAL_ACCELERATION/np.maximum(curvature,1e-6)))
    # Anticipate corners instead of waiting until the corner waypoint.
    for i in range(len(speeds)-2,-1,-1):
        speeds[i]=min(speeds[i],np.sqrt(speeds[i+1]**2+2*ACCELERATION*ds[i]))
    return speeds


def assess_path(points,occupancy,start,end,velocity=None):
    """Compare all route families in weighted predicted seconds, before execution.

    Motion time follows common horizontal/vertical speed, acceleration and turn
    limits. Risk and unseen-space weights are explicit heuristics, not energy
    measurements or a claim of globally optimal flight time.
    """
    p=sample_path(points,.25)
    if len(p)<2:
        return 0.,dict(motion_seconds=0.,entry_seconds=0.,clearance_seconds=0.,unknown_seconds=0.,deviation_seconds=0.,length_m=0.)
    ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
    measured=occupancy.distances(p)
    tangents=np.vstack((np.diff(p,axis=0),p[-1]-p[-2]))
    speeds=np.minimum(speed_limits(p),approach_speed(p,tangents,occupancy))
    for i in range(len(speeds)-2,-1,-1):
        speeds[i]=min(speeds[i],np.sqrt(speeds[i+1]**2+2*ACCELERATION*ds[i]))
    first=(p[1]-p[0])/max(ds[0],1e-9)
    velocity=np.zeros(3) if velocity is None else np.asarray(velocity)
    initial=max(0.,float(velocity@first))
    speeds[0]=min(speeds[0],initial)
    for i in range(len(speeds)-1):
        speeds[i+1]=min(speeds[i+1],np.sqrt(speeds[i]**2+2*ACCELERATION*ds[i]))
    dt=2*ds/np.maximum(speeds[:-1]+speeds[1:],.1)
    duration=float(dt.sum())
    entry=float(np.linalg.norm(velocity-first*np.linalg.norm(velocity))/ACCELERATION)
    room=np.clip((PREFERRED_CLEARANCE-measured)/(PREFERRED_CLEARANCE-EXECUTION_CLEARANCE),0,1)
    risk=float(np.sum(dt*(room[:-1]**2+room[1:]**2)/2))
    visible=occupancy.observed_free(p,margin=.2)
    unknown=float(np.sum(dt*(~visible[:-1]|~visible[1:])))
    direction=np.asarray(end[:2],dtype=float)-np.asarray(start[:2],dtype=float);direction/=max(np.linalg.norm(direction),1e-9)
    normal=np.array([-direction[1],direction[0]])
    lateral=abs((p[:,:2]-np.asarray(start[:2]))@normal)
    deviation=float(np.sum(dt*(lateral[:-1]+lateral[1:])/2))
    components=dict(motion_seconds=duration,entry_seconds=entry,clearance_seconds=2.*risk,
                    unknown_seconds=.35*unknown,deviation_seconds=.03*deviation,
                    length_m=length(p),ascent_m=float(np.maximum(np.diff(p[:,2]),0).sum()),
                    minimum_observed_clearance_m=float(np.min(measured)) if np.isfinite(measured).any() else None,
                    unknown_fraction=float(np.mean(~visible)))
    components['clearance_seconds']+=1.5*float(np.max(room)**2)
    cost=sum(components[k] for k in ['motion_seconds','entry_seconds','clearance_seconds','unknown_seconds','deviation_seconds'])
    components['total']=cost
    if hasattr(occupancy,'observation_clock'):
        components['observation_clock']=occupancy.observation_clock()
    return cost,components


class VelocityGuidance:
    """Bound normal command velocity, acceleration and jerk across path changes."""
    def __init__(self,acceleration=1.0,jerk=2.5):
        self.acceleration=acceleration;self.jerk=jerk
        self.reset()

    def reset(self):
        self.velocity=np.zeros(3);self.accel=np.zeros(3)

    @staticmethod
    def bound(vector,limit):
        return vector*min(1.,limit/max(float(np.linalg.norm(vector)),1e-9))

    def update(self,requested,dt):
        dt=float(np.clip(dt,.01,.15))
        wanted=self.bound((np.asarray(requested)-self.velocity)/.5,self.acceleration)
        self.accel+=self.bound(wanted-self.accel,self.jerk*dt)
        self.velocity+=self.accel*dt
        return self.velocity.copy()
