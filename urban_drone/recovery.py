"""Short, sensed clearance recovery when the current pose blocks inflated A*."""

import math
import numpy as np

from .planning import sample_path,length
from .clearance import (EXECUTION_CLEARANCE, CONTACT_CLEARANCE, RECOVERY_TRIGGER,
                        RECOVERY_TARGET,MIN_SEPARATION_SLOPE)


def altitude_safe(path,prior):
    p=np.asarray(path)
    return bool(np.all(p[:,2]>=1.) and np.all(p[:,2]<=prior.prior['max_height']))


def separating_segment(path,observation,prior):
    """A sensed outward escape, including an initially-too-close position.

    This exception is only for recovery: never approach another surface or
    cross an unseen region to escape. Ordinary travel keeps its full floor.
    """
    measured=observation.distances(path)
    return bool(len(measured)>1 and np.isfinite(measured).all() and measured.min()>CONTACT_CLEARANCE and
                measured.min()>=measured[0]-.002 and np.all(np.diff(measured)>=-.002) and
                measured[-1]-measured[0]>=max(1e-6,MIN_SEPARATION_SLOPE*length(path)) and
                altitude_safe(path,prior) and prior.safe(path,prior.execution_inflation) and
                observation.observed_free(path,margin=.4).all())

def clearance_recovery(position, observation, prior, forward=None, allow_up=False,allow_down=False,
                       allow_backward=False):
    """Regain measured clearance, preferring forward/lateral or allowed ascent.

    This is a last resort: the mission only calls it after ordinary local
    candidates failed.  A*'s larger planning inflation can exclude an executable
    current pose. This connector never clears occupied cells. Below the normal
    execution floor it permits only sensed outward escape above the contact
    guard; the public prior keeps its ordinary execution margin. Every sample
    must be visible and increase clearance from the measured surface.

    When ``forward`` is given, headings with a non-negative dot product are
    tried first, so clearance is regained without flying back along the route;
    backward headings require explicit positive path-blockage evidence from
    the caller and are tried only after all forward/lateral full AND partial
    escapes fail. Missing visibility or a braking request alone is insufficient.
    """
    start = np.asarray(position, dtype=float)
    initial = float(observation.distances(start[None])[0])
    if not CONTACT_CLEARANCE < initial < RECOVERY_TRIGGER:
        return None
    with observation.lock:
        _, index = observation.tree.query(start)
        away3=start-observation.points[index]
        away = away3[:2]
    headings = [np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0,2*math.pi,24,endpoint=False)]
    if np.linalg.norm(away) > 1e-6:
        headings.insert(0,away/np.linalg.norm(away))
    headings=[np.r_[h,0.] for h in headings]
    if allow_up:
        # At a roof crest, rising can regain clearance while horizontal-only
        # recovery would retreat. Visibility and every clearance check below
        # still apply; unknown overhead space is never treated as free.
        headings.append(np.array([0.,0.,1.]))
    if allow_down:headings.append(np.array([0.,0.,-1.]))
    away_heading=away3/max(np.linalg.norm(away3),1e-9)
    if ((away_heading[2]>=0 and allow_up) or (away_heading[2]<0 and allow_down)):
        headings.insert(0,away_heading)
    direction = None
    if forward is not None:
        direction = np.asarray(forward[:2], dtype=float)
        length = float(np.linalg.norm(direction))
        direction = direction/length if length > 1e-9 else None
    if direction is None:
        groups = [headings]
    else:
        if allow_up:
            # Forward-up options follow the initial route while gaining roof
            # clearance. They must pass the same visibility and safety tests.
            for angle in (30,45,60):
                a=math.radians(angle)
                headings.append(np.r_[direction*math.cos(a),math.sin(a)])
        groups = [[h for h in headings if float(h[:2] @ direction) > 0.05],
                  [h for h in headings if -1e-6 <= float(h[:2] @ direction) <= 0.05],
                  [h for h in headings if float(h[:2] @ direction) < -1e-6]]
    backwards=[]
    if direction is not None:
        backwards=[groups.pop()]
    emergency=initial<=EXECUTION_CLEARANCE
    if emergency:
        # Emergency separation has priority over route progress. A safe
        # outward move must not be rejected just because its START is close.
        away_candidates=[h for h in headings if float(h@away_heading)>.9 and
                         (direction is None or float(h[:2]@direction)>=-1e-6)]
        groups.insert(0,away_candidates)
    goals=[(RECOVERY_TARGET,(.5,1.,1.5,2.,2.5))]
    # Partial forward/outward adjustment takes precedence over reversing just
    # to reach the preferred 1.8 m. This applies above the hard floor too.
    goals.append((max(EXECUTION_CLEARANCE+.10,initial+.10),(.1,.2,.3,.4,.5,1.,1.5,2.,2.5)))
    stages=[(goals,groups)]
    if allow_backward and backwards:stages.append((goals,backwards))
    trials=[(target,distances,group) for stage_goals,stage_groups in stages
            for target,distances in stage_goals for group in stage_groups]
    for target,distances,group in trials:
        best, best_score = None, float('inf')
        for heading in group:
            for distance in distances:
                end=start+heading*distance
                path=sample_path([start,end],.08)
                if not altitude_safe(path,prior):
                    continue
                measured=observation.distances(path)
                floor=CONTACT_CLEARANCE if emergency else EXECUTION_CLEARANCE
                if (measured[-1] < target or
                        np.any(measured <= floor) or
                        measured.min()<initial-.002 or
                        measured[-1]-initial<max(1e-6,MIN_SEPARATION_SLOPE*distance) or
                        np.any(np.diff(measured) < -.002)):
                    continue
                if not prior.safe(path,prior.execution_inflation):
                    continue
                if not prior.safe(end[None]):
                    continue
                if not observation.observed_free(path,margin=.4).all():
                    continue
                score=distance-.05*min(float(measured[-1]),2.)
                if score < best_score:
                    best,best_score=path,score
                break
        if best is not None:
            return best
    return None
