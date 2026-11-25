"""Evidence-gated global route changes; costs are weighted predicted seconds."""

from dataclasses import dataclass
import numpy as np

from .clearance import EXECUTION_CLEARANCE,SEARCH_CLEARANCE
from .planning import length,sample_path
from .safety import remaining_path,path_prefix
from .trajectory import assess_path,SPEED_XY,SPEED_Z
from .recovery import altitude_safe

CHECK_INTERVAL=5.0
CONFIRM_SECONDS=2.0
CONFIRM_MAPS=3
MIN_GAIN_SECONDS=5.0
MIN_GAIN_FRACTION=.15
SWITCH_SECONDS=2.0


def join_paths(*paths):
    p=np.vstack(paths)
    return p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]]


def working_remainder(position,route,route_index,active_path,path_index,rejoin):
    """Compare the local detour PLUS its global tail, to the same final goal."""
    if (rejoin is not None and len(active_path)>1 and
            np.linalg.norm(active_path[-1]-rejoin)<.25):
        local=remaining_path(position,active_path,path_index)
        join=int(np.argmin(np.linalg.norm(route-rejoin,axis=1)))
        return join_paths(local,route[join+1:])
    return join_paths(remaining_path(position,route,route_index))


def obstruction(path,observation,prior):
    """Only positive collision evidence; unknown or search reserve is not proof."""
    p=sample_path(path,.10)
    if not prior.safe(p,prior.execution_inflation):return 'prior_collision'
    if not observation.safe(p,EXECUTION_CLEARANCE):return 'observed_collision'
    return None


def eligible_candidate(path,observation,prior):
    """Whole-route geometry plus near sensing; remote unknown stays costed."""
    if len(path)<2 or length(path)<.5:return False
    p=sample_path(path,.10)
    near=sample_path(path_prefix(path,2.),.10)
    return bool(altitude_safe(p,prior) and prior.safe(p,prior.execution_inflation) and
                observation.safe(p,EXECUTION_CLEARANCE) and
                observation.observed_free(near,margin=.4).all())


def keeps_search_reserve(raw,tail,observation,prior):
    # PriorMap.route may prepend a smaller-clearance start connector. Its
    # raw[0] marks the normal A* origin; only that connector is exempt.
    join=int(np.argmin(np.linalg.norm(tail-raw[0],axis=1)))
    normal=sample_path(tail[join:],.10)
    return bool(prior.safe(normal) and observation.safe(normal,SEARCH_CLEARANCE))


def evaluate_remaining(path,observation,position,goal,velocity):
    return assess_path(path,observation,np.asarray(position),np.asarray(goal),velocity)


def could_improve(metrics,position,goal):
    """Cheap necessary opportunity screen; do not search merely for unseen space."""
    straight=float(np.linalg.norm(np.asarray(goal)[:2]-np.asarray(position)[:2]))
    lower=max(straight/SPEED_XY,abs(float(goal[2]-position[2]))/SPEED_Z)
    avoidable=(metrics['motion_seconds']-lower+metrics['entry_seconds']+
               metrics['clearance_seconds']+metrics['deviation_seconds'])
    return avoidable>=MIN_GAIN_SECONDS+SWITCH_SECONDS


@dataclass(frozen=True)
class GlobalVerdict:
    replace: bool
    reason: str
    gain: float|None=None
    threshold: float|None=None


class GlobalReplanPolicy:
    def __init__(self):self.reset()

    def reset(self):
        self.clear_obstruction()
        self.clear_improvement()

    def clear_obstruction(self):
        self.blocked_since=None;self.blocked_versions=set()

    def clear_improvement(self):
        self.better_since=None;self.better_versions=set()

    def observe_obstruction(self,reason,local_failed,version,now):
        if not reason or not local_failed:
            self.clear_obstruction();return False
        if self.blocked_since is None:self.blocked_since=now
        self.blocked_versions.add(version)
        return now-self.blocked_since>=CONFIRM_SECONDS and len(self.blocked_versions)>=CONFIRM_MAPS

    def compare(self,old_cost,new_cost,version,now,blocked_confirmed=False):
        if not np.isfinite(new_cost):
            self.clear_improvement();return GlobalVerdict(False,'nonfinite_candidate_cost')
        if blocked_confirmed:
            self.clear_improvement();return GlobalVerdict(True,'confirmed_infeasible')
        if not np.isfinite(old_cost):
            self.clear_improvement();return GlobalVerdict(False,'unproven_old_cost')
        gain=old_cost-(new_cost+SWITCH_SECONDS)
        threshold=max(MIN_GAIN_SECONDS,MIN_GAIN_FRACTION*old_cost)
        if gain<threshold:
            self.clear_improvement()
            return GlobalVerdict(False,'insufficient_improvement',gain,threshold)
        if self.better_since is None:self.better_since=now
        self.better_versions.add(version)
        confirmed=(now-self.better_since>=CONFIRM_SECONDS and len(self.better_versions)>=CONFIRM_MAPS)
        return GlobalVerdict(confirmed,'confirmed_improvement' if confirmed else 'confirming_improvement',gain,threshold)
