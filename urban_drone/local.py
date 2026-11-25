"""Compare geometry-constrained local routes in the observed occupancy map."""

import math
import numpy as np
from .planning import Candidate, arc_smooth, astar_grid, length, sample_path
from .trajectory import assess_path
from .clearance import (EXECUTION_CLEARANCE, SEARCH_CLEARANCE, PREFERRED_CLEARANCE,
                        CONNECT_MIN_CLEARANCE)

CONNECT_MAX = 3.0
CLEARANCE_WEIGHT = 0.8


def forward_connector(start, forward, occupancy, prior, max_distance=CONNECT_MAX):
    """Short forward-or-lateral link from a pose inside the planning buffer.

    A* rejects cells closer than :data:`SEARCH_CLEARANCE`, but execution only
    requires :data:`EXECUTION_CLEARANCE`.  Drifting a few centimetres below the
    planning buffer must not deadlock the search and force a retreat.  This
    connector never moves backwards along the route: it searches headings within
    +/-90 degrees of the route direction, requires execution safety at every
    sample, and ends where A* can start again.  A pure lateral sidestep is
    allowed (it is still forward progress overall); only when no such link is
    executable does the caller recover as a last resort.
    """
    start = np.asarray(start, dtype=float)
    initial = float(occupancy.distances(start[None])[0])
    if initial > SEARCH_CLEARANCE:
        return start, None
    if initial <= EXECUTION_CLEARANCE + 0.01:
        return None, None
    base = math.atan2(float(forward[1]), float(forward[0]))
    best, best_score = None, float('inf')
    for degrees in range(-90, 91, 10):
        angle = base + math.radians(degrees)
        heading = np.array([math.cos(angle), math.sin(angle)])
        for distance in np.arange(0.4, max_distance + 1e-6, 0.2):
            end = start + np.array([*heading, 0.0]) * distance
            path = sample_path([start, end], 0.1)
            measured = occupancy.distances(path)
            if measured.min() <= CONNECT_MIN_CLEARANCE:
                continue
            if measured[-1] <= SEARCH_CLEARANCE:
                continue
            if not prior.safe(path, prior.execution_inflation):
                continue
            if not occupancy.observed_free(path, margin=0.4).all():
                continue
            # Prefer short, straight-ahead links; a wide sideways link is still
            # better than reversing, but should not win on distance alone.
            score = distance + 0.03 * abs(degrees)
            if score < best_score:
                best, best_score = (end, path), score
            break
    return best if best is not None else (None, None)


def candidates(position, rejoin, occupancy, prior, allow_over=True, diagnostics=None, velocity=None, committed_prefix=None,
               forward_hint=None):
    start, end = np.asarray(position, dtype=float), np.asarray(rejoin, dtype=float)
    cruise = prior.prior['cruise_height']
    vector = end[:2] - start[:2]
    distance = float(np.linalg.norm(vector))
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(map_version_start=occupancy.updates,map_version_end=occupancy.updates,
                       rejected=[],openings=[])
    if hasattr(occupancy,'observation_clock'):
        diagnostics['observation_clock']=occupancy.observation_clock()
    if distance < 1:
        return []
    forward = vector / distance
    route_forward=forward.copy()
    normal = np.array([-forward[1], forward[0]])
    safe = lambda p: prior.safe(p) and occupancy.safe(p,clearance=SEARCH_CLEARANCE)
    result = []
    openings = occupancy.apertures(start, forward)
    diagnostics['map_version_start'] = occupancy.updates
    diagnostics['rejected'] = []
    diagnostics['openings'] = [dict(center=c.tolist(), width=w) for c, w in openings]

    # Plan from a short forward link when the current pose is still executable
    # but inside the search buffer.  This keeps the vehicle moving forward
    # instead of retreating to satisfy an inflated start cell.
    if committed_prefix is None:
        anchor, connector = forward_connector(start, forward if forward_hint is None else forward_hint, occupancy, prior)
    else:
        connector=np.asarray(committed_prefix)
        anchor=connector[-1]
    if anchor is None or np.linalg.norm(end[:2] - anchor[:2]) < 1.0:
        anchor, connector = start, None
    origin = anchor
    origin_distance = float(np.linalg.norm(end[:2] - origin[:2]))
    forward = (end[:2] - origin[:2]) / origin_distance
    normal = np.array([-forward[1], forward[0]])
    diagnostics['connector'] = None if connector is None else float(length(connector))
    diagnostics['committed_prefix_m']=0. if committed_prefix is None else length(committed_prefix)
    diagnostics['planning_origin'] = origin.tolist()

    def reject(strategy, reason, points=None,visibility=None):
        record = dict(strategy=strategy, reason=reason)
        if points is not None:
            points = np.asarray(points)
            distances = occupancy.distances(points)
            index = int(np.argmin(distances))
            record.update(min_distance=float(distances[index]), point=points[index].tolist(),
                          prior_safe=prior.safe(points))
        if visibility is not None:record['visibility']=visibility
        diagnostics['rejected'].append(record)

    def add(strategy, points, side=None):
        side = side if side is not None else (strategy if strategy.startswith('SIDE_') else None)
        points = np.asarray(points)
        raw = sample_path(points)
        if not safe(raw):
            reject(strategy, 'raw_path_collision', raw)
            return
        # Shortcuts must preserve the clearance bought by the grid search.
        # Checking only the execution floor can turn a wide detour into
        # a line grazing the corner, leaving too little tracking reserve.
        rounding_clearance=max(SEARCH_CLEARANCE,min(PREFERRED_CLEARANCE,float(occupancy.distances(raw).min())-.03))
        round_safe=lambda p: prior.safe(p) and occupancy.safe(p,clearance=rounding_clearance)
        tail = arc_smooth(points, round_safe, radius=2.0)
        if not safe(tail):
            reject(strategy, 'smoothed_path_collision', tail)
            return
        if connector is None:
            full = tail
        else:
            bridge = sample_path([connector[-1], tail[0]], 0.1)
            if not safe(bridge):
                reject(strategy, 'connector_transition_collision', bridge)
                return
            full = np.vstack((connector[:-1], tail))
        # Local routes must never buy clearance by going backwards.  The
        # forward-only searches already enforce this; this also guards the
        # explicit aperture and direct candidates.
        if float(np.min((full[:, :2] - start[:2]) @ route_forward)) < -0.05:
            reject(strategy, 'backward_path', full)
            return
        # The near prefix must already be sensing-confirmed.  A wide, sparsely
        # sampled obstacle can leave its interior farther than the planning
        # clearance from every observed point, so distances alone would let a
        # candidate aim straight through it and then stall at the visibility
        # gate.  Unknown space farther ahead remains allowed but is costed.
        if occupancy.last_scan is not None:
            near = sample_path(full, 0.25)
            near = near[np.linalg.norm(near - start, axis=1) <= 2.0]
            if len(near):
                visible=occupancy.observed_free(near,margin=.4)
                if not visible.all():
                    point=near[np.flatnonzero(~visible)[0]]
                    detail=(occupancy.visibility_detail(point,.4) if hasattr(occupancy,'visibility_detail')
                            else dict(point=point.tolist()))
                    reject(strategy,'unobserved_prefix',near,visibility=detail)
                    return
        if strategy == 'OVER' and full[:, 2].max() < cruise + 0.4:
            return
        # A small lateral shift is still a valid detour when the direct line
        # grazes an obstacle.  The old 0.7 m floor discarded the only safe
        # candidate at the cylinder and forced a retreat.
        if strategy.startswith('SIDE') and abs((full[:, :2] - start[:2]) @ normal).max() < 0.25:
            return
        # A planar route may use an observed passage. Label what the resulting
        # trajectory actually does; never add virtual obstacles to force a label.
        if side is not None:
            for center, width in openings:
                progress = (full[:, :2] - center[:2]) @ forward
                crossing = np.where((progress[:-1] <= 0) & (progress[1:] >= 0))[0]
                for i in crossing:
                    ratio = -progress[i] / max(progress[i+1]-progress[i], 1e-9)
                    xy = full[i, :2] + ratio * (full[i+1, :2]-full[i, :2])
                    if abs((xy-center[:2]) @ normal) < width/2 - SEARCH_CLEARANCE:
                        strategy = 'PASS'
                        break
                if strategy == 'PASS':
                    break
        cost,metrics=assess_path(full,occupancy,start,end,velocity)
        unknown=metrics['unknown_fraction']
        if any(c.strategy == strategy and c.side == side and abs(c.cost - cost) < 0.05 for c in result):
            return
        result.append(Candidate(strategy, full, cost, unknown, side, metrics))

    direct = sample_path([origin, end])
    if safe(direct):
        add('PASS' if occupancy.aperture_ahead(origin, forward) else 'TRACK', direct)
    for center, width in openings[:2]:
        if float((center[:2] - origin[:2]) @ forward) <= 0.3:
            continue
        offset = np.array([*(forward * 1.3), 0])
        # The centred opening route carries the side of the aperture, so the
        # committed-side selector compares it against the grazing detour
        # instead of silently excluding it for having no side label.
        lateral = float((center[:2] - origin[:2]) @ normal)
        side = None if abs(lateral) < 0.2 else ('SIDE_LEFT' if lateral > 0 else 'SIDE_RIGHT')
        add('PASS', [origin, center - offset, center, center + offset, end], side=side)

    # Two separate 2-D searches compare both sides at normal cruise height.
    # Only forward and lateral grid moves are allowed: a local detour may not
    # reverse along the route.
    r = 0.5
    s_values = np.arange(0, origin_distance + r * 0.51, r)
    if s_values[-1] < origin_distance - 0.1:
        s_values = np.append(s_values, origin_distance)
    lateral_limit = prior.prior.get('local_lateral_limit', 8)
    lateral_values = np.arange(-lateral_limit, lateral_limit + r / 2, r)
    ss, ll = np.meshgrid(s_values, lateral_values, indexing='ij')
    xy = origin[:2] + ss[..., None] * forward + ll[..., None] * normal
    points = np.concatenate((xy, np.full((*ss.shape, 1), cruise)), axis=-1)
    flat = points.reshape(-1, 3)
    clear = occupancy.distances(flat).reshape(ss.shape)
    blocked = clear <= SEARCH_CLEARANCE
    # Known buildings remain unavailable even where the current lidar is occluded.
    blocked |= ~prior.point_clear(flat).reshape(ss.shape)
    penalty = (np.abs(ll) * 0.01 +
               CLEARANCE_WEIGHT * np.clip(PREFERRED_CLEARANCE - clear, 0, None) ** 2)
    middle = len(lateral_values) // 2
    forward_moves = [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1)]
    for strategy, side in [('SIDE_LEFT', 1), ('SIDE_RIGHT', -1)]:
        mask = blocked | (ll * side < -0.1)
        path = astar_grid(mask, (0, middle), (len(s_values) - 1, middle), r,
                          penalty=penalty, moves=forward_moves)
        if path is not None:
            add(strategy, [origin, *[points[i, j] for i, j in path], end])
        else:
            reject(strategy, 'search_no_path')

    # A search in the vertical plane of the route; lateral detours are handled
    # separately above. This is a local 2.5-D strategy, not a global 3-D claim.
    if allow_over:
        z_values = np.arange(cruise, prior.prior['max_height'] + r / 2, r)
        points = np.array([[[*(origin[:2] + s * forward), z] for z in z_values] for s in s_values])
        flat = points.reshape(-1, 3)
        clear = occupancy.distances(flat).reshape(points.shape[:2])
        blocked = clear <= SEARCH_CLEARANCE
        blocked |= ~prior.point_clear(flat).reshape(points.shape[:2])
        penalty = CLEARANCE_WEIGHT * np.clip(PREFERRED_CLEARANCE - clear, 0, None) ** 2
        start_z = int(np.clip(round((start[2] - cruise) / r), 0, len(z_values) - 1))
        path = astar_grid(blocked, (0, start_z), (len(s_values) - 1, 0), r,
                          penalty=penalty, moves=forward_moves)
        if path is not None:
            add('OVER', [origin, *[points[i, j] for i, j in path], end])
        else:
            reject('OVER', 'search_no_path')
    diagnostics['map_version_end'] = occupancy.updates
    return sorted(result, key=lambda c: c.cost)


class ManeuverSelector:
    """Compare every family; charge switching and confirm useful improvements."""

    def __init__(self, confirmation_seconds=1.5, confirmation_frames=3):
        self.confirmation_seconds = confirmation_seconds
        self.confirmation_frames = confirmation_frames
        self.reset()

    def reset(self):
        self.side = None
        self.key = None
        self.unavailable_since = None
        self.unavailable_versions = set()
        self.challenger = None
        self.challenger_since = None
        self.challenger_versions = set()

    @staticmethod
    def route_key(candidate):
        return ('OVER',None) if candidate.strategy=='OVER' else ('PLANAR',candidate.side)

    def commit(self,chosen):
        self.key=self.route_key(chosen)
        self.side=chosen.side
        self.unavailable_since=None;self.unavailable_versions.clear()
        self.challenger=None;self.challenger_since=None;self.challenger_versions.clear()

    def choose(self, options, map_version, now):
        for candidate in options:
            key=self.route_key(candidate)
            switch=0. if candidate.incumbent or self.key is None else 1.
            if self.side is not None and candidate.side is not None and candidate.side!=self.side:
                switch+=2.
            candidate.metrics.update(switch_seconds=switch,selection_total=candidate.cost+switch)
        score=lambda c:c.metrics['selection_total']
        if not options:
            return None,'no_candidate'
        best=min(options,key=score)
        incumbent=next((c for c in options if c.incumbent),None)
        same=[c for c in options if self.route_key(c)==self.key]
        current=incumbent if incumbent is not None else (min(same,key=score) if same else None)
        if current is not None:
            self.unavailable_since=None;self.unavailable_versions.clear()
            improvement=score(current)-score(best)
            threshold=max(1.5,.10*score(current))
            if best is current or improvement<threshold:
                self.challenger=None;self.challenger_versions.clear()
                return current,'retain_current' if incumbent is not None else 'retain_side'
            key=self.route_key(best)
            if self.challenger!=key:
                self.challenger=key;self.challenger_since=now;self.challenger_versions.clear()
            self.challenger_versions.add(map_version)
            if now-self.challenger_since<1.0 or len(self.challenger_versions)<2:
                return current,'confirming_cost_improvement'
            self.commit(best)
            return best,'cost_improvement_confirmed'
        if self.key is not None:
            if self.unavailable_since is None:
                self.unavailable_since = now
            self.unavailable_versions.add(map_version)
            if (now-self.unavailable_since < self.confirmation_seconds or
                    len(self.unavailable_versions) < self.confirmation_frames):
                return None, 'wait_for_observations'
            reason = 'confirmed_corridor_change'
        else:
            reason = 'lowest_cost'
        self.commit(best)
        return best, reason
