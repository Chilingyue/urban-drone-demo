"""Euclidean A*, swept-volume collision checks, and checked corner smoothing."""

from dataclasses import dataclass, field
import heapq
import itertools
import math
import numpy as np
from .clearance import SEARCH_CLEARANCE,CONNECT_MIN_CLEARANCE


def sample_path(points, spacing=0.15):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
        result.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(result)


def length(points):
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def reference_forward(route,position,lookahead=3.):
    """Forward direction of the initial route, independent of later detours."""
    r=np.asarray(route,dtype=float)[:,:2];p=np.asarray(position)[:2]
    if len(r)<2:return np.array([1.,0.])
    edges=np.diff(r,axis=0);sizes=np.linalg.norm(edges,axis=1)
    fraction=np.clip(np.sum((p-r[:-1])*edges,axis=1)/np.maximum(sizes**2,1e-9),0,1)
    projected=r[:-1]+fraction[:,None]*edges
    index=int(np.argmin(np.linalg.norm(projected-p,axis=1)))
    origin=projected[index];target=origin.copy();remaining=lookahead
    for endpoint in r[index+1:]:
        segment=endpoint-target;distance=float(np.linalg.norm(segment))
        if distance>=remaining:
            target+=segment*remaining/max(distance,1e-9);break
        target=endpoint.copy();remaining-=distance
    direction=target-origin
    if np.linalg.norm(direction)<1e-9:
        valid=np.flatnonzero(sizes>1e-9)
        direction=edges[valid[-1]] if len(valid) else np.array([1.,0.])
    return direction/np.linalg.norm(direction)


def uniform_path(points,spacing=.15):
    """Resample by arc length, including the exact final point."""
    p=np.asarray(points,dtype=float)
    if len(p)<2:return p.copy()
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(distance)>1e-9]
    distance,p=distance[keep],p[keep]
    if len(p)<2:return p.copy()
    targets=np.r_[np.arange(0,distance[-1],spacing),distance[-1]]
    return np.column_stack([np.interp(targets,distance,p[:,i]) for i in range(p.shape[1])])


def astar_grid(blocked, start, goal, resolution=1.0, penalty=None, moves=None):
    """8-neighbor search; Euclidean h, nonnegative edge costs, no corner cutting.

    ``moves`` can restrict the lattice (for example to forward-only local
    searches) without changing the heuristic or corner-cutting guarantees.
    """
    blocked = np.asarray(blocked, dtype=bool)
    if blocked[start] or blocked[goal]:
        return None
    queue, counter = [], itertools.count()
    heuristic = lambda c: math.dist(c, goal) * resolution
    heapq.heappush(queue, (heuristic(start), next(counter), 0.0, start))
    costs, parents = {start: 0.0}, {}
    if moves is None:
        moves = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    moves = list(moves)
    while queue:
        _, _, cost, cur = heapq.heappop(queue)
        if cost > costs[cur] + 1e-9:
            continue
        if cur == goal:
            path = [cur]
            while cur in parents:
                cur = parents[cur]
                path.append(cur)
            return path[::-1]
        for di, dj in moves:
            nxt = cur[0] + di, cur[1] + dj
            if not (0 <= nxt[0] < blocked.shape[0] and 0 <= nxt[1] < blocked.shape[1]):
                continue
            if blocked[nxt] or (di and dj and (blocked[cur[0] + di, cur[1]] or
                                               blocked[cur[0], cur[1] + dj])):
                continue
            extra = 0.0 if penalty is None else float(penalty[nxt])
            if extra < 0:
                raise ValueError('A* penalties must be nonnegative')
            new = cost + math.hypot(di, dj) * resolution + extra
            if new < costs.get(nxt, math.inf):
                costs[nxt], parents[nxt] = new, cur
                heapq.heappush(queue, (new + heuristic(nxt), next(counter), new, nxt))
    return None


def shortcut(points, safe):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points.copy()
    result, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not safe(sample_path([points[i], points[j]])):
            j -= 1
        result.append(points[j])
        i = j
    return np.asarray(result)


def smooth(points, safe, radius=0.65):
    """Round each corner with a quadratic Bezier; retain unsafe corners as-is."""
    points = shortcut(points, safe)
    if len(points) < 3:
        return sample_path(points)
    result = [points[0]]
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        d1, d2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        r = min(radius, d1 * 0.3, d2 * 0.3)
        if r < 1e-4:
            result.append(b)
            continue
        entry, exit = b + (a - b) * r / d1, b + (c - b) * r / d2
        t = np.linspace(0, 1, 12)[:, None]
        curve = (1 - t) ** 2 * entry + 2 * (1 - t) * t * b + t ** 2 * exit
        if safe(sample_path([result[-1], entry])) and safe(curve):
            result.extend(curve)
        else:
            result.append(b)
    result.append(points[-1])
    return sample_path(result)


def arc_smooth(points,safe,radius=2.0,shortcut_first=True):
    """Collision-checked tangent circular fillets in each corner's 3-D plane.

    The middle polyline vertex is a corner to round, not a mandatory point on
    the circle. Radius is limited by adjacent segment lengths and measured
    free space. Collinear / reversing triples remain finite straight segments.
    """
    points=shortcut(points,safe) if shortcut_first else np.asarray(points,dtype=float)
    if len(points)<3:return uniform_path(points)
    result=[points[0]]
    for a,b,c in zip(points[:-2],points[1:-1],points[2:]):
        incoming,outgoing=b-a,c-b
        d1,d2=float(np.linalg.norm(incoming)),float(np.linalg.norm(outgoing))
        if min(d1,d2)<1e-6:
            continue
        u,v=incoming/d1,outgoing/d2
        angle=math.acos(float(np.clip(u@v,-1,1)))
        normal=np.cross(u,v);size=float(np.linalg.norm(normal))
        if angle<math.radians(1) or angle>math.radians(175) or size<1e-6:
            result.append(b);continue
        normal/=size
        largest=min(radius,.35*min(d1,d2)/math.tan(angle/2))
        accepted=False
        for r in [largest,largest*.75,largest*.5,largest*.25]:
            if r<.12:continue
            trim=r*math.tan(angle/2)
            entry=b-u*trim
            center=entry+r*np.cross(normal,u)
            radial=entry-center
            count=max(4,math.ceil(r*angle/.10),math.ceil(angle/math.radians(3)))
            t=np.linspace(0,angle,count+1)[:,None]
            curve=center+np.cos(t)*radial+np.sin(t)*np.cross(normal,radial)
            if safe(sample_path([result[-1],entry])) and safe(curve):
                result.extend(curve);accepted=True;break
        if not accepted:result.append(b)
    result.append(points[-1])
    return uniform_path(result)


class PriorMap:
    """Reads only public building footprints, never scene truth."""

    def __init__(self, prior):
        self.prior = prior
        self.bounds = prior['bounds']
        self.resolution = prior['grid_resolution']
        self.execution_inflation = prior['vehicle_radius'] + prior['safety_margin']
        self.inflation = self.execution_inflation + prior.get('tracking_margin', 0.3)
        self.boxes = prior['known_buildings']
        self.dimensions = 3 if prior.get('height_aware', False) else 2
        self.centers = np.array([b['center'][:self.dimensions] for b in self.boxes], dtype=float).reshape(-1, self.dimensions)
        self.halves = np.array([b['size'][:self.dimensions] for b in self.boxes], dtype=float).reshape(-1, self.dimensions)/2

    def point_clear(self, points, clearance=None):
        """Batch point checks, including roof clearance in height-aware scenes."""
        p = np.asarray(points, dtype=float)
        inflation = self.inflation if clearance is None else clearance
        x0, x1, y0, y1 = self.bounds
        clear = ((p[:, 0] >= x0+inflation) & (p[:, 0] <= x1-inflation) &
                 (p[:, 1] >= y0+inflation) & (p[:, 1] <= y1-inflation))
        if len(p) == 0:
            return clear
        coords = p[:, :self.dimensions]
        low, high = coords.min(axis=0), coords.max(axis=0)
        for c, half in zip(self.centers, self.halves+inflation):
            if np.any(high < c-half) or np.any(low > c+half):
                continue
            clear &= ~np.all(np.abs(coords-c) <= half, axis=1)
        return clear

    def safe(self, points, clearance=None):
        p = np.asarray(points)
        inflation = self.inflation if clearance is None else clearance
        if not self.point_clear(p, inflation).all():
            return False
        if len(p) < 2:
            return True
        coords = p[:, :self.dimensions]
        low, high = coords.min(axis=0), coords.max(axis=0)
        for c, half in zip(self.centers, self.halves+inflation):
            if np.any(high < c-half) or np.any(low > c+half):
                continue
            # Exact segment/AABB slab intersection prevents a coarse sample
            # interval from missing a clipped building corner.
            if len(p) > 1:
                a, d = coords[:-1], np.diff(coords, axis=0)
                parallel = abs(d) < 1e-12
                outside = np.any(parallel & ((a < c - half) | (a > c + half)), axis=1)
                safe_d = np.where(parallel, 1.0, d)
                t1, t2 = (c - half - a) / safe_d, (c + half - a) / safe_d
                lo = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=1)
                hi = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=1)
                if np.any(~outside & (np.maximum(lo, 0) <= np.minimum(hi, 1))):
                    return False
        return True

    def _free_start(self, blocked, start_cell, xs, ys, start, occupancy):
        """Nearest executable cell when the live pose is inside the occupied set.

        Global A* blocks cells inside the shared planning clearance. A vehicle
        that is still above the execution limit can therefore be an
        invalid A* start even though it can fly out.  Pick a short, preferably
        forward, connector to a free cell instead of failing the replan.
        """
        r = self.resolution
        cruise = self.prior['cruise_height']
        direction = np.asarray(self.prior['goal'][:2], dtype=float) - np.asarray(start[:2], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm
        limit = int(math.ceil(4.0 / r))
        best = None
        for di in range(-limit, limit + 1):
            for dj in range(-limit, limit + 1):
                cell = (start_cell[0] + di, start_cell[1] + dj)
                if not (0 <= cell[0] < len(xs) and 0 <= cell[1] < len(ys)):
                    continue
                if blocked[cell]:
                    continue
                center = np.array([xs[cell[0]], ys[cell[1]], cruise])
                distance = float(np.linalg.norm(center[:2] - start[:2]))
                if distance < 1e-6 or distance > 4.0:
                    continue
                head = sample_path([start, center], 0.15)
                if not self.safe(head, self.execution_inflation):
                    continue
                if occupancy is not None and not occupancy.safe(head, CONNECT_MIN_CLEARANCE):
                    continue
                backward = 0.75 if float((center[:2] - start[:2]) @ direction) < -0.3 else 0.0
                score = distance + backward
                if best is None or score < best[0]:
                    best = (score, cell, head)
        return (best[1], best[2]) if best is not None else (None, None)

    def route(self, start=None, occupancy=None):
        x0, x1, y0, y1 = self.bounds
        r = self.resolution
        xs, ys = np.arange(x0, x1 + r / 2, r), np.arange(y0, y1 + r / 2, r)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        blocked = np.zeros(gx.shape, dtype=bool)
        for b in self.boxes:
            if self.dimensions == 3 and self.prior['cruise_height'] > b['center'][2]+b['size'][2]/2+self.inflation:
                continue
            cx, cy = b['center'][:2]
            dx, dy = b['size'][:2]
            blocked |= ((abs(gx - cx) <= dx / 2 + self.inflation) &
                        (abs(gy - cy) <= dy / 2 + self.inflation))
        blocked |= ((gx < x0 + self.inflation) | (gx > x1 - self.inflation) |
                    (gy < y0 + self.inflation) | (gy > y1 - self.inflation))
        if occupancy is not None:
            xyz = np.stack((gx, gy, np.full(gx.shape, self.prior['cruise_height'])), axis=-1)
            blocked |= (occupancy.distances(xyz.reshape(-1, 3)) <= SEARCH_CLEARANCE).reshape(gx.shape)
        cell = lambda xy: (round((xy[0] - x0) / r), round((xy[1] - y0) / r))
        start = np.asarray(self.prior['start'] if start is None else start, dtype=float)
        head = None
        origin = cell(start)
        if blocked[origin]:
            origin, head = self._free_start(blocked, origin, xs, ys, start, occupancy)
            if origin is None:
                raise RuntimeError('No global path in prior map')
        path = astar_grid(blocked, origin, cell(self.prior['goal']), r)
        if path is None:
            raise RuntimeError('No global path in prior map')
        raw = np.array([[xs[i], ys[j], self.prior['cruise_height']] for i, j in path])
        safe = self.safe if occupancy is None else lambda p: self.safe(p) and occupancy.safe(p, SEARCH_CLEARANCE)
        if head is None:
            raw[0, :2] = start[:2]
            route = smooth(raw, safe)
            if not safe(route):
                raise RuntimeError('Global path failed continuous collision validation')
        else:
            tail = smooth(raw, safe)
            if not safe(tail):
                raise RuntimeError('Global path failed continuous collision validation')
            route = np.vstack((head[:-1], tail))
        return raw, route


@dataclass
class Candidate:
    strategy: str
    points: np.ndarray
    cost: float
    unknown_fraction: float = 0.0
    side: str | None = None
    metrics: dict = field(default_factory=dict)
    incumbent: bool = False
