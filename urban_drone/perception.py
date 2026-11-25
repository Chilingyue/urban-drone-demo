"""Range-only occupancy mapping using MAVROS estimates, not simulator poses."""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
import numpy as np
from scipy.spatial import cKDTree
from .clearance import SEARCH_CLEARANCE,EXECUTION_CLEARANCE


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


@dataclass
class Scan:
    ranges: np.ndarray
    angle_min: float
    angle_step: float
    vertical_min: float
    vertical_step: float
    max_range: float
    received: float
    position: np.ndarray
    quaternion: np.ndarray
    pose_age: float


class RangeMap:
    def __init__(self, bounds, resolution=0.3, window=None, aperture_lateral_limit=5, aperture_center_limit=2.5):
        x0, x1, y0, y1 = bounds
        self.origin = np.array([x0, y0, -0.6])
        self.resolution = resolution
        self.shape = tuple(np.ceil((np.array([x1, y1, 10]) - self.origin) / resolution).astype(int))
        self.window = None if window is None else np.asarray(window, dtype=float)
        if self.window is not None:
            self.shape = tuple(np.ceil(self.window/resolution).astype(int))
            self.origin = np.floor(-self.window/2/resolution)*resolution
        self.odds = np.zeros(self.shape, dtype=np.float32)
        self.seen = np.zeros(self.shape, dtype=bool)
        self.free_seen_at = np.zeros(self.shape, dtype=np.float64)
        self.lock = threading.RLock()
        self.points = np.empty((0, 3))
        self.tree = None
        self.last_scan = None
        self.updates = 0
        self.last_update = 0.0
        self.direction_cache = {}
        self.aperture_lateral_limit = aperture_lateral_limit
        self.aperture_center_limit = aperture_center_limit
        self.evaluation_time = None

    def snapshot(self,copy_evidence=True):
        """Freeze data AND time for planning; lightweight control stays live."""
        with self.lock:
            result = RangeMap.__new__(RangeMap)
            result.evaluation_time = ((self.evaluation_time if self.evaluation_time is not None else time.monotonic())
                                      if copy_evidence else None)
            result.origin = self.origin.copy()
            result.resolution, result.shape = self.resolution, self.shape
            result.window = None if self.window is None else self.window.copy()
            result.aperture_lateral_limit = self.aperture_lateral_limit
            result.aperture_center_limit = self.aperture_center_limit
            result.odds, result.seen = (self.odds.copy(),self.seen.copy()) if copy_evidence else (self.odds,self.seen)
            result.free_seen_at = self.free_seen_at.copy() if copy_evidence else self.free_seen_at
            # integrate replaces these objects rather than modifying them.
            result.points, result.tree = self.points, self.tree
            result.last_scan = self.last_scan
            result.updates, result.last_update = self.updates, self.last_update
            # Control needs a stable geometry/scan view without copying the
            # entire rolling grid at 20 Hz. Shared evidence retains the writer
            # lock; recenter replaces arrays and cannot change this origin.
            result.lock = threading.RLock() if copy_evidence else self.lock
            result.direction_cache = {}
        return result

    def reset(self):
        with self.lock:
            self.odds.fill(0)
            self.seen.fill(False)
            self.free_seen_at.fill(0)
            self.points = np.empty((0, 3))
            self.tree = None
            self.updates = 0


    def indices(self, points):
        idx = np.floor((points - self.origin) / self.resolution).astype(np.int32)
        inside = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        return idx[inside]

    def recenter(self, position):
        """Move a bounded local volume; copy overlap without wrapping evidence."""
        if self.window is None:
            return
        with self.lock:
            center = self.origin+np.asarray(self.shape)*self.resolution/2
            if np.max(abs(np.asarray(position)-center)) < 3.0:
                return
            new_origin = np.floor((np.asarray(position)-self.window/2)/self.resolution)*self.resolution
            shift = np.rint((new_origin-self.origin)/self.resolution).astype(int)
            old_slices, new_slices = [], []
            for size, delta in zip(self.shape, shift):
                old_slices.append(slice(max(0, delta), max(0, min(size, size+delta))))
                new_slices.append(slice(max(0, -delta), max(0, min(size, size-delta))))
            for field in ('odds', 'seen', 'free_seen_at'):
                old = getattr(self, field)
                new = np.zeros_like(old)
                if np.all(abs(shift) < np.asarray(self.shape)):
                    new[tuple(new_slices)] = old[tuple(old_slices)]
                setattr(self, field, new)
            self.origin = new_origin
            occupied = np.argwhere(self.odds >= 0.7)
            self.points = self.origin+(occupied+0.5)*self.resolution
            self.tree = cKDTree(self.points) if len(self.points) else None

    def integrate(self, scan):
        self.recenter(scan.position)
        # world_pose is deliberately absent from Scan. Only estimated MAVROS
        # pose, paired by receive time, and the fixed lidar extrinsic are used.
        rows, cols = scan.ranges.shape
        key = (rows, cols, scan.angle_min, scan.angle_step, scan.vertical_min, scan.vertical_step)
        if key not in self.direction_cache:
            az = scan.angle_min + np.arange(cols) * scan.angle_step
            el = scan.vertical_min + np.arange(rows) * scan.vertical_step
            a, e = np.meshgrid(az, el)
            self.direction_cache[key] = np.stack((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a),
                                                   np.sin(e)), axis=-1).reshape(-1, 3)
        dirs = self.direction_cache[key]
        ranges = scan.ranges.ravel()
        valid = (~np.isnan(ranges)) & (ranges > 0.65)
        distance = np.minimum(ranges[valid], scan.max_range)
        rot = rotation(scan.quaternion)
        origin = scan.position + rot @ np.array([0, 0, 0.3])
        directions = dirs[valid] @ rot.T
        steps = np.arange(0.65, scan.max_range, self.resolution)
        free = origin + directions[:, None, :] * steps[None, :, None]
        free = free[steps[None, :] < distance[:, None] - self.resolution]
        free_idx = self.indices(free)
        if len(free_idx):
            free_idx = np.unique(free_idx, axis=0)
        hit = distance < scan.max_range - 0.1
        hits = origin + directions[hit] * distance[hit, None]
        # The ground is handled by the flight-height constraint. Points on the
        # vehicle and the ground must not become fictitious aerial obstacles.
        hits = hits[hits[:, 2] > 0.35]
        hit_idx = np.unique(self.indices(hits), axis=0)
        with self.lock:
            if len(free_idx):
                f = tuple(free_idx.T)
                self.odds[f] = np.maximum(-2.0, self.odds[f] - 0.6)
                self.seen[f] = True
                self.free_seen_at[f] = scan.received
            if len(hit_idx):
                h = tuple(hit_idx.T)
                self.odds[h] = np.minimum(3.0, self.odds[h] + 1.4)
                self.seen[h] = True
            occupied = np.argwhere(self.odds >= 0.7)
            self.points = self.origin + (occupied + 0.5) * self.resolution
            self.tree = cKDTree(self.points) if len(self.points) else None
            self.last_scan = scan
            self.last_update = time.monotonic()
            self.updates += 1

    def distances(self, points):
        with self.lock:
            if self.tree is None:
                return np.full(len(points), np.inf)
            return self.tree.query(np.asarray(points), workers=1)[0]

    def safe(self, points, clearance=SEARCH_CLEARANCE):
        return bool(np.all(self.distances(points) > clearance))

    def forward_distances(self,points,directions,horizon=3.,contact=False):
        """Nearest observed point in each swept forward tube, in metres.

        Side/rear points outside the safety tube do not throttle tangential or
        outward travel. They still participate in all collision/risk checks.
        """
        p=np.atleast_2d(np.asarray(points,dtype=float))
        u=np.atleast_2d(np.asarray(directions,dtype=float))
        u=np.broadcast_to(u,p.shape)
        u=u/np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-9)
        with self.lock:occupied=self.points
        result=np.full(len(p),np.inf)
        if not len(occupied):return result
        radius=EXECUTION_CLEARANCE
        for start in range(0,len(p),32):
            delta=occupied[None,:,:]-p[start:start+32,None,:]
            progress=np.sum(delta*u[start:start+32,None,:],axis=2)
            squared=np.sum(delta**2,axis=2)
            lateral=np.maximum(squared-progress**2,0.)
            mask=(progress>=0)&(progress<=horizon)&(lateral<=radius**2)
            if contact:
                # First intersection with a sphere inflated by the ordinary
                # execution radius; use along-track room, not slant range.
                values=np.maximum(progress-np.sqrt(np.maximum(radius**2-lateral,0.)),0.)
            else:
                values=np.sqrt(squared)
            result[start:start+32]=np.min(np.where(mask,values,np.inf),axis=1)
        return result

    def collision_distances(self,points,directions):
        """Along-ray distance before entering the 3-D execution envelope."""
        return self.forward_distances(points,directions,contact=True)

    def observed_free(self, points, margin=0.35):
        """Current scan's angular visibility test, including occlusion/range limit."""
        now=time.monotonic() if self.evaluation_time is None else self.evaluation_time
        with self.lock:
            scan = self.last_scan
        if scan is None or now - scan.received > 1.5:
            return np.zeros(len(points), dtype=bool)
        rot = rotation(scan.quaternion)
        origin = scan.position + rot @ np.array([0, 0, 0.3])
        local = (np.asarray(points) - origin) @ rot
        distances = np.linalg.norm(local, axis=1)
        az = np.arctan2(local[:, 1], local[:, 0])
        el = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
        rows, cols = scan.ranges.shape
        ai = np.rint((az - scan.angle_min) / scan.angle_step).astype(int)
        ei = np.rint((el - scan.vertical_min) / scan.vertical_step).astype(int)
        available = np.full(len(points), scan.max_range)
        # Neighbor bins account for finite angular resolution. NaNs and self
        # hits are unobserved, never replaced with an invented maximum range.
        for da, de in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            a = np.clip(ai + da, 0, cols - 1)
            e = np.clip(ei + de, 0, rows - 1)
            r = scan.ranges[e, a]
            r = np.where(np.isnan(r), 0, np.minimum(r, scan.max_range))
            available = np.minimum(available, r)
        angular_free = (distances + margin < available) | (distances < 0.45)
        # Descending rays can be occluded by the drone's own body. Recently
        # measured free voxels remain valid in this static-scene mapper; never
        # infer free space from a self hit or from an unobserved voxel.
        memory_free = np.zeros(len(points), dtype=bool)
        with self.lock:
            # Rolling origin and evidence arrays must be read together; a
            # recenter between indexing and lookup could invent free space.
            idx = np.floor((np.asarray(points) - self.origin) / self.resolution).astype(int)
            inside = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
            valid = tuple(idx[inside].T)
            memory_free[inside] = ((now - self.free_seen_at[valid] < 3.0) &
                                   (self.odds[valid] < -0.3))
        return angular_free | memory_free

    def observation_clock(self):
        wall_now=time.monotonic()
        evaluated=wall_now if self.evaluation_time is None else self.evaluation_time
        with self.lock:scan=self.last_scan
        return dict(clock='live' if self.evaluation_time is None else 'planning_snapshot',
                    evaluation_time=evaluated,scan_received=None if scan is None else scan.received,
                    scan_age_at_evaluation=None if scan is None else evaluated-scan.received,
                    scan_age_now=None if scan is None else wall_now-scan.received)

    def visibility_detail(self,point,margin=.4):
        """Small rejection record; never substitute missing rays with free space."""
        p=np.asarray(point,dtype=float)
        info=dict(point=p.tolist(),margin_m=margin,**self.observation_clock())
        with self.lock:scan=self.last_scan
        if scan is None:
            info.update(reason='no_scan',visible=False);return info
        if info['scan_age_at_evaluation']>1.5:
            info.update(reason='scan_stale',visible=False);return info
        rot=rotation(scan.quaternion)
        origin=scan.position+rot@np.array([0,0,.3])
        local=(p-origin)@rot
        distance=float(np.linalg.norm(local))
        ai=int(np.rint((np.arctan2(local[1],local[0])-scan.angle_min)/scan.angle_step))
        ei=int(np.rint((np.arctan2(local[2],np.hypot(local[0],local[1]))-scan.vertical_min)/scan.vertical_step))
        rows,cols=scan.ranges.shape;ranges=[];available=scan.max_range
        for da,de in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            a=int(np.clip(ai+da,0,cols-1));e=int(np.clip(ei+de,0,rows-1))
            r=float(scan.ranges[e,a])
            ranges.append(dict(azimuth_bin=a,elevation_bin=e,range_m=r if np.isfinite(r) else str(r)))
            available=min(available,0. if np.isnan(r) else min(r,scan.max_range))
        with self.lock:
            idx=np.floor((p-self.origin)/self.resolution).astype(int)
            inside=bool(np.all(idx>=0) and np.all(idx<np.array(self.shape)))
            age=info['evaluation_time']-float(self.free_seen_at[tuple(idx)]) if inside else None
            memory=inside and age<3. and self.odds[tuple(idx)]<-.3
        visible=bool(self.observed_free(p[None],margin)[0])
        info.update(visible=visible,reason='visible' if visible else 'ray_or_margin_unconfirmed',
                    distance_from_lidar_m=distance,available_m=float(available),rays=ranges,
                    recent_free_voxel=bool(memory),free_voxel_age_seconds=age)
        return info

    def apertures(self, position, direction):
        """Find measured gaps between side boundaries; verify visibility through them."""
        with self.lock:
            p = self.points.copy()
        if len(p) < 4:
            return []
        normal = np.array([-direction[1], direction[0]])
        delta = p[:, :2] - np.asarray(position[:2])
        s, lateral = delta @ direction, delta @ normal
        at_altitude = abs(p[:, 2] - position[2]) < 0.6
        openings = []
        # Look as far as the scanner reliably sees.  A 2-8 m window detects a
        # gate only a couple of metres before it, forcing a late, sharp turn
        # into the opening; the earlier detection allows a smooth merge.
        limit = 10.5
        with self.lock:
            scan = self.last_scan
        if scan is not None:
            limit = max(2.0, float(scan.max_range) - 1.0)
        for plane in np.arange(1.5, limit + 1e-6, 0.5):
            mask = at_altitude & (abs(s - plane) < 0.55)
            values = np.sort(lateral[mask & (abs(lateral) < self.aperture_lateral_limit)])
            if len(values) < 4:
                continue
            gaps = np.where(np.diff(values) > 1.8)[0]
            for gap in gaps:
                lower, upper = values[gap], values[gap + 1]
                middle = (lower + upper) / 2
                if abs(middle) > self.aperture_center_limit:
                    continue
                center = np.array([*(np.asarray(position[:2]) + direction * plane + normal * middle),
                                   position[2]])
                probe = np.array([center + np.array([*(direction * offset), 0])
                                  for offset in (-0.7, 0, 0.7)])
                if self.safe(probe, SEARCH_CLEARANCE) and self.observed_free(probe, 0.35).all():
                    openings.append((center, float(upper-lower)))
        return openings

    def aperture_ahead(self, position, direction):
        return bool(self.apertures(position, direction))


class LidarReceiver:
    """Native Gazebo range transport; no ROS/Gazebo bridge installation needed."""

    def __init__(self, occupancy):
        from gz.transport13 import Node
        from gz.msgs10.laserscan_pb2 import LaserScan
        self.map = occupancy
        self.pose_history = deque(maxlen=200)
        self.lock = threading.Lock()
        self.pending = None
        self.running = True
        self.dropped_pose = 0
        self.node = Node()
        self.node.subscribe(LaserScan, '/urban/lidar', self.receive)
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()

    def add_pose(self, position, quaternion):
        with self.lock:
            self.pose_history.append((time.monotonic(), np.array(position), np.array(quaternion)))

    def receive(self, message):
        now = time.monotonic()
        with self.lock:
            if not self.pose_history:
                return
            stamp, position, quaternion = min(self.pose_history, key=lambda v: abs(v[0] - now))
            age = abs(now - stamp)
            if age > 0.2:
                self.dropped_pose += 1
                return
            self.pending = Scan(np.array(message.ranges).reshape(message.vertical_count, message.count),
                                message.angle_min, message.angle_step, message.vertical_angle_min,
                                message.vertical_angle_step, message.range_max, now,
                                position.copy(), quaternion.copy(), age)

    def work(self):
        while self.running:
            with self.lock:
                pending, self.pending = self.pending, None
            if pending is None:
                time.sleep(0.02)
                continue
            self.map.integrate(pending)

    def close(self):
        self.running = False
        self.worker.join(timeout=3)

