"""Independent ground-truth evaluator. Never feeds geometry to the mission."""

import argparse
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np


def box_clearance(position, box):
    delta = np.asarray(position) - np.asarray(box['center'])
    if box.get('shape') == 'cylinder':
        q=np.array([np.linalg.norm(delta[:2])-box['size'][0]/2,abs(delta[2])-box['size'][2]/2])
        return float(np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0))
    yaw = box.get('yaw', 0)
    if yaw:
        from math import cos, sin
        delta = np.array([cos(yaw)*delta[0]+sin(yaw)*delta[1],
                          -sin(yaw)*delta[0]+cos(yaw)*delta[1], delta[2]])
    q = abs(delta) - np.asarray(box['size']) / 2
    return float(np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0))


class Evaluator:
    def __init__(self, truth, output):
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
        self.truth = truth
        self.output = Path(output)
        self.started = time.monotonic()
        self.rows = []
        self.centers = np.array([b['center'] for b in truth['objects']]).reshape(-1,3)
        self.halves = np.array([b['size'] for b in truth['objects']]).reshape(-1,3)/2
        angles = np.array([b.get('yaw',0) for b in truth['objects']])
        self.cosines, self.sines = np.cos(angles), np.sin(angles)
        self.cylinders = np.array([b.get('shape')=='cylinder' for b in truth['objects']],dtype=bool)
        self.lock = threading.Lock()
        self.node = Node()
        self.node.subscribe(Pose_V, '/world/urban_delivery/dynamic_pose/info', self.on_pose)

    def on_pose(self, message):
        for pose in message.pose:
            if pose.name == 'urban_x500_0':
                # Include-model pose is the model origin; x500_base's body
                # origin is 0.24 m above it, rotated by the model attitude.
                from .perception import rotation
                q = pose.orientation
                rot = rotation([q.x, q.y, q.z, q.w])
                xyz = np.array([pose.position.x, pose.position.y, pose.position.z]) + rot @ [0, 0, 0.24]
                delta = xyz-self.centers
                local = delta.copy()
                local[:,0] = self.cosines*delta[:,0]+self.sines*delta[:,1]
                local[:,1] = -self.sines*delta[:,0]+self.cosines*delta[:,1]
                q = abs(local)-self.halves
                values = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
                if self.cylinders.any():
                    d=delta[self.cylinders]; half=self.halves[self.cylinders]
                    cq=np.column_stack((np.linalg.norm(d[:,:2],axis=1)-half[:,0],abs(d[:,2])-half[:,2]))
                    values[self.cylinders]=np.linalg.norm(np.maximum(cq,0),axis=1)+np.minimum(cq.max(axis=1),0)
                clearance = float(values.min()) if len(values) else 999
                with self.lock:
                    self.rows.append([time.monotonic()-self.started, *xyz,
                                      clearance-self.truth['vehicle_radius']])

    def finish(self):
        with self.lock:
            data = np.array(self.rows)
        if len(data) == 0:
            result = dict(valid=False, reason='no_ground_truth_samples')
        else:
            np.savetxt(self.output / 'ground_truth_trajectory.csv', data, delimiter=',',
                       header='elapsed,x,y,z,body_sphere_clearance')
            result = dict(valid=True, samples=len(data),
                          path_length=float(np.linalg.norm(np.diff(data[:, 1:4], axis=0), axis=1).sum()),
                          minimum_clearance=float(data[:, 4].min()),
                          envelope_intrusion=bool(np.any(data[:, 4] < 0)),
                          final_position=data[-1, 1:4].tolist(),
                          goal_xy_error=float(np.linalg.norm(data[-1, 1:3] - self.truth['goal'])),
                          collision_metric='body sphere vs static oriented boxes/capped cylinders; ground excluded')
        (self.output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    evaluator = Evaluator(json.loads(Path(args.truth).read_text()), args.output)
    done.wait()
    evaluator.finish()


if __name__ == '__main__':
    main()

