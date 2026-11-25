import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.local import ManeuverSelector, candidates
from urban_drone.perception import RangeMap
from urban_drone.planning import Candidate, PriorMap, sample_path


class SideSwitchTests(unittest.TestCase):
    # This captured narrow corridor's original clearance budget was 1.1 m.
    # Keep testing the removed virtual aperture mask under that same budget;
    # a stricter policy is allowed to exclude a genuinely tighter corridor.
    @patch('urban_drone.local.SEARCH_CLEARANCE',1.1)
    def test_observed_aperture_does_not_close_right_corridor(self):
        data = Path(__file__).parent / 'data'
        fixture = np.load(data / 'side_corridor_scan.npz')
        prior = json.loads((data / 'side_corridor_prior.json').read_text())
        observation = RangeMap(prior['bounds'])
        observation.points = fixture['points']
        observation.tree = cKDTree(observation.points)
        # A visibility fixture, not simulator truth: exercise the aperture
        # branch on the real captured occupancy geometry.
        observation.observed_free = lambda p, margin=0.35: np.ones(len(p), dtype=bool)
        start, end = fixture['position'], fixture['rejoin']
        direction = end[:2]-start[:2]
        direction /= np.linalg.norm(direction)
        self.assertTrue(observation.apertures(start, direction))
        planner = PriorMap(prior)
        options = candidates(start, end, observation, planner)
        right = [c for c in options if c.side == 'SIDE_RIGHT']
        left = [c for c in options if c.side == 'SIDE_LEFT']
        self.assertTrue(right, 'A detected opening must not delete a feasible right detour')
        self.assertTrue(left)
        self.assertLess(min(c.cost for c in right), min(c.cost for c in left))
        for route in right:
            sampled = sample_path(route.points, 0.02)
            self.assertTrue(planner.safe(sampled))
            self.assertTrue(observation.safe(sampled, 1.0))

    def test_retain_feasible_side_despite_cost_noise_or_pass_label(self):
        selector = ManeuverSelector()
        path = np.array([[0., 0., 2.], [1., 0., 2.]])
        right = Candidate('SIDE_RIGHT', path, 10, side='SIDE_RIGHT')
        left = Candidate('SIDE_LEFT', path, 12, side='SIDE_LEFT')
        chosen, _ = selector.choose([right, left], 1, 0)
        self.assertIs(chosen, right)
        right_pass = Candidate('PASS', path, 11, side='SIDE_RIGHT')
        cheaper_left = Candidate('SIDE_LEFT', path, 8, side='SIDE_LEFT')
        chosen, reason = selector.choose([cheaper_left, right_pass], 2, 1)
        self.assertIs(chosen, right_pass)
        self.assertEqual(reason, 'retain_side')

    def test_truly_blocked_side_can_switch_after_distinct_observations(self):
        selector = ManeuverSelector()
        path = np.array([[0., 0., 2.], [1., 0., 2.]])
        right = Candidate('SIDE_RIGHT', path, 10, side='SIDE_RIGHT')
        left = Candidate('SIDE_LEFT', path, 12, side='SIDE_LEFT')
        selector.choose([right], 1, 0)
        self.assertIsNone(selector.choose([left], 2, 1)[0])
        self.assertIsNone(selector.choose([left], 2, 3)[0])  # same map is not another observation
        self.assertIsNone(selector.choose([left], 3, 3.1)[0])
        chosen, reason = selector.choose([left], 4, 3.2)
        self.assertIs(chosen, left)
        self.assertEqual(reason, 'confirmed_corridor_change')

    def test_snapshot_keeps_geometry_and_free_evidence_consistent(self):
        observation = RangeMap([-5, 20, -10, 10])
        point = np.array([[5., 0., 2.]])
        observation.points = point
        observation.tree = cKDTree(point)
        observation.odds[0, 0, 0] = 1
        observation.updates = 3
        frozen = observation.snapshot()
        observation.points = np.array([[9., 0., 2.]])
        observation.tree = cKDTree(observation.points)
        observation.odds[0, 0, 0] = -1
        observation.updates = 4
        self.assertEqual(frozen.updates, 3)
        self.assertEqual(frozen.odds[0, 0, 0], 1)
        self.assertEqual(float(frozen.distances(point)[0]), 0)
        self.assertGreater(float(observation.distances(point)[0]), 3)


if __name__ == '__main__':
    unittest.main()
