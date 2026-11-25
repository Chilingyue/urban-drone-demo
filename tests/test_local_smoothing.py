import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.local import EXECUTION_CLEARANCE, SEARCH_CLEARANCE, candidates
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap, sample_path

DATA = Path(__file__).parent / 'data'


def captured(version):
    prior = PriorMap(json.loads((DATA / 'column_clearance_prior.json').read_text()))
    fixture = np.load(DATA / f'column_clearance_{version}.npz')
    observation = RangeMap(prior.bounds, window=prior.prior['local_window'])
    observation.points = fixture['points']
    observation.tree = cKDTree(observation.points)
    observation.updates = 1
    observation.observed_free = lambda p, margin=.35: np.ones(len(p), dtype=bool)
    return prior, fixture['position'], fixture['rejoin'], observation


class LocalSmoothingTests(unittest.TestCase):
    def test_pose_inside_search_buffer_still_plans_forward(self):
        for version in (222, 230):
            prior, start, rejoin, observation = captured(version)
            self.assertLess(float(observation.distances(start[None])[0]), SEARCH_CLEARANCE)
            diagnostics = {}
            options = candidates(start, rejoin, observation, prior, diagnostics=diagnostics)
            self.assertTrue(options, 'an executable pose must not force a retreat')
            self.assertIsNotNone(diagnostics['connector'])
            forward = (rejoin[:2] - start[:2]) / np.linalg.norm(rejoin[:2] - start[:2])
            for option in options:
                sampled = sample_path(option.points, 0.1)
                np.testing.assert_allclose(sampled[0], start, atol=1e-6)
                self.assertGreater(float(observation.distances(sampled).min()), EXECUTION_CLEARANCE)
                progress = (sampled[:, :2] - start[:2]) @ forward
                self.assertGreaterEqual(float(progress.min()), -0.05,
                                        'a local detour must not move backwards')

    def test_gate_opening_is_crossed_near_its_centre_without_backtracking(self):
        prior_data = json.loads((DATA / 'column_clearance_prior.json').read_text())
        prior_data['known_buildings'] = []
        prior = PriorMap(prior_data)
        observation = RangeMap(prior.bounds,
                               aperture_lateral_limit=prior_data['aperture_lateral_limit'],
                               aperture_center_limit=prior_data['aperture_center_limit'])
        ys = np.arange(-12, 12.01, 0.15)
        zs = np.arange(8, 24.01, 0.25)
        wall_y = ys[(ys <= 2.0) | (ys >= 6.5)]
        yy, zz = np.meshgrid(wall_y, zs)
        points = np.stack((np.full(yy.size, 6.0), yy.ravel(), zz.ravel()), axis=1)
        observation.points = points
        observation.tree = cKDTree(points)
        observation.updates = 1
        observation.observed_free = lambda p, margin=.35: np.ones(len(p), dtype=bool)
        start, end = np.array([0., 0., 13.]), np.array([16., 0., 13.])
        self.assertTrue(observation.apertures(start, np.array([1., 0.])))
        options = candidates(start, end, observation, prior)
        self.assertTrue(options)
        best = options[0]
        sampled = sample_path(best.points, 0.05)
        crossing = np.where((sampled[:-1, 0] < 6) & (sampled[1:, 0] >= 6))[0]
        self.assertTrue(len(crossing), 'the chosen route must use the measured opening')
        i = crossing[0]
        ratio = (6 - sampled[i, 0]) / (sampled[i + 1, 0] - sampled[i, 0])
        crossing = sampled[i] + ratio * (sampled[i + 1] - sampled[i])
        self.assertLess(abs(crossing[1] - 4.25), 1.0,
                        'the opening should be crossed near its centre, not its edge')
        self.assertGreater(float(observation.distances(sampled).min()), 1.3)
        progress = (sampled[:, :2] - start[:2]) @ np.array([1., 0.])
        self.assertGreaterEqual(float(progress.min()), -0.05)

    def test_global_replan_can_leave_an_occupied_start_cell(self):
        prior_data = dict(bounds=[-25, 25, -25, 25], start=[-10, 0], goal=[10, 0],
                          grid_resolution=1.0, vehicle_radius=.35, safety_margin=.4,
                          tracking_margin=.3, cruise_height=5.0, max_height=8.0,
                          height_aware=False, known_buildings=[])
        prior = PriorMap(prior_data)
        observation = RangeMap(prior_data['bounds'])
        # A small occupied patch just behind the pose blocks its grid cell,
        # while the short connector still has execution clearance.
        offsets = np.array([[x, y, 0.] for x in (-.1, 0., .1) for y in (-.1, 0., .1)])
        patch = offsets + np.array([-10.95, 0., 5.])
        observation.points = patch
        observation.tree = cKDTree(patch)
        observation.updates = 1
        start = np.array([-10., 0., 5.])
        self.assertLess(float(observation.distances(start[None])[0]), 0.9)
        _, route = prior.route(start=start, occupancy=observation)
        self.assertGreater(len(route), 1)
        np.testing.assert_allclose(route[0][:2], start[:2], atol=1e-6)
        self.assertGreater(float(observation.distances(route[:5]).min()), 0.75)


if __name__ == '__main__':
    unittest.main()
