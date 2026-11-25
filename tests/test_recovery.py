import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import cKDTree

from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap
from urban_drone.local import candidates
from urban_drone.recovery import clearance_recovery
from urban_drone.clearance import CONTACT_CLEARANCE


class ClearanceRecoveryTests(unittest.TestCase):
    def observation(self,version):
        data=Path(__file__).parent/'data'
        prior=PriorMap(json.loads((data/'column_clearance_prior.json').read_text()))
        fixture=np.load(data/f'column_clearance_{version}.npz')
        observation=RangeMap(prior.bounds,window=prior.prior['local_window'])
        observation.points=fixture['points']
        observation.tree=cKDTree(observation.points)
        # These captured maps contain occupied evidence, not complete scans.
        # Visibility is tested separately below and always checked live in SITL.
        observation.observed_free=lambda p,margin=.35:np.ones(len(p),dtype=bool)
        return fixture['position'],observation,prior

    def test_captured_inflated_start_has_an_outward_executable_connector(self):
        for version in (222,230):
            start,observation,prior=self.observation(version)
            original=observation.points.copy()
            self.assertFalse(observation.safe(start[None],1.05))
            path=clearance_recovery(start,observation,prior)
            self.assertIsNotNone(path)
            np.testing.assert_allclose(path[0],start)
            np.testing.assert_allclose(path[:,2],start[2])
            distances=observation.distances(path)
            self.assertGreater(distances.min(),.75)
            self.assertGreaterEqual(distances[-1],1.6)
            self.assertGreaterEqual(np.diff(distances).min(),-.005)
            self.assertTrue(prior.safe(path,prior.execution_inflation))
            fixture=np.load(Path(__file__).parent/'data'/f'column_clearance_{version}.npz')
            self.assertTrue(candidates(path[-1],fixture['rejoin'],observation,prior),
                            'Normal planning must become feasible after recovery')
            np.testing.assert_array_equal(observation.points,original)

    def test_unobserved_connector_is_not_executed(self):
        start,observation,prior=self.observation(222)
        observation.observed_free=lambda p,margin=.35:np.zeros(len(p),dtype=bool)
        self.assertIsNone(clearance_recovery(start,observation,prior))

    def test_contact_guard_is_never_relaxed(self):
        start,observation,prior=self.observation(234)
        # An outward escape may start inside the ordinary stop floor, but
        # must never start inside the contact guard.
        distance,index=observation.tree.query(start)
        nearest=observation.points[index]
        start=nearest+(start-nearest)*(CONTACT_CLEARANCE-.01)/distance
        self.assertLess(observation.distances(start[None])[0],CONTACT_CLEARANCE)
        self.assertIsNone(clearance_recovery(start,observation,prior))


if __name__=='__main__':
    unittest.main()
