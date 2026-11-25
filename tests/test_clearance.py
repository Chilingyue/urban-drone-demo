import json
from pathlib import Path
import unittest
import numpy as np
from urban_drone.clearance import EXECUTION_CLEARANCE,SEARCH_CLEARANCE,PREFERRED_CLEARANCE,INCUMBENT_CLEARANCE
from urban_drone.local import candidates
from urban_drone.planning import PriorMap,sample_path


class PlaneObservation:
    updates=1
    last_scan=None
    def distances(self,p):return abs(np.asarray(p)[:,1])
    def safe(self,p,clearance=SEARCH_CLEARANCE):return bool(np.all(self.distances(p)>clearance))
    def observed_free(self,p,margin=.4):return np.ones(len(p),bool)
    def apertures(self,*a):return []
    def aperture_ahead(self,*a):return False


class ClearanceTests(unittest.TestCase):
    def test_execution_and_new_planning_have_distinct_reserve(self):
        observation=PlaneObservation();points=np.array([[0,.72,2],[1,1.1,2],[2,1.3,2]])
        self.assertTrue(observation.safe(points,EXECUTION_CLEARANCE))
        self.assertFalse(observation.safe(points,SEARCH_CLEARANCE))
        self.assertFalse(observation.safe([[0,EXECUTION_CLEARANCE-.01,2]],EXECUTION_CLEARANCE))
        self.assertGreaterEqual(SEARCH_CLEARANCE-EXECUTION_CLEARANCE,.49)
        self.assertTrue(EXECUTION_CLEARANCE<INCUMBENT_CLEARANCE<SEARCH_CLEARANCE<PREFERRED_CLEARANCE)

    def test_escape_connects_to_a_more_distant_complete_new_path(self):
        prior=PriorMap(dict(bounds=[-5,20,-10,10],start=[0,1.15],goal=[10,3],cruise_height=2.,
             max_height=5.,known_buildings=[],vehicle_radius=.35,safety_margin=.35,tracking_margin=.5,grid_resolution=.5))
        observation=PlaneObservation();diagnostics={}
        start=np.array([0.,1.15,2.]);end=np.array([10.,3.,2.])
        options=candidates(start,end,observation,prior,diagnostics=diagnostics)
        self.assertTrue(options)
        anchor=np.asarray(diagnostics['planning_origin'])
        self.assertGreater(observation.distances(anchor[None])[0],SEARCH_CLEARANCE)
        for candidate in options:
            path=candidate.points
            join=int(np.argmin(np.linalg.norm(path-anchor,axis=1)))
            self.assertGreater(observation.distances(sample_path(path[join:],.02)).min(),SEARCH_CLEARANCE)
            self.assertGreater(observation.distances(sample_path(path,.02)).min(),EXECUTION_CLEARANCE)

    def test_known_building_config_matches_observed_clearance_policy(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('city_300','scenes'):
            cfg=json.loads((root/f'config/{name}.json').read_text())
            execution=cfg['vehicle_radius']+cfg['safety_margin']
            self.assertAlmostEqual(execution,EXECUTION_CLEARANCE)
            self.assertAlmostEqual(execution+cfg['tracking_margin'],SEARCH_CLEARANCE)


if __name__=='__main__':unittest.main()
