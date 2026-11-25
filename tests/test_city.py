import math
import unittest
import numpy as np

from urban_drone.city import build_city
from urban_drone.evaluate import box_clearance
from urban_drone.perception import RangeMap
from urban_drone.planning import PriorMap, sample_path


class CityTests(unittest.TestCase):
    def test_compact_city_has_exactly_five_unknown_structures(self):
        prior, truth = build_city(case='city_300')
        self.assertEqual(prior['bounds'][1]-prior['bounds'][0],300)
        self.assertEqual(prior['bounds'][3]-prior['bounds'][2],120)
        self.assertLess(len(prior['known_buildings']),40)
        kinds=[s['kind'] for s in truth['demonstration_sites']]
        self.assertEqual(kinds.count('wall'),2)
        self.assertEqual(kinds.count('pillar'),1)
        self.assertEqual(kinds.count('gate'),2)
        self.assertEqual(len(kinds),5)
        self.assertGreaterEqual(truth['public_building_fraction_below_cruise'],.8)
        self.assertTrue(all(b['known'] for b in prior['known_buildings']))

    def test_offset_openings_do_not_contain_the_initial_cruise_line(self):
        prior,truth=build_city(case='city_300')
        for i,site in enumerate(truth['demonstration_sites']):
            if site['kind']!='gate': continue
            parts=[b for b in truth['objects'] if b['name'].startswith(f'unknown_{i}_')]
            point=np.array([*site['center'],prior['cruise_height']+.24])
            self.assertLess(min(box_clearance(point,b) for b in parts),0)
            normal=np.array([-math.sin(site['yaw']),math.cos(site['yaw'])])
            point[:2]+=normal*site['opening_offset']
            self.assertGreater(min(box_clearance(point,b) for b in parts),.75)

    def test_lateral_opening_is_detectable_without_reading_obstacle_metadata(self):
        from scipy.spatial import cKDTree
        occupancy=RangeMap([-20,20,-20,20],aperture_lateral_limit=10,aperture_center_limit=6)
        yy,zz=np.meshgrid(np.arange(-10,10,.2),np.arange(11.5,14.6,.2))
        mask=(yy<1.95)|(yy>6.45)
        occupancy.points=np.column_stack((np.full(mask.sum(),5),yy[mask],zz[mask]))
        occupancy.tree=cKDTree(occupancy.points)
        occupancy.observed_free=lambda p,margin=.35: np.ones(len(p),dtype=bool)
        openings=occupancy.apertures(np.array([0,0,13]),np.array([1,0]))
        self.assertTrue(openings)
        self.assertTrue(any(abs(center[1]-4.2)<.4 for center,width in openings))

    def test_cylinder_clearance_does_not_use_box_corners(self):
        cylinder=dict(center=[0,0,5],size=[4,4,10],shape='cylinder')
        self.assertGreater(box_clearance([1.8,1.8,5],cylinder),.5)
        self.assertAlmostEqual(box_clearance([0,0,5],cylinder),-2)

    def test_city_prior_has_no_unknowns_and_constant_altitude_plan(self):
        prior, truth = build_city()
        x0,x1,y0,y1 = prior['bounds']
        self.assertEqual((x1-x0,y1-y0),(1000,1000))
        self.assertTrue(all(b['known'] for b in prior['known_buildings']))
        self.assertGreater(sum(not b['known'] for b in truth['objects']),3)
        self.assertGreaterEqual(truth['public_building_fraction_below_cruise'],.8)
        _, route = PriorMap(prior).route()
        np.testing.assert_allclose(route[:,2], prior['cruise_height'])
        self.assertTrue(PriorMap(prior).safe(route))
        # Obstacles were actually placed across the public plan, not pre-avoided.
        for index in range(3):
            physical = [b for b in truth['objects'] if b['name'].startswith(f'unknown_{index}_')]
            if index != 2:  # the gate intentionally contains a real free opening
                self.assertLess(min(box_clearance(p,b) for p in route for b in physical),.35)

    def test_roofs_and_sloped_segments_have_real_height_clearance(self):
        prior = dict(bounds=[-20,20,-20,20], grid_resolution=1, vehicle_radius=.35,
                     safety_margin=.4, tracking_margin=.3, height_aware=True,
                     known_buildings=[dict(center=[0,0,5],size=[4,4,10])])
        planner = PriorMap(prior)
        self.assertTrue(planner.safe(np.array([[-5,0,12],[5,0,12]])))
        self.assertFalse(planner.safe(np.array([[-5,0,9],[5,0,9]])))
        # Both endpoints are outside, but the ascending segment cuts the roof.
        self.assertFalse(planner.safe(np.array([[-5,0,7],[5,0,13]])))

    def test_rolling_map_is_bounded_and_preserves_overlap_only(self):
        occupancy = RangeMap([-80,920,-500,500], window=[42,42,30])
        size = occupancy.odds.nbytes+occupancy.seen.nbytes+occupancy.free_seen_at.nbytes
        self.assertLess(size,30_000_000)
        points = np.array([[5.25,0.15,2.25],[-18.75,0.15,2.25]])
        idx=occupancy.indices(points)
        occupancy.odds[tuple(idx.T)] = 2
        occupancy.free_seen_at[tuple(idx.T)] = 12.5
        frozen=occupancy.snapshot()
        occupancy.recenter([8,0,2])
        self.assertEqual(len(occupancy.points),1)
        np.testing.assert_allclose(occupancy.points[0],points[0],atol=1e-9)
        new_idx=occupancy.indices(points[:1])
        self.assertEqual(occupancy.free_seen_at[tuple(new_idx.T)][0],12.5)
        self.assertEqual(frozen.odds[tuple(idx.T)][0],2)
        occupancy.recenter([800,0,31])
        self.assertEqual(len(occupancy.points),0)
        self.assertFalse(occupancy.odds.any())

    def test_rotated_wall_evaluation_uses_its_local_frame(self):
        b=dict(center=[0,0,5],size=[2,10,10],yaw=math.pi/2)
        self.assertLess(box_clearance([4,0,5],b),0)
        self.assertGreater(box_clearance([0,4,5],b),2)


if __name__ == '__main__':
    unittest.main()
