import heapq
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from urban_drone.planning import PriorMap, astar_grid, length, sample_path, smooth
from urban_drone.scene import build, generate
from urban_drone.perception import RangeMap, Scan, rotation
from urban_drone.local import candidates


class PlanningTests(unittest.TestCase):
    def test_original_route_forward_uses_local_segment_not_temporary_detour(self):
        from urban_drone.planning import reference_forward
        route=np.array([[0.,0.,13.],[10.,0.,13.],[10.,10.,13.]])
        np.testing.assert_allclose(reference_forward(route,[2.,1.,16.]),[1.,0.])
        np.testing.assert_allclose(reference_forward(route,[11.,5.,16.]),[0.,1.])
        np.testing.assert_allclose(reference_forward(route,[10.,10.,13.]),[0.,1.])

    def test_unknown_geometry_never_enters_prior(self):
        low, low_truth = build('wall_low')
        high, high_truth = build('wall_high')
        self.assertEqual(low, high)
        self.assertNotEqual(low_truth['objects'], high_truth['objects'])
        self.assertFalse(any('unknown_' in b['name'] for b in low['known_buildings']))

    def test_seed_and_gate_have_real_collisions(self):
        self.assertEqual(build('delivery', 5, 1), build('delivery', 5, 1))
        self.assertNotEqual(build('delivery', 5, 1)[1], build('delivery', 6, 1)[1])
        with tempfile.TemporaryDirectory() as path:
            generate(path, 'gate_open')
            root = ET.parse(Path(path) / 'urban_delivery.sdf')
            for name in ['unknown_0_gate_left', 'unknown_0_gate_right',
                         'unknown_0_gate_lintel', 'unknown_0_gate_sill']:
                self.assertIsNotNone(root.find(f'.//model[@name="{name}"]/link/collision'))

    def test_astar_matches_independent_dijkstra(self):
        rng = np.random.default_rng(9)
        for _ in range(15):
            grid = rng.random((12, 12)) < 0.18
            grid[0, 0] = grid[-1, -1] = False
            # Independent graph construction for a cost-optimality regression.
            distances, queue = {(0, 0): 0}, [(0, (0, 0))]
            while queue:
                cost, (x, y) = heapq.heappop(queue)
                if cost != distances[x, y]:
                    continue
                for nx in range(max(0, x-1), min(12, x+2)):
                    for ny in range(max(0, y-1), min(12, y+2)):
                        if (nx, ny) == (x, y) or grid[nx, ny]:
                            continue
                        if nx != x and ny != y and (grid[nx, y] or grid[x, ny]):
                            continue
                        nc = cost + math.hypot(nx-x, ny-y)
                        if nc < distances.get((nx, ny), math.inf):
                            distances[nx, ny] = nc
                            heapq.heappush(queue, (nc, (nx, ny)))
            path = astar_grid(grid, (0, 0), (11, 11))
            expected = distances.get((11, 11))
            if expected is None:
                self.assertIsNone(path)
            else:
                self.assertAlmostEqual(length(path), expected)

    def test_no_diagonal_corner_cutting(self):
        self.assertIsNone(astar_grid(np.array([[False, True], [True, False]]), (0, 0), (1, 1)))

    def test_global_smoothing_keeps_building_clearance(self):
        prior, _ = build('delivery')
        planner = PriorMap(prior)
        raw, path = planner.route()
        self.assertTrue(planner.safe(sample_path(path, 0.02)))
        self.assertLessEqual(length(path), length(raw) + 0.01)
        self.assertGreater(np.max(abs(path[:, 1])), 3)

    def test_tracking_reserve_survives_small_path_deviation(self):
        prior, _ = build('delivery')
        planner = PriorMap(prior)
        _, path = planner.route()
        for dx, dy in [(-0.2, -0.2), (-0.2, 0.2), (0.2, -0.2), (0.2, 0.2)]:
            shifted = path + np.array([dx, dy, 0])
            self.assertTrue(planner.safe(shifted, planner.execution_inflation))

    def test_control_modules_do_not_import_ground_truth(self):
        root = Path(__file__).resolve().parents[1] / 'urban_drone'
        for name in ['mission.py', 'local.py', 'planning.py', 'perception.py', 'recovery.py', 'trajectory.py', 'clearance.py','safety.py','global_policy.py']:
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, 'world_pose', name)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ['scene', 'evaluate', 'report'], name)

    def test_range_map_does_not_invent_free_space(self):
        occupancy = RangeMap([-5, 20, -10, 10])
        self.assertFalse(occupancy.observed_free([[1, 0, 2]])[0])
        import time
        scan = Scan(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
                    -0.1, 0.1, -0.1, 0.1, 12, time.monotonic(),
                    np.array([0, 0, 2.0]), np.array([0, 0, 0, 1]), 0)
        occupancy.integrate(scan)
        self.assertTrue(occupancy.observed_free([[1, 0, 2.3]])[0])
        self.assertFalse(occupancy.observed_free([[3, 0, 2.3]])[0])
        self.assertFalse(occupancy.safe([[2, 0, 2.3]]))

    def test_pose_rotation(self):
        r = rotation([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_height_change_changes_feasible_strategies(self):
        # Dense sensor-map fixtures, not a separate algorithm for each case.
        from scipy.spatial import cKDTree
        prior, _ = build('wall_low')
        planner = PriorMap(prior)
        for height in (2.5, 7):
            occupancy = RangeMap(prior['bounds'])
            yy, zz = np.meshgrid(np.arange(-4, 4.1, 0.25), np.arange(0.4, height, 0.25))
            occupancy.points = np.stack((np.full(yy.size, 8), yy.ravel(), zz.ravel()), axis=1)
            occupancy.tree = cKDTree(occupancy.points)
            options = candidates([2, 0, 2], [12, 0, 2], occupancy, planner)
            self.assertTrue(options)
            if height < 3:
                self.assertEqual(options[0].strategy, 'OVER')
            else:
                self.assertTrue(all(c.strategy != 'OVER' for c in options))
            for option in options:
                self.assertTrue(occupancy.safe(sample_path(option.points, 0.02), 0.8))


if __name__ == '__main__':
    unittest.main()

