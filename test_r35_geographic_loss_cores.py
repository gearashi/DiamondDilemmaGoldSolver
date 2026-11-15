"""Scope, authentic negative evidence and positive controls for geographic loss feedback."""
from pathlib import Path
import unittest

from bench_r35_geographic_loss_cores import connected_pose_count
from bench_r35_loss_boundary344 import problem_scope
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r32_gold_loss import read_json
from r35_geographic_replay import geographic_departures
from r35_loss_witness import physical_price
from test_r35_geographic_replay import triangle


class GeographicLossCoresTests(unittest.TestCase):
    def test_scope_projection_and_held_out_plant(self):
        for seed in (35, 355, 815):
            p, _ = planted(1, 2, seed, False, True)
            ref = set(physical_bonds(p, [3*i for i in range(p.n)]))
            core = triangle(p)
            edge = min(core)
            self.assertEqual(connected_pose_count(p, core), 0)
            self.assertTrue(connected_pose_count(p, (edge,)))
            self.assertFalse(geographic_departures(p, core-{edge}, set(), [core]))
            self.assertEqual(geographic_departures(p, core-{edge}, {edge}, [core, core]), (core-{edge},))
            self.assertEqual(geographic_departures(p, set(), core, [core]), (frozenset(),))
            self.assertFalse(geographic_departures(p, ref, set(), [ref, [(100000, 2)], [(3,3)]]))

    def test_original_pose_and_repriced_minimum_including_alternate_loss_sets(self):
        path = next((f for f in (Path('data/tiles.json'), Path('../data/tiles.json')) if f.is_file()), None)
        if path is None:
            self.skipTest('Original physical dataset unavailable')
        p, ref, groups, req = problem_scope(path)
        doc = read_json(Path(__file__).parent / 'docs/R35_GEOGRAPHIC_LOSS_CORES.json')
        self.assertEqual(p.fingerprint, doc['puzzle'])
        full = [frozenset(map(tuple, row)) for row in doc['geographic_cores']]
        projected = geographic_departures(p, ref, req, full)
        self.assertEqual(set(projected), {c-req for c in full})
        for core in full:
            self.assertEqual(connected_pose_count(p, core), 0)
            self.assertFalse(any(g <= core-req for g in groups))
        self.assertFalse(geographic_departures(p, ref, req & ref, full))
        for result in doc['reports']:
            phase = int(result['phase'],16)
            self.assertEqual(physical_price(p.tiles, ref, groups, req, phase)[0], result['cap'])
            new = physical_price(p.tiles, ref, [*groups, *projected], req, phase)[0]
            self.assertGreaterEqual(new, result['minimum'])
            self.assertGreater(new, result['cap'])


if __name__ == '__main__':
    unittest.main()
