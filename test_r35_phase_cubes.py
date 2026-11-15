"""Independent signed phase enumeration, authenticated reuse and branch scope checks."""
from collections import Counter
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import unittest

from r35_phase_census import census, count_equations
from r35_phase_cube_rules import certificate_rules, certified_rules, published_rules
from search_problem import SearchProblem
from verify_r35_phase_board import small_sphere, specimen
from verify_r35_phase_cube import check


DOCS = Path(__file__).resolve().parent / "docs"


class PhaseCubeTests(unittest.TestCase):
    def test_equations_against_direct_slot_histograms(self):
        from r35_loss_witness import raw_sides, reciprocal
        problem, _ = specimen(31, False, True)
        sides = raw_sides(problem.tiles)
        raw = set()
        for phase in range(1 << problem.n):
            count = Counter((slots,base ^ ((phase>>(side//3))&1))
                            for side,(slots,base) in enumerate(sides))
            if all((v % 2 == 0 if reciprocal(k) == k else count[reciprocal(k)] == v)
                   for k,v in count.items()):
                raw.add(phase)
        self.assertTrue(raw)
        rows = count_equations(problem.tiles)
        for a,b in product((0,1), repeat=2):
            pins = {1:a, 6:b}
            out = census(rows, problem.n, pins)
            expected = {x for x in raw if all((x>>t)&1 == v for t,v in pins.items())}
            self.assertTrue(out["complete"])
            self.assertEqual(set(out["phases"]), expected)
        self.assertFalse(census(rows, problem.n, {}, budget=0)["complete"])
        for bad in ({-1:0},{0:2},{10:0},{True:0}):
            with self.assertRaises(ValueError):
                census(rows, problem.n, bad)

    def test_authenticated_data_and_changed_scope(self):
        file = json.loads((DOCS/"R35_PARTIAL_PHASE_CORES.json").read_bytes())
        rules = published_rules()
        self.assertEqual(tuple(map(len,rules)), (48,49))
        self.assertIs(published_rules(), rules)
        self.assertEqual(certificate_rules(file), rules)
        damaged = deepcopy(file)
        damaged["rules"][0]["pins"][0] *= -1
        self.assertEqual(certificate_rules(damaged), ())
        for invalid in (None, (), {"leaves":{}}, {"bad":object()}):
            self.assertEqual(certificate_rules(invalid), ())
        self.assertEqual(certified_rules(small_sphere(4)), ())

    def test_original_census_and_scope(self):
        path = next((p for p in map(Path,("data/tiles.json","../data/tiles.json")) if p.is_file()), None)
        if path is None:
            self.skipTest("Original pieces unavailable.")
        data = json.loads(path.read_bytes())
        problem = SearchProblem(data)
        report = check(problem, json.loads((DOCS/"R35_PARTIAL_PHASE_CORES.json").read_bytes()),
                       json.loads((DOCS/"R35_FIXED_PHASE_PROOF.json").read_bytes()))
        self.assertEqual(report["gauges"]["labelled_roots"], 1920)
        self.assertEqual(report["gauges"]["signed_physical_matching"], 38)
        self.assertEqual(report["cubes"], [48,49])
        old = problem.edges
        problem.edges = (old[1], *old[1:])
        self.assertEqual(certified_rules(problem), ())
        problem.edges = old
        # Replacing one physical tile with another assigned the original ID
        # preserves dimensions and mesh, and changes the actual signed artwork.
        altered = [dict(t) for t in problem.tiles]
        altered[0]["segments"] = altered[1]["segments"]
        self.assertEqual(certified_rules(SearchProblem(altered)), ())


if __name__ == "__main__":
    unittest.main()
