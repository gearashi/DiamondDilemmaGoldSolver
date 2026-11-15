"""Check both phase-variable encodings and the eight-sign inventory projection."""
from importlib.util import find_spec
from itertools import product
import json
from pathlib import Path
import unittest

from r35_phase_census import census, count_equations
from search_problem import SearchProblem


@unittest.skipUnless(find_spec("pysat"), "variable board formula generation requires python-sat")
class VariablePhaseBoardTests(unittest.TestCase):
    def test_all_planted_phase_board_projections(self):
        from bench_r35_partial_phase_board import controls
        self.assertEqual(controls(), {"cases":16, "checked_phase_boards":88})

    def test_original_prefix_projection_has_exactly_six_assignments(self):
        from bench_r35_partial_phase_board import TARGET, add_balanced_projection
        path = next((p for p in map(Path,("data/tiles.json","../data/tiles.json")) if p.is_file()), None)
        if path is None:
            self.skipTest("Original pieces unavailable.")
        p = SearchProblem(json.loads(path.read_bytes()))
        free = (5,9,15,36,38,80,89,141)
        fixed = {t:(TARGET>>t)&1 for t in range(p.n) if t not in free}
        clauses = []
        report = add_balanced_projection(p,fixed,clauses.append)
        self.assertEqual((report["eligible"],len(clauses)), (6,22))
        allowed = set()
        for values in product((0,1), repeat=8):
            pins = dict(zip(free,values))
            if all(any(pins[abs(l)-1] == (l>0) for l in clause) for clause in clauses):
                allowed.add(sum(v<<t for t,v in (fixed|pins).items()))
        result = census(count_equations(p.tiles),p.n,fixed)
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["phases"]),allowed)


if __name__ == "__main__":
    unittest.main()
