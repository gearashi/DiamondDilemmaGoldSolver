"""Coverage and publication tests for interrupted geographic SAT placement branches."""
from contextlib import closing
from itertools import count
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from r31_defect_bands import solve
from r34_band_placements import intervals, region, split_placement
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_dependencies import requires_sat
from test_r34_seam_audit import ROWS


class PartitionTests(unittest.TestCase):
    def test_balanced_partitions_exhaust_cartesian_intervals_once(self):
        rows = [(1,), (0, 3, 7, 11, 12), (2, 5, 6)]
        box = [[0, 2], [1, 1]]
        pending = [box]
        leaves = set()
        self.assertIs(region(box), box)
        while pending:
            node = pending.pop()
            self.assertEqual(region(node), box)
            pins = node.get('pins', ()) if isinstance(node, dict) else ()
            bounds = intervals(rows, pins)
            children = split_placement(box, bounds, pins)
            if children:
                pending.extend(children)
            else:
                leaf = tuple(row[lo] for row, (lo, _) in zip(rows, bounds))
                self.assertNotIn(leaf, leaves)
                leaves.add(leaf)
        self.assertEqual(len(leaves), 15)


@requires_sat
class PlacementSearchTests(unittest.TestCase):
    def setUp(self):
        self.problem = SearchProblem(*planted_gold(19, 1, 3)[:2])

    def test_negative_placement_sibling_never_publishes_a_geographic_nogood(self):
        from pysat.solvers import Glucose4
        native = Glucose4.solve_limited
        calls = count()
        def interrupted_once(sat, *args, **kwargs):
            answer = native(sat, *args, **kwargs)
            return None if next(calls) == 0 else answer
        with tempfile.TemporaryDirectory() as tmp, patch.object(Glucose4, 'solve_limited', interrupted_once):
            path = Path(tmp) / 'branches.sqlite'
            def step(**kw):
                return solve(self.problem, path, domains=ROWS, target=11, width=1, split=False,
                             **kw)
            before = step(max_calls=0)
            with closing(sqlite3.connect(path)) as db:
                initial = json.loads(db.execute('SELECT stack FROM r31_bands').fetchone()[0])
            self.assertIsInstance(initial[0], list)
            self.assertEqual(before['pending'], 1)
            first = step(max_calls=1)
            self.assertEqual((first['placement_splits'], first['pending']), (1, 2))
            self.assertEqual(step(max_calls=0)['pending'], 2)
            with closing(sqlite3.connect(path)) as db:
                nodes = json.loads(db.execute('SELECT stack FROM r31_bands').fetchone()[0])
            self.assertEqual([n['pins'] for n in nodes], [[[0, 1, 2]], [[0, 0, 1]]])
            second = step(max_calls=1)
            self.assertEqual((second['status'], second['pending'], second['placement_local_unsat']),
                             ('open', 1, 1))
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r34_band_cores').fetchone()[0], 0)
            third = step(max_calls=1)
            self.assertEqual(third['status'], 'witness')
            self.assertEqual(third['witness']['codes'][0], 16)
            self.assertEqual(self.problem.validate(third['witness']['codes'])['matched_edges'], 11)

    def test_interruption_before_decisions_does_not_fragment_the_frontier(self):
        from pysat.solvers import Glucose4
        with tempfile.TemporaryDirectory() as tmp, patch.object(Glucose4, 'solve_limited', return_value=None):
            path = Path(tmp) / 'no-decisions.sqlite'
            for _ in range(3):
                result = solve(self.problem, path, domains=ROWS, split=False, target=11, max_calls=1)
                self.assertEqual((result['calls'], result['placement_splits'], result['pending']),
                                 (1, 0, 1))
            # Cross the deadline after initialization, independently of the
            # host monotonic clock's resolution or native solver startup time.
            from time import monotonic
            start = monotonic()
            ticks = iter((start,))
            with patch('r31_defect_bands.time') as clock:
                clock.monotonic.side_effect = lambda: next(ticks, start + 2)
                sliced = solve(self.problem, path, domains=ROWS, split=False, target=11, seconds=1)
            self.assertEqual((sliced['calls'], sliced['placement_splits'], sliced['pending']), (0, 0, 1))

    def test_unlimited_native_search_keeps_learnt_clauses_after_a_conflict_slice(self):
        from pysat.solvers import Glucose4
        native = Glucose4.solve_limited
        calls = count()
        def interrupted_once(sat, *args, **kwargs):
            answer = native(sat, *args, **kwargs)
            return None if next(calls) == 0 else answer
        with tempfile.TemporaryDirectory() as tmp, patch.object(Glucose4, 'solve_limited', interrupted_once):
            result = solve(self.problem, Path(tmp) / 'unlimited.sqlite', domains=ROWS,
                           split=False, target=11)
        self.assertEqual((result['status'], result['calls'], result['placement_splits']),
                         ('witness', 2, 0))

    def test_singleton_placement_assignment_covers_valid_scores_and_impossible_slices(self):
        from pysat.solvers import Glucose4
        # Interrupt each one-call reopen after reporting work. A complete
        # finite placement search has eight leaves and seven internal nodes.
        spent = count(1)
        rows = [[0, 16], [1, 15], [3], [14], [21], [19, 20], [26], [27], [6], [9]]
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(Glucose4, 'solve_limited', return_value=None),
              patch.object(Glucose4, 'accum_stats', side_effect=lambda: dict(conflicts=0, decisions=next(spent)))):
            path = Path(tmp) / 'finite.sqlite'
            def finish(target):
                splits = leaves = 0
                for _ in range(16):
                    result = solve(self.problem, path, domains=rows, split=False, target=target, max_calls=1)
                    splits += result['placement_splits']
                    leaves += result['placement_leaf_checks']
                    if result['status'] != 'open':
                        return result, splits, leaves
                self.fail('Finite placement branches failed to close across restarts')
            impossible, splits, leaves = finish(12)
            self.assertEqual(impossible['status'], 'infeasible')
            self.assertEqual((splits, leaves), (7, 8))
            possible, _, _ = finish(11)
            self.assertEqual(possible['status'], 'witness')
            self.assertEqual(self.problem.validate(possible['witness']['codes'])['matched_edges'], 11)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r34_band_cores').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
