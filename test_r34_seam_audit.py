"""Independent finite audit of persisted regional SAT clauses, cores, and partial queries."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import combinations, product
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from r27_reason_maxsat import Budget, SeamFormula, domain_rows
from r31_defect_bands import bands, exact_thresholds, solve
import r34_band_cores as cores
from r34_seam_cnf import encoded_scope, prepare
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_dependencies import requires_sat


# These domains include competing physical pieces as well as multiple orientations.
ROWS = [[0, 16], [1, 15], [3, 4], [12, 14], [20, 21, 22],
        [19, 20, 21], [25, 26], [27], [6], [9]]


def admits(vector, terms):
    return all(vector[g] >= value if sign == 1 else vector[g] <= value
               for g, sign, value in terms)


def boards_with_vectors(problem, rows, groups):
    for codes in product(*rows):
        if len({v // 3 for v in codes}) != problem.n:
            continue
        mismatches = []
        for a, first, b, second in problem.edges:
            right = format(int(problem.masks[codes[b], second]), '011b')[::-1]
            mismatches.append(int(problem.masks[codes[a], first]) != int(right, 2))
        vector = tuple(sum(mismatches[e] for e in group) for group in groups)
        if len(mismatches) - sum(vector) != problem.validate(list(codes))['matched_edges']:
            raise AssertionError('Independent score disagrees with direct eleven-position seam reversal')
        yield vector


@requires_sat
class SeamAuditTests(unittest.TestCase):
    def setUp(self):
        self.problem = SearchProblem(*planted_gold(19, 1, 3)[:2])
        self.rows = domain_rows(self.problem, ROWS)
        self.groups = bands(self.problem, 3)
        self.edges = len(self.problem.edges)
        self.vectors = list(boards_with_vectors(self.problem, self.rows, self.groups))

    def prepare(self, db, capacity, sat, stats, *, split=True, check=lambda: None,
                threshold_builder=exact_thresholds):
        return prepare(db, ':memory:', self.problem, self.rows, split, self.groups, capacity,
                       sat, check, float('inf'), stats, SeamFormula, threshold_builder,
                       encoded_scope(self.problem, self.rows, split))

    def test_every_capacity_and_two_region_literal_pair(self):
        from pysat.solvers import Glucose4
        self.assertEqual(len(self.vectors), 80)
        # A full-capacity counter comes first. Every smaller cap must bind the
        # correct element of the recovered totalizer, with either literal sign.
        with closing(sqlite3.connect(':memory:')) as warm:
            for cap in reversed(range(self.edges + 1)):
                feasible = {v for v in self.vectors if sum(v) <= cap}
                stats = {}
                with closing(sqlite3.connect(':memory:')) as cold, Glucose4() as a, Glucose4() as b:
                    _, fresh = self.prepare(cold, cap, a, {})
                    _, replayed = self.prepare(warm, cap, b, stats)
                    terms = [[(g, sign, v)
                              for sign in (1, -1)
                              for v in (range(1, min(len(group), cap) + 1) if sign == 1
                                        else range(len(group) + 1))]
                             for g, group in enumerate(self.groups)]
                    checks = [()] + [(term,) for group in terms for term in group]
                    for left, right in combinations(terms, 2):
                        checks.extend(product(left, right))
                    for query in checks:
                        expected = any(admits(v, query) for v in feasible)
                        result = (a.solve(assumptions=cores.assumptions(query, fresh, cap)),
                                  b.solve(assumptions=cores.assumptions(query, replayed, cap)))
                        self.assertEqual(result, (expected, expected), (cap, query))
                self.assertEqual(stats.get('counter_builds' if cap == self.edges else 'counter_replays'), 1)
                with Glucose4() as sat:
                    self.prepare(warm, cap, sat, {}, split=False)
                    self.assertEqual(sat.solve(), bool(feasible), cap)

    def checked_solve(self, path, target, **opts):
        result = solve(self.problem, path, domains=self.rows, target=target, **opts)
        self.assertEqual(result['status'], 'witness' if any(sum(v) <= self.edges - target
                         for v in self.vectors) else 'infeasible')
        if result['witness']:
            self.assertEqual(self.problem.validate(result['witness']['codes'])['matched_edges'],
                             result['witness']['score'])
        return result

    def test_native_nonempty_cores_reused_under_stronger_global_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cores.sqlite'
            first = self.checked_solve(path, 5, width=1)
            second = self.checked_solve(path, 6, width=1)
            third = self.checked_solve(path, 7, width=1)
            self.assertGreater(second['shared_core_checks'], 0)
            self.assertGreater(third['shared_core_skips'], 0)
            self.assertEqual(first['status'], third['status'])
            with closing(sqlite3.connect(path)) as db:
                records = db.execute('SELECT budget,terms,verified FROM r34_band_cores').fetchall()
            self.assertTrue(any(json.loads(raw) and verified for _, raw, verified in records))
            for cap, raw, _ in records:
                terms = json.loads(raw)
                self.assertFalse(any(sum(v) <= cap and admits(v, terms) for v in self.vectors),
                                 (cap, terms))

    def test_inconclusive_imported_hints_never_starve_one_call_reopens(self):
        from pysat.solvers import Glucose4
        class Probe(list):
            pass
        native = Glucose4.solve_limited
        verifier_literals = cores.assumptions
        def bounded(native_solver, *args, **kwargs):
            # Model a core query whose broader assumptions need more conflicts
            # than one slice. Each actual geographic branch gets a normal solve.
            if isinstance(kwargs['assumptions'], Probe):
                return None
            return native(native_solver, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'slices.sqlite'
            self.checked_solve(path, 5, width=1)
            with patch.object(Glucose4, 'solve_limited', bounded), patch.object(
                    cores, 'assumptions', side_effect=lambda *args: Probe(verifier_literals(*args))):
                checks = calls = 0
                for _ in range(24):
                    result = solve(self.problem, path, domains=self.rows, target=6, width=1, max_calls=1)
                    calls += result['calls']
                    checks += result['shared_core_checks']
                    if result['status'] != 'open':
                        break
            self.assertEqual(result['status'], 'witness')
            self.assertGreater(calls, checks)
            self.assertGreater(checks, 0)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r34_band_core_deferrals').fetchone()[0],
                                 checks)
            # A different request may still independently establish the same cores.
            self.assertGreater(self.checked_solve(path, 7, width=1)['shared_core_checks'], 0)

    def test_interrupted_partial_builds_recover_without_publishing_prefixes(self):
        from pysat.solvers import Glucose4
        calls = 0
        def interrupt_structure():
            nonlocal calls
            calls += 1
            if calls == 4:
                raise Budget()
        def interrupt_counters(*args):
            exact_thresholds(*args)
            raise Budget()
        with closing(sqlite3.connect(':memory:')) as db:
            with Glucose4() as sat, self.assertRaises(Budget):
                self.prepare(db, 6, sat, {}, check=interrupt_structure)
            self.assertEqual(db.execute('SELECT count(*) FROM r34_band_cnf').fetchone()[0], 0)
            with Glucose4() as sat, self.assertRaises(Budget):
                self.prepare(db, 6, sat, {}, threshold_builder=interrupt_counters)
            self.assertEqual(db.execute('SELECT count(*) FROM r34_band_cnf').fetchone()[0], 1)
            with Glucose4() as sat:
                stats = {}
                self.prepare(db, 6, sat, stats)
                self.assertEqual(stats, {'structural_replays': 1, 'counter_builds': 1})
                self.assertEqual(sat.solve(), any(sum(v) <= 6 for v in self.vectors))

    def test_changed_domains_and_concurrent_mixed_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'concurrent.sqlite'
            self.checked_solve(path, 5, width=1)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda arg: self.checked_solve(path, arg[0], width=arg[1]),
                                        [(6, 1), (9, 2), (12, 3), (6, 1)]))
            self.assertEqual(results[0]['witness'], results[3]['witness'])
            widened = [list(row) for row in self.rows]
            widened[0] += [1, 2]
            narrowed = [[row[0]] for row in self.rows]
            shuffled = domain_rows(self.problem, [tuple(reversed(row)) + row[:1] for row in self.rows])
            current = encoded_scope(self.problem, self.rows, True)
            self.assertEqual(encoded_scope(self.problem, shuffled, True), current)
            for changed in (widened, narrowed):
                self.assertNotEqual(encoded_scope(self.problem, domain_rows(self.problem, changed), True),
                                    current)
                vectors = list(boards_with_vectors(self.problem, changed, self.groups))
                for target in (6, 10, 12):
                    result = solve(self.problem, path, domains=changed, target=target, width=1)
                    self.assertEqual(result['status'], 'witness' if any(
                        sum(v) <= self.edges - target for v in vectors) else 'infeasible')


if __name__ == '__main__':
    unittest.main()
