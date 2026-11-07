"""Independent finite cycle-crossover/controller checks; no GPU execution."""
from copy import deepcopy
import itertools
import json
import unittest
from unittest.mock import patch

import sampling_crossover_round25 as crossover
from sampling_crossover_round25 import CycleCrossover, relative_cycles
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles, brute, board


def complementary_fixture():
    """Two complementary two-cell swaps of a separately checked Gold board."""
    problem = SearchProblem(*tetrahedron_tiles())
    parents = [[3, 0, 6, 9], [0, 3, 9, 6]]
    return problem, [{'codes': row} for row in parents], [0, 3, 6, 9]


def rehash(record):
    record['digest'] = crossover._digest({key: value for key, value in record.items() if key != 'digest'})
    return record


class CycleCrossoverReferenceTests(unittest.TestCase):
    def test_every_relative_cycle_union_preserves_inventory_for_all_four_piece_parents(self):
        permutations = list(itertools.permutations(range(4)))
        for left_pieces in permutations:
            left = [3*piece+cell%3 for cell, piece in enumerate(left_pieces)]
            for right_pieces in permutations:
                right = [3*piece+(cell+1)%3 for cell, piece in enumerate(right_pieces)]
                cycles = relative_cycles(left, right)
                self.assertEqual(sorted(cell for cycle in cycles for cell in cycle), list(range(4)))
                for mask in range(1 << len(cycles)):
                    child = left.copy()
                    for index, cycle in enumerate(cycles):
                        if mask & (1 << index):
                            for cell in cycle: child[cell] = right[cell]
                    self.assertEqual(sorted(code//3 for code in child), list(range(4)))

    def test_complementary_physical_cycles_produce_independent_gold_from_non_gold_parents(self):
        for gold_aware in (False, True):
            problem, archive, expected = complementary_fixture()
            self.assertTrue(problem.validate(expected)['valid'])
            self.assertTrue(all(not problem.validate(row['codes'])['valid'] for row in archive))
            self.assertEqual(sorted(map(len, relative_cycles(archive[0]['codes'], archive[1]['codes']))), [2, 2])
            solver = CycleCrossover(problem, gold_aware=gold_aware)
            result = solver.propose(archive)
            self.assertEqual(result['solution'], expected); self.assertEqual(result['candidate'], expected)
            self.assertIn(tuple(result['solution']), brute(problem, 'gold'))
            self.assertEqual(result['stats']['cycle_candidates'], 2)
            self.assertEqual(result['stats']['cycle_parent_scores'], 2)
            self.assertGreater(result['stats']['cycle_improvements'], 0)
            self.assertEqual(result['stats']['cycle_gold_solutions'], 1)
            self.assertEqual(result['stats']['cycle_injections'], 0)
            solver.record_injection(); self.assertEqual(solver.stats['cycle_injections'], 1)
            with self.assertRaisesRegex(RuntimeError, 'No unrecorded'): solver.record_injection()
            result['candidate'][0] = 99; result['stats']['cycle_candidates'] = 99
            self.assertEqual(solver.stats['cycle_candidates'], 2)

    def test_one_candidate_budget_resumes_next_cycle_direction_without_repeating_scores(self):
        problem, archive, expected = complementary_fixture()
        baseline = CycleCrossover(problem, max_pairs=1, max_candidates=1)
        resumed = CycleCrossover(problem, max_pairs=1, max_candidates=1)
        solutions = []
        for step in range(5):
            first = baseline.propose(archive)
            resumed = CycleCrossover(problem, max_pairs=1, max_candidates=1, resume=json.loads(json.dumps(resumed.snapshot())))
            second = resumed.propose(archive)
            self.assertEqual(first, second); self.assertEqual(baseline.snapshot(), resumed.snapshot())
            solutions.append(first['solution'])
        self.assertIn(expected, solutions)
        self.assertEqual(resumed.stats['cycle_candidates'], 2)
        self.assertGreater(resumed.stats['cycle_duplicate_skips'], 0)

    def test_fixed_placements_and_domain_rejections_are_enforced(self):
        problem = SearchProblem(*tetrahedron_tiles(), fixed={0: 0})
        archive = [{'codes': [0, 3, 9, 6]}, {'codes': [0, 4, 6, 9]}]
        result = CycleCrossover(problem).propose(archive)
        self.assertEqual(result['solution'], [0, 3, 6, 9]); self.assertEqual(result['candidate'][0], 0)
        bad = CycleCrossover(problem); before = bad.snapshot()
        with self.assertRaisesRegex(ValueError, 'fixed'): bad.propose([{'codes': [3, 0, 6, 9]}, archive[0]])
        self.assertEqual(before, bad.snapshot())
        problem, archive, _ = complementary_fixture(); problem.domains[0] = [3]
        result = CycleCrossover(problem).propose(archive)
        self.assertIsNone(result['solution']); self.assertGreater(result['stats']['cycle_domain_rejections'], 0)
        if result['candidate'] is not None:
            self.assertTrue(all(code in problem.domains[cell] for cell, code in enumerate(result['candidate'])))

    def test_parent_scores_are_recomputed_archive_order_is_not_cursor_identity(self):
        problem, archive, _ = complementary_fixture()
        spoofed = [{'codes': row['codes'], 'objective': 10**12, 'domain_violations': -1} for row in archive]
        a, b = CycleCrossover(problem), CycleCrossover(problem)
        self.assertEqual(a.propose(spoofed), b.propose(list(reversed(archive))))
        self.assertEqual(a.snapshot(), b.snapshot())
        duplicate = CycleCrossover(problem).propose(archive+[archive[0]])
        self.assertEqual(duplicate['stats']['cycle_parent_scores'], 2)
        self.assertEqual(duplicate['stats']['cycle_parent_duplicates'], 1)

    def test_history_is_bounded_and_eviction_discloses_possible_future_repetition(self):
        problem, archive, _ = complementary_fixture(); solver = CycleCrossover(problem, history_limit=1)
        first = solver.propose(archive); second = solver.propose(archive)
        self.assertEqual(first['stats']['cycle_candidates'], 2)  # No repeats within one atomic round.
        self.assertGreater(second['stats']['cycle_candidates'], first['stats']['cycle_candidates'])
        self.assertEqual(len(solver.snapshot()['state']['history']), 1)
        self.assertEqual(solver.stats['cycle_history_evictions'], solver.stats['cycle_candidates']-1)
        restored = CycleCrossover(problem, history_limit=1, resume=solver.snapshot())
        self.assertEqual(restored.snapshot(), solver.snapshot())

    def test_pair_budget_round_robin_and_expiring_offsets_are_bounded(self):
        problem, archive, _ = complementary_fixture()
        archive += [{'codes': [0, 4, 6, 9]}, {'codes': [1, 3, 6, 9]}]
        solver = CycleCrossover(problem, max_pairs=1, max_candidates=1)
        previous = solver.stats
        for iteration in range(6):
            result = solver.propose(archive); current = result['stats']
            self.assertLessEqual(current['cycle_parent_pairs']-previous['cycle_parent_pairs'], 1)
            self.assertLessEqual(current['cycle_candidates']-previous['cycle_candidates'], 1)
            previous = current
        self.assertEqual(len(solver.snapshot()['state']['offsets']), 6)
        solver.propose(archive[:2])
        self.assertEqual(len(solver.snapshot()['state']['offsets']), 1)
        self.assertEqual(solver.stats['cycle_pair_cursor_evictions'], 5)

    def test_larger_orientation_family_continues_under_sixteen_score_cap(self):
        n = 20
        problem = SearchProblem([{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(n)],
                                board(n, [(cell, 1, (cell+1)%n, 0) for cell in range(n)]))
        archive = [{'codes': [3*cell for cell in range(n)]}, {'codes': [3*cell+1 for cell in range(n)]}]
        solver = CycleCrossover(problem)
        first = solver.propose(archive); second = solver.propose(archive)
        self.assertEqual(first['stats']['cycle_candidates'], 16)
        self.assertEqual(second['stats']['cycle_candidates'], 32)
        self.assertEqual(len(solver.snapshot()['state']['history']), 32)
        self.assertGreater(second['stats']['cycle_domain_rejections'], 0)

    def test_corrupt_scope_policy_and_malformed_state_rejected(self):
        problem, archive, _ = complementary_fixture(); solver = CycleCrossover(problem); solver.propose(archive)
        original = solver.snapshot()
        changed = deepcopy(original); changed['state']['pair_cursor'] = 1
        with self.assertRaisesRegex(ValueError, 'digest'): CycleCrossover(problem, resume=changed)
        for mutate in (lambda state: state['stats'].__setitem__('cycle_candidates', True),
                       lambda state: state.__setitem__('injection_pending', 1),
                       lambda state: state['history'].append(state['history'][0]),
                       lambda state: state['offsets'][0].__setitem__('next', 999),
                       lambda state: state['offsets'].append(deepcopy(state['offsets'][0])),
                       lambda state: state['stats'].__setitem__('cycle_injections', 10)):
            changed = deepcopy(original); mutate(changed['state']); rehash(changed)
            with self.assertRaises(ValueError): CycleCrossover(problem, resume=changed)
        with self.assertRaisesRegex(ValueError, 'policy'): CycleCrossover(problem, max_candidates=15, resume=original)
        for field, malformed in (('gold_aware', 0), ('max_candidates', True)):
            changed = CycleCrossover(problem, max_candidates=1).snapshot()
            changed['policy'][field] = malformed; rehash(changed)
            with self.assertRaisesRegex(ValueError, 'policy'): CycleCrossover(problem, max_candidates=1, resume=changed)
        problem.domains[0] = problem.domains[0][1:]
        with self.assertRaisesRegex(ValueError, 'scope'): CycleCrossover(problem, resume=original)
        with self.assertRaisesRegex(ValueError, 'changed'): solver.snapshot()

    def test_failures_are_atomic_and_scope_mutation_cannot_publish_candidates(self):
        import gpu_sampling
        problem, archive, _ = complementary_fixture(); solver = CycleCrossover(problem); original = solver.snapshot()
        for invalid in ([{'codes': [0, 0, 6, 9]}], [{'codes': [True, 3, 6, 9]}], archive*5, [{'missing': []}]):
            with self.assertRaises(ValueError): solver.propose(invalid)
            self.assertEqual(solver.snapshot(), original)
        with patch.object(gpu_sampling, 'score_board', side_effect=RuntimeError('scorer failed')):
            with self.assertRaisesRegex(RuntimeError, 'scorer failed'): solver.propose(archive)
        self.assertEqual(solver.snapshot(), original)
        old = problem.domains[0].copy(); real = gpu_sampling.score_board; changed = [False]
        def mutate(p, codes):
            if not changed[0]: p.domains[0] = p.domains[0][1:]; changed[0] = True
            return real(p, codes)
        with patch.object(gpu_sampling, 'score_board', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed'): solver.propose(archive)
        problem.domains[0] = old
        self.assertEqual(solver.snapshot(), original)
        with patch.object(crossover, 'MAX_SNAPSHOT_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'byte'): solver.snapshot()
            with self.assertRaisesRegex(ValueError, 'byte'): CycleCrossover(problem, resume=original)
        self.assertEqual(solver.snapshot(), original)


if __name__ == '__main__': unittest.main()
