"""Independent finite checks for shared constraints, loop pruning, and DFS resume."""
from itertools import permutations, product
import json
from types import SimpleNamespace
import unittest

from constraint_dfs import solve
from search_problem import SearchProblem
from validator import validate_arrangement


def board(n, edges):
    return SimpleNamespace(cells=tuple(range(n)), edges=tuple(edges))


def double_face():
    return board(2, [(0, s, 1, s) for s in range(3)])


def simple_tiles():
    return [{'id': 'A', 'segments': [[[0, 2], [1, 4]]]},
            {'id': 'B', 'segments': [[[0, 10], [1, 8]]]}]


def multiple_loops():
    return [{'id': str(i), 'segments': [[[s, 2], [s, 10]] for s in range(3)]} for i in range(2)]


def tetrahedron_tiles():
    edges = [(0, 0, 1, 0), (0, 1, 2, 0), (0, 2, 3, 0),
             (1, 1, 2, 1), (1, 2, 3, 1), (2, 2, 3, 2)]
    paths = [[] for _ in range(4)]
    for index, p in ((0, 2), (2, 3), (3, 4), (5, 5)):
        a, sa, b, sb = edges[index]
        paths[a].append([sa, p]); paths[b].append([sb, 12-p])
    return [{'id': str(i), 'segments': [points]} for i, points in enumerate(paths)], board(4, edges)


def brute(problem, target):
    found = []
    for pieces in permutations(range(problem.n)):
        for rotations in product(range(3), repeat=problem.n):
            codes = [3*tile+rot for tile, rot in zip(pieces, rotations)]
            if any(codes[cell] not in domain for cell, domain in enumerate(problem.domains)):
                continue
            report = validate_arrangement(problem.tiles, codes, problem.board)
            if report['valid'] or target == 'edges' and report['unique_tiles'] and report['matched_edges'] == len(problem.edges):
                found.append(tuple(codes))
    return found


class SharedProblemTests(unittest.TestCase):
    def test_noncentral_positions_require_reversal_and_validate_independently(self):
        problem = SearchProblem(simple_tiles(), double_face())
        self.assertTrue(problem.validate([0, 3])['valid'])
        self.assertEqual(problem.masks.shape, (6, 3))
        self.assertEqual(str(problem.masks.dtype), 'uint16')
        self.assertEqual(problem.neighbors.tolist(), [[1, 1, 1], [0, 0, 0]])
        self.assertEqual(problem.sides.tolist(), [[0, 1, 2], [0, 1, 2]])
        wrong = simple_tiles(); wrong[1]['segments'] = [[[0, 2], [1, 4]]]
        self.assertFalse(SearchProblem(wrong, double_face()).validate([0, 3])['valid'])

    def test_boundaries_fixed_cells_and_unique_tiles_filter_domains(self):
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(2)]
        problem = SearchProblem(tiles, board(2, [(0, 0, 1, 0)]), fixed={0: 0})
        self.assertEqual(problem.domains, [[0], [3]])
        self.assertTrue(problem.validate([0, 3])['valid'])
        rotated = SearchProblem(tiles, problem.board, fixed={0: 1})
        self.assertEqual(rotated.domains[0], [])

    def test_partial_closed_proper_loop_and_sound_support_cut(self):
        edges = [(0, s, 1, s) for s in range(3)] + [(2, s, 3, s) for s in range(3)]
        tiles = [{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(4)]
        problem = SearchProblem(tiles, board(4, edges))
        self.assertFalse(problem.partial_loop_impossible([0, -1, -1, -1]))
        self.assertTrue(problem.partial_loop_impossible([0, 3, -1, -1]))
        cut = problem.gold_nogood([0, 3, 6, 9])
        self.assertEqual(cut, [(0, 0), (1, 3)])
        for order in permutations((2, 3)):
            for rotations in product(range(3), repeat=2):
                codes = [0, 3] + [tile*3+rot for tile, rot in zip(order, rotations)]
                self.assertFalse(problem.validate(codes)['valid'])
        single = SearchProblem(simple_tiles(), double_face())
        self.assertFalse(single.partial_loop_impossible([0, 3]))

    def test_fingerprint_includes_interior_pairing_and_fixed_placements(self):
        tiles = multiple_loops()
        first = SearchProblem(tiles, double_face())
        tiles[0]['segments'] = [[[0, 2], [1, 2]], [[0, 10], [1, 10]], [[2, 2], [2, 10]]]
        second = SearchProblem(tiles, double_face())
        self.assertEqual(first.masks.tolist(), second.masks.tolist())
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, SearchProblem(multiple_loops(), double_face(), fixed={0: 0}).fingerprint)

    def test_invalid_geometry_and_fixed_placements_are_rejected(self):
        for fixture, fixed in ((board(1, []), None),
                               (board(2, [(0, 0, 1, 0), (0, 0, 1, 1)]), None),
                               (double_face(), {2: 0}), (double_face(), {0: 6}),
                               (double_face(), {True: 0})):
            with self.subTest(fixed=fixed), self.assertRaises(ValueError):
                SearchProblem(simple_tiles(), fixture, fixed)


class ConstraintDFSTests(unittest.TestCase):
    def test_tiny_searches_agree_with_independent_bruteforce(self):
        tetra, tetra_board = tetrahedron_tiles()
        altered = simple_tiles(); altered[1]['segments'] = [[[0, 2], [1, 4]]]
        cases = [(simple_tiles(), double_face(), {}),
                 (multiple_loops(), double_face(), {}),
                 (altered, double_face(), {}),
                 (tetra, tetra_board, {}),
                 (tetra, tetra_board, {0: 0}),
                 (tetra, tetra_board, {0: 0, 1: 0})]
        for tiles, geometry, fixed in cases:
            for target in ('gold', 'edges'):
                with self.subTest(n=len(tiles), target=target, fixed=fixed, tiles=tiles):
                    problem = SearchProblem(tiles, geometry, fixed)
                    expected = brute(problem, target)
                    result = solve(problem, seconds=20, seed=23, target=target)
                    self.assertTrue(result['complete'], result)
                    if expected:
                        self.assertIn(tuple(result['codes']), expected)
                        self.assertIn(result['status'], ('solved', 'edge_perfect'))
                    else:
                        self.assertEqual(result['status'], 'infeasible')
                        self.assertIsNone(result['codes'])

    def test_gold_rejects_multiloop_but_edge_target_accepts(self):
        problem = SearchProblem(multiple_loops(), double_face())
        edge = solve(problem, target='edges', seconds=2)
        self.assertEqual(edge['status'], 'edge_perfect')
        self.assertEqual(edge['validation']['line_components'], 3)
        self.assertFalse(edge['validation']['valid'])
        gold = solve(problem, target='gold', seconds=2)
        self.assertEqual(gold['status'], 'infeasible')
        self.assertEqual(gold['stats']['leaves'], 18)

    def test_closed_proper_loop_prunes_before_remaining_cells_are_chosen(self):
        geometry = board(4, [(0, s, 1, s) for s in range(3)] + [(2, s, 3, s) for s in range(3)])
        tiles = [{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(4)]
        problem = SearchProblem(tiles, geometry, fixed={0: 0, 1: 3})
        result = solve(problem, seconds=2, target='gold')
        self.assertEqual(result['status'], 'infeasible')
        self.assertGreater(result['stats']['loop_prunes'], 0)
        self.assertEqual(result['stats']['leaves'], 0)
        self.assertEqual(solve(problem, seconds=2, target='edges')['status'], 'edge_perfect')

    def test_zero_budget_and_requested_stop_are_incomplete_not_infeasible(self):
        problem = SearchProblem(simple_tiles(), double_face())
        for options, status in (({'seconds': 0}, 'timeout'), ({'seconds': 2, 'stop': lambda: True}, 'stopped')):
            result = solve(problem, **options)
            self.assertEqual(result['status'], status)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['codes'])
            restored = solve(problem, seconds=2, resume=json.loads(json.dumps(result['checkpoint'])))
            self.assertEqual(restored['status'], 'solved')

    def test_multiple_stop_resume_visits_every_leaf_once_in_the_same_order(self):
        problem = SearchProblem(multiple_loops(), double_face())
        baseline = []
        solve(problem, seconds=5, seed=71, progress=lambda event: baseline.append(tuple(event['codes'])) if event['event'] == 'candidate' else None)
        self.assertEqual(len(baseline), 18)
        self.assertEqual(len(set(baseline)), 18)
        seen, checkpoint, prior_nodes = [], None, 0
        for limit in (4, 9, 13, 100):
            result = solve(problem, seconds=5, seed=71, resume=checkpoint,
                           stop=lambda: len(seen) >= limit,
                           progress=lambda event: seen.append(tuple(event['codes'])) if event['event'] == 'candidate' else None)
            self.assertGreaterEqual(result['nodes'], prior_nodes)
            prior_nodes = result['nodes']
            self.assertEqual(result['stats']['leaves'], len(seen))
            if result['status'] == 'infeasible': break
            self.assertEqual(result['status'], 'stopped')
            self.assertFalse(result['complete'])
            checkpoint = json.loads(json.dumps(result['checkpoint']))
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(seen, baseline)
        self.assertEqual(len(set(seen)), 18)

    def test_interruption_inside_propagation_preserves_exact_leaf_coverage(self):
        problem = SearchProblem(multiple_loops(), double_face())
        baseline = []
        solve(problem, seconds=2, seed=14, progress=lambda event: baseline.append(tuple(event['codes'])) if event['event'] == 'candidate' else None)
        for limit in (1, 2, 3, 4, 7, 11, 20, 35, 60, 100):
            with self.subTest(stop_check=limit):
                calls, seen = [0], []
                def stop():
                    calls[0] += 1
                    return calls[0] >= limit
                def progress(event):
                    if event['event'] == 'candidate': seen.append(tuple(event['codes']))
                interrupted = solve(problem, seconds=2, seed=14, stop=stop, progress=progress)
                if interrupted['status'] == 'stopped':
                    resumed = solve(problem, seconds=2, resume=json.loads(json.dumps(interrupted['checkpoint'])), progress=progress)
                    self.assertEqual(resumed['status'], 'infeasible')
                self.assertEqual(seen, baseline)

    def test_hints_reorder_but_never_remove_possible_solutions(self):
        problem = SearchProblem(simple_tiles(), double_face())
        for hints in ([0, 0], [-1, -1], {0: [2, 1, 0], 1: 0}):
            with self.subTest(hints=hints):
                result = solve(problem, seconds=2, hints=hints)
                self.assertEqual(result['status'], 'solved')
                self.assertTrue(problem.validate(result['codes'])['valid'])

    def test_resume_rejects_wrong_problem_target_or_invalid_frame(self):
        problem = SearchProblem(simple_tiles(), double_face())
        checkpoint = solve(problem, seconds=0)['checkpoint']
        with self.assertRaises(ValueError): solve(problem, seconds=2, target='edges', resume=checkpoint)
        with self.assertRaises(ValueError): solve(SearchProblem(simple_tiles(), double_face(), fixed={0: 0}), seconds=2, resume=checkpoint)
        malformed = json.loads(json.dumps(checkpoint)); malformed['stack'][0]['domains'][0] = hex(1 << 100)
        with self.assertRaises(ValueError): solve(problem, seconds=2, resume=malformed)

    def test_resume_rejects_missing_siblings_and_wrong_parent_branches(self):
        problem = SearchProblem(multiple_loops(), double_face())
        seen = []
        stopped = solve(problem, seconds=2, stop=lambda: len(seen) >= 1,
                        progress=lambda event: seen.append(event) if event['event'] == 'candidate' else None)
        checkpoint = stopped['checkpoint']
        self.assertGreaterEqual(len(checkpoint['stack']), 2)
        missing = json.loads(json.dumps(checkpoint))
        missing['stack'][0]['choices'].pop()
        with self.assertRaises(ValueError): solve(problem, seconds=2, resume=missing)
        wrong = json.loads(json.dumps(checkpoint))
        parent = wrong['stack'][0]
        wrong['stack'][1]['domains'][parent['cell']] = hex(1 << parent['choices'][-1])
        with self.assertRaises(ValueError): solve(problem, seconds=2, resume=wrong)
        for field in ('seed', 'stats', 'stack'):
            bad = json.loads(json.dumps(checkpoint)); del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError): solve(problem, seconds=2, resume=bad)

    def test_damaged_domain_digest_cannot_turn_satisfiable_into_infeasible(self):
        problem = SearchProblem(simple_tiles(), double_face())
        saved = json.loads(json.dumps(solve(problem, seconds=0)['checkpoint']))
        self.assertEqual(solve(problem, seconds=2, resume=saved)['status'], 'solved')
        saved['stack'][0]['domains'][0] = '0x0'
        with self.assertRaisesRegex(ValueError, 'integrity'):
            solve(problem, seconds=2, resume=saved)

    def test_resigned_checkpoint_still_checks_sibling_structure(self):
        from constraint_dfs import _checkpoint_digest
        problem = SearchProblem(multiple_loops(), double_face())
        seen=[]
        result=solve(problem,seconds=2,stop=lambda:bool(seen),progress=lambda event:seen.append(event) if event['event']=='candidate' else None)
        saved=json.loads(json.dumps(result['checkpoint']))
        saved['stack'][0]['choices'].pop()
        saved['digest']=_checkpoint_digest(saved)
        with self.assertRaises(ValueError):solve(problem,seconds=2,resume=saved)

    def test_invalid_budgets_and_targets_are_rejected(self):
        problem = SearchProblem(simple_tiles(), double_face())
        for options in ({'seconds': -1}, {'seconds': float('inf')}, {'seconds': True}, {'target': 'unknown'}, {'hints': [0]}):
            with self.subTest(options=options), self.assertRaises(ValueError): solve(problem, **options)


if __name__ == '__main__': unittest.main()
