"""Finite exhaustive oracle checks for all three constraint engines; no GPU."""
from copy import deepcopy
from itertools import permutations, product
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import MagicMock, patch

from constraint_models import solve
from search_problem import SearchProblem
from test_dependencies import HAS_ORTOOLS, HAS_SAT, requires_ortools, requires_solver

ENGINES = (['cp-sat', 'cp'] if HAS_ORTOOLS else []) + (['sat'] if HAS_SAT else [])


def pair_problem(*, multiple_loops=False, fixed=None, boundary=False):
    if multiple_loops:
        first = [[[0, 2], [0, 4]], [[1, 2], [1, 4]]]
        second = [[[0, 8], [0, 10]], [[1, 8], [1, 10]]]
    else:
        first = [[[0, 2], [1, 4]]]
        second = [[[0, 10], [1, 8]]]
    data = [{'id': 'A', 'segments': first}, {'id': 'B', 'segments': second}]
    board = SimpleNamespace(cells=(0, 1), edges=tuple((0, side, 1, side) for side in range(2 if boundary else 3)))
    return SearchProblem(data, board, fixed)


def brute(problem, target):
    solutions = set()
    for tiles in permutations(range(problem.n)):
        for rotations in product(range(3), repeat=problem.n):
            codes = tuple(3*tile+rotation for tile, rotation in zip(tiles, rotations))
            if any(code not in problem.domains[cell] for cell, code in enumerate(codes)):
                continue
            report = problem.validate(list(codes))
            if report['valid'] if target == 'gold' else report['matched_edges'] == len(problem.edges):
                solutions.add(codes)
    return solutions


class ConstraintModelTests(unittest.TestCase):
    @requires_solver
    def test_gold_witnesses_match_independent_exhaustive_oracle(self):
        problem = pair_problem()
        expected = brute(problem, 'gold')
        self.assertTrue(expected)
        for engine in ENGINES:
            with self.subTest(engine=engine):
                result = solve(problem, engine=engine, seconds=5, target='gold', hints=[0, 4])
                self.assertEqual(result['status'], 'solved')
                self.assertIn(tuple(result['codes']), expected)
                self.assertTrue(result['complete'])
                self.assertTrue(problem.validate(result['codes'])['valid'])

    @requires_solver
    def test_edge_objective_and_gold_objective_are_distinct(self):
        problem = pair_problem(multiple_loops=True)
        expected = brute(problem, 'edges')
        self.assertTrue(expected)
        self.assertFalse(brute(problem, 'gold'))
        for engine in ENGINES:
            with self.subTest(engine=engine):
                edges = solve(problem, engine=engine, seconds=5, target='edges')
                self.assertEqual(edges['status'], 'edge_perfect')
                self.assertIn(tuple(edges['codes']), expected)
                gold = solve(problem, engine=engine, seconds=5, target='gold')
                self.assertEqual(gold['status'], 'infeasible')
                self.assertTrue(gold['complete'])
                self.assertGreater(gold['stats']['gold_cuts'], 0)
                self.assertLessEqual(gold['stats']['candidates_checked'], len(expected))

    @requires_solver
    def test_nonzero_reversal_fixed_domains_and_boundaries(self):
        for fixed, boundary in (({0: 0}, False), ({0: 0, 1: 4}, False), ({0: 0}, True)):
            problem = pair_problem(fixed=fixed, boundary=boundary)
            expected = brute(problem, 'gold')
            for engine in ENGINES:
                with self.subTest(engine=engine, fixed=fixed, boundary=boundary):
                    result = solve(problem, engine=engine, seconds=5, hints=[5, 1])
                    if expected:
                        self.assertIn(tuple(result['codes']), expected)
                        self.assertEqual(result['status'], 'solved')
                    else:
                        self.assertEqual(result['status'], 'infeasible')
                    self.assertTrue(result['complete'])

    @requires_solver
    def test_duplicate_fixed_tile_is_infeasible(self):
        problem = pair_problem(fixed={0: 0, 1: 0})
        for engine in ENGINES:
            with self.subTest(engine=engine):
                result = solve(problem, engine=engine, seconds=5)
                self.assertEqual(result['status'], 'infeasible')
                self.assertTrue(result['complete'])

    def test_zero_budget_and_requested_stop_are_never_infeasible(self):
        problem = pair_problem()
        stop = threading.Event()
        stop.set()
        for engine in ('sat','cp-sat','cp','cp-sat-circuit'):
            with self.subTest(engine=engine):
                timeout = solve(problem, engine=engine, seconds=0)
                stopped = solve(problem, engine=engine, seconds=5, stop=stop)
                self.assertEqual(timeout['status'], 'timeout')
                self.assertEqual(stopped['status'], 'stopped')
                self.assertFalse(timeout['complete'])
                self.assertFalse(stopped['complete'])

    @requires_solver
    def test_stop_during_model_build_is_checked_before_native_search(self):
        problem = pair_problem()
        for engine in ENGINES:
            event = threading.Event()
            def progress(value):
                if value['phase'] == 'building_model': event.set()
            with self.subTest(engine=engine):
                result = solve(problem, engine=engine, seconds=5, stop=event, progress=progress)
                self.assertEqual(result['status'], 'stopped')
                self.assertEqual(result['nodes'], 0)
                self.assertFalse(result['complete'])

    @requires_solver
    def test_native_sat_and_cpsat_calls_are_interrupted_on_stop_and_deadline(self):
        problem = pair_problem()
        for engine in [value for value in ENGINES if value != 'cp']+(['cp-sat-circuit'] if HAS_ORTOOLS else []):
            for reason in ('stopped', 'timeout'):
                with self.subTest(engine=engine, reason=reason):
                    entered, interrupted = threading.Event(), threading.Event()
                    native = MagicMock()
                    native.__enter__.return_value = native
                    native.accum_stats.return_value = {'decisions': 2}
                    native.num_branches = 2
                    native.status_name.return_value = 'UNKNOWN'
                    def native_call(*args, **kwargs):
                        entered.set()
                        if not interrupted.wait(timeout=2):
                            raise AssertionError('The native-call interrupt was not delivered')
                        return None if engine == 'sat' else 0
                    native.solve_limited.side_effect = native_call
                    native.solve.side_effect = native_call
                    native.interrupt.side_effect = interrupted.set
                    native.stop_search.side_effect = interrupted.set
                    path = 'pysat.solvers.Glucose4' if engine == 'sat' else 'ortools.sat.python.cp_model.CpSolver'
                    # A deterministic clock crosses the budget only inside the
                    # native call; no sleep or wall-time race decides this test.
                    clock = lambda: 2.0 if reason == 'timeout' and entered.is_set() else 0.0
                    with patch(path, return_value=native), patch('constraint_models.time.monotonic', side_effect=clock):
                        result = solve(problem, engine=engine, seconds=1,
                                       stop=entered.is_set if reason == 'stopped' else None)
                    self.assertTrue(entered.is_set())
                    self.assertTrue(interrupted.is_set())
                    self.assertEqual(result['status'], reason)
                    self.assertFalse(result['complete'])

    @requires_ortools
    def test_traditional_cp_stop_is_observed_inside_native_limit_callback(self):
        from ortools.constraint_solver import pywrapcp
        original = pywrapcp.Solver.CustomLimit
        checked = threading.Event()
        def monitored(solver, callback):
            def invoke():
                checked.set()
                return callback()
            return original(solver, invoke)
        with patch.object(pywrapcp.Solver, 'CustomLimit', monitored):
            result = solve(pair_problem(), engine='cp', seconds=5, stop=checked.is_set)
        self.assertTrue(checked.is_set())
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(result['complete'])

    @requires_solver
    def test_gold_cut_checkpoint_restarts_with_revalidated_witnesses(self):
        problem = pair_problem(multiple_loops=True)
        for engine in ENGINES:
            event = threading.Event()
            def progress(value):
                if value['phase'] == 'gold_cut': event.set()
            with self.subTest(engine=engine):
                first = solve(problem, engine=engine, seconds=5, stop=event, progress=progress)
                self.assertEqual(first['status'], 'stopped')
                self.assertEqual(len(first['checkpoint']['rejected_boards']), 1)
                resumed = solve(problem, engine=engine, seconds=5, resume=first['checkpoint'])
                self.assertEqual(resumed['status'], 'infeasible')
                self.assertTrue(resumed['stats']['resumed'])
                self.assertEqual(resumed['stats']['restored_gold_cuts'], 1)
                self.assertEqual(resumed['stats']['resumption_mode'], 'restart_with_revalidated_gold_cuts')
                tampered = deepcopy(first['checkpoint'])
                tampered['fingerprint'] = 'different puzzle'
                with self.assertRaisesRegex(ValueError, 'different inputs'):
                    solve(problem, engine=engine, seconds=5, resume=tampered)

    @requires_solver
    def test_progress_checkpoints_are_independent_snapshots_of_cut_witnesses(self):
        problem = pair_problem(multiple_loops=True)
        for engine in ENGINES:
            events = []
            with self.subTest(engine=engine):
                result = solve(problem, engine=engine, seconds=5, progress=events.append)
                checkpoints = [event['checkpoint'] for event in events if 'checkpoint' in event]
                self.assertGreaterEqual(len(checkpoints), 2)
                self.assertEqual(checkpoints[0]['rejected_boards'], [])
                self.assertEqual(checkpoints[-1]['rejected_boards'], result['checkpoint']['rejected_boards'])
                self.assertGreater(len(checkpoints[-1]['rejected_boards']), 0)
                self.assertTrue(all('stats' in event for event in events))

    def test_checkpoint_may_not_block_a_real_gold_solution(self):
        problem = pair_problem()
        checkpoint = solve(problem, engine='cp', seconds=0)['checkpoint']
        checkpoint['rejected_boards'] = [list(next(iter(brute(problem, 'gold'))))]
        with self.assertRaisesRegex(ValueError, 'valid Gold solution'):
            solve(problem, engine='cp', seconds=5, resume=checkpoint)

    @requires_solver
    def test_component_cut_can_be_strictly_smaller_than_board(self):
        pair = pair_problem()
        data = [{'id': str(i), 'segments': pair.tiles[i % 2]['segments']} for i in range(4)]
        board = SimpleNamespace(cells=tuple(range(4)), edges=tuple((a, side, a+1, side) for a in (0, 2) for side in range(3)))
        problem = SearchProblem(data, board, fixed={0: 0, 1: 3, 2: 6, 3: 9})
        for engine in ENGINES:
            cuts = []
            with self.subTest(engine=engine):
                result = solve(problem, engine=engine, seconds=5,
                               progress=lambda event: cuts.append(event.get('cut_cells')) if event['phase'] == 'gold_cut' else None)
                self.assertEqual(result['status'], 'infeasible')
                self.assertEqual(cuts, [2])

    @requires_solver
    def test_native_engines_use_verified_rotation_symmetry_on_fresh_and_legacy_resume(self):
        from search_fixtures import planted_gold
        from constraint_symmetry import proper_rotations
        data, board, reference = planted_gold(17,1,0)
        problem = SearchProblem(data,board)
        # Deliberately hint a noncanonical copy. Filtering an inadmissible hint
        # must not change completeness or produce an invalid native model.
        noncanonical = next(codes for rotation in proper_rotations(board)
                            for codes in [rotation.apply(reference)]
                            if board.cells[next(i for i,code in enumerate(codes) if code//3==0)].face != 0)
        original = deepcopy(problem.domains)
        for engine in ENGINES:
            with self.subTest(engine=engine):
                legacy = solve(problem,engine=engine,seconds=0)['checkpoint']
                self.assertNotIn('root_symmetry',legacy)
                result = solve(problem,engine=engine,seconds=5,resume=legacy,hints=noncanonical)
                self.assertEqual(result['status'],'solved')
                self.assertTrue(problem.validate(result['codes'])['valid'])
                self.assertTrue(result['root_symmetry']['reduction']['applied'])
                anchor = next(i for i,code in enumerate(result['codes']) if code//3==0)
                self.assertEqual(board.cells[anchor].face,0)
        self.assertEqual(problem.domains,original)

    def test_native_symmetry_is_skipped_for_fixed_or_narrowed_input_domains(self):
        from search_fixtures import planted_gold
        from constraint_models import _Run, _domains
        data, board, reference = planted_gold(17,1,0)
        cases=[SearchProblem(data,board,fixed={0:reference[0]}),SearchProblem(data,board)]
        cases[1].domains[1].pop()
        for problem in cases:
            run=_Run(problem,'cp-sat',5,0,'gold',None,None)
            rows=_domains(run)
            self.assertFalse(run.root_symmetry['reduction']['applied'])
            self.assertEqual(rows,problem.domains)

    @requires_ortools
    def test_partial_and_discarded_hints_remain_preferences_in_traditional_cp(self):
        problem=pair_problem();expected=brute(problem,'gold')
        for hints in ({0:0},{1:4},{0:999},{0:0,1:999}):
            with self.subTest(hints=hints):
                result=solve(problem,engine='cp',seconds=5,hints=hints)
                self.assertEqual(result['status'],'solved')
                self.assertIn(tuple(result['codes']),expected)

    def test_invalid_limits_and_options_rejected(self):
        problem = pair_problem()
        for seconds in (-1, True, float('nan'), float('inf')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                solve(problem, seconds=seconds)
        for options in ({'engine': 'unknown'}, {'target': 'anything'}, {'seed': True}, {'stop': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve(problem, **options)


if __name__ == '__main__':
    unittest.main()
