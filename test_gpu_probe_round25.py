"""CPU-only correctness and continuation checks; no GPU or performance workload."""
import copy
import itertools
import json
import unittest
from unittest.mock import patch

import constraint_repair_resume_round25 as resume_api
from constraint_repair_round24 import (core_guided_repair, fragment_crossover,
                                      scoped_repair_failure, loop_merge_trade, matched_fragments)
from constraint_lns import _Cancelled
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, double_face, multiple_loops, tetrahedron_tiles, brute
from test_dependencies import requires_ortools


def resign(record):
    record['digest'] = resume_api._digest({key: value for key, value in record.items() if key != 'digest'})
    return record


class FiniteOracle:
    """Independent exhaustive Gold oracle with deliberately interruptible calls."""
    def __init__(self, problem, control, harness):
        control.check(); self.problem, self.control = problem, control
        self.cp_model = None; self.failures = []; self.harness = harness
        self.stats = {name: 0 for name in ('oracle_queries', 'oracle_branches', 'oracle_conflicts', 'oracle_unknown', 'oracle_core_cells')}
        harness['models'] += 1
    def query(self, assumptions):
        self.control.check(); assumptions = tuple(sorted(assumptions))
        self.stats['oracle_queries'] += 1; self.harness['queries'].append(assumptions)
        number = len(self.harness['queries'])
        if number in self.harness.get('unknown', set()):
            self.stats['oracle_unknown'] += 1
            return {'status': 'unknown'}
        if number in self.harness.get('cancel', set()):
            self.control.reason = 'stopped'; raise _Cancelled()
        choices = [row for row in self.harness['solutions'] if all(row[cell] == code for cell, code in assumptions)]
        if choices:
            codes = list(choices[0])
            return {'status': 'solved', 'codes': codes, 'validation': self.problem.validate(codes)}
        self.failures.append(assumptions); self.stats['oracle_core_cells'] += len(assumptions)
        return {'status': 'infeasible', 'core': assumptions}


def finite_hitting(cores, cells, control, cp):
    control.check()
    for size in range(len(cells)+1):
        for candidate in itertools.combinations(cells, size):
            if all(set(candidate).intersection(core) for core in cores): return set(candidate), 0
    return None, 'infeasible'


def finite_master(fragments, conflicts, control, cp):
    control.check(); best = None; objective = -1
    for mask in range(1 << len(fragments)):
        chosen = [i for i in range(len(fragments)) if mask & (1 << i)]
        if any(set(row) <= set(chosen) for row in conflicts): continue
        placements = [pair for i in chosen for pair in fragments[i]['placements']]
        if len({cell for cell, code in placements}) != len(placements): continue
        if len({code//3 for cell, code in placements}) != len(placements): continue
        if len(placements) > objective: best, objective = chosen, len(placements)
    return best, 0


class RepairContinuationTests(unittest.TestCase):
    def fixture(self):
        problem = SearchProblem(*tetrahedron_tiles())
        harness = {'solutions': brute(problem, 'gold'), 'models': 0, 'queries': []}
        return problem, harness

    def oracle(self, harness):
        return patch('constraint_repair_round24.GoldRepairOracle', side_effect=lambda p, c: FiniteOracle(p, c, harness))

    def test_core_pending_master_is_not_repeated_and_minimum_is_exact(self):
        problem, h = self.fixture(); h['unknown'] = {1}; seed = [0, 4, 6, 9]
        with self.oracle(h), patch('constraint_repair_round24._minimum_hitting_set', side_effect=finite_hitting) as master:
            first = core_guided_repair(problem, seed, durable=True, seconds=2)
            self.assertFalse(first['continuation_complete']); self.assertEqual(first['checkpoint']['state']['phase'], 'query')
            self.assertEqual(master.call_count, 1)
            second = core_guided_repair(problem, seed, seconds=2, resume=first['checkpoint'])
            self.assertEqual(second['status'], 'solved', second)
            self.assertEqual(second['stats']['repair_native_retries'], 1)
            self.assertEqual(master.call_count, second['stats']['repair_completed_queries'])
            self.assertEqual(h['queries'][0], h['queries'][1])
            self.assertEqual(second['minimum_changes'], min(sum(a != b for a, b in zip(seed, row)) for row in h['solutions']))
            self.assertGreater(second['stats']['repair_completed_work_reused'], 0)
            count = len(h['queries'])
            again = core_guided_repair(problem, seed, resume=second['checkpoint'], seconds=0)
            self.assertTrue(again['continuation_complete']); self.assertEqual(len(h['queries']), count)

    def test_completed_core_query_survives_stop_and_caps_are_cumulative(self):
        problem, h = self.fixture(); seed = [0, 4, 6, 9]
        def stop(): return len(h['queries']) >= 1
        with self.oracle(h), patch('constraint_repair_round24._minimum_hitting_set', side_effect=finite_hitting):
            first = core_guided_repair(problem, seed, durable=True, seconds=2, stop=stop, max_queries=1)
            self.assertEqual(first['stats']['repair_completed_queries'], 1)
            self.assertTrue(first['cores']); self.assertEqual(len(first['exclusions']['cores']), 1)
            models = h['models']
            last = core_guided_repair(problem, seed, resume=first['checkpoint'], seconds=2, max_queries=1)
            self.assertEqual(h['models'], models)
            self.assertTrue(last['continuation_complete']); self.assertEqual(last['reason'], 'query_limit')
            self.assertEqual(len(h['queries']), 1)
            self.assertNotEqual(last['status'], 'infeasible')

    def test_interrupted_master_is_counted_and_retry_does_not_invent_a_query(self):
        problem, h = self.fixture(); seed = [0, 4, 6, 9]
        master_calls = [0]
        def interrupted(cores, cells, control, cp):
            master_calls[0] += 1
            if master_calls[0] == 1: return None, 'unknown'
            return finite_hitting(cores, cells, control, cp)
        with self.oracle(h), patch('constraint_repair_round24._minimum_hitting_set', side_effect=interrupted):
            first = core_guided_repair(problem, seed, durable=True, seconds=2)
            self.assertFalse(first['continuation_complete']); self.assertEqual(h['queries'], [])
            self.assertEqual(first['stats']['repair_completed_masters'], 0)
            last = core_guided_repair(problem, seed, resume=first['checkpoint'], seconds=2)
            self.assertEqual(last['status'], 'solved'); self.assertEqual(last['stats']['repair_master_retries'], 1)
            self.assertEqual(last['stats']['repair_master_attempts'], last['stats']['repair_completed_masters']+1)
        problem, h = self.fixture(); boards = [[1, 3, 6, 9], [0, 4, 6, 9]]
        with self.oracle(h), patch.object(resume_api, '_fragment_master', return_value=(None, 3)):
            first = fragment_crossover(problem, boards, durable=True, seconds=2, max_fragment=1)
        self.assertEqual(first['stats']['fragment_master_branches'], 3); self.assertEqual(h['queries'], [])
        with self.oracle(h), patch.object(resume_api, '_fragment_master', side_effect=finite_master):
            last = fragment_crossover(problem, boards, resume=first['checkpoint'], seconds=2, max_fragment=1)
        self.assertEqual(last['status'], 'solved'); self.assertEqual(last['stats']['repair_master_retries'], 1)

    def test_fragment_keeps_completed_selection_and_completed_master_cuts(self):
        problem, h = self.fixture(); boards = [[1, 3, 6, 9], [0, 4, 6, 9]]; h['unknown'] = {1, 3}
        with self.oracle(h), patch.object(resume_api, '_fragment_master', side_effect=finite_master) as master:
            first = fragment_crossover(problem, boards, durable=True, seconds=2, max_fragment=1)
            self.assertEqual(master.call_count, 1)
            second = fragment_crossover(problem, boards, resume=first['checkpoint'], seconds=2, max_fragment=1)
            self.assertFalse(second['continuation_complete']); self.assertEqual(len(second['checkpoint']['state']['conflicts']), 1)
            third = fragment_crossover(problem, boards, resume=second['checkpoint'], seconds=2, max_fragment=1)
            self.assertEqual(third['status'], 'solved', third); self.assertTrue(problem.validate(third['best_board'])['valid'])
            self.assertEqual(third['stats']['repair_native_retries'], 2)
            self.assertEqual(master.call_count, third['stats']['repair_completed_queries'])
            self.assertEqual(h['queries'][0], h['queries'][1]); self.assertEqual(h['queries'][2], h['queries'][3])
            self.assertTrue(third['exclusions']['cores'])

    def test_scoped_failure_unknown_and_cancel_retry_without_global_claim(self):
        problem, h = self.fixture(); seed = [1, 3, 6, 9]; h['unknown'] = {1}; h['cancel'] = {2}
        with self.oracle(h):
            first = scoped_repair_failure(problem, seed, [1, 2, 3], durable=True, seconds=2)
            self.assertFalse(first['continuation_complete']); self.assertFalse(first['exclusions']['cores'])
            second = scoped_repair_failure(problem, seed, [1, 2, 3], resume=first['checkpoint'], seconds=2)
            self.assertEqual(second['status'], 'stopped'); self.assertFalse(second['continuation_complete'])
            last = scoped_repair_failure(problem, seed, [1, 2, 3], resume=second['checkpoint'], seconds=2)
            self.assertEqual(last['status'], 'infeasible'); self.assertTrue(last['continuation_complete'])
            self.assertEqual(last['core'], ((0, 1),)); self.assertEqual(last['stats']['repair_native_retries'], 2)
            self.assertEqual(last['stats']['repair_native_attempts'], 3)
            for row in h['solutions']: self.assertFalse(all(row[cell] == code for cell, code in last['core']))
            again = scoped_repair_failure(problem, seed, [1, 2, 3], resume=last['checkpoint'], seconds=0)
            self.assertEqual(len(h['queries']), 3); self.assertEqual(again['status'], 'infeasible')

    def test_trades_resume_every_boundary_without_repeating_candidates_or_histories(self):
        problem = SearchProblem(multiple_loops(), double_face()); seed = [0, 3]
        full = loop_merge_trade(problem, seed, [0, 1], durable=True, seconds=2)
        self.assertTrue(full['complete_local_query']); self.assertNotEqual(full['status'], 'solved')
        legacy = loop_merge_trade(problem, seed, [0, 1], seconds=2)
        self.assertEqual(full['stats']['trade_histories'], legacy['stats']['trade_histories'])
        self.assertEqual(full['stats']['trade_states'], legacy['stats']['trade_states'])
        for allowance in range(1, 10):
            checkpoint = None
            for iteration in range(200):
                calls = [0]
                def stop():
                    calls[0] += 1
                    return calls[0] > allowance
                part = loop_merge_trade(problem, seed, [0, 1], durable=True, resume=checkpoint, seconds=2, stop=stop)
                if part['continuation_complete']: break
                checkpoint = json.loads(json.dumps(part['checkpoint']))
            else: self.fail('Trade continuation did not finish.')
            for key in ('trade_states', 'trade_histories', 'trade_improvements'):
                self.assertEqual(part['stats'][key], full['stats'][key], (allowance, key))
            self.assertEqual(part['best_board'], full['best_board'])

    def test_trade_cap_never_proves_exhaustion_and_fixed_boundary_is_preserved(self):
        problem = SearchProblem(multiple_loops(), double_face()); seed = [0, 3]
        part = loop_merge_trade(problem, seed, [0, 1], durable=True, seconds=2, max_states=1)
        self.assertTrue(part['continuation_complete']); self.assertFalse(part['complete_local_query'])
        self.assertEqual(part['status'], 'unknown'); self.assertEqual(part['stats']['trade_states'], 1)
        again = loop_merge_trade(problem, seed, [0, 1], resume=part['checkpoint'], seconds=2, max_states=1)
        self.assertEqual(again['stats']['trade_states'], 1)
        tiles = [{'id': i, 'segments': [[[0, 2], [1, 2]], [[0, 10], [1, 10]], [[2, 2], [2, 10]]]} for i in range(2)]
        fixed = SearchProblem(tiles, double_face(), fixed={0: 0})
        solved = loop_merge_trade(fixed, seed, [0, 1], durable=True, seconds=2)
        self.assertEqual(solved['status'], 'solved'); self.assertEqual(solved['best_board'][0], 0)

    def test_zero_budget_no_native_build_and_unavailable_is_finished(self):
        problem = SearchProblem(simple_tiles(), double_face())
        with patch('constraint_repair_round24.GoldRepairOracle', side_effect=AssertionError('must not build')):
            for fn, args in ((core_guided_repair, ([0, 3],)), (fragment_crossover, ([[0, 3]],)),
                             (scoped_repair_failure, ([0, 3], [0]))):
                result = fn(problem, *args, durable=True, seconds=0)
                self.assertEqual(result['status'], 'timeout'); self.assertFalse(result['continuation_complete'])
        with patch('constraint_repair_round24.GoldRepairOracle', side_effect=ImportError):
            result = core_guided_repair(problem, [0, 3], durable=True, seconds=1)
            self.assertEqual(result['status'], 'unavailable'); self.assertTrue(result['continuation_complete'])

    def test_scope_policy_semantics_and_digest_changes_rejected_before_native(self):
        problem, h = self.fixture(); seed = [0, 4, 6, 9]
        record = core_guided_repair(problem, seed, durable=True, seconds=0)['checkpoint']
        with patch('constraint_repair_round24.GoldRepairOracle', side_effect=AssertionError('must not build')):
            for kwargs in ({'max_queries': 127}, {'codes': [1, 3, 6, 9]}):
                options = {'codes': seed, 'resume': record, 'seconds': 1, **kwargs}
                with self.assertRaises(ValueError): core_guided_repair(problem, **options)
            broken = copy.deepcopy(record); broken['state']['stats']['queries'] = 1
            with self.assertRaisesRegex(ValueError, 'digest'): core_guided_repair(problem, seed, resume=broken)
            problem.domains[0] = problem.domains[0][1:]
            with self.assertRaisesRegex(ValueError, 'scope'): core_guided_repair(problem, seed, resume=record)
            problem.domains[0].insert(0, 0)
            segment = problem.tiles[0]['segments'][0]
            problem.tiles[0]['segments'] = (((segment[0][0], segment[0][1]+1), segment[1]),)
            with self.assertRaisesRegex(ValueError, 'scope'): core_guided_repair(problem, seed, resume=record)

    def test_mutation_during_slice_cannot_publish_old_scope_checkpoint(self):
        problem = SearchProblem(multiple_loops(), double_face())
        def change():
            problem.domains[0] = problem.domains[0][1:]
            return True
        with self.assertRaisesRegex(ValueError, 'changed during'):
            loop_merge_trade(problem, [0, 3], [0, 1], durable=True, seconds=2, stop=change)

    def test_strict_malformed_payloads_and_cursor_rejected_atomically(self):
        problem = SearchProblem(multiple_loops(), double_face()); seed = [0, 3]
        original = loop_merge_trade(problem, seed, [0, 1], durable=True, seconds=0)['checkpoint']
        mutations = [lambda s: s['stats'].__setitem__('trade_states', True),
                     lambda s: s['stats'].__setitem__('trade_states', 200001),
                     lambda s: s.__setitem__('phase', 'query'),
                     lambda s: s['walk'].__setitem__('next', [1, 0]),
                     lambda s: s['walk'].update(prefix=[0], next=[0, 0]),
                     lambda s: s.__setitem__('retry', 1),
                     lambda s: s.__setitem__('status', 'solved')]
        for mutate in mutations:
            bad = copy.deepcopy(original); mutate(bad['state']); resign(bad)
            with self.assertRaises(ValueError): loop_merge_trade(problem, seed, [0, 1], resume=bad)
        self.assertEqual(original['state']['walk'], {'prefix': [], 'next': [0]})
        with patch.object(resume_api, 'MAX_BYTES', 10):
            with self.assertRaisesRegex(ValueError, 'byte cap'): loop_merge_trade(problem, seed, [0, 1], resume=original)

    def test_results_own_checkpoint_and_stop_callback_errors_propagate(self):
        problem = SearchProblem(multiple_loops(), double_face())
        result = loop_merge_trade(problem, [0, 3], [0, 1], durable=True, seconds=0)
        result['stats']['trade_states'] = 999; result['best_board'][0] = 5
        self.assertEqual(result['checkpoint']['state']['stats']['trade_states'], 0)
        self.assertEqual(result['checkpoint']['state']['best_board'], [0, 3])
        def broken(): raise RuntimeError('callback failed')
        with self.assertRaisesRegex(RuntimeError, 'callback failed'):
            loop_merge_trade(problem, [0, 3], [0, 1], durable=True, seconds=1, stop=broken)
        for error in (ImportError('callback import'), RuntimeError('callback runtime')):
            def failing_callback(): raise error
            with self.assertRaises(type(error)):
                core_guided_repair(problem, [0, 3], durable=True, seconds=1, stop=failing_callback)
        with self.assertRaises(ValueError): core_guided_repair(problem, [0, 3], durable=1)
        for seconds in (True, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError): core_guided_repair(problem, [0, 3], durable=True, seconds=seconds)

    @requires_ortools
    def test_actual_native_modes_preserve_full_gold_and_finished_replay(self):
        problem = SearchProblem(*tetrahedron_tiles()); seed = [0, 4, 6, 9]
        cases = ((core_guided_repair, (seed,), {}),
                 (fragment_crossover, ([[1, 3, 6, 9], seed],), {'max_fragment': 1}),
                 (scoped_repair_failure, ([1, 3, 6, 9], [1, 2, 3]), {}))
        for fn, args, kwargs in cases:
            result = fn(problem, *args, durable=True, seconds=5, **kwargs)
            self.assertTrue(result['continuation_complete'], result)
            self.assertIn(result['status'], ('solved', 'infeasible'))
            if result['status'] == 'solved': self.assertTrue(problem.validate(result['best_board'])['valid'])
            with patch('constraint_repair_round24.GoldRepairOracle', side_effect=AssertionError('finished query replayed')):
                again = fn(problem, *args, resume=result['checkpoint'], seconds=0, **kwargs)
            self.assertEqual(again['status'], result['status'])


if __name__ == '__main__': unittest.main()
