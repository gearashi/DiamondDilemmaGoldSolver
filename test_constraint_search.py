"""CPU-only lifecycle tests for Gold controls, saved state, and live sampling."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import queue
import signal
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import constraint_search as runner
from io_utils import atomic_json
from search_problem import SearchProblem
from test_dependencies import requires_ortools, requires_sat


def pair_problem(multiple_loops=False):
    if multiple_loops:
        first = [[[0, 2], [0, 4]], [[1, 2], [1, 4]]]
        second = [[[0, 8], [0, 10]], [[1, 8], [1, 10]]]
    else:
        first, second = [[[0, 2], [1, 4]]], [[[0, 10], [1, 8]]]
    data = [{'id': 'A', 'segments': first}, {'id': 'B', 'segments': second}]
    board = SimpleNamespace(cells=(0, 1), edges=tuple((0, side, 1, side) for side in range(3)))
    return data, SearchProblem(data, board)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(runner, 'validate_runtime_selection'))
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix='diamond-constraint-runner-'))
        self.root = Path(self.directory).resolve()
        self.run = self.root/'runtime'
        self.input = self.root/'tiles.json'
        self.data, self.problem = pair_problem()
        self.input.write_text(json.dumps(self.data), encoding='utf-8')
        self.model = self.enterContext(patch.object(runner, 'SearchProblem', return_value=self.problem))
        self.render = self.enterContext(patch('render_board.render'))
        self.statuses = []
        def record(path, value):
            if Path(path).name == 'status.json':
                self.statuses.append(deepcopy(value))
            atomic_json(path, value)
        self.write = self.enterContext(patch.object(runner, 'atomic_json', side_effect=record))
        self.handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def args(self, algorithm='cp-sat', *extra):
        return ['--algorithm', algorithm, '--seconds', '5', '--replicas', '128',
                '--data', str(self.input), '--output', str(self.run), *extra]

    def read(self, relative):
        return json.loads((self.run/relative).read_text(encoding='utf-8'))

    def terminal(self, status='timeout', **extra):
        return {'status': status, 'nodes': 7, 'stats': {'branches': 7}, 'complete': False,
                'checkpoint': {'saved': 'final'}, **extra}

    def assert_released(self):
        self.assertFalse((self.run/'run.lock').exists())
        self.assertEqual(self.handlers, {sig: signal.getsignal(sig) for sig in self.handlers})

    @requires_ortools
    def test_direct_circuit_route_runs_and_snapshots_its_model(self):
        runner.main(self.args('cp-sat-circuit'))
        state=self.read('status.json')
        self.assertEqual(state['state'],'solved')
        self.assertTrue(self.read('solution.json')['validation']['valid'])
        self.assertEqual(state['stats']['connectivity_encoding'],'direct_optional_endpoint_circuit')
        self.assertEqual(state['stats']['gold_cuts'],0)
        record=self.read('searches/cp-sat-circuit-gold/solution.json')
        self.assertTrue((Path(record['input_snapshot']).parent/'constraint_circuit.py').is_file())
        self.assertTrue((Path(record['input_snapshot']).parent/'constraint_residual.py').is_file())
        self.assertFalse((self.run/'searches/cp-sat-gold').exists())
        self.assert_released()

    @requires_ortools
    def test_direct_circuit_model_cap_is_visible_and_never_exhausted(self):
        from constraint_circuit import CircuitLimit
        with patch('constraint_circuit.build_circuit_model',side_effect=CircuitLimit('fixture arc limit')), self.assertRaisesRegex(SystemExit,'fixture arc limit'):
            runner.main(self.args('cp-sat-circuit'))
        state=self.read('status.json')
        self.assertEqual(state['state'],'error');self.assertFalse(state['complete'])
        self.assertIn('unsupported',state['error'])
        self.assertEqual(self.read('searches/cp-sat-circuit-gold/last-result.json')['status'],'unsupported')
        self.assertFalse((self.run/'searches/cp-sat-circuit-gold/completed-search.json').exists())
        self.assertFalse((self.run/'solution.json').exists())
        self.assert_released()

    def test_progress_checkpoint_phase_and_final_result_are_saved(self):
        def solve(problem, algorithm, **kwargs):
            self.assertIs(problem, self.problem)
            self.assertEqual(kwargs['target'], 'gold')
            kwargs['progress']({'phase': 'building_model', 'nodes': 2, 'stats': {'clauses': 31},
                                'checkpoint': {'saved': 'periodic'}})
            kwargs['progress']({'phase': 'searching', 'codes': [0, -1], 'nodes': 3})
            return self.terminal()
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args())
        state = self.read('status.json')
        self.assertEqual(state['state'], 'time_limit')
        self.assertEqual(state['nodes_checked'], 7)
        self.assertEqual(state['stats'], {'branches': 7})
        self.assertFalse(state['complete'])
        saved = self.read('searches/cp-sat-gold/checkpoint.json')
        self.assertEqual(saved['payload'], {'saved': 'final'})
        self.assertEqual(saved['algorithm'], 'cp-sat')
        self.assertEqual(saved['target'], 'gold')
        live = self.read('live.json')
        self.assertEqual(live['codes'], [0, -1])
        self.assertTrue(live['is_partial'])
        self.assertIsNone(live['validation'])
        snapshot = Path(live['input_snapshot'])
        self.assertEqual(snapshot.read_bytes(), self.input.read_bytes())
        self.assertTrue((snapshot.parent/'sampling_kernels.wgsl').is_file())
        self.assertFalse((snapshot.parent/'gpu_sampling.wgsl').exists())
        checkpoint_writes = [call.args[1]['payload'] for call in self.write.call_args_list
                             if Path(call.args[0]).name == 'checkpoint.json']
        self.assertIn({'saved': 'periodic'}, checkpoint_writes)
        self.assertEqual(self.read('searches/cp-sat-gold/last-result.json')['status'], 'timeout')
        self.assert_released()

    def test_terminal_return_writes_final_once_for_timeout_stop_and_error(self):
        for status in ('timeout', 'stopped', 'error'):
            with self.subTest(status=status):
                self.run = self.root/status
                self.write.reset_mock()
                result = self.terminal(status, error='fixture final error')
                with patch.object(runner, 'run_solver', return_value=result):
                    if status == 'error':
                        with self.assertRaisesRegex(SystemExit, 'fixture final error'):
                            runner.main(self.args())
                    else:
                        runner.main(self.args())
                writes = [call.args[1]['payload'] for call in self.write.call_args_list
                          if Path(call.args[0]).name == 'checkpoint.json']
                self.assertEqual(writes, [{'saved': 'final'}])
                self.assertEqual(self.read('searches/cp-sat-gold/checkpoint.json')['payload'], result['checkpoint'])
                self.assert_released()

    def test_continuing_slice_checkpoint_and_changed_final_both_persist(self):
        results = [self.terminal('timeout', checkpoint={'saved': 'periodic'}),
                   self.terminal('stopped', checkpoint={'saved': 'changed final'})]
        def solve(problem, algorithm, **kwargs):
            if len(results) == 1:
                self.assertEqual(kwargs['resume'], {'saved': 'periodic'})
            return results.pop(0)
        with patch.object(runner, 'run_solver', side_effect=solve) as producer:
            runner.main(self.args('dfs'))
        self.assertEqual(producer.call_count, 2)
        writes = [call.args[1]['payload'] for call in self.write.call_args_list
                  if Path(call.args[0]).name == 'checkpoint.json']
        self.assertEqual(writes, [{'saved': 'periodic'}, {'saved': 'changed final'}])
        self.assertEqual(self.read('searches/dfs-gold/checkpoint.json')['payload'], {'saved': 'changed final'})
        self.assert_released()

    def test_stop_or_time_crossing_progress_enqueue_keeps_latest_result_metadata(self):
        original_queue = queue.Queue
        for crossing in ('time', 'stop'):
            with self.subTest(crossing=crossing):
                self.run = self.root/crossing
                clock = [0.0]
                stop_request = []
                class CrossingQueue(original_queue):
                    def put(self, event, *args, **kwargs):
                        super().put(event, *args, **kwargs)
                        if isinstance(event, dict) and event.get('event') == 'checkpoint':
                            # Deterministically model time/Stop arriving during
                            # a potentially blocking producer progress enqueue.
                            if crossing == 'time':
                                clock[0] = 6.0
                            else:
                                stop_request[0]()
                def new_queue(*, maxsize):
                    return CrossingQueue(maxsize=maxsize) if maxsize == runner.EVENT_CAPACITY else original_queue(maxsize=maxsize)
                def solve(problem, algorithm, **kwargs):
                    stop_request.append(kwargs['stop'].__self__.set)
                    return self.terminal('timeout', nodes=71, stats={'branches':71},
                                         metadata={'latest':True}, native_backend='python')
                with patch.object(runner.queue, 'Queue', side_effect=new_queue), \
                        patch.object(runner.time, 'monotonic', side_effect=lambda:clock[0]), \
                        patch.object(runner, 'run_solver', side_effect=solve) as producer:
                    runner.main(self.args('dfs'))
                producer.assert_called_once()
                result = self.read('searches/dfs-gold/last-result.json')
                self.assertEqual(result['metadata'], {'latest':True})
                self.assertEqual(result['nodes'], 71)
                self.assertEqual(result['stats'], {'branches':71})
                self.assertEqual(result['native_backend'], 'python')
                self.assertEqual(self.read('searches/dfs-gold/checkpoint.json')['payload'], {'saved':'final'})
                self.assertEqual(self.read('status.json')['state'], 'time_limit' if crossing == 'time' else 'stopped')
                self.assert_released()

    def test_cancelled_producer_without_progress_still_flushes_final(self):
        def solve(problem, algorithm, **kwargs):
            kwargs['stop'].__self__.set()
            kwargs['progress']({'phase': 'finished', 'checkpoint': {'saved': 'suppressed'}})
            return self.terminal('stopped')
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args())
        writes = [call.args[1]['payload'] for call in self.write.call_args_list
                  if Path(call.args[0]).name == 'checkpoint.json']
        self.assertEqual(writes, [{'saved': 'final'}])
        self.assertEqual(self.read('status.json')['state'], 'stopped')
        self.assert_released()

    def test_final_checkpoint_write_failure_preserves_prior_file_and_unlocks(self):
        with patch.object(runner, 'run_solver', return_value=self.terminal('stopped', checkpoint={'saved': 'prior'})):
            runner.main(self.args())
        checkpoint_path = self.run/'searches/cp-sat-gold/checkpoint.json'
        before = checkpoint_path.read_bytes()
        def fail_checkpoint(path, value):
            if Path(path) == checkpoint_path:
                raise OSError('fixture final checkpoint failure')
            atomic_json(path, value)
        with patch.object(runner, 'run_solver', return_value=self.terminal('stopped')), \
                patch.object(runner, 'atomic_json', side_effect=fail_checkpoint), \
                self.assertRaisesRegex(SystemExit, 'fixture final checkpoint failure'):
            runner.main(self.args('cp-sat', '--resume'))
        self.assertEqual(checkpoint_path.read_bytes(), before)
        self.assertEqual(self.read('status.json')['state'], 'error')
        self.assert_released()

    def test_resume_passes_only_matching_algorithm_checkpoint(self):
        with patch.object(runner, 'run_solver', return_value=self.terminal()):
            runner.main(self.args())
        saved_bytes = (self.run/'searches/cp-sat-gold/checkpoint.json').read_bytes()
        with patch.object(runner, 'run_solver', return_value=self.terminal('stopped')) as solve:
            runner.main(self.args('sat', '--resume'))
        self.assertIsNone(solve.call_args.kwargs['resume'])
        self.assertEqual((self.run/'searches/cp-sat-gold/checkpoint.json').read_bytes(), saved_bytes)
        with patch.object(runner, 'run_solver', return_value=self.terminal('stopped')) as solve:
            runner.main(self.args('cp-sat', '--resume'))
        self.assertEqual(solve.call_args.kwargs['resume'], {'saved': 'final'})
        self.assert_released()

    def test_wrong_checkpoint_binding_is_preserved_and_rejected(self):
        with patch.object(runner, 'run_solver', return_value=self.terminal()):
            runner.main(self.args())
        path = self.run/'searches/cp-sat-gold/checkpoint.json'
        saved = self.read('searches/cp-sat-gold/checkpoint.json')
        saved['binding'] = 'different input'
        atomic_json(path, saved)
        before = path.read_bytes()
        with patch.object(runner, 'run_solver') as solve, self.assertRaisesRegex(SystemExit, 'different data'):
            runner.main(self.args('cp-sat', '--resume'))
        solve.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.read('status.json')['state'], 'error')
        self.assert_released()

    def test_existing_stop_request_is_seen_before_native_search(self):
        self.run.mkdir()
        (self.run/'stop.request').write_text('stop', encoding='utf-8')
        def solve(problem, algorithm, **kwargs):
            self.assertTrue(kwargs['stop']())
            return self.terminal('stopped')
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('cp', '--stop-file-initialized'))
        self.assertEqual(self.read('status.json')['state'], 'stopped')
        self.assertEqual(self.read('searches/cp-gold/checkpoint.json')['payload'], {'saved': 'final'})
        self.assertTrue(any(status['state'] == 'stopping' for status in self.statuses))
        self.assert_released()

    def test_stop_is_not_starved_by_continuous_progress(self):
        produced = []
        def solve(problem, algorithm, **kwargs):
            (self.run/'stop.request').write_text('stop', encoding='utf-8')
            for index in range(10000):
                if kwargs['stop']():
                    break
                produced.append(index)
                kwargs['progress']({'phase': 'searching', 'nodes': index})
            self.assertTrue(kwargs['stop']())
            return self.terminal('stopped')
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args())
        self.assertLess(len(produced), 10000)
        self.assertEqual(self.read('status.json')['state'], 'stopped')
        self.assert_released()

    def test_unlimited_passes_large_budget_and_still_stops_on_gold(self):
        self.assertTrue(self.problem.validate([0, 3])['valid'])
        with patch.object(runner, 'run_solver', return_value=self.terminal('solved', codes=[0, 3], complete=True)) as solve:
            runner.main(self.args('cp-sat', '--seconds', '0'))
        self.assertGreater(solve.call_args.kwargs['seconds'], 86400)
        self.assertEqual(self.read('status.json')['time_limit_seconds'], 0)
        self.assertEqual(self.read('status.json')['state'], 'solved')
        self.assertTrue(self.read('solution.json')['validation']['valid'])
        self.assert_released()

    def test_multi_loop_cannot_be_promoted_to_gold(self):
        self.data, self.problem = pair_problem(multiple_loops=True)
        self.model.return_value = self.problem
        self.input.write_text(json.dumps(self.data), encoding='utf-8')
        report = self.problem.validate([0, 3])
        self.assertEqual(report['matched_edges'], 3)
        self.assertFalse(report['valid'])
        with patch.object(runner, 'run_solver', return_value=self.terminal('solved', codes=[0, 3], complete=True)), self.assertRaisesRegex(SystemExit, 'single-loop Gold'):
            runner.main(self.args())
        self.assertFalse((self.run/'solution.json').exists())
        self.assertEqual(self.read('status.json')['state'], 'error')
        self.assert_released()

    def test_prior_solution_is_freshly_validated_and_rebound_to_current_run(self):
        saved = self.run/'searches/cp-sat-gold/solution.json'
        atomic_json(saved, {'codes': [0, 3], 'validation': {'valid': False, 'bogus': True},
                            'run_id': 'old-run', 'input_sha256': 'stale', 'algorithm': 'wrong'})
        with patch.object(runner, 'run_solver') as solve:
            runner.main(self.args('cp-sat', '--resume'))
        solve.assert_not_called()
        record = self.read('solution.json')
        self.assertEqual(record['algorithm'], 'cp-sat')
        self.assertNotEqual(record['run_id'], 'old-run')
        self.assertEqual(record['source_solution_run_id'], 'old-run')
        self.assertEqual(record['input_sha256'], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(record['validation'], self.problem.validate([0, 3]))
        self.assertEqual(Path(record['input_snapshot']).read_bytes(), self.input.read_bytes())
        self.assertEqual(self.read('best.json'), record)
        self.assertEqual(self.read('live.json'), record)
        self.assertTrue(self.read('status.json')['complete'])
        self.render.assert_called_once()
        self.assert_released()

    def test_initialization_failure_and_publish_failure_release_lock(self):
        original_mkdir = Path.mkdir
        target = self.run/'searches/cp-sat-gold'
        def failing_mkdir(path, *args, **kwargs):
            if path == target:
                raise OSError('fixture directory failure')
            return original_mkdir(path, *args, **kwargs)
        with patch.object(Path, 'mkdir', failing_mkdir), self.assertRaisesRegex(SystemExit, 'directory failure'):
            runner.main(self.args())
        self.assert_released()
        with patch.object(runner, 'atomic_json', side_effect=OSError('fixture disk failure')), self.assertRaisesRegex(SystemExit, 'disk failure'):
            runner.main(self.args())
        self.assert_released()

    def test_invalid_live_event_cancels_and_joins_worker_before_unlock(self):
        finished = threading.Event()
        def solve(problem, algorithm, **kwargs):
            try:
                kwargs['progress']({'codes': [999, -1]})
                while not kwargs['stop']():
                    kwargs['progress']({'phase': 'searching'})
                return self.terminal('stopped')
            finally:
                finished.set()
        with patch.object(runner, 'run_solver', side_effect=solve), self.assertRaisesRegex(SystemExit, 'invalid orientation'):
            runner.main(self.args())
        self.assertTrue(finished.is_set())
        self.assert_released()

    def test_existing_lock_is_never_removed(self):
        self.run.mkdir()
        (self.run/'run.lock').write_text('other-process', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'solver lock exists'):
            runner.main(self.args())
        self.assertEqual((self.run/'run.lock').read_text(), 'other-process')

    def test_gpu_sampling_progress_and_final_report_are_visible(self):
        report = {'status': 'timeout', 'backend': 'webgpu', 'device': 'Mock adapter', 'replicas': 128,
                  'samples_considered': 256, 'move_proposals': 128, 'best_score': 1,
                  'top_board': [0, 4], 'top_samples': [], 'complete': False,
                  'duplicates_tracked': False, 'unique_boards': None}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'sampling', **report})
            return self.terminal('stopped', gpu_sampling=report)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('hybrid'))
        state = self.read('status.json')
        self.assertEqual(state['gpu_sampling'], report)
        self.assertEqual(state['backend'], 'webgpu')
        self.assertIn('Mock adapter', state['gpu'])
        live = self.read('live.json')
        self.assertEqual(live['snapshot_kind'], 'gpu_sample')
        self.assertEqual(state.get('max_depth', 0), 0)
        self.assertEqual(live['codes'], [0, 4])
        self.assertEqual(live['validation'], self.problem.validate([0, 4]))
        self.assertFalse((self.run/'solution.json').exists())
        self.assert_released()

    def test_final_sampling_report_does_not_overwrite_later_dfs_snapshot(self):
        report = {'backend': 'cuda', 'top_board': [0, 4], 'best_score': 1}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'sampling', **report})
            kwargs['progress']({'event': 'progress', 'codes': [0, -1]})
            return self.terminal('stopped', gpu_sampling=report)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('hybrid'))
        self.assertEqual(self.read('live.json')['codes'], [0, -1])
        self.assertEqual(self.read('live.json')['snapshot_kind'], 'search')
        self.assertEqual(self.read('status.json')['gpu_sampling'], report)
        self.assert_released()

    def test_final_only_sampling_report_provides_snapshot(self):
        report = {'backend': 'cuda', 'top_board': [0, 4], 'best_score': 1}
        with patch.object(runner, 'run_solver', return_value=self.terminal('stopped', gpu_sampling=report)):
            runner.main(self.args('hybrid'))
        self.assertEqual(self.read('live.json')['snapshot_kind'], 'gpu_sample')
        self.assertEqual(self.read('status.json')['gpu_sampling'], report)
        self.assert_released()

    def test_exhausted_result_resume_does_not_replay_search(self):
        with patch.object(runner, 'run_solver', return_value=self.terminal('infeasible', complete=True, checkpoint=None)):
            runner.main(self.args())
        recorded = self.read('searches/cp-sat-gold/completed-search.json')
        self.assertEqual(recorded['sha256'], runner.terminal_digest(recorded))
        with patch.object(runner, 'run_solver') as solve:
            runner.main(self.args('cp-sat', '--resume'))
        solve.assert_not_called()
        state = self.read('status.json')
        self.assertEqual(state['state'], 'exhausted')
        self.assertTrue(state['complete'])
        self.assertTrue(state['resumed_complete'])
        self.assertEqual(state['source_result_run_id'], recorded['run_id'])
        self.assertEqual(state['nodes_checked'], 7)
        self.assert_released()

    def test_corrupted_exhaustion_marker_is_rejected_without_overwriting_it(self):
        with patch.object(runner, 'run_solver', return_value=self.terminal('infeasible', complete=True, checkpoint=None)):
            runner.main(self.args())
        path = self.run/'searches/cp-sat-gold/completed-search.json'
        recorded = self.read('searches/cp-sat-gold/completed-search.json')
        recorded['result']['nodes'] = 123456
        atomic_json(path, recorded)
        before = path.read_bytes()
        with patch.object(runner, 'run_solver') as solve, self.assertRaisesRegex(SystemExit, 'Completed search evidence'):
            runner.main(self.args('cp-sat', '--resume'))
        solve.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assert_released()

    def test_saved_solution_final_status_failure_is_not_hidden_by_early_return(self):
        atomic_json(self.run/'searches/cp-sat-gold/solution.json', {'codes': [0, 3], 'run_id': 'old'})
        def failing_publish(path, value):
            if Path(path).name == 'status.json' and value.get('state') == 'solved':
                raise OSError('fixture final status failure')
            atomic_json(path, value)
        with patch.object(runner, 'atomic_json', side_effect=failing_publish), self.assertRaisesRegex(SystemExit, 'final status failure'):
            runner.main(self.args('cp-sat', '--resume'))
        self.assert_released()

    def test_seed_uses_fresh_score_and_ignores_malformed_saved_board(self):
        self.run.mkdir()
        a, b, bad = [self.run/name for name in ('a.json', 'b.json', 'bad.json')]
        atomic_json(a, {'codes': [0, 4], 'validation': {'matched_edges': 999}})
        atomic_json(b, {'codes': [0, 3]})
        atomic_json(bad, {'codes': [0, 1]})
        self.assertEqual(runner.load_seed_board(self.problem, [bad, a, b]), [0, 3])
        self.assertIsNone(runner.load_seed_board(self.problem, [bad, self.run/'missing']))

    def test_repair_result_is_visible_and_not_counted_as_dfs_depth(self):
        report = {'status': 'timeout', 'best_board': [0, 4], 'best_score': 1, 'attempts': 3}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'repair', 'phase': 'repair', **report})
            return self.terminal('stopped', cpu_repair=report)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('hybrid'))
        self.assertEqual(self.read('status.json')['cpu_repair'], report)
        self.assertEqual(self.read('live.json')['snapshot_kind'], 'repair_sample')
        self.assertEqual(self.read('status.json').get('max_depth', 0), 0)
        self.assertEqual(self.read('searches/hybrid-gold/heuristic-best.json')['codes'], [0, 4])
        self.assertFalse((self.run/'solution.json').exists())
        self.assert_released()

    def test_repair_final_report_preserves_later_dfs_live_branch(self):
        report = {'status': 'timeout', 'best_board': [0, 4], 'best_score': 1, 'attempts': 3}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'repair', **report})
            kwargs['progress']({'event': 'progress', 'codes': [0, -1]})
            return self.terminal('stopped', cpu_repair=report)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('hybrid'))
        self.assertEqual(self.read('live.json')['codes'], [0, -1])
        self.assertEqual(self.read('live.json')['snapshot_kind'], 'search')

    def test_hybrid_passes_saved_seed_and_keeps_learning_between_slices(self):
        atomic_json(self.run/'best.json', {'codes': [0, 3]})
        seen = []
        def solve(problem, algorithm, **kwargs):
            self.assertEqual(kwargs['seed_board'], [0, 3])
            if not seen:
                kwargs['learning_state']['fixture'] = 42
                seen.append(kwargs['learning_state'])
                return self.terminal('timeout')
            self.assertIs(kwargs['learning_state'], seen[0])
            self.assertEqual(kwargs['learning_state']['fixture'], 42)
            return self.terminal('stopped')
        with patch.object(runner, 'run_solver', side_effect=solve) as search:
            runner.main(self.args('hybrid'))
        self.assertEqual(search.call_count, 2)
        self.assert_released()

    def test_actual_lns_event_schema_is_a_repair_snapshot(self):
        report = {'status': 'timeout', 'best_board': [0, 4], 'best_score': 1, 'stats': {'attempts': 2}}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'repair_progress', 'phase': 'repair',
                                'codes': [0, 4], 'best_score': 1, 'stats': {'attempts': 2}})
            return self.terminal('stopped', local_repair=report)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('hybrid'))
        self.assertEqual(self.read('live.json')['snapshot_kind'], 'repair_sample')
        self.assertEqual(self.read('status.json')['cpu_repair'], report)
        self.assertEqual(self.read('status.json').get('max_depth', 0), 0)
        self.assert_released()

    def test_symmetry_status_survives_progress_and_final_result(self):
        symmetry = {'version': 1, 'state': 'ready', 'reduction': {'applied': True, 'rotations': 10}}
        def solve(problem, algorithm, **kwargs):
            kwargs['progress']({'event': 'progress', 'root_symmetry': symmetry})
            return self.terminal('stopped', root_symmetry=symmetry)
        with patch.object(runner, 'run_solver', side_effect=solve):
            runner.main(self.args('dfs'))
        self.assertEqual(self.read('status.json')['root_symmetry'], symmetry)
        self.assert_released()

    def test_parallel_import_preserves_original_dfs_and_passes_options(self):
        source = self.run/'searches/dfs-gold/checkpoint.json'
        payload = {'fixture':'legacy DFS pending tree'}
        binding = hashlib.sha256(self.input.read_bytes()+b'dfs:gold:v1').hexdigest()
        atomic_json(source, {'version':1,'algorithm':'dfs','target':'gold','binding':binding,'payload':payload})
        original = source.read_bytes()
        def solve(problem, algorithm, **kwargs):
            self.assertEqual(algorithm,'parallel-dfs')
            self.assertEqual(kwargs['workers'],2)
            self.assertFalse(kwargs['options']['lookahead'])
            self.assertEqual(kwargs['resume'],payload)
            return self.terminal('stopped',native_backend='python',parallel={'pending_jobs':3})
        with patch.object(runner,'run_solver',side_effect=solve):
            runner.main(self.args('parallel-dfs','--resume','--workers','2',
                '--search-options','{"lookahead":false,"native":"python"}'))
        self.assertEqual(source.read_bytes(),original)
        status=self.read('status.json')
        self.assertEqual(status['workers'],2)
        self.assertEqual(status['native_backend'],'python')
        self.assertEqual(status['parallel']['pending_jobs'],3)
        self.assertEqual(self.read('searches/parallel-dfs-gold/checkpoint.json')['payload'],{'saved':'final'})
        self.assert_released()

    def test_parallel_import_rejects_different_input_before_search(self):
        source=self.run/'searches/dfs-gold/checkpoint.json'
        atomic_json(source,{'version':1,'algorithm':'dfs','target':'gold','binding':'other','payload':{}})
        with patch.object(runner,'run_solver') as solve,self.assertRaisesRegex(SystemExit,'cannot be imported'):
            runner.main(self.args('parallel-dfs','--resume'))
        solve.assert_not_called()
        self.assert_released()

    def test_parallel_existing_queue_takes_precedence_over_dfs_import(self):
        source=self.run/'searches/dfs-gold/checkpoint.json'
        atomic_json(source,{'binding':'intentionally unrelated'})
        binding=hashlib.sha256(self.input.read_bytes()+b'parallel-dfs:gold:v1').hexdigest()
        atomic_json(self.run/'searches/parallel-dfs-gold/checkpoint.json',{'version':1,'algorithm':'parallel-dfs','target':'gold','binding':binding,'payload':{'queue':'existing'}})
        def solve(problem,algorithm,**kwargs):
            self.assertEqual(kwargs['resume'],{'queue':'existing'})
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=solve):
            runner.main(self.args('parallel-dfs','--resume'))
        self.assert_released()

    def test_queue_drain_has_explicit_bound_and_polls_stop_per_event(self):
        events, consumed, polls = queue.Queue(), [], []
        events.put({'value': 0})
        def consume(event):
            consumed.append(event['value'])
            events.put({'value': event['value']+1})
        count = runner.drain_events(events, consume, limit=7, poll_stop=lambda: polls.append(True))
        self.assertEqual(count, 7)
        self.assertEqual(consumed, list(range(7)))
        self.assertEqual(len(polls), 7)
        self.assertEqual(events.qsize(), 1)


    def loop_fixture(self):
        from constraint_knowledge import KnowledgeStore
        self.data,self.problem=pair_problem(multiple_loops=True)
        self.model.return_value=self.problem
        self.input.write_text(json.dumps(self.data),encoding='utf-8')
        store=KnowledgeStore(self.problem)
        store.add([[0,0],[1,3]])
        return store.snapshot()

    def test_verified_knowledge_survives_algorithm_change_without_rewriting(self):
        payload=self.loop_fixture()
        def first(problem,algorithm,**kwargs):
            self.assertEqual(len(kwargs['learning_state']['_knowledge']),0)
            kwargs['progress']({'event':'progress','knowledge':payload})
            return self.terminal('stopped',knowledge=payload)
        with patch.object(runner,'run_solver',side_effect=first):runner.main(self.args('dfs'))
        path=self.run/'knowledge-gold.json';original=path.read_bytes()
        self.assertEqual(self.read('status.json')['knowledge']['records'],1)
        def second(problem,algorithm,**kwargs):
            self.assertEqual(kwargs['learning_state']['_knowledge'].snapshot(),payload)
            return self.terminal('stopped',knowledge=payload)
        self.write.reset_mock()
        with patch.object(runner,'run_solver',side_effect=second):runner.main(self.args('parallel-dfs'))
        self.assertEqual(path.read_bytes(),original)
        self.assertEqual(self.read('status.json')['knowledge']['loaded'],1)
        self.assertFalse(any(Path(call.args[0]).name=='knowledge-gold.json' for call in self.write.call_args_list))
        self.assert_released()

    def test_forged_saved_loop_is_rejected_as_optional_knowledge(self):
        from constraint_knowledge import _digest
        payload=self.loop_fixture()
        payload['records']=[[[0,0]]];payload['digest']=_digest(payload)
        path=self.run/'knowledge-gold.json';atomic_json(path,payload);original=path.read_bytes()
        def solve(problem,algorithm,**kwargs):
            self.assertEqual(len(kwargs['learning_state']['_knowledge']),0)
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=solve):runner.main(self.args('dfs'))
        self.assertIn('rejected',self.read('status.json')['knowledge_warning'])
        self.assertEqual(path.read_bytes(),original)
        self.assert_released()

    def test_oversized_optional_cache_is_not_parsed_and_disabled_cache_is_ignored(self):
        from constraint_knowledge import MAX_BYTES
        path=self.run/'knowledge-gold.json';path.parent.mkdir(parents=True)
        path.write_text('x'*(MAX_BYTES+1),encoding='utf-8')
        read=runner.read_json_shared
        def guarded(path_arg,*args,**kwargs):
            self.assertNotEqual(Path(path_arg),path)
            return read(path_arg,*args,**kwargs)
        with patch.object(runner,'read_json_shared',side_effect=guarded),patch.object(runner,'run_solver',return_value=self.terminal('stopped')):
            runner.main(self.args('dfs'))
        self.assertIn('bounded file size',self.read('status.json')['knowledge_warning'])
        def disabled(problem,algorithm,**kwargs):
            self.assertNotIn('_knowledge',kwargs['learning_state'])
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=disabled):
            runner.main(self.args('dfs','--resume','--search-options','{"persistent_learning":false}'))
        self.assertNotIn('knowledge_warning',self.read('status.json'))
        self.assertNotIn('knowledge',self.read('status.json'))
        self.assertEqual(path.stat().st_size,MAX_BYTES+1)
        self.assert_released()

    def test_knowledge_write_failure_stops_worker_and_releases_lock(self):
        payload=self.loop_fixture();finished=threading.Event()
        def solve(problem,algorithm,**kwargs):
            try:
                kwargs['progress']({'event':'progress','knowledge':payload})
                while not kwargs['stop']():finished.wait(.005)
                return self.terminal('stopped')
            finally:finished.set()
        def failed_write(path,value):
            if Path(path).name=='knowledge-gold.json':raise OSError('fixture knowledge write failure')
            atomic_json(path,value)
        with patch.object(runner,'run_solver',side_effect=solve),patch.object(runner,'atomic_json',side_effect=failed_write),self.assertRaisesRegex(SystemExit,'knowledge write failure'):
            runner.main(self.args('dfs'))
        self.assertTrue(finished.is_set());self.assert_released()

    def conflict_fixture(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        store=VerifiedConflictStore(self.problem)
        try:
            store.offer([[0,0],[1,0]])
            for _ in range(4):
                store.advance(seconds=.25,attempts=4)
                if len(store):break
            self.assertEqual(len(store),1)
            return store.snapshot()
        finally:store.close()

    @requires_sat
    def test_shared_conflicts_are_reproved_saved_and_restored_in_separate_stores(self):
        payload=self.conflict_fixture();producer=[]
        def first(problem,algorithm,**kwargs):
            store=kwargs['learning_state']['_shared_conflicts'];producer.append(store)
            self.assertEqual(len(store),0)
            kwargs['progress']({'event':'progress','shared_conflicts':payload})
            return self.terminal('stopped',shared_conflicts=payload)
        with patch.object(runner,'run_solver',side_effect=first):
            runner.main(self.args('dfs','--search-options','{"shared_conflicts":true}'))
        path=self.run/'conflicts-gold.json';original=path.read_bytes()
        self.assertEqual(json.loads(original)['records'],payload['records'])
        self.assertEqual(self.read('status.json')['shared_conflicts']['records'],1)
        # The disk consumer could not mutate the producer's empty store.
        self.assertEqual(len(producer[0]),0);self.assertEqual(producer[0].status,'closed')
        def second(problem,algorithm,**kwargs):
            store=kwargs['learning_state']['_shared_conflicts']
            self.assertEqual(len(store),0);self.assertEqual(store.pending_count,1)
            store.advance(seconds=.25,attempts=4)
            self.assertEqual(store.snapshot(),payload)
            return self.terminal('stopped',shared_conflicts=payload)
        self.write.reset_mock()
        with patch.object(runner,'run_solver',side_effect=second):
            runner.main(self.args('parallel-dfs','--search-options','{"shared_conflicts":true}'))
        self.assertEqual(path.read_bytes(),original)
        # 'loaded' counts proofs completed in the initial 20 ms allowance.
        # Later consumer work may finish verification; it must still report a
        # verified record without requiring a particular machine's timing.
        restored=self.read('status.json')['shared_conflicts']
        self.assertIn(restored['loaded'],(0,1))
        self.assertEqual(restored['records'],1)
        self.assertFalse(any(Path(call.args[0]).name=='conflicts-gold.json' for call in self.write.call_args_list))
        self.assert_released()

    @requires_sat
    def test_false_claim_with_recomputed_digest_never_becomes_a_disk_exclusion(self):
        from constraint_shared_conflicts import _digest
        payload=self.conflict_fixture();payload['records']=[[[0,0]]];payload['digest']=_digest(payload)
        path=self.run/'conflicts-gold.json';atomic_json(path,payload);original=path.read_bytes()
        with patch.object(runner,'run_solver',return_value=self.terminal('stopped',shared_conflicts=payload)):
            runner.main(self.args('dfs','--search-options','{"shared_conflicts":true}'))
        state=self.read('status.json')['shared_conflicts']
        self.assertEqual(state['records'],0);self.assertGreater(state['rejected'],0)
        self.assertEqual(path.read_bytes(),original);self.assert_released()

    def test_unknown_shared_claims_remain_pending_and_are_not_serialized(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import snapshot_claims
        payload=snapshot_claims(self.problem,[[[0,0],[1,0]]])
        with patch.object(VerifiedConflictStore,'advance',return_value=0),patch.object(runner,'run_solver',return_value=self.terminal('stopped',shared_conflicts=payload)):
            runner.main(self.args('dfs','--search-options','{"shared_conflicts":true}'))
        state=self.read('status.json')['shared_conflicts']
        self.assertEqual(state['records'],0);self.assertEqual(state['pending'],1)
        self.assertFalse((self.run/'conflicts-gold.json').exists());self.assert_released()

    def test_oversized_and_disabled_shared_cache_never_get_parsed(self):
        from constraint_shared_conflicts import MAX_BYTES
        path=self.run/'conflicts-gold.json';path.parent.mkdir(parents=True)
        path.write_text('x'*(MAX_BYTES+1),encoding='utf-8');read=runner.read_json_shared
        def guarded(path_arg,*args,**kwargs):
            self.assertNotEqual(Path(path_arg),path);return read(path_arg,*args,**kwargs)
        with patch.object(runner,'read_json_shared',side_effect=guarded),patch.object(runner,'run_solver',return_value=self.terminal('stopped')):
            runner.main(self.args('dfs','--search-options','{"shared_conflicts":true}'))
        self.assertIn('bounded file size',self.read('status.json')['shared_conflicts_warning'])
        def disabled(problem,algorithm,**kwargs):
            self.assertNotIn('_shared_conflicts',kwargs['learning_state'])
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=disabled):runner.main(self.args('dfs','--resume'))
        self.assertNotIn('shared_conflicts',self.read('status.json'))
        self.assertEqual(path.stat().st_size,MAX_BYTES+1);self.assert_released()

    @requires_sat
    def test_shared_cache_write_failure_stops_worker_and_releases_lock(self):
        payload=self.conflict_fixture();finished=threading.Event()
        def solve(problem,algorithm,**kwargs):
            try:
                kwargs['progress']({'event':'progress','shared_conflicts':payload})
                while not kwargs['stop']():finished.wait(.005)
                return self.terminal('stopped')
            finally:finished.set()
        def failed_write(path,value):
            if Path(path).name=='conflicts-gold.json':raise OSError('fixture conflict write failure')
            atomic_json(path,value)
        with patch.object(runner,'run_solver',side_effect=solve),patch.object(runner,'atomic_json',side_effect=failed_write),self.assertRaisesRegex(SystemExit,'conflict write failure'):
            runner.main(self.args('dfs','--search-options','{"shared_conflicts":true}'))
        self.assertTrue(finished.is_set());self.assert_released()

    def test_new_reason_ordering_and_shared_conflict_sources_are_snapshotted(self):
        with patch.object(runner,'run_solver',return_value=self.terminal('stopped')):runner.main(self.args('dfs'))
        evidence=self.run/'runs'/self.read('status.json')['run_id']
        for name in ('constraint_ordering.py','constraint_explanations.py','constraint_shared_conflicts.py'):
            self.assertEqual((evidence/name).read_bytes(),(runner.ROOT/name).read_bytes())
        self.assert_released()

    def test_parallel_keeps_workers_for_full_remaining_run_budget(self):
        def solve(problem,algorithm,**kwargs):
            self.assertGreater(kwargs['seconds'],80)
            self.assertLessEqual(kwargs['seconds'],90)
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=solve) as search:
            runner.main(self.args('parallel-dfs','--seconds','90'))
        self.assertEqual(search.call_count,1);self.assert_released()


if __name__ == '__main__':
    unittest.main()
