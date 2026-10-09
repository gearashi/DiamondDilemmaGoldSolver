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


if __name__ == '__main__':
    unittest.main()
