"""CPU-only stop/checkpoint regressions with deterministic event ordering.

The real runner, validator, JSON writer, and position cache are exercised with a
small host-array GPU double. These tests do not benchmark GPU or disk throughput.
Temporary data stays below this package; actual runtime state is never touched.
"""
from contextlib import contextmanager, redirect_stdout
import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import solve
from checkpoint_writer import CheckpointWriter
from gpu_engine import score_boards_cpu

ROOT = Path(__file__).resolve().parent


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory(prefix='stop-test-', dir=ROOT) as name:
        directory = Path(name).resolve()
        if directory.parent != ROOT.resolve():
            raise RuntimeError('Test directory escaped the solver folder')
        yield directory


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class HostArray:
    """The subset of the CuPy get/index interface read by solve.main."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def get(self):
        return self.value.copy()

    def __getitem__(self, item):
        return HostArray(self.value[item])


class Scenario:
    def __init__(self, output, mode):
        self.output = output
        self.mode = mode
        self.clock = Clock()
        self.limit_seconds = 1000
        self.step_seconds = 2
        self.steps = 0
        self.reseeds = 0
        self.mutations = 0
        self.stop_issued = False
        self.snapshots = []
        self.writes = []
        self.write_states = []
        self.statuses = []
        self.status_records = []
        self.status_at_gpu_init = None
        self.previous_status_bytes = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.writer = None

    def issue_stop(self):
        self.stop_issued = True
        (self.output / 'stop.request').write_text('stop', encoding='utf-8')

    def writer_factory(self, snapshot, write, *, clock):
        self.writer = CheckpointWriter(snapshot, write, clock=self.clock)
        return self.writer

    def gpu_class(self):
        scenario = self

        class FakeGPU:
            def __init__(self, masks, neighbors, neighbor_sides, n, seed, cache_slots):
                scenario.status_at_gpu_init = json.loads(
                    (scenario.output / 'status.json').read_text(encoding='utf-8'))
                self.n = n
                self.device = 'CPU stop-test double; no CUDA execution'
                codes = np.arange(160, dtype=np.int16) * 3
                scores = score_boards_cpu(masks, neighbors, neighbor_sides, codes)
                if not 0 <= scores[0] < 240:
                    raise AssertionError('Stop fixture must be a legal unfinished board')
                self.boards = HostArray(np.repeat(codes[:, None], n, axis=1))
                self.bestboards = HostArray(self.boards.get())
                self.scores = HostArray(np.repeat(scores, n))
                self.bestscores = HostArray(self.scores.get())
                self.counters = HostArray(np.zeros((3, n), np.uint64))
                self.seeds = None

            def initialize(self):
                pass

            def cool(self, elapsed):
                pass

            def step(self, count):
                if scenario.stop_issued:
                    raise AssertionError('GPU step was called after Stop was issued')
                scenario.steps += 1
                scenario.mutations += 1
                self.counters.value[0] += count
                scenario.clock.advance(scenario.step_seconds)
                if scenario.steps > 5:
                    raise AssertionError('Stop was not honored within the deterministic fixture')
                if scenario.mode in ('batch', 'error'):
                    scenario.issue_stop()
                elif scenario.mode == 'during_write' and scenario.steps == 2:
                    if not scenario.started.wait(timeout=5):
                        raise AssertionError('Checkpoint worker did not start')
                    # Keep the worker blocked until the runner publishes stopping.
                    scenario.issue_stop()
                elif scenario.mode == 'completion_gap' and scenario.steps == 2:
                    if not scenario.started.wait(timeout=5):
                        raise AssertionError('Checkpoint worker did not start')
                    scenario.release.set()
                    # It finishes after the loop-top poll, before the bottom
                    # checkpoint interval test. No wall-clock sleep is needed.
                    scenario.writer.future.result(timeout=5)
                elif scenario.mode == 'completion_gap' and scenario.steps == 3:
                    scenario.issue_stop()

            def reseed(self, fraction):
                if scenario.stop_issued:
                    raise AssertionError('Reseed was called after Stop was issued')
                scenario.reseeds += 1
                scenario.mutations += 1

            def has_pending(self):
                return False

            def pending_indices(self):
                return np.array([], dtype=np.int64)

            def acknowledge(self, indices):
                return len(indices)

            def verify_device_scores(self, indices):
                pass

            def cache_stats(self):
                return {'enabled': False, 'slots_per_replica': 0}

            def snapshot_checkpoint(self):
                scenario.snapshots.append(scenario.mutations)
                return {
                    'test_generation': np.array(scenario.mutations),
                    'boards': self.boards.get(),
                    'bestboards': self.bestboards.get(),
                    'scores': self.scores.get(),
                    'bestscores': self.bestscores.get(),
                    'counters': self.counters.get(),
                }

            @staticmethod
            def write_checkpoint(path, payload, compressed=False):
                if compressed:
                    raise AssertionError('Runner should request uncompressed checkpoints')
                generation = int(payload['test_generation'])
                scenario.writes.append(generation)
                status_path = scenario.output / 'status.json'
                state = json.loads(status_path.read_text(encoding='utf-8'))['state']
                scenario.write_states.append(state)
                if scenario.mode == 'error':
                    raise OSError('injected checkpoint write failure')
                if generation == 1 and scenario.mode in ('during_write', 'completion_gap'):
                    scenario.started.set()
                    if not scenario.release.wait(timeout=5):
                        raise TimeoutError('Test did not release the checkpoint worker')
                np.savez(path, **payload)
                return {'file_bytes': Path(path).stat().st_size, 'compressed': False}

        return FakeGPU

    def execute(self):
        self.output.mkdir()
        # Two altered U paths on nonadjacent cells make a balanced but unfinished
        # synthetic fixture. No source diagrams or extracted tile set is needed.
        tiles = [{'id': f'fixture-{i}', 'number': i + 1, 'group': 'silver',
                  'segments': [[[side, 5], [side, 7]] for side in range(3)]}
                 for i in range(160)]
        adjacent = {cell for cell, _ in solve.build_board().neighbors[0]}
        remote = next(cell for cell in range(1, 160) if cell not in adjacent)
        for cell in (0, remote):
            tiles[cell]['segments'][0] = [[0, 4], [0, 8]]
        source = self.output.parent / 'fixture.json'
        source.write_text(json.dumps({'tiles': tiles}), encoding='utf-8')
        if self.previous_status_bytes is not None:
            (self.output / 'status.json').write_bytes(self.previous_status_bytes)
        if self.mode == 'prelaunch':
            self.issue_stop()
        original_atomic = solve.atomic_json

        def record_json(path, data):
            original_atomic(path, data)
            if Path(path) == self.output / 'status.json':
                self.statuses.append(data['state'])
                self.status_records.append(json.loads(json.dumps(data)))
                if self.mode == 'during_write' and data['state'] == 'stopping':
                    self.release.set()

        command = [
            str(ROOT / 'solve.py'), '--seconds', str(self.limit_seconds), '--replicas', '128',
            '--cache-slots', '0', '--checkpoint-seconds', '1',
            '--stop-file-initialized', '--data', str(source), '--output', str(self.output),
        ]
        try:
            with patch.object(solve, 'GPU', self.gpu_class()), \
                 patch.object(solve, 'CheckpointWriter', self.writer_factory), \
                 patch.object(solve, 'time', SimpleNamespace(monotonic=self.clock)), \
                 patch.object(solve, 'atomic_json', side_effect=record_json), \
                 patch.object(solve.signal, 'signal'), \
                 patch.object(sys, 'argv', command), redirect_stdout(io.StringIO()):
                return solve.main()
        finally:
            self.release.set()


class CheckpointWriterTests(unittest.TestCase):
    def test_one_inflight_snapshot_and_no_identical_final_save(self):
        clock = Clock()
        started = threading.Event()
        release = threading.Event()
        snapshots = []
        writes = []

        def snapshot():
            snapshots.append(1)
            return {'value': len(snapshots)}

        def write(payload, generation):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError('Test worker was not released')
            writes.append((payload, generation))
            return {'test_metric': 7}

        writer = CheckpointWriter(snapshot, write, clock=clock)
        try:
            self.assertTrue(writer.request(1))
            self.assertTrue(started.wait(timeout=5))
            self.assertFalse(writer.request(2))
            self.assertEqual(len(snapshots), 1)
            release.set()
            writer.finish(1)
            self.assertEqual(len(snapshots), 1)
            self.assertEqual([generation for _, generation in writes], [1])
            self.assertEqual(writer.info['phase'], 'idle')
            self.assertEqual(writer.info['test_metric'], 7)
        finally:
            release.set()
            writer.close()

    def test_finish_saves_newer_generation_once(self):
        current = {'value': 1}
        snapshots = []
        writes = []

        def snapshot():
            payload = dict(current)
            snapshots.append(payload)
            return payload

        writer = CheckpointWriter(snapshot, lambda payload, generation: writes.append((payload, generation)))
        try:
            writer.request(1)
            current['value'] = 3
            writer.finish(3)
            writer.finish(3)
            self.assertEqual([generation for _, generation in writes], [1, 3])
            self.assertEqual([value['value'] for value in snapshots], [1, 3])
            self.assertEqual(writer.completed_generation, 3)
        finally:
            writer.close()

    def test_completion_timestamp_is_after_capture(self):
        clock = Clock()
        release = threading.Event()
        writer = CheckpointWriter(lambda: {}, lambda *_: release.wait(timeout=5) and {}, clock=clock)
        try:
            clock.advance(2)
            writer.request(1)
            clock.advance(19)
            release.set()
            writer.poll(wait=True)
            self.assertEqual(writer.last_completed_at, 21)
        finally:
            release.set()
            writer.close()

    def test_worker_error_propagates(self):
        def fail(*_):
            raise OSError('injected writer failure')
        writer = CheckpointWriter(lambda: {}, fail)
        try:
            writer.request(1)
            with self.assertRaisesRegex(OSError, 'injected writer failure'):
                writer.poll(wait=True)
            with self.assertRaisesRegex(OSError, 'injected writer failure'):
                writer.finish(2)
            self.assertIsNone(writer.completed_generation)
        finally:
            writer.close()

    def test_closed_writer_rejects_new_requests(self):
        writer = CheckpointWriter(lambda: {}, lambda *_: {})
        writer.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            writer.request(1)


class SolverStopTests(unittest.TestCase):
    def check_finished(self, scenario, expected_steps, expected_snapshots):
        self.assertEqual(scenario.execute(), 0)
        self.assertEqual(scenario.steps, expected_steps)
        self.assertEqual(scenario.reseeds, 0)
        self.assertEqual(scenario.snapshots, expected_snapshots)
        self.assertEqual(scenario.writes, expected_snapshots)
        self.assertEqual(scenario.statuses[-1], 'stopped')
        self.assertIn('stopping', scenario.statuses)
        self.assertEqual(scenario.write_states[-1], 'stopping')
        output = scenario.output
        self.assertFalse((output / 'run.lock').exists())
        self.assertFalse((output / 'solution.json').exists())
        self.assertTrue((output / 'best.json').is_file())
        status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['checkpoint']['completed_generation'], scenario.mutations)
        with np.load(output / 'checkpoint.npz', allow_pickle=False) as checkpoint:
            self.assertEqual(int(checkpoint['test_generation']), scenario.mutations)
            self.assertEqual(int(checkpoint['counters'][0, 0]), expected_steps * 32)
        metadata = json.loads((output / 'checkpoint-meta.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['generation'], scenario.mutations)

    def test_startup_exposes_selected_replicas_before_gpu_initialization(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'batch')
            self.check_finished(scenario, 1, [1])
            first = scenario.status_records[0]
            self.assertEqual(first['state'], 'starting')
            self.assertEqual(first['startup_phase'], 'loading_input')
            self.assertEqual(first['replicas'], 128)
            self.assertEqual(first['time_limit_seconds'], 1000)
            self.assertTrue(first['phase_message'])
            at_init = scenario.status_at_gpu_init
            self.assertEqual(at_init['state'], 'starting')
            self.assertEqual(at_init['startup_phase'], 'initializing_gpu')
            self.assertEqual(at_init['replicas'], 128)
            self.assertEqual(at_init['run_id'], first['run_id'])
            phases = [item.get('startup_phase') for item in scenario.status_records
                      if item['state'] == 'starting']
            self.assertEqual(phases, ['loading_input', 'initializing_gpu',
                                     'preparing_population'])

    def test_previous_status_archive_preserves_old_bytes_not_new_startup(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'batch')
            previous = b' {\r\n "state": "stopped", "run_id": "old-run", "replicas": 4096\r\n}\r\n'
            scenario.previous_status_bytes = previous
            self.check_finished(scenario, 1, [1])
            first = scenario.status_records[0]
            archived = scenario.output / 'runs' / first['run_id'] / 'previous-status.json'
            self.assertEqual(archived.read_bytes(), previous)
            self.assertEqual(json.loads(archived.read_bytes())['run_id'], 'old-run')
            self.assertNotEqual(first['run_id'], 'old-run')
            self.assertEqual(first['replicas'], 128)
    def test_unlimited_stochastic_run_past_one_day_still_honors_stop(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'during_write')
            scenario.limit_seconds = 0
            scenario.step_seconds = 90000
            self.check_finished(scenario, 2, [1, 2])
            status = json.loads((scenario.output / 'status.json').read_text(encoding='utf-8'))
            self.assertGreater(status['elapsed_seconds'], 86400)
            self.assertEqual(status['time_limit_seconds'], 0)
            config = json.loads((scenario.output / 'run-config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['seconds'], 0)

    def test_stop_before_first_launch_preserves_initial_state(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'prelaunch')
            self.check_finished(scenario, 0, [0])
            self.assertFalse((scenario.output / 'live.json').exists())

    def test_stop_during_batch_skips_optional_maintenance(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'batch')
            self.check_finished(scenario, 1, [1])
            self.assertFalse((scenario.output / 'live.json').exists())

    def test_stop_during_writer_preserves_latest_generation_without_duplicates(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'during_write')
            self.check_finished(scenario, 2, [1, 2])
            self.assertEqual(scenario.write_states, ['running', 'stopping'])

    def test_periodic_interval_restarts_after_worker_completes_during_batch(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'completion_gap')
            self.check_finished(scenario, 3, [1, 3])

    def test_worker_failure_sets_error_status_and_releases_lock(self):
        with scratch() as directory:
            scenario = Scenario(directory / 'run', 'error')
            with self.assertRaisesRegex(OSError, 'injected checkpoint write failure'):
                scenario.execute()
            status = json.loads((scenario.output / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['state'], 'error')
            self.assertIn('injected checkpoint write failure', status['error'])
            self.assertFalse((scenario.output / 'run.lock').exists())
            self.assertEqual(scenario.steps, 1)
            self.assertEqual(scenario.snapshots, [1])


if __name__ == '__main__':
    names = ('test_stop.py', 'solve.py', 'checkpoint_writer.py', 'gpu_engine.py',
             'geometry.py', 'validator.py', 'position_cache.py', 'io_utils.py',
             'render_board.py')

    def source_hashes():
        return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}

    before = source_hashes()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = source_hashes()
    report = {
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'started_at': started_at,
        'runtime_seconds': time.monotonic() - started,
        'command': list(sys.orig_argv), 'working_directory': str(ROOT),
        'exit_status': 0 if result.wasSuccessful() else 1,
        'python': sys.version, 'platform': platform.platform(),
        'scope': 'CPU control-flow tests with a host-array GPU double; no GPU or disk throughput claim',
        'source_sha256_before': before, 'source_sha256_after': after,
        'execution_sources_unchanged': before == after,
    }
    (ROOT / 'stop-tests.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(report['exit_status'])
