"""CPU integration checks for durable systematic job ownership.

The real runner, frontier, ledger, JSON/NPZ persistence, and final CPU validator
run against a deterministic host DFS double. Its three work units per job test
stop/resume scheduling, not CUDA traversal or real-puzzle solution existence.
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
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import systematic_search as runner
from checkpoint_writer import CheckpointWriter

ROOT = Path(__file__).resolve().parent


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def get(self):
        return self.values.copy()

    def __getitem__(self, item):
        return Array(self.values[item])


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory(prefix='systematic-runner-test-', dir=ROOT) as name:
        root = Path(name).resolve()
        if root.parent != ROOT.resolve():
            raise RuntimeError('Test directory escaped standalone root')
        data = {'tiles': [
            {'id': f'T{i}', 'number': i + 1, 'group': 'silver',
             'segments': [[[side, 5], [side, 7]] for side in range(3)]}
            for i in range(160)
        ]}
        source = root / 'tiles.json'
        source.write_text(json.dumps(data), encoding='utf-8')
        yield root, source


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class Scenario:
    def __init__(self, output, source, mode='exhaust'):
        self.output = output
        self.source = source
        self.mode = mode
        self.clock = Clock()
        self.stop_after = None
        self.step_seconds = .001
        self.pending_after = 1
        self.pending_lane = 0
        self.steps_this_run = 0
        self.all_steps = 0
        self.constructions = 0
        self.events = []
        self.loaded_jobs = []
        self.instances = []
        self.restored_jobs = []

    def gpu_class(self):
        scenario = self

        class FakeDFS:
            DEVICE_FIELDS = ('boards', 'used', 'depths', 'maxdepths', 'states', 'floors',
                             'firsts', 'lengths', 'cursors', 'shifts', 'rng', 'counters')
            CHECKPOINT_VERSION = 2

            def __init__(self, masks, neighbors, sides, n, seed, order, prefixes):
                scenario.constructions += 1
                scenario.instances.append(self)
                self.n = n
                self.order = np.asarray(order)
                self.device = 'CPU integration DFS double'
                self.boards = Array(np.full((160, n), -1, np.int16))
                self.states = Array(np.full(n, 3, np.uint8))
                self.floors = Array(np.zeros(n, np.int16))
                self.depths = Array(np.zeros(n, np.int16))
                self.maxdepths = Array(np.zeros(n, np.int16))
                self.counters = Array(np.zeros((3, n), np.uint64))
                self.prefix_codes = np.full((n, 160), -1, np.int16)
                self.used = Array(np.zeros((5, n), np.uint32))
                self.firsts = Array(np.zeros((160, n), np.uint16))
                self.lengths = Array(np.zeros((160, n), np.uint16))
                self.cursors = Array(np.zeros((160, n), np.uint16))
                self.shifts = Array(np.zeros((160, n), np.uint16))
                self.rng = Array(np.arange(1, n + 1, dtype=np.uint32))
                self.fingerprint = 'deterministic-host-dfs-integration-fixture'
                if len(prefixes):
                    raise AssertionError('Runner should construct initially idle lanes')

            @property
            def work_cursor(self):
                # A real checkpointed field carries synthetic job progress through
                # the production bank/slicing path; no test-only restoration hook.
                return self.cursors.values[159]

            @property
            def roots(self):
                return self.prefix_codes[:, 0].copy()

            def load_prefixes(self, prefixes, lanes):
                for prefix, lane in zip(prefixes, lanes):
                    lane = int(lane)
                    if self.states.values[lane] not in (2, 3):
                        raise AssertionError('Runner overwrote an owned active or pending job')
                    depth = len(prefix)
                    self.prefix_codes[lane] = -1
                    self.prefix_codes[lane, :depth] = prefix
                    self.boards.values[:, lane] = -1
                    self.boards.values[self.order[:depth], lane] = prefix
                    self.floors.values[lane] = depth
                    self.depths.values[lane] = depth
                    self.maxdepths.values[lane] = depth
                    self.work_cursor[lane] = 0
                    self.states.values[lane] = 1 if depth == 160 else 0
                    scenario.loaded_jobs.append(tuple(map(int, prefix)))

            def step(self, budget):
                scenario.steps_this_run += 1
                scenario.all_steps += 1
                scenario.clock.value += scenario.step_seconds
                if scenario.steps_this_run > 2000:
                    raise AssertionError('Deterministic integration failed to terminate')
                if scenario.mode == 'pending' and scenario.steps_this_run >= scenario.pending_after:
                    lane = scenario.pending_lane
                    codes = np.arange(160, dtype=np.int16) * 3
                    for depth, code in enumerate(self.prefix_codes[lane, :self.floors.values[lane]]):
                        cell = int(self.order[depth])
                        other = int(np.flatnonzero(codes // 3 == code // 3)[0])
                        codes[cell], codes[other] = codes[other], codes[cell]
                        codes[cell] = code
                    self.boards.values[:, lane] = codes
                    self.depths.values[lane] = self.maxdepths.values[lane] = 160
                    self.states.values[lane] = 1
                else:
                    for lane in np.flatnonzero(self.states.values == 0):
                        job = tuple(map(int, self.prefix_codes[lane, :self.floors.values[lane]]))
                        cursor = int(self.work_cursor[lane])
                        scenario.events.append((job, cursor))
                        self.work_cursor[lane] += 1
                        self.counters.values[0, lane] += 1
                        if self.work_cursor[lane] == 3:
                            self.states.values[lane] = 2
                            self.depths.values[lane] = 0
                            self.boards.values[:, lane] = -1
                if scenario.stop_after == scenario.steps_this_run:
                    (scenario.output / 'stop.request').write_text('stop', encoding='utf-8')

            def verify(self, indices):
                return len(indices)

            def snapshot_checkpoint(self):
                payload = {field: getattr(self, field).get() for field in self.DEVICE_FIELDS}
                payload.update(prefix_codes=self.prefix_codes.copy(), roots=self.roots,
                               order=self.order.copy(), fingerprint=np.array(self.fingerprint),
                               dfs_version=np.int32(self.CHECKPOINT_VERSION),
                               test_work_cursor=self.work_cursor.copy())
                return payload

            def read_checkpoint(self, archive):
                payload = {field: np.asarray(archive[field]).copy() for field in self.DEVICE_FIELDS}
                payload.update(prefix_codes=np.asarray(archive['prefix_codes']).copy(),
                               roots=np.asarray(archive['roots']).copy(),
                               order=np.asarray(archive['order']).copy(),
                               fingerprint=np.asarray(archive['fingerprint']).copy(),
                               dfs_version=np.asarray(archive['dfs_version']).copy())
                self._validate_host(payload, payload['prefix_codes'])
                return payload

            def _validate_host(self, arrays, prefix_rows, check_stack=True):
                count = arrays['boards'].shape[1]
                for field in self.DEVICE_FIELDS:
                    value = np.asarray(arrays[field])
                    expected = getattr(self, field).values
                    shape = (*expected.shape[:-1], count)
                    if value.shape != shape or value.dtype != expected.dtype:
                        raise ValueError('Malformed host DFS field: ' + field)
                if np.asarray(prefix_rows).shape != (count, 160):
                    raise ValueError('Malformed host DFS prefix rows')
                if np.any(arrays['states'] > 3) or np.any(arrays['cursors'][159] > 3):
                    raise ValueError('Invalid deterministic host DFS progress')
                return count

            def restore_checkpoint(self, payload):
                count = self._validate_host(payload, payload['prefix_codes'])
                if count != self.n:
                    raise ValueError('Host DFS restore has the wrong lane count')
                for field in self.DEVICE_FIELDS:
                    getattr(self, field).values = np.asarray(payload[field]).copy()
                self.prefix_codes = np.asarray(payload['prefix_codes']).copy()
                return True

            def restore_lanes(self, payload, lanes):
                ids = np.asarray(lanes, dtype=np.int64)
                count = self._validate_host(payload, payload['prefix_codes'])
                if count != len(ids) or np.any(~np.isin(self.states.values[ids], [2, 3])):
                    raise ValueError('Exact restore may replace only idle or exhausted lanes')
                for field in self.DEVICE_FIELDS:
                    target = getattr(self, field).values
                    if target.ndim == 2:
                        target[:, ids] = payload[field]
                    else:
                        target[ids] = payload[field]
                self.prefix_codes[ids] = payload['prefix_codes']
                for lane in ids:
                    scenario.restored_jobs.append(tuple(map(int, self.prefix_codes[lane, :self.floors.values[lane]])))
                return ids

            def resume(self, path):
                with np.load(path, allow_pickle=False) as saved:
                    return self.restore_checkpoint(self.read_checkpoint(saved))

        return FakeDFS

    def execute(self, resume=False, replicas=4, seconds=1000):
        self.steps_this_run = 0
        command = [
            str(ROOT / 'systematic_search.py'), '--backend', 'cuda', '--seconds', str(seconds),
            '--replicas', str(replicas), '--nodes', '4', '--checkpoint-seconds', '1000',
            '--data', str(self.source), '--output', str(self.output),
        ]
        if resume:
            command.append('--resume')

        def writer_factory(snapshot, write, **kwargs):
            return CheckpointWriter(snapshot, write, clock=self.clock)

        with patch.object(runner, 'choose_backend', return_value='cuda'), \
             patch.object(runner, 'DFS', self.gpu_class()), \
             patch.object(runner, 'time', SimpleNamespace(monotonic=self.clock)), \
             patch.object(runner, 'CheckpointWriter', writer_factory), \
             patch.object(runner.signal, 'signal'), \
             patch.object(sys, 'argv', command), redirect_stdout(io.StringIO()):
            runner.main()


class SystematicRunnerTests(unittest.TestCase):
    def test_unlimited_runs_past_one_day_and_still_honors_stop(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.step_seconds = 90000
            scenario.stop_after = 2
            scenario.execute(seconds=0)
            status = read(scenario.output / 'status.json')
            config = read(scenario.output / 'run-config.json')
            self.assertEqual(scenario.steps_this_run, 2)
            self.assertGreater(status['elapsed_seconds'], 86400)
            self.assertEqual(status['state'], 'stopped')
            self.assertEqual(status['time_limit_seconds'], 0)
            self.assertEqual(config['seconds'], 0)
            self.assertFalse((scenario.output / 'run.lock').exists())
            with np.load(scenario.output / 'systematic' / 'checkpoint.npz',
                         allow_pickle=False) as saved:
                self.assertEqual(saved['test_work_cursor'].tolist(), [2, 2, 2, 2])

    def test_unlimited_still_stops_on_first_240_after_one_day(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source, mode='pending')
            scenario.step_seconds = 90000
            scenario.pending_after = 2
            scenario.execute(replicas=1, seconds=0)
            status = read(scenario.output / 'status.json')
            self.assertEqual(scenario.steps_this_run, 2)
            self.assertGreater(status['elapsed_seconds'], 86400)
            self.assertEqual(status['time_limit_seconds'], 0)
            self.assertEqual(status['state'], 'edge_perfect')
            report = read(scenario.output / 'edge-perfect.json')['validation']
            self.assertEqual(report['matched_edges'], 240)
            self.assertEqual(report['line_components'], 240)
            self.assertFalse((scenario.output / 'solution.json').exists())

    def test_normal_stop_resume_covers_each_job_work_unit_once(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 2
            scenario.execute()
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['state'], 'stopped')
            self.assertEqual(status['assigned_jobs'], 4)
            self.assertEqual(status['exhausted_jobs'], 0)
            checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(saved['exact_job_ids'].tolist(), [0, 1, 2, 3])
                self.assertEqual(saved['test_work_cursor'].tolist(), [2, 2, 2, 2])
            scenario.stop_after = None
            scenario.execute(resume=True)
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['state'], 'exhausted')
            self.assertEqual(status['total_jobs'], 480)
            self.assertEqual(status['assigned_jobs'], 480)
            self.assertEqual(status['exhausted_jobs'], 480)
            self.assertEqual(status['active_jobs'], 0)
            self.assertEqual(len(scenario.events), 480 * 3)
            self.assertEqual(len(set(scenario.events)), 480 * 3)
            expected = {((code,), cursor) for code in range(480) for cursor in range(3)}
            self.assertEqual(set(scenario.events), expected)
            self.assertEqual(len(scenario.loaded_jobs), 480)
            self.assertEqual(len(set(scenario.loaded_jobs)), 480)
            self.assertFalse((scenario.output / 'run.lock').exists())
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(str(saved['exact_terminal']), 'exhausted')
                self.assertEqual(int(saved['exact_cursor']), 480)
                self.assertEqual(int(saved['exact_completed']), 480)
                self.assertTrue(np.all(saved['exact_job_ids'] == -1))
            steps = scenario.all_steps
            scenario.execute(resume=True)
            self.assertEqual(scenario.all_steps, steps, 'Terminal resume launched extra work')
            self.assertEqual(read(scenario.output / 'status.json')['state'], 'exhausted')

    def test_first_240_multiloop_candidate_stops_and_terminal_resume_does_no_work(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source, mode='pending')
            scenario.execute(replicas=1)
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['state'], 'edge_perfect')
            record = read(scenario.output / 'edge-perfect.json')
            self.assertEqual(record['validation']['matched_edges'], 240)
            self.assertEqual(record['validation']['line_components'], 240)
            self.assertFalse(record['solved'])
            self.assertFalse((scenario.output / 'solution.json').exists())
            checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(saved['states'].tolist(), [1])
                self.assertEqual(saved['exact_job_ids'].tolist(), [0])
                self.assertEqual(str(saved['exact_terminal']), 'edge_perfect')
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            steps = scenario.all_steps
            constructions = scenario.constructions
            scenario.execute(resume=True, replicas=1)
            self.assertEqual(read(scenario.output / 'status.json')['state'], 'edge_perfect')
            self.assertEqual(scenario.all_steps, steps)
            self.assertEqual(scenario.constructions, constructions)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_resume_rejects_lane_prefix_bound_to_wrong_job(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 2
            scenario.execute()
            checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
            with np.load(checkpoint, allow_pickle=False) as saved:
                fields = {name: saved[name] for name in saved.files}
            # Both fixed prefixes are individually valid, but their job IDs stay
            # unchanged. Separate GPU and ledger validation cannot catch this swap.
            fields['prefix_codes'][[0, 1]] = fields['prefix_codes'][[1, 0]]
            fields['boards'][:, [0, 1]] = fields['boards'][:, [1, 0]]
            np.savez(checkpoint, **fields)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with self.assertRaises(SystemExit):
                scenario.execute(resume=True)
            self.assertEqual(read(scenario.output / 'status.json')['state'], 'error')
            self.assertEqual(scenario.steps_this_run, 0)
            self.assertFalse((scenario.output / 'run.lock').exists())
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_terminal_checkpoint_claims_require_matching_saved_state(self):
        for terminal in ('unknown', 'exhausted', 'edge_perfect', 'solved'):
            with self.subTest(terminal=terminal), scratch() as (root, source):
                scenario = Scenario(root / 'run', source)
                scenario.stop_after = 2
                scenario.execute()
                checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
                with np.load(checkpoint, allow_pickle=False) as saved:
                    fields = {name: saved[name] for name in saved.files}
                fields['exact_terminal'] = np.array(terminal)
                np.savez(checkpoint, **fields)
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(SystemExit):
                    scenario.execute(resume=True)
                self.assertEqual(read(scenario.output / 'status.json')['state'], 'error')
                self.assertEqual(scenario.steps_this_run, 0)
                self.assertFalse((scenario.output / 'run.lock').exists())
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_pending_terminal_rebuilds_missing_certificate_without_search(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source, mode='pending')
            scenario.execute(replicas=1)
            original = read(scenario.output / 'edge-perfect.json')
            (scenario.output / 'best.json').unlink()
            (scenario.output / 'edge-perfect.json').unlink()
            steps = scenario.all_steps
            scenario.execute(resume=True, replicas=1)
            restored = read(scenario.output / 'edge-perfect.json')
            self.assertEqual(scenario.all_steps, steps)
            self.assertEqual(restored['codes'], original['codes'])
            self.assertEqual(restored['validation'], original['validation'])
            self.assertTrue(restored['revalidated_previous_record'])
            self.assertFalse(restored['solved'])
            self.assertFalse((scenario.output / 'solution.json').exists())
            self.assertEqual(read(scenario.output / 'status.json')['state'], 'edge_perfect')

    def test_resume_rejects_changed_frontier_before_launch(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 2
            scenario.execute()
            frontier = scenario.output / 'systematic' / 'frontier.npz'
            with np.load(frontier, allow_pickle=False) as saved:
                fields = {name: saved[name] for name in saved.files}
            fields['prefixes'][0, 0] = 479
            np.savez(frontier, **fields)
            constructions = scenario.constructions
            with self.assertRaises(SystemExit):
                scenario.execute(resume=True)
            self.assertEqual(read(scenario.output / 'status.json')['state'], 'error')
            self.assertEqual(scenario.constructions, constructions)
            self.assertEqual(scenario.steps_this_run, 0)


    def test_shrink_preserves_paused_cursors_and_cumulative_work(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 1
            scenario.execute(replicas=8)
            self.assertEqual(len(scenario.events), 8)
            scenario.execute(resume=True, replicas=2)
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['replicas'], 2)
            self.assertEqual(status['active_jobs'], 2)
            self.assertEqual(status['paused_jobs'], 6)
            self.assertEqual(status['assigned_jobs'], 8)
            self.assertEqual(status['exhausted_jobs'], 0)
            self.assertEqual(status['nodes_checked'], 10)
            with np.load(scenario.output / 'systematic' / 'checkpoint.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['boards'].shape, (160, 2))
                self.assertEqual(len(saved['exact_paused_job_ids']), 6)
                self.assertEqual(saved['exact_paused_cursors'][159].tolist(), [1] * 6)
                self.assertEqual(saved['cursors'][159].tolist(), [2] * 2)
                self.assertEqual(int(saved['exact_completed']), 0)
                self.assertEqual(int(saved['exact_cursor']), 8)
            self.assertEqual(len(scenario.loaded_jobs), 8, 'Shrink restarted or assigned a job')
            self.assertEqual(len(set(scenario.events)), len(scenario.events))

    def test_multiple_resizes_cover_every_work_unit_once_with_stable_totals(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            for i, replicas in enumerate((8, 2, 5, 1, 7)):
                scenario.stop_after = 1 if i < 4 else None
                scenario.execute(resume=i > 0, replicas=replicas)
                status = read(scenario.output / 'status.json')
                self.assertEqual(status['replicas'], replicas)
                self.assertLessEqual(status['active_jobs'], replicas)
                self.assertEqual(status['nodes_checked'], len(scenario.events))
                self.assertEqual(len(set(scenario.events)), len(scenario.events))
                self.assertEqual(status['assigned_jobs'], status['exhausted_jobs'] +
                                 status['active_jobs'] + status['paused_jobs'])
            expected = {((code,), cursor) for code in range(480) for cursor in range(3)}
            self.assertEqual(set(scenario.events), expected)
            self.assertEqual(len(scenario.events), 1440)
            self.assertEqual(len(scenario.loaded_jobs), 480)
            self.assertEqual(len(set(scenario.loaded_jobs)), 480)
            self.assertEqual(status['state'], 'exhausted')
            self.assertEqual(status['exhausted_jobs'], 480)
            self.assertEqual(status['active_jobs'], 0)
            self.assertEqual(status['paused_jobs'], 0)

    def test_growth_promotes_paused_jobs_before_new_prefixes(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 1
            scenario.execute(replicas=8)
            scenario.execute(resume=True, replicas=2)
            before = len(scenario.loaded_jobs)
            scenario.execute(resume=True, replicas=6)
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['replicas'], 6)
            self.assertEqual(status['assigned_jobs'], 8)
            self.assertEqual(len(scenario.loaded_jobs), before)
            self.assertEqual(status['nodes_checked'], len(scenario.events))
            with np.load(scenario.output / 'systematic' / 'checkpoint.npz', allow_pickle=False) as saved:
                owned = set(map(int, saved['exact_job_ids'][saved['exact_job_ids'] >= 0]))
                paused = set(map(int, saved['exact_paused_job_ids']))
                self.assertFalse(owned & paused)
                self.assertTrue(owned | paused <= set(range(8)))

    def test_legacy_checkpoint_without_bank_fields_can_shrink_without_replay(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source)
            scenario.stop_after = 2
            scenario.execute(replicas=8)
            checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
            with np.load(checkpoint, allow_pickle=False) as saved:
                old = {name: saved[name].copy() for name in saved.files
                       if not name.startswith(('exact_paused_', 'exact_bank_', 'exact_retired_'))}
            np.savez(checkpoint, **old)
            scenario.stop_after = None
            scenario.execute(resume=True, replicas=2)
            status = read(scenario.output / 'status.json')
            expected = {((code,), cursor) for code in range(480) for cursor in range(3)}
            self.assertEqual(set(scenario.events), expected)
            self.assertEqual(len(scenario.events), len(expected))
            self.assertEqual(status['nodes_checked'], len(expected))
            self.assertGreater(len(scenario.restored_jobs), 0, 'Paused branches were never promoted into free lanes')
            self.assertEqual(status['replicas'], 2)
            self.assertEqual(status['state'], 'exhausted')

    def test_corrupt_paused_ownership_is_rejected_without_replacing_checkpoint(self):
        for fault in ('duplicate_job', 'wrong_prefix_binding'):
            with self.subTest(fault=fault), scratch() as (root, source):
                scenario = Scenario(root / 'run', source)
                scenario.stop_after = 1
                scenario.execute(replicas=8)
                scenario.execute(resume=True, replicas=2)
                checkpoint = scenario.output / 'systematic' / 'checkpoint.npz'
                with np.load(checkpoint, allow_pickle=False) as saved:
                    fields = {name: saved[name].copy() for name in saved.files}
                if fault == 'duplicate_job':
                    fields['exact_paused_job_ids'][0] = fields['exact_job_ids'][0]
                else:
                    fields['exact_paused_prefix_codes'][[0, 1]] = fields['exact_paused_prefix_codes'][[1, 0]]
                    fields['exact_paused_boards'][:, [0, 1]] = fields['exact_paused_boards'][:, [1, 0]]
                np.savez(checkpoint, **fields)
                original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(SystemExit):
                    scenario.execute(resume=True, replicas=3)
                self.assertEqual(scenario.steps_this_run, 0)
                self.assertEqual(read(scenario.output / 'status.json')['state'], 'error')
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original)
                self.assertFalse((scenario.output / 'run.lock').exists())

    def test_pending_last_lane_survives_shrink_and_rebuilds_certificate_without_steps(self):
        with scratch() as (root, source):
            scenario = Scenario(root / 'run', source, mode='pending')
            scenario.pending_lane = 7
            scenario.execute(replicas=8)
            original = read(scenario.output / 'edge-perfect.json')
            self.assertEqual(original['job_id'], 7)
            (scenario.output / 'best.json').unlink()
            (scenario.output / 'edge-perfect.json').unlink()
            steps = scenario.all_steps
            scenario.execute(resume=True, replicas=1)
            restored = read(scenario.output / 'edge-perfect.json')
            status = read(scenario.output / 'status.json')
            self.assertEqual(status['replicas'], 1)
            self.assertEqual(scenario.all_steps, steps)
            self.assertEqual(restored['job_id'], original['job_id'])
            self.assertEqual(restored['codes'], original['codes'])
            self.assertEqual(restored['validation'], original['validation'])
            self.assertTrue(restored['revalidated_previous_record'])
            self.assertEqual(status['state'], 'edge_perfect')


if __name__ == '__main__':
    names = ('test_systematic_runner.py', 'systematic_search.py', 'systematic_jobs.py',
             'exact_frontier.py', 'dfs_gpu.py', 'geometry.py', 'validator.py',
             'checkpoint_writer.py', 'io_utils.py')
    def hashes():
        return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    before = hashes()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = hashes()
    report = {
        'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'started_at': started_at, 'runtime_seconds': time.monotonic() - started,
        'command': list(sys.orig_argv), 'working_directory': str(ROOT),
        'exit_status': 0 if result.wasSuccessful() else 1,
        'python': sys.version, 'platform': platform.platform(),
        'scope': 'CPU runner integration with deterministic three-unit-per-job host DFS double and independently validated multiloop candidate; not a CUDA traversal proof',
        'source_sha256_before': before, 'source_sha256_after': after,
        'execution_sources_unchanged': before == after,
    }
    (ROOT / 'systematic-runner-tests.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(report['exit_status'])
