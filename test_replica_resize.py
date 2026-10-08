"""CPU checks for exact lane-bank resizing; optional small CUDA leaf enumeration.

Set DIAMOND_TEST_CUDA=1 to enable the finite 54-leaf GPU regression. The default
suite does not import CuPy, launch CUDA, or read any production checkpoint.
"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from dfs_gpu import DFS, fill_order
from geometry import build_board
from systematic_jobs import JobLedger


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def dtype(self): return self.value.dtype
    def __array__(self, dtype=None, copy=None): return np.asarray(self.value, dtype=dtype) if not copy else np.array(self.value, dtype=dtype, copy=True)
    def get(self): return self.value.copy()
    def __getitem__(self, key): return Array(self.value[key])
    def __setitem__(self, key, value): self.value[key] = value


def host_dfs(n):
    """Use the real DFS host validation/install methods with NumPy-backed storage."""
    gpu = DFS.__new__(DFS)
    gpu.n = n
    board = build_board()
    gpu.masks = np.zeros((480, 3), np.uint16)
    gpu.neighbors = np.asarray(board.neighbor_cells, np.int16)
    gpu.neighbor_sides = np.asarray(board.neighbor_sides, np.uint8)
    gpu.order = fill_order(gpu.neighbors)
    gpu.order_rank = np.argsort(gpu.order)
    gpu.fingerprint = hashlib.sha256(gpu.masks.tobytes() + gpu.neighbors.tobytes() + gpu.neighbor_sides.tobytes() + gpu.order.tobytes()).hexdigest()
    gpu.host_faces = gpu.host_reverse = np.zeros((480, 3), np.uint16)
    gpu.colors = 1
    gpu.host_single_offsets = np.arange(4, dtype=np.int32) * 480 + 480
    gpu.host_pair_offsets = np.arange(4, dtype=np.int32) * 480 + 1920
    gpu.host_pool = np.tile(np.arange(480, dtype=np.int16), 7)
    gpu.cp = SimpleNamespace(asarray=np.asarray, maximum=np.maximum,
                            cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    gpu.boards = Array(np.full((160, n), -1, np.int16))
    gpu.used = Array(np.zeros((5, n), np.uint32))
    for name in ('depths', 'maxdepths', 'floors'):
        setattr(gpu, name, Array(np.zeros(n, np.int16)))
    gpu.states = Array(np.full(n, 3, np.uint8))
    for name in ('firsts', 'lengths', 'shifts'):
        setattr(gpu, name, Array(np.zeros((160, n), np.uint16)))
    gpu.cursors = Array(np.full((160, n), 65535, np.uint16))
    gpu.rng = Array(np.arange(1, n + 1, dtype=np.uint32))
    gpu.counters = Array(np.zeros((3, n), np.uint64))
    gpu.prefix_codes = np.full((n, 160), -1, np.int16)
    gpu.roots = np.full(n, -1, np.int16)
    gpu.prefixes = [None] * n
    return gpu


def frontier(count=8):
    rows = np.full((count, 160), -1, np.int16)
    rows[:, 0] = np.arange(count)
    return rows, np.ones(count, np.int16)


def snapshot(gpu, ledger):
    payload = gpu.snapshot_checkpoint()
    payload.update(ledger.fields())
    return payload


def owned_states(gpu, ledger):
    result = {}
    for payload, ids in ((gpu.snapshot_checkpoint(), ledger.ids),
                         (ledger.paused.remaining_arrays(), ledger.paused.remaining_ids())):
        for lane, job in enumerate(ids):
            if job >= 0:
                result[int(job)] = {name: (payload[name][:, lane].copy() if payload[name].ndim == 2 else payload[name][lane].copy())
                                    for name in DFS.DEVICE_FIELDS}
                result[int(job)]['prefix_codes'] = payload['prefix_codes'][lane].copy()
    return result


class ResizeTests(unittest.TestCase):
    def setUp(self):
        self.prefixes, self.lengths = frontier()
        self.gpu = host_dfs(8)
        self.ledger = JobLedger.empty(8, 8)
        self.ledger.refill(self.gpu, self.prefixes, self.lengths)
        # Enter each current frame at a distinct exact cursor, without descending.
        cell = self.gpu.order[1]
        side = next(s for s in range(3) if self.gpu.neighbors[cell, s] == self.gpu.order[0])
        self.gpu.firsts.value[1] = self.gpu.host_single_offsets[side]
        self.gpu.lengths.value[1] = 480
        self.gpu.cursors.value[1] = np.arange(30, 38)
        self.gpu.shifts.value[1] = np.arange(70, 78)
        self.gpu.counters.value[:] = np.arange(24, dtype=np.uint64).reshape(3, 8)
        self.gpu.verify()

    def assert_same_owned(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for job in expected:
            for name in expected[job]:
                np.testing.assert_array_equal(expected[job][name], actual[job][name], err_msg=f'job {job}, {name}')

    def test_multiple_resize_preserves_every_stack_and_total(self):
        expected = owned_states(self.gpu, self.ledger)
        counts = self.ledger.counter_totals(self.gpu)
        payload = snapshot(self.gpu, self.ledger)
        for n in (2, 5, 1, 12, 3):
            gpu = host_dfs(n)
            ledger = JobLedger.resume(gpu, payload, self.prefixes, self.lengths)
            self.assertEqual(gpu.n, n)
            self.assertEqual(np.count_nonzero(ledger.ids >= 0), min(8, n))
            self.assertEqual(ledger.paused_count, max(0, 8 - n))
            self.assert_same_owned(expected, owned_states(gpu, ledger))
            np.testing.assert_array_equal(counts, ledger.counter_totals(gpu))
            payload = snapshot(gpu, ledger)

    def test_paused_promotion_preserves_overwritten_history_and_frozen_snapshot(self):
        gpu = host_dfs(2)
        ledger = JobLedger.resume(gpu, snapshot(self.gpu, self.ledger), self.prefixes, self.lengths)
        frozen = ledger.fields()
        copies = {key: value.copy() for key, value in frozen.items()}
        counts = ledger.counter_totals(gpu)
        paused_job = int(ledger.paused.remaining_ids()[0])
        expected_cursor = int(ledger.paused.remaining_arrays()['cursors'][1, 0])
        # Finish a lane after eleven additional attempts; its historical totals
        # must move to the accumulator when a paused stack takes its place.
        gpu.counters.value[0, 0] += np.uint64(11)
        gpu.states.value[0] = 2
        gpu.depths.value[0] = 0
        gpu.boards.value[:, 0] = -1
        gpu.used.value[:, 0] = 0
        ledger.refill(gpu, self.prefixes, self.lengths)
        self.assertEqual(ledger.completed, 1)
        self.assertEqual(ledger.paused_count, 5)
        self.assertEqual(ledger.ids[0], paused_job)
        self.assertEqual(gpu.cursors.value[1, 0], expected_cursor)
        np.testing.assert_array_equal(ledger.counter_totals(gpu), counts + np.array([11, 0, 0], np.uint64))
        for key, value in frozen.items():
            self.assertFalse(value.flags.writeable)
            np.testing.assert_array_equal(value, copies[key], err_msg=key)
        rebuilt = host_dfs(5)
        resized = JobLedger.resume(rebuilt, snapshot(gpu, ledger), self.prefixes, self.lengths)
        self.assertEqual(resized.completed, 1)
        np.testing.assert_array_equal(resized.counter_totals(rebuilt), ledger.counter_totals(gpu))

    def test_legacy_checkpoint_without_bank_is_supported(self):
        payload = snapshot(self.gpu, self.ledger)
        payload = {key: value for key, value in payload.items()
                   if key != 'exact_bank_version' and key != 'exact_retired_counters' and not key.startswith('exact_paused_')}
        gpu = host_dfs(1)
        ledger = JobLedger.resume(gpu, payload, self.prefixes, self.lengths)
        self.assertEqual(ledger.paused_count, 7)
        self.assert_same_owned(owned_states(self.gpu, self.ledger), owned_states(gpu, ledger))

    def test_unknown_or_missing_bank_version_rejected(self):
        for replacement in (None, np.int32(2)):
            payload = snapshot(self.gpu, self.ledger)
            if replacement is None:
                payload.pop('exact_bank_version')
            else:
                payload['exact_bank_version'] = replacement
            gpu = host_dfs(1)
            with self.assertRaisesRegex(ValueError, 'version'):
                JobLedger.resume(gpu, payload, self.prefixes, self.lengths)
            self.assertTrue(np.all(gpu.states.get() == 3))

    def test_invalid_unselected_source_lane_is_not_hidden_by_shrink(self):
        payload = snapshot(self.gpu, self.ledger)
        payload['rng'] = payload['rng'].copy()
        payload['rng'][-1] = 0
        with self.assertRaisesRegex(ValueError, 'RNG'):
            JobLedger.resume(host_dfs(1), payload, self.prefixes, self.lengths)

    def test_corrupt_paused_cursor_is_rejected_on_next_resume(self):
        small = host_dfs(1)
        ledger = JobLedger.resume(small, snapshot(self.gpu, self.ledger), self.prefixes, self.lengths)
        payload = snapshot(small, ledger)
        payload['exact_paused_cursors'] = payload['exact_paused_cursors'].copy()
        payload['exact_paused_cursors'][1, -1] = 600
        with self.assertRaisesRegex(ValueError, 'cursor'):
            JobLedger.resume(host_dfs(2), payload, self.prefixes, self.lengths)

    def test_standalone_resume_rejects_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'checkpoint.npz'
            np.savez(target, **snapshot(self.gpu, self.ledger))
            with self.assertRaisesRegex(ValueError, 'JobLedger.resume'):
                host_dfs(8).resume(target)

    @unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA') == '1', 'Optional finite CUDA regression')
    def test_cuda_resize_enumerates_54_unique_leaves(self):
        from itertools import permutations, product
        board = build_board()
        masks = np.zeros((480, 3), np.uint16)
        args = (masks, board.neighbor_cells, board.neighbor_sides)
        rows = np.full((3, 160), -1, np.int16)
        rows[:, :158] = np.arange(158, dtype=np.int16) * 3
        rows[:, 0] = np.arange(3)
        lengths = np.full(3, 158, np.int16)
        expected = set()
        for prefix in rows[:, :158]:
            for tail in permutations((158, 159)):
                for rotations in product(range(3), repeat=2):
                    expected.add(tuple(prefix.tolist() + [3 * tile + rot for tile, rot in zip(tail, rotations)]))
        gpu = DFS(*args, n=3, prefixes=[], seed=491)
        ledger = JobLedger.empty(3, 3)
        ledger.refill(gpu, rows, lengths)
        gpu.step(32)
        payload = snapshot(gpu, ledger)
        gpu = DFS(*args, n=1, prefixes=[], seed=201)
        ledger = JobLedger.resume(gpu, payload, rows, lengths)
        seen = set()
        for iteration in range(2000):
            if iteration == 20:
                payload = snapshot(gpu, ledger)
                gpu = DFS(*args, n=2, prefixes=[], seed=17)
                ledger = JobLedger.resume(gpu, payload, rows, lengths)
            gpu.step(128)
            for lane in gpu.pending():
                result = tuple(map(int, gpu.boards[:, lane].get()[gpu.order]))
                self.assertNotIn(result, seen)
                seen.add(result)
                gpu.acknowledge([int(lane)])
            ledger.refill(gpu, rows, lengths)
            if ledger.completed == 3:
                break
        self.assertEqual(ledger.completed, 3)
        self.assertEqual(seen, expected)
        self.assertEqual(len(seen), 54)


if __name__ == '__main__':
    unittest.main()
