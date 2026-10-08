"""Native WebGPU exact DFS tests; GPU cases opt in with DIAMOND_TEST_WEBGPU=1."""
from itertools import permutations, product
import os
import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from dfs_gpu import DFS, fill_order
from gpu_engine import reverse11
from dfs_webgpu import DeviceField, WebGPUDFS
from geometry import build_board
from systematic_jobs import JobLedger


def arguments():
    board = build_board()
    return np.zeros((480, 3), np.uint16), board.neighbor_cells, board.neighbor_sides


def checkpoint(gpu, ledger):
    payload = gpu.snapshot_checkpoint()
    payload.update(ledger.fields())
    return payload


class ArrayTests(unittest.TestCase):
    def test_selected_storage_views_and_unsigned_counter_limbs(self):
        class Memory:
            n = 4
            def _read_rows(self, buffer, rows, cols): return buffer[np.ix_(rows, cols)].copy()
            def _write_rows(self, buffer, rows, cols, values): buffer[np.ix_(rows, cols)] = values
        engine = Memory()
        buffer = np.zeros((8, 4), np.uint32)
        signed = DeviceField(engine, buffer, 0, (2, 4), np.int16)
        signed.set(-1)
        np.testing.assert_array_equal(signed.get(), np.full((2, 4), -1))
        signed[:, [3, 1]] = np.array([[4, 5], [6, 7]], np.int16)
        np.testing.assert_array_equal(signed[:, 1].get(), [5, 7])
        self.assertEqual(signed[1, 3].get(), 6)
        counters = DeviceField(engine, buffer, 2, (3, 4), np.uint64, counters=True)
        values = np.arange(12, dtype=np.uint64).reshape(3, 4) + np.uint64(2**40)
        counters.set(values)
        np.testing.assert_array_equal(counters.get(), values)
        np.testing.assert_array_equal(counters[:, [2, 0]].get(), values[:, [2, 0]])
        counters[1, 2] = np.uint64(2**64 - 1)
        self.assertEqual(int(counters[1, 2].get()), 2**64 - 1)

    def test_cpu_and_opengl_adapters_are_rejected(self):
        for info in ({'adapter_type': 'CPU', 'backend_type': 'Vulkan'},
                     {'adapter_type': 'DiscreteGPU', 'backend_type': 'OpenGL'}):
            with self.subTest(info=info), patch.dict('sys.modules', {'wgpu': SimpleNamespace()}):
                with self.assertRaisesRegex(RuntimeError, 'software|Native'):
                    WebGPUDFS(*arguments(), n=1, prefixes=[], adapter=SimpleNamespace(info=info))

    def test_storage_limits_fail_before_requesting_a_device(self):
        for limits, message in (
            ({'max-storage-buffer-binding-size': 128, 'max-buffer-size': 2**30,
              'max-storage-buffers-per-shader-stage': 8}, 'storage-buffer limit'),
            ({'max-storage-buffer-binding-size': 2**30, 'max-buffer-size': 2**30,
              'max-storage-buffers-per-shader-stage': 6}, 'seven compute storage')):
            adapter = SimpleNamespace(info={'adapter_type': 'DiscreteGPU', 'backend_type': 'Vulkan'}, limits=limits)
            with self.subTest(limits=limits), patch.dict('sys.modules', {'wgpu': SimpleNamespace()}):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    WebGPUDFS(*arguments(), n=128, prefixes=[], adapter=adapter)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU') == '1', 'Optional native hardware GPU test')
class NativeTests(unittest.TestCase):
    def new(self, n=3, seed=41):
        gpu = WebGPUDFS(*arguments(), n=n, prefixes=[], seed=seed, allow_software=os.environ.get("DIAMOND_TEST_SOFTWARE_GPU") == "1")
        self.addCleanup(gpu.close)
        return gpu

    def test_finite_leaves_shrink_grow_pending_freeze_and_checkpoint(self):
        rows = np.full((3, 160), -1, np.int16)
        rows[:, :158] = np.arange(158, dtype=np.int16) * 3
        rows[:, 0] = np.arange(3)
        lengths = np.full(3, 158, np.int16)
        expected = set()
        for prefix in rows[:, :158]:
            for tail in permutations((158, 159)):
                for rotations in product(range(3), repeat=2):
                    expected.add(tuple(prefix.tolist() + [3 * tile + rot for tile, rot in zip(tail, rotations)]))
        gpu = self.new()
        ledger = JobLedger.empty(3, 3)
        ledger.refill(gpu, rows, lengths)
        seen = set()
        resized = False
        for iteration in range(2000):
            if iteration in (4, 16):
                payload = checkpoint(gpu, ledger)
                previous_counts = ledger.counter_totals(gpu)
                gpu = self.new(1 if iteration == 4 else 2, seed=77)
                ledger = JobLedger.resume(gpu, payload, rows, lengths)
                np.testing.assert_array_equal(ledger.counter_totals(gpu), previous_counts)
                resized = True
            before = gpu.readback_bytes
            gpu.step(128)
            self.assertEqual(gpu.readback_bytes - before, 4, 'A launch should fence four bytes, not read back stacks.')
            gpu.verify()
            pending = gpu.pending()
            if len(pending):
                frozen = gpu.boards[:, pending].get()
                gpu.step(32)
                np.testing.assert_array_equal(frozen, gpu.boards[:, pending].get())
            for lane in pending:
                result = tuple(map(int, gpu.boards[:, lane].get()[gpu.order]))
                self.assertNotIn(result, seen)
                self.assertIn(result, expected)
                seen.add(result)
                gpu.acknowledge([int(lane)])
            ledger.refill(gpu, rows, lengths)
            if ledger.completed == 3:
                break
        self.assertTrue(resized)
        self.assertEqual(ledger.completed, 3)
        self.assertEqual(ledger.paused_count, 0)
        self.assertEqual(seen, expected)
        self.assertEqual(len(seen), 54)

    def test_counter_carry_selective_readback_and_deterministic_resume(self):
        gpu = self.new(2, seed=3)
        gpu.load_prefixes([list(np.arange(158) * 3)], lanes=[1])
        counts = np.full((3, 2), 2**32 - 16, np.uint64)
        gpu.counters.set(counts)
        before = gpu.readback_bytes
        board = gpu.boards[:, 1].get()
        self.assertEqual(gpu.readback_bytes - before, 160 * 4)
        self.assertEqual(np.count_nonzero(board >= 0), 158)
        gpu.step(32)
        after = gpu.counters.get()
        self.assertEqual(int(after[0, 1]), int(counts[0, 1]) + 32)
        np.testing.assert_array_equal(after[:, 0], counts[:, 0])
        payload = gpu.snapshot_checkpoint()
        restored = self.new(2, seed=991)
        restored.restore_checkpoint(payload)
        for _ in range(4):
            gpu.step(64)
            restored.step(64)
        for name in DFS.DEVICE_FIELDS:
            np.testing.assert_array_equal(getattr(gpu, name).get(), getattr(restored, name).get(), err_msg=name)
        restored.verify()

    def test_asymmetric_masks_varied_order_and_pruning(self):
        board = build_board()
        rng = np.random.default_rng(107)
        base = np.zeros((160, 3), np.uint16)
        palette = np.array([3, 6, 0x111, 0x205, 0x241, 0x403], np.uint16)
        for a, sa, b, sb in board.edges:
            value = rng.choice(palette)
            base[a, sa], base[b, sb] = value, reverse11(value)
        masks = np.array([row[(np.arange(3)-rotation)%3] for row in base for rotation in range(3)], np.uint16)
        self.assertTrue(np.any(masks != reverse11(masks)))
        order = fill_order(board.neighbor_cells, start=17)
        gpu = WebGPUDFS(masks, board.neighbor_cells, board.neighbor_sides, n=3, seed=99,
                        prefixes=[[0], [31], [62]], order=order,
                        allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU') == '1')
        self.addCleanup(gpu.close)
        for _ in range(20):
            gpu.step(32)
            gpu.verify()
        counts = gpu.counters.get().sum(axis=1)
        self.assertGreater(int(counts[0]), 0)
        self.assertGreater(int(counts[0]), int(counts[1]))

    @unittest.skipUnless(os.environ.get('DIAMOND_CUDA_PYTHON'), 'Optional separate CUDA interpreter comparison')
    def test_cuda_checkpoint_interchange_every_launch(self):
        board = build_board()
        order = fill_order(board.neighbor_cells, start=17)
        if os.environ.get('DIAMOND_TEST_TILE_DATA'):
            from validator import orientation_masks
            data = json.loads(Path(os.environ['DIAMOND_TEST_TILE_DATA']).read_text(encoding='utf-8-sig'))
            masks = np.asarray(orientation_masks(data), np.uint16)
        else:
            rng = np.random.default_rng(981)
            base = rng.choice(np.array([3, 6, 0x111, 0x205, 0x241, 0x403], np.uint16), size=(160, 3))
            masks = np.array([row[(np.arange(3)-rotation)%3] for row in base for rotation in range(3)], np.uint16)
        gpu = WebGPUDFS(masks, board.neighbor_cells, board.neighbor_sides, n=3, seed=297,
                        prefixes=[[0], [31], [62]], order=order)
        self.addCleanup(gpu.close)
        with tempfile.TemporaryDirectory(prefix='diamond-cross-gpu-') as folder:
            folder = Path(folder)
            initial = folder / 'initial.npz'
            payload = gpu.snapshot_checkpoint()
            payload.update(test_masks=masks, test_neighbors=np.asarray(board.neighbor_cells, np.int16),
                           test_sides=np.asarray(board.neighbor_sides, np.uint8))
            np.savez(initial, **payload)
            script = """import sys,numpy as np
from pathlib import Path
from dfs_gpu import DFS
root=Path(sys.argv[1])
with np.load(root/'initial.npz',allow_pickle=False) as source:
    g=DFS(source['test_masks'],source['test_neighbors'],source['test_sides'],n=3,prefixes=[],order=source['order'])
g.resume(root/'initial.npz')
result={}
for step in range(20):
    g.step(32)
    g.verify()
    for name,value in g.snapshot_checkpoint().items():
        result[str(step)+'_'+name]=value
np.savez(root/'cuda-steps.npz',**result)
print(g.device)
"""
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUPY_CACHE_DIR=str(folder/'cuda-cache'))
            result = subprocess.run([os.environ['DIAMOND_CUDA_PYTHON'], '-c', script, str(folder)],
                                    cwd=Path(__file__).resolve().parent, env=environment,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with np.load(folder/'cuda-steps.npz', allow_pickle=False) as archive:
                for step in range(20):
                    gpu.step(32)
                    for name in DFS.DEVICE_FIELDS:
                        np.testing.assert_array_equal(getattr(gpu,name).get(), archive[str(step)+'_'+name],
                                                      err_msg=f'CUDA/WebGPU step {step}, {name}')
                latest = {key[len('19_'):]: archive[key] for key in archive.files if key.startswith('19_')}
            restored = WebGPUDFS(masks, board.neighbor_cells, board.neighbor_sides, n=3, seed=8,
                                 prefixes=[], order=order)
            self.addCleanup(restored.close)
            restored.restore_checkpoint(latest)
            gpu.step(64)
            restored.step(64)
            for name in DFS.DEVICE_FIELDS:
                np.testing.assert_array_equal(getattr(gpu,name).get(), getattr(restored,name).get(), err_msg=name)


if __name__ == '__main__': unittest.main()
