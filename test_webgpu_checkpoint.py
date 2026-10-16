"""CPU-only checks for full-buffer WebGPU checkpoint capture.

Arbitrary stored bit patterns test representation, not legal search states.
Hardware traversal, resizing and checkpoint interchange live in test_webgpu.
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from dfs_gpu import DFS
from dfs_webgpu import WebGPUDFS, DeviceField
from checkpoint_writer import CheckpointWriter

ROOT = Path(__file__).resolve().parent

class Queue:
    def __init__(self):
        self.calls = []
        self.fail_at = None
        self.pending = None
        self.mapped = []
    def read_buffer(self, buffer, offset=0, size=None):
        self.calls.append((offset, size))
        if self.fail_at == len(self.calls):
            raise RuntimeError('Injected checkpoint map failure')
        if self.pending:
            callback, self.pending = self.pending, None
            callback()
        value = buffer.view(np.uint8).reshape(-1)
        mapped = bytearray(value[offset:offset + size].tobytes())
        self.mapped.append(mapped)
        return memoryview(mapped)

def memory_engine(n):
    gpu = WebGPUDFS.__new__(WebGPUDFS)
    gpu.n, gpu.readback_bytes, gpu.queue = n, 0, Queue()
    rng = np.random.default_rng(218)
    gpu.buffers = [rng.integers(0, 2**32, size=(rows, n), dtype=np.uint32)
                   for rows in (165, 160, 160, 160, 160, 11)]
    gpu.buffers[0][0] = np.uint32(0xffffffff)
    gpu.buffers[5][0] = np.uint32(0xffffffff)
    counter_values = np.resize(np.array([0, 1, 2**32-1, 2**32, 2**40+17, 2**63, 2**64-1], np.uint64), (3,n))
    gpu.buffers[5][5::2] = (counter_values & np.uint64(0xffffffff)).astype(np.uint32)
    gpu.buffers[5][6::2] = (counter_values >> np.uint64(32)).astype(np.uint32)
    gpu.boards = DeviceField(gpu, gpu.buffers[0], 0, (160, n), np.int16)
    gpu.used = DeviceField(gpu, gpu.buffers[0], 160, (5, n), np.uint32)
    for name, buffer in zip(('firsts', 'lengths', 'cursors', 'shifts'), gpu.buffers[1:5]):
        setattr(gpu, name, DeviceField(gpu, buffer, 0, (160, n), np.uint16))
    for name, offset, dtype in (('depths', 0, np.int16), ('maxdepths', 1, np.int16),
                               ('floors', 2, np.int16), ('states', 3, np.uint8),
                               ('rng', 4, np.uint32)):
        setattr(gpu, name, DeviceField(gpu, gpu.buffers[5], offset, (n,), dtype))
    gpu.counters = DeviceField(gpu, gpu.buffers[5], 5, (3, n), np.uint64, counters=True)
    gpu.cp = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:
                             SimpleNamespace(synchronize=gpu._synchronize)))
    gpu.fingerprint = 'independent-storage-oracle'
    gpu.order = np.arange(160, dtype=np.int16)[::-1].copy()
    gpu.prefix_codes = np.full((n, 160), -1, np.int16)
    gpu.roots = np.arange(n, dtype=np.int16)
    return gpu

def oracle(gpu):
    # Independent scalar interpretation of the physical representation.
    result = {}
    specs = [('boards',0,0,160,np.int16),('used',0,160,5,np.uint32),
             ('depths',5,0,1,np.int16),('maxdepths',5,1,1,np.int16),
             ('states',5,3,1,np.uint8),('floors',5,2,1,np.int16),
             ('firsts',1,0,160,np.uint16),('lengths',2,0,160,np.uint16),
             ('cursors',3,0,160,np.uint16),('shifts',4,0,160,np.uint16),
             ('rng',5,4,1,np.uint32)]
    for name, buffer, offset, rows, dtype in specs:
        bits = np.dtype(dtype).itemsize * 8
        values = [[int(gpu.buffers[buffer][offset + row, col]) & ((1 << bits)-1)
                   for col in range(gpu.n)] for row in range(rows)]
        if np.dtype(dtype).kind == 'i':
            values = [[value-(1 << bits) if value >= 1 << (bits-1) else value
                       for value in row] for row in values]
        result[name] = np.array(values, dtype=dtype)
        if getattr(gpu, name).ndim == 1:
            result[name] = result[name][0]
    result['counters'] = np.array([[int(gpu.buffers[5][5+2*row,col]) +
                                  (int(gpu.buffers[5][6+2*row,col]) << 32)
                                  for col in range(gpu.n)] for row in range(3)], np.uint64)
    return result

class RepresentationTests(unittest.TestCase):
    def test_exact_schema_dtypes_scalar_oracle_and_baseline(self):
        for n in (1,3,128):
            gpu = memory_engine(n)
            expected = oracle(gpu)
            baseline = DFS.snapshot_checkpoint(gpu)
            self.assertEqual(len(gpu.queue.calls), 13)
            gpu.queue.calls.clear()
            before = gpu.readback_bytes
            actual = WebGPUDFS.snapshot_checkpoint(gpu)
            self.assertEqual(list(actual), list(baseline))
            self.assertEqual(gpu.queue.calls, [(0, rows*n*4) for rows in (165,160,160,160,160,11)])
            self.assertEqual(gpu.readback_bytes-before, 816*n*4)
            for name, value in actual.items():
                np.testing.assert_array_equal(value, baseline[name], err_msg=name)
                self.assertEqual(np.asarray(value).dtype, np.asarray(baseline[name]).dtype)
                if name in expected:
                    np.testing.assert_array_equal(value, expected[name], err_msg=name)

    def test_independent_read_only_outputs_survive_all_input_mutations(self):
        gpu = memory_engine(3)
        actual = WebGPUDFS.snapshot_checkpoint(gpu)
        expected = {name: np.asarray(value).copy() for name,value in actual.items()}
        for buffer in gpu.buffers: buffer.fill(0)
        for mapped in gpu.queue.mapped: mapped[:] = bytes(len(mapped))
        gpu.order.fill(0); gpu.roots.fill(0); gpu.prefix_codes.fill(0)
        for name,value in actual.items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)
            if isinstance(value,np.ndarray):
                self.assertFalse(value.flags.writeable)
                for buffer in gpu.buffers:
                    self.assertFalse(np.shares_memory(value,buffer))
                with self.assertRaises(ValueError): value.flat[0] = 0

    def test_first_map_fences_pending_work_and_all_rows_come_after_it(self):
        gpu = memory_engine(3)
        def pending():
            for buffer in gpu.buffers: buffer.fill(37)
        gpu.queue.pending = pending
        actual = WebGPUDFS.snapshot_checkpoint(gpu)
        expected = oracle(gpu)
        for name,value in expected.items():
            np.testing.assert_array_equal(actual[name],value,err_msg=name)
        self.assertIsNone(gpu.queue.pending)

    def test_each_read_failure_never_submits_partial_payload_or_replaces_old_file(self):
        for failure in range(1,7):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="atomic-io-test-webgpu-", dir=ROOT) as folder:
                path=Path(folder)/'checkpoint.npz'
                path.write_bytes(b'previous-good-checkpoint')
                gpu=memory_engine(3);gpu.queue.fail_at=failure
                writes=[]
                def write(payload,generation):
                    writes.append(generation)
                    with path.open('wb') as stream: np.savez(stream,**payload)
                writer=CheckpointWriter(lambda:WebGPUDFS.snapshot_checkpoint(gpu),write)
                try:
                    with self.assertRaisesRegex(RuntimeError,'map failure'): writer.request(7)
                    self.assertIsNone(writer.future)
                    self.assertIsNone(writer.pending_generation)
                    self.assertEqual(writes,[])
                    self.assertEqual(path.read_bytes(),b'previous-good-checkpoint')
                finally: writer.close()

    def test_stop_finish_writes_independent_frozen_snapshot_and_npz_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="atomic-io-test-webgpu-", dir=ROOT) as folder:
            path=Path(folder)/'checkpoint.npz';gpu=memory_engine(3)
            expected=WebGPUDFS.snapshot_checkpoint(gpu)
            def write(payload,generation):
                for buffer in gpu.buffers:buffer.fill(0)
                with path.open('wb') as stream:np.savez(stream,**payload)
                return {'generation':generation}
            writer=CheckpointWriter(lambda:WebGPUDFS.snapshot_checkpoint(gpu),write)
            try:
                writer.finish(5)
                self.assertEqual(writer.completed_generation,5)
                with np.load(path,allow_pickle=False) as saved:
                    self.assertEqual(saved.files,list(expected))
                    for name,value in expected.items():
                        np.testing.assert_array_equal(saved[name],value,err_msg=name)
            finally:writer.close()


if __name__ == "__main__": unittest.main()
