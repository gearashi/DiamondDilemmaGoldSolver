"""Exact WebGPU status totals; hardware cases opt in with DIAMOND_TEST_WEBGPU."""
from pathlib import Path
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from dfs_webgpu import DeviceField, WebGPUDFS, _CounterReduction
from geometry import build_board


class Buffer:
    def __init__(self, label, log):
        self.label, self.log, self.fail = label, log, False
    def destroy(self):
        self.log.append(self.label)
        if self.fail:
            raise RuntimeError('destroy ' + self.label)


class Queue:
    def __init__(self):
        self.values = np.array([0, 2**32, 2**64-1], np.uint64)
        self.submissions = []
        self.fail_submit = self.fail_read = False
        self.mapped = None
    def write_buffer(self, buffer, offset, value):
        self.parameters = value.copy()
    def submit(self, commands):
        if self.fail_submit:
            raise RuntimeError('submit failure')
        self.submissions.append(commands)
    def read_buffer(self, buffer, offset, size):
        if self.fail_read:
            raise RuntimeError('map failure')
        assert offset == 0 and size == 24
        raw = np.empty((3, 2), np.uint32)
        for row, value in enumerate(self.values):
            raw[row] = [int(value) & 0xffffffff, int(value) >> 32]
        self.mapped = bytearray(raw.tobytes())
        return memoryview(self.mapped)


class Device:
    def __init__(self):
        self.log, self.buffers, self.dispatches = [], [], []
        self.fail_shader = False
        self.fail_destroy_first = False
    def create_buffer(self, **kwargs):
        buffer = Buffer(str(len(self.buffers)), self.log)
        buffer.fail = self.fail_destroy_first and not self.buffers
        self.buffers.append(buffer)
        return buffer
    def create_bind_group_layout(self, **kwargs): return object()
    def create_pipeline_layout(self, **kwargs): return object()
    def create_shader_module(self, **kwargs):
        if self.fail_shader:
            raise RuntimeError('shader failure')
        return object()
    def create_compute_pipeline(self, **kwargs): return object()
    def create_bind_group(self, **kwargs): return object()
    def create_command_encoder(self):
        compute = SimpleNamespace(set_pipeline=lambda _: None, set_bind_group=lambda *_: None,
                                  dispatch_workgroups=self.dispatches.append, end=lambda: None)
        return SimpleNamespace(begin_compute_pass=lambda: compute, finish=lambda: object())


def fake_gpu(n=32768):
    device, queue = Device(), Queue()
    return SimpleNamespace(n=n, gpu_device=device, queue=queue, buffers=[object()]*6,
                           readback_bytes=0, wgpu=SimpleNamespace(
                               BufferUsage=SimpleNamespace(STORAGE=1, COPY_SRC=2, COPY_DST=4, UNIFORM=8),
                               ShaderStage=SimpleNamespace(COMPUTE=1)))


def field_fixture(n=32768):
    gpu = SimpleNamespace(n=n)
    field = DeviceField(gpu, object(), 5, (3, n), np.uint64, counters=True)
    gpu.counters = field
    values = np.array([[2**64-1, 3], [2**32, 7], [3, 9]], np.uint64)
    field.get = Mock(return_value=values)
    gpu._counter_totals = Mock(return_value=np.array([21, 22, 23], np.uint64))
    return gpu, field, values


class CounterHostTests(unittest.TestCase):
    def test_gate_and_persistent_view_only(self):
        for n in (1, 128, 4096, 32767, 32768, 131072):
            gpu, field, values = field_fixture(n)
            result = field.sum(axis=1, dtype=np.uint64).get()
            expected = gpu._counter_totals.return_value if n >= 32768 else values.sum(axis=1, dtype=np.uint64)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(gpu._counter_totals.call_count, int(n >= 32768))
        alternate = DeviceField(gpu, object(), 7, (2, n), np.uint64, counters=True)
        alternate.get = Mock(return_value=values[:2])
        before = gpu._counter_totals.call_count
        np.testing.assert_array_equal(alternate.sum(axis=1, dtype=np.uint64).get(), values[:2].sum(axis=1, dtype=np.uint64))
        self.assertEqual(gpu._counter_totals.call_count, before)

    def test_other_numpy_call_forms_keep_values_errors_and_out(self):
        gpu, field, values = field_fixture()
        cases = [((1,), {'dtype': np.uint64}), ((), {}), ((), {'axis': 0, 'dtype': np.uint64}),
                 ((), {'axis': 1, 'dtype': np.dtype('uint64')}),
                 ((), {'axis': 1, 'dtype': np.uint64, 'keepdims': True}),
                 ((), {'axis': True, 'dtype': np.uint64}),
                 ((), {'axis': np.int64(1), 'dtype': np.uint64})]
        for args, kwargs in cases:
            try:
                expected = values.sum(*args, **kwargs)
            except Exception as exc:
                with self.assertRaises(type(exc)):
                    field.sum(*args, **kwargs)
            else:
                np.testing.assert_array_equal(field.sum(*args, **kwargs).get(), expected)
        output = np.empty(3, np.uint64)
        self.assertIs(field.sum(axis=1, dtype=np.uint64, out=output).get(), output)
        np.testing.assert_array_equal(output, values.sum(axis=1, dtype=np.uint64))
        gpu._counter_totals.assert_not_called()

    def test_lazy_helper_is_published_only_after_complete_construction(self):
        gpu = WebGPUDFS.__new__(WebGPUDFS)
        gpu._counter_reduction = None
        helper = Mock()
        helper.get.side_effect = [np.array([1, 2, 3], np.uint64), np.array([4, 5, 6], np.uint64)]
        with patch('dfs_webgpu._CounterReduction', side_effect=[RuntimeError('construction'), helper]) as factory:
            with self.assertRaisesRegex(RuntimeError, 'construction'):
                gpu._counter_totals()
            self.assertIsNone(gpu._counter_reduction)
            np.testing.assert_array_equal(gpu._counter_totals(), [1, 2, 3])
            np.testing.assert_array_equal(gpu._counter_totals(), [4, 5, 6])
            self.assertEqual(factory.call_count, 2)
            self.assertIs(gpu._counter_reduction, helper)

    def test_two_pass_decode_is_fresh_independent_and_bounded(self):
        gpu = fake_gpu(131072)
        reducer = _CounterReduction(gpu)
        self.addCleanup(reducer.close)
        first = reducer.get()
        np.testing.assert_array_equal(first, gpu.queue.values)
        self.assertEqual(gpu.readback_bytes, 24)
        self.assertEqual(gpu.gpu_device.dispatches, [512, 1])
        np.testing.assert_array_equal(gpu.queue.parameters, [131072, 512, 0, 0])
        gpu.queue.mapped[:] = bytes(24)
        np.testing.assert_array_equal(first, [0, 2**32, 2**64-1])
        gpu.queue.values[:] = [2**64-1, 1, 2**32-1]
        second = reducer.get()
        np.testing.assert_array_equal(second, gpu.queue.values)
        self.assertFalse(np.shares_memory(first, second))

    def test_submit_and_map_errors_propagate_without_stale_totals(self):
        gpu = fake_gpu()
        reducer = _CounterReduction(gpu)
        self.addCleanup(reducer.close)
        for name, message in [('fail_submit', 'submit failure'), ('fail_read', 'map failure')]:
            setattr(gpu.queue, name, True)
            with self.assertRaisesRegex(RuntimeError, message):
                reducer.get()
            self.assertEqual(gpu.readback_bytes, 0)
            setattr(gpu.queue, name, False)
        gpu.queue.values[:] = [1, 2, 3]
        np.testing.assert_array_equal(reducer.get(), [1, 2, 3])
        self.assertEqual(gpu.readback_bytes, 24)

    def test_partial_construction_frees_all_buffers_preserving_original_error(self):
        gpu = fake_gpu()
        gpu.gpu_device.fail_shader = gpu.gpu_device.fail_destroy_first = True
        with self.assertRaisesRegex(RuntimeError, 'shader failure'):
            _CounterReduction(gpu)
        self.assertEqual(gpu.gpu_device.log, ['0', '1', '2'])

    def test_close_attempts_all_buffers_once_and_rejects_more_queries(self):
        gpu = fake_gpu()
        reducer = _CounterReduction(gpu)
        gpu.gpu_device.buffers[0].fail = True
        with self.assertRaisesRegex(RuntimeError, 'destroy 0'):
            reducer.close()
        self.assertEqual(gpu.gpu_device.log, ['0', '1', '2'])
        reducer.close()
        self.assertEqual(gpu.gpu_device.log, ['0', '1', '2'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            reducer.get()

    def test_engine_closes_reducer_before_metadata_and_device(self):
        gpu = WebGPUDFS.__new__(WebGPUDFS)
        events = []
        gpu._synchronize = lambda: events.append('fence')
        gpu._counter_reduction = SimpleNamespace(close=lambda: events.append('reducer'))
        gpu.buffers = [Buffer('metadata', events)]
        gpu.parameters = Buffer('parameters', events)
        gpu.gpu_device = Buffer('device', events)
        gpu.close()
        self.assertEqual(events, ['fence', 'reducer', 'metadata', 'parameters', 'device'])
        events.clear()
        def failed_close():
            events.append('reducer')
            raise RuntimeError('reducer destroy failure')
        gpu._counter_reduction.close = failed_close
        with self.assertRaisesRegex(RuntimeError, 'reducer destroy failure'):
            gpu.close()
        self.assertEqual(events, ['fence', 'reducer', 'metadata', 'parameters', 'device'])


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU') == '1', 'native WebGPU tests opt in')
class CounterHardwareTests(unittest.TestCase):
    def test_shader_carry_overflow_padding_and_full_pool(self):
        board = build_board()
        rng = np.random.default_rng(173)
        for n in (1, 255, 256, 257, 32768, 131072):
            with self.subTest(replicas=n):
                gpu = WebGPUDFS(np.zeros((480, 3), np.uint16), board.neighbor_cells, board.neighbor_sides,
                                n=n, prefixes=[], allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU') == '1')
                reducer = None
                try:
                    values = rng.integers(0, np.iinfo(np.uint64).max, size=(3, n), dtype=np.uint64)
                    values[:, 0] = [2**64-1, 2**32, 2**32-1]
                    gpu.counters.set(values)
                    reducer = _CounterReduction(gpu)
                    for data in (values, np.full((3, n), 2**64-1, np.uint64)):
                        gpu.counters.set(data)
                        expected = np.array([sum(map(int, row)) & ((1<<64)-1) for row in data], np.uint64)
                        before = gpu.readback_bytes
                        np.testing.assert_array_equal(reducer.get(), expected)
                        self.assertEqual(gpu.readback_bytes-before, 24)
                        np.testing.assert_array_equal(gpu.counters.get(), data)
                    if n >= 32768:
                        np.testing.assert_array_equal(gpu.counters.sum(axis=1, dtype=np.uint64).get(), expected)
                finally:
                    if reducer is not None:
                        reducer.close()
                    gpu.close()


if __name__ == '__main__':
    unittest.main()
