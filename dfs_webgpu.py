"""Persistent native WebGPU DFS (Metal/Vulkan/DX12), with CUDA-compatible states.

All search transitions run in WGSL on a hardware GPU. NumPy initializes and
validates states and supports checkpoint I/O; there is no CPU search fallback.
"""
from __future__ import annotations
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from dfs_gpu import DFS

ROOT = Path(__file__).resolve().parent


class _Readback:
    def __init__(self, field, key):
        self.field, self.key = field, key
    def get(self):
        return self.field._get(self.key)
    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.get(), dtype=dtype)
        return value.copy() if copy else value


class _HostResult:
    def __init__(self, value): self.value = value
    def get(self): return self.value


class _CounterReduction:
    """Private two-pass exact uint64 reduction for large-pool status queries."""
    def __init__(self, gpu):
        self.gpu = gpu
        self.buffers = []
        self.closed = False
        if not 1 <= gpu.n <= 131072:
            raise ValueError('Counter reduction requires 1..131072 lanes.')
        self.groups = (gpu.n + 255) // 256
        device, wgpu = gpu.gpu_device, gpu.wgpu
        try:
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            self.partial = device.create_buffer(size=self.groups * 24, usage=usage)
            self.buffers.append(self.partial)
            self.result = device.create_buffer(size=24, usage=usage)
            self.buffers.append(self.result)
            self.parameters = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self.buffers.append(self.parameters)
            gpu.queue.write_buffer(self.parameters, 0, np.array([gpu.n, self.groups, 0, 0], np.uint32))
            layout = device.create_bind_group_layout(entries=[
                {'binding': i, 'visibility': wgpu.ShaderStage.COMPUTE,
                 'buffer': {'type': kind}}
                for i, kind in enumerate(('read-only-storage', 'storage', 'storage', 'uniform'))])
            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
            shader = device.create_shader_module(code=(ROOT / 'dfs_counter_reduce.wgsl').read_text(encoding='utf-8'))
            self.first = device.create_compute_pipeline(layout=pipeline_layout,
                compute={'module': shader, 'entry_point': 'reduce_lanes'})
            self.last = device.create_compute_pipeline(layout=pipeline_layout,
                compute={'module': shader, 'entry_point': 'reduce_groups'})
            self.bind_group = device.create_bind_group(layout=layout, entries=[
                {'binding': i, 'resource': {'buffer': buffer}}
                for i, buffer in enumerate((gpu.buffers[5], self.partial, self.result, self.parameters))])
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass  # Preserve the original initialization failure.
            raise

    def get(self):
        if self.closed: raise RuntimeError('Counter reduction is closed.')
        encoder = self.gpu.gpu_device.create_command_encoder()
        for pipeline, groups in ((self.first, self.groups), (self.last, 1)):
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, self.bind_group)
            compute.dispatch_workgroups(groups)
            compute.end()
        self.gpu.queue.submit([encoder.finish()])
        raw = np.frombuffer(self.gpu.queue.read_buffer(self.result, 0, 24), np.uint32).reshape(3, 2)
        self.gpu.readback_bytes += 24
        values = raw[:, 1].astype(np.uint64)
        values <<= np.uint64(32)
        values |= raw[:, 0]
        return values

    def close(self):
        if not self.closed:
            self.closed = True
            first_error = None
            for buffer in self.buffers:
                try:
                    buffer.destroy()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            self.buffers.clear()
            if first_error is not None:
                raise first_error


class DeviceField:
    """Typed logical array over u32 storage; selected reads copy only needed rows.

    The public shape/dtype/checkpoint representation matches the CUDA engine.
    A matrix column read gathers its rows on the GPU before one host transfer.
    """
    def __init__(self, engine, buffer, row_offset, shape, dtype, counters=False):
        self.engine, self.buffer, self.row_offset = engine, buffer, row_offset
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.ndim, self.counters = len(shape), counters

    def _indices(self, key):
        if key is Ellipsis: key = slice(None)
        if self.ndim == 1:
            rows, col_key = np.array([0], np.int64), key
            row_scalar = True
        else:
            row_key, col_key = key if isinstance(key, tuple) else (key, slice(None))
            rows = np.arange(self.shape[0], dtype=np.int64)[row_key]
            row_scalar = np.ndim(rows) == 0
            rows = np.atleast_1d(rows)
        cols = np.arange(self.shape[-1], dtype=np.int64)[col_key]
        col_scalar = np.ndim(cols) == 0
        return rows, np.atleast_1d(cols), row_scalar, col_scalar

    def _get(self, key):
        rows, cols, row_scalar, col_scalar = self._indices(key)
        physical = (self.row_offset + (rows[:, None] * 2 + [0, 1]).ravel()
                    if self.counters else self.row_offset + rows)
        raw = self.engine._read_rows(self.buffer, physical, cols)
        if self.counters:
            values = raw[0::2].astype(np.uint64) | (raw[1::2].astype(np.uint64) << np.uint64(32))
        else:
            values = raw.astype(self.dtype)
        if row_scalar: values = values[0]
        if col_scalar: values = values[..., 0]
        return values

    def get(self): return self._get(slice(None))
    def __getitem__(self, key): return _Readback(self, key)
    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.get(), dtype=dtype)
        return result.copy() if copy else result
    def max(self): return _HostResult(self.get().max())
    def sum(self, *args, **kwargs):
        # Only the persistent whole-counter view has the shader's fixed layout.
        # Smaller pools and all other NumPy call forms retain their old behavior.
        if (self.counters and self is getattr(self.engine, 'counters', None)
                and self.engine.n >= 32768 and not args
                and kwargs.keys() == {'axis', 'dtype'}
                and type(kwargs['axis']) is int and kwargs['axis'] == 1
                and kwargs['dtype'] is np.uint64):
            return _HostResult(self.engine._counter_totals())
        return _HostResult(self.get().sum(*args, **kwargs))
    def set(self, value): self.__setitem__(slice(None), value)

    def __setitem__(self, key, value):
        rows, cols, row_scalar, col_scalar = self._indices(key)
        target_shape = (() if row_scalar else (len(rows),)) + (() if col_scalar else (len(cols),))
        value = np.broadcast_to(np.asarray(value, dtype=self.dtype), target_shape).reshape(len(rows), len(cols))
        if self.counters:
            physical = self.row_offset + (rows[:, None] * 2 + [0, 1]).ravel()
            encoded = np.empty((2 * len(rows), len(cols)), np.uint32)
            encoded[0::2] = value & np.uint64(0xffffffff)
            encoded[1::2] = value >> np.uint64(32)
        else:
            physical = self.row_offset + rows
            encoded = value.astype(np.uint32)
        self.engine._write_rows(self.buffer, physical, cols, encoded)


class WebGPUDFS(DFS):
    """Same exact DFS/checkpoint interface, backed by persistent WebGPU buffers."""
    backend = 'webgpu'

    def __init__(self, masks, neighbors, neighbor_sides, n=3840, seed=20261007,
                 root_codes=None, order=None, prefixes=None, adapter=None, allow_software=False, gold_data=None, gpu_features=None):
        initial = self._prepare_host(masks, neighbors, neighbor_sides, n, root_codes, order, prefixes)
        self._prepare_gold(gold_data)
        from gpu_reasoning import prepare_policy
        prepare_policy(self, gpu_features, gold_data)
        self._prepare_tables()
        try:
            import wgpu
        except ImportError as exc:
            raise RuntimeError('WebGPU requires wgpu==0.32.0. Run setup with the WebGPU backend.') from exc
        self.wgpu = wgpu
        self.adapter = adapter or wgpu.gpu.request_adapter_sync(power_preference='high-performance', force_fallback_adapter=False)
        if self.adapter is None:
            raise RuntimeError('No WebGPU adapter found. A native Metal, Vulkan, or DX12 hardware GPU is required.')
        self.adapter_info = dict(self.adapter.info)
        adapter_type = str(self.adapter_info.get('adapter_type', '')).lower()
        description = ' '.join(str(v) for v in self.adapter_info.values()).lower()
        software = adapter_type in ('cpu', 'software') or any(token in description for token in ('llvmpipe', 'lavapipe', 'swiftshader', 'microsoft basic render'))
        self.is_hardware = not software
        # Explicit shader-test opt-in only; production factories never enable it.
        if software and not allow_software:

            raise RuntimeError('WebGPU selected a software adapter; hardware GPU search is required and CPU fallback is disabled.')
        if self.adapter_info.get('backend_type') not in ('Metal', 'Vulkan', 'D3D12'):
            raise RuntimeError('Native Metal, Vulkan, or DX12 is required; OpenGL and software search adapters are unsupported.')
        self.device = 'WebGPU: ' + str(self.adapter_info.get('device') or self.adapter_info.get('description') or 'hardware adapter')
        self.device += ' [' + str(self.adapter_info.get('backend_type', 'native')) + ']'
        tables = np.concatenate((self.order, self.host_faces.ravel(), self.host_reverse.ravel(),
                                 self.neighbors.ravel(), self.neighbor_sides.ravel(),
                                 self.host_single_offsets, self.host_pair_offsets, self.host_pool,
                                 self.host_gold_partners.ravel(), self.order_rank, self.host_policy)).astype(np.uint32)
        sizes = [165 * n * 4] + [160 * n * 4] * 4 + [12 * n * 4, tables.nbytes]
        limits = self.adapter.limits
        if max(sizes) > limits['max-storage-buffer-binding-size'] or max(sizes) > limits['max-buffer-size']:
            raise ValueError('Requested replica pool exceeds this GPU storage-buffer limit; select fewer replicas.')
        if limits['max-storage-buffers-per-shader-stage'] < 7:
            raise RuntimeError('This WebGPU device exposes fewer than seven compute storage buffers.')
        self.gpu_device = self.adapter.request_device_sync(required_limits={
            'max-storage-buffer-binding-size': max(sizes), 'max-buffer-size': max(sizes),
            'max-storage-buffers-per-shader-stage': 7})
        self.queue = self.gpu_device.queue
        self._counter_reduction = None
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self.buffers = [self.gpu_device.create_buffer(label=f'DFS storage {i}', size=size, usage=usage)
                        for i, size in enumerate(sizes)]
        self.queue.write_buffer(self.buffers[6], 0, tables)
        self.parameters = self.gpu_device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        source = (ROOT / 'dfs_kernels.wgsl').read_text(encoding='utf-8')
        source = 'const REASONING_FLAGS: u32 = '+str(self.feature_bits)+'u;\n'+source.replace('parameters.flags', 'REASONING_FLAGS')
        shader = self.gpu_device.create_shader_module(code=source)
        self.pipeline = self.gpu_device.create_compute_pipeline(layout='auto', compute={'module': shader, 'entry_point': 'dfs_search'})
        entries = [{'binding': i, 'resource': {'buffer': buffer}} for i, buffer in enumerate(self.buffers)]
        entries.append({'binding': 7, 'resource': {'buffer': self.parameters}})
        self.bind_group = self.gpu_device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
        self.readback_bytes = 0
        self.boards = DeviceField(self, self.buffers[0], 0, (160, n), np.int16)
        self.used = DeviceField(self, self.buffers[0], 160, (5, n), np.uint32)
        for name, buffer in zip(('firsts', 'lengths', 'cursors', 'shifts'), self.buffers[1:5]):
            setattr(self, name, DeviceField(self, buffer, 0, (160, n), np.uint16))
        for name, offset, dtype in [('depths', 0, np.int16), ('maxdepths', 1, np.int16),
                                    ('floors', 2, np.int16), ('states', 3, np.uint8), ('rng', 4, np.uint32)]:
            setattr(self, name, DeviceField(self, self.buffers[5], offset, (n,), dtype))
        self.counters = DeviceField(self, self.buffers[5], 5, (3, n), np.uint64, counters=True)
        self.gold_checked = DeviceField(self, self.buffers[5], 11, (n,), np.uint8)
        self.cp = SimpleNamespace(asarray=np.asarray, maximum=np.maximum,
                                  cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=self._synchronize)))
        self.boards.set(-1)
        self.cursors.set(65535)
        self.states.set(3)
        self.rng.set(np.random.default_rng(seed).integers(1, 2**32, n, dtype=np.uint32))
        self.prefix_codes = np.full((n, 160), -1, np.int16)
        self.prefixes = [None] * n
        self.roots = np.full(n, -1, np.int16)
        if len(initial):
            self.load_prefixes(initial, lanes=np.arange(len(initial), dtype=np.int32))

    def _read_rows(self, buffer, rows, cols):
        if not len(rows) or not len(cols): return np.empty((len(rows), len(cols)), np.uint32)
        left, right = int(cols.min()), int(cols.max()) + 1
        width = right - left
        if left == 0 and right == self.n and np.all(np.diff(rows) == 1):
            result = np.frombuffer(self.queue.read_buffer(buffer, int(rows[0]) * self.n * 4, len(rows) * self.n * 4), np.uint32).reshape(len(rows), self.n)
        else:
            # Gather strided board cells with GPU buffer copies, then map once.
            size = len(rows) * width * 4
            staging = self.gpu_device.create_buffer(size=size, usage=self.wgpu.BufferUsage.COPY_SRC | self.wgpu.BufferUsage.COPY_DST)
            encoder = self.gpu_device.create_command_encoder()
            for i, row in enumerate(rows):
                encoder.copy_buffer_to_buffer(buffer, (int(row) * self.n + left) * 4, staging, i * width * 4, width * 4)
            self.queue.submit([encoder.finish()])
            result = np.frombuffer(self.queue.read_buffer(staging), np.uint32).reshape(len(rows), width)
            staging.destroy()
        self.readback_bytes += len(rows) * width * 4
        return result[:, cols - left].copy()

    def _write_rows(self, buffer, rows, cols, values):
        if not len(rows) or not len(cols): return
        if np.array_equal(cols, np.arange(self.n)) and np.all(np.diff(rows) == 1):
            self.queue.write_buffer(buffer, int(rows[0]) * self.n * 4, np.ascontiguousarray(values))
            return
        breaks = np.flatnonzero(np.diff(cols) != 1) + 1
        starts, stops = np.r_[0, breaks], np.r_[breaks, len(cols)]
        for i, row in enumerate(rows):
            for start, stop in zip(starts, stops):
                self.queue.write_buffer(buffer, (int(row) * self.n + int(cols[start])) * 4,
                                        np.ascontiguousarray(values[i, start:stop]))

    def _synchronize(self):
        # Mapping a four-byte copy fences preceding queue work. This also avoids
        # the work-done callback ABI mismatch in some wgpu-native 0.32 builds.
        self.queue.read_buffer(self.buffers[5], 0, 4)
        self.readback_bytes += 4

    def _upload_policy(self):
        offset = (4000 + 3*self.colors + 1 + 3*self.colors*self.colors + 1 + 3360 + 15840 + 160)*4
        self.queue.write_buffer(self.buffers[6], offset, self.host_policy)

    def _dispatch_step(self, nodes):
        if type(nodes) is not int or not 1 <= nodes <= 512:
            raise ValueError('Use a bounded work budget of 1..512 per launch.')
        self.queue.write_buffer(self.parameters, 0, np.array([self.n, nodes, self.colors, self.gold_segments, self.feature_bits, 0, 0, 0], np.uint32))
        encoder = self.gpu_device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.pipeline)
        compute.set_bind_group(0, self.bind_group)
        compute.dispatch_workgroups((self.n + 63) // 64)
        compute.end()
        self.queue.submit([encoder.finish()])

    def step(self, nodes=128):
        started = time.perf_counter()
        held=self._pause_cpu_owned()
        try:
            self._dispatch_step(nodes)
            self._synchronize()
            return (time.perf_counter() - started) * 1000
        finally:self._restore_cpu_owned(held)

    def step_and_states(self, nodes=128):
        """Complete a bounded launch and read its lane states with one mapping.

        The states read fences the submitted work. Callers already needing these
        flags can avoid step()'s separate four-byte fence; step() retains its
        synchronous timing contract for every other caller.
        """
        held=self._pause_cpu_owned()
        try:
            self._dispatch_step(nodes)
            self._synchronize() if held else None
        finally:self._restore_cpu_owned(held)
        return self.states.get()

    def snapshot_checkpoint(self):
        """Capture independent frozen logical arrays from six storage buffers."""
        # The first synchronous map fences all prior queue work. There are no
        # submissions between these reads on the search thread. Decode each mapping
        # before releasing it; no returned array aliases device/staging memory.
        fields = {}
        def read(index, rows):
            size = rows * self.n * 4
            raw = np.frombuffer(self.queue.read_buffer(self.buffers[index], 0, size),
                                dtype=np.uint32).reshape(rows, self.n)
            self.readback_bytes += size
            return raw
        raw = read(0, 165)
        fields['boards'] = raw[:160].astype(np.int16)
        fields['used'] = raw[160:].astype(np.uint32, copy=True)
        del raw
        for index, name in enumerate(('firsts', 'lengths', 'cursors', 'shifts'), 1):
            raw = read(index, 160)
            fields[name] = raw.astype(np.uint16)
            del raw
        raw = read(5, 11)
        for name, offset, dtype in (('depths', 0, np.int16), ('maxdepths', 1, np.int16),
                                   ('floors', 2, np.int16), ('states', 3, np.uint8),
                                   ('rng', 4, np.uint32)):
            fields[name] = raw[offset].astype(dtype, copy=True)
        counters = raw[6::2].astype(np.uint64)
        counters <<= np.uint64(32)
        counters |= raw[5::2]
        fields['counters'] = counters
        del raw
        payload = {name: fields[name] for name in self.DEVICE_FIELDS}
        payload.update(dfs_version=np.int32(self.CHECKPOINT_VERSION), gold_fingerprint=np.array(self.gold_fingerprint),
                       fingerprint=np.array(self.fingerprint), order=self.order.copy(),
                       roots=self.roots.copy(), prefix_codes=self.prefix_codes.copy())
        if getattr(self, 'cpu_assist', None) is not None:
            import json
            payload['cpu_assist_state'] = np.array(json.dumps(self.cpu_assist.snapshot(), separators=(',', ':')))
        for value in payload.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return payload

    def _counter_totals(self):
        reducer = self._counter_reduction
        if reducer is None:
            # Publish only a fully initialized helper. Failed construction frees
            # its own buffers; failed submissions/maps never return old totals.
            reducer = _CounterReduction(self)
            self._counter_reduction = reducer
        return reducer.get()

    def has_pending(self):
        return bool(np.any(self.states.get() == 1))

    def close(self):
        """Release this engine's native allocations after its last checkpoint."""
        self._synchronize()
        try:
            reducer = getattr(self, '_counter_reduction', None)
            if reducer is not None:
                reducer.close()
        finally:
            for buffer in self.buffers: buffer.destroy()
            self.parameters.destroy()
            self.gpu_device.destroy()
