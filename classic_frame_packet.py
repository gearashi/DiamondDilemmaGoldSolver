"""Read four physical classic stack fields in one lossless 8-byte record per selected cell."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np


_DESCRIPTOR_BYTES = (2 + 4 * 160) * 4
_FIELDS = ('firsts', 'lengths', 'cursors', 'shifts')


def _blocks(windows, capacity):
    """Preserve window/row/column order when a device grant requires multiple reads."""
    if capacity < 1:
        raise ValueError('Device grant cannot hold a frame descriptor and one record.')
    block = []
    left = capacity
    for start, stop, columns in windows:
        width = len(columns)
        if not width or start >= stop:
            continue
        if width > capacity:
            if block:
                yield block
                block = []
            for row in range(start, stop):
                for j in range(0, width, capacity):
                    yield [(row, row + 1, columns[j:j + capacity])]
            left = capacity
            continue
        while start < stop:
            fits = left // width
            if not fits:
                yield block
                block = []
                left = capacity
                fits = left // width
            end = min(stop, start + fits)
            block.append((start, end, columns))
            left -= (end - start) * width
            start = end
    if block:
        yield block


def _successors(descriptor, n):
    """Physical first cell of the following nonempty row, wrapping at field boundaries."""
    table = descriptor[2:2+3*160].reshape(160,3)
    active = np.flatnonzero(table[:,1])
    active = active[np.argsort(table[active,0])]
    addresses = table[active,2]
    direct = (addresses & 0x80000000) != 0
    lane = addresses & 0x7fffffff
    lane[~direct] = descriptor[addresses[~direct]]
    first = (active * n + lane).astype(np.uint32)
    after = np.zeros(160,np.uint32)
    after[active] = np.roll(first,-1)
    return np.concatenate((descriptor,after))


class FramePacket:
    def __init__(self, gpu):
        self.gpu = gpu
        self.web = getattr(gpu, 'backend', None) == 'webgpu'
        self.pipeline = None
        self.maps = self.read_bytes = 0

    def _capacity(self):
        if self.web:
            # The requested device grant can be smaller than the adapter limit.
            limits = self.gpu.gpu_device.limits
            size = min(limits['max-storage-buffer-binding-size'], limits['max-buffer-size'])
            return min(size // 8, (size - _DESCRIPTOR_BYTES) // 4)
        return 160 * self.gpu.n

    def _read(self, descriptor, width, cells):
        gpu = self.gpu
        base = Path(__file__).resolve().parent
        descriptor[1] = cells
        if not self.web:
            if self.pipeline is None:
                self.pipeline = gpu.cp.RawKernel((base/'dfs_checkpoint_packet.cu').read_text(encoding='utf8'), 'gather_frame_packet')
            layout = gpu.cp.asarray(descriptor)
            out = gpu.cp.empty((4, cells), gpu.cp.uint16)
            self.pipeline(((width+63)//64,160), (64,), (*[getattr(gpu, name) for name in _FIELDS], layout, out))
            return out.get()
        descriptor = _successors(descriptor,gpu.n)
        device, usage = gpu.gpu_device, gpu.wgpu.BufferUsage
        if self.pipeline is None:
            shader = device.create_shader_module(code=(base/'dfs_checkpoint_packet.wgsl').read_text(encoding='utf8'))
            self.pipeline = device.create_compute_pipeline(layout='auto', compute={'module':shader,'entry_point':'gather_frame_packet'})
        with ExitStack() as buffers:
            layout = device.create_buffer(size=descriptor.nbytes, usage=usage.STORAGE | usage.COPY_DST)
            buffers.callback(layout.destroy)
            out = device.create_buffer(size=8*cells, usage=usage.STORAGE | usage.COPY_SRC)
            buffers.callback(out.destroy)
            gpu.queue.write_buffer(layout,0,descriptor)
            group = device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=[
                {'binding': i, 'resource': {'buffer': buf}}
                for i,buf in enumerate((*gpu.buffers[1:5],layout,out))])
            cmd = device.create_command_encoder()
            compute = cmd.begin_compute_pass()
            compute.set_pipeline(self.pipeline)
            compute.set_bind_group(0,group)
            compute.dispatch_workgroups((width+127)//128,160)
            compute.end()
            gpu.queue.submit([cmd.finish()])
            data = gpu.queue.read_buffer(out,0,8*cells)
            gpu.readback_bytes += 8*cells
            return np.frombuffer(data, dtype='<u2').reshape(4,cells).astype(np.uint16,copy=False)

    def capture(self, windows):
        from classic_checkpoint_ranges import _packed_frames
        self.maps = self.read_bytes = 0
        cells = sum((stop-start)*len(cols) for start, stop, cols in windows)
        blocks = list(_blocks(windows, self._capacity()))
        result = tuple(np.empty(cells,np.uint16) for _ in _FIELDS) if len(blocks)!=1 else None
        pos = 0
        for block in blocks:
            descriptor, width, count = _packed_frames(self.gpu.n, block)
            fields = self._read(descriptor, width, count)
            self.maps += 1
            self.read_bytes += count*8
            if result is None:
                return tuple(fields)
            for target,source in zip(result,fields):
                target[pos:pos+count] = source
            pos += count
        return result
