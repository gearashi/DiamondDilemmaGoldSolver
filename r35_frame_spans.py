"""Transport checkpoint writes from validated per-lane initialization and cursor spans."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np


FIELDS = ('firsts', 'lengths', 'cursors', 'shifts')
# Each half contains two one-based inclusive endpoints. All four fields are
# overwritten by host installation, including otherwise unused stack storage.
FULL = np.uint32(1 | (160 << 8) | (1 << 16) | (160 << 24))


def extents(marks):
    if marks.dtype != np.uint32 or marks.ndim != 1:
        raise ValueError('Expected one unsigned 32-bit mark per selected lane.')
    a, b, c, d = ((marks >> shift & 255).astype(np.int16) for shift in (0, 8, 16, 24))
    if np.any(((a == 0) != (b == 0)) | ((c == 0) != (d == 0)) |
              (b < a) | (d < c) | (d > 160) | ((a != 0) & ((c == 0) | (a < c) | (b > d)))):
        raise ValueError('Invalid native frame write span.')
    return np.maximum(a - 1, 0), b, np.maximum(c - 1, 0), d


def _descriptor(n, ids, marks, widths):
    """Constant and contiguous columns need no per-column descriptor uploads."""
    k = len(ids)
    parts = []
    header = [n, k, 0, 0, 0, 0]
    at = len(header)
    def variable(field, data):
        nonlocal at
        header[5] |= 1 << (field - 2)
        header[field] = at
        parts.append(data)
        at += len(data)
    if int(ids[-1]) - int(ids[0]) + 1 == k:
        header[2] = int(ids[0])
    else:
        variable(2, ids)
    if np.all(marks == marks[0]):
        header[3] = int(marks[0])
    else:
        variable(3, marks)
    if np.all(widths == widths[0]):
        header[4] = int(widths[0])
    else:
        variable(4, np.cumsum(widths, dtype=np.uint32) - widths)
    usual = np.concatenate((np.array(header, np.uint32), *parts))
    # Dense spans with a few retired columns need no per-column IDs or offsets.
    # Native groups compute 32 local word offsets from the uploaded marks.
    count = int(ids[-1]) - int(ids[0]) + 1
    groups = (count + 31) // 32
    if 6 + groups + count >= len(usual) or count > 2 * k:
        return usual
    compact = np.zeros(6 + groups + count, np.uint32)
    compact[:6] = (n, count, ids[0], 6 + groups, 6, 8)
    pos = ids - ids[0]
    compact[6 + groups + pos] = marks
    totals = np.bincount(pos // 32, weights=widths, minlength=groups).astype(np.uint32)
    compact[6:6 + groups] = np.cumsum(totals, dtype=np.uint32) - totals
    if count < 128:
        return compact
    # Nearby lanes commonly share exact endpoints. An indexed palette avoids
    # re-uploading a full 32-bit mark for every such lane.
    palette, codes = np.unique(compact[6 + groups:], return_inverse=True)
    bits = next(b for b in (1, 2, 4, 8, 16, 32) if len(palette) <= 1 << b)
    per_word = 32 // bits
    packed_words = (count + per_word - 1) // per_word
    if 8 + groups + len(palette) + packed_words >= len(compact):
        return compact
    indexed = np.zeros(8 + groups + len(palette) + packed_words, np.uint32)
    indexed[:8] = (n, count, ids[0], 8 + groups + len(palette), 8, 24, bits, 8 + groups)
    indexed[8:8 + groups] = compact[6:6 + groups]
    indexed[8 + groups:8 + groups + len(palette)] = palette
    padded = np.zeros(packed_words * per_word, np.uint32)
    padded[:count] = codes
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    indexed[-packed_words:] = np.bitwise_or.reduce(padded.reshape(-1, per_word) << shifts, axis=1)
    return indexed


def prepare(n, lanes, raw, resets):
    """Caller must establish ownership and native-only execution before passing marks."""
    lanes = np.asarray(lanes)
    raw = np.asarray(raw)
    resets = np.asarray(resets)
    if (not 0 < n <= 131072 or lanes.ndim != 1 or lanes.dtype.kind not in 'iu' or
            np.any(lanes < 0) or np.any(lanes >= n) or np.any(lanes[1:] <= lanes[:-1]) or
            raw.shape != lanes.shape or resets.shape != lanes.shape or resets.dtype != bool):
        raise ValueError('Invalid frame capture lanes.')
    if raw.dtype != np.uint32:
        raise ValueError('Invalid frame capture mark type.')
    marks = np.where(resets, FULL, raw)
    a, b, c, d = extents(marks)
    active = d != 0
    widths = ((3 * (b - a) + (d - c) + 1) // 2).astype(np.uint32)
    return lanes[active].astype(np.uint32), marks[active], widths[active]


def blocks(n, ids, marks, widths, limit):
    limit //= 4
    # A block independently fits the requested grant for its plan and output.
    ends = np.cumsum(widths, dtype=np.int64)
    first = 0
    while first < len(ids):
        base = int(ends[first - 1]) if first else 0
        stop = int(np.searchsorted(ends, base + limit, side='right'))
        if first == stop or limit < 6:
            raise ValueError('Frame grant cannot hold one lane.')
        while True:
            sl = slice(first, stop)
            desc = _descriptor(n, ids[sl], marks[sl], widths[sl])
            if len(desc) <= limit:
                break
            stop = first + (stop - first) // 2
            if stop == first:
                raise ValueError('Frame grant cannot hold one lane descriptor.')
        yield sl, desc, int(ends[stop - 1]) - base
        first = stop


def rectangle_wire_bytes(gpu, windows):
    """Cost of the competing range reader's physical data and lane descriptors."""
    from classic_frame_packet import FramePacket, _blocks
    if not windows:
        return 0
    n = gpu.n
    web = getattr(gpu, 'backend', None) == 'webgpu'
    cells = sum((b - a) * len(cols) for a, b, cols in windows)
    if len(windows) == 1 and cells < 1024:
        a, b, cols = windows[0]
        width = int(cols[-1]) - int(cols[0]) + 1
        if not web:
            return 8 * cells + (4 * cols.nbytes if width != len(cols) else 0)
        from dfs_webgpu import PACKED_READ_MIN_ROWS
        if (width > 2 * len(cols) and width * (b-a) >= 2048) or (width < n and b-a >= PACKED_READ_MIN_ROWS):
            return 16 * cells + 4 * (20 + (4 * len(cols) if width != len(cols) else 0))
        return 16 * width * (b - a)
    size = 8 * cells
    for block in _blocks(windows, FramePacket(gpu)._capacity()):
        size += 4 * (2 + 480 + (160 if web else 0))
        size += 4 * sum(len(cols) for a, b, cols in block if int(cols[-1])-int(cols[0])+1 != len(cols))
    return size


def merge_numpy(fields, ids, marks, widths, packet):
    a, b, c, d = extents(marks)
    if packet.dtype != np.uint32 or packet.shape != (int(widths.sum(dtype=np.uint64)),):
        raise ValueError('Truncated physical frame packet.')
    start = (np.cumsum(widths, dtype=np.int64) - widths) * 2
    pitch = (b - a).astype(np.int64)
    raw = packet.astype('<u4', copy=False).view('<u2')
    for row in range(int(c.min()), int(d.max())):
        init = np.flatnonzero((a <= row) & (b > row))
        if len(init):
            pos = start[init] + row - a[init]
            lanes = ids[init]
            fields[0][row, lanes] = raw[pos]
            fields[1][row, lanes] = raw[pos + pitch[init]]
            fields[3][row, lanes] = raw[pos + 2 * pitch[init]]
        cursor = np.flatnonzero((c <= row) & (d > row))
        fields[2][row, ids[cursor]] = raw[start[cursor] + 3 * pitch[cursor] + row - c[cursor]]


class FrameSpans:
    def __init__(self, gpu, *, merge=None, native_directory=None):
        self.gpu = gpu
        self.web = getattr(gpu, 'backend', None) == 'webgpu'
        self.pipelines = {}
        if merge is None:
            from r35_board_codec import verified_library
            from r35_frame_span_decode import native_merge
            library = verified_library(native_directory)
            try:
                merge = native_merge(library) if library is not None else merge_numpy
            except (AttributeError, TypeError):
                merge = merge_numpy
        self.merge = merge
        self.native = merge is not merge_numpy
        self.read_bytes = self.upload_bytes = self.maps = 0

    def _read(self, desc, words):
        gpu = self.gpu
        root = Path(__file__).resolve().parent
        compact = bool(desc[5] & 8)
        threads = 32 if compact else 64
        entry = 'gather_dense_spans' if compact else 'gather_spans'
        groups = (int(desc[1]) + threads - 1) // threads
        if not self.web:
            if entry not in self.pipelines:
                self.pipelines[entry] = gpu.cp.RawKernel((root/'r35_frame_spans.cu').read_text('utf8'), entry)
            plan = gpu.cp.asarray(desc)
            out = gpu.cp.empty(words, gpu.cp.uint32)
            self.pipelines[entry]((groups,), (threads,), (*[getattr(gpu, key) for key in FIELDS], plan, out))
            return out.get()
        dev, usage = gpu.gpu_device, gpu.wgpu.BufferUsage
        if entry not in self.pipelines:
            shader = dev.create_shader_module(code=(root/'r35_frame_spans.wgsl').read_text('utf8'))
            self.pipelines[entry] = dev.create_compute_pipeline(layout='auto', compute={'module': shader, 'entry_point': entry})
        with ExitStack() as resources:
            plan = dev.create_buffer(size=desc.nbytes, usage=usage.STORAGE | usage.COPY_DST)
            resources.callback(plan.destroy)
            out = dev.create_buffer(size=words * 4, usage=usage.STORAGE | usage.COPY_SRC)
            resources.callback(out.destroy)
            gpu.queue.write_buffer(plan, 0, desc)
            group = dev.create_bind_group(layout=self.pipelines[entry].get_bind_group_layout(0), entries=[
                {'binding': i, 'resource': {'buffer': buf}} for i, buf in enumerate((*gpu.buffers[1:5], plan, out))])
            commands = dev.create_command_encoder()
            work = commands.begin_compute_pass()
            work.set_pipeline(self.pipelines[entry])
            work.set_bind_group(0, group)
            work.dispatch_workgroups(groups)
            work.end()
            gpu.queue.submit([commands.finish()])
            data = gpu.queue.read_buffer(out, 0, words * 4)
            gpu.readback_bytes += words * 4
            return np.frombuffer(data, dtype='<u4').astype(np.uint32, copy=False)

    def capture(self, old, lanes, raw, resets, *, max_wire_bytes=None, limit=None):
        """Return None without device work when a competing transport costs no more."""
        self.read_bytes = self.upload_bytes = self.maps = 0
        n = self.gpu.n
        ids, marks, widths = prepare(n, lanes, raw, resets)
        if not len(ids):
            return old
        grant = 2**31
        if self.web:
            limits = self.gpu.gpu_device.limits
            grant = min(limits['max-storage-buffer-binding-size'], limits['max-buffer-size'])
        if limit is not None:
            grant = min(grant,limit)
        work = list(blocks(n, ids, marks, widths, grant))
        if max_wire_bytes is not None and sum(desc.nbytes + 4 * words for _, desc, words in work) >= max_wire_bytes:
            return None
        if len(old) != 4 or any(array.shape != (160, n) or array.dtype != np.uint16 for array in old):
            raise ValueError('Invalid previous native frames.')
        only_cursors = not np.any(marks & 65535)
        fields = tuple(array.copy() if not only_cursors or key == 'cursors' else array
                       for key, array in zip(FIELDS, old))
        for sl, desc, words in work:
            data = self._read(desc, words)
            self.read_bytes += data.nbytes
            self.upload_bytes += desc.nbytes
            self.maps += 1
            self.merge(fields, ids[sl], marks[sl], widths[sl], data)
        return fields
