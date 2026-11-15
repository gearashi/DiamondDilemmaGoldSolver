"""Capture only classic DFS lanes and stack frames that can have changed."""
from pathlib import Path

import numpy as np

FRAME_FIELDS = ('firsts', 'lengths', 'cursors', 'shifts')


def _selection(lanes):
    if len(lanes) and int(lanes[-1]) - int(lanes[0]) + 1 == len(lanes):
        return slice(int(lanes[0]), int(lanes[-1]) + 1)
    return lanes


def _frame_windows(lanes, floors, maximum, installed):
    """Priced rectangles around all writable frames, including full lane installs."""
    low = floors[lanes].astype(int)
    high = np.minimum(160, maximum[lanes].astype(int) + 1)
    resets = installed[lanes]
    low[resets], high[resets] = 0, 160
    valid = low < high
    if not valid.any():
        return []
    lanes, low, high = lanes[valid], low[valid], high[valid]
    first, last = int(low.min()), int(high.max())
    if (last-first)*len(lanes) <= 4096 or (np.all(low==first) and np.all(high==last)):
        return [(first,last,lanes)]

    # For a strip [a,b), exactly the lanes with floor < b and top > a
    # can change a frame. A histogram prices every possible strip without
    # scanning the lane array for every pair of boundaries.
    boundaries, inverse = np.unique(np.r_[low,high],return_inverse=True)
    count = len(boundaries)
    left,right = np.split(inverse,2)
    hist = np.bincount(left*count+right,minlength=count*count).reshape(count,count)
    summed = hist.cumsum(axis=0).cumsum(axis=1)
    cost = np.zeros(count,np.int64)
    parent = np.zeros(count,np.int16)
    for end in range(1,count):
        width = summed[end-1,-1]-summed[end-1,:end]
        price = cost[:end]+(boundaries[end]-boundaries[:end])*width+(width>0)*4096
        parent[end] = np.argmin(price)
        cost[end] = price[parent[end]]
    # Reconstruct row-disjoint transfers. No column scatter is needed for
    # the often dense shared bottom of a large pool.
    strips = []
    end = count-1
    while end:
        begin = int(parent[end])
        a,b = int(boundaries[begin]),int(boundaries[end])
        ids = lanes[(low<b)&(high>a)]
        if len(ids):
            strips.append((a,b,ids))
        end = begin
    return strips[::-1]


def _packed_frames(n, windows):
    """Describe each strip once, including a zero-upload encoding for consecutive lanes."""
    table = np.zeros((160,3),np.uint32)
    header = 2+table.size
    parts = []
    width = offset = 0
    for start,stop,cols in windows:
        size = len(cols)
        width = max(width,size)
        if int(cols[-1])-int(cols[0])+1 == size:
            address = 0x80000000 | int(cols[0])
        else:
            address = header
            header += size
            parts.append(cols.astype(np.uint32,copy=False))
        table[start:stop,0] = offset + np.arange(stop-start,dtype=np.uint32)*size
        table[start:stop,1] = size
        table[start:stop,2] = address
        offset += (stop-start)*size
    return np.concatenate((np.array([n,width],np.uint32),table.ravel(),*parts)),width,offset


class CheckpointRanges:
    """One search-thread reader for the lifetime of one native engine and ledger."""
    def __init__(self, gpu, ledger):
        self.gpu = gpu
        self.ledger = ledger
        self.prior = None
        self.frame_kernel = None
        self.enabled = callable(getattr(type(gpu), '_checkpoint_payload', None))
        self.web = getattr(gpu, 'backend', None) == 'webgpu'
        self.read_bytes = 0
        self.maps = 0
        if self.enabled:
            ledger.checkpoint_installs = np.zeros(gpu.n, bool)

    def _read(self, name, lanes, start=None, stop=None):
        arr = getattr(self.gpu, name)
        cols = _selection(lanes)
        result = (arr[cols] if arr.ndim == 1 else arr[slice(start,stop),cols]).get()
        self.maps += 1
        self.read_bytes += result.nbytes
        return result

    def _physical(self, index, start, stop, lanes):
        gpu = self.gpu
        before = gpu.readback_bytes
        result = gpu._read_rows(gpu.buffers[index],np.arange(start,stop),lanes)
        self.maps += 1
        self.read_bytes += gpu.readback_bytes - before
        return result

    def _frames(self, windows):
        gpu = self.gpu
        params,width,total = _packed_frames(gpu.n,windows)
        if self.web:
            packed = gpu._columns().frames(params,width,total)
        else:
            cp = gpu.cp
            if self.frame_kernel is None:
                source = (Path(__file__).resolve().parent / 'dfs_checkpoint_frames.cu').read_text(encoding='utf8')
                self.frame_kernel = cp.RawKernel(source,'gather_frames')
            description = cp.asarray(params)
            target = cp.empty(total,cp.uint16)
            packed = []
            for name in FRAME_FIELDS:
                self.frame_kernel(((width+63)//64,160),(64,),(getattr(gpu,name),description,target))
                packed.append(target.get())
        self.maps += len(FRAME_FIELDS)
        self.read_bytes += total*(4 if self.web else 2)*len(FRAME_FIELDS)
        return packed

    def capture(self):
        gpu = self.gpu
        self.maps = self.read_bytes = 0
        if not self.enabled:
            return gpu.snapshot_checkpoint()
        installed = self.ledger.checkpoint_installs
        if self.prior is None or np.all(installed):
            before = getattr(gpu, 'readback_bytes', 0)
            payload = gpu.snapshot_checkpoint()
            self.maps = 6 if self.web else len(gpu.DEVICE_FIELDS)
            self.read_bytes = (gpu.readback_bytes-before if self.web else
                               sum(payload[name].nbytes for name in gpu.DEVICE_FIELDS))
        else:
            # Exhausted and unassigned lanes cannot run. Include pending boards:
            # a Gold rewind or acknowledgement can change them before the next save.
            lanes = np.flatnonzero((self.prior['states'] <= 1) | installed)
            fields = self.prior.copy()
            def put(name, values, ids, start=0):
                old = self.prior[name]
                if values.shape == old.shape:
                    fields[name] = values
                    return
                if fields[name] is old:
                    if old.ndim == 2 and len(ids) == old.shape[1]:
                        # This first full-width strip immediately replaces its rows.
                        value = np.empty_like(old)
                        value[:start] = old[:start]
                        value[start+len(values):] = old[start+len(values):]
                        fields[name] = value
                    else:
                        fields[name] = old.copy()
                cols = _selection(ids)
                if old.ndim == 1:
                    fields[name][cols] = values
                else:
                    target = fields[name][start:start+len(values)]
                    if not isinstance(cols,slice) and len(ids)>=1024:
                        # Keep large irregular writes in row-major memory order.
                        # NumPy's two-axis advanced indexing walks whole columns.
                        for row,part in zip(target,values):
                            row[ids] = part
                    else:
                        target[:,cols] = values

            if len(lanes):
                if self.web:
                    raw = self._physical(5,0,11,lanes)
                    for i,name in enumerate(('depths','maxdepths','floors','states','rng')):
                        put(name,raw[i].astype(self.prior[name].dtype),lanes)
                    counts = raw[6::2].astype(np.uint64) << np.uint64(32)
                    counts |= raw[5::2]
                    put('counters',counts,lanes)
                    raw = self._physical(0,0,165,lanes)
                    put('boards',raw[:160].astype(np.int16),lanes)
                    put('used',raw[160:].copy(),lanes)
                else:
                    for name in ('boards','used','depths','maxdepths','states','rng','counters'):
                        put(name,self._read(name,lanes),lanes)
                    resets = np.flatnonzero(installed)
                    if len(resets):
                        put('floors',self._read('floors',resets),resets)
                # Normal transitions cannot write a frame below the immutable
                # prefix or beyond the highest attained depth and its next cursor.
                # Replacing a lane can overwrite all 160 rows, even if it retires
                # again before this capture. Preserve arbitrary old bytes elsewhere.
                windows = _frame_windows(lanes,fields['floors'],fields['maxdepths'],installed)
                packed = (self._frames(windows) if len(windows)>1 and
                          (self.web or hasattr(gpu.cp,'RawKernel')) else None)
                offset = 0
                for a,b,ids in windows:
                    size = (b-a)*len(ids)
                    for index,name in enumerate(FRAME_FIELDS,1):
                        if packed is not None:
                            values = packed[index-1][offset:offset+size].reshape(b-a,len(ids))
                        elif self.web:
                            values = self._physical(index,a,b,ids).astype(np.uint16)
                        else:
                            values = self._read(name,ids,a,b)
                        put(name,values,ids,a)
                    offset += size
            payload = gpu._checkpoint_payload(fields, fixed=None if np.any(installed) else self.prior)
        self.prior = {k:payload[k] for k in (*gpu.DEVICE_FIELDS,'roots','order','prefix_codes')}
        installed.fill(False)
        return payload
