"""Capture only classic DFS lanes and stack frames that can have changed."""
from r35_board_codec import PhysicalBoardCodec
from classic_checkpoint_delta import BoardDelta
from weakref import WeakSet
from classic_frame_packet import FramePacket
from classic_frame_defaults import FrameDefaults
from r35_frame_spans import FrameSpans, rectangle_wire_bytes

import numpy as np

FRAME_FIELDS = ('firsts', 'lengths', 'cursors', 'shifts')


def _selection(lanes):
    if len(lanes) and int(lanes[-1]) - int(lanes[0]) + 1 == len(lanes):
        return slice(int(lanes[0]), int(lanes[-1]) + 1)
    return lanes


def _attained_bounds(prior, current, lanes, installed, native_only):
    """Bound touched ranks using actual placements since the last frozen snapshot."""
    floors, high = current['floors'], current['maxdepths']
    if not native_only:
        # A CPU endgame may install a whole solved suffix in a single operation.
        return lanes,floors,high
    # A native placement is the only operation which increases depth. Every
    # touched rank can be reached from the start, and must be able to reach the
    # final depth, using at most this many placements in either segment.
    added = np.minimum(current['counters'][1]-prior['counters'][1],160).astype(np.int16)
    low = np.maximum(floors,current['depths']-added)
    top = np.minimum(high,prior['depths']+added)
    # Initializing a frame consumes a candidate or a backtrack in that same
    # iteration. A newly installed job resets all rows independently of counters.
    changed = np.any(current['counters'][:,lanes]!=prior['counters'][:,lanes],axis=0)
    return lanes[changed|installed[lanes]],low,top


def _frame_windows(lanes, floors, maximum, installed):
    """Priced rectangles around all writable frames, including full lane installs."""
    if not len(lanes):
        return []
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
    def __init__(self, gpu, ledger, *, board_mode=None, trace=None):
        self.gpu = gpu
        self.ledger = ledger
        self.prior = None
        self.frame_kernel = None
        self.frame_defaults = None
        self.defaults_disabled = False
        self.board_mode = board_mode
        self.auto_trace = trace is None
        self.trace = True if self.auto_trace else trace
        self.trace_owner = object()
        self.trace_rows = None
        self.trace_scope = None
        self.span_reader = None
        self.board_projection = None
        self.board_fallbacks = 0
        self.delta_disabled = False
        self.physical_disabled = False
        self.board_delta = None
        self.enabled = callable(getattr(type(gpu), '_checkpoint_payload', None))
        self.web = getattr(gpu, 'backend', None) == 'webgpu'
        self.read_bytes = 0
        self.maps = 0
        if self.enabled:
            self.installed = np.zeros(gpu.n, bool)
            ledger.checkpoint_installs = self.installed
            if callable(getattr(type(gpu),'_mark_checkpoint_installs',None)):
                if not hasattr(gpu,'_checkpoint_readers'):
                    gpu._checkpoint_readers = WeakSet()
                gpu._checkpoint_readers.add(self)

    def _want_delta(self):
        native = self.web or hasattr(self.gpu.cp,'RawKernel')
        return native and self.board_mode not in ('physical','ranges')

    def _disable_delta(self):
        self.delta_disabled = True
        self.board_fallbacks += 1
        if self.board_delta is not None:
            self.board_delta.close()
            self.board_delta = None

    def _arm_trace(self, first):
        gpu = self.gpu
        if not self.trace or getattr(gpu,'cpu_assist',None) is not None:
            return
        if not first and getattr(gpu,'_frame_mark_owner',None) is not self.trace_owner:
            return
        try:
            if self.web:
                encoder = gpu.gpu_device.create_command_encoder()
                encoder.clear_buffer(gpu.buffers[5],12*gpu.n*4,gpu.n*4)
                gpu.queue.submit([encoder.finish()])
            elif hasattr(gpu.cp,'RawKernel'):
                if not hasattr(gpu,'_frame_marks'):
                    gpu._frame_marks = gpu.cp.zeros(gpu.n,gpu.cp.uint32)
                else:
                    gpu._frame_marks.fill(0)
            else:
                return
            gpu._frame_mark_owner = self.trace_owner
        except Exception:
            gpu._frame_mark_owner = None
            self.trace = False

    def _trace_bounds(self, lanes, low, high, installed):
        gpu = self.gpu
        if (not len(lanes) or getattr(gpu,'cpu_assist',None) is not None
                or getattr(gpu,'_frame_mark_owner',None) is not self.trace_owner):
            return lanes,low,high
        try:
            marks = self._physical(5,12,13,lanes)[0] if self.web else self._read('_frame_marks',lanes)
        except Exception:
            gpu._frame_mark_owner = None
            self.trace = False
            return lanes,low,high
        start=((marks>>16)&255).astype(np.int16)-1
        stop=(marks>>24).astype(np.int16)-1
        init_start=(marks&255).astype(np.int16)-1
        init_stop=((marks>>8)&255).astype(np.int16)-1
        usable=(~installed[lanes])&(start>=0)&(stop>=start)&(stop<160)
        init_good=((marks&65535)==0)|((init_start>=start)&(init_stop>=init_start)&(init_stop<=stop))
        if np.any((marks!=0)&~((usable&init_good)|installed[lanes])):
            return lanes,low,high
        self.trace_rows=(int(np.maximum(0,stop-start+1)[usable].sum()),
                         int(np.where(init_start>=0,init_stop-init_start+1,0)[usable].sum()))
        self.trace_scope = lanes,marks
        ids=lanes[usable]
        low[ids]=np.maximum(low[ids],start[usable])
        high[ids]=np.minimum(high[ids],stop[usable])
        return lanes[(marks!=0)|installed[lanes]],low,high

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
        if self.frame_kernel is None:
            self.frame_kernel = FramePacket(self.gpu)
        result = self.frame_kernel.capture(windows)
        self.maps += self.frame_kernel.maps
        self.read_bytes += self.frame_kernel.read_bytes
        return result

    def _full_frames(self,maxdepths=None,floors=None):
        gpu = self.gpu
        if not (self.web or hasattr(gpu.cp,'RawKernel')):
            return None
        if not self.defaults_disabled:
            try:
                if self.frame_defaults is None:
                    self.frame_defaults = FrameDefaults(gpu)
                prefer_direct = getattr(self.frame_defaults,'prefer_direct',None)
                result = None if prefer_direct and prefer_direct(maxdepths,floors) else self.frame_defaults.capture()
            except Exception:
                self.defaults_disabled = True
                result = None
            if self.frame_defaults is not None:
                self.maps += self.frame_defaults.maps
                self.read_bytes += self.frame_defaults.read_bytes
            if result is not None:
                return result
        if self.web:
            try:
                return self._frames([(0,160,np.arange(gpu.n,dtype=np.int32))])
            except Exception:
                pass
        return None

    def _physical_board(self, fields, lanes):
        if (self.physical_disabled or self.board_mode=='ranges'
                or not (self.web or self.board_mode=='physical')):
            return None
        try:
            if self.board_projection is None:
                self.board_projection = PhysicalBoardCodec(self.gpu)
            result = self.board_projection.capture(fields,lanes)
        except Exception:
            self.physical_disabled = True
            result = None
        if self.board_projection is not None:
            self.maps += self.board_projection.maps
            self.read_bytes += self.board_projection.read_bytes
        if result is None:
            self.board_fallbacks += 1
        return result

    def capture(self):
        gpu = self.gpu
        self.maps = self.read_bytes = 0
        self.trace_scope = self.trace_rows = None
        if not self.enabled:
            return gpu.snapshot_checkpoint()
        installed = self.installed
        if self.prior is None or np.all(installed):
            before = getattr(gpu, 'readback_bytes', 0)
            if hasattr(gpu,'_snapshot_fields'):
                payload = gpu._snapshot_fields(self)
            else:
                payload = gpu.snapshot_checkpoint()
                self.maps = 6 if self.web else len(gpu.DEVICE_FIELDS)
                self.read_bytes = sum(payload[name].nbytes for name in gpu.DEVICE_FIELDS)
            if self._want_delta() and not self.delta_disabled:
                try:
                    if self.board_delta is None:
                        self.board_delta = BoardDelta(gpu, native=self.board_mode!='device')
                    self.board_delta.seed(payload)
                except Exception:
                    # Optional GPU workspace may exceed the available memory.
                    self._disable_delta()
            if self.web:
                self.read_bytes = gpu.readback_bytes-before
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
                else:
                    for name in ('depths','maxdepths','states','rng','counters'):
                        put(name,self._read(name,lanes),lanes)
                    resets = np.flatnonzero(installed)
                    if len(resets):
                        put('floors',self._read('floors',resets),resets)
                # Normal transitions cannot write a frame below the immutable
                # prefix or beyond the highest attained depth and its next cursor.
                # Replacing a lane can overwrite all 160 rows, even if it retires
                # again before this capture. Preserve arbitrary old bytes elsewhere.
                frame_lanes,low,high = _attained_bounds(self.prior,fields,lanes,installed,
                                                        getattr(gpu,'cpu_assist',None) is None)
                # Large pools sometimes make only a handful of candidate tries
                # per lane before Stop or a scheduled save. In these intervals
                # fetching a separate word per lane can cost more than the
                # frame rows it eliminates. Existing counters are already local.
                use_marks = True
                if self.auto_trace and len(frame_lanes) >= 32768:
                    attempts = fields['counters'][0,frame_lanes] - self.prior['counters'][0,frame_lanes]
                    use_marks = not (0 < attempts.sum(dtype=np.uint64) <= 8 * len(frame_lanes))
                if use_marks:
                    frame_lanes,low,high = self._trace_bounds(frame_lanes,low,high,installed)
                windows = _frame_windows(frame_lanes,low,high,installed)
                if self.trace_scope is not None and windows:
                    traced,marks = self.trace_scope
                    if self.span_reader is None:
                        self.span_reader = FrameSpans(gpu)
                    if self.span_reader.native:
                        try:
                            spans = self.span_reader.capture(tuple(self.prior[k] for k in FRAME_FIELDS),
                                traced,marks,installed[traced],max_wire_bytes=rectangle_wire_bytes(gpu,windows))
                        except Exception:
                            spans = None
                        self.maps += self.span_reader.maps
                        self.read_bytes += self.span_reader.read_bytes
                        if spans is not None:
                            fields.update(zip(FRAME_FIELDS,spans))
                            windows = []
                packed = None
                cells = sum((b-a)*len(ids) for a,b,ids in windows)
                if windows and (self.web or hasattr(gpu.cp,'RawKernel')) and (len(windows)>1 or cells>=1024):
                    try:
                        packed = self._frames(windows)
                    except Exception:
                        # Read the same individual fields if extra GPU workspace is unavailable.
                        packed = None
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
                projected = None
                if self.board_delta is not None:
                    codec = self.board_delta
                    try:
                        projected = codec.capture(lanes)
                    except Exception:
                        # An interrupted dispatch may already have advanced its device mirror.
                        self._disable_delta()
                    self.maps += codec.maps
                    self.read_bytes += codec.read_bytes
                if projected is None:
                    projected = self._physical_board(fields,lanes)
                if projected is not None:
                    put('boards',projected[0],lanes)
                    put('used',projected[1],lanes)
                elif self.web:
                    raw = self._physical(0,0,165,lanes)
                    put('boards',raw[:160].astype(np.int16),lanes)
                    put('used',raw[160:].copy(),lanes)
                else:
                    for name in ('boards','used'):
                        put(name,self._read(name,lanes),lanes)
            payload = gpu._checkpoint_payload(fields, fixed=None if np.any(installed) else self.prior)
        self._arm_trace(first=self.prior is None)
        self.prior = {k:payload[k] for k in (*gpu.DEVICE_FIELDS,'roots','order','prefix_codes')}
        installed.fill(False)
        return payload
