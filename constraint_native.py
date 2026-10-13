"""Optional compiled seam propagation, with explicit and honest Python fallback.

The C library processes an exact bounded arc queue with reusable storage. Changed
rows or ordered reason events are committed before polling. Cancellation retains
only sound work and invalidates the synchronization shadow for the next call.
Only the local source-matching build manifest is loaded; importing never builds.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
POLL_CALLBACK=C.CFUNCTYPE(C.c_int)
STAT_KEYS=('native_calls','native_arcs_checked','native_arc_revisions','python_arcs_checked','native_support_hits','native_support_scans','native_input_rows','native_output_rows','native_reason_events')

def load_library(directory=None):
    directory=Path(directory or ROOT/'.native')
    metadata=json.loads((directory/'build.json').read_text(encoding='utf-8'))
    if metadata.get('abi')!=1 or metadata.get('platform')!=sys.platform:
        raise RuntimeError('Native propagation ABI/platform differs from this installation.')
    name=metadata.get('library')
    if not isinstance(name,str) or Path(name).name!=name:
        raise RuntimeError('Native propagation build manifest is malformed.')
    source=ROOT/'constraint_native.c';library=directory/name
    if metadata.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError('Native propagation source changed; rebuild the local library.')
    if metadata.get('library_sha256')!=hashlib.sha256(library.read_bytes()).hexdigest():
        raise RuntimeError('Native propagation library differs from its local build manifest.')
    lib=C.CDLL(str(library));lib.dd_native_abi.restype=C.c_int
    if lib.dd_native_abi()!=1:raise RuntimeError('Native propagation library ABI differs.')
    u64=C.POINTER(C.c_uint64);i32=C.POINTER(C.c_int32);u8=C.POINTER(C.c_uint8)
    lib.dd_arc_batch.argtypes=[u64,u64,i32,i32,C.c_int,C.c_int,C.c_int,C.c_int,i32,u8,i32,C.c_int,u64]
    lib.dd_arc_batch.restype=C.c_int
    if hasattr(lib,'dd_arc_batch_cached'):
        lib.dd_arc_batch_cached.argtypes=[*lib.dd_arc_batch.argtypes,C.POINTER(C.c_int16)]
        lib.dd_arc_batch_cached.restype=C.c_int
    if hasattr(lib,'dd_arc_batch_workspace'):
        lib.dd_arc_batch_workspace.argtypes=[*lib.dd_arc_batch.argtypes,
            C.POINTER(C.c_int16),u8,i32,u64,C.c_int,i32]
        lib.dd_arc_batch_workspace.restype=C.c_int
    if hasattr(lib,'dd_cyclic_blocks'):
        lib.dd_cyclic_blocks.argtypes=[C.c_int,C.c_int,i32,i32,POLL_CALLBACK]
        lib.dd_cyclic_blocks.restype=C.c_int
    if hasattr(lib,'dd_hall_allowed'):
        lib.dd_hall_allowed.argtypes=[C.c_int,C.c_int,u64,u64,i32,i32,POLL_CALLBACK]
        lib.dd_hall_allowed.restype=C.c_int
    return lib


def build_signature(directory=None):
    directory=Path(directory or ROOT/'.native')
    metadata=json.loads((directory/'build.json').read_text(encoding='utf-8'))
    source=hashlib.sha256((ROOT/'constraint_native.c').read_bytes()).hexdigest()
    if metadata.get('source_sha256')!=source:raise RuntimeError('Native source changed; rebuild locally.')
    return (source,metadata.get('library_sha256'),metadata.get('library'),metadata.get('abi'),metadata.get('platform'))


class NativePropagation:
    def __init__(self,tables,library=None):
        if sys.byteorder!='little':raise ValueError('Native propagation currently requires a little-endian host.')
        self.lib=library if library is not None else load_library()
        self.tables=tables
        self.signature=build_signature() if library is None else None
        self.n=len(tables.tile_bits);self.count=3*self.n;self.words=(self.count+63)//64
        if not 1<=self.n<=160:raise ValueError('Native propagation supports 1..160 cells.')
        self.width=self.words*8;self.arc_count=len(tables.arcs)
        if self.arc_count>3*self.n:raise ValueError('Native propagation requires triangular side incidence.')
        raw=b''.join(value.to_bytes(self.width,'little') for row in tables.compatible for side in row for value in side)
        self.support=(C.c_uint64*(len(raw)//8)).from_buffer_copy(raw)
        self.arcs=(C.c_int32*(4*self.arc_count))(*(v for arc in tables.arcs for v in arc))
        indices=[]
        for row in tables.incoming:
            if len(row)>3:raise ValueError('Native propagation requires triangular side incidence.')
            indices.extend([*row,*([-1]*(3-len(row)))])
        self.incoming=(C.c_int32*len(indices))(*indices)
        self.residues=(C.c_int16*(self.arc_count*self.count)).from_buffer_copy(b'\xff\xff'*(self.arc_count*self.count))
        self.use_residues=hasattr(self.lib,'dd_arc_batch_cached')
        self.use_workspace=hasattr(self.lib,'dd_arc_batch_workspace')
        self.buffer=(C.c_uint64*(self.n*self.words))()
        self.capacity=max(1,self.arc_count)
        self.queue=(C.c_int32*self.capacity)()
        self.pending=(C.c_uint8*self.capacity)()
        self.state=(C.c_int32*3)()
        self.counts=(C.c_uint64*4)()
        self.dirty=(C.c_uint8*self.n)()
        self._shadow=[None]*self.n
        self._buffer_address=C.addressof(self.buffer)
        self._event_capacity=0
        self._event_arcs=self._event_domains=None
        self._event_count=C.c_int32(0)
        self._active=False

    def matches(self,tables):
        if self.tables is not tables:return False
        try:return self.signature is not None and self.signature==build_signature()
        except (OSError,ValueError,RuntimeError,KeyError,TypeError):return False

    def _sync(self,domains,stats):
        changed=[cell for cell,value in enumerate(domains) if self._shadow[cell]!=value]
        # Dense restorations use one packed copy; sibling/unchanged calls copy
        # only changed rows. Equality is exact, so widening never reuses stale bits.
        if len(changed)>self.n//2:
            raw=b''.join(value.to_bytes(self.width,'little') for value in domains)
            C.memmove(self._buffer_address,raw,len(raw));self._shadow[:]=domains
            copied=self.n
        else:
            for cell in changed:
                value=domains[cell]
                C.memmove(self._buffer_address+cell*self.width,value.to_bytes(self.width,'little'),self.width)
                self._shadow[cell]=value
            copied=len(changed)
        stats['native_input_rows']=stats.get('native_input_rows',0)+copied

    def _events(self,batch):
        if batch>self._event_capacity:
            self._event_arcs=(C.c_int32*batch)()
            self._event_domains=(C.c_uint64*(batch*self.words))()
            self._event_capacity=batch

    def _replay(self,domains,reason,batch):
        count=int(self._event_count.value)
        if not 0<=count<=batch or count!=int(self.counts[1]):
            raise RuntimeError('Malformed native revision event count.')
        # Validate the complete bounded batch before exposing any events.
        # Domain mutation occurs only during replay, in exact C revision order.
        previous=domains.copy();events=[]
        address=C.addressof(self._event_domains)
        for i in range(count):
            index=int(self._event_arcs[i])
            if not 0<=index<self.arc_count:raise RuntimeError('Malformed native revision arc.')
            cell,_,neighbor,_=self.tables.arcs[index]
            value=int.from_bytes(C.string_at(address+i*self.width,self.width),'little')
            if value==previous[cell] or value&~previous[cell]:
                raise RuntimeError('Malformed native revision domain.')
            previous[cell]=value;events.append((cell,value,neighbor))
        for cell,value,neighbor in events:
            domains[cell]=value
            reason(cell,value,neighbor)
        self._shadow[:]=previous
        return count

    def propagate(self,domains,initial_arcs,check,stats,batch=128,reason=None):
        """Propagate exact seams, optionally replaying ordered direct reasons.

        A reason callback observes domains[cell] already revised and only earlier
        revisions visible, and receives (cell,new_domain,neighbor). It must not
        mutate domains or recursively call this engine. Exceptions preserve the
        applied prefix; the next call re-syncs every row before using C again.
        Old libraries use single-arc batches when tracing, preserving this API.
        """
        if self._active:raise RuntimeError('Native propagation workspace is already active.')
        if len(domains)!=self.n or any(type(v)is not int or v<0 or v.bit_length()>self.count for v in domains):
            raise ValueError('Malformed native propagation domains.')
        if not all(domains):return False
        if type(batch)is not int or batch<1 or batch>4096:raise ValueError('Invalid native arc batch size.')
        if reason is not None and not callable(reason):raise ValueError('Native reason callback is not callable.')
        initial=tuple(initial_arcs)
        if len(initial)!=len(set(initial)) or any(type(i)is not int or not 0<=i<self.arc_count for i in initial):
            raise ValueError('Malformed native initial arc queue.')
        self._active=True
        try:
            check();self._sync(domains,stats)
            C.memset(C.addressof(self.pending),0,self.capacity)
            for i,index in enumerate(initial):self.queue[i]=index;self.pending[index]=1
            self.state[:]=(0,len(initial)%self.capacity,len(initial))
            traced=reason is not None
            if traced and self.use_workspace:self._events(batch)
            effective_batch=1 if traced and not self.use_workspace else batch
            first=True
            while True:
                if not first:check()
                first=False
                next_arc=int(self.queue[self.state[0]]) if self.state[2] else None
                args=(self.buffer,self.support,self.arcs,self.incoming,self.n,self.count,self.words,
                      self.arc_count,self.queue,self.pending,self.state,effective_batch,self.counts)
                if self.use_workspace:
                    status=self.lib.dd_arc_batch_workspace(*args,self.residues if self.use_residues else None,
                        self.dirty,self._event_arcs if traced else None,self._event_domains if traced else None,
                        self._event_capacity if traced else 0,C.byref(self._event_count) if traced else None)
                else:
                    # The original export writes only its original two counters.
                    if not self.use_residues:self.counts[2]=self.counts[3]=0
                    status=(self.lib.dd_arc_batch_cached(*args,self.residues) if self.use_residues
                            else self.lib.dd_arc_batch(*args))
                if status not in (-1,0,1):raise RuntimeError('Native propagation rejected malformed internal state.')
                stats['native_calls']+=1;stats['native_arcs_checked']+=int(self.counts[0])
                stats['native_support_hits']+=int(self.counts[2]);stats['native_support_scans']+=int(self.counts[3])
                stats['native_arc_revisions']+=int(self.counts[1]);stats['arc_revisions']+=int(self.counts[1])
                if traced and self.use_workspace:
                    revised=self._replay(domains,reason,effective_batch)
                else:
                    changed=(range(self.n) if not self.use_workspace else
                             [cell for cell in range(self.n) if self.dirty[cell]])
                    revised=0
                    for cell in changed:
                        value=int.from_bytes(C.string_at(self._buffer_address+cell*self.width,self.width),'little')
                        if value!=domains[cell]:
                            domains[cell]=value;revised+=1
                            if traced:
                                if next_arc is None or self.tables.arcs[next_arc][0]!=cell:
                                    raise RuntimeError('Malformed legacy native revision.')
                                reason(cell,value,self.tables.arcs[next_arc][2])
                        self._shadow[cell]=value
                stats['native_output_rows']=stats.get('native_output_rows',0)+revised
                if traced:stats['native_reason_events']=stats.get('native_reason_events',0)+revised
                check()
                if status:return status>0
        except BaseException:
            # A callback can stop halfway through replay while C is already at
            # batch end. Never trust that native state on the next invocation.
            self._shadow[:]=[None]*self.n
            raise
        finally:self._active=False


def select_native(tables,mode='auto'):
    if mode not in ('auto','python','compiled'):raise ValueError('Unknown native propagation mode.')
    if mode=='python':return None,'Python propagation was requested.'
    try:return NativePropagation(tables),None
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        reason='Compiled propagation unavailable: '+str(exc)
        if mode=='compiled':raise RuntimeError(reason+' Run python build_native.py to build it locally.') from exc
        return None,reason


class NativeGraphHelpers:
    """Exact bounded port-graph certificates; interrupted results are discarded."""
    def __init__(self,library=None):
        self.lib=library if library is not None else load_library()
        if not hasattr(self.lib,'dd_cyclic_blocks'):raise RuntimeError('Native graph helper missing; rebuild locally.')
    def cyclic_edge_blocks(self,vertex_count,edges,check=lambda:None):
        if type(vertex_count)is not int or not 0<=vertex_count<=5280:raise ValueError('Invalid native graph vertex count.')
        check();edges=tuple(edges)
        if len(edges)>262144:raise ValueError('Native graph edge bound exceeded.')
        flat=[]
        for index,pair in enumerate(edges):
            if index%256==0:check()
            if (not isinstance(pair,(tuple,list)) or len(pair)!=2 or
                    any(type(v)is not int or not 0<=v<vertex_count for v in pair)):
                raise ValueError('Malformed native graph edge.')
            flat.extend(pair)
        inputs=(C.c_int32*len(flat))(*flat)
        return self._cyclic_blocks(vertex_count,len(edges),inputs,check)

    def cyclic_edge_blocks_packed(self,vertex_count,packed,check=lambda:None):
        """Exact blocks from immutable native-endian signed32 endpoint pairs.

        The byte representation is intentionally explicit: no float conversion,
        truncating integer cast, strided view, or borrowed mutable storage. Array
        construction by the caller checks signed32 range; every possible byte
        pattern denotes signed32 values. C validates every endpoint before using
        it as an index. Empty graphs and arbitrary invalid bytes are safe inputs.
        """
        if type(vertex_count)is not int or not 0<=vertex_count<=5280:raise ValueError('Invalid native graph vertex count.')
        if type(packed)is not bytes or len(packed)%8 or len(packed)>8*262144:
            raise ValueError('Packed native graph must contain bounded signed32 endpoint pairs.')
        if not vertex_count and packed:raise ValueError('A zero-vertex graph cannot have edges.')
        check()
        # One bounded memcpy gives aligned, privately owned C storage. A poll
        # callback cannot mutate the original graph while traversal is active.
        inputs=(C.c_int32*(len(packed)//4)).from_buffer_copy(packed)
        return self._cyclic_blocks(vertex_count,len(packed)//8,inputs,check,packed=True)

    def _cyclic_blocks(self,vertex_count,edge_count,inputs,check,packed=False):
        output=(C.c_int32*edge_count)();interrupted=[]
        def poll():
            try:check();return 0
            except BaseException as exc:interrupted.append(exc);return 1
        callback=POLL_CALLBACK(poll)
        status=self.lib.dd_cyclic_blocks(vertex_count,edge_count,inputs,output,callback)
        if interrupted:raise interrupted[0]
        if status==-2 and packed:raise ValueError('Malformed packed native graph endpoint.')
        if status!=1:raise RuntimeError('Native graph helper failed with status '+str(status))
        check();return tuple(output)


def select_graph_helpers(mode='auto'):
    if mode not in ('auto','python','compiled'):raise ValueError('Unknown native graph mode.')
    if mode=='python':return None,'Python graph helpers were requested.'
    try:return NativeGraphHelpers(),None
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        reason='Compiled graph helper unavailable: '+str(exc)
        if mode=='compiled':raise RuntimeError(reason+' Run python build_native.py to build it locally.') from exc
        return None,reason


class NativeHall:
    """Exact balanced AllDifferent filtering, with revalidated matching hints."""
    def __init__(self,library=None):
        if sys.byteorder!='little':raise ValueError('Native Hall filtering currently requires a little-endian host.')
        self.lib=library if library is not None else load_library()
        if not hasattr(self.lib,'dd_hall_allowed'):raise RuntimeError('Native Hall helper missing; rebuild locally.')
    def filter(self,rows,previous=None,check=lambda:None):
        rows=tuple(rows);n=len(rows)
        if not 1<=n<=160 or any(type(v)is not int or v<0 or v.bit_length()>n for v in rows):
            raise ValueError('Malformed native physical-tile domains.')
        if previous is None:previous=(-1,)*n
        if len(previous)!=n or any(type(v)is not int or not -1<=v<n for v in previous):raise ValueError('Malformed matching hints.')
        check();words=(n+63)//64;width=words*8
        raw=b''.join(value.to_bytes(width,'little') for value in rows)
        inputs=(C.c_uint64*(n*words)).from_buffer_copy(raw);output=(C.c_uint64*(n*words))()
        matching=(C.c_int32*n)(*previous);failed=C.c_int32(-1);interrupted=[]
        def poll():
            try:check();return 0
            except BaseException as exc:interrupted.append(exc);return 1
        status=self.lib.dd_hall_allowed(n,words,inputs,output,matching,C.byref(failed),POLL_CALLBACK(poll))
        if interrupted:raise interrupted[0]
        if status not in (0,1):raise RuntimeError('Native Hall helper failed with status '+str(status))
        check()
        if not status:return None,None,int(failed.value)
        data=bytes(output)
        return tuple(int.from_bytes(data[i*width:(i+1)*width],'little') for i in range(n)),tuple(matching),None


def select_hall(mode='auto'):
    if mode not in ('auto','python','compiled'):raise ValueError('Unknown native Hall mode.')
    if mode=='python':return None,'Python Hall filtering was requested.'
    try:return NativeHall(),None
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        reason='Compiled Hall helper unavailable: '+str(exc)
        if mode=='compiled':raise RuntimeError(reason+' Run python build_native.py to build it locally.') from exc
        return None,reason
