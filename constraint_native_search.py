"""Optional compiled DFS control with exact checkpoint frames and an undo trail.

The C engine, not a Python generator, owns branch iteration, its decision stack,
rollback, and node consumption. Python callbacks retain the existing puzzle
rules. This is optional: crossing the FFI for every node is not always faster.
Masks in frame choices must be disjoint and cover that frame's selected domain.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,platform,sys

ROOT=Path(__file__).resolve().parent
STAT_KEYS=('native_search_nodes','native_search_branches','native_search_pops',
           'native_search_trail_writes','native_search_trail_undos',
           'native_search_peak_trail','native_search_peak_depth','native_search_transitions',
           'native_fast_nodes','native_fast_rejects','native_fast_rows','native_fast_arcs',
           'native_fast_support_scans','native_fast_support_hits','native_fast_polls',
           'native_fast_enqueues','native_fast_incoming_visits','native_fast_seeded_nodes')
_U64=C.POINTER(C.c_uint64)
_I32=C.POINTER(C.c_int)
_PROCESS=C.CFUNCTYPE(C.c_int,C.c_void_p,_U64,_I32,_U64,_I32,_I32)
_POLL=C.CFUNCTYPE(C.c_int,C.c_void_p)
_EVENT=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int)
STATUS={0:'ready',1:'exhausted',2:'solved',3:'stopped',4:'limit'}

def load_library(directory=None):
    directory=Path(directory or ROOT/'.native')
    metadata=json.loads((directory/'search-build.json').read_text(encoding='utf-8'))
    if (metadata.get('abi')!=1 or metadata.get('platform')!=sys.platform
            or metadata.get('machine')!=platform.machine()):
        raise RuntimeError('Native search ABI/platform/machine differs from this installation.')
    name=metadata.get('library')
    if not isinstance(name,str) or Path(name).name!=name:
        raise RuntimeError('Native search build manifest is malformed.')
    source=ROOT/'constraint_native_search.c';library=directory/name
    if metadata.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError('Native search source changed; rebuild the local library.')
    if metadata.get('library_sha256')!=hashlib.sha256(library.read_bytes()).hexdigest():
        raise RuntimeError('Native search library differs from its local build manifest.')
    lib=C.CDLL(str(library));lib.dd_search_abi.restype=C.c_int
    if lib.dd_search_abi()!=1:raise RuntimeError('Native search library ABI differs.')
    lib.dd_search_create.argtypes=[C.c_int,C.c_int,C.c_int];lib.dd_search_create.restype=C.c_void_p
    lib.dd_search_configure.argtypes=[C.c_void_p,_U64,_I32,C.c_int];lib.dd_search_configure.restype=C.c_int
    lib.dd_search_destroy.argtypes=[C.c_void_p];lib.dd_search_destroy.restype=None
    lib.dd_search_add_frame.argtypes=[C.c_void_p,_U64,C.c_int,_U64,C.c_int,C.c_int];lib.dd_search_add_frame.restype=C.c_int
    lib.dd_search_copy_frame.argtypes=[C.c_void_p,C.c_int,_U64,_I32,_U64,_I32,_I32];lib.dd_search_copy_frame.restype=C.c_int
    lib.dd_search_decisions.argtypes=[C.c_void_p,_I32,_U64];lib.dd_search_decisions.restype=C.c_int
    lib.dd_search_stats.argtypes=[C.c_void_p,_U64];lib.dd_search_stats.restype=C.c_int
    lib.dd_search_fast_stats.argtypes=[C.c_void_p,_U64];lib.dd_search_fast_stats.restype=C.c_int
    lib.dd_search_queue_stats.argtypes=[C.c_void_p,_U64];lib.dd_search_queue_stats.restype=C.c_int
    lib.dd_search_run.argtypes=[C.c_void_p,_PROCESS,_POLL,_EVENT,C.c_void_p,C.c_uint64];lib.dd_search_run.restype=C.c_int
    for name in ('dd_search_depth','dd_search_status','dd_search_basic_ready'):
        getattr(lib,name).argtypes=[C.c_void_p];getattr(lib,name).restype=C.c_int
    return lib,metadata


def select_native_search(n,count=None,mode='auto',directory=None):
    """Return (engine, reason); explicit compiled mode must never silently fall back."""
    if mode not in ('auto','python','compiled'):raise ValueError('Unknown native search backend.')
    if mode=='python':return None,'Python search control selected.'
    try:return NativeSearch(n,count,directory=directory),None
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        reason=str(exc)
        if mode=='compiled':raise RuntimeError('Compiled search unavailable; run build_native.py: '+reason) from exc
        return None,reason


class NativeSearch:
    """Bounded, non-reentrant search workspace. Use run() with mask-based frames.

    process_node(domains, decisions) returns {'action':'branch', 'domains':...,
    'cell':int, 'choices':[disjoint masks]} or {'action':'reject'/'solved'}.
    Rejection may include backjump_depth (number of ancestors to retain).
    on_event receives (event_name, result_dict); it runs after committing the
    action, so it may safely snapshot or stop without replaying consumed leaves.
    Returning {'replace_stack': frames} transactionally replaces the remaining
    frontier after that commit. The caller must prove all discarded work invalid.
    A callback exception is re-raised with .native_search_state when possible;
    snapshot() and last_result remain available even for exceptions without a
    writable __dict__. Uncommitted callback domain changes are discarded.
    """
    backend='compiled'
    def __init__(self,n,count=None,*,directory=None,max_depth=4096):
        count=3*n if count is None else count
        if type(n)is not int or not 1<=n<=160 or type(count)is not int or not 1<=count<=480:
            raise ValueError('Native search supports 1..160 cells and 1..480 codes.')
        if type(max_depth)is not int or not 1<=max_depth<=4096:raise ValueError('Invalid native search depth limit.')
        if sys.byteorder!='little':raise ValueError('Native search currently requires a little-endian host.')
        self._directory=Path(directory or ROOT/'.native').resolve()
        self.lib,self.metadata=load_library(self._directory)
        self.signature=(self.metadata['source_sha256'],self.metadata['library_sha256'],self.metadata['abi'],self.metadata['platform'],self.metadata['machine'])
        self.n=n;self.count=count;self.words=(count+63)//64;self.width=self.words*8;self.max_depth=max_depth
        self._handle=None;self._active=False;self.last_result=None;self._propagation=None
        self._stats_base={key:0 for key in STAT_KEYS}
        self._domain_buffer=(C.c_uint64*(n*self.words))()
        self._choice_buffer=(C.c_uint64*(count*self.words))()
        self._decision_cells=(C.c_int*max_depth)()
        self._decision_masks=(C.c_uint64*(max_depth*self.words))()
        self._new_handle()

    def _new_handle(self):
        self._stats_base={key:0 for key in STAT_KEYS}
        if self._handle:self.lib.dd_search_destroy(self._handle)
        self._handle=self.lib.dd_search_create(self.n,self.count,self.max_depth)
        if not self._handle:raise RuntimeError('Unable to allocate the bounded native search workspace.')
        if self._propagation is not None:self._configure(*self._propagation)

    def _configure(self,support,arcs,arc_count):
        result=self.lib.dd_search_configure(self._handle,support,arcs,arc_count)
        if result:raise RuntimeError('Native propagation configuration failed ('+str(result)+').')

    def configure_propagation(self,tables):
        """Enable exact seam and tile rules. The caller must conservatively account
        for prefilter consequences, or disable this when precise event traces
        are required. Copies are immutable and
        retained across load/run. Does not replace final independent validation.
        """
        if self._active:raise RuntimeError('Cannot configure an active native search workspace.')
        if not self._handle:raise RuntimeError('Native search workspace is closed.')
        if self.count!=3*self.n:raise ValueError('Puzzle propagation requires exactly three codes per tile.')
        compatible=tables.compatible;arcs=tables.arcs
        if (len(compatible)!=3 or any(len(row)!=3 for row in compatible)
                or any(len(side)!=self.count for row in compatible for side in row)):
            raise ValueError('Malformed seam support dimensions.')
        masks=[value for row in compatible for side in row for value in side]
        if any(type(v)is not int or v<0 or v.bit_length()>self.count for v in masks):
            raise ValueError('Malformed seam support mask.')
        if (len(arcs)>960 or any(len(arc)!=4 or any(type(v)is not int for v in arc)
                or not 0<=arc[0]<self.n or not 0<=arc[2]<self.n
                or not 0<=arc[1]<3 or not 0<=arc[3]<3 for arc in arcs)):
            raise ValueError('Malformed seam arcs.')
        support=(C.c_uint64*(9*self.count*self.words)).from_buffer_copy(self._pack(masks))
        edges=(C.c_int*(len(arcs)*4))(*(v for arc in arcs for v in arc))
        self._configure(support,edges,len(arcs));self._propagation=(support,edges,len(arcs))

    def close(self):
        if self._active:raise RuntimeError('Cannot close an active native search workspace.')
        if self._handle:self.lib.dd_search_destroy(self._handle);self._handle=None

    def __del__(self):
        handle=getattr(self,'_handle',None)
        if handle and not getattr(self,'_active',False):self.lib.dd_search_destroy(handle);self._handle=None

    def _domains(self,values):
        if (not isinstance(values,(list,tuple)) or len(values)!=self.n
                or any(type(v)is not int or v<=0 or v.bit_length()>self.count for v in values)):
            raise ValueError('Malformed native search domains.')
        return values

    def _pack(self,values):
        return b''.join(value.to_bytes(self.width,'little') for value in values)

    def _unpack(self,pointer,length):
        data=C.string_at(pointer,length*self.width)
        return [int.from_bytes(data[i*self.width:(i+1)*self.width],'little') for i in range(length)]

    def _frame(self,frame):
        if not isinstance(frame,dict) or not {'domains','cell','choices','next'}<=frame.keys():
            raise ValueError('Malformed native search frame.')
        domains=self._domains(frame['domains']);cell=frame['cell'];choices=frame['choices'];offset=frame['next']
        if (type(cell)is not int or not -1<=cell<self.n or not isinstance(choices,(list,tuple))
                or len(choices)>self.count or type(offset)is not int or not 0<=offset<=len(choices)):
            raise ValueError('Malformed native search frame choices.')
        if cell==-1:
            if choices or offset:raise ValueError('Unexpanded native search frame has choices.')
        else:
            if len(choices)<2:raise ValueError('Native search branch must have at least two choices.')
            union=0
            for value in choices:
                if type(value)is not int or value<=0 or value&~domains[cell] or value&union:
                    raise ValueError('Native search choices must be a disjoint domain partition.')
                union|=value
            if union!=domains[cell]:raise ValueError('Native search choices do not cover their domain.')
        return domains,cell,choices,offset

    def _validated_stack(self,stack):
        if not isinstance(stack,(list,tuple)) or len(stack)>self.max_depth:raise ValueError('Malformed native search stack.')
        validated=[self._frame(frame) for frame in stack]
        # Validate the whole parent/child chain before replacing prior state.
        for previous,current in zip(validated,validated[1:]):
            pd,pc,pch,pnext=previous;d,_,_,_=current
            if pc<0 or not 1<=pnext<=len(pch) or any(v&~p for v,p in zip(d,pd)) or d[pc]&~pch[pnext-1]:
                raise ValueError('Native search child does not belong to its selected parent choice.')
        return validated

    def _fill_handle(self,handle,validated):
        for domains,cell,choices,offset in validated:
            C.memmove(self._domain_buffer,self._pack(domains),self.n*self.width)
            if choices:C.memmove(self._choice_buffer,self._pack(choices),len(choices)*self.width)
            result=self.lib.dd_search_add_frame(handle,self._domain_buffer,cell,self._choice_buffer,len(choices),offset)
            if result:raise RuntimeError('Native search could not import a validated frame ('+str(result)+').')

    def load(self,stack):
        if self._active:raise RuntimeError('Native search workspace is already active.')
        validated=self._validated_stack(stack)
        self._new_handle();self.last_result=None
        self._fill_handle(self._handle,validated)

    def snapshot(self):
        if not self._handle:raise RuntimeError('Native search workspace is closed.')
        depth=self.lib.dd_search_depth(self._handle);stack=[]
        for index in range(depth):
            cell=C.c_int();count=C.c_int();offset=C.c_int()
            result=self.lib.dd_search_copy_frame(self._handle,index,self._domain_buffer,C.byref(cell),self._choice_buffer,C.byref(count),C.byref(offset))
            if result:raise RuntimeError('Native search snapshot failed.')
            stack.append({'domains':self._unpack(self._domain_buffer,self.n),'cell':cell.value,
                          'choices':self._unpack(self._choice_buffer,count.value),'next':offset.value})
        return stack

    def decisions(self):
        length=self.lib.dd_search_decisions(self._handle,self._decision_cells,self._decision_masks)
        if length<0:raise RuntimeError('Native search decisions are malformed.')
        return list(zip(self._decision_cells[:length],self._unpack(self._decision_masks,length)))

    @property
    def basic_propagated(self):
        """True only inside a callback after this C workspace completed its rules.

        This private live certificate is never loaded from a checkpoint. The
        callback may skip its initial basic pass; later narrowing still needs
        normal propagation to a fixpoint.
        """
        return bool(self._handle and self.lib.dd_search_basic_ready(self._handle))

    def stats(self):
        values=(C.c_uint64*len(STAT_KEYS))()
        if self.lib.dd_search_stats(self._handle,values):raise RuntimeError('Native search statistics unavailable.')
        extra=(C.c_uint64*7)()
        if self.lib.dd_search_fast_stats(self._handle,extra):raise RuntimeError('Native fast statistics unavailable.')
        queue=(C.c_uint64*3)()
        if self.lib.dd_search_queue_stats(self._handle,queue):raise RuntimeError('Native queue statistics unavailable.')
        current=dict(zip(STAT_KEYS,[*map(int,values[:8]),*map(int,extra),*map(int,queue)]))
        return {key:max(self._stats_base[key],value) if key in ('native_search_peak_trail','native_search_peak_depth')
                else self._stats_base[key]+value for key,value in current.items()}

    def _replace_frontier(self,stack):
        """Build and validate replacement before swapping the committed handle.

        Reuse the loaded, hash-verified library; no file hashing or additional
        Python wrapper is required for each jump. All new native state remains
        private until configuration and frame imports have succeeded.
        """
        validated=self._validated_stack(stack)
        replacement=self.lib.dd_search_create(self.n,self.count,self.max_depth)
        if not replacement:raise RuntimeError('Unable to allocate replacement native search workspace.')
        try:
            if self._propagation is not None:
                result=self.lib.dd_search_configure(replacement,*self._propagation)
                if result:raise RuntimeError('Replacement native propagation configuration failed ('+str(result)+').')
            self._fill_handle(replacement,validated)
            accumulated=self.stats()
        except BaseException:
            self.lib.dd_search_destroy(replacement)
            raise
        old=self._handle;self._handle=replacement;self._stats_base=accumulated
        self.lib.dd_search_destroy(old)

    def run(self,stack,process_node,check=lambda:None,on_event=None,*,max_steps=0):
        if self._active:raise RuntimeError('Native search workspace is already active.')
        if type(max_steps)is not int or not 0<=max_steps<1<<64:raise ValueError('Invalid native search transition budget.')
        if not callable(process_node) or not callable(check) or (on_event is not None and not callable(on_event)):
            raise ValueError('Native search callbacks must be callable.')
        if stack is not None:self.load(stack)
        if not self._handle:raise RuntimeError('Native search workspace is closed.')
        error=[];pending=[None];replacement=[None]
        solved=[self.last_result.get('value') if self.last_result and self.last_result.get('status')=='solved' else None]
        def capture(exc):
            if not error:error.append(exc)
            return -1
        @_POLL
        def poll(_):
            try:check();return 0
            except BaseException as exc:return capture(exc)
        @_PROCESS
        def process(_,buffer,cell,choices,count,jump):
            try:
                before=self._unpack(buffer,self.n);result=process_node(before.copy(),self.decisions())
                if not isinstance(result,dict):raise ValueError('Native node callback must return a dictionary.')
                action=result.get('action','branch')
                if action not in ('branch','reject','solved'):raise ValueError('Unknown native search node action.')
                target=result.get('backjump_depth',-1)
                depth=self.lib.dd_search_depth(self._handle)
                if type(target)is not int or not -1<=target<depth or (action!='reject' and target!=-1):
                    raise ValueError('Invalid native search backjump depth.')
                jump[0]=target
                if action=='branch':
                    frame={'domains':result.get('domains',before),'cell':result.get('cell'),
                           'choices':result.get('choices'),'next':0}
                    narrowed,selected,partition,_=self._frame(frame)
                    if selected<0:raise ValueError('Native branch callback must select a cell.')
                    if any(value&~old for value,old in zip(narrowed,before)):raise ValueError('Native callback may not widen domains.')
                    C.memmove(buffer,self._pack(narrowed),self.n*self.width);cell[0]=selected;count[0]=len(partition)
                    C.memmove(choices,self._pack(partition),len(partition)*self.width)
                pending[0]=result
                if action=='solved':solved[0]=result.get('value',result)
                return {'reject':0,'branch':1,'solved':2}[action]
            except BaseException as exc:return capture(exc)
        @_EVENT
        def event(_,code):
            try:
                if on_event is not None:
                    payload={'action':'reject','native_fast_reject':True} if code==4 else pending[0] if code!=1 else None
                    response=on_event(('reject','backtrack','expanded','solved','fast_reject')[code],payload)
                    if response is not None:
                        if not isinstance(response,dict) or set(response)!={'replace_stack'}:
                            raise ValueError('Native event callback must return None or a replacement frontier.')
                        if code==3:raise ValueError('A solved native search cannot replace its frontier.')
                        replacement[0]=response['replace_stack']
                        pending[0]=None
                        return 1
                pending[0]=None
                return 0
            except BaseException as exc:return capture(exc)
        self._active=True
        start_transitions=self.stats()['native_search_transitions']
        try:
            while True:
                used=self.stats()['native_search_transitions']-start_transitions
                if max_steps and used>=max_steps:status=3;break
                status=self.lib.dd_search_run(self._handle,process,poll,event,None,max_steps-used if max_steps else 0)
                if replacement[0] is None or error or status not in STATUS:break
                proposed=replacement[0];replacement[0]=None
                try:self._replace_frontier(proposed)
                except BaseException as exc:capture(exc);break
                if not proposed:status=1;break
        finally:self._active=False
        if status not in STATUS:raise RuntimeError('Native search rejected its callback contract ('+str(status)+').')
        state={'status':STATUS[status],'stack':self.snapshot(),'stats':self.stats(),'value':solved[0],
               'backend':self.backend,'signature':self.signature}
        self.last_result=state
        if error:
            try:error[0].native_search_state=state
            except (AttributeError,TypeError):pass
            raise error[0]
        return state
