"""Bounded heuristic GPU sampling followed by complete, strongly pruned CPU DFS.

Sampling preserves physical-tile uniqueness and explicit fixed placements. Its
boards may violate seams/domains, may repeat, and are only value-order hints.
GPU time includes imports, allocation, compilation, and CPU verification. There
is no CPU sampling fallback, and sample failure never establishes infeasibility.
"""
from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from gpu_backends import choose_backend
from validator import reverse_mask
from constraint_lns import diverse_archive

ROOT = Path(__file__).resolve().parent

CUDA_SOURCE = r'''
__device__ unsigned rnd(unsigned &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
__device__ int legal(const unsigned*t,const unsigned*b,int n,int r,int f,int c,int lane){return t[25*n+f+c*3*n+b[c*r+lane]];}
__device__ int edge(const unsigned*t,const unsigned*b,int n,int r,int c,int side,int lane){
 unsigned nb=t[18*n+c*3+side];if(nb>=n)return 0;
 return t[b[c*r+lane]*3+side]==t[9*n+b[nb*r+lane]*3+t[21*n+c*3+side]];
}
__device__ int total(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int lane){
 int value=0;for(int c=0;c<n;c++){value+=legal(t,b,n,r,f,c,lane)*(edges+1);
 for(int s=0;s<3;s++){unsigned nb=t[18*n+c*3+s];if(nb<n&&nb>c)value+=edge(t,b,n,r,c,s,lane);}}return value;
}
// Score only changed domain terms and incident seams; old board stays intact.
__device__ int change(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int a,int d,unsigned na,unsigned nd,int lane){
 int delta=0;
 for(int which=0;which<(a==d?1:2);which++){
 int c=which?d:a;unsigned old=b[c*r+lane],next=which?nd:na;int base=25*n+f+c*3*n;
 delta+=(int(t[base+next])-int(t[base+old]))*(edges+1);
 for(int side=0;side<3;side++){
 unsigned nb=t[18*n+c*3+side];if(nb>=n||(which&&nb==a))continue;
 unsigned opposite=t[21*n+c*3+side],other=b[nb*r+lane];
 unsigned replacement=nb==a?na:(nb==d?nd:other);
 delta+=int(t[next*3+side]==t[9*n+replacement*3+opposite])-int(t[old*3+side]==t[9*n+other*3+opposite]);
 }}return delta;
}

__device__ unsigned partner(const unsigned*t,int n,int f,unsigned code,unsigned port){return t[25*n+f+3*n*n+1+code*33+port];}
// Count only closed proper components touching a moved cell (a<0: all cells).
// Among touched segments the smallest (cell, local endpoint pair) counts once.
__device__ int loops(const unsigned*t,const unsigned*b,int n,int r,int f,int a,int d,int lane,int total){
 if(!total)return 0;int result=0;
 int roots=a<0?n:(a==d?1:2);
 for(int root=0;root<roots;root++) {int start=a<0?root:(root?d:a);unsigned code=b[start*r+lane];
 for(int port=0;port<33;port++) {unsigned pair=partner(t,n,f,code,port);if(pair>=33||pair<unsigned(port))continue;
 int cell=start,incoming=port,key=start*33+port;
 for(int count=1;count<=total;count++) {
   unsigned outgoing=partner(t,n,f,b[cell*r+lane],incoming);if(outgoing>=33)break;
   int local=cell*33+min(incoming,int(outgoing));if((a<0||cell==a||cell==d)&&local<key)break;
   int side=outgoing/11;unsigned next=t[18*n+cell*3+side];if(next>=unsigned(n))break;
   incoming=t[21*n+cell*3+side]*11+10-outgoing%11;cell=next;
   if(cell==start&&incoming==port){if(count<total)++result;break;}
 }
 }}return result;
}
extern "C" __global__ void sample_search(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int steps,int phase,int edges,int gold_total){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;unsigned random=m[lane];
 if(phase==0){
 for(int c=0;c<n;c++)b[c*r+lane]=t[24*n+c];
 for(int count=f;count>1;count--){int a=t[25*n+count-1],d=t[25*n+rnd(random)%count];unsigned old=b[a*r+lane];b[a*r+lane]=b[d*r+lane];b[d*r+lane]=old;}
 for(int i=0;i<f;i++){int c=t[25*n+i];b[c*r+lane]=3*(b[c*r+lane]/3)+rnd(random)%3;}
 m[r+lane]=m[2*r+lane]=total(t,b,n,r,f,edges,lane)*(gold_total+1)+gold_total-loops(t,b,n,r,f,-1,-1,lane,gold_total);m[3*r+lane]=0;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
 }else if(f>0){
 int score=m[r+lane],bestscore=m[2*r+lane];unsigned iteration=m[3*r+lane];
 for(int step=0;step<steps;step++){
 int a=t[25*n+rnd(random)%f],d=t[25*n+rnd(random)%f];unsigned ca=b[a*r+lane],cd=b[d*r+lane];
 unsigned na=3*(cd/3)+rnd(random)%3,nd=cd;if(a!=d)nd=3*(ca/3)+rnd(random)%3;
 int base_delta=change(t,b,n,r,f,edges,a,d,na,nd,lane),loop_delta=0;
 if(gold_total){int old_loops=loops(t,b,n,r,f,a,d,lane,gold_total);b[a*r+lane]=na;if(a!=d)b[d*r+lane]=nd;
 loop_delta=old_loops-loops(t,b,n,r,f,a,d,lane,gold_total);b[a*r+lane]=ca;if(a!=d)b[d*r+lane]=cd;}
 int delta=base_delta*(gold_total+1)+loop_delta;int acceptance=base_delta?base_delta:loop_delta;
 float heat=.25f+1.25f*(1.f-float(iteration%4096)/4096.f);
 bool accept=acceptance>=0||float(rnd(random)%1000000)<1000000.f*expf(float(acceptance)/heat);iteration++;
 if(accept){b[a*r+lane]=na;if(a!=d)b[d*r+lane]=nd;score+=delta;
 if(score>bestscore||(score==bestscore&&rnd(random)%64==0)){bestscore=score;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}}
 }m[r+lane]=score;m[2*r+lane]=bestscore;m[3*r+lane]=iteration;
 }m[lane]=random;
}
'''


def _inputs(problem, gold_aware=False):
    n = problem.n
    if not 1 <= n <= 160:
        raise ValueError('GPU sampler supports 1..160 cells.')
    fixed = dict(problem.fixed)
    pieces = [code // 3 for code in fixed.values()]
    if len(set(pieces)) != len(pieces):
        return None  # Let exact DFS prove this, without inventing a valid permutation.
    free = [cell for cell in range(n) if cell not in fixed]
    unused = [tile for tile in range(n) if tile not in pieces]
    base = np.zeros(n, np.uint32)
    for cell, code in fixed.items(): base[cell] = code
    for cell, tile in zip(free, unused): base[cell] = 3 * tile
    domains = np.zeros((n, 3*n), np.uint32)
    for cell, values in enumerate(problem.domains): domains[cell, values] = 1
    reverse = np.array([[reverse_mask(v) for v in row] for row in problem.masks], np.uint32)
    table = np.concatenate((np.asarray(problem.masks).ravel(), reverse.ravel(),
                            np.asarray(problem.neighbors).ravel(), np.asarray(problem.sides).ravel(),
                            base, np.asarray(free, np.uint32), domains.ravel())).astype(np.uint32)
    if gold_aware:
        partners=np.full((3*n,33),255,np.uint32)
        for tile,item in enumerate(problem.tiles):
            for rotation in range(3):
                for left,right in item['segments']:
                    a=((left[0]+rotation)%3)*11+left[1]-1
                    b=((right[0]+rotation)%3)*11+right[1]-1
                    partners[3*tile+rotation,a]=b;partners[3*tile+rotation,b]=a
        table=np.concatenate((table,np.array([problem.total_segments],np.uint32),partners.ravel()))
    return table, len(free)


def score_board(problem, codes):
    """Independent raw-mask CPU verification of every returned sampling hint."""
    raw = np.asarray(codes)
    if raw.shape != (problem.n,) or raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= 3*problem.n):
        raise ValueError('GPU sample has malformed orientation codes.')
    if not np.array_equal(np.sort(raw // 3), np.arange(problem.n)):
        raise ValueError('GPU sample repeats or omits physical tiles.')
    if any(int(raw[cell]) != code for cell, code in problem.fixed.items()):
        raise ValueError('GPU sample changed an explicit fixed placement.')
    matched = sum(int(problem.masks[raw[a], sa]) == reverse_mask(problem.masks[raw[b], sb])
                  for a, sa, b, sb in problem.edges)
    violations = sum(int(code) not in problem.domains[cell] for cell, code in enumerate(raw))
    return {'matched_edges': int(matched), 'required_edges': len(problem.edges),
            'domain_violations': int(violations),
            'objective': int((problem.n-violations)*(len(problem.edges)+1)+matched)}


def gold_score(problem, codes):
    metrics=score_board(problem,codes)
    loops=sum(size<problem.total_segments for size,_ in problem._closed_components(list(map(int,codes))))
    return {**metrics,'closed_proper_loops':loops,
            'gold_objective':metrics['objective']*(problem.total_segments+1)+problem.total_segments-loops}


def _verification_view(problem, count):
    """Fresh membership tables for a batch; the raw-mask validator is unchanged."""
    if count < 32:
        return problem
    domains, shared = [], {}
    try:
        for values in problem.domains:
            # Small domains do not amortize the setup cost across this batch.
            if len(values) < 64:
                domains.append(values)
                continue
            key = tuple(values)
            domain = shared.get(key)
            if domain is None:
                domain = frozenset(key)
                shared[key] = domain
            domains.append(domain)
    except TypeError:
        return problem  # Preserve ordinary membership semantics for unusual inputs.
    if not shared:
        return problem
    # Nothing survives this collection: domain changes before the next sample
    # are observed, and GPU tables/scores never supply validation evidence.
    return SimpleNamespace(n=problem.n, fixed=problem.fixed, masks=problem.masks,
                           edges=problem.edges, domains=domains)


def _top_ids(scores, count):
    """Match stable descending score order, including descending lane-ID ties."""
    size = len(scores)
    count = min(count, size)
    if size < 2048 or count <= 0 or count * 8 >= size:
        return np.argsort(scores, kind='stable')[-count:][::-1].copy()
    # Only the best few uint32 scores are needed. Partition finds the cutoff;
    # explicitly keep the highest lane IDs on that boundary before sorting the
    # selected values. Arbitrary partition tie order must never choose hints.
    cutoff = np.partition(scores, size - count)[size - count]
    better = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[-(count - len(better)):]
    ids = np.concatenate((better, tied))
    return ids[np.lexsort((ids, scores[ids]))[::-1]].copy()


class _CUDA:
    def __init__(self, problem, table, free, replicas, seed, *, experiments=None):
        import cupy as cp
        self.cp, self.n, self.r, self.f, self.e = cp, problem.n, replicas, free, len(problem.edges)
        self.flags=sampling_flags(experiments);self.meta_rows=18 if self.flags else 4
        base=25*self.n+free+3*self.n*self.n
        self.gold_total=int(table[base]) if len(table)>base else 0
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        self.device = name.decode() if isinstance(name, bytes) else str(name)
        self.boards = cp.zeros((self.n, replicas), np.uint32)
        self.best = cp.zeros_like(self.boards)
        self.meta = cp.zeros((self.meta_rows, replicas), np.uint32)
        self.meta[0] = cp.asarray(np.random.default_rng(seed).integers(1, 2**32, replicas, dtype=np.uint32))
        self.table = cp.asarray(table)
        source=(ROOT/'sampling_round24.cu').read_text() if self.flags else CUDA_SOURCE
        self.kernel = cp.RawKernel(source, 'sample_search', options=('--std=c++17',))
        self.exchange_kernel=cp.RawKernel(source,'replica_exchange',options=('--std=c++17',)) if self.flags&4 else None
        self.launch(0, 0)
    def launch(self, steps, phase=1):
        self.kernel(((self.r+127)//128,), (128,), (self.boards, self.best, self.meta, self.table,
                    *map(np.int32, (self.n, self.r, self.f, steps, phase, self.e, self.gold_total)),
                    *((np.int32(self.flags),) if self.flags else ())))
        self.cp.cuda.get_current_stream().synchronize()
    def exchange(self, parity=0):
        if self.exchange_kernel is None:return
        self.exchange_kernel(((self.r+255)//256,),(128,),
            (self.boards,self.best,self.meta,*map(np.int32,(self.n,self.r,self.gold_total,parity))))
        self.cp.cuda.get_current_stream().synchronize()
    def experiment_stats(self):
        if not self.flags:return {}
        counts=self.meta[13:18].sum(axis=1,dtype=self.cp.uint64).get()
        return dict(zip(('accepted_moves','tabu_rejections','chain_proposals','exchange_attempts','exchange_accepts'),map(int,counts)))
    def top(self, count=32):
        scores = self.meta[2].get()
        ids = _top_ids(scores, count)
        return self.best[:, ids].get().T.astype(int), scores[ids].astype(int)
    def close(self):
        self.boards = self.best = self.meta = self.table = self.kernel = self.exchange_kernel = None


def _validate_sampling_adapter(info, *, allow_software=False):
    # Only finite tests pass this private opt-in. Runtime sample()/solve() and
    # the dashboard expose no software-fallback setting or environment switch.
    text = ' '.join(map(str, info.values())).lower()
    software = (str(info.get('adapter_type')).lower() in ('cpu', 'software')
                or any(term in text for term in ('llvmpipe', 'lavapipe', 'swiftshader', 'basic render')))
    if info.get('backend_type') not in ('Metal', 'Vulkan', 'D3D12') or (software and allow_software is not True):
        raise RuntimeError('Sampling requires a native hardware GPU; software/OpenGL fallback is disabled.')


class _WebGPU:
    def __init__(self, problem, table, free, replicas, seed, *, allow_software=False, experiments=None):
        import wgpu
        self.wgpu, self.n, self.r, self.f, self.e = wgpu, problem.n, replicas, free, len(problem.edges)
        self.flags=sampling_flags(experiments);self.meta_rows=18 if self.flags else 4
        base=25*self.n+free+3*self.n*self.n
        self.gold_total=int(table[base]) if len(table)>base else 0
        adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance', force_fallback_adapter=False)
        if adapter is None: raise RuntimeError('No native GPU adapter is available for sampling.')
        info = dict(adapter.info)
        _validate_sampling_adapter(info, allow_software=allow_software)
        self.device = f"{info.get('device', 'GPU')} [{info.get('backend_type')}]"
        sizes = [problem.n*replicas*4]*2+[self.meta_rows*replicas*4, table.nbytes]
        required_size = max(*sizes,(problem.n+1)*replicas*4)
        if required_size > min(adapter.limits['max-buffer-size'], adapter.limits['max-storage-buffer-binding-size']):
            raise ValueError('Sampling replica count exceeds this GPU buffer limit.')
        self.gpu = adapter.request_device_sync(required_limits={'max-buffer-size': required_size,
                    'max-storage-buffer-binding-size': required_size, 'max-storage-buffers-per-shader-stage': 4})
        self.queue = self.gpu.queue
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self.buffers = [self.gpu.create_buffer(size=size, usage=usage) for size in sizes]
        self.queue.write_buffer(self.buffers[2], 0, np.random.default_rng(seed).integers(1, 2**32, replicas, dtype=np.uint32))
        self.queue.write_buffer(self.buffers[3], 0, table)
        self.params = self.gpu.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        module = self.gpu.create_shader_module(code=(ROOT/('sampling_round24.wgsl' if self.flags else 'sampling_kernels.wgsl')).read_text())
        self.pipeline = self.gpu.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'sample_search'})
        self.exchange_pipeline = self.gpu.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'replica_exchange'}) if self.flags&4 else None
        self.gather_pipeline = self.gpu.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'gather_best'})
        self.gather_buffer, self.gather_capacity, self.gather_group = None, 0, None
        entries = [{'binding': i, 'resource': {'buffer': buffer}} for i, buffer in enumerate(self.buffers)]
        entries.append({'binding': 4, 'resource': {'buffer': self.params}})
        self.group = self.gpu.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
        self.exchange_group=self.gpu.create_bind_group(layout=self.exchange_pipeline.get_bind_group_layout(0),entries=[entries[i] for i in (0,1,2,4)]) if self.exchange_pipeline is not None else None
        self.launch(0, 0)
    def launch(self, steps, phase=1):
        self.queue.write_buffer(self.params, 0, np.array([self.n,self.r,self.f,steps,phase,self.e,self.gold_total,self.flags],np.uint32))
        encoder = self.gpu.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.pipeline); compute.set_bind_group(0, self.group)
        compute.dispatch_workgroups((self.r+63)//64); compute.end()
        self.queue.submit([encoder.finish()])
        self.queue.read_buffer(self.buffers[2], 0, 4)  # Tiny synchronization fence.
    def top(self, count=32):
        scores = np.frombuffer(self.queue.read_buffer(self.buffers[2], 2*self.r*4, self.r*4), np.uint32).copy()
        ids = _top_ids(scores, count)
        # One dispatch gathers selected lanes, instead of n*count host copy commands.
        count = len(ids)
        if count > self.gather_capacity:
            if self.gather_buffer is not None: self.gather_buffer.destroy()
            self.gather_buffer = self.gpu.create_buffer(size=count*(self.n+1)*4,
                usage=self.wgpu.BufferUsage.STORAGE | self.wgpu.BufferUsage.COPY_SRC | self.wgpu.BufferUsage.COPY_DST)
            self.gather_capacity = count
            self.gather_group = self.gpu.create_bind_group(layout=self.gather_pipeline.get_bind_group_layout(0), entries=[
                {'binding':1,'resource':{'buffer':self.buffers[1]}},
                {'binding':4,'resource':{'buffer':self.params}},
                {'binding':5,'resource':{'buffer':self.gather_buffer}}])
        if not count: return np.empty((0,self.n),int),scores[ids].astype(int)
        self.queue.write_buffer(self.gather_buffer,0,ids.astype(np.uint32))
        self.queue.write_buffer(self.params,0,np.array([self.n,self.r,self.f,count,0,self.e,0,0],np.uint32))
        encoder = self.gpu.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.gather_pipeline);compute.set_bind_group(0,self.gather_group)
        compute.dispatch_workgroups((count*self.n+63)//64);compute.end()
        self.queue.submit([encoder.finish()])
        boards = np.frombuffer(self.queue.read_buffer(self.gather_buffer,count*4,count*self.n*4),np.uint32).reshape(count,self.n).astype(int)
        return boards, scores[ids].astype(int)
    def exchange(self, parity=0):
        if self.exchange_pipeline is None:return
        self.queue.write_buffer(self.params,0,np.array([self.n,self.r,self.f,0,parity,self.e,self.gold_total,self.flags],np.uint32))
        encoder=self.gpu.create_command_encoder();compute=encoder.begin_compute_pass()
        compute.set_pipeline(self.exchange_pipeline);compute.set_bind_group(0,self.exchange_group)
        compute.dispatch_workgroups((self.r+127)//128);compute.end();self.queue.submit([encoder.finish()])
    def experiment_stats(self):
        if not self.flags:return {}
        counts=np.frombuffer(self.queue.read_buffer(self.buffers[2],13*self.r*4,5*self.r*4),np.uint32).reshape(5,self.r).sum(axis=1,dtype=np.uint64)
        return dict(zip(('accepted_moves','tabu_rejections','chain_proposals','exchange_attempts','exchange_accepts'),map(int,counts)))
    def close(self):
        if self.gather_buffer is not None: self.gather_buffer.destroy()
        for buffer in self.buffers: buffer.destroy()
        self.params.destroy(); self.gpu.destroy()


def sampling_flags(experiments=None):
    experiments={} if experiments is None else experiments
    names={'ejection_chains':1,'move_tabu':2,'replica_exchange':4}
    if not isinstance(experiments,dict) or set(experiments)-set(names) or any(type(value) is not bool for value in experiments.values()):
        raise ValueError('Invalid sampling experiment switches.')
    return sum(bit for name,bit in names.items() if experiments.get(name,False))

def _make_sampler(problem, table, free, replicas, seed, backend, *, allow_software=False, experiments=None):
    extra={'experiments':experiments} if sampling_flags(experiments) else {}
    if backend == 'cuda': return _CUDA(problem, table, free, replicas, seed,**extra)
    return _WebGPU(problem, table, free, replicas, seed, allow_software=allow_software,**extra)


def sample(problem, *, seconds, seed=0, replicas=128, backend='auto', stop=None, progress=None, gold_aware=False, experiments=None):
    began = time.monotonic()
    flags=sampling_flags(experiments)
    if isinstance(seconds, bool) or not isinstance(seconds, (int,float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative.')
    if type(replicas) is not int or not 1 <= replicas <= 131072 or type(seed) is not int or seed < 0:
        raise ValueError('replicas must be 1..131072 and seed a nonnegative integer.')
    if backend not in ('auto', 'cuda', 'webgpu'): raise ValueError('Unknown sampling backend.')
    result = {'status': 'timeout', 'backend': backend, 'device': None, 'replicas': replicas,
              'initial_boards': 0, 'move_proposals': 0, 'samples_considered': 0,
              'unique_boards': None, 'duplicates_tracked': False, 'complete': False,
              'top_board': None, 'best_score': None, 'top_samples': [],
              'archive_capacity':8, 'archive_checks':0, 'archive_repeated_observations':0, 'archive_unique_boards':0,
              'hint_scope': 'Value order only; explicit fixed cells/tile ownership preserved; domains and seams may be violated.'}
    def finish():
        result['elapsed_seconds'] = time.monotonic()-began
        return result
    if stop is not None and stop(): result['status']='stopped'; return finish()
    if seconds == 0: return finish()
    if type(gold_aware) is not bool:raise ValueError('gold_aware must be boolean.')
    inputs = _inputs(problem, gold_aware=gold_aware)
    result['gold_aware'] = gold_aware
    if inputs is None:
        result.update(status='skipped', reason='Explicit fixed cells reuse a physical tile; exact DFS will decide feasibility.')
        return finish()
    engine = None
    archive = []
    def collect():
        nonlocal archive
        boards,scores = engine.top()
        verification = _verification_view(problem, len(boards))
        checked = []
        observed = {tuple(item['codes']) for item in archive}
        for codes,objective in zip(boards,scores):
            metrics = gold_score(problem,codes) if gold_aware else score_board(verification,codes)
            if metrics['gold_objective' if gold_aware else 'objective'] != int(objective):
                raise RuntimeError('GPU sample failed independent CPU scoring.')
            values = codes.astype(int).tolist()
            key = tuple(values)
            result['archive_checks'] += 1
            if key in observed: result['archive_repeated_observations'] += 1
            observed.add(key)
            checked.append({'codes':values,**metrics})
        archive = diverse_archive([*archive,*checked],capacity=result['archive_capacity'], rank_key='gold_objective' if gold_aware else 'objective')
        if archive:
            result.update(top_samples=archive,top_board=archive[0]['codes'],best_score=archive[0]['matched_edges'],
                          best_objective=archive[0]['objective'],domain_violations=archive[0]['domain_violations'],
                          archive_unique_boards=len(archive))
    try:
        selected = choose_backend(backend)
        if stop is not None and stop(): result['status']='stopped'; return finish()
        if time.monotonic()-began >= seconds: return finish()
        engine = _make_sampler(problem, *inputs, replicas, seed, selected,**({'experiments':experiments} if flags else {}))
        result.update(backend=selected, device=engine.device, initial_boards=replicas, samples_considered=replicas)
        last_progress = time.monotonic();launches=0
        while time.monotonic()-began < seconds and inputs[1]:
            if stop is not None and stop(): result['status']='stopped'; break
            engine.launch(64);launches+=1
            if flags&4 and launches%8==0:engine.exchange((launches//8)%2)
            result['move_proposals'] += replicas*64
            result['samples_considered'] += replicas*64
            if time.monotonic()-last_progress >= 1:
                collect()
                last_progress = time.monotonic()
                if progress is not None:
                    progress({'event':'sampling_progress', **result, 'elapsed_seconds':last_progress-began})
        collect()
        if flags:result['sampling_experiments']={**experiments,**engine.experiment_stats()}
        if stop is not None and stop(): result['status']='stopped'
        elif not inputs[1]: result['status']='sampled'
        if progress is not None: progress({'event':'sampling', **result, 'elapsed_seconds':time.monotonic()-began})
        return finish()
    finally:
        if engine is not None: engine.close()
        result['elapsed_seconds'] = time.monotonic()-began


def solve(problem, *, seconds=10, seed=0, target='gold', replicas=128, backend='auto',
          sampling_fraction=0.2, repair_fraction=0.3, hints=None, seed_board=None,
          stop=None, progress=None, resume=None, learning_state=None, options=None):
    """Bounded GPU sampling, bounded CPU repair, then the full exact DFS tree.

    A saved full-board hint competes with GPU samples rather than being replaced
    by a worse sample. Heuristic budgets are capped at 30+60 seconds even for a
    very long run; construction/native return can overrun cooperative budgets.
    Resume skips both phases and preserves the exact DFS stack.
    """
    began = time.monotonic()
    from constraint_dfs import solve as exact_solve
    from constraint_lns import repair
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative.')
    if target not in ('gold','edges') or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in (sampling_fraction,repair_fraction)):
        raise ValueError('Invalid target or heuristic fractions.')
    if sampling_fraction+repair_fraction > 1:
        raise ValueError('Heuristic fractions must sum to at most one.')
    sampling_report, repair_report, checked = None, None, 0
    was_resume = resume is not None
    def remaining(): return max(0.0, seconds-(time.monotonic()-began))
    def emit_phase(phase, **extra):
        if progress is not None:
            progress({'event':'phase', 'phase':phase, 'elapsed_seconds':time.monotonic()-began, **extra})
    def phase_progress(phase):
        def notify(event):
            if progress is not None:
                progress({**event, 'phase':phase, 'phase_elapsed_seconds':event.get('elapsed_seconds',0),
                          'elapsed_seconds':time.monotonic()-began})
        return notify
    if was_resume and isinstance(resume,dict) and 'hybrid_version' in resume:
        if resume['hybrid_version'] not in (1,2): raise ValueError('Unsupported sampling/DFS checkpoint version.')
        sampling_report, repair_report, resume = resume.get('sampling'), resume.get('repair'), resume['dfs']
        if resume is None: raise ValueError('Hybrid resume is missing its exact DFS stack.')
    candidates = []
    if not was_resume:
        if seed_board is not None:
            score_board(problem, seed_board)  # Explicit seeds fail clearly if malformed.
            candidates.append(list(map(int,seed_board)))
        if hints is not None and not isinstance(hints,dict):
            try:
                score_board(problem,hints)
                candidates.append(list(map(int,hints)))
            except (ValueError,TypeError):
                pass  # General value-order hints need not form a complete board.
    def certificate(codes, metrics, phase):
        nonlocal checked
        if metrics['domain_violations'] or metrics['matched_edges'] != len(problem.edges): return None
        validation = problem.validate(codes)
        checked += 1
        if not validation['valid'] and not (target == 'edges' and validation['unique_tiles'] and validation['matched_edges'] == len(problem.edges)):
            return None
        if progress is not None:
            progress({'event':'candidate', 'phase':phase, 'codes':codes, 'validation':validation,
                      'elapsed_seconds':time.monotonic()-began, 'nodes':0})
        return {'status':'solved' if validation['valid'] else 'edge_perfect',
                'codes':codes, 'validation':validation, 'complete':True, 'nodes':0,
                'stats':{'gpu_candidates_checked':checked}, 'dfs_elapsed_seconds':0.0,
                'elapsed_seconds':time.monotonic()-began, 'gpu_sampling':sampling_report,
                'local_repair':repair_report, 'method':'gpu_sampling_dfs', 'phase':phase}
    # Revalidate saved candidates before allocating a GPU or moving any tiles.
    for codes in candidates if seconds > 0 and not (stop is not None and stop()) else []:
        found = certificate(codes,score_board(problem,codes),'saved_hint')
        if found is not None: return found
    if not was_resume and seconds and not (stop is not None and stop()):
        emit_phase('sampling', budget_seconds=min(30.0,seconds*sampling_fraction,remaining()))
        sampling_report = sample(problem, seconds=min(30.0,seconds*sampling_fraction,remaining()), seed=seed,
                                 replicas=replicas, backend=backend, stop=stop, progress=phase_progress('sampling'),
                                 **({'gold_aware':True} if options and options.get('gpu_gold_sampling') and target=='gold' else {}),
                                 **({'experiments':{name:True for name in ('ejection_chains','move_tabu','replica_exchange') if options.get(name)}} if options and any(options.get(name) for name in ('ejection_chains','move_tabu','replica_exchange')) else {}))
        samples = sampling_report.get('top_samples', [])
        if not samples and sampling_report.get('top_board') is not None:
            samples = [{'codes':sampling_report['top_board']}]
        candidates.extend(candidate['codes'] for candidate in samples)
    archive = diverse_archive([{'codes':codes,**score_board(problem,codes)} for codes in candidates])
    ranked = [(item['objective'],item['codes']) for item in archive]
    for _, codes in ranked if seconds > 0 else []:
        found = certificate(codes,score_board(problem,codes),'sampling')
        if found is not None: return found
    if ranked:
        hints = list(ranked[0][1])
    if not was_resume and ranked and remaining() > 0 and not (stop is not None and stop()):
        budget = min(60.0,seconds*repair_fraction,remaining())
        emit_phase('repair', budget_seconds=budget)
        repair_report = repair(problem, hints, seconds=budget, seed=seed, target=target,
                               stop=stop, progress=phase_progress('repair'),
                               candidate_boards=[item['codes'] for item in archive[1:]],
                               **({'dependency_chains':True} if options and options.get('dependency_repair') else {}),
                               **({'experiments':{name:True for name in ('core_repair','loop_trades','fragment_crossover','repair_learning') if options.get(name)}} if options and any(options.get(name) for name in ('core_repair','loop_trades','fragment_crossover','repair_learning')) else {}))
        # Repair cores are scoped claims. Independent mask verification is
        # required before they can prune exact DFS or enter its saved store.
        if target=='gold' and learning_state is not None and options and options.get('smart_learning'):
            from constraint_mask_conflicts import VerifiedMaskConflictStore
            store=learning_state.get('_mask_conflicts')
            if store is None:
                store=VerifiedMaskConflictStore(problem,target,proof_transport=options.get('proof_transport',False),resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False))
                learning_state['_mask_conflicts']=store
            store.configure_policy(proof_transport=options.get('proof_transport',False),
                resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False))
            offered=0
            for trial in repair_report.get('experimental_repair',[]):
                payload=trial.get('exclusions',{})
                if payload.get('kind')!='gold_repair_assumption_cores' or payload.get('fingerprint')!=problem.fingerprint or payload.get('domains')!=[list(row) for row in problem.domains]:continue
                for core in payload.get('cores',[]):
                    if all(code in problem.domains[cell] for cell,code in core):
                        offered+=bool(store.offer(tuple((cell,1<<code) for cell,code in core)))
            repair_report['claims_queued_for_independent_proof']=offered
        if repair_report.get('best_board') is not None:
            improved = repair_report['best_board']
            if score_board(problem,improved)['objective'] >= score_board(problem,hints)['objective']:
                hints = improved
            found = certificate(improved,score_board(problem,improved),'repair')
            if found is not None: return found
    emit_phase('exact_dfs', budget_seconds=remaining(), resumed=was_resume)
    kwargs = {'seconds':remaining(), 'seed':seed, 'target':target, 'hints':hints,
              'stop':stop, 'progress':phase_progress('exact_dfs'), 'resume':resume, 'options':options}
    if learning_state is not None: kwargs['learning_state'] = learning_state
    result = exact_solve(problem, **kwargs)
    result['stats']['gpu_candidates_checked'] = checked
    result['dfs_elapsed_seconds'] = result['elapsed_seconds']
    result['elapsed_seconds'] = time.monotonic()-began
    result['gpu_sampling'], result['local_repair'] = sampling_report, repair_report
    result['method'], result['phase'] = 'gpu_sampling_dfs', 'exact_dfs'
    if 'checkpoint' in result:
        result['checkpoint'] = {'hybrid_version':2, 'dfs':result['checkpoint'],
                                'sampling':sampling_report, 'repair':repair_report,
                                'heuristics_complete':True}
    return result