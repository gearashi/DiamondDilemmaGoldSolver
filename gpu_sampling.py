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
extern "C" __global__ void sample_search(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int steps,int phase,int edges){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;unsigned random=m[lane];
 if(phase==0){
 for(int c=0;c<n;c++)b[c*r+lane]=t[24*n+c];
 for(int count=f;count>1;count--){int a=t[25*n+count-1],d=t[25*n+rnd(random)%count];unsigned old=b[a*r+lane];b[a*r+lane]=b[d*r+lane];b[d*r+lane]=old;}
 for(int i=0;i<f;i++){int c=t[25*n+i];b[c*r+lane]=3*(b[c*r+lane]/3)+rnd(random)%3;}
 m[r+lane]=m[2*r+lane]=total(t,b,n,r,f,edges,lane);m[3*r+lane]=0;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
 }else if(f>0){
 int score=m[r+lane],bestscore=m[2*r+lane];unsigned iteration=m[3*r+lane];
 for(int step=0;step<steps;step++){
 int a=t[25*n+rnd(random)%f],d=t[25*n+rnd(random)%f];unsigned ca=b[a*r+lane],cd=b[d*r+lane];
 unsigned na=3*(cd/3)+rnd(random)%3,nd=cd;if(a!=d)nd=3*(ca/3)+rnd(random)%3;
 int delta=change(t,b,n,r,f,edges,a,d,na,nd,lane);float heat=.25f+1.25f*(1.f-float(iteration%4096)/4096.f);
 bool accept=delta>=0||float(rnd(random)%1000000)<1000000.f*expf(float(delta)/heat);iteration++;
 if(accept){b[a*r+lane]=na;if(a!=d)b[d*r+lane]=nd;score+=delta;
 if(score>bestscore||(score==bestscore&&rnd(random)%64==0)){bestscore=score;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}}
 }m[r+lane]=score;m[2*r+lane]=bestscore;m[3*r+lane]=iteration;
 }m[lane]=random;
}
'''


def _inputs(problem):
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
    def __init__(self, problem, table, free, replicas, seed):
        import cupy as cp
        self.cp, self.n, self.r, self.f, self.e = cp, problem.n, replicas, free, len(problem.edges)
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        self.device = name.decode() if isinstance(name, bytes) else str(name)
        self.boards = cp.zeros((self.n, replicas), np.uint32)
        self.best = cp.zeros_like(self.boards)
        self.meta = cp.zeros((4, replicas), np.uint32)
        self.meta[0] = cp.asarray(np.random.default_rng(seed).integers(1, 2**32, replicas, dtype=np.uint32))
        self.table = cp.asarray(table)
        self.kernel = cp.RawKernel(CUDA_SOURCE, 'sample_search', options=('--std=c++17',))
        self.launch(0, 0)
    def launch(self, steps, phase=1):
        self.kernel(((self.r+127)//128,), (128,), (self.boards, self.best, self.meta, self.table,
                    *map(np.int32, (self.n, self.r, self.f, steps, phase, self.e))))
        self.cp.cuda.get_current_stream().synchronize()
    def top(self, count=32):
        scores = self.meta[2].get()
        ids = _top_ids(scores, count)
        return self.best[:, ids].get().T.astype(int), scores[ids].astype(int)
    def close(self):
        self.boards = self.best = self.meta = self.table = self.kernel = None


def _validate_sampling_adapter(info, *, allow_software=False):
    # Only finite tests pass this private opt-in. Runtime sample()/solve() and
    # the dashboard expose no software-fallback setting or environment switch.
    text = ' '.join(map(str, info.values())).lower()
    software = (str(info.get('adapter_type')).lower() in ('cpu', 'software')
                or any(term in text for term in ('llvmpipe', 'lavapipe', 'swiftshader', 'basic render')))
    if info.get('backend_type') not in ('Metal', 'Vulkan', 'D3D12') or (software and allow_software is not True):
        raise RuntimeError('Sampling requires a native hardware GPU; software/OpenGL fallback is disabled.')


class _WebGPU:
    def __init__(self, problem, table, free, replicas, seed, *, allow_software=False):
        import wgpu
        self.wgpu, self.n, self.r, self.f, self.e = wgpu, problem.n, replicas, free, len(problem.edges)
        adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance', force_fallback_adapter=False)
        if adapter is None: raise RuntimeError('No native GPU adapter is available for sampling.')
        info = dict(adapter.info)
        _validate_sampling_adapter(info, allow_software=allow_software)
        self.device = f"{info.get('device', 'GPU')} [{info.get('backend_type')}]"
        sizes = [problem.n*replicas*4]*2+[4*replicas*4, table.nbytes]
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
        module = self.gpu.create_shader_module(code=(ROOT/'sampling_kernels.wgsl').read_text())
        self.pipeline = self.gpu.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'sample_search'})
        self.gather_pipeline = self.gpu.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'gather_best'})
        self.gather_buffer, self.gather_capacity, self.gather_group = None, 0, None
        entries = [{'binding': i, 'resource': {'buffer': buffer}} for i, buffer in enumerate(self.buffers)]
        entries.append({'binding': 4, 'resource': {'buffer': self.params}})
        self.group = self.gpu.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
        self.launch(0, 0)
    def launch(self, steps, phase=1):
        self.queue.write_buffer(self.params, 0, np.array([self.n,self.r,self.f,steps,phase,self.e,0,0],np.uint32))
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
    def close(self):
        if self.gather_buffer is not None: self.gather_buffer.destroy()
        for buffer in self.buffers: buffer.destroy()
        self.params.destroy(); self.gpu.destroy()


def _make_sampler(problem, table, free, replicas, seed, backend, *, allow_software=False):
    if backend == 'cuda': return _CUDA(problem, table, free, replicas, seed)
    return _WebGPU(problem, table, free, replicas, seed, allow_software=allow_software)


def sample(problem, *, seconds, seed=0, replicas=128, backend='auto', stop=None, progress=None):
    began = time.monotonic()
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
    inputs = _inputs(problem)
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
            metrics = score_board(verification,codes)
            if metrics['objective'] != int(objective):
                raise RuntimeError('GPU sample failed independent CPU scoring.')
            values = codes.astype(int).tolist()
            key = tuple(values)
            result['archive_checks'] += 1
            if key in observed: result['archive_repeated_observations'] += 1
            observed.add(key)
            checked.append({'codes':values,**metrics})
        archive = diverse_archive([*archive,*checked],capacity=result['archive_capacity'])
        if archive:
            result.update(top_samples=archive,top_board=archive[0]['codes'],best_score=archive[0]['matched_edges'],
                          best_objective=archive[0]['objective'],domain_violations=archive[0]['domain_violations'],
                          archive_unique_boards=len(archive))
    try:
        selected = choose_backend(backend)
        if stop is not None and stop(): result['status']='stopped'; return finish()
        if time.monotonic()-began >= seconds: return finish()
        engine = _make_sampler(problem, *inputs, replicas, seed, selected)
        result.update(backend=selected, device=engine.device, initial_boards=replicas, samples_considered=replicas)
        last_progress = time.monotonic()
        while time.monotonic()-began < seconds and inputs[1]:
            if stop is not None and stop(): result['status']='stopped'; break
            engine.launch(64)
            result['move_proposals'] += replicas*64
            result['samples_considered'] += replicas*64
            if time.monotonic()-last_progress >= 1:
                collect()
                last_progress = time.monotonic()
                if progress is not None:
                    progress({'event':'sampling_progress', **result, 'elapsed_seconds':last_progress-began})
        collect()
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
                                 replicas=replicas, backend=backend, stop=stop, progress=phase_progress('sampling'))
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
                               candidate_boards=[item['codes'] for item in archive[1:]])
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