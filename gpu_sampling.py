"""Bounded heuristic GPU sampling followed by complete, strongly pruned CPU DFS.

Sampling preserves physical-tile uniqueness and explicit fixed placements. Its
boards may violate seams/domains, may repeat, and are only value-order hints.
GPU time includes imports, allocation, compilation, and CPU verification. There
is no CPU sampling fallback, and sample failure never establishes infeasibility.
"""
from __future__ import annotations
import math
from pathlib import Path
import time
import numpy as np
from gpu_backends import choose_backend
from validator import reverse_mask

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
__device__ int local(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int a,int d,int lane){
 int value=legal(t,b,n,r,f,a,lane)*(edges+1);for(int s=0;s<3;s++)value+=edge(t,b,n,r,a,s,lane);
 if(d!=a){value+=legal(t,b,n,r,f,d,lane)*(edges+1);for(int s=0;s<3;s++)if(t[18*n+d*3+s]!=a)value+=edge(t,b,n,r,d,s,lane);}return value;
}
extern "C" __global__ void sample_search(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int steps,int phase,int edges){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;unsigned random=m[lane];
 if(phase==0){
 for(int c=0;c<n;c++)b[c*r+lane]=t[24*n+c];
 for(int count=f;count>1;count--){int a=t[25*n+count-1],d=t[25*n+rnd(random)%count];unsigned old=b[a*r+lane];b[a*r+lane]=b[d*r+lane];b[d*r+lane]=old;}
 for(int i=0;i<f;i++){int c=t[25*n+i];b[c*r+lane]=3*(b[c*r+lane]/3)+rnd(random)%3;}
 m[r+lane]=m[2*r+lane]=total(t,b,n,r,f,edges,lane);m[3*r+lane]=0;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
 }else if(f>0){for(int step=0;step<steps;step++){
 int a=t[25*n+rnd(random)%f],d=t[25*n+rnd(random)%f];unsigned ca=b[a*r+lane],cd=b[d*r+lane];
 int before=local(t,b,n,r,f,edges,a,d,lane);b[a*r+lane]=3*(cd/3)+rnd(random)%3;if(a!=d)b[d*r+lane]=3*(ca/3)+rnd(random)%3;
 int delta=local(t,b,n,r,f,edges,a,d,lane)-before;float heat=.25f+1.25f*(1.f-float(m[3*r+lane]%4096)/4096.f);
 bool accept=delta>=0||float(rnd(random)%1000000)<1000000.f*expf(float(delta)/heat);m[3*r+lane]++;
 if(accept){m[r+lane]=int(m[r+lane])+delta;if(m[r+lane]>m[2*r+lane]||(m[r+lane]==m[2*r+lane]&&rnd(random)%64==0)){m[2*r+lane]=m[r+lane];for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}}
 else{b[a*r+lane]=ca;b[d*r+lane]=cd;}
 }}m[lane]=random;
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
    def top(self, count=4):
        scores = self.meta[2].get()
        ids = np.argsort(scores, kind='stable')[-min(count, self.r):][::-1].copy()
        return self.best[:, ids].get().T.astype(int), scores[ids].astype(int)
    def close(self):
        self.boards = self.best = self.meta = self.table = self.kernel = None


class _WebGPU:
    def __init__(self, problem, table, free, replicas, seed):
        import wgpu
        self.wgpu, self.n, self.r, self.f, self.e = wgpu, problem.n, replicas, free, len(problem.edges)
        adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance', force_fallback_adapter=False)
        if adapter is None: raise RuntimeError('No native GPU adapter is available for sampling.')
        info = dict(adapter.info)
        text = ' '.join(map(str, info.values())).lower()
        if (str(info.get('adapter_type')).lower() in ('cpu', 'software')
                or any(term in text for term in ('llvmpipe', 'lavapipe', 'swiftshader', 'basic render'))
                or info.get('backend_type') not in ('Metal', 'Vulkan', 'D3D12')):
            raise RuntimeError('Sampling requires a native hardware GPU; software/OpenGL fallback is disabled.')
        self.device = f"{info.get('device', 'GPU')} [{info.get('backend_type')}]"
        sizes = [problem.n*replicas*4]*2+[4*replicas*4, table.nbytes]
        if max(sizes) > min(adapter.limits['max-buffer-size'], adapter.limits['max-storage-buffer-binding-size']):
            raise ValueError('Sampling replica count exceeds this GPU buffer limit.')
        self.gpu = adapter.request_device_sync(required_limits={'max-buffer-size': max(sizes),
                    'max-storage-buffer-binding-size': max(sizes), 'max-storage-buffers-per-shader-stage': 4})
        self.queue = self.gpu.queue
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self.buffers = [self.gpu.create_buffer(size=size, usage=usage) for size in sizes]
        self.queue.write_buffer(self.buffers[2], 0, np.random.default_rng(seed).integers(1, 2**32, replicas, dtype=np.uint32))
        self.queue.write_buffer(self.buffers[3], 0, table)
        self.params = self.gpu.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        module = self.gpu.create_shader_module(code=(ROOT/'sampling_kernels.wgsl').read_text())
        self.pipeline = self.gpu.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'sample_search'})
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
    def top(self, count=4):
        scores = np.frombuffer(self.queue.read_buffer(self.buffers[2], 2*self.r*4, self.r*4), np.uint32).copy()
        ids = np.argsort(scores, kind='stable')[-min(count, self.r):][::-1].copy()
        scratch = self.gpu.create_buffer(size=len(ids)*self.n*4,
                  usage=self.wgpu.BufferUsage.COPY_SRC | self.wgpu.BufferUsage.COPY_DST)
        encoder = self.gpu.create_command_encoder()
        for index, lane in enumerate(ids):
            for cell in range(self.n):
                encoder.copy_buffer_to_buffer(self.buffers[1], (cell*self.r+int(lane))*4,
                                               scratch, (index*self.n+cell)*4, 4)
        self.queue.submit([encoder.finish()])
        boards = np.frombuffer(self.queue.read_buffer(scratch), np.uint32).reshape(len(ids),self.n).astype(int)
        scratch.destroy()
        return boards, scores[ids].astype(int)
    def close(self):
        for buffer in self.buffers: buffer.destroy()
        self.params.destroy(); self.gpu.destroy()


def _make_sampler(problem, table, free, replicas, seed, backend):
    return (_CUDA if backend == 'cuda' else _WebGPU)(problem, table, free, replicas, seed)


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
    try:
        selected = choose_backend(backend)
        if stop is not None and stop(): result['status']='stopped'; return finish()
        if time.monotonic()-began >= seconds: return finish()
        engine = _make_sampler(problem, *inputs, replicas, seed, selected)
        result.update(backend=selected, device=engine.device, initial_boards=replicas, samples_considered=replicas)
        while time.monotonic()-began < seconds and inputs[1]:
            if stop is not None and stop(): result['status']='stopped'; break
            engine.launch(64)
            result['move_proposals'] += replicas*64
            result['samples_considered'] += replicas*64
        boards, scores = engine.top()
        checked = []
        for codes, objective in zip(boards, scores):
            metrics = score_board(problem, codes)
            if metrics['objective'] != int(objective):
                raise RuntimeError('GPU sample failed independent CPU scoring.')
            checked.append({'codes': codes.astype(int).tolist(), **metrics})
        checked.sort(key=lambda item: item['objective'], reverse=True)
        if checked:
            result.update(top_samples=checked, top_board=checked[0]['codes'], best_score=checked[0]['matched_edges'],
                          best_objective=checked[0]['objective'], domain_violations=checked[0]['domain_violations'])
        if stop is not None and stop(): result['status']='stopped'
        elif not inputs[1]: result['status']='sampled'
        if progress is not None: progress({'event':'sampling', **result, 'elapsed_seconds':time.monotonic()-began})
        return finish()
    finally:
        if engine is not None: engine.close()
        result['elapsed_seconds'] = time.monotonic()-began


def solve(problem, *, seconds=10, seed=0, target='gold', replicas=128, backend='auto',
          sampling_fraction=0.2, hints=None, stop=None, progress=None, resume=None):
    """Sample for an initial fraction, then search the full legal CPU DFS tree."""
    from constraint_dfs import solve as exact_solve
    began = time.monotonic()
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative.')
    if target not in ('gold','edges') or not 0 <= sampling_fraction <= 1:
        raise ValueError('Invalid target or sampling_fraction.')
    report = None
    if resume is not None and isinstance(resume,dict) and 'hybrid_version' in resume:
        if resume['hybrid_version'] != 1: raise ValueError('Unsupported sampling/DFS checkpoint version.')
        report, resume = resume.get('sampling'), resume['dfs']
    if resume is None and seconds and not (stop is not None and stop()):
        report = sample(problem, seconds=seconds*sampling_fraction, seed=seed,
                        replicas=replicas, backend=backend, stop=stop, progress=progress)
        if report['top_board'] is not None: hints = report['top_board']
    # A found certificate remains useful even when cold GPU startup used the
    # time budget. Never promote an edge-perfect multi-loop sample to Gold.
    checked = 0
    if resume is None and report is not None:
        candidates = report.get('top_samples', [])
        if not candidates and report.get('top_board') is not None:
            candidates = [{'codes': report['top_board']}]
        for candidate in candidates:
            codes = candidate['codes']
            metrics = score_board(problem, codes)
            if metrics['domain_violations'] or metrics['matched_edges'] != len(problem.edges):
                continue
            validation = problem.validate(codes)
            checked += 1
            if validation['valid'] or target == 'edges' and validation['unique_tiles'] and validation['matched_edges'] == len(problem.edges):
                if progress is not None:
                    progress({'event':'candidate', 'codes':codes, 'validation':validation,
                              'elapsed_seconds':time.monotonic()-began, 'nodes':0})
                return {'status':'solved' if validation['valid'] else 'edge_perfect',
                        'codes':codes, 'validation':validation, 'complete':True, 'nodes':0,
                        'stats':{'gpu_candidates_checked':checked}, 'dfs_elapsed_seconds':0.0,
                        'elapsed_seconds':time.monotonic()-began, 'gpu_sampling':report,
                        'method':'gpu_sampling_dfs'}
    result = exact_solve(problem, seconds=max(0.0,seconds-(time.monotonic()-began)), seed=seed,
                         target=target, hints=hints, stop=stop, progress=progress, resume=resume)
    result['stats']['gpu_candidates_checked'] = checked
    result['dfs_elapsed_seconds'] = result['elapsed_seconds']
    result['elapsed_seconds'] = time.monotonic()-began
    result['gpu_sampling'] = report
    result['method'] = 'gpu_sampling_dfs'
    if 'checkpoint' in result:
        result['checkpoint'] = {'hybrid_version':1, 'dfs':result['checkpoint'], 'sampling':report}
    return result
