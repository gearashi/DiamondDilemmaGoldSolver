"""Replay identical original puzzle states through rectangle and exact-span frame transports."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from bench_r35_parked_public import inputs
from classic_checkpoint_ranges import CheckpointRanges, _frame_windows, _selection
from classic_frame_packet import FramePacket
from dfs_gpu import DFS
from r35_frame_spans import FIELDS, FrameSpans, FULL, extents, merge_numpy, rectangle_wire_bytes
from search_problem import SearchProblem
from systematic_jobs import JobLedger


class MeteredRectangles(FramePacket):
    def _read(self, descriptor, width, cells):
        self.upload_bytes += descriptor.nbytes + (640 if self.web else 0)
        return super()._read(descriptor, width, cells)


def legacy(old, transport, marks, lanes, reset):
    n = old[0].shape[1]
    _, _, lo, hi = extents(np.where(reset,FULL,marks))
    selected = hi > 0
    low = np.zeros(n,np.int16)
    high = low.copy()
    low[lanes], high[lanes] = lo,hi - 1
    windows = _frame_windows(lanes[selected], low, high, np.zeros(n,bool))
    if not windows:
        return old, dict(upload=0,read=0,maps=0)
    transport.upload_bytes=0
    cells=sum((b-a)*len(cols) for a,b,cols in windows)
    if len(windows)>1 or cells>=1024:
        arrays=transport.capture(windows)
    else:
        a,b,cols=windows[0]
        gpu=transport.gpu
        arrays=[]
        before=getattr(gpu,'readback_bytes',0)
        width=int(cols[-1])-int(cols[0])+1
        if transport.web:
            from dfs_webgpu import PACKED_READ_MIN_ROWS
            if width>2*len(cols) and width*(b-a)>=2048 or width<gpu.n and b-a>=PACKED_READ_MIN_ROWS:
                transport.upload_bytes=4*(20+(4*len(cols) if width!=len(cols) else 0))
        elif width!=len(cols):
            transport.upload_bytes=4*cols.nbytes
        for index,name in enumerate(FIELDS,1):
            if transport.web:
                part=gpu._read_rows(gpu.buffers[index],np.arange(a,b),cols).astype(np.uint16)
            else:
                part=getattr(gpu,name)[a:b,_selection(cols)].get()
            arrays.append(part.ravel())
        transport.read_bytes=(gpu.readback_bytes-before) if transport.web else cells*8
        transport.maps=4
    a,b,firstcols=windows[0]
    if len(firstcols)==n:
        result=tuple(np.empty_like(arr) for arr in old)
        for dst,src in zip(result,old):
            dst[:a]=src[:a]
            dst[b:]=src[b:]
    else:
        result=tuple(arr.copy() for arr in old)
    offset = 0
    for a,b,ids in windows:
        count = (b-a)*len(ids)
        sl = _selection(ids)
        for target, values in zip(result,arrays):
            cells = values[offset:offset+count].reshape(b-a,len(ids))
            if not isinstance(sl,slice) and len(ids)>=1024:
                for row,src in zip(target[a:b],cells):
                    row[sl]=src
            else:
                target[a:b,sl]=cells
        offset += count
    assert rectangle_wire_bytes(transport.gpu,windows)==transport.upload_bytes+transport.read_bytes
    return result,dict(upload=transport.upload_bytes,read=transport.read_bytes,maps=transport.maps)


def trial(p, n, used, api, steps, repeats, scattered=False, native=None, dynamic=False):
    args, prefixes, sizes = inputs(p,used,False)
    if dynamic: args['gpu_features']={'dynamic_mrv':True,'forced':True}
    if api == 'CUDA':
        gpu = DFS(**args,n=n,prefixes=[],seed=35068)
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        adapter = next(a for a in wgpu.gpu.enumerate_adapters_sync()
                       if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
        gpu=WebGPUDFS(**args,n=n,prefixes=[],seed=35068,adapter=adapter)
    ledger=JobLedger.empty(len(prefixes),n)
    if scattered:
        lanes=np.linspace(0,n-1,len(prefixes),dtype=np.int32)
        gpu.load_prefixes([row[:size] for row,size in zip(prefixes,sizes)],lanes=lanes)
        ledger.ids[lanes]=np.arange(len(prefixes))
        ledger.cursor=len(prefixes)
        ledger.validate(gpu.states.get())
    else:
        ledger.refill(gpu,prefixes,sizes)
    reader=CheckpointRanges(gpu,ledger,board_mode='ranges',trace=True)
    prior=reader.capture()
    spans=FrameSpans(gpu,merge=merge_numpy)
    packed=MeteredRectangles(gpu)
    result=[]
    try:
        for budget in steps:
            left=budget
            while left:
                nodes=min(512,left)
                gpu.step(nodes)
                ledger.refill(gpu,prefixes,sizes)
                left-=nodes
            expected=gpu.snapshot_checkpoint()
            for lane in np.flatnonzero(expected['states']==1):
                codes=list(map(int,expected['boards'][:,lane]))
                gold=p.validate(codes)
                if gold['valid']:
                    (Path(__file__).parent/'audit/r35/checkpoint-gold.json').write_text(
                        json.dumps({'codes':codes,'validation':gold}),encoding='utf8')
                    raise SystemExit('Verified Gold saved; search stopped.')
            lanes=np.flatnonzero((prior['states']<=1)|ledger.checkpoint_installs)
            reset=ledger.checkpoint_installs[lanes]
            assert gpu._frame_mark_owner is reader.trace_owner
            mark_before=getattr(gpu,"readback_bytes",0)
            if not len(lanes):
                marks=np.empty(0,np.uint32)
            elif api=='CUDA':
                marks=reader._read('_frame_marks',lanes)
            else:
                marks=reader._physical(5,12,13,lanes)[0]
            old=tuple(prior[f] for f in FIELDS)
            a,b,c,d=extents(np.where(reset,FULL,marks))
            metrics=dict(steps=budget,lanes=len(lanes),installed=int(reset.sum()),
                         initialized=int((b-a).sum()),cursor=int((d-c).sum()),
                         marks=(gpu.readback_bytes-mark_before if api!='CUDA' else marks.nbytes))
            for mode in ('rectangle','span','native','bands','bands0','bands4096','fringe','fringe0','fringe4096'):
                if mode=='native' and native is None: continue
                if mode.startswith(('bands','fringe')):
                    try:
                        from bench_r35_frame_bands import FrameBands
                    except ImportError:
                        continue
                    if not hasattr(spans, mode):
                        penalty=0 if mode.endswith('0') else 4096 if mode.endswith('4096') else 256
                        setattr(spans,mode,FrameBands(gpu,penalty=penalty,fringe=mode.startswith('fringe')))
                times=[]
                for attempt in range(repeats+1):
                    begin=perf_counter()
                    if mode=='rectangle':
                        answer,traffic=legacy(old,packed,marks,lanes,reset)
                    elif mode.startswith(('bands','fringe')):
                        obj=getattr(spans,mode)
                        answer=obj.capture(old,lanes,marks,reset)
                        traffic=dict(upload=obj.upload_bytes,read=obj.read_bytes,maps=obj.maps)
                    else:
                        if mode=='native':
                            if not hasattr(spans,'fast'): spans.fast=FrameSpans(gpu,merge=native)
                            obj=spans.fast
                        else:
                            obj=spans
                        answer=obj.capture(old,lanes,marks,reset)
                        traffic=dict(upload=obj.upload_bytes,read=obj.read_bytes,maps=obj.maps)
                    elapsed=(perf_counter()-begin)*1000
                    assert all(np.array_equal(src,expected[f]) for f,src in zip(FIELDS,answer)),(mode,n,used,budget)
                    if attempt: times.append(elapsed)
                metrics[mode]=dict(**traffic,median_ms=round(median(times),4))
            result.append(metrics)
            reader._arm_trace(False)
            prior=expected
            ledger.checkpoint_installs.fill(False)
    finally:
        if hasattr(gpu,'close'): gpu.close()
    return result


if __name__=='__main__':
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('--api',choices=('CUDA','Vulkan','D3D12','Metal'),required=True)
    cmd.add_argument('--tiles',type=Path,required=True)
    cmd.add_argument('--output',type=Path,required=True)
    cmd.add_argument('--steps',nargs='+',type=int,default=[1,8,64,512,4096])
    cmd.add_argument('--shapes',nargs='+',default=['128/128','2048/128','2048/2048','32768/32768'])
    cmd.add_argument('--repeats',type=int,default=3)
    cmd.add_argument('--native',type=Path)
    cmd.add_argument('--dynamic',action='store_true')
    cli=cmd.parse_args()
    fast=None
    if cli.native:
        import ctypes
        from r35_frame_span_decode import native_merge
        fast=native_merge(ctypes.CDLL(str(cli.native.resolve())))
    p=SearchProblem(json.loads(cli.tiles.read_bytes()))
    out={}
    for shape in cli.shapes:
        n,k=map(int,shape.split('/'))
        out[shape]=trial(p,n,k,cli.api,cli.steps,cli.repeats,k<n,fast,cli.dynamic)
        cli.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
