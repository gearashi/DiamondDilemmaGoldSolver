"""Field-selective row bands for an independent checkpoint-traffic comparison."""
from pathlib import Path
from contextlib import ExitStack

import numpy as np

from classic_checkpoint_ranges import _selection
from r35_frame_spans import extents, FULL, FIELDS


def _strips(ids, low, high, penalty):
    valid=high>low
    ids,low,high=ids[valid],low[valid],high[valid]
    if not len(ids): return []
    # Build an independent set of row breaks for each field group. The
    # original rectangle planner's fixed 4096 overhead can leave short
    # irregular windows very wide, so also price small and zero overhead.
    cuts, inverse=np.unique(np.r_[low,high],return_inverse=True)
    k=len(cuts)
    lo,hi=np.split(inverse,2)
    hist=np.bincount(lo*k+hi,minlength=k*k).reshape(k,k)
    sums=hist.cumsum(axis=0).cumsum(axis=1)
    cost=np.zeros(k,np.int64)
    parent=np.zeros(k,np.int16)
    for b in range(1,k):
        width=sums[b-1,-1]-sums[b-1,:b]
        price=cost[:b]+(cuts[b]-cuts[:b])*width+(width>0)*penalty
        parent[b]=np.argmin(price)
        cost[b]=price[parent[b]]
    bands=[]
    b=k-1
    while b:
        a=int(parent[b])
        start,stop=int(cuts[a]),int(cuts[b])
        selected=ids[(low<stop)&(high>start)]
        if len(selected): bands.append((start,stop,selected))
        b=a
    return bands[::-1]


def plan(n,lanes,marks,reset,penalty,fringe=False):
    lo,hi,cl,ch=extents(np.where(reset,FULL,marks))
    if fringe:
        groups=((_strips(lanes,lo,hi,penalty),(0,1,2,3)),
                (_strips(lanes,cl,np.where(hi==0,ch,lo),penalty),(2,)),
                (_strips(lanes,np.where(hi==0,ch,hi),ch,penalty),(2,)))
    else:
        groups=((_strips(lanes,lo,hi,penalty),(0,1,3)),(_strips(lanes,cl,ch,penalty),(2,)))
    table=[]; columns=[]; blocks=[]
    words=cols=width=0
    for strips,fields in groups:
        for a,b,ids in strips:
            size=len(ids)
            packed=(size+1)//2
            if int(ids[-1])-int(ids[0])+1==size:
                address=0x80000000|int(ids[0])
            else:
                address=cols
                columns.append(ids.astype(np.uint32))
                cols+=size
            width=max(width,packed)
            for field in fields:
                blocks.append((field,a,b,ids,words))
                for row in range(a,b):
                    table.append((field*160+row,size,address,words))
                    words+=packed
    rows=len(table)
    header=3+4*rows
    return np.concatenate((np.asarray([n,rows,header],np.uint32),np.array(table,np.uint32).ravel(),*columns)),blocks,width,words


class FrameBands:
    def __init__(self,gpu,penalty=256,fringe=False):
        self.gpu=gpu
        self.penalty=penalty
        self.fringe=fringe
        self.web=getattr(gpu,'backend',None)=='webgpu'
        self.pipe=None

    def _read(self,desc,width,words):
        root=Path(__file__).resolve().parent
        gpu=self.gpu
        rows=int(desc[1])
        if not self.web:
            if self.pipe is None:
                self.pipe=gpu.cp.RawKernel((root/'bench_r35_frame_bands.cu').read_text('utf8'),'gather_bands')
            inp=gpu.cp.asarray(desc)
            out=gpu.cp.empty(words,gpu.cp.uint32)
            self.pipe(((width+63)//64,rows),(64,),(*[getattr(gpu,f) for f in FIELDS],inp,out))
            return out.get()
        dev,usage=gpu.gpu_device,gpu.wgpu.BufferUsage
        if self.pipe is None:
            shader=dev.create_shader_module(code=(root/'bench_r35_frame_bands.wgsl').read_text('utf8'))
            self.pipe=dev.create_compute_pipeline(layout='auto',compute={'module':shader,'entry_point':'gather_bands'})
        with ExitStack() as resources:
            inp=dev.create_buffer(size=desc.nbytes,usage=usage.STORAGE|usage.COPY_DST)
            resources.callback(inp.destroy)
            out=dev.create_buffer(size=words*4,usage=usage.STORAGE|usage.COPY_SRC)
            resources.callback(out.destroy)
            gpu.queue.write_buffer(inp,0,desc)
            group=dev.create_bind_group(layout=self.pipe.get_bind_group_layout(0),entries=[
                 {'binding':i,'resource':{'buffer':buf}} for i,buf in enumerate((*gpu.buffers[1:5],inp,out))])
            cmd=dev.create_command_encoder()
            work=cmd.begin_compute_pass();work.set_pipeline(self.pipe);work.set_bind_group(0,group)
            work.dispatch_workgroups((width+63)//64,rows);work.end()
            gpu.queue.submit([cmd.finish()])
            data=gpu.queue.read_buffer(out,0,words*4)
            gpu.readback_bytes+=words*4
            return np.frombuffer(data,'<u4').astype(np.uint32,copy=False)

    def capture(self,old,lanes,marks,reset):
        desc,parts,width,words=plan(self.gpu.n,lanes,marks,reset,self.penalty,self.fringe)
        self.read_bytes=self.upload_bytes=self.maps=0
        if not words: return old
        values=self._read(desc,width,words).astype('<u4',copy=False).view('<u2')
        self.read_bytes=words*4;self.upload_bytes=desc.nbytes;self.maps=1
        changed={f for f,_,_,_,_ in parts}
        out=tuple(arr.copy() if i in changed else arr for i,arr in enumerate(old))
        for f,a,b,ids,start in parts:
            size=len(ids)
            width=((size+1)//2)*2
            part=values[2*start:2*start+(b-a)*width].reshape(b-a,width)[:,:size]
            cols=_selection(ids)
            if not isinstance(cols,slice) and len(ids)>=1024:
                for row,src in zip(out[f][a:b],part):
                    row[cols]=src
            else:
                out[f][a:b,cols]=part
        return out
