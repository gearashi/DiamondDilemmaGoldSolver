"""Exact device-side chunk comparison versus complete CUDA classic checkpoint readback."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from dfs_gpu import DFS, fill_order
from exact_frontier import plan_frontier
from search_problem import SearchProblem
from systematic_jobs import JobLedger


KERNELS = r"""
extern "C" __global__ void changed(const unsigned char *cur, const unsigned char *last,
 unsigned char *flags, int size, int width) {
 int begin = blockIdx.x * width, end = min(size, begin + width);
 bool diff = false;
 for (int i=begin + threadIdx.x; i<end; i+=blockDim.x)
     diff |= cur[i]!=last[i];
 int any = __syncthreads_or(diff);
 if (threadIdx.x==0) flags[blockIdx.x]=any;
}
extern "C" __global__ void gather(const unsigned char *cur, unsigned char *last,
 const int *index, unsigned char *out, int size, int width) {
 int begin=index[blockIdx.x]*width;
 for (int j=threadIdx.x; j<width; j+=blockDim.x) {
    int i=begin+j;
    if (i<size) out[blockIdx.x*width+j]=last[i]=cur[i];
 }
}
"""


class CUDAChunkReader:
    """Prototype. Old CPU arrays are never mutated after publication."""
    def __init__(self, gpu, width):
        self.gpu=gpu
        self.cp=gpu.cp
        self.width=width
        code=self.cp.RawModule(code=KERNELS,name_expressions=('changed','gather'))
        self.changed=code.get_function('changed')
        self.gather=code.get_function('gather')
        devices=[getattr(gpu,k) for k in gpu.DEVICE_FIELDS]
        assert all(v.flags.c_contiguous and v.nbytes < 2**31 for v in devices)
        self.last={k:v.copy() for k,v in zip(gpu.DEVICE_FIELDS,devices)}
        self.host={k:v.get() for k,v in zip(gpu.DEVICE_FIELDS,devices)}
        for value in self.host.values():
            value.setflags(write=False)
        self.groups={k:(v.nbytes+width-1)//width for k,v in zip(gpu.DEVICE_FIELDS,devices)}
        self.flags=self.cp.empty(sum(self.groups.values()),np.uint8)
        self.raw_bytes=sum(v.nbytes for v in devices)

    def capture(self):
        start=perf_counter()
        width=self.width
        off=0
        for name,count in self.groups.items():
            cur=getattr(self.gpu,name)
            self.changed((count,),(256,),(cur,self.last[name],self.flags[off:],np.int32(cur.nbytes),np.int32(width)))
            off+=count
        flags=self.flags.get()
        reads=1
        size=flags.nbytes
        upload=changed=0
        off=0
        results={}
        for name,count in self.groups.items():
            cur=getattr(self.gpu,name)
            old=self.host[name]
            dirty=np.flatnonzero(flags[off:off+count]).astype(np.int32)
            off+=count
            changed+=len(dirty)
            if not len(dirty):
                results[name]=old
                continue
            reads+=1
            if len(dirty)*width >= .85*cur.nbytes:
                value=cur.get()
                self.cp.copyto(self.last[name],cur)
                size+=cur.nbytes
            else:
                packed=self.cp.empty(len(dirty)*width,np.uint8)
                index=self.cp.asarray(dirty)
                self.gather((len(dirty),),(256,),(cur,self.last[name],index,packed,np.int32(cur.nbytes),np.int32(width)))
                transfer=packed.get().reshape(-1,width)
                size+=transfer.nbytes
                upload+=dirty.nbytes
                value=old.copy()
                raw=value.reshape(-1).view(np.uint8)
                for i,chunk in enumerate(dirty):
                    at=int(chunk)*width
                    n=min(width,len(raw)-at)
                    raw[at:at+n]=transfer[i,:n]
            value.setflags(write=False)
            results[name]=value
        self.cp.cuda.get_current_stream().synchronize()
        self.host=results
        return results,dict(read_bytes=size,upload_bytes=upload,maps=reads,changed=changed,
                            seconds=round(perf_counter()-start,6))


def run(p,n,used,budgets,widths):
    order=fill_order(p.neighbors)
    plan=plan_frontier(p.masks,p.neighbors,p.sides,order,used)
    prefix,lengths=plan.prefixes[:used],plan.lengths[:used]
    gpu=DFS(p.masks,p.neighbors,p.sides,n=n,order=order,seed=35068,prefixes=[],gold_data=p.tiles)
    ledger=JobLedger.empty(len(prefix),n)
    ledger.refill(gpu,prefix,lengths)
    readers=[CUDAChunkReader(gpu,width) for width in widths]
    rows=[]
    for nodes in budgets:
        remaining=nodes
        while remaining:
            amount=min(512,remaining)
            gpu.step(amount)
            ledger.refill(gpu,prefix,lengths)
            remaining-=amount
        start=perf_counter()
        expected=gpu.snapshot_checkpoint()
        wall=perf_counter()-start
        for lane in np.flatnonzero(expected['states']==1):
            codes=list(map(int,expected['boards'][:,lane]))
            report=p.validate(codes)
            if report['valid']:
                (Path(__file__).parent/'audit/r35/checkpoint-gold.json').write_text(
                    json.dumps({'codes':codes,'validation':report}),encoding='utf8')
                raise SystemExit('Verified Gold saved; search stopped.')
        record=dict(nodes_per_lane=nodes,active=int((expected['states']==0).sum()),baseline_bytes=readers[0].raw_bytes,
                    baseline_maps=len(gpu.DEVICE_FIELDS),baseline_seconds=round(wall,6),variants={})
        for reader in readers:
            prior=dict(reader.host)
            actual,metrics=reader.capture()
            for k,value in actual.items():
                assert value.dtype==expected[k].dtype and np.array_equal(value,expected[k]),k
                assert not prior[k].flags.writeable and not value.flags.writeable
            record['variants'][str(reader.width)]=metrics
        rows.append(record)
    return rows


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--sustained',action='store_true')
    args=cli.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    out={}
    cases=((128,128),(2048,128),(2048,2048),(32768,512),(32768,32768))
    for n,used in cases:
        budgets=(0,1,8,64,256,4096,32768) if args.sustained else (0,1,8,64,256)
        out[f'{n}/{used}']=run(p,n,used,budgets,(4096,16384,65536))
        args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
