"""Count actual native WebGPU sparse lane uploads against the serial write layout."""
import argparse
import json
from unittest.mock import patch
from time import perf_counter
from pathlib import Path

import numpy as np

from dfs_webgpu import WebGPUDFS
from test_webgpu import arguments


class CountQueue:
    def __init__(self,queue):
        self.queue=queue
        self.writes=self.uploaded=self.submits=0

    def __getattr__(self,name):
        return getattr(self.queue,name)

    def write_buffer(self,buf,offset,data):
        self.writes+=1
        self.uploaded+=memoryview(data).nbytes
        return self.queue.write_buffer(buf,offset,data)

    def submit(self,cmd):
        self.submits+=1
        return self.queue.submit(cmd)


def run(api,n,count,threshold=None,anchor=None):
    import wgpu
    adapter=anchor.adapter if anchor is not None else next(a for a in wgpu.gpu.enumerate_adapters_sync()
                 if a.info['backend_type']==api and a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
    gpu=WebGPUDFS(*arguments(),n=n,prefixes=[],adapter=adapter,reuse_from=anchor)
    gpu.queue=CountQueue(gpu.queue)
    original=gpu._write_rows
    if threshold is not None:
        cutoff=patch('dfs_webgpu.PACKED_UPLOAD_MIN_WRITES',threshold)
        cutoff.start()
    calls=[]
    def observe(buffer,rows,cols,values):
        serial=(1 if np.array_equal(cols,np.arange(n)) and np.all(np.diff(rows)==1)
                else len(rows)*(1+np.count_nonzero(np.diff(cols)!=1)))
        before=gpu.queue.writes
        original(buffer,rows,cols,values)
        calls.append((int(serial),gpu.queue.writes-before))
    gpu._write_rows=observe
    try:
        ids=np.linspace(0,n-1,count,dtype=np.int32)
        start=perf_counter()
        gpu.load_prefixes([[int(i)] for i in range(count)],lanes=ids)
        gpu._synchronize()
        wall=perf_counter()-start
        new=gpu.queue.writes
        uploaded=gpu.queue.uploaded
        submits=gpu.queue.submits
        field_writes=sum(v for _,v in calls)
        assert field_writes<=new
        snap=gpu.snapshot_checkpoint()
        gpu.verify(ids)
        expected=np.zeros(n,bool)
        expected[ids]=True
        np.testing.assert_array_equal(snap['states']==0,expected)
        np.testing.assert_array_equal(snap['prefix_codes'][ids,0],np.arange(count))
        return dict(lanes=n,installed=count,seconds=round(wall,6),serial_write_calls=sum(v for v,_ in calls),
                    actual_write_calls=new,field_write_calls=field_writes,
                    read_selection_uploads=new-field_writes,uploaded_bytes=uploaded,
                    install_submissions=submits,verified_native_stacks=count)
    finally:
        if threshold is not None:
            cutoff.stop()
        gpu.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--api',choices=('Vulkan','D3D12','Metal'),required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--thresholds',nargs='+',type=int)
    args=cli.parse_args()
    out=[]
    cases=((128,1),(128,128),(512,32),(2048,256),(32768,384))
    if args.thresholds:
        cases=((128,1),(128,4),(128,16),(128,128),(2048,1),(2048,16))
    anchor=None
    if args.thresholds:
        import wgpu
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync() if a.info['backend_type']==args.api
                     and a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
        anchor=WebGPUDFS(*arguments(),n=max(n for n,_ in cases),prefixes=[],adapter=adapter)
    try:
        for limit in (args.thresholds or [None]):
            for n,count in cases:
                out.append({**run(args.api,n,count,limit,anchor),'tested_threshold':limit})
                args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    finally:
        if anchor is not None:
            anchor.close()
