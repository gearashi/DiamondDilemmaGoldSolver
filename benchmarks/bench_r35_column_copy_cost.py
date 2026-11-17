"""Price strided GPU transfer dispatch against emitted row copies on frozen native data."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

from bench_r35_sparse_upload import CountQueue
from dfs_webgpu import WebGPUDFS
from test_webgpu import arguments


def run(api,n):
    import wgpu
    adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync() if
                 a.info['backend_type']==api and a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
    gpu=WebGPUDFS(*arguments(),n=n,prefixes=[],adapter=adapter)
    try:
        values=(np.arange(160*n,dtype=np.uint32).reshape(160,n)*17+192).astype(np.uint16)
        gpu.firsts[:,:]=values
        gpu.queue=CountQueue(gpu.queue)
        rows=[]
        for width in (1,8,min(128,n//2),n):
          for height in (1,4,8,16,32,64,96,128,160):
            ids=np.arange(width) if width==n else np.arange(2,2+width)
            physical=np.arange(height)
            results=[]
            for cutoff in (2,8,16,32,64,161):
                times=[]
                with patch('dfs_webgpu.PACKED_READ_MIN_ROWS',cutoff):
                    for rep in range(5):
                        before=(gpu.queue.writes,gpu.queue.uploaded,gpu.queue.submits,gpu.readback_bytes)
                        start=perf_counter()
                        result=gpu._read_rows(gpu.buffers[1],physical,ids)
                        times.append(perf_counter()-start)
                        np.testing.assert_array_equal(result,values[:height,ids])
                        after=(gpu.queue.writes,gpu.queue.uploaded,gpu.queue.submits,gpu.readback_bytes)
                uploads,up_bytes,submits,read=(b-a for a,b in zip(before,after))
                results.append(dict(cutoff=cutoff,steady_ms=round(median(times[1:])*1000,4),
                    cold_ms=round(times[0]*1000,4),selection_uploads=uploads,selection_bytes=up_bytes,
                    submissions=submits,read_bytes=read,physical_rows=height if submits and not uploads else 0))
            rows.append(dict(lanes=n,columns=width,rows=height,results=results))
        return rows
    finally:
        gpu.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--api',required=True,choices=('D3D12','Vulkan','Metal'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    rows=run(args.api,2048)
    args.output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
