"""Compare the physical four-field frame read at 9a3403b with the paired packet."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from classic_frame_packet import FramePacket
from dfs_gpu import DFS
from test_gpu_gold import gold_fixture
from validator import orientation_masks


def measure(baseline,backend,width,repeats):
    source=baseline/'classic_checkpoint_ranges.py'
    if hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!='7e3401ce273d45ef45595e2a0fbf509c97b3c2a6e8e06f9c72ff88cbfb0b0548':
        raise ValueError('Expected the original range reader from 9a3403b.')
    spec=importlib.util.spec_from_file_location('r35_original_frames',source)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    data,board=gold_fixture()
    args=(np.array(orientation_masks(data),np.uint16),board.neighbor_cells,board.neighbor_sides)
    if backend=='CUDA':gpu=DFS(*args,n=width,prefixes=[])
    else:
        from dfs_webgpu import WebGPUDFS
        import wgpu
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync() if a.info['backend_type']==backend and
                     a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
        gpu=WebGPUDFS(*args,n=width,prefixes=[],adapter=adapter)
    try:
        legacy=old.CheckpointRanges(gpu,SimpleNamespace())
        pair=FramePacket(gpu)
        ids=np.arange(width,dtype=np.int32)
        cases={'single':[(119,120,ids[-1:])],'scattered':[(3,20,ids[5::257])],
               'contiguous':[(2,20,ids)],'mixed':[(2,14,ids),(19,40,ids[3::47]),(76,151,ids[-177:])],
               'complete':[(0,160,ids)]}
        def previous(windows):
            legacy.maps=legacy.read_bytes=0
            if len(windows)>1:return legacy._frames(windows)
            a,b,cols=windows[0]
            if legacy.web:return [legacy._physical(i,a,b,cols).astype(np.uint16).ravel() for i in range(1,5)]
            return [legacy._read(name,cols,a,b).ravel() for name in old.FRAME_FIELDS]
        out={}
        for name,windows in cases.items():
            def capture(arm):
                start=perf_counter()
                arrays=previous(windows) if arm=='old' else pair.capture(windows)
                elapsed=perf_counter()-start
                obj=legacy if arm=='old' else pair
                return elapsed,obj.maps,obj.read_bytes,[hashlib.sha256(x).hexdigest() for x in arrays]
            frozen=capture('old')[-1]
            assert capture('new')[-1]==frozen
            samples={'old':[],'new':[]}
            for i in range(repeats):
                for arm in (('new','old') if i%2 else ('old','new')):
                    elapsed,maps,size,hashes=capture(arm)
                    assert hashes==frozen
                    samples[arm].append({'seconds':elapsed,'maps':maps,'read_bytes':size})
            out[name]={'field_hashes':frozen,'samples':samples,
                       'median_seconds':{k:median(s['seconds'] for s in v) for k,v in samples.items()}}
        return dict(baseline_commit='9a3403b7f940d09f4780437e16dc0aabdaf668f2',backend=backend,width=width,observations=out)
    finally:
        if backend!='CUDA':gpu.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--baseline',type=Path,required=True,help='Extracted tree of commit 9a3403b.')
    cli.add_argument('--backend',choices=('CUDA','Vulkan','D3D12'),required=True)
    cli.add_argument('--width',type=int,default=131072)
    cli.add_argument('--repeats',type=int,default=3)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    result=measure(args.baseline,args.backend,args.width,args.repeats)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({name:{arm:(s[0]['maps'],s[0]['read_bytes'],round(row['median_seconds'][arm],4))
                           for arm,s in row['samples'].items()} for name,row in result['observations'].items()}))
