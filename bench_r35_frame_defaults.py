"""Compare public first/full checkpoint capture on a fresh pool and an actual restored archive."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from classic_checkpoint_ranges import CheckpointRanges
from classic_frame_defaults import FrameDefaults
from dfs_gpu import DFS,fill_order
from search_problem import SearchProblem
from systematic_jobs import JobLedger


def identity(payload):
    return {k:hashlib.sha256(np.ascontiguousarray(v).view(np.uint8)).hexdigest() for k,v in payload.items()}


def measure(api,checkpoint,tiles,repeats,previous=None):
    p=SearchProblem(json.loads(tiles.read_bytes()))
    with np.load(checkpoint,allow_pickle=False) as archive:
        n=len(archive['states'])
        # Preserve H/W independent logical fields and metadata. Leave ledger installation to the runner.
        saved={k:archive[k] for k in (*DFS.DEVICE_FIELDS,'prefix_codes','order','roots','fingerprint','dfs_version')}
        for key in ('gold_fingerprint','gpu_features','cpu_assist_state'):
            if key in archive:saved[key]=archive[key]
    options=dict(masks=p.masks,neighbors=p.neighbors,neighbor_sides=p.sides,n=n,prefixes=[],
                 order=fill_order(p.neighbors),gold_data=p.tiles)
    if api=='CUDA':gpu=DFS(**options)
    else:
        from dfs_webgpu import WebGPUDFS
        import wgpu
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync() if a.info['backend_type']==api and
                     a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
        gpu=WebGPUDFS(**options,adapter=adapter)
    old=None
    if previous:
        mod=importlib.util.spec_from_file_location('r35_plain_frames',previous/'classic_frame_defaults.py')
        old=importlib.util.module_from_spec(mod)
        mod.loader.exec_module(old)
    arms={'packed':None,'bounded':FrameDefaults}
    if old is not None:arms['rows']=old.FrameDefaults
    stages={}
    def reader(arm):
        r=CheckpointRanges(gpu,JobLedger.empty(0,n))
        if arms[arm] is None:r.defaults_disabled=True
        else:r.frame_defaults=arms[arm](gpu)
        return r
    def close(r):
        if r.board_delta is not None:r.board_delta.close()
    def sample(r,reference):
        start=perf_counter()
        payload=r.capture()
        seconds=perf_counter()-start
        assert identity(payload)==reference
        s={'seconds':seconds,'maps':r.maps,'bytes':r.read_bytes}
        if r.frame_defaults is not None:
            s.update(frame_bytes=r.frame_defaults.read_bytes,frame_maps=r.frame_defaults.maps,
                     modes=r.frame_defaults.distribution)
            if r.defaults_disabled:raise ValueError('Frame compression failed and silently fell back.')
        return s
    def stage(name):
        reference=identity(gpu.snapshot_checkpoint())
        samples={arm:[] for arm in arms}
        for it in range(repeats):
            for arm in (list(arms) if it%2==0 else reversed(list(arms))):
                r=reader(arm)
                try:samples[arm].append(sample(r,reference))
                finally:close(r)
        stages[name]={'full_returned_fields':len(reference),'samples':samples,
            'median_seconds':{arm:median(item['seconds'] for item in res) for arm,res in samples.items()}}
    try:
        stage('fresh')
        gpu.load_prefixes([[i] for i in range(min(n,480))])
        stage('installed_prefixes')
        # The warm test also keeps a previously used reader through physical full reinstallation.
        prior=reader('bounded')
        sample(prior,identity(gpu.snapshot_checkpoint()))
        begin=perf_counter()
        gpu.restore_checkpoint(saved)
        restore_seconds=perf_counter()-begin
        prior.ledger.checkpoint_installs[:]=True
        stages['reinstalled_after_first_capture']=sample(prior,identity(gpu.snapshot_checkpoint()))
        close(prior)
        stage('restored')
    finally:
        if api!='CUDA':gpu.close()
    return {'backend':api,'width':n,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'restore_seconds':restore_seconds,'stages':stages}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--backend',choices=('CUDA','Vulkan','D3D12'),required=True)
    cli.add_argument('--checkpoint',type=Path,required=True)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--previous',type=Path,help='Optional frozen e9c2f15 default-row prototype.')
    cli.add_argument('--repeats',type=int,default=3)
    cli.add_argument('--output',type=Path,required=True)
    a=cli.parse_args()
    result=measure(a.backend,a.checkpoint,a.tiles,a.repeats,a.previous)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({name:row.get('median_seconds',row) for name,row in result['stages'].items()}))
