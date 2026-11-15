"""Replay a physical CUDA checkpoint produced by the published v1.14 classic runner."""
import argparse
import json
import os
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parent


def run(runner, path, data, replicas, seconds, resume=False, *, success=True):
    command = [sys.executable,'-B',str(runner/'systematic_search.py'),'--backend','cuda','--target','gold',
               '--replicas',str(replicas),'--seconds',str(seconds),'--data',str(data),'--output',str(path),
               '--nodes','1' if seconds else '128','--fixed-nodes']
    if resume:
        command.append('--resume')
    completed=subprocess.run(command,cwd=runner,check=False,env={**os.environ,'CUPY_CACHE_DIR':str(ROOT.parent/'runtime'/'cuda-cache')})
    assert (completed.returncode==0)==success,(command,completed.returncode)
    return json.loads((path/'status.json').read_text(encoding='utf-8'))


def experiment(old):
    with TemporaryDirectory(dir=ROOT, prefix='classic-origin-') as directory:
        root=Path(directory)
        data=root/'tiles.json'
        data.write_text(json.dumps({'tiles':[{'id':str(i),'segments':[[[s,5],[s,7]] for s in range(3)]}
                                             for i in range(160)]}))
        results={}
        for case in ('partial','terminal','altered_partial','pristine','pristine_changed_frontier'):
            out=root/case
            initial=run(old,out,data,4 if case!='terminal' else 480,(1e-9 if case.startswith('pristine') else 4) if case!='terminal' else 0)
            assert initial['state']==('stopped' if case!='terminal' else 'exhausted'), initial['state']
            with np.load(out/'systematic'/'checkpoint.npz',allow_pickle=False) as archive:
                assert 'exact_source_origin' not in archive
            if case.startswith('pristine'):
                assert initial['checkpoint']['completed_generation']==initial['nodes_checked']==0
                (out/'runs'/initial['run_id']/'final-status.json').unlink()
            if case=='pristine_changed_frontier':
                folder=out/'systematic'
                path=folder/'frontier.npz'
                with np.load(path,allow_pickle=False) as archive:
                    saved={k:archive[k] for k in archive.files}
                saved['prefixes']=saved['prefixes'][:-1]
                saved['lengths']=saved['lengths'][:-1]
                meta=json.loads(str(saved['metadata']))
                meta['jobs']=len(saved['prefixes'])
                meta['prefixes_sha256']=hashlib.sha256(saved['prefixes'].tobytes()+saved['lengths'].tobytes()).hexdigest()
                saved['metadata']=np.array(json.dumps(meta))
                with path.open('wb') as stream: np.savez(stream,**saved)
                checksum=hashlib.sha256(path.read_bytes()).hexdigest()
                path=folder/'checkpoint.npz'
                with np.load(path,allow_pickle=False) as archive:
                    saved={k:archive[k] for k in archive.files}
                saved.update(exact_frontier_sha256=np.array(checksum),exact_total=np.array(meta['jobs'],np.int64))
                with path.open('wb') as stream: np.savez(stream,**saved)
                current=run(ROOT,out,data,480,0,resume=True,success=False)
                assert 'planner refutes' in current['error'], current
                results[case]=dict(prior_nodes=0,forged_jobs=meta['jobs'],state=current['state'])
                continue
            if case=='altered_partial':
                path=out/'systematic'/'checkpoint.npz'
                size=path.stat().st_size
                with np.load(path,allow_pickle=False) as archive:
                    fields={k:archive[k] for k in archive.files}
                fields['rng'][0] ^= np.uint32(128)
                with path.open('wb') as stream: np.savez(stream,**fields)
                assert path.stat().st_size==size
            first=run(ROOT,out,data,480,0,resume=True)
            if case=='altered_partial':
                assert first['state']=='unverified_exhaustion' and not first['verified_coverage_origin'],first
                results[case]=dict(same_bytes=True,unchanged_counters=True,history_verified=False,final_jobs=first['exhausted_jobs'])
                continue
            if case=='partial':assert first['historical_stack_replayed'] is True,first
            assert first['state']=='exhausted' and first['verified_coverage_origin'], first
            second=run(ROOT,out,data,128,0,resume=True)
            assert second['state']=='exhausted' and second['verified_coverage_origin'], second
            assert second['counters']==first['counters'] and second['terminal_reopened_on_host']
            results[case]=dict(legacy_generation=initial['checkpoint']['completed_generation'],
                legacy_checked=initial['nodes_checked'],first_checked=first['nodes_checked'],second_checked=second['nodes_checked'],
                final_jobs=second['exhausted_jobs'],preserved=True,second_gpu_allocations=0)
        return results


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--v114-root',required=True,type=Path)
    print(json.dumps(experiment(arg.parse_args().v114_root.resolve()),indent=2))
