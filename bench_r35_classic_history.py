"""Exercise native pure, mixed and probe-only classic histories in isolated public-run directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic,sleep
from unittest.mock import patch

import numpy as np
from classic_parked_storage import REFERENCE
from io_utils import atomic_json
import systematic_evidence as proof
import r35_classic_sources as policy


ROOT=Path(__file__).resolve().parent


def execute(root,data,backend,*,features=None,resume=False,stop_after_work=False,seconds=0,small=False):
    root.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-B',str(ROOT/'systematic_search.py'),'--backend',backend,'--target','gold',
          '--data',str(data),'--output',str(root),'--replicas',('16' if small else '480'),
          '--nodes',('1' if small else '128'),'--fixed-nodes','--seconds',str(seconds),'--checkpoint-seconds','1']
    if features:
        args+=['--gpu-features',json.dumps(features)]
    if resume:
        args+=['--resume']
    log=root/('resume.log' if resume else 'initial.log')
    env={**os.environ,'CUPY_CACHE_DIR':str(root.parent/'kernel-cache')}
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
        start=monotonic()
        while process.poll() is None:
            if stop_after_work and (root/'status.json').exists():
                status=proof.read(root/'status.json')
                if status.get('state')=='running' and status.get('nodes_checked',0)>0:
                    (root/'stop.request').touch()
            if monotonic()-start>180:
                process.terminate()
                raise TimeoutError(log)
            sleep(.05)
        if process.returncode:
            raise RuntimeError(log.read_text(encoding='utf8',errors='replace')[-2000:])
    return proof.read(root/'status.json')


def restore(root,current,change_root):
    folder=next(root.glob('systematic*/exhaustion.json')).parent
    marker=proof.read(folder/'exhaustion.json')
    scope={k:marker[k] for k in ('input_sha256','frontier_sha256','target','gpu_policy')}
    checkpoint=folder/'checkpoint.npz'
    with patch.object(proof,'__file__',str(change_root/'systematic_evidence.py')),np.load(checkpoint,allow_pickle=False) as a:
        return proof.reopen_terminal(root,checkpoint,folder/'exhaustion.json',a,current,scope,marker['total_jobs'])


def paired(root,features):
    marker=proof.read(next(root.glob('systematic*/exhaustion.json')))
    source=marker['source_sha256']
    origin=root/'runs'/marker['run_id']
    normalized=proof.archived_sources(origin,source)
    used=tuple(marker['proof_backends'])
    live=policy.active_image(origin,normalized,backends=used)
    assert policy.image_digest(live) in policy.GUARDED[used]
    raw=proof.read(root/'status.json')
    assert raw['verified_coverage_origin'] and raw['exhausted_jobs']==480
    trials=[]
    shared={'dfs_checkpoint_decode.c','r35_board_codec_decode.c','constraint_work.py','gpu_cpu_assist.py'}
    for name in sorted((policy.CUDA_RESOURCES|policy.WEBGPU_RESOURCES|shared)&source.keys()):
        with tempfile.TemporaryDirectory(dir=root.parent) as folder:
            changed=Path(folder)
            for key in source:
                shutil.copyfile(origin/key,changed/key)
            file=changed/name
            comment=b'\n# reviewed source mutation\n' if name.endswith('.py') else b'\n// reviewed shader mutation\n'
            file.write_bytes(file.read_bytes()+comment)
            hashes={**source,name:hashlib.sha256(file.read_bytes()).hexdigest()}
            try:
                result=restore(root,hashes,changed)
            except ValueError as exc:
                verdict='rejected'
            else:
                assert result['counters']==raw['counters'] and result['terminal_reopened_on_host']
                verdict='reopened'
            assert (verdict=='reopened')==(name not in live),(name,used,verdict)
            trials.append(dict(changed=name,state=verdict))
    # Individually falsify every operative source digest on this archive.
    with tempfile.TemporaryDirectory(dir=root.parent) as folder:
        changed=Path(folder)
        for key in source:
            shutil.copyfile(origin/key,changed/key)
        for name in sorted(live):
            path=changed/name
            original=path.read_bytes()
            try:
                added=original+b'\n'
                path.write_bytes(added)
                variant={**normalized,name:hashlib.sha256(added.replace(b'\r\n',b'\n')).hexdigest()}
                assert not policy.compatible(origin,normalized,changed,variant,marker['gpu_policy'],backends=list(used)),name
            finally:
                path.write_bytes(original)
    return dict(engines=used,scope=features,full_sources=len(normalized),active=len(live),
                individually_rejected_active_files=len(live),active_digest=policy.image_digest(live),
                jobs=480,counters=raw['counters'],native_probe=raw.get('cpu_assist'),trials=trials,origin=marker['run_id'])


def partial_source_replay(root, status):
    checkpoint=next(root.glob('systematic*/checkpoint.npz'))
    origin=proof.run_dir(root,status['run_id'])
    config=proof.read(origin/'run-config.json')
    source=config['source_sha256']
    with np.load(checkpoint,allow_pickle=False) as archive, tempfile.TemporaryDirectory(dir=root.parent) as folder:
        assert str(archive['exact_source_origin'])==status['run_id']
        binding=str(archive['exact_source_binding'])
        from gpu_reasoning import feature_identity
        scope=proof.checkpoint_scope(archive,dict(target=config['target'],gpu_policy=feature_identity(config['gpu_features']),
                                               frontier_sha256=config['frontier_sha256'],input_sha256=source['tiles.json']))
        proof.check_cpu_scope(archive,scope['gpu_policy'])
        proof.check_receipt(root,status['run_id'],int(archive['exact_generation']),checkpoint,binding,
                            parked=REFERENCE in archive)
        destination=Path(folder)
        for key in source:
            shutil.copyfile(origin/key,destination/key)
        path=destination/'constraint_work.py'
        path.write_bytes(path.read_bytes()+b'\n# dormant on this guarded probe-only traversal\n')
        current={**source,path.name:hashlib.sha256(path.read_bytes()).hexdigest()}
        with patch.object(proof,'__file__',str(destination/'systematic_evidence.py')):
            assert proof.verify_origin(root,status['run_id'],binding,current,scope,required_backend='cuda')==source
            active=destination/'constraint_pruning.py'
            active.write_bytes(active.read_bytes()+b'\n# live prefix probe mutation\n')
            current[active.name]=hashlib.sha256(active.read_bytes()).hexdigest()
            try:
                proof.verify_origin(root,status['run_id'],binding,current,scope,required_backend='cuda')
            except ValueError as exc:
                assert 'source has changed' in str(exc)
            else:
                raise AssertionError('accepted a modified live probe implementation')
        return dict(generation=int(archive['exact_generation']),source_run=status['run_id'],
                    signed_engines=scope['proof_backends'],partial_source_reopened_on_dormant_change=True,
                    verified_probe_implementation_required=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--original',type=Path,default=Path('../data/tiles.json'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    root=a.audit.resolve()
    root.mkdir(parents=True,exist_ok=True)
    report={}
    for name,backend,flags in [('cuda','cuda',{}),('webgpu','webgpu',{}),
                               ('mixed','cuda',{}),('probe','cuda',{'cpu_probe':True}),
                               ('proof-probe','cuda',{'proof_probes':True})]:
        run=root/name
        if not list(run.glob('systematic*/exhaustion.json')):
            if name=='mixed' and not (run/'systematic/checkpoint.npz').exists():
                old=execute(run,a.fixture.resolve(),backend,stop_after_work=True,small=True)
                assert old['state']=='stopped' and old['nodes_checked']>0
            state=execute(run,a.fixture.resolve(),'webgpu' if name=='mixed' else backend,
                          features=flags,resume=name=='mixed')
            assert state['state']=='exhausted'
        report[name]=paired(run,flags)
        if name=='mixed':
            assert report[name]['engines']==('cuda','webgpu')
        print(name,report[name]['engines'],report[name]['counters'],flush=True)
        atomic_json(a.output,report)
    assert report['cuda']['counters']==report['webgpu']['counters']==report['mixed']['counters']
    for name,flags in [('deep-probe',{'cpu_probe':True}),('deep-targeted',{'proof_probes':True})]:
        run=root/name
        status=proof.read(run/'status.json') if (run/'status.json').exists() else None
        if not status or not status.get('cpu_assist',{}).get('cpu_prefix_probes'):
            status=execute(run,a.original.resolve(),'cuda',features=flags,seconds=22,resume=bool(status))
        report[name]=dict(state=status['state'],active=status['active_jobs'],completed=status['exhausted_jobs'],
                          max_depth=status['max_depth'],counters=status['counters'],native_probe=status.get('cpu_assist'),
                          **partial_source_replay(run,status))
        assert report[name]['native_probe']['cpu_prefix_probes']>0
        if name=='deep-targeted':
            assert report[name]['native_probe']['cpu_probe_selection_contexts']>0
        print(name,report[name],flush=True)
        atomic_json(a.output,report)


if __name__=='__main__':
    main()
