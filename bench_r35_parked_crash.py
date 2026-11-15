"""Kill only an isolated test runner after a signed parked interval and resume it normally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

from classic_parked_storage import REFERENCE, restore_parked
from systematic_evidence import check_receipt


ROOT=Path(__file__).resolve().parent


def inspect(output):
    path=output/'systematic/checkpoint.npz'
    with np.load(path,allow_pickle=False) as head:
        gen=int(head['exact_generation'])
        origin=str(head['exact_source_origin'])
        external=REFERENCE in head.files
        check_receipt(output,origin,gen,path,str(head['exact_source_binding']),parked=external)
        with restore_parked(path,head) as full:
            counters=full['counters'].sum(axis=1,dtype=np.uint64)+full['exact_retired_counters']
            paused=full['exact_paused_counters']
            counters+=paused.sum(axis=1,dtype=np.uint64)
            return dict(generation=gen,origin=origin,external=external,total=int(full['exact_total']),
                completed=int(full['exact_completed']),cursor=int(full['exact_cursor']),
                paused=len(full['exact_paused_job_ids']),counters=list(map(int,counters)))


def job(output,tiles,size,action,previous=None):
    # Launch the interpreter itself. The Windows venv redirector has its own
    # PID and terminating it alone can leave its actual solver child running.
    exe=sys._base_executable if os.name=='nt' else sys.executable
    env=dict(os.environ)
    if os.name=='nt':
        env['__PYVENV_LAUNCHER__']=sys.executable
    cmd=[exe,'-B',str(ROOT/'systematic_search.py'),'--backend','cuda','--target','gold',
         '--data',str(tiles),'--output',str(output),'--replicas',str(size),'--seconds','0','--nodes','8',
         '--fixed-nodes','--checkpoint-seconds','.25']
    if previous is not None:
        cmd.append('--resume')
    log=output.parent/f'job-{size}.log'
    with log.open('wb') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise AssertionError(log.read_text(encoding='utf8',errors='replace')[-5000:])
                try:
                    status=json.loads((output/'status.json').read_bytes())
                    current=(status.get('pid')==proc.pid and status.get('state')=='running')
                    if current and action=='stop' and status.get('nodes_checked',0)>(previous or {}).get('counters',[0])[0]:
                        (output/'stop.request').touch()
                        assert proc.wait(timeout=90)==0,log.read_bytes()[-2000:]
                        break
                    if current and action=='crash' and list((output/'systematic').glob('parked-bank-*.npz')):
                        path=output/'systematic/checkpoint.npz'
                        with np.load(path,allow_pickle=False) as head:
                            eligible=(REFERENCE in head.files and str(head['exact_source_origin'])!=previous['origin'])
                        if eligible:
                            proc.kill()
                            proc.wait(timeout=30)
                            lock=output/'run.lock'
                            assert lock.parent.resolve()==output.resolve() and lock.read_text()==str(proc.pid)
                            lock.unlink()
                            break
                except (FileNotFoundError,PermissionError,json.JSONDecodeError):
                    pass
                time.sleep(.15)
            else:
                raise TimeoutError((action,size,log.read_bytes()[-1200:]))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
    return inspect(output)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    with TemporaryDirectory(dir=ROOT/'audit/r35',prefix='isolated-crash-') as tmp:
        run=Path(tmp)/'run'
        first=job(run,args.tiles.resolve(),2048,'stop')
        interrupted=job(run,args.tiles.resolve(),128,'crash',first)
        resumed=job(run,args.tiles.resolve(),64,'stop',interrupted)
        assert not first['external'] and interrupted['external'] and not resumed['external']
        assert first['total']==interrupted['total']==resumed['total']
        assert all(a<=b<=c for a,b,c in zip(first['counters'],interrupted['counters'],resumed['counters']))
        assert resumed['generation']>interrupted['generation']>=first['generation']
        assert not list((run/'systematic').glob('parked-bank-*.npz'))
        status=json.loads((run/'status.json').read_bytes())
        assert status['state']=='stopped' and status['verified_coverage_origin']
        result=dict(initial=first,after_kill=interrupted,normal_resume=resumed,verified_origin=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
