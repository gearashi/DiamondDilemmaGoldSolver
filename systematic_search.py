"""GPU systematic search: disjoint branches, durable stacks, and exact edge pruning.

Full-board uniqueness follows from prefix-free jobs and monotone DFS cursors.
Normal Stop/Resume retains all progress. A crash can replay work after the last
completed atomic checkpoint; it never skips uncommitted work to claim coverage.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from geometry import build_board
from validator import orientation_masks, validate_arrangement
from dfs_gpu import DFS, fill_order, ensure_disjoint_prefixes
from gpu_backends import choose_backend
from exact_frontier import plan_frontier
from systematic_jobs import JobLedger
from checkpoint_writer import CheckpointWriter
from io_utils import atomic_json, replace_with_retry
from render_board import render

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'runtime'/'cuda-cache'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_npz(path, fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    started=time.monotonic()
    try:
        with temporary.open('wb') as stream:
            np.savez(stream,**fields); stream.flush(); os.fsync(stream.fileno())
        size=temporary.stat().st_size
        replace_with_retry(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'compressed':False,'file_bytes':size,'write_seconds':time.monotonic()-started}

class StartupStopped(Exception): pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=3600,help='Run duration in seconds; 0 means unlimited.')
    parser.add_argument('--replicas',type=int,default=131072)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--backend',choices=('auto','cuda','webgpu'),default='auto',help='CUDA for NVIDIA; WebGPU for Metal/Vulkan/DirectX; auto detects the platform.')
    parser.add_argument('--nodes',type=int,default=32)
    parser.add_argument('--checkpoint-seconds',type=float,default=30)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-file-initialized',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--data',type=Path,default=ROOT/'data'/'tiles.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runtime')
    args=parser.parse_args()
    try: backend=choose_backend(args.backend)
    except ValueError as exc: parser.error(str(exc))
    if not math.isfinite(args.seconds) or args.seconds<0 or not 1<=args.replicas<=131072 or not 1<=args.nodes<=512 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<=0:
        parser.error('Seconds must be 0 (unlimited) or positive; checkpoint interval positive, replicas 1..131072, nodes 1..512 required.')
    run=args.output.resolve(); run.mkdir(parents=True,exist_ok=True)
    exact=run/'systematic'; exact.mkdir(exist_ok=True)
    lock=run/'run.lock'
    try: fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise SystemExit('A solver lock exists; wait for the current solver to stop.')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    started=dt.datetime.now(dt.timezone.utc); run_id=started.strftime('%Y%m%dT%H%M%S%fZ')
    evidence=run/'runs'/run_id; evidence.mkdir(parents=True)
    stopfile=run/'stop.request'
    if not args.stop_file_initialized: stopfile.unlink(missing_ok=True)
    stop=False; began=time.monotonic(); gpu=None; writer=None; ledger=None
    state='starting'; error=None; generation=0; maximum=0; checked=0; trusted=False
    checkpoint=exact/'checkpoint.npz'; frontier_path=exact/'frontier.npz'
    status={'state':'starting','method':'systematic','search_mode':'systematic','pid':os.getpid(),
            'run_id':run_id,'backend':backend,'replicas':args.replicas,'time_limit_seconds':args.seconds,
            'no_repeat_scope':'Disjoint branches and saved DFS cursors; crashes may replay since the last checkpoint.'}
    hashes={}; frontier_hash=None; frontier_metadata={}; previous_best=None
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    def stopping(): return stop or stopfile.exists()
    def publish(phase=None,message=None):
        status.update(state=state,elapsed_seconds=round(time.monotonic()-began,2),updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
        if phase: status.update(startup_phase=phase,phase_message=message)
        elif state!='starting':
            status.pop('startup_phase',None); status.pop('phase_message',None)
        if ledger is not None:
            status.update(total_jobs=ledger.total,exhausted_jobs=ledger.completed,assigned_jobs=ledger.cursor,
                          active_jobs=int(np.count_nonzero(ledger.ids>=0)),paused_jobs=ledger.paused_count,
                          queued_jobs=ledger.paused_count+ledger.total-ledger.cursor,unassigned_jobs=ledger.total-ledger.cursor,
                          max_depth=maximum,candidates_checked=checked)
        if gpu is not None:
            counts=ledger.counter_totals(gpu) if ledger is not None else gpu.counters.get().sum(axis=1).astype(object)
            status.update(gpu=gpu.device,replicas=gpu.n,nodes_checked=int(counts[0]),counters=[int(v) for v in counts])
        if writer is not None: status['checkpoint']=dict(writer.info)
        if error: status['error']=error
        atomic_json(run/'status.json',status)
    try:
        for name in ('status.json','run-config.json'):
            old=run/name
            if old.exists(): (evidence/('previous-'+name)).write_bytes(old.read_bytes())
        publish('loading_input','Loading the puzzle for systematic search.')
        input_bytes=args.data.read_bytes(); data_sha=hashlib.sha256(input_bytes).hexdigest()
        data=json.loads(input_bytes.decode('utf-8-sig'))
        masks=np.asarray(orientation_masks(data),np.uint16); board=build_board(); order=fill_order(board.neighbor_cells)
        neighbors=np.asarray(board.neighbor_cells,np.int16); neighbor_sides=np.asarray(board.neighbor_sides,np.uint8)
        if masks.shape!=(480,3): raise ValueError('Gold requires exactly 160 pieces.')
        inputs=run/'inputs'; inputs.mkdir(exist_ok=True)
        input_snapshot=inputs/(data_sha+'.json')
        if not input_snapshot.exists(): input_snapshot.write_bytes(input_bytes)
        (evidence/'tiles.json').write_bytes(input_bytes)
        for name in ('gpu_backends.py','systematic_search.py','systematic_jobs.py','exact_frontier.py','dfs_gpu.py','dfs_kernels.cu','gpu_engine.py','geometry.py','validator.py','io_utils.py','checkpoint_writer.py'):
            body=(ROOT/name).read_bytes(); (evidence/name).write_bytes(body); hashes[name]=hashlib.sha256(body).hexdigest()
        if backend=='webgpu':
            for name in ('dfs_webgpu.py','dfs_kernels.wgsl'):
                body=(ROOT/name).read_bytes(); (evidence/name).write_bytes(body); hashes[name]=hashlib.sha256(body).hexdigest()
        hashes['tiles.json']=data_sha; status['source_sha256']=hashes
        if (run/'best.json').exists():
            previous_best=json.loads((run/'best.json').read_text(encoding='utf-8-sig'))
            report=validate_arrangement(data,previous_best['codes'])
            status['best_matched_edges']=report['matched_edges']; status['best_components']=report['line_components']
            status['required_matched_edges']=240
            if report['matched_edges']==240:
                state='solved' if report['valid'] else 'edge_perfect'
                publish(); return
        if stopping(): raise StartupStopped()
        if checkpoint.exists() and not args.resume:
            raise ValueError('A systematic checkpoint already exists. Resume it to preserve completed work.')
        frontier_binding=hashlib.sha256(masks.tobytes()+neighbors.tobytes()+neighbor_sides.tobytes()+order.tobytes()+data_sha.encode()).hexdigest()
        if frontier_path.exists():
            publish('loading_frontier','Loading the saved, disjoint search branches.')
            with np.load(frontier_path,allow_pickle=False) as archive:
                if str(archive['binding'])!=frontier_binding: raise ValueError('Saved branches belong to different puzzle data or geometry.')
                prefixes=archive['prefixes'].copy(); lengths=archive['lengths'].copy()
                frontier_metadata=json.loads(str(archive['metadata']))
            if (prefixes.ndim!=2 or prefixes.shape[1]!=160 or lengths.shape!=(len(prefixes),)
                    or prefixes.dtype!=np.int16 or lengths.dtype!=np.int16
                    or np.any(lengths<0) or np.any(lengths>160)
                    or np.any(prefixes < -1) or np.any(prefixes>479)):
                raise ValueError('Malformed saved frontier.')
            for begin in range(0,len(prefixes),2048):
                rows=prefixes[begin:begin+2048]; sizes=lengths[begin:begin+2048]
                if np.any((rows>=0)!=(np.arange(160)[None,:]<sizes[:,None])):
                    raise ValueError('Saved frontier padding disagrees with its lengths.')
            prefix_hash=hashlib.sha256(prefixes.tobytes()+lengths.tobytes()).hexdigest()
            if prefix_hash!=frontier_metadata.get('prefixes_sha256') or frontier_metadata.get('cover_complete') is not True:
                raise ValueError('Saved frontier failed its coverage/integrity metadata check.')
            ensure_disjoint_prefixes([tuple(map(int,row[:length])) for row,length in zip(prefixes,lengths)])
        else:
            last_progress=[0.0]
            def progress(info):
                if time.monotonic()-last_progress[0]>=1:
                    status['frontier_progress']=info
                    publish('partitioning','Dividing the search into non-overlapping branches.')
                    last_progress[0]=time.monotonic()
            publish('partitioning','Dividing the search into non-overlapping branches.')
            plan=plan_frontier(masks,board.neighbor_cells,board.neighbor_sides,order,args.replicas,
                               stop_requested=stopping,progress=progress)
            if stopping(): raise StartupStopped()
            prefixes=plan.prefixes; lengths=plan.lengths; frontier_metadata=plan.metadata
            atomic_npz(frontier_path,{'prefixes':prefixes,'lengths':lengths,'order':order,
                                     'binding':np.array(frontier_binding),'metadata':np.array(json.dumps(frontier_metadata))})
        frontier_hash=digest(frontier_path)
        status['frontier_metadata']=frontier_metadata
        status.pop('frontier_progress',None)
        replicas=args.replicas
        if args.resume and checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as archive:
                if str(archive['exact_frontier_sha256'])!=frontier_hash or str(archive['exact_data_sha256'])!=data_sha:
                    raise ValueError('Systematic checkpoint does not match its saved frontier and puzzle.')
                # Saved lanes are repacked to the requested GPU population below.
        status['replicas']=replicas
        if stopping(): raise StartupStopped()
        if len(prefixes)==0:
            state='exhausted'; status.update(total_jobs=0,exhausted_jobs=0,active_jobs=0,paused_jobs=0,queued_jobs=0,unassigned_jobs=0,max_depth=0,candidates_checked=0)
            atomic_json(exact/'exhaustion.json',{'state':state,'frontier_sha256':frontier_hash,'source_sha256':hashes,'metadata':frontier_metadata})
            publish(); return
        publish('initializing_gpu','Preparing GPU lanes for disjoint branches.')
        engine=DFS
        if backend=='webgpu':
            from dfs_webgpu import WebGPUDFS
            engine=WebGPUDFS
        gpu=engine(masks,board.neighbor_cells,board.neighbor_sides,n=replicas,seed=args.seed,order=order,prefixes=[])
        if args.resume and checkpoint.exists():
            publish('restoring_checkpoint','Restoring saved branches; excess branches will pause with their progress preserved.')
            with np.load(checkpoint,allow_pickle=False) as archive:
                ledger=JobLedger.resume(gpu,archive,prefixes,lengths)
                ledger.validate_prefixes(gpu.prefix_codes,gpu.floors.get(),prefixes,lengths)
                generation=int(archive['exact_generation']); maximum=int(archive['exact_max_depth']); checked=int(archive['exact_candidates'])
                terminal=str(archive['exact_terminal'])
            if ledger.total!=len(prefixes): raise ValueError('Job ledger frontier size changed.')
        else:
            ledger=JobLedger.empty(len(prefixes),replicas); terminal=''
            publish('assigning_branches','Assigning a different fixed prefix to each active GPU lane.')
            ledger.refill(gpu,prefixes,lengths)
        trusted=True
        def snapshot():
            ledger.validate(gpu.states.get())
            ledger.validate_prefixes(gpu.prefix_codes,gpu.floors.get(),prefixes,lengths)
            payload=gpu.snapshot_checkpoint(); payload.update(ledger.fields())
            payload.update(exact_frontier_sha256=np.array(frontier_hash),exact_data_sha256=np.array(data_sha),
                           exact_generation=np.array(generation,np.int64),exact_max_depth=np.array(maximum,np.int16),
                           exact_candidates=np.array(checked,np.int64),
                           exact_terminal=np.array(state if state in ('solved','edge_perfect','exhausted') else ''))
            return payload
        writer=CheckpointWriter(snapshot,lambda payload,gen: atomic_npz(checkpoint,payload))
        config={'method':'systematic','backend':backend,'run_id':run_id,'pid':os.getpid(),'started_at':started.isoformat(),
                'replicas':gpu.n,'requested_replicas':args.replicas,'seconds':args.seconds,'seed':args.seed,
                'source_sha256':hashes,'frontier_sha256':frontier_hash,'total_jobs':ledger.total,
                'paused_jobs':ledger.paused_count,'checkpoint_seconds':args.checkpoint_seconds,'stop_on_matched_edges':240,'resumed':args.resume and checkpoint.exists()}
        atomic_json(run/'run-config.json',config); atomic_json(evidence/'run-config.json',config)
        if terminal not in ('','solved','edge_perfect','exhausted'):
            trusted=False
            raise ValueError('Unknown terminal checkpoint state.')
        if terminal=='exhausted':
            if ledger.completed!=ledger.total or np.any(ledger.ids>=0) or ledger.paused_count:
                trusted=False
                raise ValueError('Checkpoint claims exhaustion while branches remain unfinished.')
            state=terminal; publish(); return
        if terminal in ('solved','edge_perfect'):
            pending=np.flatnonzero(gpu.states.get()==1)
            if not len(pending):
                trusted=False
                raise ValueError('Terminal checkpoint has no pending full-board certificate.')
            idx=int(pending[0]); codes=gpu.boards[:,idx].get().astype(int).tolist()
            report=validate_arrangement(data,codes)
            if report['matched_edges']!=240 or not report['unique_tiles'] or (terminal=='solved' and not report['valid']):
                trusted=False
                raise ValueError('Terminal checkpoint failed independent full-board validation.')
            record={'solved':report['valid'],'codes':codes,'validation':report,'method':'systematic',
                    'run_id':run_id,'source_sha256':hashes,'input_snapshot':str(input_snapshot),
                    'job_id':int(ledger.ids[idx]),'replica':idx,'revalidated_previous_record':True}
            atomic_json(run/'best.json',record); atomic_json(run/'edge-perfect.json',record)
            render(run/'best.html',data['tiles'],codes,report)
            if report['valid']: atomic_json(run/'solution.json',record)
            state='solved' if report['valid'] else 'edge_perfect'
            status.update(best_matched_edges=240,best_components=report['line_components'])
            publish(); return
        state='running'; writer.finish(generation); publish()
        last_live=last_status=last_verify=0.0
        while (args.seconds==0 or time.monotonic()-began<args.seconds) and not stopping():
            trusted=False
            gpu.step(args.nodes); generation+=1
            flags=gpu.states.get()
            ledger.retire(flags)
            trusted=True
            pending=np.flatnonzero(flags==1)
            if len(pending):
                idx=int(pending[0]); codes=gpu.boards[:,idx].get().astype(int).tolist()
                report=validate_arrangement(data,codes)
                if report['matched_edges']!=240 or not report['unique_tiles']:
                    raise RuntimeError('A complete GPU candidate failed independent validation.')
                checked+=1; maximum=160
                record={'solved':report['valid'],'codes':codes,'validation':report,'method':'systematic',
                        'run_id':run_id,'source_sha256':hashes,'input_snapshot':str(input_snapshot),'job_id':int(ledger.ids[idx]),
                        'replica':idx,'elapsed_seconds':time.monotonic()-began}
                atomic_json(exact/f"candidate-job-{int(ledger.ids[idx]):09d}.json",record)
                atomic_json(run/'best.json',record); atomic_json(run/'edge-perfect.json',record)
                render(run/'best.html',data['tiles'],codes,report)
                if report['valid']: atomic_json(run/'solution.json',record)
                state='solved' if report['valid'] else 'edge_perfect'
                status.update(best_matched_edges=240,best_components=report['line_components'])
                break
            if stopping(): break
            trusted=False
            ledger.refill(gpu,prefixes,lengths)
            trusted=True
            if ledger.completed==ledger.total:
                state='exhausted'; break
            elapsed=time.monotonic()-began
            if elapsed-last_live>=1:
                depths=gpu.depths.get(); high=gpu.maxdepths.get()
                maximum=max(maximum,int(high.max()))
                active=np.flatnonzero(ledger.ids>=0)
                if len(active):
                    idx=int(active[np.argmax(depths[active])]); codes=gpu.boards[:,idx].get().astype(int).tolist()
                    assigned=int(depths[idx])
                    known=sum(codes[a]>=0 and codes[b]>=0 for a,sa,b,sb in board.edges)
                    atomic_json(run/'live.json',{'method':'systematic','run_id':run_id,'codes':codes,'is_partial':True,
                        'assigned_tiles':assigned,'known_matched_edges':known,'validation':None,'replica':idx,
                        'job_id':int(ledger.ids[idx]),'elapsed_seconds':elapsed,'updated_at':dt.datetime.now(dt.timezone.utc).isoformat()})
                last_live=elapsed
            if elapsed-last_verify>=30:
                gpu.verify(np.arange(min(16,gpu.n))); last_verify=elapsed
            writer.poll()
            if not stopping(): writer.request(generation,minimum_interval=args.checkpoint_seconds)
            if elapsed-last_status>=1:
                publish(); last_status=elapsed
        if state=='running': state='stopped'
        if state=='exhausted':
            atomic_json(exact/'exhaustion.json',{'state':state,'frontier_sha256':frontier_hash,'source_sha256':hashes,
                'completed_jobs':ledger.completed,'total_jobs':ledger.total,'candidates_checked':checked})
    except StartupStopped:
        state='stopped'
    except Exception as exc:
        state='error'; error=f'{type(exc).__name__}: {exc}'
        print(error,flush=True)
    finally:
        try:
            if gpu is not None and writer is not None and trusted:
                terminal_state=state
                status['state_after_save']=terminal_state
                state='saving_checkpoint'; publish()
                state=terminal_state
                try: writer.finish(generation)
                except Exception as exc:
                    state='error'; error=f'Checkpoint save failed: {exc}'
            if writer is not None: writer.close()
            publish()
            atomic_json(evidence/'final-status.json',status)
        finally:
            if lock.exists() and lock.read_text()==str(os.getpid()): lock.unlink()
    if error: raise SystemExit(error)

if __name__=='__main__': main()
