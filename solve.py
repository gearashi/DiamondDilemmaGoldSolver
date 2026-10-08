"""CUDA search with independent loop certificates, live snapshots and resumable states."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'runtime'/'cuda-cache'))
import numpy as np
from geometry import build_board
from validator import validate_arrangement,reverse_mask,normalize_tiles
from gpu_engine import GPU
from render_board import render

from io_utils import atomic_json
from position_cache import PositionCache
from checkpoint_writer import CheckpointWriter
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def oriented_masks(tiles):
    masks=np.zeros((len(tiles)*3,3),np.uint16)
    for t,tile in enumerate(tiles):
        for segment in tile['segments']:
            for s,p in segment:
                for r in range(3):masks[t*3+r,(s+r)%3]|=1<<(p-1)
    return masks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=3600,help='Run duration in seconds; 0 means unlimited.')
    ap.add_argument('--data',type=Path,default=ROOT/'data'/'tiles.json')
    ap.add_argument('--replicas',type=int,default=4096)
    ap.add_argument('--seed',type=int,default=20261007)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--stop-file-initialized',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--checkpoint-seconds',type=float,default=120,help='Minimum time between completed periodic checkpoints.')
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--cache-slots',type=int,default=64,help='Recent exact full boards retained per GPU replica; 0 disables the GPU cache.')
    ap.add_argument('--output',type=Path,default=ROOT/'runtime')
    args=ap.parse_args()
    if not math.isfinite(args.seconds) or args.seconds<0 or not 128<=args.replicas<=131072 or not 1<=args.steps<=128 or not 0<=args.cache_slots<=256 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<=0:
        ap.error('seconds must be 0 (unlimited) or positive, replicas128..131072, steps1..128, cache-slots0..256, checkpoint-seconds positive')
    if args.replicas*args.cache_slots*160>2147483647:
        ap.error('This replica/cache combination exceeds the CUDA cache index limit; reduce cache-slots.')
    run=args.output.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=run/'run.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit(f'{lock} exists. Check its PID; remove only if that process has ended.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=dt.datetime.now(dt.timezone.utc);run_id=started.strftime('%Y%m%dT%H%M%S%fZ')
    evidence=run/'runs'/run_id;evidence.mkdir(parents=True)
    start=time.monotonic();stop=False;state='starting';gpu=None;status={};position_cache=None;checkpoint_writer=None;generation=0
    best_score=-1;best_components=None;best_codes=None;best_record=None
    last_save=last_check=last_reseed=last_verify=last_live=0.
    candidates_checked=0;history=[];pending_cursor=0
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    stopfile=run/'stop.request'
    if not args.stop_file_initialized:stopfile.unlink(missing_ok=True)
    try:
        # Preserve the previous status before publishing this process's startup metadata.
        previous_status=run/'status.json'
        if previous_status.exists():(evidence/'previous-status.json').write_bytes(previous_status.read_bytes())
        def startup(phase,message):
            status.update(state='starting',pid=os.getpid(),run_id=run_id,replicas=args.replicas,
                          time_limit_seconds=args.seconds,elapsed_seconds=round(time.monotonic()-start,2),
                          startup_phase=phase,phase_message=message,updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(run/'status.json',status)
        startup('loading_input','Loading puzzle data')
        input_bytes=args.data.read_bytes();data_sha=hashlib.sha256(input_bytes).hexdigest()
        raw=json.loads(input_bytes.decode('utf-8-sig'));tiles=raw['tiles'];normalize_tiles(tiles)
        if len(tiles)!=160:raise ValueError('Gold requires160tiles')
        input_dir=run/'inputs';input_dir.mkdir(exist_ok=True)
        snapshot=input_dir/(data_sha+'.json')
        if not snapshot.exists():snapshot.write_bytes(input_bytes)
        (evidence/'tiles.json').write_bytes(input_bytes)
        board=build_board();masks=oriented_masks(tiles)
        inventory=Counter(int(v)for row in masks[::3]for v in row)
        bad=[m for m,c in inventory.items()if c!=inventory[reverse_mask(m)]or(m==reverse_mask(m)and c%2)]
        if bad:raise ValueError(f'Tile edge inventory cannot close: {len(bad)} unpaired signature types. Audit the source transcription first.')
        source_files=['solve.py','gpu_engine.py','kernels.cu','geometry.py','validator.py','render_board.py','extract_tiles.py','audit_endpoints.py','io_utils.py','position_cache.py','checkpoint_writer.py']
        hashes={}
        for name in source_files:
            if (ROOT/name).exists():
                body=(ROOT/name).read_bytes();(evidence/name).write_bytes(body)
                hashes[name]=hashlib.sha256(body).hexdigest()
        hashes['tiles.json']=data_sha
        cache_namespace=hashlib.sha256(json.dumps({key:hashes[key] for key in ('tiles.json','geometry.py','validator.py')},sort_keys=True).encode()).hexdigest()
        position_cache=PositionCache(run/'positions.sqlite3',cache_namespace)
        def checked_report(codes):
            report,hit=position_cache.get_or_compute(codes,lambda:validate_arrangement(tiles,codes))
            # A cache can save work, but every stopping candidate gets a fresh certificate.
            if report['matched_edges']==240:
                report=validate_arrangement(tiles,codes)
            return report,hit
        atomic_json(run/'geometry.json',board.as_dict())
        # Archive old certificates before a new run can create differently sourced results.
        for name in ('solution.json','edge-perfect.json','run-config.json'):
            old=run/name
            if old.exists():
                (evidence/('previous-'+name)).write_bytes(old.read_bytes())
                if name in ('solution.json','edge-perfect.json'):old.unlink()
        startup('initializing_gpu','Preparing GPU replicas')
        gpu=GPU(masks,board.neighbor_cells,board.neighbor_sides,n=args.replicas,seed=args.seed,cache_slots=args.cache_slots)
        checkpoint=run/'checkpoint.npz';checkpoint_meta=run/'checkpoint-meta.json'
        resumed=False;resume_meta=None;checkpoint_sha=None;seeded_from_checkpoint=False
        if args.resume and checkpoint.exists():
            checkpoint_sha=sha(checkpoint)
            if checkpoint_meta.exists():resume_meta=json.loads(checkpoint_meta.read_text())
            # GPU.resume rechecks every stored permutation, score, and puzzle fingerprint.
            startup('restoring_checkpoint','Restoring saved search and validating its cache')
            resumed=gpu.resume(checkpoint)
        if not resumed:
            startup('preparing_population','Preparing the search population')
            if args.resume and checkpoint.exists():
                # A changed population size cannot resume per-replica RNG state, but all
                # saved placements can seed the new population after full rescoring.
                with np.load(checkpoint,allow_pickle=False) as saved:
                    current=saved['boards'].T
                    historical=saved['bestboards'].T
                    if (not np.array_equal(gpu.score_cpu(current),saved['scores'])
                            or not np.array_equal(gpu.score_cpu(historical),saved['bestscores'])
                            or current.shape!=historical.shape
                            or np.any(saved['bestscores']<saved['scores'])):
                        raise ValueError('Population-resize checkpoint failed independent scoring.')
                    seeds=np.concatenate((historical,current),axis=0)
                shutil.copy2(checkpoint,evidence/'starting-checkpoint.npz')
                gpu.initialize(seeds=seeds,perturbations=0)
                seeded_from_checkpoint=True
            else:gpu.initialize()
        print(f'GPU: {gpu.device}; replicas={args.replicas}; resumed={resumed}; seeded_from_checkpoint={seeded_from_checkpoint}',flush=True)
        config={'run_id':run_id,'started_at':started.isoformat(),'pid':os.getpid(),'command':sys.argv,'source_sha256':hashes,'input_snapshot':str(snapshot),'replicas':args.replicas,'seed':args.seed,'seconds':args.seconds,'gpu':gpu.device,'cache_slots':args.cache_slots,'cache_namespace':cache_namespace,'stop_on_matched_edges':240,'checkpoint_seconds':args.checkpoint_seconds,'checkpoint_compressed':False,'resumed':resumed,'seeded_from_checkpoint':seeded_from_checkpoint,'resumed_checkpoint_sha256':checkpoint_sha,'resume_metadata':resume_meta}
        atomic_json(run/'run-config.json',config);atomic_json(evidence/'run-config.json',config)
        state='running'
        status.pop('phase_message',None);status.pop('startup_phase',None)
        def write_best(codes,report,elapsed,revalidated=False):
            nonlocal best_score,best_components,best_codes,best_record
            best_score=report['matched_edges'];best_components=report['line_components'];best_codes=list(codes)
            best_record={'solved':report['valid'],'codes':best_codes,'validation':report,'elapsed_seconds':elapsed,'source_sha256':hashes,'input_snapshot':str(snapshot),'seed':args.seed,'gpu':gpu.device,'run_id':run_id,'revalidated_previous_record':revalidated}
            atomic_json(run/'best.json',best_record);render(run/'best.html',tiles,best_codes,report)
            if report['matched_edges']==240:atomic_json(run/'edge-perfect.json',best_record)
            if report['valid']:atomic_json(run/'solution.json',best_record)
            history.append({'elapsed_seconds':round(elapsed,2),'score':best_score,'components':best_components})
            atomic_json(run/'history.json',history)
        previous=run/'best.json'
        if args.resume and previous.exists():
            prior=json.loads(previous.read_text());report,_=checked_report(prior['codes'])
            if report['matched_edges']!=240:report=validate_arrangement(tiles,prior['codes'])
            if not report['unique_tiles']:raise ValueError('Previous best record has invalid tile use')
            write_best(prior['codes'],report,0,True)
            gpu.seeds=np.asarray([best_codes],np.int16)
            if report['matched_edges']==240:
                state='solved' if report['valid'] else 'edge_perfect';stop=True
        def observe(final=False):
            nonlocal best_score,best_components,best_codes,candidates_checked,state,stop,pending_cursor
            elapsed=time.monotonic()-start
            scores=gpu.bestscores.get();top=int(scores.max())
            pending=gpu.pending_indices()
            ordered=np.concatenate((pending[pending>=pending_cursor],pending[pending<pending_cursor]))
            selected=ordered if final else ordered[:256]
            if len(selected):pending_cursor=(int(selected[-1])+1)%gpu.n
            if not len(selected) and top>best_score:selected=np.array([int(np.argmax(scores))])
            for idx in selected:
                codes=gpu.bestboards[:,int(idx)].get().astype(int).tolist()
                report,hit=checked_report(codes)
                if not hit:candidates_checked+=1
                score=int(scores[int(idx)])
                if not report['unique_tiles']or report['matched_edges']!=score:raise RuntimeError('GPU candidate failed independent validation')
                comps=report['line_components']
                if score>best_score or(score==240 and(best_components is None or comps<best_components)):
                    write_best(codes,report,elapsed)
                    print(f'{elapsed:.1f}s: {score}/240 matched edges; components={comps}; verified_solution={report["valid"]}',flush=True)
                if score==240:
                    # The user requested a stop at the first full edge match, even with multiple loops.
                    state='solved' if report['valid'] else 'edge_perfect';stop=True
                    if best_record is not None:
                        atomic_json(run/'edge-perfect.json',best_record)
                        if report['valid']:atomic_json(run/'solution.json',best_record)
                    break
                gpu.acknowledge([int(idx)])
        def write_snapshot(payload,captured_generation):
            metrics=GPU.write_checkpoint(checkpoint,payload,compressed=False)
            atomic_json(checkpoint_meta,{'source_sha256':hashes,'replicas':args.replicas,'run_id':run_id,'generation':captured_generation,'saved_at':dt.datetime.now(dt.timezone.utc).isoformat(),'checkpoint_metrics':metrics})
            return metrics
        checkpoint_writer=CheckpointWriter(gpu.snapshot_checkpoint,write_snapshot,clock=time.monotonic)
        def publish(elapsed):
            nonlocal status
            counters=gpu.counters.get().sum(axis=1).astype(int).tolist()
            status={'state':state,'pid':os.getpid(),'run_id':run_id,'elapsed_seconds':round(elapsed,2),'time_limit_seconds':args.seconds,'best_matched_edges':best_score,'required_matched_edges':240,'best_components':best_components,'candidates_checked':candidates_checked,'gpu':gpu.device,'replicas':args.replicas,'counters':counters,'duplicate_cache':{'gpu':gpu.cache_stats(),'disk':position_cache.stats()},'checkpoint':dict(checkpoint_writer.info),'source_sha256':hashes,'updated_at':dt.datetime.now(dt.timezone.utc).isoformat()}
            atomic_json(run/'status.json',status)
        while not stop and (args.seconds==0 or time.monotonic()-start<args.seconds):
            # Honor Stop before launching or doing any optional maintenance.
            if stopfile.exists():stop=True;break
            checkpoint_writer.poll()
            elapsed=time.monotonic()-start
            gpu.cool(elapsed);gpu.step(args.steps);generation+=1
            # A perfect pending candidate has priority over a simultaneous Stop.
            if gpu.has_pending():
                observe()
                if stop:break
            if stop or stopfile.exists():stop=True;break
            if elapsed-last_live>=.5 or last_live==0:
                idx=0;codes=gpu.boards[:,idx].get().astype(int).tolist()
                report,_=checked_report(codes)
                if report['matched_edges']!=int(gpu.scores[idx].get()):raise RuntimeError('Live GPU state failed validation')
                atomic_json(run/'live.json',{'codes':codes,'validation':report,'replica':idx,'elapsed_seconds':round(elapsed,2),'run_id':run_id})
                last_live=elapsed
            if stop or stopfile.exists():stop=True;break
            if elapsed-last_check>=2 or last_check==0:
                observe();publish(elapsed);last_check=elapsed
            if stop or stopfile.exists():stop=True;break
            if elapsed-last_verify>=15 or last_verify==0:
                gpu.verify_device_scores(np.arange(min(16,args.replicas)));last_verify=elapsed
            if stop or stopfile.exists():stop=True;break
            if elapsed-last_reseed>=60:
                if best_codes is not None:gpu.seeds=np.asarray([best_codes],np.int16)
                gpu.reseed(fraction=.125);generation+=1;last_reseed=time.monotonic()-start
            if stop or stopfile.exists():stop=True;break
            checkpoint_writer.request(generation,minimum_interval=args.checkpoint_seconds)
        if state not in ('solved','edge_perfect'):observe(final=True)
        final_state=state if state in ('solved','edge_perfect') else ('stopped' if stop else 'time_limit')
        state='stopping'
        checkpoint_writer.info['phase']='saving'
        publish(time.monotonic()-start)
        gpu.verify_device_scores(np.arange(min(64,args.replicas)))
        checkpoint_writer.finish(generation)
        state=final_state
        publish(time.monotonic()-start)
        status['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(run/'status.json',status);atomic_json(evidence/'status.json',status)
        if best_record is not None:atomic_json(evidence/'best.json',best_record)
        print(f'Finished: {state}; best {best_score}/240. Saved {run}',flush=True)
        return 0
    except BaseException as exc:
        status.update(state='error',error=str(exc),elapsed_seconds=round(time.monotonic()-start,2))
        atomic_json(run/'status.json',status);atomic_json(evidence/'status.json',status)
        raise
    finally:
        try:
            if checkpoint_writer is not None:checkpoint_writer.close()
            if position_cache is not None:position_cache.close()
        finally:lock.unlink(missing_ok=True)
if __name__=='__main__':raise SystemExit(main())
