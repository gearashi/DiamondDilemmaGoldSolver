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
import zipfile
import numpy as np
from geometry import build_board
from validator import orientation_masks, validate_arrangement
from dfs_gpu import DFS, fill_order, ensure_disjoint_prefixes
from gpu_backends import choose_backend
from gpu_reasoning import feature_identity, normalize_features
from exact_frontier import plan_frontier
import systematic_evidence as proof
from systematic_jobs import JobLedger
from classic_checkpoint_ranges import CheckpointRanges
from classic_parked_storage import ParkedCheckpointWriter, REFERENCE, restore_parked
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
            compressed = 'exact_source_origin' in fields
            output = proof.HashedStream(stream) if compressed else stream
            if compressed:
                with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for name, value in fields.items():
                        with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                            np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
                    parked_bytes = sum(item.compress_size for item in archive.infolist()
                                       if item.filename.startswith('exact_paused_'))
            else:
                np.savez(output, **fields)
            stream.flush()
            os.fsync(stream.fileno())
        size=temporary.stat().st_size
        if isinstance(output,proof.HashedStream):
            directory = proof.run_dir(path.parent.parent,str(fields['exact_source_origin']))
            generation = int(fields['exact_generation'])
            suffix = '-parked' if REFERENCE in fields else ''
            atomic_json(directory/f'checkpoint-{generation}{suffix}.json',
                        {'sha256':output.hash.hexdigest(),'bytes':size,'source_binding':str(fields['exact_source_binding'])})
        replace_with_retry(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'compressed':compressed,'file_bytes':size,'write_seconds':time.monotonic()-started}
    if compressed:
        result.update(sha256=output.hash.hexdigest(), parked_member_bytes=parked_bytes)
    return result

class AdaptiveBatchBudget:
    """Tune work per synchronized launch without altering per-lane DFS order."""
    def __init__(self, initial=32, target_ms=50.0, adaptive=True):
        if type(initial) is not int or not 1 <= initial <= 512:
            raise ValueError('Initial GPU work budget must be 1..512.')
        if not math.isfinite(target_ms) or target_ms <= 0:
            raise ValueError('GPU batch target must be finite and positive.')
        self.nodes = initial
        self.target_ms = float(target_ms)
        self.adaptive = bool(adaptive)
        self.cost_ms = None
        self.last_ms = None

    def observe(self, elapsed_ms):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return self.nodes
        self.last_ms = float(elapsed_ms)
        if not self.adaptive:
            return self.nodes
        cost = elapsed_ms / self.nodes
        self.cost_ms = cost if self.cost_ms is None else .75 * self.cost_ms + .25 * cost
        # React immediately to a slow launch; grow at most twofold to avoid a
        # large jump when easy lanes suddenly enter harder branching regions.
        estimate = max(cost, self.cost_ms)
        desired = max(1, min(512, int(self.target_ms / estimate)))
        self.nodes = min(desired, 2 * self.nodes)
        return self.nodes

class StartupStopped(Exception): pass

def terminal_certificate(gpu, ledger, data, terminal, target):
    """Find an independently verified pending certificate, including parked jobs."""
    def candidates():
        for lane in np.flatnonzero(gpu.states.get()==1):
            idx=int(lane)
            yield int(ledger.ids[idx]),idx,gpu.boards[:,idx].get()
        bank=ledger.paused.remaining_arrays()
        ids=ledger.paused.remaining_ids()
        for lane in np.flatnonzero(bank['states']==1):
            idx=int(lane)
            yield int(ids[idx]),None,bank['boards'][:,idx]

    first=None
    for job,replica,board in candidates():
        codes=board.astype(int).tolist()
        report=validate_arrangement(data,codes)
        if report['matched_edges']!=240 or not report['unique_tiles']:
            raise ValueError('Terminal checkpoint failed independent full-board validation.')
        current=job,replica,codes,report
        if report['valid'] or (terminal=='edge_perfect' and target=='edges'):
            return current
        if first is None:
            first=current
    if first is None:
        raise ValueError('Terminal checkpoint has no pending full-board certificate.')
    if terminal=='solved':
        raise ValueError('Terminal checkpoint failed independent full-board validation.')
    return first


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=3600,help='Run duration in seconds; 0 means unlimited.')
    parser.add_argument('--replicas',type=int,default=131072)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--backend',choices=('auto','cuda','webgpu'),default='auto',help='CUDA for NVIDIA; WebGPU for Metal/Vulkan/DirectX; auto detects the platform.')
    parser.add_argument('--gpu-features',type=json.loads,default=None,help='Experimental reasoning switches; each policy has an isolated resumable frontier.')
    parser.add_argument('--nodes',type=int,default=32,help='Initial GPU work budget per launch (1..512).')
    parser.add_argument('--fixed-nodes',action='store_true',help='Disable adaptive GPU batch sizing for reproducible comparisons.')
    parser.add_argument('--target',choices=('gold','edges'),default='gold',help='Gold requires one complete loop; edges is an explicit diagnostic search.')
    parser.add_argument('--checkpoint-seconds',type=float,default=30)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--host-terminal-only',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--stop-file-initialized',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--data',type=Path,default=ROOT/'data'/'tiles.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runtime')
    args=parser.parse_args()
    try:
        backend=args.backend if args.resume else choose_backend(args.backend)
        gpu_features=normalize_features(args.gpu_features)
        if args.target!='gold' and any(gpu_features.values()):raise ValueError('Experimental GPU reasoning profiles require the Gold target.')
    except ValueError as exc: parser.error(str(exc))
    if not math.isfinite(args.seconds) or args.seconds<0 or not 1<=args.replicas<=131072 or not 1<=args.nodes<=512 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<=0:
        parser.error('Seconds must be 0 (unlimited) or positive; checkpoint interval positive, replicas 1..131072, nodes 1..512 required.')
    batches=AdaptiveBatchBudget(args.nodes,adaptive=not args.fixed_nodes)
    run=args.output.resolve(); run.mkdir(parents=True,exist_ok=True)
    profile=proof.profile_name(gpu_features)
    exact=run/profile; exact.mkdir(exist_ok=True)
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
            'run_id':run_id,'backend':backend,'replicas':args.replicas,'gpu_features':gpu_features,'search_profile':profile,'time_limit_seconds':args.seconds,'target':args.target,
            'no_repeat_scope':'Disjoint branches and saved DFS cursors; crashes may replay since the last checkpoint.'}
    hashes={}; frontier_hash=None; frontier_metadata={}; previous_best=None
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    def stopping(): return stop or stopfile.exists()
    def dispatch_stopped():
        return stopping() or (args.seconds>0 and time.monotonic()-began>=args.seconds)
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
            counts=ledger.counter_totals(gpu) if ledger is not None else gpu.counters.sum(axis=1,dtype=np.uint64).get()
            status.update(gpu=gpu.device,replicas=gpu.n,nodes_checked=int(counts[0]),counters=[int(v) for v in counts])
        status.update(gpu_work_budget=batches.nodes,last_gpu_batch_ms=batches.last_ms,
                      gpu_batch_target_ms=batches.target_ms,adaptive_gpu_batches=batches.adaptive)
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
        hashes = proof.capture(ROOT, evidence, data_sha)
        status['source_sha256'] = hashes
        if (run/'best.json').exists():
            previous_best=json.loads((run/'best.json').read_text(encoding='utf-8-sig'))
            report=validate_arrangement(data,previous_best['codes'])
            status['best_matched_edges']=report['matched_edges']; status['best_components']=report['line_components']
            status['required_matched_edges']=240
            if report['matched_edges']==240 and (report['valid'] or args.target=='edges'):
                if report['valid'] and not (run/'solution.json').exists():
                    atomic_json(run/'solution.json',{**previous_best,'solved':True,'validation':report,
                                                     'revalidated_previous_record':True})
                state='solved' if report['valid'] else 'edge_perfect'
                publish(); return
        if stopping(): raise StartupStopped()
        if args.host_terminal_only and proof.terminal_features(checkpoint) is None:
            raise ValueError('A current completed Gold checkpoint is required for host verification.')
        if checkpoint.exists() and not args.resume:
            raise ValueError('A systematic checkpoint already exists. Resume it to preserve completed work.')
        frontier_binding=hashlib.sha256(masks.tobytes()+neighbors.tobytes()+neighbor_sides.tobytes()+order.tobytes()+data_sha.encode()).hexdigest()
        saved_frontier = frontier_path.exists()
        if saved_frontier:
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
        legacy_stack=False
        if args.resume and checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as archive:
                if str(archive['exact_frontier_sha256'])!=frontier_hash or str(archive['exact_data_sha256'])!=data_sha:
                    raise ValueError('Systematic checkpoint does not match its saved frontier and puzzle.')
                legacy_stack = 'exact_source_origin' not in archive.files and 'exact_source_binding' not in archive.files
                # Saved lanes are repacked to the requested GPU population below.
        status['replicas']=replicas
        if stopping(): raise StartupStopped()
        scope = {'frontier_sha256': frontier_hash, 'input_sha256': data_sha, 'target': args.target,
                 'gpu_policy': feature_identity(gpu_features)}
        source_binding = proof.origin_binding(hashes, scope)
        origin_verified = True
        marker = exact/'exhaustion.json'
        def verify_partition():
            target = frontier_metadata.get('target_jobs')
            if type(target) is not int or not 1 <= target <= 131072:
                raise ValueError('Invalid saved systematic partition size.')
            plan = plan_frontier(masks, neighbors, neighbor_sides, order, target, stop_requested=stopping)
            if stopping(): raise StartupStopped()
            if (not plan.metadata.get('cover_complete') or not np.array_equal(plan.prefixes,prefixes)
                    or not np.array_equal(plan.lengths,lengths)):
                raise ValueError('Current planner refutes the coverage of the saved systematic frontier.')

        if saved_frontier and not checkpoint.exists() and len(prefixes):
            verify_partition()
        if len(prefixes)==0:
            # An empty legacy partition has no GPU stack. Replay the deterministic planner
            # before accepting its claim, even when its own prefix digest is self-consistent.
            if marker.exists() and proof.read(marker).get('schema') == proof.SCHEMA:
                old = proof.read(marker)
                old_source = proof.verify_origin(run, old['run_id'], old.get('origin_binding'), hashes, scope)
                if old != proof.record(old_source,scope,old['run_id'],0,0):
                    raise ValueError('Empty frontier exhaustion evidence differs.')
            else:
                if saved_frontier:
                    verify_partition()
                atomic_json(evidence/'run-config.json', {'run_id':run_id, 'source_sha256':hashes, 'verified_coverage_origin':True})
                atomic_json(marker, proof.record(hashes,scope,run_id,0,0))
            state='exhausted'; status.update(total_jobs=0,exhausted_jobs=0,active_jobs=0,paused_jobs=0,queued_jobs=0,unassigned_jobs=0,max_depth=0,candidates_checked=0)
            publish(); return
        if args.resume and checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as archive:
                if (str(archive['exact_terminal']) == 'exhausted' and 'exact_source_origin' in archive
                        and 'exact_source_binding' in archive and str(archive['exact_source_binding'])):
                    status.update(proof.reopen_terminal(run,checkpoint,marker,archive,hashes,scope,len(prefixes)))
                    state = 'exhausted'
                    publish()
                    return
        if args.host_terminal_only:
            raise ValueError('A current completed checkpoint is required for verification without a GPU backend.')
        if args.resume:
            backend=choose_backend(args.backend)
            status['backend']=backend
        publish('initializing_gpu','Preparing GPU lanes for disjoint branches.')
        engine=DFS
        if backend=='webgpu':
            from dfs_webgpu import WebGPUDFS
            engine=WebGPUDFS
        engine_options={'gpu_features':gpu_features} if any(gpu_features.values()) else {}
        if backend=='webgpu' and legacy_stack:
            engine_options['reserve_replay']=True
        gpu=engine(masks,board.neighbor_cells,board.neighbor_sides,n=replicas,seed=args.seed,order=order,
                   prefixes=[],gold_data=data if args.target=='gold' else None,**engine_options)
        prior_backends=set()
        if args.resume and checkpoint.exists():
            publish('restoring_checkpoint','Restoring saved branches; excess branches will pause with their progress preserved.')
            with np.load(checkpoint,allow_pickle=False) as archive:
                previous_scope=proof.checkpoint_scope(archive,scope)
                proof.check_cpu_scope(archive,gpu_features)
                # A predecessor without a bound contribution history, including
                # a migrated legacy stack, can have traversed both kernels.
                prior_backends.update(previous_scope.get('proof_backends',('cuda','webgpu')))
                with restore_parked(checkpoint,archive) as stack:
                    ledger=JobLedger.resume(gpu,stack,prefixes,lengths)
                external_bank = REFERENCE in archive.files
                ledger.validate_prefixes(gpu.prefix_codes,gpu.floors.get(),prefixes,lengths)
                generation=int(archive['exact_generation']); maximum=int(archive['exact_max_depth']); checked=int(archive['exact_candidates'])
                terminal=str(archive['exact_terminal'])
                old_origin = str(archive['exact_source_origin']) if 'exact_source_origin' in archive else None
                old_binding = str(archive['exact_source_binding']) if 'exact_source_binding' in archive else None
                old_record = str(archive['exact_exhaustion_record']) if 'exact_exhaustion_record' in archive else ''
            closed = ledger.completed == ledger.total
            if terminal in ('exhausted','unverified_exhaustion') and not closed:
                raise ValueError('Checkpoint claims exhaustion while branches remain unfinished.')
            if old_origin is None and old_binding is None:
                summary = {'total_jobs':ledger.total, 'exhausted_jobs':ledger.completed, 'assigned_jobs':ledger.cursor,
                           'candidates_checked':checked, 'max_depth':maximum,
                           'counters':[int(v) for v in ledger.counter_totals(gpu)]}
                history = proof.legacy_origin(run, evidence, scope, gpu_features, summary,
                                                      checkpoint.stat().st_size, generation, terminal, marker)
                if history:
                    verify_partition()
                origin_verified = bool(history and closed and terminal == 'exhausted' and len(history)==1)
                if history and not origin_verified:
                    replay_inputs = dict(masks=masks,neighbors=board.neighbor_cells,neighbor_sides=board.neighbor_sides,
                                         order=order,gold_data=data if args.target=='gold' else None)
                    if getattr(gpu, 'backend', None) == 'webgpu':
                        replay_inputs['reuse_from'] = gpu
                    origin_verified = proof.verify_native_history(engine, replay_inputs, gpu_features, checkpoint,
                                                                 history, prefixes, lengths, stopping)
                    if stopping(): raise StartupStopped()
                    status['historical_stack_replayed'] = origin_verified
                # A truly untouched job ledger contains no prior GPU pruning to attest.
                if (not closed and generation == checked == ledger.completed == 0 and not ledger.paused_count
                        and not any(summary['counters']) and not terminal
                        and np.array_equal(gpu.depths.get(),gpu.floors.get())
                        and np.all(gpu.cursors.get()==65535)):
                    verify_partition()
                    origin_verified = True
                if origin_verified and closed:
                    if marker.exists():
                        (evidence/'previous-exhaustion.json').write_bytes(marker.read_bytes())
                    status['historical_exhaustion_verified'] = True
            else:
                try:
                    proof.check_receipt(run,old_origin,generation,checkpoint,old_binding,parked=external_bank)
                    previous_source = proof.verify_origin(run, old_origin, old_binding, hashes, previous_scope,
                                                        required_backend=backend if not closed else None)
                except (KeyError, TypeError, OSError, ValueError) as exc:
                    origin_verified = False
                    if closed and terminal != 'unverified_exhaustion':
                        raise ValueError('Systematic exhaustion source evidence failed: '+str(exc)) from exc
                if closed and origin_verified:
                    proof.check_terminal_record(run,old_origin,previous_source,previous_scope,ledger.total,checked,old_record,marker)
            if closed and terminal in ('', 'exhausted', 'unverified_exhaustion'):
                terminal = 'exhausted' if origin_verified else 'unverified_exhaustion'

            if ledger.total!=len(prefixes): raise ValueError('Job ledger frontier size changed.')
        else:
            ledger=JobLedger.empty(len(prefixes),replicas); terminal=''
            publish('assigning_branches','Assigning a different fixed prefix to each active GPU lane.')
            ledger.refill(gpu,prefixes,lengths)
        # Include the engine entrusted with initializing or restoring native state.
        # Earlier untracked stacks retain both engines throughout later resumes.
        used_backends=sorted(prior_backends | {backend})
        scope={**scope,'proof_backends':used_backends}
        source_binding=proof.origin_binding(hashes,scope)
        trusted=True
        ranges=CheckpointRanges(gpu,ledger)
        def snapshot():
            payload=ranges.capture()
            ledger.validate(payload['states'])
            ledger.validate_prefixes(payload['prefix_codes'],payload['floors'],prefixes,lengths)
            payload.update(ledger.fields())
            payload.update(exact_proof_backends=np.array(json.dumps(used_backends)),exact_gpu_features=np.array(json.dumps(feature_identity(gpu_features),sort_keys=True)),exact_frontier_sha256=np.array(frontier_hash),exact_data_sha256=np.array(data_sha),
                           exact_generation=np.array(generation,np.int64),exact_max_depth=np.array(maximum,np.int16),
                           exact_candidates=np.array(checked,np.int64),
                           exact_source_origin=np.array(run_id),exact_source_target=np.array(args.target),
                           exact_source_binding=np.array(source_binding if origin_verified else ''),
                           exact_exhaustion_record=np.array(proof.record(hashes,scope,run_id,ledger.total,checked)['sha256']
                                                            if state=='exhausted' else ''),
                           exact_terminal=np.array(state if state in ('solved','edge_perfect','exhausted','unverified_exhaustion') else ''))
            return payload, ledger.paused, ledger.paused.offset, state=='running'
        parked_writer=ParkedCheckpointWriter(checkpoint,atomic_npz)
        writer=CheckpointWriter(snapshot,lambda packet,gen: parked_writer.write(packet))
        config={'method':'systematic','backend':backend,'run_id':run_id,'pid':os.getpid(),'started_at':started.isoformat(),
                'replicas':gpu.n,'requested_replicas':args.replicas,'gpu_features':gpu_features,'search_profile':profile,'seconds':args.seconds,'seed':args.seed,
                'source_sha256':hashes,'verified_coverage_origin':origin_verified,'proof_backends':used_backends,'frontier_sha256':frontier_hash,'total_jobs':ledger.total,
                'paused_jobs':ledger.paused_count,'checkpoint_seconds':args.checkpoint_seconds,
                'target':args.target,'stop_on_gold_solution':args.target=='gold',
                'adaptive_gpu_batches':batches.adaptive,'gpu_batch_target_ms':batches.target_ms,
                'stop_on_matched_edges':240 if args.target=='edges' else None,'resumed':args.resume and checkpoint.exists()}
        atomic_json(run/'run-config.json',config); atomic_json(evidence/'run-config.json',config)
        status['verified_coverage_origin'] = origin_verified
        if not origin_verified:
            status['coverage_note'] = 'Saved work is available for finding Gold. Its earlier branch exclusions have unverified source evidence.'
        if terminal not in ('','solved','edge_perfect','exhausted','unverified_exhaustion'):
            trusted=False
            raise ValueError('Unknown terminal checkpoint state.')
        if terminal in ('exhausted','unverified_exhaustion'):
            if ledger.completed!=ledger.total or np.any(ledger.ids>=0) or ledger.paused_count:
                trusted=False
                raise ValueError('Checkpoint claims exhaustion while branches remain unfinished.')
            state=terminal
            trusted = bool(status.get('historical_exhaustion_verified'))
            publish(); return
        if terminal in ('solved','edge_perfect'):
            trusted=False
            job,idx,codes,report=terminal_certificate(gpu,ledger,data,terminal,args.target)
            record={'solved':report['valid'],'codes':codes,'validation':report,'method':'systematic',
                    'run_id':run_id,'source_sha256':hashes,'input_snapshot':str(input_snapshot),
                    'job_id':job,'replica':idx,'revalidated_previous_record':True}
            atomic_json(run/'best.json',record); atomic_json(run/'edge-perfect.json',record)
            render(run/'best.html',data['tiles'],codes,report)
            if report['valid']: atomic_json(run/'solution.json',record)
            state='solved' if report['valid'] else 'edge_perfect'
            status.update(best_matched_edges=240,best_components=report['line_components'])
            if report['valid'] or args.target=='edges':
                publish()
                trusted=True
                return
            trusted=True
            # A legacy edge-perfect terminal is unfinished for the Gold target.
            # Its frozen candidate is retained until the first Gold dispatch
            # proves and unwinds the closed proper component.
            terminal=''
        state='running'; writer.finish(generation); publish()
        last_live=last_status=last_verify=0.0
        # Larger pools did not show a stable full-iteration benefit. Use the
        # actual restored/resized engine size and retain ordinary step above it.
        if hasattr(gpu,'_page_stop'):
            gpu._page_stop=dispatch_stopped
        step_and_states=getattr(gpu,'step_and_states',None) if gpu.n<=4096 else None
        while not dispatch_stopped():
            trusted=False
            batch_started=time.perf_counter()
            if step_and_states is None:
                gpu.step(batches.nodes); generation+=1
                flags=gpu.states.get()
            else:
                # WebGPU's required state read also completes the launch.
                # A failed read leaves trusted false, preserving the old checkpoint.
                flags=step_and_states(batches.nodes); generation+=1
            if hasattr(gpu,'cpu_assist_stats'):status['cpu_assist']=gpu.cpu_assist_stats()
            batches.observe((time.perf_counter()-batch_started)*1000)
            ledger.retire(flags)
            trusted=True
            pending=np.flatnonzero(flags==1)
            # A Stop request can arrive as a page completes. Check every
            # already-pending board for Gold before acknowledging or stopping.
            for lane in pending:
                trusted=False
                idx=int(lane); codes=gpu.boards[:,idx].get().astype(int).tolist()
                report=validate_arrangement(data,codes)
                if report['matched_edges']!=240 or not report['unique_tiles']:
                    raise RuntimeError('A complete GPU candidate failed independent validation.')
                trusted=True
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
                if report['valid'] or args.target=='edges':
                    break
                state='running'
            if state in ('solved','edge_perfect'):
                break
            if len(pending):
                # Independent validation remains authoritative even for a
                # legacy/mocked device that emits a multi-loop full board.
                trusted=False
                gpu.acknowledge(pending.tolist())
                ledger.retire(gpu.states.get())
                trusted=True
            if ledger.completed==ledger.total:
                state='exhausted' if origin_verified else 'unverified_exhaustion'; break
            if dispatch_stopped(): break
            # retire(flags) already validated this completed launch. Reading and
            # validating every lane again cannot assign work unless both a free
            # lane and queued/paused jobs exist.
            if (ledger.paused_count or ledger.cursor<ledger.total) and np.any(ledger.ids<0):
                trusted=False
                ledger.refill(gpu,prefixes,lengths)
                trusted=True
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
                try:
                    writer.finish(generation)
                    if writer.info.get('parked_bank_external'):
                        writer.info.update(parked_writer.portable())
                    if state=='exhausted':
                        atomic_json(marker,proof.record(hashes,scope,run_id,ledger.total,checked))
                except Exception as exc:
                    state='error'; error=f'Checkpoint save failed: {exc}'
            if writer is not None: writer.close()
            publish()
            atomic_json(evidence/'final-status.json',status)
        finally:
            if lock.exists() and lock.read_text()==str(os.getpid()): lock.unlink()
    if error: raise SystemExit(error)

if __name__=='__main__': main()
