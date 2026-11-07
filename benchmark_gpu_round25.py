"""Serial finite GPU discriminator records; not an original-puzzle speed claim."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions


def _load_fixtures():
    # Main takes its before-snapshot before loading solver or fixture modules.
    require_assertions()
    global DomainCUDA,joint_hall_fixture,repeated_neighbor_fixture,word_boundary_fixture,planted160_fixture,independent_solutions,fork_gold_fixture,precise_reason_fixture,enumerate_engine
    from gpu_domain_round24 import DomainCUDA
    from test_gpu_domain_round25 import joint_hall_fixture,repeated_neighbor_fixture,word_boundary_fixture,planted160_fixture,independent_solutions,fork_gold_fixture,precise_reason_fixture
    from test_gpu_domain_round24 import enumerate_engine


def _hardware_snapshot():
    import cupy as cp
    return {'cuda_device_index':int(cp.cuda.Device().id),'cuda_driver_api_version':int(cp.cuda.runtime.driverGetVersion()),
            'cuda_runtime_version':int(cp.cuda.runtime.runtimeGetVersion())}

def run26():
    _load_fixtures()
    rows=[]
    for name,factory,target in [('joint-neighbor-inventory',joint_hall_fixture,'edges'),('conjoined-repeated-seams',repeated_neighbor_fixture,'edges'),('tile-word-boundary',word_boundary_fixture,'edges'),('controlled-planted160',planted160_fixture,'gold')]:
        problem=factory();expected=independent_solutions(problem,target)
        for enabled in (False,True):
            engine=DomainCUDA(problem,target=target,joint_support=enabled);began=time.monotonic()
            try:
                found=enumerate_engine(engine,17)
                if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Independent complete enumeration differs.')
                rows.append({'fixture':name,'joint_support':enabled,'target':target,'cells':problem.n,'solutions':len(found),'complete':True,'stats':engine.stats(),'wall_seconds':time.monotonic()-began,'device':engine.device,'input_fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:engine.close()
    return rows

def run27():
    _load_fixtures()
    from gpu_domain_runner import split_cubes
    from constraint_work import WorkScope
    rows=[]
    for name,factory,target in [('root-strand-reuse',fork_gold_fixture,'gold'),('joint-neighbor-inventory',joint_hall_fixture,'edges'),('tile-word-boundary',word_boundary_fixture,'edges')]:
        problem=factory();expected=independent_solutions(problem,target)
        for enabled in (False,True):
            parent=None;engine=None;began=time.monotonic();root_stats={}
            try:
                if enabled:
                    parent=DomainCUDA(problem,target=target,joint_support=True)
                    for _ in range(10000):
                        states,ready=parent.propagate_root(17)
                        if ready:break
                    else:raise RuntimeError('Preparation stalled')
                    root_stats=parent.stats();engine=parent.fork_root(split_cubes(parent.remaining_domain_cubes(),8))
                else:engine=DomainCUDA(problem,split_cubes([WorkScope.for_problem(problem,target).root],8),target=target,joint_support=True)
                found=enumerate_engine(engine,17)
                if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Independent complete enumeration differs.')
                stats=engine.stats();keys=('decisions','domain_removals','support_checks','strand_joins','joint_word_scans','joint_tile_visits','joint_subset_checks')
                rows.append({'fixture':name,'propagated_forks':enabled,'target':target,'cells':problem.n,'solutions':len(found),'complete':True,
                    'stats':{key:stats[key]+root_stats.get(key,0) for key in keys},'parent_stats':root_stats,'child_stats':stats,
                    'wall_seconds':time.monotonic()-began,'device':engine.device,'input_fingerprint':problem.fingerprint,
                    'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:
                if engine is not None:engine.close()
                if parent is not None:parent.close()
    return rows

def run28():
    _load_fixtures()
    rows=[]
    for name,factory,target in [('root-strand-reuse',fork_gold_fixture,'gold'),('joint-neighbor-inventory',joint_hall_fixture,'edges'),('tile-word-boundary',word_boundary_fixture,'edges'),('controlled-planted160',planted160_fixture,'gold')]:
        problem=factory();expected=independent_solutions(problem,target)
        for enabled in (False,True):
            engine=DomainCUDA(problem,target=target,joint_support=True,event_propagation=enabled);began=time.monotonic()
            try:
                found=enumerate_engine(engine,17)
                if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Independent complete enumeration differs.')
                rows.append({'fixture':name,'event_propagation':enabled,'target':target,'cells':problem.n,'solutions':len(found),'complete':True,'stats':engine.stats(),'wall_seconds':time.monotonic()-began,'device':engine.device,'input_fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:engine.close()
    return rows

def run29():
    _load_fixtures()
    rows=[]
    for name,factory,target in [('prism-independent-failure',precise_reason_fixture,'edges'),('prism-gold-failure',precise_reason_fixture,'gold'),('tile-word-boundary',word_boundary_fixture,'edges'),('controlled-planted160',planted160_fixture,'gold')]:
        problem=factory();expected=independent_solutions(problem,target)
        configs=[(False,None),(True,None),(True,8)] if problem.n<160 else [(False,None),(True,None)]
        for enabled,slots in configs:
            engine=DomainCUDA(problem,target=target,precise_reasons=enabled,reason_slots=slots);began=time.monotonic()
            try:
                found=enumerate_engine(engine,17)
                if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Independent complete enumeration differs.')
                rows.append({'fixture':name,'precise_reasons':enabled,'reason_slots':engine.reason_slots if enabled else None,'target':target,'cells':problem.n,'solutions':len(found),'complete':True,'stats':engine.stats(),'wall_seconds':time.monotonic()-began,'device':engine.device,'input_fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:engine.close()
    return rows

def run30():
    _load_fixtures()
    from constraint_work import WorkScope,subtract,disjoint
    from constraint_shared_conflicts import VerifiedConflictStore
    from gpu_domain_runner import split_cubes
    from test_constraint_shared_conflicts import finish
    problem=joint_hall_fixture();expected=independent_solutions(problem,'edges');cube=list(WorkScope.for_problem(problem,'edges').root)
    cube[1]=(1<<0)|(1<<9);cube[2]=cube[3]=(1<<0)|(1<<6)
    source=DomainCUDA(problem,[tuple(cube)],target='edges');store=VerifiedConflictStore(problem,'edges');rows=[]
    try:
        for _ in range(3000):
            if any(source.step(1)==1):break
        else:raise RuntimeError('Finite source did not reach its first candidate')
        claims,_,_=source.learned_claims(limit=32);source_stats=source.stats()
        for claim in claims:store.offer(claim)
        finish(store)
        if not store.records() or not store.stats['shared_conflicts_rejected']:raise RuntimeError('Expected globally verified and locally-only claims were not distinguished.')
        parts=list(subtract(WorkScope.for_problem(problem,'edges').root,tuple(cube)))
        if not disjoint([tuple(cube),*parts]):raise RuntimeError('Producer and recipient ownership overlap.')
        expected={answer for answer in expected if not all(mask&(1<<code) for mask,code in zip(cube,answer))}
        for enabled in (False,True):
            engine=DomainCUDA(problem,parts,target='edges',event_propagation=True,precise_reasons=True);began=time.monotonic()
            try:
                imported=engine.import_verified_clauses(store) if enabled else {'inserted':0,'comparisons':0}
                found=enumerate_engine(engine,17)
                if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Independent complete enumeration differs.')
                rows.append({'fixture':'verified-cross-lane-neighbor-inventory','shared_clauses':enabled,'cells':problem.n,'target':'edges',
                    'solutions':len(found),'complete':True,'completion_scope':'Recipient cubes only; producer prefix remains unfinished.',
                    'producer_cube':[hex(x) for x in cube],'recipient_cubes':[[hex(x) for x in row] for row in parts],'receiver_stats':engine.stats(),'producer_prefix_stats':source_stats,
                    'verification_setup_stats':dict(store.stats),'verification_scope':'Original domains; producer prefix and verification setup are reported separately, not free work.',
                    'gpu_import':imported,'claims':claims,'verified_clauses':store.records(),'wall_seconds':time.monotonic()-began,
                    'device':engine.device,'input_fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:engine.close()
    finally:source.close();store.close()
    return rows

def run31():
    _load_fixtures()
    from unittest.mock import patch
    from gpu_domain_runner import solve,split_cubes
    from constraint_work import WorkScope,frontier_snapshot
    problem=precise_reason_fixture();expected=independent_solutions(problem,'edges')
    if expected:raise RuntimeError('The independent complete prism enumeration must be infeasible.')
    scope=WorkScope.for_problem(problem,'edges');frontier=frontier_snapshot(scope,split_cubes([scope.root],12));rows=[]
    original_step=DomainCUDA.step
    for quantum in (1,8,17,64):
        for enabled in (False,True):
            counts={'dispatches':0,'idle_lane_quanta':0};device=[]
            def step(engine,work):
                counts['dispatches']+=1;counts['idle_lane_quanta']+=int((engine.state[0].get()==2).sum())*quantum
                device[:]=[engine.device];return original_step(engine,quantum)
            began=time.monotonic()
            with patch.object(DomainCUDA,'step',step):
                result=solve(problem,target='edges',seconds=30,replicas=4,work_frontier=frontier,
                    options={'gpu_lane_balance':enabled,'gpu_event_propagation':True,'gpu_precise_reasons':True})
            if result['status']!='infeasible' or not result['complete'] or result['checkpoint']['frontier']['boxes']:
                raise RuntimeError('Lane balancing did not exhaust the independently infeasible full fixture.')
            rows.append({'fixture':'prism-twelve-queued-roots','lane_balance':enabled,'cells':problem.n,'target':'edges',
                'solutions':0,'complete':True,'quantum':quantum,'lanes':4,'input_roots':12,**counts,'stats':result['stats'],
                'measurement_scope':'Fixed primitive budget per dispatch. Idle lane quanta measure unused scheduling capacity, not GPU instructions or time.',
                'wall_seconds':time.monotonic()-began,'device':device[0],'input_fingerprint':problem.fingerprint,
                'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
    return rows

def run32():
    _load_fixtures()
    import tempfile,base64
    from gpu_domain_runner import split_cubes,pack_native,COUNTERS
    from gpu_checkpoint_codec_round25 import encode_native,decode_native
    from gpu_checkpoint_round25 import CheckpointStore,resolve_checkpoint
    from constraint_work import WorkScope,frontier_snapshot,digest
    rows=[]
    for name,factory,target in [('prism-idle-lane-reuse',precise_reason_fixture,'edges'),('controlled-planted160-live-prefixes',planted160_fixture,'gold')]:
        problem=factory();scope=WorkScope.for_problem(problem,target);expected=independent_solutions(problem,target)
        engine=DomainCUDA(problem,split_cubes([scope.root],4),target=target,event_propagation=True,precise_reasons=True)
        try:
            if problem.n<160:
                for _ in range(3000):
                    states=engine.step(1)
                    if any(states==2) and any(states==0):break
                else:raise RuntimeError('No finite mixed idle/active checkpoint barrier.')
            else:engine.step(8)
            with tempfile.TemporaryDirectory() as folder:
                store=CheckpointStore(folder,scope.payload())
                for generation in (1,2):
                    if generation==2:engine.step(17 if problem.n<160 else 1)
                    snapshot=engine.checkpoint();native,blobs,metrics=encode_native(engine,snapshot,store.cache)
                    payload={'version':1,'scope':scope.payload(),'native':native,'queue':[],'totals':dict.fromkeys(COUNTERS,0),
                        'completed_lanes':0,'active_stats':engine.stats(snapshot=snapshot['state']),
                        'frontier':frontier_snapshot(scope,engine.remaining_domain_cubes(snapshot=snapshot['state']))}
                    payload['digest']=digest(payload);before=dict(store.stats);ref=store.write(payload,blobs);restored=resolve_checkpoint(ref,folder)
                    legacy=pack_native(engine,snapshot);legacy_payload={**payload,'native':legacy};legacy_payload['digest']=digest({k:v for k,v in legacy_payload.items() if k!='digest'})
                    other=DomainCUDA(problem,engine.cubes,target=target,event_propagation=True,precise_reasons=True)
                    try:
                        other.restore(decode_native(restored['native'],other));found=enumerate_engine(other,17 if problem.n<160 else 256)
                        if set(found)!=expected or len(found)!=len(expected):raise RuntimeError('Checkpoint continuation lost an independent solution.')
                    finally:other.close()
                    rows.append({'fixture':name,'generation':generation,'cells':problem.n,'lanes':engine.lanes,'target':target,
                        'solutions':len(found),'complete':True,'completion_scope':'Each decoded snapshot independently completes its exact remaining search.',
                        'native_state_bytes':snapshot['state'].nbytes,'packed_live_bytes':metrics['packed_bytes'],'omitted_padding_bytes':metrics['omitted_bytes'],
                        'legacy_compressed_native_bytes':len(base64.b64decode(legacy['compressed_state'])),
                        'factored_compressed_native_bytes':metrics['compressed_bytes'],'reused_compression_chunks':metrics['reused_chunks'],
                        'new_chunk_objects':store.stats['written_chunks']-before['written_chunks'],
                        'logical_write_bytes':store.stats['written_bytes']-before['written_bytes'],
                        'legacy_payload_json_bytes':len(json.dumps(legacy_payload,sort_keys=True,separators=(',',':')).encode()),
                        'retained_store_bytes':store._usage()[0],
                        'copy_scope':'One GPU-to-host native snapshot supplies encoding, frontier and statistics. Legacy path copies the full native state twice. Byte counts omit filesystem block/journal overhead.',
                        'device':engine.device,'input_fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
        finally:engine.close()
    return rows

def main():
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--proposal',type=int,choices=[26,27,28,29,30,31,32],default=26);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.output.suffix.lower()!='.json' or args.output.exists() or args.output.is_symlink():raise ValueError('GPU benchmark output must be a fresh .json path.')
    root=Path(__file__).resolve().parent;names=('gpu_domain_round24.py','domain_search_round24.cu','gpu_domain_runner.py','test_gpu_domain_round25.py','test_gpu_domain_round24.py','test_constraint_dfs.py','search_problem.py','search_fixtures.py','validator.py','geometry.py','gpu_shared_round25.py','gpu_lane_pool_round25.py','gpu_checkpoint_round25.py','gpu_checkpoint_codec_round25.py','test_gpu_checkpoint_round25.py','constraint_shared_conflicts.py','constraint_sat_assist.py','constraint_knowledge.py','constraint_pruning.py','constraint_work.py','test_constraint_shared_conflicts.py',Path(__file__).name)
    all_before=source_snapshot(root);runtime_before=runtime_snapshot();repository_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repository_before.get('commit') or repository_before.get('dirty') is not False):
        raise RuntimeError('Frozen GPU evidence requires a clean, readable committed checkout.')
    before={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names};rows={26:run26,27:run27,28:run28,29:run29,30:run30,31:run31,32:run32}[args.proposal]();after={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    if before!=after:raise RuntimeError('Sources changed during finite benchmark.')
    all_after=source_snapshot(root);runtime_after=runtime_snapshot();repository_after=repository_snapshot(root)
    if all_before!=all_after or runtime_before!=runtime_after:raise RuntimeError('Sources or runtime changed during finite GPU evaluation.')
    if args.evidence_stage=='frozen' and repository_before!=repository_after:raise RuntimeError('Git identity changed during frozen GPU evaluation.')
    import importlib.metadata
    try:sat_version=importlib.metadata.version('python-sat')
    except importlib.metadata.PackageNotFoundError:sat_version=None
    report={'python_sat':sat_version,'evidence_stage':args.evidence_stage,'source_before':all_before,'source_after':all_after,'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repository_before,'repository_after':repository_after,'source_unchanged':all_before==all_after,'runtime_unchanged':runtime_before==runtime_after,'hardware':_hardware_snapshot(),'proposal':args.proposal,'python':sys.version,'platform':platform.platform(),'scope':'Completed finite CUDA enumeration; independent raw-board validation. No claim about original160-cell solution speed.','source_sha256':before,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'proposal':args.proposal,'evidence_stage':args.evidence_stage,'rows':len(rows),'source_files':len(all_before),'source_unchanged':True,'runtime_unchanged':True}))
if __name__=='__main__':main()
