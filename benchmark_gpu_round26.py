"""Finite work comparisons for the complete round26 GPU exact-reasoning family.

Counts of different operations stay separate. Elapsed times are diagnostic only;
cooperative execution needs the separately controlled frozen performance matrix.
"""
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot


def exact_rows():
    from gpu_domain_round24 import DomainCUDA
    from test_gpu_domain_round24 import enumerate_engine
    from test_gpu_domain_round25 import independent_solutions,word_boundary_fixture,precise_reason_fixture,planted160_fixture
    from test_gpu_exact_round26 import inventory_fixture
    from test_constraint_dfs import tetrahedron_tiles
    from search_problem import SearchProblem
    cases=[('support','tetra',lambda:SearchProblem(*tetrahedron_tiles()),'gold'),
           ('support','word-boundary',word_boundary_fixture,'edges'),('support','prism',precise_reason_fixture,'gold'),
           ('watch','tetra',lambda:SearchProblem(*tetrahedron_tiles()),'gold'),('watch','prism',precise_reason_fixture,'gold'),
           ('matching','nonlocal-deficiency',lambda:inventory_fixture('failure'),'edges'),
           ('matching','unsupported-tile',inventory_fixture,'edges'),('matching','disjoint-components',lambda:inventory_fixture('components'),'edges'),
           ('cooperative','word-boundary',word_boundary_fixture,'edges'),('cooperative','controlled160',planted160_fixture,'gold')]
    rows=[]
    for feature,name,factory,target in cases:
        problem=factory();answers=independent_solutions(problem,target)
        for enabled in (False,True):
            policy=dict(joint_support=True,event_propagation=True,precise_reasons=True,support_metrics=True,clause_metrics=True)
            if feature=='support':policy['support_reuse']=enabled
            else:policy['support_reuse']=True
            if feature=='watch':policy['watched_clauses']=enabled
            elif feature in ('matching','cooperative'):policy['watched_clauses']=True
            if feature=='matching':policy['global_matching']=enabled
            elif feature=='cooperative':policy.update(global_matching=True,cooperative=enabled)
            engine=DomainCUDA(problem,target=target,**policy);began=time.monotonic()
            try:
                found=enumerate_engine(engine,71)
                if set(found)!=answers or len(found)!=len(answers):raise RuntimeError('Independent completed solution set differs.')
                rows.append({'feature':feature,'fixture':name,'enabled':enabled,'cells':problem.n,'target':target,'policy':policy,
                    'complete':True,'solutions':len(found),'stats':engine.stats(),'device':engine.device,
                    'diagnostic_seconds':time.monotonic()-began,'fingerprint':problem.fingerprint,
                    'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest()})
            finally:engine.close()
    return rows


def proof_rows():
    from gpu_domain_round24 import DomainCUDA
    from gpu_deduction_round26 import produce
    from gpu_proof_cache_round26 import ProofCache
    from gpu_shared_round25 import COUNTERS
    from constraint_shared_conflicts import VerifiedConflictStore
    from test_constraint_dfs import tetrahedron_tiles
    from search_problem import SearchProblem
    problem=SearchProblem(*tetrahedron_tiles());root=[sum(1<<v for v in row) for row in problem.domains];root[0]=root[1]=1;root[2]=1<<6
    engine=DomainCUDA(problem,[tuple(root)],proof_transport=True);trace=None
    try:
        for _ in range(1000):
            engine.step(1);trace=engine.deduction_input()
            if trace:break
        if trace is None:raise RuntimeError('No actual native failed trace.')
        native_stats=engine.stats();certificate,production=produce(problem,'gold',trace['root'],trace['decisions'],trace['trail'])
        if certificate is None:raise RuntimeError('No structural certificate for failed trace.')
    finally:engine.close()
    rows=[]
    for enabled in (False,True):
        store=VerifiedConflictStore(problem,'gold');stats=dict.fromkeys(COUNTERS,0)
        try:
            if enabled:
                cache=ProofCache(problem,'gold',stats);cache.offer(trace['identity'],certificate)
                for _ in range(100):
                    cache.advance(store)
                    if not cache.pending():break
                else:raise RuntimeError('Finite proof minimization unfinished.')
                saved=cache.snapshot();fresh=VerifiedConflictStore(problem,'gold');fresh.import_snapshot(store.snapshot())
                try:
                    restored=ProofCache(problem,'gold',stats,resume=saved)
                    while restored.pending():restored.advance(fresh)
                    if fresh.records()!=store.records() or fresh.stats['shared_conflicts_calls']!=0:raise RuntimeError('Durable structural replay differed.')
                    restore_stats=dict(fresh.stats)
                finally:fresh.close()
            else:
                store.offer(((0,0),(1,0)))
                for _ in range(100):
                    store.advance(seconds=.05,attempts=1,conflicts=10000)
                    if not store.pending_count:break
                else:raise RuntimeError('Finite independent SAT verification unfinished.')
                restore_stats=None
            if not store.records():raise RuntimeError('No independently established conclusion.')
            rows.append({'feature':'structural-proof','enabled':enabled,'fixture':'duplicate-piece-native-failure','complete':True,
                'native_producer_stats':native_stats,'structural_production_stats':production if enabled else None,
                'proof_stats':stats if enabled else None,'store_stats':dict(store.stats),'verified_records':store.records(),
                'durable_restore_store_stats':restore_stats,
                'scope':'Both approaches independently establish a globally invalid duplicate-piece conjunction. GPU failed-root preparation is charged separately; structural guard minimization/replay and SAT model/query work are different units.'})
        finally:store.close()
    return rows


def cooperative_rows(*,repeats=5,dispatches=32,quantum=16,replicas=128):
    """Paired fixed-work synchronized dispatch timing after separate warmup.

    Initial propagation is the measured phase. Status-copy cost is included;
    source compilation/allocation, state verification and closure are excluded
    and separately charged. No checkpoint or application Stop-latency claim.
    """
    import numpy as np
    from gpu_domain_round24 import DomainCUDA
    from gpu_domain_runner import split_cubes
    from test_gpu_domain_round25 import word_boundary_fixture
    from search_problem import SearchProblem
    root=Path(__file__).resolve().parent;data_path=root/'data/tiles.json';input_bytes=data_path.read_bytes()
    fixtures=(('word-boundary',word_boundary_fixture(),'edges'),('original160',SearchProblem(json.loads(input_bytes)),'gold'))
    rows=[]
    for name,problem,target in fixtures:
        cubes=split_cubes([tuple(sum(1<<v for v in domain) for domain in problem.domains)],replicas)
        policy=dict(target=target,joint_support=True,event_propagation=True,precise_reasons=True,
                    support_reuse=True,support_metrics=True,watched_clauses=True,global_matching=True)
        warm=[]
        for enabled in (False,True):
            began=time.perf_counter();engine=DomainCUDA(problem,cubes,cooperative=enabled,**policy)
            try:engine.step(1);warm.append({'cooperative':enabled,'constructor_and_one_step_seconds':time.perf_counter()-began})
            finally:engine.close()
        for repetition in range(repeats):
            pair=[];expected_initial=None;expected_final=None;expected_stats=None
            for enabled in ((False,True) if repetition%2==0 else (True,False)):
                began=time.perf_counter();engine=DomainCUDA(problem,cubes,cooperative=enabled,**policy)
                allocation=time.perf_counter()-began
                try:
                    keep=np.ones(engine.rows,bool)
                    keep[engine.offsets['support_counters']+6:engine.offsets['support_counters']+8]=False
                    initial=engine.state.get();initial_digest=hashlib.sha256(np.ascontiguousarray(initial[keep]).tobytes()).hexdigest()
                    if expected_initial is None:expected_initial=initial_digest
                    elif initial_digest!=expected_initial:raise RuntimeError('Cooperative paired initialization differs.')
                    statuses=initial[engine.offsets['meta']].copy();del initial
                    durations=[];active=[]
                    for _ in range(dispatches):
                        active.append(int(np.count_nonzero(statuses==0)))
                        began=time.perf_counter();statuses=engine.step(quantum);durations.append(time.perf_counter()-began)
                    measured=sum(durations);began=time.perf_counter();state=engine.state.get()
                    final_digest=hashlib.sha256(np.ascontiguousarray(state[keep]).tobytes()).hexdigest()
                    stats=engine.stats(snapshot=state);comparable={k:v for k,v in stats.items() if k!='support_word_scans'}
                    if expected_final is None:expected_final=final_digest;expected_stats=comparable
                    elif final_digest!=expected_final or comparable!=expected_stats:raise RuntimeError('Fixed cooperative work changes logical state or counters.')
                    # Any incidental candidate still requires the independent
                    # engine/raw validator. Search is deliberately not continued.
                    candidates=engine.candidates()
                    if any(target=='gold' and not row['validation']['valid'] for row in candidates):raise RuntimeError('Cooperative candidate failed Gold validation.')
                    verification=time.perf_counter()-began
                    pair.append({'cooperative':enabled,'device':engine.device,'lanes':engine.lanes,'state_bytes':engine.state_bytes,
                        'dispatches':dispatches,'quantum':quantum,'dispatch_seconds':durations,'total_dispatch_seconds':measured,
                        'active_lanes_before_dispatch':active,'active_lane_dispatches':sum(active),'final_lane_states':statuses.tolist(),
                        'constructor_seconds':allocation,'verification_seconds':verification,'initial_logical_sha256':initial_digest,
                        'final_logical_sha256':final_digest,'stats':stats,'independently_checked_candidates':len(candidates)})
                finally:engine.close()
            rows.append({'feature':'cooperative-timing','fixture':name,'cells':problem.n,'target':target,'repetition':repetition,
                'warmup':warm if repetition==0 else [],'ordered_pair':pair,'equal_logical_work':True,
                'fingerprint':problem.fingerprint,'domains_sha256':hashlib.sha256(json.dumps(problem.domains).encode()).hexdigest(),
                'input_sha256':hashlib.sha256(input_bytes).hexdigest() if name=='original160' else None,
                'scope':'Serial alternating paired order; fixed early-propagation dispatch work. Synchronized native launch plus status transfer is timed. Wider support-word reads remain charged. Finished/pending lanes can become idle; active counts are disclosed, and no deep-search, whole-run or Stop-latency conclusion follows.'})
    if data_path.read_bytes()!=input_bytes:raise RuntimeError('Original puzzle data changed during cooperative comparison.')
    return rows


def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development')
    parser.add_argument('--mode',choices=('finite','cooperative'),default='finite');parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--dispatches',type=int,default=32);parser.add_argument('--quantum',type=int,default=16);parser.add_argument('--replicas',type=int,choices=(1,128),default=128);args=parser.parse_args()
    if not 3<=args.repeats<=9 or not 1<=args.dispatches<=128 or not 1<=args.quantum<=64:raise ValueError('Invalid bounded cooperative comparison policy.')
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Fresh .json output required.')
    root=Path(__file__).resolve().parent;before=source_snapshot(root);runtime=runtime_snapshot();repo=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty')is not False):raise RuntimeError('Frozen evidence requires a clean committed checkout.')
    rows=exact_rows()+proof_rows() if args.mode=='finite' else cooperative_rows(repeats=args.repeats,dispatches=args.dispatches,quantum=args.quantum,replicas=args.replicas);after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime!=runtime_after:raise RuntimeError('Source/runtime changed during finite work comparison.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository identity changed.')
    report={'family':'F14','mode':args.mode,'evidence_stage':args.evidence_stage,'source_before':before,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,
            'repository_before':repo,'repository_after':repo_after,'source_unchanged':True,'runtime_unchanged':True,'results':rows,
            'scope':('Completed independent finite solution enumeration and proof work. Diagnostic seconds are not controlled performance evidence.' if args.mode=='finite' else 'Fixed-work paired serial cooperative comparison after separate warmup, with equal logical state checked. Timings require an externally quiet GPU/CPU for interpretation. No universal or whole-puzzle timing claim.')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:out.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'family':'F14','rows':len(rows),'passed':True}))
if __name__=='__main__':main()
