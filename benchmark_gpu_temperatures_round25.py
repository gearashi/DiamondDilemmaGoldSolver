"""Paired proposal37 Gold search with production feedback cadence.

Random initial populations are paired; planted witnesses certify the fixtures
but are never supplied to the sampler. Capped/adverse seeds remain in evidence.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot
ROOT=Path(__file__).resolve().parent

def summarize(pairs):
    both=[p for p in pairs if p['baseline']['status']=='solved' and p['adaptive']['status']=='solved']
    return {'pairs':len(pairs),'baseline_solved':sum(p['baseline']['status']=='solved' for p in pairs),
        'adaptive_solved':sum(p['adaptive']['status']=='solved' for p in pairs),'both_solved':len(both),
        'baseline_both_solved_proposals':sum(p['baseline']['lane_proposals'] for p in both),
        'adaptive_both_solved_proposals':sum(p['adaptive']['lane_proposals'] for p in both),
        'adaptive_both_solved_wins':sum(p['adaptive']['lane_proposals']<p['baseline']['lane_proposals'] for p in both),
        'adaptive_both_solved_losses':sum(p['adaptive']['lane_proposals']>p['baseline']['lane_proposals'] for p in both),
        'post_adjustment_wins':sum(p['adaptive']['status']=='solved' and p['adaptive']['controller_stats']['temperature_adjustments']>0 and
            (p['baseline']['status']!='solved' or p['adaptive']['lane_proposals']<p['baseline']['lane_proposals']) for p in pairs)}

def signature(engine):
    import numpy as np
    arrays=engine.snapshot_arrays()
    return hashlib.sha256(b''.join(np.asarray(a,dtype='<u4').tobytes() for a in (*arrays[:2],arrays[2][:18]))).hexdigest()

def run_case(problem,*,backend,seed,adaptive,replicas,max_dispatches):
    from gpu_sampling_resume_round25 import SamplerSession
    from gpu_sampling import gold_score
    policy={'ejection_chains':True,'replica_exchange':True,**({'adaptive_temperatures':True} if adaptive else {})}
    session=SamplerSession(problem,backend=backend,seed=seed,replicas=replicas,gold_aware=True,experiments=policy)
    try:
        session._ensure_engine();initial=signature(session.engine);identity=session.engine.runtime_identity();solution=None;prefix=None
        boards,scores=session.engine.top(count=replicas)
        for board,score in zip(boards,scores):
            metrics=gold_score(problem,board)
            if int(score)!=metrics['gold_objective']:raise ValueError('Initial score mismatch.')
            if not metrics['domain_violations'] and metrics['matched_edges']==len(problem.edges) and problem.validate(board)['valid']:solution=list(map(int,board))
        began=time.monotonic()
        while solution is None and session.launches<max_dispatches:
            # Record a matched pre-adjustment prefix at15, preserving the real
            # 8-dispatch collection and16-dispatch adaptation schedule.
            chunk=min(8,max_dispatches-session.launches)
            if session.launches==8:chunk=min(chunk,7)
            elif session.launches==15:chunk=1
            before=session.launches;session.advance(30,max_dispatches=chunk)
            if session.launches==15:prefix=signature(session.engine)
            if session.solution is not None:solution=list(session.solution)
            if session.launches==before:raise RuntimeError('Finite sampler made no dispatch progress.')
            if time.monotonic()-began>60:raise RuntimeError('Finite case exceeded60s guard.')
        report=session.report();best=report['top_board']
        if best is not None:
            metrics=gold_score(problem,best)
            if metrics['gold_objective']!=report['top_samples'][0]['gold_objective']:raise ValueError('Final score mismatch.')
        if solution is not None and not problem.validate(solution)['valid']:raise ValueError('Finite Gold certificate invalid.')
        stats=session.temperatures.stats if adaptive else {'temperature_adjustments':0}
        return {'status':'solved' if solution is not None else 'capped','solution':solution,'initial_gold':session.launches==0 and solution is not None,
            'initial_native_sha256':initial,'pre_adjustment_native_sha256':prefix,'dispatches':session.launches,'lane_proposals':session.launches*64*replicas,
            'native_runtime':identity,'controller_stats':stats,'native_counters':session.engine.experiment_stats(),
            'final_ladder':session.engine.temperature_feedback()['temperatures'] if adaptive else None,
            'best_matched_edges':report['best_score'],'best_objective':report['top_samples'][0]['gold_objective'] if report['top_samples'] else None}
    finally:session.close()

def comparisons(backend,seeds=8,replicas=8,max_dispatches=256):
    from search_fixtures import planted_gold
    from search_problem import SearchProblem
    results=[]
    for fixture_seed,extra in ((2537,0),(91,4),(17,2)):
        data,shape,witness=planted_gold(fixture_seed,size=1,extra_edges=extra);problem=SearchProblem(data,shape)
        if not problem.validate(witness)['valid']:raise ValueError('Planted fixture certificate invalid.')
        pairs=[]
        for seed in range(1,seeds+1):
            runs={}
            for adaptive in ((False,True) if seed%2 else (True,False)):
                runs['adaptive' if adaptive else 'baseline']=run_case(problem,backend=backend,seed=seed,adaptive=adaptive,replicas=replicas,max_dispatches=max_dispatches)
            if runs['baseline']['initial_native_sha256']!=runs['adaptive']['initial_native_sha256']:raise ValueError('Initial random populations differ.')
            a,b=(runs[name]['pre_adjustment_native_sha256'] for name in ('baseline','adaptive'))
            if a is not None and b is not None and a!=b:raise ValueError('Pre-adjustment trajectories differ.')
            pairs.append({'seed':seed,**runs})
        result={'fixture_seed':fixture_seed,'extra_edges':extra,'cells':problem.n,'fingerprint':problem.fingerprint,
            'fixture':'synthetic planted Gold; witness not supplied to sampler','planted_witness_sha256':hashlib.sha256(json.dumps(witness).encode()).hexdigest(),
            'pairs':pairs,'summary':summarize(pairs)};results.append(result)
        print(json.dumps({'fixture_seed':fixture_seed,**result['summary']}),flush=True)
    return results

def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('cuda','webgpu'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seeds',type=int,default=8);parser.add_argument('--max-dispatches',type=int,default=256)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if not 1<=args.seeds<=32 or args.max_dispatches<16 or args.max_dispatches>1024 or args.max_dispatches%8:raise ValueError('Invalid bounded finite matrix.')
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Output must be a fresh JSON file.')
    before=source_snapshot(ROOT);runtime=runtime_snapshot();repo=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty') is not False):raise RuntimeError('Frozen evidence requires clean committed sources.')
    rows=comparisons(args.backend,seeds=args.seeds,max_dispatches=args.max_dispatches)
    after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repo_after=repository_snapshot(ROOT)
    if before!=after or runtime!=runtime_after or repo.get('commit')!=repo_after.get('commit'):raise RuntimeError('Finite source/runtime changed.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository changed.')
    value={'proposal':37,'evidence_status':'passed','evidence_stage':args.evidence_stage,'backend':args.backend,'seeds':list(range(1,args.seeds+1)),
        'replicas':8,'max_dispatches':args.max_dispatches,'results':rows,'source_before':before,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,
        'repository_before':repo,'repository_after':repo_after,
        'assertion_tested':'Same initial random populations and pre-adjustment trajectories, independently validated Gold, production adaptation/collection cadence, all capped/adverse outcomes retained.',
        'limits':['Synthetic10-cell Gold fixtures, not the original puzzle; no whole solution-set enumeration claim.',
            'Goal detection after a dispatch collection gives proposal upper bounds; initial Gold is charged zero.',
            'Each dispatch attempts64 moves per lane; all adaptation burn-in proposals are included.',
            'Feedback/native transfer and CPU ladder work are additional work, not included in GPU proposal units.',
            'No stationary-distribution, exact coverage, universal speed or original-puzzle gain claim.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
