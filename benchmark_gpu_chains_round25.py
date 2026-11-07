"""Finite proposal34 work discriminator, not a timing or coverage claim.

Each paired population starts with identical native boards, RNG and scores.
Every lane advances exactly one proposal between independent Gold checks.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot
ROOT=Path(__file__).resolve().parent

def cases():
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles,brute
    output=[]
    for name in ('tetrahedron','singleton_domains'):
        problem=SearchProblem(*tetrahedron_tiles())
        if name=='singleton_domains':problem.domains=[[3*c] for c in range(problem.n)]
        solutions=set(brute(problem,'gold'))
        if not solutions:raise ValueError('Finite fixture unexpectedly lacks Gold solutions.')
        output.append((name,problem,solutions))
    return output

def run_case(problem,solutions,*,seed,replicas,backend,guided,max_proposals):
    import numpy as np
    from gpu_sampling import _inputs,_make_sampler,gold_score
    engine=_make_sampler(problem,*_inputs(problem,gold_aware=True),replicas,seed,backend,
        experiments={'ejection_chains':True,**({'obstruction_chains':True} if guided else {})})
    began=time.monotonic();steps=0;solution=None
    try:
        arrays=engine.snapshot_arrays();initial=hashlib.sha256(b''.join(np.asarray(a,dtype='<u4').tobytes() for a in (*arrays[:2],arrays[2][:4]))).hexdigest()
        identity=engine.runtime_identity()
        while True:
            boards,objectives=engine.top(count=replicas)
            for codes,objective in zip(boards,objectives):
                values=tuple(map(int,codes));metrics=gold_score(problem,values)
                if metrics['gold_objective']!=int(objective):raise ValueError('Finite GPU score mismatch.')
                if values in solutions:
                    report=problem.validate(values)
                    if not report['valid'] or metrics['domain_violations']:raise ValueError('Finite Gold certificate failed independent validation.')
                    solution=list(values)
            if solution is not None or steps==max_proposals:break
            if time.monotonic()-began>20:raise RuntimeError('Finite one-proposal case exceeded its20s guard; no work comparison claimed.')
            engine.launch(1);steps+=1
        return {'seed':seed,'guided':guided,'replicas':replicas,'initial_population_sha256':initial,'native_runtime':identity,
            'status':'solved' if solution is not None else 'capped','gold_codes':solution,'proposal_rounds':steps,'lane_proposals':steps*replicas,
            'initial_gold':solution is not None and steps==0,'counters':engine.experiment_stats()}
    finally:engine.close()

def compare(backend,seeds=12,replicas=8,max_proposals=64):
    reports=[]
    for name,problem,solutions in cases():
        pairs=[]
        for seed in range(1,seeds+1):
            baseline=run_case(problem,solutions,seed=seed,replicas=replicas,backend=backend,guided=False,max_proposals=max_proposals)
            guided=run_case(problem,solutions,seed=seed,replicas=replicas,backend=backend,guided=True,max_proposals=max_proposals)
            if baseline['initial_population_sha256']!=guided['initial_population_sha256'] or baseline['initial_gold']!=guided['initial_gold']:
                raise ValueError('Paired native initial state differs.')
            pairs.append({'baseline':baseline,'guided':guided})
        complete=[p for p in pairs if p['baseline']['status']==p['guided']['status']=='solved']
        reports.append({'fixture':name,'cells':problem.n,'fingerprint':problem.fingerprint,'domains':[list(r) for r in problem.domains],
            'independent_gold_count':len(solutions),'gold_set_sha256':hashlib.sha256(json.dumps(sorted(solutions)).encode()).hexdigest(),
            'pairs':pairs,'both_solved_pairs':len(complete),'both_solved_lane_proposals':{k:sum(p[k]['lane_proposals'] for p in complete) for k in ('baseline','guided')},
            'capped_pairs':len(pairs)-len(complete),'guided_orientation_checks':sum(p['guided']['counters']['guided_candidate_checks'] for p in pairs)})
    return reports

def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('cuda','webgpu'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development')
    parser.add_argument('--seeds',type=int,default=12);parser.add_argument('--replicas',type=int,default=8);parser.add_argument('--max-proposals',type=int,default=64);args=parser.parse_args()
    if not 1<=args.seeds<=64 or not 1<=args.replicas<=128 or not 1<=args.max_proposals<=256:raise ValueError('Finite bounds exceeded.')
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Output must be a fresh JSON path.')
    before=source_snapshot(ROOT);runtime=runtime_snapshot();repo=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty') is not False):raise RuntimeError('Frozen evidence requires a clean committed checkout.')
    results=compare(args.backend,args.seeds,args.replicas,args.max_proposals)
    after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repo_after=repository_snapshot(ROOT)
    if before!=after or runtime!=runtime_after or repo.get('commit')!=repo_after.get('commit'):raise RuntimeError('Finite evidence inputs changed.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository changed.')
    report={'proposal':34,'evidence_stage':args.evidence_stage,'evidence_status':'passed','backend':args.backend,'seeds':list(range(1,args.seeds+1)),
        'replicas':args.replicas,'max_proposals':args.max_proposals,'results':results,'source_before':before,'source_after':after,
        'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after,
        'assertion_tested':'Identical paired populations; independently checked finite Gold solutions; actual lane proposals until first observed Gold at each single-proposal boundary.',
        'limits':['Finite small synthetic fixtures, not the original puzzle; no timing winner or global coverage claim.',
                  'The first successful round still charges all replicas in that round; it is not a count of distinct arrangements.',
                  'Initial Gold populations cost zero move proposals. Capped cases are not infeasibility results.',
                  'Guidance orientation evaluations are extra work and are not added to proposal counts.',
                  'Production controllers normally collect less frequently; this discriminator checks every native proposal round.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    for row in results:print(json.dumps({k:row[k] for k in ('fixture','both_solved_pairs','both_solved_lane_proposals','capped_pairs','guided_orientation_checks')}),flush=True)
if __name__=='__main__':main()
