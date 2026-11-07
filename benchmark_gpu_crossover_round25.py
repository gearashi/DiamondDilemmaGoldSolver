"""Proposal35 finite native integration, given explicitly prepared parent inputs.

This measures recombination after parents are supplied, not discovery of those
parents. Fixture targets are planted and are not original-puzzle solutions.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot
ROOT=Path(__file__).resolve().parent

def fixtures():
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles,brute
    from search_fixtures import planted_gold
    rows=[]
    for name,parents in (('physical_cycles',[[3,0,6,9],[0,3,9,6]]),('rotation_cycles',[[1,3,6,9],[0,4,6,9]])):
        p=SearchProblem(*tetrahedron_tiles());rows.append((name,p,parents,set(brute(p,'gold'))))
    data,geometry,witness=planted_gold(2535,size=4,extra_edges=12);p=SearchProblem(data,geometry)
    a=witness.copy();a[0],a[17]=a[17],a[0];b=witness.copy();b[93],b[100]=b[100],b[93]
    rows.append(('nearby_160',p,[a,b],{tuple(witness)}))
    for name,p,parents,solutions in rows:
        if any(p.validate(board)['valid'] for board in parents):raise ValueError('Prepared parents unexpectedly include Gold.')
        if not all(p.validate(board)['valid'] for board in solutions):raise ValueError('Fixture target failed independent validation.')
    return rows

def prepare(engine,problem,parents):
    import numpy as np
    from gpu_sampling import gold_score
    arrays=engine.snapshot_arrays()
    for lane in range(engine.r):
        values=parents[lane%len(parents)];arrays[0][:,lane]=values;arrays[1][:,lane]=values
        arrays[2][1:3,lane]=gold_score(problem,values)['gold_objective']
    engine.restore_arrays(arrays)
    return hashlib.sha256(b''.join(np.asarray(row,dtype='<u4').tobytes() for row in arrays)).hexdigest()

def baseline(problem,parents,solutions,*,backend,seed,replicas,max_rounds):
    from gpu_sampling import _make_sampler,_inputs,gold_score
    engine=_make_sampler(problem,*_inputs(problem,gold_aware=True),replicas,seed,backend,experiments={'ejection_chains':True})
    try:
        initial=prepare(engine,problem,parents);rounds=0;began=time.monotonic();solution=None
        while True:
            boards,scores=engine.top(count=replicas)
            for codes,score in zip(boards,scores):
                values=tuple(map(int,codes));metrics=gold_score(problem,values)
                if int(score)!=metrics['gold_objective']:raise ValueError('Prepared baseline score failed independent check.')
                if values in solutions:
                    if not problem.validate(values)['valid']:raise ValueError('Baseline Gold validation failed.')
                    solution=list(values)
            if solution is not None or rounds==max_rounds:break
            if time.monotonic()-began>20:raise RuntimeError('Finite baseline exceeded20s guard.')
            engine.launch(1);rounds+=1
        return {'initial_native_sha256':initial,'status':'solved' if solution is not None else 'capped','gold_codes':solution,
            'native_proposal_rounds':rounds,'gpu_lane_proposals':rounds*replicas,'initial_gold':rounds==0 and solution is not None,'native_counters':engine.experiment_stats()}
    finally:engine.close()

def recombine(problem,parents,solutions,*,backend,seed,replicas):
    from gpu_sampling_resume_round25 import SamplerSession
    from sampling_crossover_round25 import CycleCrossover
    settings={'backend':backend,'seed':seed,'replicas':replicas,'gold_aware':True,'experiments':{'ejection_chains':True,'cycle_crossover':True}}
    session=SamplerSession(problem,**settings);restored=None
    try:
        session._ensure_engine();initial=prepare(session.engine,problem,parents);identity=session.engine.runtime_identity();session.collect()
        if session.solution is None or tuple(session.solution) not in solutions or not problem.validate(session.solution)['valid']:raise ValueError('Actual archive crossover did not recover the independently checked fixture Gold.')
        saved=json.loads(json.dumps(session.snapshot()));stats=session.crossover.stats
        if stats['cycle_parent_scores']!=2 or len(session.crossover_parents)<2:raise ValueError('Both nearby parents did not survive actual collection.')
        restored=SamplerSession(problem,resume=saved,**settings);restored.advance(1)
        if restored.engine is not None or restored.crossover.snapshot()!=saved['crossover'] or restored.counts['move_proposals']!=0:raise ValueError('Finished crossover replayed work or allocated a new GPU.')
        reuse=CycleCrossover(problem,gold_aware=True,resume=saved['crossover']);before=reuse.stats;again=reuse.propose([{'codes':v} for v in parents])
        if again['candidate'] is not None or reuse.stats['cycle_candidates']!=before['cycle_candidates']:raise ValueError('Saved exact child history repeated child scoring.')
        return {'initial_native_sha256':initial,'status':'solved','gold_codes':session.solution,'gpu_lane_proposals':session.counts['move_proposals'],
            'native_runtime':identity,'crossover_counters':stats,'saved_parent_pool_size':len(saved['crossover_parents']),
            'finished_resume_allocated_gpu':False,'repeated_parent_child_score_delta':reuse.stats['cycle_candidates']-before['cycle_candidates'],
            'repeated_parent_rescore_delta':reuse.stats['cycle_parent_scores']-before['cycle_parent_scores'],
            'checkpoint_sha256':hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()}
    finally:
        session.close()
        if restored:restored.close()

def comparisons(backend,seeds=4,replicas=4,max_rounds=64):
    rows=[]
    for name,p,parents,solutions in fixtures():
        paired=[]
        for seed in range(1,(seeds if p.n==4 else 1)+1):
            base=baseline(p,parents,solutions,backend=backend,seed=seed,replicas=replicas,max_rounds=max_rounds) if p.n==4 else None
            child=recombine(p,parents,solutions,backend=backend,seed=seed,replicas=replicas)
            if base is not None and base['initial_native_sha256']!=child['initial_native_sha256']:raise ValueError('Prepared native states differ.')
            paired.append({'seed':seed,'baseline':base,'crossover':child})
        rows.append({'fixture':name,'cells':p.n,'fingerprint':p.fingerprint,'prepared_parents':parents,
            'target_scope':'complete independent finite enumeration' if p.n==4 else 'one independently validated planted witness; no enumeration claim',
            'known_gold_count':len(solutions),'gold_sha256':hashlib.sha256(json.dumps(sorted(solutions)).encode()).hexdigest(),'pairs':paired,
            'solved_baseline_pairs':sum(x['baseline'] is not None and x['baseline']['status']=='solved' for x in paired),
            'solved_baseline_gpu_lane_proposals':sum(x['baseline']['gpu_lane_proposals'] for x in paired if x['baseline'] is not None and x['baseline']['status']=='solved')})
    return rows

def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('cuda','webgpu'),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Output must be a fresh JSON file.')
    before=source_snapshot(ROOT);runtime=runtime_snapshot();repo=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty') is not False):raise RuntimeError('Frozen evidence requires clean committed sources.')
    rows=comparisons(args.backend);after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repo_after=repository_snapshot(ROOT)
    if before!=after or runtime!=runtime_after or repo.get('commit')!=repo_after.get('commit'):raise RuntimeError('Finite source/runtime changed.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository changed.')
    result={'proposal':35,'evidence_stage':args.evidence_stage,'evidence_status':'passed','backend':args.backend,'seeds':list(range(1,5)),'replicas':4,'max_proposal_rounds':64,
        'results':rows,'source_before':before,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after,
        'assertion_tested':'Given explicit non-Gold parents, actual native archive collection produces validated finite Gold children, preserves bounded durable history, and does not replay finished work.',
        'limits':['Prepared parents are stipulated inputs derived from known synthetic witnesses; their discovery is not measured.',
                  'Each native engine first initializes a random population, which is discarded by explicit fixture preparation; this setup is not a search gain.',
                  'CPU parent/child scoring and native GPU proposals are different work units and are reported separately.',
                  'Four-cell Gold targets are independently fully enumerated;160-cell target is only one planted witness, not the original puzzle.',
                  'No baseline random search is run for the160-cell fixture. Capped four-cell runs are incomplete, never UNSAT.',
                  'No original-puzzle speed, global deduplication, pruning or coverage claim. Production collections normally occur less frequently than this prepared-archive discriminator.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for row in rows:print(json.dumps({'fixture':row['fixture'],'baseline_solved':row['solved_baseline_pairs'],'baseline_solved_proposals':row['solved_baseline_gpu_lane_proposals'],
        'crossovers':len(row['pairs']),'child_checks':[p['crossover']['crossover_counters']['cycle_candidates'] for p in row['pairs']],
        'gpu_proposals':[p['crossover']['gpu_lane_proposals'] for p in row['pairs']]}),flush=True)
if __name__=='__main__':main()
