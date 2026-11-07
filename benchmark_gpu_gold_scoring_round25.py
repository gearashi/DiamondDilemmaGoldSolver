"""Finite proposal36 exact-trajectory and repeated score-pass discriminator.

Fixed proposals continue even after synthetic Gold; this is not a time-to-solve
comparison. Entire score passes, not variable-length segment visits, are counted.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot
ROOT=Path(__file__).resolve().parent
CHUNKS=(1,7,64,64)

def work_counts(stats,proposals):
    keys=('gold_old_base_scans_saved','gold_old_loop_traces_saved','gold_candidate_loop_traces','gold_tabu_traces_skipped','gold_metropolis_traces_skipped')
    if any(type(stats.get(k))is not int or stats[k]<0 for k in keys) or type(proposals)is not int or proposals<=0:raise ValueError('Invalid finite score-pass accounting.')
    oldbase,oldloops,traces,tabu,metro=(stats[k] for k in keys)
    if oldbase!=proposals or oldloops!=proposals or traces+tabu+metro!=proposals:raise ValueError('Finite score-pass counters disagree.')
    return {'lane_proposals':proposals,'baseline_old_and_candidate_base_passes':2*proposals,'lazy_candidate_base_passes':proposals,
        'baseline_old_and_candidate_loop_passes':2*proposals,'lazy_candidate_loop_passes':traces,
        'omitted_base_passes':oldbase,'omitted_loop_passes':oldloops+tabu+metro,
        'baseline_pass_counts':'Derived from one old and one candidate pass per proposal in the unoptimized kernel; initialization excluded.'}

def cases(data):
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles
    from search_fixtures import planted_gold
    planted,shape,_=planted_gold(2536,size=4,extra_edges=12)
    return [('tetrahedron',SearchProblem(*tetrahedron_tiles()),16,(2536,2537)),
        ('planted160',SearchProblem(planted,shape),4,(2536,)),('original160',SearchProblem(json.loads(data.decode('utf-8-sig'))),4,(2536,))]

def native_signature(arrays,rows):
    import numpy as np
    return hashlib.sha256(b''.join(np.asarray(x,dtype='<u4').tobytes() for x in (*arrays[:2],arrays[2][:rows]))).hexdigest()

def compare(problem,*,backend,seed,replicas,policy):
    import numpy as np
    from gpu_sampling import _make_sampler,_inputs,gold_score
    inputs=_inputs(problem,gold_aware=True);a=b=None;began=time.monotonic()
    try:
        a=_make_sampler(problem,*inputs,replicas,seed,backend,experiments=policy)
        b=_make_sampler(problem,*inputs,replicas,seed,backend,experiments={**policy,'lazy_gold_scoring':True})
        signatures=[];checked=0;initial_gold=0
        for chunk in (0,*CHUNKS):
            if chunk:
                a.launch(chunk);b.launch(chunk)
                if policy.get('replica_exchange'):a.exchange(len(signatures)%2);b.exchange(len(signatures)%2)
            old=a.snapshot_arrays();new=b.snapshot_arrays()
            for x,y in zip(old[:2],new[:2]):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(old[2],new[2][:a.meta_rows])
            signature=native_signature(old,a.meta_rows)
            if signature!=native_signature(new,a.meta_rows):raise ValueError('Native trajectory hash mismatch.')
            signatures.append(signature)
            for lane in range(replicas):
                for board,row in ((new[0],1),(new[1],2)):
                    metrics=gold_score(problem,board[:,lane]);checked+=1
                    if metrics['gold_objective']!=int(new[2][row,lane]):raise ValueError('Independent native score mismatch.')
                    if not chunk and row==1 and metrics['matched_edges']==len(problem.edges) and not metrics['domain_violations']:
                        initial_gold+=int(problem.validate(board[:,lane])['valid'])
            if time.monotonic()-began>30:raise RuntimeError('Finite case exceeded30s guard; no score-pass evidence emitted.')
        stats=b.experiment_stats();counts=work_counts(stats,sum(CHUNKS)*replicas)
        return {'seed':seed,'replicas':replicas,'policy':policy,'native_runtime':b.runtime_identity(),'initial_gold_count':initial_gold,
            'trajectory_sha256_after_each_chunk':signatures,'native_old_rows_equal':True,'independently_checked_boards':checked,
            'work':counts,'lazy_counters':stats}
    finally:
        if a:a.close()
        if b:b.close()

def comparisons(data,backend):
    policies=[{'ejection_chains':True},{'ejection_chains':True,'move_tabu':True},
        {'ejection_chains':True,'move_tabu':True,'replica_exchange':True},
        {'obstruction_chains':True,'move_tabu':True,'replica_exchange':True}]
    return [{'fixture':name,'cells':p.n,'fingerprint':p.fingerprint,
             'scope':'Original supplied tiles, fixed work only; no solution or speed claim' if name=='original160' else 'Synthetic fixture, not original-puzzle progress',
             'pairs':[compare(p,backend=backend,seed=seed,replicas=r,policy=policy) for policy in policies for seed in seeds]}
            for name,p,r,seeds in cases(data)]

def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('cuda','webgpu'),required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/tiles.json');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Output must be a fresh JSON file.')
    before=source_snapshot(ROOT);runtime=runtime_snapshot();repo=repository_snapshot(ROOT);data=args.data.read_bytes()
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty') is not False):raise RuntimeError('Frozen evidence requires clean committed sources.')
    rows=comparisons(data,args.backend);after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repo_after=repository_snapshot(ROOT)
    if before!=after or runtime!=runtime_after or repo.get('commit')!=repo_after.get('commit') or data!=args.data.read_bytes():raise RuntimeError('Finite source/runtime/input changed.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository changed.')
    result={'proposal':36,'evidence_stage':args.evidence_stage,'evidence_status':'passed','backend':args.backend,'chunks':CHUNKS,
        'input':{'path':str(args.data.resolve()),'sha256':hashlib.sha256(data).hexdigest()},'results':rows,
        'source_before':before,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after,
        'assertion_tested':'Same native boards, best boards, RNG, iteration, tabu and prior counters after every chunk; independently equal scores; fewer complete score passes.',
        'limits':['Fixed work continues even after a synthetic Gold; not a time-to-solve or proposal-reduction result.',
            'Counts measure complete base/loop passes, not variable-length loop segment visits or all primitive operations.',
            'Counter rows are uint32 per lane; these bounded runs cannot wrap. Unbounded direct sampler telemetry remains modular.',
            'No original-puzzle solution, timing gain or new exact pruning/coverage claim.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for row in rows:
        print(json.dumps({'fixture':row['fixture'],'pairs':len(row['pairs']),**{k:sum(pair['work'][k] for pair in row['pairs']) for k in ('lane_proposals','omitted_base_passes','omitted_loop_passes','lazy_candidate_loop_passes')}}),flush=True)
if __name__=='__main__':main()
