"""Completed CPU residual labels and leave-family-out contextual selection.

Reports CPU decisions and propagation work separately. Unknown runs are excluded
from training and never counted as complete. No original-puzzle optimum claim.
"""
import hashlib,json,random,sys,time
from pathlib import Path
from constraint_context_policy_round26 import ContextPolicy,PROFILES,features,profile_options
from constraint_dfs import _solve_core
from search_options import BOOLEAN_OPTIONS
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,simple_tiles,multiple_loops,double_face,brute
from test_constraint_separator import ring,permutation_ring


def cases():
    rng=random.Random(2608)
    raw=[('double',SearchProblem(simple_tiles(),double_face())),
         ('double-unsat',SearchProblem(multiple_loops(),double_face())),
         ('tetra',SearchProblem(*tetrahedron_tiles())),
         ('ring4',ring(4)),('ring5',ring(5)),
         ('permutation4',permutation_ring([(1,0,2),(0,2,1)]*2))]
    for family,problem in raw:
        root=tuple(sum(1<<c for c in row) for row in problem.domains)
        answers=brute(problem,'gold')
        for variant in range(4):
            masks=root if variant==0 else tuple(sum(1<<c for c in row if rng.random()<(.5 if variant==1 else .8)) or 1<<row[0] for row in problem.domains)
            expected=[board for board in answers if all(mask>>c&1 for mask,c in zip(masks,board))]
            yield family,str(variant),problem,masks,expected


def compare(seconds=15):
    rows=[]
    base=dict.fromkeys(BOOLEAN_OPTIONS,False)
    base.update(native='python',lookahead_ms=1000,incremental=True,persistent_learning=True)
    for family,variant,problem,masks,expected in cases():
        row={'family':family,'case':family+':'+variant,'features':features(problem,masks),'domains':masks,'gold_witness_count':len(expected),'results':{}}
        for profile in PROFILES:
            state={};started=time.monotonic()
            try:
                result=_solve_core(problem,seconds=seconds,seed=2608,target='gold',_initial_domains=masks,
                    learning_state=state,options=profile_options(base,profile))
                if result['complete']:
                    if expected and tuple(result.get('codes') or ()) not in expected:raise RuntimeError('CPU profile lost the independent Gold witness.')
                    if not expected and result['status']!='infeasible':raise RuntimeError('CPU profile invented a Gold witness.')
                row['results'][profile]={'status':result['status'],'complete':result['complete'],'cpu_nodes':result['nodes'],
                    'arc_revisions':result['stats']['arc_revisions'],'probe_propagations':result['stats'].get('lookahead_probe_propagations',0)+result['stats'].get('pair_lookahead_propagations',0),
                    'elapsed_seconds':time.monotonic()-started}
            finally:
                if state.get('_runtime') is not None:state['_runtime'].close()
        rows.append(row)
    held=[]
    for row in rows:
        model=ContextPolicy()
        for train in rows:
            if train['family']==row['family']:continue
            for profile,result in train['results'].items():model.observe(train['features'],profile,nodes=result['cpu_nodes'],complete=result['complete'],case=train['case'])
        selected=model.choose(row['features'],explore=False)
        if any(not r['complete'] for r in row['results'].values()):
            held.append({'case':row['case'],'selected':selected,'comparison':'inconclusive'});continue
        held.append({'case':row['case'],'selected':selected,'base_nodes':row['results']['base']['cpu_nodes'],
            'selected_nodes':row['results'][selected]['cpu_nodes'],'best_observed_nodes':min(r['cpu_nodes'] for r in row['results'].values()),
            'training_records':len(model.records)})
    return {'rows':rows,'held_out':held,'policy_role':'priorities only; never pruning authority',
            'summary':{'completed_rows':sum(all(r['complete'] for r in row['results'].values()) for row in rows),
                       'base_nodes':sum(r.get('base_nodes',0) for r in held),'selected_nodes':sum(r.get('selected_nodes',0) for r in held)}}

def main(argv=None):
    import argparse
    from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions
    require_assertions();parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parent
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Fresh JSON output required.')
    source=source_snapshot(root);runtime=runtime_snapshot();repo=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty')is not False):raise RuntimeError('Frozen evidence requires a clean readable commit.')
    report=compare()
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if source!=after or runtime!=runtime_after or repo!=repo_after:raise RuntimeError('Sources/runtime/repository changed during comparison.')
    report.update(evidence_stage=args.evidence_stage,family='F08',provenance={'source_before':source,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(report,out,indent=2);out.write('\n')
    print(json.dumps(report['summary']));return report

if __name__=='__main__':main()
