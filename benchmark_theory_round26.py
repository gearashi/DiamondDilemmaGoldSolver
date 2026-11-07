"""Paired completed SAT cases for checked clause versus direct theory inference."""
import hashlib,json,sys,random,time
from pathlib import Path
from unittest.mock import patch
from constraint_sat_session import PersistentSatSession
from constraint_sat_theory_round26 import direct_propagator
from constraint_work import WorkScope
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,brute
from test_constraint_sat_domains import fixture
from test_constraint_separator import permutation_ring

def compare():
    rows=[]
    for label,problem in [('double',SearchProblem(simple_tiles(),double_face())),('double-unsat',SearchProblem(multiple_loops(),double_face())),('tetra',fixture()[0]),('permutation',permutation_ring([(1,0,2),(0,2,1)]*2))]:
        solutions=brute(problem,'gold');modes=[]
        for direct in (False,True):
            with patch('constraint_sat_theory_round26.direct_propagator',side_effect=lambda *a,**kw:direct_propagator(*a,**kw,direct=direct)):
                owner=PersistentSatSession(problem,WorkScope.for_problem(problem,'gold'),theory=True,guard_reuse=True)
                try:
                    result=owner.advance(seconds=15)
                    if result['complete']:
                        if solutions and tuple(result.get('codes') or ()) not in solutions:raise RuntimeError('Theory lost the independent Gold witness.')
                        if not solutions and result['status']!='infeasible':raise RuntimeError('Theory accepted an invalid Gold witness.')
                    modes.append({'direct':direct,'status':result['status'],'complete':result['complete'],'sat_decisions':result['nodes'],
                        'native':dict(owner.solver.accum_stats()) if owner.solver is not None else {},
                        'stats':{k:v for k,v in result['stats'].items() if k.startswith(('theory_','sat_external_','session_model'))}})
                finally:owner.close()
        rows.append({'case':label,'modes':modes})
    return rows

def query_sequences():
    from constraint_separator import _Budget
    from constraint_pruning import values
    rows=[]
    for label,problem in [('tetra',fixture()[0]),('double-unsat',SearchProblem(multiple_loops(),double_face())),
                          ('permutation',permutation_ring([(1,0,2),(0,2,1)]*2))]:
        solutions=brute(problem,'gold');rng=random.Random(260403)
        root=tuple(sum(1<<code for code in row) for row in problem.domains)
        queries=[tuple(sum(1<<code for code in values(mask) if rng.random()<probability) for mask in root)
                 for probability in (.5,.8) for _ in range(32)]
        if label=='tetra':queries.insert(0,(2428,28,1730,1851))
        expected=[any(all(mask>>code&1 for mask,code in zip(query,board)) for board in solutions) for query in queries]
        modes=[]
        for direct in (False,True):
            with patch('constraint_sat_theory_round26.direct_propagator',side_effect=lambda *a,**kw:direct_propagator(*a,**kw,direct=direct)):
                owner=PersistentSatSession(problem,WorkScope.for_problem(problem,'gold'),theory=True)
                try:
                    owner.prepare(lambda:None,10);answers=[];complete=True
                    for query in queries:
                        deadline=time.monotonic()+10
                        def check():
                            if time.monotonic()>=deadline:raise TimeoutError('Finite theory query exceeded bound.')
                        assumptions=[-lit for cell,row in enumerate(owner.variables) for code,lit in row.items() if not query[cell]>>code&1]
                        answer=None
                        while answer is None:answer=owner._native_call(owner.solver,assumptions,check,deadline)
                        answers.append(answer)
                        if answer:
                            model=set(owner.solver.get_model())
                            codes=[next(code for code,lit in row.items() if lit in model) for row in owner.variables]
                            if not problem.validate(codes)['valid']:raise RuntimeError('Invalid native Gold witness.')
                    if answers!=expected:raise RuntimeError('Theory query answers disagree with independent full-board enumeration.')
                    modes.append({'direct':direct,'complete':complete,'native':dict(owner.solver.accum_stats()),
                                  'stats':{**owner._external.theory_stats,**owner._external.stats},'sat_answers':sum(answers)})
                finally:owner.close()
        rows.append({'case':label,'seed':260403,'queries':queries,'independent_gold_solutions':len(solutions),'modes':modes})
    return rows

def main(argv=None):
    import argparse
    from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Fresh JSON output required.')
    root=Path(__file__).resolve().parent;source=source_snapshot(root);runtime=runtime_snapshot();repo=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty')is not False):raise RuntimeError('Frozen evidence requires a clean readable commit.')
    rows=compare();sequences=query_sequences()
    if any(not row['complete'] for item in sequences for row in item['modes']):raise RuntimeError('Incomplete finite theory query.')
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if source!=after or runtime!=runtime_after or repo!=repo_after:raise RuntimeError('Source/runtime/repository changed during comparison.')
    output={'scope':'Finite completed Gold cases; no original-puzzle optimality claim','rows':rows,'query_sequences':sequences,
      'evidence_stage':args.evidence_stage,'family':'F04','provenance':{'sources_before':source,'sources_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(output,out,indent=2);out.write('\n')
    print(json.dumps({'cases':len(rows),'sequences':len(sequences),'passed':True}));return output

if __name__=='__main__':main()
