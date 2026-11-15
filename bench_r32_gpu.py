"""Compare CUDA Gold supports and verify migration from a v1.12 runner checkpoint."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

from constraint_work import WorkScope,restore_frontier
from gpu_domain_round24 import DomainCUDA, _device_table_cache
from gpu_domain_runner import solve,split_cubes
from r28_exchange import read_json
from search_problem import SearchProblem
from test_constraint_dfs import brute,double_face,multiple_loops,tetrahedron_tiles
from test_gpu_domain_round24 import enumerate_engine


def compare(problem,unrestricted=False):
    cubes=split_cubes([tuple(sum(1<<v for v in d) for d in problem.domains)],128)
    rows=[]
    for mode in ('old','current'):
        # This experiment changes the support compiler itself within one process.
        _device_table_cache.pop(problem,None)
        context=patch('r32_gold_bonds.forbidden_codes',return_value=()) if mode=='old' else nullcontext()
        with context:
            gpu=DomainCUDA(problem,cubes,target='gold',global_matching=True,support_reuse=True,cross_lane=True,
                           support_metrics=True,proof_transport=True)
        try:
            if unrestricted:
                for _ in range(32):
                    gpu.step(16)
                count=len(gpu.candidates())
            else:
                found=enumerate_engine(gpu,31)
                assert sorted(found)==sorted(brute(problem,'gold'))
                count=len(found)
            s=gpu.stats()
            rows.append(dict(mode=mode,lanes=len(cubes),candidates=count,device=gpu.device,
                        work={k:int(s[k]) for k in ('decisions','backtracks','support_word_scans','primitives','exhausted') if k in s},
                        complete=not unrestricted))
        finally:
            gpu.close()
    return rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--old-checkpoint',type=Path)
    cli.add_argument('--output',required=True)
    a=cli.parse_args()
    p=SearchProblem(read_json(a.tiles))
    report={'original':compare(p,True),'closed_loops':compare(SearchProblem(multiple_loops(),double_face())),
            'positive':compare(SearchProblem(*tetrahedron_tiles()))}
    if a.old_checkpoint:
        saved=read_json(a.old_checkpoint)
        problem=SearchProblem(*tetrahedron_tiles())
        assert 'physical_bond_support' not in saved['native']['binding']
        remaining=restore_frontier(saved['frontier'],WorkScope.for_problem(problem,'gold'))
        expected={codes for codes in map(tuple,brute(problem,'gold')) if any(
            all((1<<code)&mask for code,mask in zip(codes,row)) for row in remaining)}
        run=solve(problem,replicas=3,resume=saved,seconds=30,target='gold')
        assert run['status']=='solved' and tuple(run['codes']) in expected and problem.validate(run['codes'])['valid']
        assert run['stats']['gpu_domain_portable_restart']==1
        if remaining:
            gpu=DomainCUDA(problem,remaining,target='gold')
            try:
                assert set(enumerate_engine(gpu,31))==expected
            finally:
                gpu.close()
        report['migration']=dict(source_schema=saved['native'].get('schema'),native_old=True,
                                 frontier_cubes=len(remaining),gold_answers_in_remainder=len(expected),
                                 status=run['status'],portable_restart=1)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
