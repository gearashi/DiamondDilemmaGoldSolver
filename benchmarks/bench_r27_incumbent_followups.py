"""Higher-budget screens and equal-budget controls for alternative board construction."""
import json,time
from pathlib import Path
import numpy as np
from r27_incumbent import BoardModel
from bench_r27_incumbent_population import construct,ensemble,project,propose
from constraint_ordering import SupportOrdering,STAT_KEYS
from constraint_dfs import _build_support_tables,_values
from r27_structure_fixtures import planted
from search_problem import SearchProblem


def rank_test():
    rows=[]
    for seed in range(5):
        p,domains=planted(1,free=6,seed=seed+7,split=False)
        # Order six physical tiles on the same residual domains as the released heuristic.
        p.domains=[list(_values(mask)) for mask in domains]
        model=BoardModel(p)
        tables=_build_support_tables(p,domains,'gold',lambda:None)
        maps=[np.zeros((p.n,3*p.n)),np.zeros((p.n,3*p.n))]
        order=SupportOrdering(tables)
        stats=dict.fromkeys(STAT_KEYS,0)
        known=np.arange(p.n)*3
        rng=np.random.default_rng(seed)
        for c,row in enumerate(p.domains):
            rank={v:i for i,v in enumerate(rng.permutation(row))}
            options=order.order(c,domains,row,{},rank,lambda:None,stats)
            for i,v in enumerate(options):
                maps[0][c,v]=1/(i+1)
        control=project(model,maps[0],rng)
        start=project(model,rng.gumbel(size=model.legal.shape),rng)
        for mode in ('support','belief','belief_pin','fractional'):
            before=time.perf_counter()
            if mode=='support':
                candidate,_=model.orientations(control)
                work=stats['value_order_scored']
            else:
                candidate,work=construct(model,start,rng,'belief' if mode.startswith('belief') else mode,
                                         decimate=mode=='belief_pin')
            val=p.validate(candidate.tolist())
            rows.append(dict(seed=seed,mode=mode,score=val['matched_edges'],gold=val['valid'],
                             seconds=time.perf_counter()-before,work=work))
    return rows


def main():
    src=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    p=SearchProblem(src('data/tiles.json'))
    model=BoardModel(p)
    original=model.legal_board(src('audit/r27/incumbent.json')['codes'])
    rows=[]
    for seed in range(4):
        for starts in ('original','fresh'):
            rng=np.random.default_rng(seed+10)
            board=original if starts=='original' else project(model,rng.random(model.legal.shape),rng)
            for method in ('exchange','population','flat','cross_entropy'):
                before=time.perf_counter()
                trial=ensemble(model,board,seed+10,method,proposals=4096)
                report=p.validate(trial['best'].tolist())
                rows.append(dict(seed=seed,start=starts,method=method,score=report['matched_edges'],
                    components=report['line_components'],first208=trial['first208'],proposals=trial['steps'],
                    extra=trial['extra'],seconds=time.perf_counter()-before))
    path=Path('audit/r27/incumbent-followups.json')
    path.write_text(json.dumps(dict(ensemble=rows,rank=rank_test()),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

