"""Measure donor occupancy dominance and reusable whole-budget fronts on original pools."""
from pathlib import Path
import argparse
import random

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r34_component_budget import ComponentBudget
from search_problem import SearchProblem


def run(tiles,out):
    model=BoardModel(SearchProblem(read_json(tiles)))
    bound=ExchangeBound(model,read_json('docs/R29_INCUMBENT.json')['codes'])
    rng=random.Random(1534)
    result=[]
    for i in range(36):
        pool=rng.sample(sorted(map(int,model.free)),8)
        answers={}
        for name,dominance,order,full in [('released',False,None,False),('dom',True,None,False),
            ('small',True,'small',False),('large',True,'large',False),('front',True,None,True),('meet',True,None,True),('meet_small',True,'small',True)]:
            c=ComponentBudget(bound,pool,dominance=dominance,order=order)
            rows=[]
            for swaps,floor in [(4,0),(3,0),(2,0),(4,7)]:
                rows.append(c.upper(swaps,min_moved=floor,retain_front=full and not rows,meet=name.startswith('meet')))
            answers[name]=rows
            assert [r['upper'] for r in rows]==[r['upper'] for r in answers['released']]
        result.append(dict(pool=pool,measurements=answers))
        atomic_json(out,result)
        print(i,[(tag,sum(r['work']['joins'] for r in rows)) for tag,rows in answers.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='../data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.tiles,a.output)
