"""Compare exact endpoint/family bounds with and without separator conditioning."""
import argparse
from pathlib import Path
from random import Random
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r34_component_budget import ComponentBudget
from r35_donor_chains import family_budget
from io_utils import atomic_json
from search_problem import SearchProblem


def run(tiles,path):
    b=ExchangeBound(BoardModel(SearchProblem(read_json(tiles))),read_json('docs/R29_INCUMBENT.json')['codes'])
    rng=Random(1534)
    all_rows=[]
    free=sorted(map(int,b.model.free))
    for i in range(36):
        pool=rng.sample(free,8)
        if i>=28:
            pool=[rng.choice(free)]
            while len(pool)<8:
                boundary=set(int(v) for c in pool for v in b.neighbors[c] if v in free)-set(pool)
                pool.append(rng.choice(sorted(boundary)))
        c=ComponentBudget(b,pool)
        for fam in ((2,2,3),(2,2,2,2),(6,)):
            broad=c.upper(sum(fam)-len(fam),min_moved=sum(fam))
            plain=family_budget(c,fam,max_rows=120000,separator=0)
            cut=plain if plain['complete'] else family_budget(c,fam,max_rows=120000,separator=2)
            all_rows.append(dict(pool=pool,family=fam,broad=broad,plain=plain,cut=cut))
        atomic_json(path,all_rows)
        print(i,[(r['broad']['upper'],r['plain']['upper'],r['cut']['upper']) for r in all_rows[-3:]],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    run(args.tiles,args.output)
