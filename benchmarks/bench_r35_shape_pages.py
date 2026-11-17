"""Measure route stages where prior ranked filters and disconnected shape proofs both apply."""
import argparse
from collections import Counter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r30_multicenter import run
from r31_routes import rank
from r34_cycle_orders import CycleOrders,FIVE_FAMILIES
from io_utils import atomic_json
from search_problem import SearchProblem


def test_pages(model,board,dest):
    b=ExchangeBound(model,board)
    rng=Random(6401)
    movable=sorted(map(int,model.free))
    # The original Round 34 random support experiment ran before these pages.
    for fam in FIVE_FAMILIES:
        for _ in range(250):
            rng.sample(movable,sum(fam))
            rng.randrange(len(CycleOrders(fam)))
    rows=[]
    with TemporaryDirectory(dir=dest.parent) as root:
        for family in FIVE_FAMILIES:
            if len(family)<2:
                continue
            starts=[]
            for anchor in ((30,62,105,104,107,106),(43,123,30,51,62,42)):
                near=sorted(set(int(j) for c in anchor for j in b.neighbors[c] if j!=b.n)-set(anchor))
                for _ in range(2):
                    cells=sorted([*anchor,*rng.sample(near,sum(family)-len(anchor))])
                    starts.append(rank(cells,model.free))
            variants={}
            for mode in ('existing','late','early'):
                filters=['contact','core','connected','star','family','ranges']
                if mode=='early':
                    filters.insert(3,'shapes')
                elif mode=='late':
                    filters.append('shapes')
                counts=Counter()
                best=[]
                for start in starts:
                    record=run(model.problem,[board],Path(root)/(str(family)+mode+'.sqlite'),
                               families=[family],start=start,stop=start+2,page=2,filters=tuple(filters))
                    assert record['range_finished']
                    counts.update(record['work'])
                    best.append(record['best'])
                variants[mode]=dict(work=dict(counts),best=best)
                assert variants[mode]['best']==variants['existing']['best']
            rows.append(dict(family=family,starts=starts,variants=variants))
            atomic_json(dest,rows)
            print(family,[(mode,vals['work'].get('first_shapes',0),vals['work'].get('shape_regional_prices',0))
                          for mode,vals in variants.items()],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--output',type=Path,required=True)
    opt=cli.parse_args()
    model=BoardModel(SearchProblem(read_json(opt.tiles)))
    test_pages(model,read_json('docs/R29_INCUMBENT.json')['codes'],opt.output)
