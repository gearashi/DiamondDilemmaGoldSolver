"""Compare single-cycle shortcuts and wide conditional rotation bounds on original supports."""
from collections import Counter
from itertools import combinations
import random
from pathlib import Path
import argparse

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r30_cycles import canonical_cycles
from r34_cycle_orders import CycleOrders,FIVE_FAMILIES
from r34_exchange_undo import CoveredUndo
from r35_route_difference import differential
from search_problem import SearchProblem


def run(tiles,output):
    model=BoardModel(SearchProblem(read_json(tiles)))
    b=ExchangeBound(model,read_json('docs/R29_INCUMBENT.json')['codes'])
    pool=sorted(map(int,model.free))
    out=dict(samples={},regions=[])
    # The archived owner is in the 208-closed radius-five ball, the six
    # occupied (2,4), (3,3), and (2,2,2) classes have separate proofs.
    from r29_exchange_route import existing_balls,is_owned,owner_copies
    from r28_exchange import domain_binding
    from r30_multicenter import released_six
    from r29_exchange_route import physical_hash
    binding=domain_binding(model)
    centers=existing_balls(model,208)
    proved={r['center'] for r in read_json('docs/R29_EXCHANGE_RESULTS.json')['triples']
            if r['complete'] is True and r['hits']==0 and r['target']==208 and r['swap_distance']==3
            and r['moved']==6 and r['objective']=='matched_edges'}
    centers+=[(row,-1) for row,_ in tuple(centers) if physical_hash(row) in proved]
    centers+=[(row,-2) for row,radius in tuple(centers) if radius>=0 and released_six(binding,[row],[(2,4),(3,3)],208)]
    rng=random.Random(6401)
    for family in FIVE_FAMILIES:
        owners=list(owner_copies(model,centers,b.tiles,sum(family)))
        from functools import lru_cache
        @lru_cache(8192)
        def covered(g):
            cells=sorted(c for group in g for c in group)
            index={c:i for i,c in enumerate(cells)}
            moved={c:d for row in g for c,d in zip(row,(*row[1:],row[0]))}
            return is_owned(owners,cells,tuple(index[moved[c]] for c in cells))
        before=CoveredUndo(b,covered)
        after=CoveredUndo(b,covered,predecessor_upper=lambda g:207 if covered(g) else None,target=208)
        counts=Counter()
        opts=CycleOrders(family)
        for _ in range(250):
            cells=sorted(rng.sample(pool,sum(family)))
            cycles=canonical_cycles(cells,opts[rng.randrange(len(opts))])
            first=before.rejects(cycles)
            second=after.rejects(cycles)
            assert not first or second
            counts['examined']+=1
            counts['whole_cycle']+=int(first)
            counts['including_splits']+=int(second)
        out['samples'][str(family)]=dict(counts)
        atomic_json(output,out)
        print(family,counts,flush=True)
    rng=random.Random(3520)
    for length in (3,4,5,6,7,8):
        for i in range(48):
            cells=[rng.choice(pool)]
            while len(cells)<length:
                near=set().union(*(before.near[c] for c in cells)) - set(cells)
                cells.append(rng.choice(sorted(near & set(pool))))
            exterior=set().union(*(before.near[c] for c in cells))-set(cells)
            common={}
            if i%2:
                port=rng.choice(sorted(exterior))
                source=rng.choice(sorted(set(pool)-set(cells)-exterior))
                common={port:source,source:port}
            old=dict(zip(cells,cells))
            after=dict(zip(cells,(*cells[1:],cells[0])))
            counts={}
            gain=differential(b,old,after,common,work=counts)
            if len(cells)+len(exterior)<=12:
                expected=before.region_gain(tuple(cells),tuple(after[c] for c in cells),tuple(sorted(common.items())))
                assert gain==expected,(cells,common,gain,expected)
            out['regions'].append(dict(cells=cells,boundary=len(exterior),changed=common,
                                       gain=gain,work=counts))
        atomic_json(output,out)
        print('width',length,len(out['regions']),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--output',type=Path,required=True)
    a=cli.parse_args()
    run(a.tiles,a.output)
