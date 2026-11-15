from collections import Counter
from pathlib import Path
import json
import numpy as np
from io_utils import atomic_json
from r32_gold_loss import read_json
from r32_gold_bonds import inventory
from r27_incumbent import BoardModel
from r32_fresh_phases import SignedRepair
from r35_signed_refit import changed_dart_prices
from search_problem import SearchProblem

p=SearchProblem(read_json('../data/tiles.json'))
model=BoardModel(p)
forbidden=inventory(p).forbidden
saved=read_json('docs/R35_CONNECTED_TOURS.json')
old=saved['raw']['phases']
other=saved['contract']['phases']
baseline=SignedRepair(model,old,forbidden)
for dart in range(3*p.n):
    baseline.prices[dart]=baseline.geometry.price(dart,baseline.dart_phases,baseline.stats)
records=[]
for name,phase in [('identical',old),('global-complement',[1-x for x in old]),('one-flip',[1-old[0],*old[1:]]),('other-verified-tour',other)]:
    new=SignedRepair(model,phase,forbidden)
    arrays=[]
    for dart in range(3*p.n):
        got=changed_dart_prices(baseline,new,dart)
        want=new.geometry.price(dart,new.dart_phases,Counter())
        np.testing.assert_array_equal(got,want)
        arrays.append(got is baseline.prices[dart])
    records.append(dict(name=name,flips=sum(a!=b for a,b in zip(old,phase)),cold_bits=(3*p.n)**2,
                        shared_unchanged_rows=sum(arrays),rows=len(arrays),work=dict(new.stats)))
atomic_json(Path('docs/R35_SIGNED_DART_DELTA.json'),records)
print(records)
