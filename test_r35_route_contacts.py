from collections import Counter
from itertools import combinations
import math
import random
import unittest
import numpy as np

from r31_route_geometry import eligible_orders
from r32_route_ranges import PrefixRanges
from r35_route_contacts import attainable


class PhysicalWildcardTests(unittest.TestCase):
    def test_every_fixed_prefix_against_physical_subsets(self):
        rng=random.Random(3516)
        free=(1,3,4,7,8,10,13,15,16,17)
        fam=(2,2)
        for case in range(6):
            close=np.zeros((18,18),dtype=bool)
            for a,b in combinations(free,2):
                close[a,b]=close[b,a]=rng.randrange(4)==0
            cores=[tuple(sum(1 << c for c in rng.sample(free,4)) for _ in range(2)),
                   tuple(sum(1 << c for c in rng.sample(free,3)) for _ in range(2))]
            required=[3,2]
            engine=PrefixRanges(free,close,cores,required,[{fam},set()],(fam,),[208,208],208)
            def physical(cells):
                mask=sum(1 << c for c in cells)
                edges=sum(close[a,b] for a,b in combinations(cells,2))
                return any(all(mask & core for core in scope) and edges>=need and
                           (i==1 or bool(tuple(eligible_orders(close,cells,fam))))
                           for i,(scope,need) in enumerate(zip(cores,required)))
            for width in (2,3):
                for high in combinations(range(width,len(free)),4-width):
                    fixed=tuple(free[i] for i in high)
                    limit=high[0]
                    expected=any(physical((*low,*fixed)) for low in combinations(free[:limit],width))
                    self.assertEqual(expected,attainable(engine,fixed,width,limit))
                    self.assertEqual(expected,engine.possible(fixed,width,limit))
            for lo,hi in [(0,math.comb(len(free),4)),(1,72),(37,181)]:
                work=Counter()
                spans=list(engine.spans(4,lo,hi,work))
                viable={r for r,group in enumerate(sorted(combinations(free,4),key=lambda s:s[::-1])) if lo<=r<hi and physical(group)}
                retained={r for a,b in spans for r in range(a,b)}
                self.assertLessEqual(viable,retained)
                self.assertEqual(hi-lo-len(retained),work['supports_before_generation'])
