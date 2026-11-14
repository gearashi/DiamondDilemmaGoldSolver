from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path
import unittest
import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,combination_batch
from r29_exchange_clusters import StarClusters
from r30_cycles import canonical_cycles
from r30_multicenter import orders
from r31_route_geometry import ConnectedJoin, connected, eligible_orders
from r32_route_ranges import PrefixRanges
from r32_route_join import bridge_upper,traverse
from r32_route_pool import defect_cover
from search_problem import SearchProblem
from search_fixtures import planted_gold


class RouteImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,codes = planted_gold(45,1,5)
        cls.model = BoardModel(SearchProblem(tiles,geo))
        cls.bound = ExchangeBound(cls.model,codes)
        cls.stars = StarClusters(cls.bound)

    def test_ranges_conservative_with_holes_and_non_aligned_boundaries(self):
        n,k = 19,7
        free = tuple(2*i+1 for i in range(n))
        close = np.eye(2*n,dtype=bool)
        for i in range(0,10,2):
            a,b=free[i],free[i+1]
            close[a,b]=close[b,a]=True
        family = (2,2,3)
        core = tuple(sum(1 << free[j] for j in g) for g in ((1,3),(5,6,9)))
        engine = PrefixRanges(free,close,[core],[2],[{family}],(family,),[208],208)
        lo,hi = 287,1905
        w=Counter()
        spans=list(engine.spans(k,lo,hi,w))
        covered = {j for a,b in spans for j in range(a,b)}
        self.assertGreater(w['supports_before_generation'],100)
        self.assertEqual(hi-lo-len(covered),w['supports_before_generation'])
        for r,cells in enumerate(combination_batch(free,k,lo,hi-lo),lo):
            mask=sum(1<<int(c) for c in cells)
            if all(m&mask for m in core) and tuple(eligible_orders(close,cells,family)):
                self.assertIn(r,covered)

    def test_exhaustive_pool_pricing_owner_and_shared_centers(self):
        pool=tuple(range(8))
        for family in ((2,2,3),(2,2,2,2)):
            all_rows=set()
            for cells in combinations(pool,sum(family)):
                connected_orders = tuple(eligible_orders(self.stars.close,cells,family))
                cap = self.stars.support_upper_twice(cells,orders=connected_orders)
                for order in connected_orders:
                    value=self.stars.price(cells,order)
                    self.assertGreaterEqual(cap,value)
                    group=canonical_cycles(cells,order)
                    bridge=bridge_upper(self.stars,group)
                    if bridge is not None:
                        self.assertEqual(value,bridge)
                    if value>=24:
                        all_rows.add((cells,order))
            for pairs,buckets in ((False,False),(False,True),(True,True)):
                j=ConnectedJoin(self.stars,pool,family,12)
                rows,work=traverse([j,j],pair=pairs,buckets=buckets)
                got=list(rows)
                self.assertEqual(len(got),len(set(got)))
                for center in (0,1):
                    self.assertEqual({(c,o) for i,c,o in got if i==center},all_rows)
    @unittest.skipUnless(__import__('importlib').util.find_spec('ortools'), 'optional CP-SAT dependency')
    def test_defect_cover(self):
        pool=tuple(range(8))
        forced=((0,1),)
        self.assertFalse(defect_cover(self.bound,forced,pool,6,0)['closed'])
        self.assertTrue(defect_cover(self.bound,forced,pool,0,len(self.model.problem.edges)+1)['closed'])

    def test_every_small_stream_item_has_exact_optimum(self):
        from unittest.mock import patch
        from r28_exchange_rotations import RotationOracle
        from r32_route_pool import stream_prefixes
        import tempfile
        oracle=RotationOracle(self.model)
        for f in ((2,2,3),(2,2,2,2)):
            cells=tuple(range(sum(f)))
            best=max(oracle.optimum(self.bound.witness_seed(cells,o))[0] for o in eligible_orders(self.stars.close,cells,f))
            with tempfile.TemporaryDirectory() as temp, patch('r31_route_geometry.proved_connected',return_value=True):
                res=stream_prefixes(self.model.problem,[self.bound.board],cells,f,best,str(Path(temp)/'tiny.sqlite'))
                self.assertTrue(res['complete'])
                self.assertEqual(res['best']['score'],best)

    def test_streamed_pair_price_and_resume(self):
        from unittest.mock import patch
        import tempfile
        from r32_route_join import PhysicalPrices
        from r32_route_pool import stream_prefixes
        for family in ((2,2,3),(2,2,2,2)):
            pool=tuple(range(8))
            j=ConnectedJoin(self.stars,pool,family,8)
            price=PhysicalPrices(j)
            group=((0,1,2),) if len(family)==3 else ((0,1),(2,3))
            available=sorted(set(pool)-{c for cyc in group for c in cyc})
            pairs=list(combinations(available,2))
            cat=price.pair_catalog(group,pairs)
            for first in pairs:
                got=set(price.partners(cat,first,Counter()))
                expected={second for second in pairs if second>first and set(first).isdisjoint(second)
                          and price.one_score((*group,first,second))>=8}
                self.assertEqual(got,expected)
            with tempfile.TemporaryDirectory() as tmp, patch('r31_route_geometry.proved_connected',return_value=True):
                args=dict(pool=pool,family=family,target=14,database=str(Path(tmp)/'stream.db'),start=0,stop=16)
                partial=stream_prefixes(self.model.problem,[self.bound.board],**args,max_pages=3)
                self.assertEqual(partial['completed_items'],3)
                done=stream_prefixes(self.model.problem,[self.bound.board],**args)
                self.assertTrue(done['range_finished'])
                again=stream_prefixes(self.model.problem,[self.bound.board],**args)
                self.assertFalse(again['new_pages'])
                self.assertFalse(again['work'])
