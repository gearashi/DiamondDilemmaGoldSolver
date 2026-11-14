"""Independent checks of sparse pricing, ownership, support bounds, and resumption."""
from collections import Counter
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_clusters import StarClusters, interacting_supports
from r29_exchange_route import (NativeTriples, _order_count, best_swap_path, is_owned,
                                owner_copies, run, support_owned, swap_orders, transpositions)
from search_fixtures import planted_gold
from search_problem import SearchProblem


class Exchanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,cls.codes = planted_gold(32,1,3)
        cls.problem = SearchProblem(tiles,geo)
        cls.bound = ExchangeBound(BoardModel(cls.problem),cls.codes)

    def test_only_requested_transpositions_each_once(self):
        for n in range(2,9):
            expected = Counter(transpositions(p) for p in itertools.permutations(range(n)) if all(i!=x for i,x in enumerate(p)))
            self.assertEqual(sum(expected.values()),sum(_order_count(n,k) for k in range(1,n//2+1)))
            for cost,count in expected.items():
                orders = list(swap_orders(n,cost))
                self.assertEqual(len(orders),count)
                self.assertEqual(len(set(orders)),count)
                self.assertTrue(all(transpositions(p)==cost for p in orders))

    def test_prices_all_interactions_and_whole_support_screen(self):
        clusters = StarClusters(self.bound)
        for cells in [(0,1,2,3),(0,2,6),(1,2,3,4,5,6)]:
            orders = [order for order in itertools.permutations(range(len(cells))) if all(i!=j for i,j in enumerate(order))]
            vals = []
            for order in orders:
                direct = self.bound.bounds([cells],order)[0]
                self.assertEqual(clusters.price(cells,order)//2,direct)
                vals.append(direct)
            self.assertFalse(clusters.support_screen(cells,int(max(vals))))
        for size in (2,3,5,6):
            got = list(interacting_supports(range(self.problem.n),size,clusters.close))
            self.assertCountEqual(got,[s for s in itertools.combinations(range(self.problem.n),size)
                                    if any(clusters.close[a,b] for a,b in itertools.combinations(s,2))])

    def test_separated_dual_cap_fails_open_and_witness_is_possible(self):
        bound = self.bound
        clusters = StarClusters(bound)
        score = int(bound.model.matches(bound.board).sum())
        def exact(seed):
            return dict(score=int(bound.model.matches(seed).sum()))
        self.assertFalse(clusters.separated_certificate(0,exact,max_leaves=0)['closed'])
        self.assertFalse(clusters.separated_certificate(score,exact)['closed'])
        self.assertTrue(clusters.separated_certificate(len(self.problem.edges)+1,exact)['closed'])

    def test_ownership_against_physical_distance(self):
        b = self.bound
        first = np.arange(b.n)
        first[[2,3]] = first[[3,2]]
        candidates = [(tuple(first),3),(tuple(first),-1)]
        for cells in itertools.combinations(range(b.n),6):
            if sum(cells)%7:
                continue
            for order in swap_orders(6,3):
                target = np.arange(b.n)
                target[list(cells)] = [cells[j] for j in order]
                relative = np.argsort(first)[target]
                diff = sum(relative!=np.arange(b.n))
                expected = diff<=3 or (diff==6 and transpositions(relative)==3)
                self.assertEqual(is_owned(candidates,cells,order),expected)
                if support_owned(candidates,cells):
                    self.assertTrue(expected)
        tiles,geo,codes = planted_gold(32,1,3)
        fixed = BoardModel(SearchProblem(tiles,geo,{0:codes[0]}))
        owners = list(owner_copies(fixed,[(self.bound.tiles,3)],self.bound.tiles,20))
        self.assertEqual(len(owners),1)

    def test_resume_and_shortest_path_preserve_checked_boards(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.sqlite'
            knobs=dict(size=2,distance=1,batch=8,backend='cpu')
            partly=run(self.problem,self.codes,12,path,max_pages=1,**knobs)
            self.assertFalse(partly['complete'])
            end=run(self.problem,self.codes,12,path,**knobs)
            cold=run(self.problem,self.codes,12,':memory:',cluster_screen=False,**knobs)
            self.assertEqual(end['hits'],cold['hits'])
            self.assertTrue(end['complete'])
            self.assertEqual(run(self.problem,self.codes,12,path,**knobs)['offset'],end['offset'])
        goal=list(self.codes)
        goal[0],goal[1],goal[2]=goal[2],goal[0],goal[1]
        path=best_swap_path(self.problem,self.codes,goal)
        self.assertTrue(path['reachable'])
        self.assertEqual(len(path['swaps']),2)
        self.assertEqual(path['floor'],min(self.problem.validate(x['codes'])['matched_edges'] for x in path['boards']))
        self.assertEqual([v//3 for v in path['boards'][-1]['codes']],[v//3 for v in goal])

    def test_sparse_native_survivors_and_ownership_vs_all_derangements(self):
        from build_native import find_compiler
        try:
            find_compiler()
        except RuntimeError:
            self.skipTest('No native compiler.')
        b=self.bound
        cells=np.array(list(itertools.combinations(range(b.n),6)))
        first=np.arange(b.n)
        first[[0,1]]=first[[1,0]]
        owners=[(tuple(first),5),(tuple(first),-1)]
        def identity(s,o):
            row=b.tiles.copy()
            row[s]=b.tiles[s[list(o)]]
            return tuple(row)
        for goal in (5,8,10,12,15):
            graph=NativeTriples(b,goal,owners)
            try:
                for start,stop,cap in [(-1,1,1),(0,len(graph.pairs)+1,1),(0,2,0),(1,0,1),(0.5,1,1)]:
                    with self.assertRaises(ValueError):
                        graph.page(start,stop,cap)
                counts,rows=graph.page(0,len(graph.pairs),capacity=11)
            finally:
                graph.close()
            with self.assertRaises(ValueError):
                graph.page(0,1)
            expected=[]
            for order in swap_orders(6,3):
                for s in cells[b.bounds(cells,order)>=goal]:
                    if not is_owned(owners,s,order):
                        expected.append(identity(s,order))
            self.assertCountEqual([identity(s,(1,0,3,2,5,4)) for s in rows],expected)
            self.assertEqual(counts['star_pass'],len(expected))


if __name__=='__main__':
    unittest.main()
