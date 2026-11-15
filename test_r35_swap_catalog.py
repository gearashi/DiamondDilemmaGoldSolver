"""Check dependency invalidation, orientation ties and exact move-list reuse."""
from collections import Counter
from itertools import combinations, product
import unittest

import numpy as np

from bench_r35_swap_catalog import full_ranking, modified
from r27_incumbent import BoardModel
from r28_exchange_patches import AssignmentPrices
from r30_fresh import fresh_start
from r34_swap_packing import pack_swaps
from r35_swap_catalog import SwapCatalog
from search_fixtures import planted_gold
from search_problem import SearchProblem


class CatalogTests(unittest.TestCase):
    def setUp(self):
        tiles,geometry,reference=planted_gold(3535,1,5)
        problem=SearchProblem(tiles,geometry)
        for cell,code in enumerate(reference):
            problem.domains[cell]=[v for v in problem.domains[cell] if v==code or (v+2*cell)%7]
        self.model=BoardModel(problem)
        self.board=np.array(reference)
        self.rng=np.random.default_rng(3535)

    def physical(self,board,cells):
        initial=int(self.model.matches(board).sum())
        result=[]
        for a,b in combinations(sorted(cells),2):
            if b in self.model.problem.neighbors[a]:
                continue
            best=None
            for r,s in product(range(3),repeat=2):
                p,q=3*(board[b]//3)+r,3*(board[a]//3)+s
                if not (self.model.legal[a,p] and self.model.legal[b,q]):
                    continue
                row=board.copy()
                row[a],row[b]=p,q
                weight=int(self.model.matches(row).sum())-initial
                if weight>0 and (best is None or weight>best[-1]):
                    best=(a,b,p,q,weight)
            if best is not None:
                result.append(best)
        return sorted(result,key=lambda row:-row[-1])

    def test_sparse_changes_varying_protection_and_reuse_against_direct_rotation_scores(self):
        model=self.model
        board=self.board.copy()
        prices=AssignmentPrices(model)
        catalog=SwapCatalog(model)
        for step in range(28):
            cells=self.rng.permutation(model.free)
            cells=cells[:(len(cells)-step%3)]
            if step==8:
                cells=cells[:0]
            if step==14:
                cells=cells[:1]
            prices.update(board)
            ranked=catalog.rank(prices,board,cells)
            self.assertEqual(list(zip(*ranked)),self.physical(board,cells))
            full=full_ranking(model,prices,board,cells)
            for a,b in zip(ranked,full):
                np.testing.assert_array_equal(a,b)
            # The same price-table instance is also used by unrelated repair
            # stages, which can update it with a third board between calls.
            prices.update(modified(model,board,self.rng,1))
            if step%4:
                board=modified(model,board,self.rng,1)
        self.assertLess(catalog.work['pairs_reconsidered'],catalog.work['available_pairs'])
        stable=catalog.rank(prices,prices.board,model.free)
        again=catalog.rank(prices,prices.board,model.free)
        self.assertIs(stable,again)
        self.assertFalse(stable[0].flags.writeable)

    def test_inactive_neighbor_rotation_retracts_an_unchanged_eligible_pair(self):
        model=self.model
        rng=self.rng
        prices=AssignmentPrices(model)
        for trial in range(32):
            board=modified(model,self.board,rng,3)
            for cell in model.free:
                cells=np.setdiff1d(model.free,[cell])
                prices.update(board)
                first=full_ranking(model,prices,board,cells)
                before=set(zip(*first[:2]))
                for rotation in range(3):
                    code=3*(board[cell]//3)+rotation
                    if code==board[cell] or not model.legal[cell,code]:
                        continue
                    changed=board.copy()
                    changed[cell]=code
                    prices.update(changed)
                    second=full_ranking(model,prices,changed,cells)
                    if before-set(zip(*second[:2])):
                        catalog=SwapCatalog(model)
                        prices.update(board)
                        catalog.rank(prices,board,cells)
                        prices.update(changed)
                        actual=catalog.rank(prices,changed,cells)
                        for a,b in zip(actual,second):
                            np.testing.assert_array_equal(a,b)
                        return
        self.fail('Fixture did not remove a nonincident candidate.')

    def test_conflict_choice_reuses_only_matching_geometry_weight_priority_and_mode(self):
        model=self.model
        catalog=SwapCatalog(model)
        left=np.array([0,1,3])
        right=np.array([2,5,6])
        ranking=(left,right,np.zeros(3,dtype=int),np.ones(3,dtype=int),np.array([3,2,1]))
        def solve(model,pairs,weights,**kwargs):
            return pack_swaps(model,pairs,weights,**kwargs)
        first=catalog.noncontacting(ranking,0,Counter(),solve)
        # Fixed weights and endpoints can use wholly different oriented codes.
        updated=ranking[:2]+(np.array([26,23,10]),np.array([4,7,24]),ranking[-1])
        self.assertEqual(first,catalog.noncontacting(updated,0,Counter(),solve))
        self.assertEqual(catalog.work['nonempty_packs_reused'],1)
        catalog.noncontacting(updated,3,Counter(),solve)
        catalog.noncontacting(updated[:-1]+(np.array([4,2,1]),),3,Counter(),solve)
        order=[1,0,2]
        catalog.noncontacting(tuple(x[order] for x in updated),3,Counter(),solve)
        self.assertEqual(catalog.work['nonempty_packs_solved'],4)
        self.assertEqual(first,catalog.noncontacting(ranking,0,Counter(),solve))
        self.assertEqual(catalog.work['nonempty_packs_reused'],2)
        self.assertLessEqual(catalog.pack_key_bytes,262144)


if __name__=='__main__':
    unittest.main()
