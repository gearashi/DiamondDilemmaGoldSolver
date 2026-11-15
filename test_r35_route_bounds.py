from itertools import combinations, permutations, product
import unittest
import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r28_exchange_rotations import RotationOracle
from r30_exact_routes import cycle_type
from r34_component_budget import ComponentBudget
from r34_exchange_undo import CoveredUndo
from r35_donor_chains import family_budget, stitch
from r35_regional_frontier import frontier
from r35_route_difference import differential
from r35_route_splits import predecessor_splits
from search_fixtures import planted_gold
from search_problem import SearchProblem


class RegionalRouteBounds(unittest.TestCase):
    def test_dominance_direction_and_exact_budget_reuse(self):
        states={(1,2,1):5,(1,2,3):5,(1,3,2):4,(1,3,5):1,(2,3,1):0}
        self.assertEqual(frontier(states),{(1,2,3):5,(1,3,5):1,(2,3,1):0})
        tiles,geo,codes=planted_gold(68,2,4)
        b=ExchangeBound(BoardModel(SearchProblem(tiles,geo)),codes)
        pool=(0,2,12,18,22,36)
        cold=ComponentBudget(b,pool,dominance=False)
        warm=ComponentBudget(b,pool)
        self.assertEqual(cold.upper(4)['upper'],warm.upper(4,retain_front=True)['upper'])
        for swap in range(4):
            for moved in range(7):
                got=warm.upper(swap,min_moved=moved)
                self.assertEqual(got['upper'],cold.upper(swap,min_moved=moved)['upper'])
                self.assertEqual(got['work']['joins'],0)

    def test_chain_scope_stitches_all_family_types_and_separators(self):
        self.assertEqual(stitch(((4,1,2),(1,5,1),(5,4,3),(3,3,1),(0,6,2))),(((0,6,2),),(6,)))
        tiles,geo,codes=planted_gold(62,2,4)
        b=ExchangeBound(BoardModel(SearchProblem(tiles,geo)),codes)
        checked=set()
        for pool in ((0,2,12,18,22,36),(0,1,2,3,4,5),(0,1,3,8,12,15)):
            c=ComponentBudget(b,pool)
            maxima={}
            for order in permutations(range(len(pool))):
                fam=tuple(sorted(cycle_type(order)))
                if not fam:
                    continue
                physical=b.pieces.copy()
                physical[list(pool)]=b.tiles[[pool[j] for j in order]]
                value=int(b.stars(physical,physical[b.neighbors]).sum())//2
                maxima[fam]=max(maxima.get(fam,-1),value)
            for fam,expected in maxima.items():
                value=family_budget(c,fam)
                if value['complete']:
                    checked.add(len(value.get('separator',())))
                    self.assertEqual(value['upper'],min(value['coarse']['upper'],expected))
                else:
                    self.assertEqual(value['upper'],value['coarse']['upper'])
        self.assertIn(0,checked)
        self.assertTrue(checked & {1,2})

    def test_certified_transposition_and_differential_with_changed_donors(self):
        tiles,geo,codes=planted_gold(1134,1,4)
        b=ExchangeBound(BoardModel(SearchProblem(tiles,geo)),codes)
        oracle=RotationOracle(b.model)
        def optimum(group):
            row=b.board.copy()
            for cycle in group:
                for c,d in zip(cycle,(*cycle[1:],cycle[0])):
                    row[c]=3*b.tiles[d]
            return oracle.optimum(row)[0]
        for group in (((0,1,2,3,4,5),),((0,1),(2,3,4,5,6)),((0,3,7),(1,2,4,6))):
            endpoint=optimum(group)
            for pred,cells,old,new,moves in predecessor_splits(group):
                self.assertEqual(sum(len(row)-1 for row in pred),sum(len(row)-1 for row in group)-1)
                before=dict(zip(cells,old))
                after=dict(zip(cells,new))
                common={c:d for c,d in moves.items() if c not in cells}
                delta=differential(b,before,after,common)
                self.assertGreaterEqual(optimum(pred)+delta,endpoint)
                self.assertIsNone(differential(b,before,after,common,limit=1))
            strict=CoveredUndo(b,lambda g:False,predecessor_upper=optimum,target=endpoint)
            self.assertFalse(strict.rejects(group))
            unknown=CoveredUndo(b,lambda g:False,predecessor_upper=lambda g:None,target=endpoint+1)
            self.assertFalse(unknown.rejects(group))

    def test_differencing_after_boundary_maximization_can_reverse_the_sign(self):
        tiles,geo,codes=planted_gold(15,1,6)
        b=ExchangeBound(BoardModel(SearchProblem(tiles,geo)),codes)
        cell,donor=0,2
        near=sorted(set(int(c) for c in b.neighbors[cell] if c!=b.n))
        values=[[],[]]
        for outside in product(range(3),repeat=len(near)):
            local=np.array(b.board)
            local[near]=3*b.tiles[near]+outside
            for index,d in enumerate((cell,donor)):
                scores=[]
                for r in range(3):
                    local[cell]=3*b.tiles[d]+r
                    scores.append(sum(int(b.model.masks[local[a],s]==b.model.reverse[b.model.masks[local[n],t]])
                                      for a,s,n,t in b.model.problem.edges if cell in (a,n)))
                values[index].append(max(scores))
        self.assertEqual(max(values[1])-max(values[0]),-2)
        self.assertEqual(max(y-x for x,y in zip(*values)),1)
        self.assertEqual(differential(b,{cell:cell},{cell:donor},{}),1)

    def test_numeric_predecessor_ceiling_can_pay_for_positive_gain(self):
        tiles,geo,codes=planted_gold(1134,1,4)
        b=ExchangeBound(BoardModel(SearchProblem(tiles,geo)),codes)
        oracle=RotationOracle(b.model)
        group=((0,1,2,3,4,5),)
        seen=False
        for pred,cells,old,new,moves in predecessor_splits(group):
            exterior={c:d for c,d in moves.items() if c not in cells}
            gain=differential(b,dict(zip(cells,old)),dict(zip(cells,new)),exterior)
            if gain<=0:
                continue
            placement=b.board.copy()
            for cycle in pred:
                for c,d in zip(cycle,(*cycle[1:],cycle[0])):
                    placement[c]=3*b.tiles[d]
            exact=oracle.optimum(placement)[0]
            ceiling=lambda g:exact if g==pred else None
            equal=CoveredUndo(b,lambda g:False,predecessor_upper=ceiling,target=exact+gain)
            self.assertFalse(equal.rejects(group))
            past=CoveredUndo(b,lambda g:False,predecessor_upper=ceiling,target=exact+gain+1)
            self.assertTrue(past.rejects(group))
            seen=True
            break
        self.assertTrue(seen)
