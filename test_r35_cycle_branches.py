"""Independent witnesses for disjoint donor assignments and restricted Hungarian repair."""
from collections import Counter
from itertools import combinations, permutations, product
from importlib.util import find_spec
from unittest.mock import patch
import unittest
import numpy as np

from r27_incumbent import BoardModel
from r28_exchange_patches import AssignmentPrices
from r31_fresh import pack_cycles
from r35_cycle_branching import CycleBranches
from r35_restricted_assignment import identity_dual,matching
from search_fixtures import planted_gold
from search_problem import SearchProblem


class BranchTests(unittest.TestCase):
    def test_zero_gain_duals_respect_positive_cycles_and_absent_arcs(self):
        good=np.array([[0,2],[-3,0]])
        bad=np.array([[0,2],[-1,0]])
        allowed=np.ones((2,2),dtype=bool)
        self.assertEqual(identity_dual(good,allowed,0),(False,0))
        self.assertEqual(identity_dual(good,allowed,1),(True,1))
        self.assertFalse(identity_dual(bad,allowed,2)[0])
        allowed[1,0]=False
        self.assertTrue(identity_dual(bad,allowed,2)[0])
        rng=np.random.default_rng(3688)
        for n in range(2,7):
            for _ in range(20):
                w=rng.integers(-3,4,(n,n))
                ok=rng.random((n,n))>.3
                np.fill_diagonal(w,0)
                np.fill_diagonal(ok,True)
                optimum=max(sum(w[i,j] for i,j in enumerate(d)) for d in permutations(range(n))
                            if all(ok[i,j] for i,j in enumerate(d)))
                certified,_=identity_dual(w,ok,n)
                self.assertEqual(certified,optimum==0)

    def test_restricted_matching_agrees_with_all_injections_including_infeasible(self):
        rng=np.random.default_rng(3536)
        for size in range(2,7):
            for _ in range(28):
                weights=rng.integers(-3,4,(size,size))
                legal=rng.random((size,size))>.3
                parent,_=matching(weights,legal)
                tightened=weights-rng.integers(0,2,(size,size))
                restrictive=legal & (rng.random((size,size))>.15)
                for value,domain,seed in ((weights,legal,None),(tightened,restrictive,parent)):
                    result,_=matching(value,domain,seed)
                    candidates=[sum(value[i,p[i]] for i in range(size))
                                for p in permutations(range(size)) if all(domain[i,p[i]] for i in range(size))]
                    self.assertEqual(bool(candidates),result is not None)
                    if result is not None:
                        self.assertEqual(int(value[np.arange(size),result[0]].sum()),max(candidates))

    def test_changed_code_partition_has_no_intersection_and_forced_diagonal_can_rotate(self):
        tiles,geo,reference=planted_gold(3536,1,4)
        model=BoardModel(SearchProblem(tiles,geo))
        cells=np.array([0,*map(int,model.problem.neighbors[0]),6])
        cells=np.unique(cells)
        board=np.array(reference)
        prices=AssignmentPrices(model)
        prices.update(board)
        grid=prices.prices[cells][:,board[cells]//3]
        prior=grid[np.arange(len(cells)),np.arange(len(cells)),board[cells]%3]
        for policy in ('disjoint','neighbors','warm'):
            rules=CycleBranches(model,cells,board,grid,prior,policy)
            conflict=next((i,j) for i,links in enumerate(rules.neighbors) for j in links)
            children=list(rules.children(conflict,(),()))
            for selected in product((False,True),repeat=len(cells)):
                if any(selected[i] and selected[j] for i,links in enumerate(rules.neighbors) for j in links):
                    continue
                hits=sum(all(not selected[i] for i in stay) and all(selected[i] for i in change)
                         for stay,change in children)
                self.assertEqual(hits,1)
            allowed=np.full((len(cells),len(cells)),True)
            gain=np.zeros_like(allowed,dtype=int)
            changed=rules.require_changes((0,),allowed,gain,grid.argmax(axis=2))
            self.assertNotEqual(changed[0,0],board[cells[0]]%3)
            self.assertTrue(allowed[0,0])
            self.assertEqual(gain[0,0],grid[0,0,changed[0,0]]-prior[0])
        only_current=grid.copy()
        only_current[0,0,:]=-10
        only_current[0,0,board[cells[0]]%3]=prior[0]
        restricted=CycleBranches(model,cells,board,only_current,prior,'disjoint')
        allowed=np.ones((len(cells),len(cells)),dtype=bool)
        restricted.require_changes((0,),allowed,np.zeros_like(allowed,dtype=int),grid.argmax(axis=2))
        self.assertFalse(allowed[0,0])

    def test_explicit_regional_only_preserves_incomplete_status_without_calling_cp(self):
        from r31_fresh import Engine
        tiles,geo,_=planted_gold(35361,1,4)
        problem=SearchProblem(tiles,geo)
        for options in ({'regional_cycle_budget':0},{'regional_cycle_budget':3,'contacts':1}):
            with self.assertRaises(ValueError):
                Engine(problem,regional_only=True,**options)
        def incomplete(model,board,*args,**kwargs):
            return board.copy(),dict(complete=False,relaxations=1,cycles=0)
        engine=Engine(problem,pool=4,regional_cycle_budget=1,regional_only=True)
        with patch('r31_fresh.pack_cycles',incomplete),patch('r31_fresh.joint_repair',side_effect=AssertionError):
            row=engine.run(391,rounds=1)
            self.assertEqual(row['work']['joint_CIRCULATION_FEASIBLE'],1)
            self.assertEqual(row['work'].get('joint_CIRCULATION_OPTIMAL',0),0)

    @unittest.skipUnless(find_spec('ortools'), 'direct no-contact comparison requires OR-Tools')
    def test_skip_only_proved_identical_no_contact_queries(self):
        from r31_fresh import Engine, joint_repair
        tiles,geo,_=planted_gold(35361,1,4)
        problem=SearchProblem(tiles,geo)
        for options,expected in (({},0),({'contacts':0},0),({'adjacent':True},1),
                                 ({'contacts':2},1),({'disjoint':True},1)):
            engine=Engine(problem,pool=4,regional_cycle_budget=256,**options)
            with patch('r31_fresh.joint_repair',wraps=joint_repair) as direct:
                row=engine.run(391,rounds=1)
                self.assertEqual(direct.call_count,expected)
                self.assertEqual(row['work'].get('joint_CIRCULATION_OPTIMAL',0),int(not expected))
        engine=Engine(problem,pool=4,regional_cycle_budget=1)
        def incomplete(model,board,*args,**kwargs):
            return board.copy(),dict(complete=False,relaxations=1,cycles=0)
        with patch('r31_fresh.pack_cycles',incomplete),patch('r31_fresh.joint_repair',wraps=joint_repair) as direct:
            row=engine.run(391,rounds=1)
            self.assertEqual(direct.call_count,1)
            self.assertEqual(row['work'].get('joint_CIRCULATION_OPTIMAL',0),0)

    def test_complete_regional_optimum_matches_all_independent_code_assignments(self):
        tiles,geo,ref=planted_gold(35360,1,5)
        rng=np.random.default_rng(3536)
        for trial in range(18):
            problem=SearchProblem(tiles,geo)
            cells=np.sort(rng.choice(len(ref),5,replace=False))
            board=np.array(ref)
            board[cells]=board[rng.permutation(cells)]
            for c in range(len(ref)):
                problem.domains[c]=[v for v in problem.domains[c] if v==board[c] or rng.random()>.35]
            model=BoardModel(problem)
            best=int(model.matches(board).sum())
            for n in range(1,len(cells)+1):
                for moved in combinations(cells,n):
                    if any(b in model.problem.neighbors[a] for a,b in combinations(moved,2)):
                        continue
                    for order in permutations(moved):
                        for rot in product(range(3),repeat=n):
                            test=board.copy()
                            codes=3*(board[list(order)]//3)+np.array(rot)
                            if not np.all(model.legal[list(moved),codes]):
                                continue
                            test[list(moved)]=codes
                            best=max(best,int(model.matches(test).sum()))
            for policy in (None,'disjoint','neighbors','warm'):
                work=Counter()
                physical,status=pack_cycles(model,board,cells,AssignmentPrices(model),
                                            relaxations=2000,partition=policy,audit=work)
                self.assertTrue(status['complete'])
                self.assertEqual(int(model.matches(model.legal_board(physical)).sum()),best)
                self.assertEqual(work['open_upper'],best-int(model.matches(board).sum()))
                if policy:
                    self.assertEqual(work['duplicate_relaxed_boards'],0)


if __name__=='__main__':
    unittest.main()
