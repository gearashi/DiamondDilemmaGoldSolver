"""Physical evidence for move-cycle rollback and forbidden-mate local domains."""
from itertools import permutations, product
from importlib.util import find_spec
import unittest
import numpy as np

from r27_incumbent import BoardModel
from r31_fresh import joint_repair,physical_bonds
from r35_forbidden_mates import ForbiddenMates,GuardedPrices
from search_fixtures import planted_gold
from search_problem import SearchProblem


class MateTests(unittest.TestCase):
    def fixture(self,uniform=False):
        tiles,geo,reference=planted_gold(3537,1,4)
        if uniform:
            for tile in tiles['tiles']:
                tile['segments']=[[[0,4],[1,8]],[[1,4],[2,8]],[[2,4],[0,8]]]
        return BoardModel(SearchProblem(tiles,geo)),np.array(reference)

    def test_reverting_one_collision_can_create_a_different_forbidden_mate(self):
        model,trial=self.fixture(True)
        old=trial.copy()
        cells=[0,*map(int,model.problem.neighbors[0])]
        for c in cells:
            old[c]=3*(old[c]//3)+(old[c]%3+1)%3
        reverted=trial.copy()
        reverted[0]=old[0]
        prior=physical_bonds(model,old)
        final=physical_bonds(model,trial)
        intermediate=physical_bonds(model,reverted)
        first=next(iter(final-prior-intermediate))
        second=next(iter(intermediate-prior-final))
        rules=ForbiddenMates(model,[first,second])
        self.assertTrue(rules.violations(trial).any())
        self.assertTrue(rules.violations(reverted).any())
        winner,work=rules.project(old,trial)
        self.assertFalse(rules.violations(winner).any())
        self.assertFalse(physical_bonds(model,winner)&{first,second})
        self.assertGreater(work['exact_cycle_subsets'],1)

    def test_constraints_and_blocked_codes_agree_with_physical_darts(self):
        model,board=self.fixture(True)
        trial=board.copy()
        trial[[0,2]]=trial[[2,0]]
        new=physical_bonds(model,trial)-physical_bonds(model,board)
        rules=ForbiddenMates(model,new)
        blocked=set(rules.blocked_codes(trial))
        for c in range(model.n):
            for side,other in enumerate(model.problem.neighbors[c]):
                facing=model.problem.sides[c,side]
                for code in range(model.n*3):
                    self.assertEqual((c,code,int(other)) in blocked,
                                     bool(rules.edge_bans(code,side,trial[other],facing)))
        flat=np.arange(3*model.n)[::2]
        label=np.full(3*model.n,-1,dtype=int)
        label[flat]=np.arange(len(flat))
        for s,t in product(range(3),repeat=2):
            dense=set(zip(*np.nonzero(rules.edge_bans(flat[:,None],s,flat[None,:],t))))
            self.assertEqual(sorted(dense),list(map(tuple,rules.local_pairs(label,s,t))))
        prices=GuardedPrices(model,rules)
        for values in (board,trial,board):
            prices.update(values)
            current=set(rules.blocked_codes(values))
            for c,code,other in current:
                if values[c]!=code:
                    self.assertEqual(prices.prices[c,code//3,code%3],-10)
        no_rules=ForbiddenMates(model,[])
        self.assertIs(no_rules.project(board,trial)[0],trial)

    def test_three_donor_cycles_against_all_allowed_rollback_choices(self):
        model,base=self.fixture()
        rng=np.random.default_rng(3537)
        tested=0
        for _ in range(100):
            changed=rng.choice(model.n,6,replace=False)
            old=base.copy()
            trial=base.copy()
            for a,b in np.asarray(changed).reshape(3,2):
                old[a],old[b]=old[b],old[a]
            for c in changed:
                old[c]=3*(old[c]//3)+rng.integers(3)
            groups=[list(map(int,p)) for p in np.asarray(changed).reshape(3,2)]
            states=[]
            for active in product((False,True),repeat=len(groups)):
                row=old.copy()
                for keep,positions in zip(active,groups):
                    if keep:row[positions]=trial[positions]
                states.append(row)
            banned=physical_bonds(model,trial)-physical_bonds(model,old)
            rules=ForbiddenMates(model,banned)
            if not banned:continue
            answer,work=rules.project(old,trial)
            allowed=[v for v in states if not banned & physical_bonds(model,v)]
            if work['active_cycles']==len(groups):
                self.assertEqual(int(model.matches(answer).sum()),max(int(model.matches(v).sum()) for v in allowed))
                tested+=1
        self.assertGreater(tested,15)

    @unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
    def test_internal_and_exterior_physical_pairs_in_contacting_cp(self):
        model,target=self.fixture()
        board=target.copy()
        cells=[0,1,2,3]
        board[cells]=board[cells[::-1]]
        banned=physical_bonds(model,target)-physical_bonds(model,board)
        rules=ForbiddenMates(model,banned)
        for limit in (0,1,-1):
            best=0
            for ids in permutations(cells):
                for rot in product(range(3),repeat=len(cells)):
                    row=board.copy()
                    row[cells]=3*(board[list(ids)]//3)+np.array(rot)
                    moved=set(np.flatnonzero(board!=row))
                    contacts=sum(a in moved and b in moved for a,_,b,_ in model.problem.edges)
                    if ((limit>=0 and contacts>limit) or physical_bonds(model,row)&banned):
                        continue
                    best=max(best,int(model.matches(row).sum()))
            winner,info=joint_repair(model,board,cells,contacts=limit,forbidden=rules,effort=4)
            self.assertEqual(info['status'],'OPTIMAL')
            self.assertFalse(physical_bonds(model,winner)&banned)
            self.assertEqual(int(model.matches(winner).sum()),best)


if __name__=='__main__':
    unittest.main()
