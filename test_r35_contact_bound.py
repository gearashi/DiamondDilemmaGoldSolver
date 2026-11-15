import itertools
from collections import Counter
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r31_fresh import physical_bonds
from r35_contact_bound import bounded_assignment,no_contact_gain
from r35_forbidden_mates import ForbiddenMates
from search_fixtures import planted_gold
from search_problem import SearchProblem


class ContactBoundTests(unittest.TestCase):
    def test_all_small_relocations_and_rotations_with_contact_policies(self):
        data,geometry,reference=planted_gold(354112,1,5)
        m=BoardModel(SearchProblem(data,geometry))
        rng=np.random.default_rng(1904)
        target=np.array(reference)
        for run in range(18):
            cells=rng.choice(m.n,4,replace=False).tolist()
            b=target.copy()
            b[cells]=3*(b[cells[::-1]]//3)+rng.integers(3,size=4)
            banned=sorted(physical_bonds(m,target)-physical_bonds(m,b))[:1]
            rules=ForbiddenMates(m,banned)
            limit=(0,1,-1)[run%3]
            disjoint=bool(run%3==2)
            best=int(m.matches(b).sum())
            for order in itertools.permutations(b[cells]//3):
                for rot in itertools.product(range(3),repeat=4):
                    c=b.copy()
                    c[cells]=3*np.array(order)+np.array(rot)
                    changed=set(np.flatnonzero(c!=b))
                    degrees=[sum(nb in changed for nb in m.problem.neighbors[t]) for t in changed]
                    if limit>=0 and sum(degrees)//2>limit:continue
                    if disjoint and max(degrees,default=0)>1:continue
                    if rules.violations(c).any():continue
                    best=max(best,int(m.matches(c).sum()))
            for level in ('dual','supports','exact'):
                witness=[]
                closed,stats=no_contact_gain(m,b,cells,contacts=limit,disjoint=disjoint,forbidden=rules,level=level,witness=witness)
                if closed:self.assertEqual(best,int(m.matches(b).sum()))
                if level=='exact':
                    self.assertEqual(best,int(m.matches(witness[0] if witness else b).sum()))
                    self.assertTrue(closed != bool(witness))

    def test_threshold_upper_can_be_reopened_and_exact_witness_reused(self):
        one=np.array([[-1000,3,-3],[0,-1000,1],[2,-1,-1000]])
        units=((0,),(1,),(2,))
        rows=(0,1,2)
        prices=np.zeros(3,dtype=int)
        memo={}
        stats=Counter()
        def run(target):
            return bounded_assignment(rows,units,one,{},True,target,prices,stats,memo)
        self.assertLess(run(10)[0],0)
        value,_,placement=run(0)
        self.assertEqual(value,6)
        self.assertEqual(placement,{(0,):(1,),(1,):(2,),(2,):(0,)})
        self.assertEqual(run(3)[0],6)
        self.assertGreater(stats['bounded_hits'],0)

    def test_perfect_complete_loop_and_unsupported_overlapping_changes(self):
        data,g,ref=planted_gold(354117,1,4)
        m=BoardModel(SearchProblem(data,g))
        b=np.array(ref)
        c=0
        cells=[c,*m.problem.neighbors[c]]
        self.assertEqual(no_contact_gain(m,b,cells,contacts=-1,disjoint=False)[1],{'outside_scope':1})
        closed,stats=no_contact_gain(m,b,cells,contacts=1,disjoint=False,level='exact')
        self.assertTrue(closed)
        self.assertTrue(m.problem.validate(b.tolist())['valid'])


if __name__=='__main__':
    unittest.main()
