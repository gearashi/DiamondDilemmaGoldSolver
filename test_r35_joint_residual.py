import itertools
from importlib.util import find_spec
import unittest

import numpy as np

from constraint_symmetry import proper_rotations
from r27_incumbent import BoardModel
from r31_fresh import joint_repair,physical_bonds
from r35_forbidden_mates import ForbiddenMates
from r35_joint_residual import JointResidualCache
from search_fixtures import planted_gold
from search_problem import SearchProblem


class ResidualTests(unittest.TestCase):
    def test_rotated_query_decodes_exact_optimum_and_keeps_other_cells(self):
        tiles,geo,reference=planted_gold(3541,1,5)
        model=BoardModel(SearchProblem(tiles,geo))
        cells=[0,1,2,3]
        board=np.array(reference)
        board[cells]=board[cells[::-1]]
        transform=proper_rotations(geo)[6]
        rotated=np.array(transform.apply(board.tolist()))
        moved=[transform.cells[c] for c in cells]
        cache=JointResidualCache()
        # A validated planted full-score answer is an exact solver-independent
        # admission witness for the NumPy-only profile. Full builds also solve it.
        def planted(*args,**kw):
            self.assertTrue(model.problem.validate(reference)['valid'])
            return np.array(reference),dict(status='OPTIMAL',branches=0)
        source=joint_repair if find_spec('ortools') else planted
        old,report=cache.solve(source,model,board,cells,contacts=-1,effort=3)
        self.assertEqual(report['status'],'OPTIMAL')
        self.assertTrue(model.problem.validate(old.tolist())['valid'])
        def forbidden_build(*args,**kwargs):
            self.fail('A certified transformed residual was rebuilt.')
        new,report=cache.solve(forbidden_build,model,rotated,moved,contacts=-1,effort=3)
        np.testing.assert_array_equal(new,transform.apply(old.tolist()))
        self.assertEqual(report['branches'],0)
        self.assertEqual(cache.work['proved_hits'],1)

    def test_contact_limits_forbidden_pairs_and_domains_do_not_alias(self):
        tiles,gold,reference=planted_gold(35412,1,5)
        model=BoardModel(SearchProblem(tiles,gold))
        b=np.array(reference)
        p,s,q,t=map(int,model.problem.edges[0])
        cells=[p,q]
        cache=JointResidualCache()
        def key(limit,forbidden=None,disjoint=False):
            return cache.describe(model,b,cells,limit,disjoint,forbidden)[0]
        self.assertEqual(len({key(0),key(1),key(-1),key(-1,disjoint=True)}),4)
        current=set(physical_bonds(model,b))
        inside=3*(b[p]//3)+(s-b[p]%3)%3
        outside=3*(b[q]//3)+(t-b[q]%3)%3
        pair=tuple(sorted((inside,outside)))
        self.assertIn(pair,current)
        self.assertNotEqual(key(1,ForbiddenMates(model,[])),key(1,ForbiddenMates(model,[pair])))
        before=key(1)
        model.legal[p,3*(b[q]//3)]=False
        self.assertNotEqual(before,key(1))

    def test_replay_of_all_contact_policies_matches_independent_finite_search(self):
        tiles,geometry,reference=planted_gold(35419,1,4)
        m=BoardModel(SearchProblem(tiles,geometry))
        rng=np.random.default_rng(35419)
        for run in range(12):
            slots=rng.choice(m.n,3,replace=False).tolist()
            b=np.asarray(reference).copy()
            b[slots]=3*(b[slots[::-1]]//3)+rng.integers(3,size=3)
            protected=set(physical_bonds(m,np.asarray(reference)))-set(physical_bonds(m,b))
            rules=ForbiddenMates(m,sorted(protected)[:1])
            cache=JointResidualCache()
            limit=(0,1,-1)[run%3]
            disjoint=bool(run%2)
            best=-1
            for perm in itertools.permutations(b[slots]//3):
                for rotations in itertools.product(range(3),repeat=3):
                    candidate=b.copy()
                    candidate[slots]=3*np.array(perm)+np.array(rotations)
                    if not m.legal[slots,candidate[slots]].all() or rules.violations(candidate).any():continue
                    changed=set(np.flatnonzero(b!=candidate))
                    degrees=[sum(nb in changed for nb in m.problem.neighbors[c]) for c in changed]
                    if (limit>=0 and sum(degrees)>2*limit) or (disjoint and max(degrees,default=0)>1):continue
                    score=int(m.matches(candidate).sum())
                    if score>best:
                        best,best_witness=score,candidate
            def finite_optimum(*args,**kw):
                return best_witness.copy(),dict(status='OPTIMAL',branches=0)
            source=joint_repair if find_spec('ortools') else finite_optimum
            answer,_=cache.solve(source,m,b,slots,contacts=limit,disjoint=disjoint,forbidden=rules,effort=2)
            self.assertEqual(int(m.matches(answer).sum()),best)
            replay=JointResidualCache()
            receipt=cache.snapshot('physical-fixture')
            replay.restore('physical-fixture',receipt)
            copy,work=replay.solve(joint_repair,m,b,slots,contacts=limit,disjoint=disjoint,forbidden=rules)
            np.testing.assert_array_equal(answer,copy)
            self.assertEqual(work,dict(status='OPTIMAL',branches=0))
            with self.assertRaisesRegex(ValueError,'Foreign'):
                replay.restore('other',receipt)
            bad=dict(receipt,sha256='1'*64)
            with self.assertRaisesRegex(ValueError,'Invalid'):
                replay.restore('physical-fixture',bad)

    def test_incomplete_results_are_never_proofs_and_payloads_are_bounded(self):
        tiles,g,reference=planted_gold(35411,1,5)
        model=BoardModel(SearchProblem(tiles,g))
        b=np.array(reference)
        cache=JointResidualCache(capacity=1,max_bytes=1024)
        def fake(model,board,cells,**kw):
            return board.copy(),dict(status=kw['status'],branches=1)
        for status in ('FEASIBLE','UNKNOWN','INFEASIBLE'):
            cache.solve(fake,model,b,[0,1],contacts=1,status=status)
        self.assertFalse(cache.rows)
        for cells in ([0,1],[1,2],[2,3]):
            cache.solve(fake,model,b,cells,contacts=1,status='OPTIMAL')
            self.assertLessEqual(len(cache.rows),1)
            self.assertLessEqual(cache.occupied,1024)
        self.assertEqual(cache.work['optima_admitted'],3)


if __name__=='__main__':
    unittest.main()
