from collections import Counter
from itertools import product,combinations
from unittest import TestCase
import numpy as np

from r27_incumbent import BoardModel
from r32_fresh_phases import SignedRepair
from r35_signed_refit import RefitBounds,changed_score,changed_dart_prices
from test_r33_gold_refine import fixture


class RefitTests(TestCase):
    def test_transferred_ceiling_dominates_every_changed_objective(self):
        limits=RefitBounds()
        terms=[(0,1),(2,3),(4,5),(6,7),(8,9)]
        allowed=[{terms[i] for i,x in enumerate(mask) if x} for mask in product((False,True),repeat=5) if sum(mask)<=2]
        for size in range(6):
            for subset in combinations(terms,size):
                old=set(subset)
                limits.record(old,max(len(old&x) for x in allowed))
                for k in range(6):
                    for row in combinations(terms,k):
                        goal=set(row)
                        self.assertGreaterEqual(limits.upper(goal),max(len(goal&x) for x in allowed))
        self.assertEqual(limits.upper(set(terms)),2)
        saved=limits.snapshot('abc',set())
        for fingerprint in ('abc','changed'):
            restored=RefitBounds()
            restored.restore(saved,fingerprint,set())
            self.assertEqual(restored.upper(set(terms)),2 if fingerprint=='abc' else 5)
        saved['checksum']='invalid'
        resumed=RefitBounds()
        resumed.restore(saved,'abc',set())
        self.assertEqual(resumed.upper(set(terms)),5)

    def test_sparse_dense_complement_dart_rows_and_seam_delta(self):
        p,ref,g,req,out=fixture(True,True)
        model=BoardModel(p)
        initial=[0]*p.n
        prior=SignedRepair(model,initial,forbidden=[out[0]])
        for dart in range(3*p.n):
            prior.prices[dart]=prior.geometry.price(dart,prior.dart_phases,prior.stats)
        for phases in ([1]+initial[1:],[i%2 for i in range(p.n)],[1]*p.n):
            current=SignedRepair(model,phases,forbidden=[out[0]])
            for dart in range(3*p.n):
                patch=changed_dart_prices(prior,current,dart)
                cold=current.geometry.price(dart,current.dart_phases,Counter())
                np.testing.assert_array_equal(patch,cold)
            board=np.arange(p.n)*3
            score=current.score(board)
            for cells in ((0,),tuple(range(p.n)),()):
                other=board.copy()
                for cell in cells:
                    other[cell]+=1
                self.assertEqual(changed_score(current,other,board,score),current.score(other))
                self.assertEqual(changed_score(current,other,board,prior.score(board),prior),
                                 current.score(other))

    def test_matching_prior_database_scope_required(self):
        p,ref,g,req,out=fixture(True,True)
        model=BoardModel(p)
        a=SignedRepair(model,[0]*p.n)
        b=SignedRepair(model,[1]*p.n,[out[0]])
        self.assertIsNone(changed_dart_prices(a,b,0))

    def test_raw_ties_are_lexicographic_for_every_legal_donor_permutation(self):
        from itertools import permutations,product
        p,_,_,_,_=fixture(True,True)
        model=BoardModel(p)
        board=np.arange(p.n)*3
        current=SignedRepair(model,[i%2 for i in range(p.n)],raw_ties=True)
        rng=np.random.default_rng(12)
        cells=model.independent_set(board,rng)[:3]
        result=current.block(board,cells)
        def score(codes):
            return current.score(codes)[0],int(model.matches(codes).sum()),int((codes[cells]==board[cells]).sum())
        best=score(board)
        for ids in permutations(board[cells]//3):
            for turns in product(range(3),repeat=len(cells)):
                other=board.copy()
                other[cells]=3*np.asarray(ids)+turns
                if model.legal[cells,other[cells]].all():
                    best=max(best,score(other))
        self.assertEqual(score(result),best)
