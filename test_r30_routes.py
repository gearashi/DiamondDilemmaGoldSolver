"""Differentials against independent complete donor permutations and rotations."""
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_native import find_compiler
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_route import swap_orders
from r30_cycle_joins import TypedJoin, matching_bound
from r30_cycles import Predecessors, canonical_cycles
from r30_route import NativeCycles,search
from search_fixtures import planted_gold
from search_problem import SearchProblem


def oracle(bound, pool, family, target):
    n=sum(family)
    cells=np.array(list(itertools.combinations(pool,n)),int)
    result=[]
    for order in swap_orders(n,n-len(family)):
        if sorted(map(len,canonical_cycles(range(n),order)))!=sorted(family):continue
        for row in cells[bound.bounds(cells,order)>=target]:
            result.append(tuple(sorted(zip(map(int,row),(int(row[j]) for j in order)))))
    return set(result)


def identity(cycles):
    return tuple(sorted((a,b) for cycle in cycles for a,b in zip(tuple(cycle),tuple(cycle[1:])+tuple(cycle[:1]))))


class CycleRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geometry,cls.codes=planted_gold(39,1,3)
        cls.p=SearchProblem(tiles,geometry)
        cls.bound=ExchangeBound(BoardModel(cls.p),cls.codes)

    def test_simultaneous_first_build_publishes_once(self):
        import r30_route
        try:
            find_compiler()
        except RuntimeError:
            self.skipTest('No native compiler')
        # Windows retains mapped DLLs until the process exits. Run both imports in
        # a separate interpreter so the temporary build directory is then removable.
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            from shutil import copyfile
            copyfile(Path(r30_route.ROOT)/'r30_cycle_native.c',Path(tmp)/'r30_cycle_native.c')
            program = """from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
from pathlib import Path
import sys
import r30_route
r30_route.ROOT = Path(sys.argv[1])
gate = Barrier(2)
def load():
    gate.wait()
    return r30_route.library().cycle_abi()
with patch.object(r30_route,'_build_one',wraps=r30_route._build_one) as build:
    with ThreadPoolExecutor(2) as pool:
        values = [pool.submit(load) for _ in range(2)]
        assert [v.result() for v in values] == [1,1]
    assert build.call_count == 1, build.call_count
"""
            subprocess.run([sys.executable,'-B','-c',program,tmp],cwd=r30_route.ROOT,check=True,timeout=90)

    def test_native_prices_and_overflow(self):
        try:find_compiler()
        except RuntimeError:self.skipTest('No native compiler')
        for family in ((4,2),(3,3)):
            for target in (6,10,13):
                expected=oracle(self.bound,range(self.p.n),family,target)
                engine=NativeCycles(self.bound,target,family)
                try:
                    counts,survivors=engine.page(0,self.p.n**2,False,cap=73)
                finally:engine.close()
                got=[identity((row[:family[0]],row[family[0]:])) for row in survivors]
                self.assertCountEqual(expected,got)
                self.assertEqual(counts['survivors'],len(got))
                self.assertLessEqual(counts['joins'],math.comb(self.p.n,6)*(90 if family[0]==4 else 40))

    def test_typed_three_and_four_way_and_threshold_order(self):
        for family in ((4,2),(3,3),(3,2,2),(2,2,2,2)):
            k=sum(family)
            target=10
            pool=range(k)
            reference=oracle(self.bound,pool,family,target)
            for order_by_gain in (True,False):
                join=TypedJoin(self.bound,pool,family,target)
                rows=[identity(cycles) for cycles,_,_ in join.survivors(order_by_gain=order_by_gain)]
                self.assertCountEqual(rows,reference)
            split=TypedJoin(self.bound,pool,family,target)
            self.assertCountEqual([identity(cs) for cs,_,_ in split.half_survivors()],reference)

    def test_matching_and_predecessor_safe_scope(self):
        rng=np.random.default_rng(13)
        for family in ((2,4),(3,3),(3,2,2),(2,2,2,2)):
            n=sum(family);weights=rng.integers(-4,4,(n,n))
            expected=max(sum(weights[i,j] for i,j in enumerate(row)) for row in swap_orders(n,n-len(family))
                         if sorted(map(len,canonical_cycles(range(n),row)))==sorted(family))
            self.assertEqual(matching_bound(weights,family),expected)
        pred=Predecessors(self.bound,10)
        self.assertIsNone(pred.reject(((0,1),(2,3),(4,5),(6,7)),deep=True))

    def test_one_contact_complete_enumeration_includes_planted_hit(self):
        from bench_r30_single_contact import one_contact, prices
        from r29_exchange_clusters import StarClusters
        from r29_exchange_route import rotation_optimum
        from r28_exchange_rotations import RotationOracle
        data,geo,solution=planted_gold(41,2,3)
        moved=[26,2,28,13,3,6,37]
        fixed={c:k for c,k in enumerate(solution) if c not in moved}
        problem=SearchProblem(data,geo,fixed=fixed)
        center=np.array(solution)
        for cycle in (moved[:3],moved[3:5],moved[5:]):
            center[cycle]=center[cycle[1:]+cycle[:1]]
        model=BoardModel(problem)
        bound=ExchangeBound(model,center)
        stars=StarClusters(bound)
        top={k:[z[0] for z in prices(bound,stars,k,-200)[:4]] for k in (2,3)}
        family=(3,2,2);target=len(problem.edges)
        result=one_contact(bound,target,family,top,stars,{})
        self.assertFalse(result['second_contact_required'])
        cells=tuple(sorted(moved))
        seen={}
        oracle=RotationOracle(model)
        for order in swap_orders(7,4):
            cs=canonical_cycles(cells,order)
            if sorted(map(len,cs))!=[2,2,3]:continue
            if sum(stars.close[a,b] for i,C in enumerate(cs) for D in cs[:i] for a in C for b in D)!=1:continue
            if stars.price(cells,order)<2*target:continue
            seed=bound.witness_seed(cells,order)
            if seed is not None:
                seen[identity(cs)]=rotation_optimum(bound,oracle,seed)['score']
        got={identity(r['cycles']):r['score'] for r in result['star_survivors']}
        self.assertEqual(got,seen)

    def test_contacting_component_price_catalogue(self):
        from collections import Counter
        from bench_r30_cycle_groups import coupled,joined_order
        from r29_exchange_clusters import StarClusters
        try:find_compiler()
        except RuntimeError:self.skipTest('No native compiler')
        bound=self.bound
        stars=StarClusters(bound)
        engine=NativeCycles(bound,12,(4,2),component=True)
        try:
            for k in (2,3):
                size=k+2
                expected=[]
                threshold=18-bound.total
                for cells in itertools.combinations(range(bound.n),size):
                    for order in swap_orders(size,k):
                        cs=canonical_cycles(cells,order)
                        if sorted(map(len,cs))!=[2,k]:continue
                        if not any(stars.close[a,b] for a in cs[0] for b in cs[1]):continue
                        score=stars.price(cells,order)-bound.total
                        if score>=threshold:expected.append((identity(cs),score))
                rows=coupled(bound,stars,k,threshold,engine,Counter())
                self.assertCountEqual([(identity(cs),g) for g,_,cs in rows],expected)
        finally:engine.close()

    def test_budget_certificate_closes_before_native_and_resumes(self):
        from unittest.mock import patch
        tiles,geometry,codes=planted_gold(72,2,4)
        p=SearchProblem(tiles,geometry)
        codes=np.random.default_rng(304).permutation(codes)
        with tempfile.TemporaryDirectory() as tmp, patch('r30_route.library',side_effect=AssertionError):
            db=Path(tmp)/'budget.db'
            result=search(p,codes,len(p.edges),db,(3,3))
            self.assertTrue(result['complete'])
            self.assertEqual(result['star_queries'],0)
            self.assertLess(result['budget_upper'],len(p.edges))
            with patch('r30_route.budget_upper',side_effect=AssertionError):
                again=search(p,codes,len(p.edges),db,(3,3))
            self.assertEqual(result['budget_upper'],again['budget_upper'])

    def test_resume_and_checked_witness(self):
        try:find_compiler()
        except RuntimeError:self.skipTest('No native compiler')
        board=np.asarray(self.codes).copy()
        cycle1=[0,1,2];cycle2=[3,4,5]
        for c in (cycle1,cycle2):board[c]=board[c[1:]+c[:1]]
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'cycle.db'
            r=search(self.p,board,len(self.p.edges),db,(3,3),max_pages=1,batch=2)
            self.assertFalse(r['complete'])
            r=search(self.p,board,len(self.p.edges),db,(3,3))
            self.assertTrue(r['complete']);self.assertGreaterEqual(r['hits'],1)
            self.assertEqual(self.p.validate(r['best']['codes'])['matched_edges'],len(self.p.edges))
            again=search(self.p,board,len(self.p.edges),db,(3,3))
            self.assertEqual(again['star_queries'],r['star_queries'])
            self.assertEqual(again['hits'],r['hits'])
