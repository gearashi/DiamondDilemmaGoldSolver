from itertools import combinations
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import is_owned
from r30_cycles import canonical_cycles
from r30_multicenter import orders, run
from r31_route_coverage import register,inherited_end
from r31_route_geometry import ConnectedJoin,connected,eligible_orders,proved_connected
from r31_route_price import DefectPrices,Exterior,SharedStarPrice,six_comparisons,comparison_relations
from r31_routes import rank,search_roots
from search_fixtures import planted_gold
from search_problem import SearchProblem


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,cls.codes = planted_gold(42,1,5)
        cls.model = BoardModel(SearchProblem(tiles,geo))
        cls.bound = ExchangeBound(cls.model,cls.codes)
        cls.stars = StarClusters(cls.bound)

    def test_connected_and_canonical_at_positive_and_zero_floor(self):
        pool = tuple(range(8))
        for f in ((2,2,3),(2,2,2,2)):
            expected = []
            for cells in combinations(pool,sum(f)):
                direct = [o for o in orders(f) if connected(canonical_cycles(cells,o),self.stars.close)]
                self.assertEqual(tuple(eligible_orders(self.stars.close,cells,f)),tuple(direct))
                expected.extend((cells,o) for o in direct)
            self.assertFalse(proved_connected(self.bound,15,f))
            for target in (0,12):
                reference = {(cells,o) for cells,o in expected if self.stars.price(cells,o)>=2*target}
                for priced in (False,True):
                    join = ConnectedJoin(self.stars,pool,f,target)
                    got = list(join.survivors(price=priced))
                    self.assertEqual(set(got),reference)
                    self.assertEqual(len(got),len(set(got)))

    def test_donor_and_shared_exterior_against_rotation_oracle(self):
        p = DefectPrices(self.bound)
        cells = tuple(range(7))
        family = (2,2,3)
        rows = orders(family)[::18]
        oracle = RotationOracle(self.model)
        ext = Exterior(self.bound,cells)
        joint = SharedStarPrice(self.bound,cells)
        self.assertEqual(joint.upper(rows),max(self.stars.price(cells,o) for o in rows))
        for o in rows:
            score = oracle.optimum(self.bound.witness_seed(cells,o))[0]
            self.assertEqual(ext.score(o),score)
            self.assertGreaterEqual(p.endpoint_upper(cells,o),score)
            group = canonical_cycles(cells,o)
            self.assertGreaterEqual(p.actor_upper(group[:1],5,pool=cells),score)
        self.assertIsNone(Exterior(self.bound,cells,cap=1).score(rows[0]))
        self.assertEqual(ext.score(rows[0]),ext.score(rows[0],shared=False))

    def test_comparisons_are_six_cells_and_only_reject_dominated_endpoints(self):
        cells = tuple(range(8))
        oracle = RotationOracle(self.model)
        for family in ((2,2,3),(2,2,2,2)):
            chosen = cells[:sum(family)]
            order = orders(family)[1]
            group = canonical_cycles(chosen,order)
            candidates = six_comparisons(group)
            self.assertTrue(candidates)
            for comp in candidates:
                self.assertEqual(len(comp),6)
                self.assertEqual(sorted(dict(comp)),sorted(dict(comp).values()))
            keep,work = comparison_relations(self.bound,group,max_comparisons=3)
            if not keep:
                score = oracle.optimum(self.bound.witness_seed(chosen,order))[0]
                possible = []
                for comp in candidates:
                    m = dict(comp)
                    moved = tuple(m)
                    perm = tuple(moved.index(m[c]) for c in moved)
                    possible.append(oracle.optimum(self.bound.witness_seed(moved,perm))[0])
                    if possible[-1]>=score:
                        break
                self.assertGreaterEqual(max(possible),score)

    def test_paged_cycles_match_oracle_without_replaying_prefixes(self):
        from r31_route_cursor import advance,restore_frames
        from r31_route_geometry import cycle_options
        from collections import Counter
        for family in ((2,2,3),(2,2,2,2)):
            pool = tuple(range(8))
            join = ConnectedJoin(self.stars,pool,family,0)
            observed = []
            for size in sorted(set(family)):
                for root in cycle_options(set(pool),size,set(pool)):
                    stack = [((root,),(0,0,0))]
                    while stack:
                        advance(join,stack,{},Counter(),lambda c,o:(observed.append((c,o)) or True),lambda:False,37)
                        stack = restore_frames(json.loads(json.dumps(stack)))
            expected = set(join.survivors())
            self.assertEqual(set(observed),expected)
            self.assertEqual(len(observed),len(expected))

    def test_root_database_resumes_same_cursor_and_concurrent_interval(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        # The synthetic test explicitly selects a connected-only comparison.
        # Production rejects these uncertified centers in the preceding tests.
        family = (2,2,2,2)
        args = dict(pool=tuple(range(8)),start=0,stop=1,quantum=7)
        with tempfile.TemporaryDirectory() as temp, patch('r31_routes.proved_connected',return_value=True):
            d1 = str(Path(temp)/'direct.sqlite')
            d2 = str(Path(temp)/'shared.sqlite')
            def run_page(db,**kw):
                return search_roots(self.model.problem,self.codes,family,15,db,**args,**kw)
            direct = run_page(d1)
            with ThreadPoolExecutor(2) as workers:
                partial = list(workers.map(lambda _:run_page(d2,max_pages=1),range(2)))
            end = run_page(d2)
            self.assertEqual(direct['best'],end['best'])
            self.assertEqual(direct['work']['cursor_steps'],
                             end['work'].get('cursor_steps',0)+sum(a['work']['cursor_steps'] for a in partial))
            self.assertTrue(end['range_finished'])
            warm = run_page(d2)
            self.assertEqual(warm['work'],{'roots_reused':1})
            for pool in ((0,1,2,3,4,5,6,6),(0,1,2,3,4,5,6,400)):
                with self.assertRaises(ValueError):
                    search_roots(self.model.problem,self.codes,family,15,d1,pool=pool)

    def test_six_owner_and_rank(self):
        mapping = tuple(range(9))
        for f in ((2,4),(3,3),(2,2,2),(6,)):
            from r29_exchange_route import swap_orders
            o = next(o for o in swap_orders(6,6-len(f)) if tuple(sorted(map(len,canonical_cycles(range(6),o))))==tuple(sorted(f)))
            self.assertEqual(is_owned([(mapping,-2)],tuple(range(6)),o), f in ((2,4),(3,3)))
        self.assertEqual(sorted(rank(c,range(9)) for c in combinations(range(9),7)),list(range(36)))

    def test_coverage_transfer_never_imports_ambiguous_owner(self):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        db.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
        f = ((2,2,3),)
        register(db,'both','test',f,10,['a','b'])
        db.execute("INSERT INTO r30_pages VALUES('both',0,10,'{}','null')")
        self.assertEqual(inherited_end(db,'z','test',f,10,['a'],0,10),0)
        register(db,'single','test',f,10,['a'])
        db.execute("INSERT INTO r30_pages VALUES('single',0,8,'{}',?)",(json.dumps({'score':11}),))
        self.assertEqual(inherited_end(db,'z','test',f,11,['a'],0,10),0)
        self.assertEqual(inherited_end(db,'z','test',f,12,['a'],0,10),8)
        self.assertEqual(inherited_end(db,'z','other',f,12,['a'],0,10),0)

    def test_uncertified_planted_center_preserves_real_witness(self):
        # Alter a planted center by exactly the requested physical exchange.
        for f in ((2,2,3),(2,2,2,2)):
            n = sum(f)
            o = orders(f)[0]
            center = list(self.codes)
            center[:n] = [self.codes[o.index(i)] for i in range(n)]
            with tempfile.TemporaryDirectory() as d:
                db = str(Path(d)/'job.sqlite')
                result = run(self.model.problem,[center],db,families=[f],target=len(self.model.problem.edges),start=0,stop=1)
                self.assertTrue(result['best']['gold'])
                self.assertFalse(result['work'].get('first_connected',0))

if __name__=='__main__':
    unittest.main()
