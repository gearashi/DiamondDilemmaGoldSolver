import json
import sqlite3
import unittest
from contextlib import closing
from itertools import combinations

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r30_cycle_joins import TypedJoin
from r30_multicenter import orders
from r31_route_coverage import register,inherited_end
from r31_route_geometry import ConnectedJoin,eligible_orders,parent
from r31_routes import rank
from r32_catalogue import install
from r32_exterior import CompactExterior
from r32_route_store import RootCoverage,publish_page,schema
from r30_cycles import canonical_cycles
from search_fixtures import planted_gold
from search_problem import SearchProblem


class StoredRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,board,cls.codes=planted_gold(71,1,5)
        cls.model=BoardModel(SearchProblem(tiles,board))
        cls.bound=ExchangeBound(cls.model,cls.codes)
        cls.stars=StarClusters(cls.bound)

    def test_dependency_order_threshold_and_old_pages(self):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
            f=((2,2,3),)
            register(db,'shared','d',f,208,('a','b','c'))
            def covered(owner,target=208):
                return inherited_end(db,'new','d',f,target,(owner,),10,12)
            def page(lo,hi,deps,score=None):
                db.execute('INSERT INTO r30_pages VALUES(?,?,?,?,?)',('shared',lo,hi,'{}',json.dumps(score)))
                publish_page(db,'shared',lo,hi,('a','b','c'),deps)
                db.commit()
            page(10,12,{'c':{('b',20)}})
            self.assertEqual(covered('a'),12)
            self.assertEqual(covered('c'),10)
            page(20,21,{'b':{('a',30)}})
            self.assertEqual(covered('c'),10)
            page(30,31,{},dict(score=208))
            self.assertEqual(covered('c'),10)
            self.assertEqual(covered('c',209),12)
            self.assertEqual(covered('c',207),10)

    def test_root_order_mask_exact_on_every_order(self):
        pool=tuple(range(9))
        for family in ((2,2,3),(2,2,2,2)):
            with closing(sqlite3.connect(':memory:')) as db:
                schema(db)
                db.execute('CREATE TABLE r31_roots(scope,idx,best)')
                db.execute('INSERT INTO r32_root_binding VALUES(?,?,?,?,?,?)',
                           ('root','d','o',json.dumps(family),208,json.dumps(pool)))
                roots=[]
                from itertools import permutations
                for k in sorted(set(family)):
                    for cells in sorted(combinations(pool,k),key=lambda s:rank(s,pool)):
                        for tail in permutations(cells[1:]):
                            roots.append((cells[0],*tail))
                done=set(roots[::3])
                db.executemany('INSERT INTO r31_roots VALUES(?,?,?)',
                               (('root',i,'null') for i,c in enumerate(roots) if c in done))
                rc=RootCoverage(db,'d','o',family,208)
                for support in combinations(pool,sum(family)):
                    for order in eligible_orders(self.stars.close,support,family):
                        g=tuple(sorted(canonical_cycles(support,order)))
                        while len(g)>1:
                            g=parent(g,self.stars.close)
                        self.assertEqual(rc.covered(support,order,self.stars.close),g[0] in done)
                self.assertFalse(RootCoverage(db,'d','o',family,207).scopes)

    def test_catalogue_replay_and_center_specific_prices(self):
        with closing(sqlite3.connect(':memory:')) as db:
            def joined(tag):
                j=ConnectedJoin(self.stars,range(9),(2,2,3),12)
                install(j,db,'binding',tag)
                return j
            a,b,c=joined('a'),joined('b'),joined('a')
            for size in (2,3):
                ref=a.buckets(size,1)
                for j in (b,c):
                    vals,rows=j.buckets(size,1)
                    self.assertEqual(ref[0],vals)
                    self.assertEqual(ref[1].tolist(),rows.tolist())
            self.assertGreater(b.work['catalogue_cells_reused'],0)
            self.assertEqual(c.work['catalogue_rows'],0)
            self.assertGreater(c.work['catalogue_prices_reused'],0)

    def test_exterior_and_response_joins_against_cold(self):
        oracle=RotationOracle(self.model)
        for fam in ((2,2,3),(2,2,2,2)):
            cells=tuple(range(sum(fam)))
            ext=CompactExterior(self.bound,cells)
            for o in orders(fam)[::13]:
                self.assertEqual(ext.score(o),oracle.optimum(self.bound.witness_seed(cells,o))[0])
            for target in (0,12):
                a=TypedJoin(self.bound,range(8),fam,target,contact=True)
                b=TypedJoin(self.bound,range(8),fam,target,contact=True)
                self.assertEqual(sorted(a.half_survivors()),sorted(b.half_survivors(True)))
