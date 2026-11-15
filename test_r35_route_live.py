"""Exercise arriving root and ranked evidence over two independent database connections."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from r31_route_coverage import register
from r32_route_store import RootCoverage,publish_page,publish_scores
from r35_route_live import RankedCoverage,ReceiptEpoch
from r27_structure_fixtures import planted
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_clusters import StarClusters
from r31_route_geometry import eligible_orders,parent
from r30_cycles import canonical_cycles
from r32_route_store import colex


class LiveRouteTests(unittest.TestCase):
    def test_late_singleton_multicenter_family_threshold_and_root_registration(self):
        p,_=planted(1,3,71);model=BoardModel(p)
        near=StarClusters(ExchangeBound(model,[3*t for t in range(p.n)])).close
        family=(2,2,3);other=(3,4);free={i:i for i in range(10)}
        cells=tuple(range(7));point=colex(cells,free)
        with TemporaryDirectory() as tmp, closing(sqlite3.connect(Path(tmp)/'x.db',isolation_level=None)) as a, closing(sqlite3.connect(Path(tmp)/'x.db',isolation_level=None)) as b:
            a.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
            a.execute('CREATE TABLE r31_roots(scope,idx,best)')
            register(a,'ranked','d',(family,),208,('owner',))
            epoch=ReceiptEpoch(a);ranked=RankedCoverage(a,epoch,'d','owner',family,208,'root','ranked',free)
            rc=RootCoverage(a,'d','owner',family,208,epoch=epoch)
            order=next(iter(eligible_orders(near,cells,family)))
            self.assertFalse(rc.covered(cells,order,near));self.assertFalse(ranked.covered(cells))
            register(b,'peer','d',(family,other),207,('owner','other'))
            b.execute("INSERT INTO r30_pages VALUES('peer',0,1,'{}',?)",(json.dumps({'score':211}),))
            publish_page(b,'peer',0,1,('owner','other'),{'owner':[('other',8)]})
            publish_scores(b,'peer',0,('owner','other'),(family,other),{(1,other):211})
            epoch.refresh(True)
            self.assertFalse(ranked.covered(cells)) # unresolved dependency
            b.execute("INSERT INTO r30_pages VALUES('peer',8,14,'{}','null')")
            publish_page(b,'peer',8,14,('owner','other'),{})
            epoch.refresh(True);self.assertTrue(ranked.covered(cells))
            b.execute('INSERT INTO r32_root_binding VALUES(?,?,?,?,?,?)',('root-peer','d','owner',json.dumps(family),207,json.dumps(list(free))))
            epoch.refresh(True);self.assertFalse(rc.covered(cells,order,near))
            group=tuple(sorted(canonical_cycles(cells,order)))
            while len(group)>1:group=parent(group,near)
            from itertools import permutations
            from math import comb,factorial
            cycle,=group;n=len(cycle)
            idx=sum(comb(10,k)*factorial(k-1) for k in set(family) if k<n)+colex(cycle,free)*factorial(n-1)+list(permutations(sorted(cycle)[1:])).index(cycle[1:])
            b.execute('INSERT INTO r31_roots VALUES(?,?,?)',('root-peer',idx,json.dumps({'score':207})))
            epoch.refresh(True);self.assertTrue(rc.covered(cells,order,near))
            stricter=RootCoverage(a,'d','owner',family,207,epoch=epoch)
            self.assertFalse(stricter.covered(cells,order,near))
            # An otherwise valid source cannot claim a support with an outside donor.
            b.execute('UPDATE r32_root_binding SET pool=? WHERE scope=?',(json.dumps(list(range(6))),'root-peer'))
            epoch.refresh(True);self.assertFalse(rc.covered(cells,order,near))

    def test_coarse_positive_on_other_support_requires_local_resolution(self):
        from unittest.mock import patch
        from r32_route_pool import evaluate
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        tiles,geo,board=planted_gold(42,1,5);p=SearchProblem(tiles,geo)
        family=(2,2,3)
        with TemporaryDirectory() as tmp,patch('r31_route_geometry.proved_connected',return_value=True):
            path=Path(tmp)/'pool.db'
            direct=evaluate(p,[board],range(7),family,0,database=path)
            with closing(sqlite3.connect(path,isolation_level=None)) as db:
                key=db.execute('SELECT scope FROM r30_pages').fetchone()[0]
                db.execute('DELETE FROM r30_pages')
                # A valid aggregate witness in rank one is not the answer to rank zero.
                isolated=evaluate(p,[board],(0,1,2,3,4,5,7),family,0,database=Path(tmp)/'other.db')
                coarse=isolated['best']
                db.execute('INSERT INTO r30_pages VALUES(?,?,?,?,?)',(key,0,2,'{}',json.dumps(coarse)))
            refined=evaluate(p,[board],range(7),family,0,database=path)
            self.assertEqual(refined['best'],direct['best'])
            self.assertNotEqual(refined['best']['codes'],coarse['codes'])
            with patch('r32_route_join.traverse',side_effect=AssertionError('repeated local search')):
                self.assertEqual(evaluate(p,[board],range(7),family,0,database=path)['best'],direct['best'])
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r30_pages').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT rank FROM r35_pool_results').fetchone()[0],0)

if __name__=='__main__':unittest.main()
