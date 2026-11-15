from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from r34_rank_leases import LostRankLease,RankedLeases
from r34_cycle_orders import CycleOrders
from r35_order_frontier import OrderFrontier


class OrderFrontierTests(unittest.TestCase):
    def test_seek_profile_partial_witness_and_expired_owner(self):
        with closing(sqlite3.connect(':memory:',isolation_level=None)) as db:
            db.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
            holder=RankedLeases(db,'support','s')
            with db:
                db.execute('BEGIN IMMEDIATE');claim,_=holder.take(0,4)
            f=(2,2,2,3);orders=CycleOrders(f)
            a=OrderFrontier(db,'s',40);full=a.catalogue(f,orders,orders)
            subset=list(orders)[::2]
            self.assertNotEqual(full,a.catalogue(f,subset,orders))
            self.assertNotEqual(a.catalogue(f,subset,orders),a.catalogue(f,subset[::-1],orders))
            self.assertIsNone(a.catalogue(f,subset[:20],orders))
            best={'score':17}
            a.save(2,'a',full,40,Counter(first_owned=40),{(0,1,(3,4))},best,holder,claim)
            self.assertEqual(a.load(2,'a',full,len(orders))[3],best)
            self.assertEqual(a.best(),best)
            self.assertEqual(a.load(2,'b',full,len(orders))[0],0)
            self.assertEqual(a.load(1,'a',full,len(orders))[0],0)
            db.execute("UPDATE r35_order_cursor SET done=45")
            self.assertEqual(a.load(2,'a',full,len(orders))[0],0)
            self.assertIsNone(a.best())
            db.execute("UPDATE r34_rank_leases SET expires=0")
            with self.assertRaises(LostRankLease):a.save(2,'a',full,80,Counter(first_owned=80),set(),best,holder,claim)
            self.assertEqual(db.execute('SELECT done FROM r35_order_cursor').fetchone()[0],45)
            a.promoted(0,4);self.assertIsNone(a.best())

    def test_public_restart_restores_checked_maximum_and_typed_links(self):
        from bench_r35_order_frontier import compare,planted_case
        p,boards,family=planted_case()
        with TemporaryDirectory() as tmp:
            result=compare(p,boards,family,len(p.edges),0,tmp,3301)
        self.assertEqual(result['128']['witness_score'],len(p.edges))
        self.assertGreater(result['128']['typed_dependency_rows'],0)
        self.assertLess(result['128']['total_checks'],result['0']['total_checks'])


    def test_hard_exit_preserves_only_fenced_prefix_and_partial_positive(self):
        import os,subprocess,sys
        from unittest.mock import patch
        from bench_r35_order_frontier import planted_case
        from r30_multicenter import run
        from r35_order_frontier import OrderSliceExpired
        script="""from bench_r35_order_frontier import planted_case
from r30_multicenter import run
from r35_order_frontier import OrderFrontier
from unittest.mock import patch
import os,sys
p,boards,f=planted_case()
save=OrderFrontier.save
def crash(self,*a):
    save(self,*a)
    os._exit(17)
with patch.object(OrderFrontier,'save',crash):
    run(p,boards,sys.argv[1],families=(f,),target=len(p.edges),start=0,stop=1,page=1,filters=(),order_quantum=64)
"""
        p,boards,f=planted_case()
        with TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'hard.db')
            child=subprocess.run([sys.executable,'-B','-c',script,path],cwd=Path(__file__).parent,timeout=45)
            self.assertEqual(child.returncode,17)
            with closing(sqlite3.connect(path,isolation_level=None)) as db:
                self.assertEqual(db.execute('SELECT done FROM r35_order_cursor').fetchone()[0],64)
                self.assertEqual(db.execute('SELECT count(*) FROM r30_pages').fetchone()[0],0)
                db.execute('UPDATE r34_rank_leases SET expires=0')
            opts=dict(families=(f,),target=len(p.edges),start=0,stop=1,page=1,filters=(),order_quantum=128)
            save=OrderFrontier.save
            def limit(self,*a):
                save(self,*a)
                if a[6] is not None:raise OrderSliceExpired()
            with patch.object(OrderFrontier,'save',limit):
                partial=run(p,boards,path,**opts)
            self.assertFalse(partial['range_finished'])
            self.assertIsNotNone(partial['best'])
            self.assertEqual(p.validate(partial['best']['codes'])['matched_edges'],len(p.edges))
            complete=run(p,boards,path,**opts)
            self.assertTrue(complete['range_finished'])
            self.assertEqual(complete['best']['score'],len(p.edges))

if __name__=='__main__':unittest.main()
