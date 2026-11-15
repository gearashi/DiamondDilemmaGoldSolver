from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bench_r35_live_roots import parallel_roots
from r31_route_cursor import advance
from r31_routes import search_roots
from search_fixtures import planted_gold
from search_problem import SearchProblem


class RootFenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,cls.board=planted_gold(42,1,5)
        cls.p=SearchProblem(tiles,geo)
        cls.family=(2,2,3)

    def run_root(self,path,**kw):
        return search_roots(self.p,self.board,self.family,0,path,pool=range(7),catalogue=False,quantum=99999,**kw)

    def test_concurrent_roots_price_outside_writer_and_warm_skip(self):
        with TemporaryDirectory() as tmp, patch('r31_routes.proved_connected',return_value=True):
            shared,baseline=[str(Path(tmp)/(n+'.db')) for n in ('s','b')]
            serial=self.run_root(baseline,start=0,stop=2)
            one,two,end=parallel_roots(self.run_root,shared)
            self.assertEqual(1,two['work']['root_lease_conflicts'])
            self.assertEqual(2,end['completed_roots'])
            self.assertEqual(serial['work']['cursor_steps'],sum(w['work'].get('cursor_steps',0) for w in (one,two,end)))
            self.assertEqual(serial['best']['score'],end['best']['score'])
            with patch('r31_routes.ConnectedJoin',side_effect=AssertionError('warm join')),patch('r31_routes.RotationOracle',side_effect=AssertionError('warm oracle')):
                self.assertEqual(self.run_root(shared,start=0,stop=2)['work'],{'roots_reused':2})

    def test_crash_after_exact_positive_reuses_witness_without_advancing_frontier(self):
        with TemporaryDirectory() as tmp,patch('r31_routes.proved_connected',return_value=True):
            path=str(Path(tmp)/'x.db')
            def crash(join,stack,frames,work,consume,expired,quantum):
                def interrupt(cells,order):
                    self.assertTrue(consume(cells,order))
                    raise InterruptedError('after exact result')
                return advance(join,stack,frames,work,interrupt,expired,quantum)
            with patch('r31_route_cursor.advance',side_effect=crash), self.assertRaises(InterruptedError):
                self.run_root(path,start=0,stop=1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r31_roots').fetchone()[0],0)
                self.assertEqual(db.execute('SELECT count(*) FROM optima').fetchone()[0],1)
            result=self.run_root(path,start=0,stop=1)
            self.assertTrue(result['range_finished']);self.assertGreater(result['work']['exact_reused'],0)
            best=result['best'];self.assertEqual(self.p.validate(best['codes'])['matched_edges'],best['score'])

    def test_expired_root_cannot_publish_its_old_cursor(self):
        with TemporaryDirectory() as tmp,patch('r31_routes.proved_connected',return_value=True):
            path=str(Path(tmp)/'x.db')
            calls=[]
            def interleave(*args):
                result=advance(*args)
                if not calls:
                    calls.append(True)
                    with closing(sqlite3.connect(path,isolation_level=None)) as db:
                        db.execute("UPDATE r34_rank_leases SET expires=0 WHERE family='root-cursor'")
                    self.assertTrue(self.run_root(path,start=0,stop=1)['range_finished'])
                return result
            with patch('r31_route_cursor.advance',side_effect=interleave):
                outer=self.run_root(path,start=0,stop=1)
            self.assertEqual(outer['work']['cursor_fences_lost'],1)
            self.assertEqual(outer['completed_roots'],1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM r31_root_cursor').fetchone()[0],0)


if __name__=='__main__':unittest.main()
