"""Durable in-prefix continuation, process death, and concurrent disjoint search pages."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import patch

from r32_gold import run_syndrome
from r32_gold_syndrome import OutsideSyndromes
from r34_rank_leases import RankedLeases, LostRankLease
from test_r33_gold_lookup import fixture


class PartialLossTests(unittest.TestCase):
    def test_order_scope_checksum_and_retry_of_timed_out_joint(self):
        _,_,_,q=fixture()
        expected=[]
        def collect(indices,deadline):
            expected.append(tuple(sorted(indices)))
            return False
        q.indexed(9,4,start=0,stop=1,accept_errors=collect)
        self.assertGreater(len(expected),1)
        for i in range(len(expected)):
            attempted=[]
            def stop(indices,deadline):
                attempted.append(tuple(sorted(indices)))
                if len(attempted)==i+1:
                    raise TimeoutError
                return False
            old=q.indexed(9,4,start=0,stop=1,accept_errors=stop)
            self.assertEqual((old["status"],old["next_rank"]),("UNKNOWN",0))
            resumed=[]
            def finish(indices,deadline):
                resumed.append(tuple(sorted(indices)))
                return False
            out=q.indexed(9,4,start=0,stop=1,accept_errors=finish,cursor=old["cursor"])
            self.assertEqual(out["status"],"INFEASIBLE")
            self.assertEqual(attempted[:-1]+resumed,expected)
            self.assertEqual(out["work"]["saved_joint_decisions_reused"],1)
            broken=[old["cursor"][0],"corrupted"]
            resumed.clear()
            q.indexed(9,4,start=0,stop=1,accept_errors=finish,cursor=broken)
            self.assertEqual(resumed,expected)
            # A cursor for a refined query cannot skip ordinary phase solutions.
            plain=q.indexed(9,4,start=0,stop=1,cursor=old["cursor"])
            self.assertEqual(plain["work"].get("suffix_cursors_resumed",0),0)
        with patch.object(q,"joint",side_effect=TimeoutError):
            stopped=q.indexed(9,4,start=0,stop=1)
        again=q.indexed(9,4,start=0,stop=1,cursor=stopped["cursor"])
        self.assertGreater(again["work"]["joint_queries"],0)
        self.assertEqual(again["status"],"PHASE_RELAXATION_FEASIBLE")

    def test_killed_process_keeps_exact_integer_candidate_in_same_prefix(self):
        p,ref,g,q=fixture()
        answers=[]
        def second(indices,deadline):
            answers.append(tuple(sorted(indices)))
            return len(answers)==2
        expected=q.indexed(9,4,start=0,stop=1,accept_errors=second)
        source="""import os,sys
from unittest.mock import patch
from r32_gold import run_syndrome
from test_r33_gold_lookup import fixture
p,r,g,q=fixture()
hits=[]
def abrupt(indices,deadline):
    hits.append(1)
    if len(hits)==2: os._exit(93)
    return False
with patch('r33_gold_refine.make_integer_filter',return_value=abrupt):
    run_syndrome(p,r,g,9,[4],sys.argv[1],0,100,refine_integer=True)
"""
        with TemporaryDirectory() as folder:
            path=Path(folder)/"dead.sqlite"
            proc=subprocess.run([sys.executable,"-B","-c",source,str(path)],cwd=Path(__file__).parent,
                                capture_output=True,timeout=40)
            self.assertEqual(proc.returncode,93,proc.stderr)
            with closing(sqlite3.connect(path)) as db,db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM r34_rank_leases").fetchone()[0],1)
                self.assertEqual(db.execute("SELECT prefix FROM r34_loss_cursors").fetchall(),[(0,)])
                db.execute("UPDATE r34_rank_leases SET expires=?",(time.time()+0.75,))
            waits=[]
            raw_wait=RankedLeases.wait
            def wait(holder,rank,*args,**kwargs):
                waits.append(rank)
                return raw_wait(holder,rank,*args,**kwargs)
            with (patch("r33_gold_refine.make_integer_filter",return_value=lambda *_:True),
                  patch.object(RankedLeases,"wait",wait)):
                warm=run_syndrome(p,ref,g,9,[4],path,0,3,refine_integer=True)
            self.assertEqual(waits,[0])
            self.assertEqual(list(map(tuple,warm["outside_errors"])),expected["outside_errors"])
            self.assertEqual(warm["work"]["saved_joint_decisions_reused"],1)
            self.assertEqual(warm["work"].get("joint_queries",0),0)

    def test_worker_runs_other_loss_prefixes_while_first_holds_a_lease(self):
        p,ref,g,q=fixture()
        ready,finish,waiting=Event(),Event(),Event()
        raw_wait=RankedLeases.wait
        def wait(holder,*args,**kwargs):
            waiting.set()
            return raw_wait(holder,*args,**kwargs)
        calls=[]
        raw=OutsideSyndromes.indexed
        def delayed(query,*args,start,stop,**kwargs):
            calls.extend(range(start,stop))
            if start==0:
                ready.set()
                if not finish.wait(15):raise RuntimeError("lost test peer")
            return raw(query,*args,start=start,stop=stop,**kwargs)
        with TemporaryDirectory() as folder,patch.object(OutsideSyndromes,"indexed",delayed):
            path=Path(folder)/"workers.sqlite"
            def search():
                return run_syndrome(p,ref,g,6,[3],path,0,page=1)
            with ThreadPoolExecutor(max_workers=2) as pool,patch.object(RankedLeases,"wait",wait):
                first=pool.submit(search)
                try:
                    self.assertTrue(ready.wait(15))
                    with closing(sqlite3.connect(path,timeout=1)) as probe:
                        probe.execute("BEGIN IMMEDIATE")
                        probe.rollback()
                    second=pool.submit(search)
                    self.assertTrue(waiting.wait(10))
                finally:
                    finish.set()
                self.assertEqual(first.result()["status"],"INFEASIBLE")
                self.assertEqual(second.result()["status"],"INFEASIBLE")
            self.assertEqual(sorted(calls),list(range(4)))
            self.assertFalse(search()["work"])


class RankLeaseTests(unittest.TestCase):
    def test_expired_owner_cannot_publish_or_cancel_successor(self):
        with closing(sqlite3.connect(":memory:",isolation_level=None)) as db:
            manager=RankedLeases(db,"support","scope",ttl=5)
            with db,patch("r34_rank_leases.time.time",return_value=100):
                db.execute("BEGIN IMMEDIATE")
                first,_=manager.take(4,8)
                left,_=manager.take(0,10)
                self.assertEqual((left.lo,left.hi),(0,4))
                self.assertEqual(manager.take(6,12),(None,8))
            with db,patch("r34_rank_leases.time.time",return_value=107):
                db.execute("BEGIN IMMEDIATE")
                later,_=manager.take(4,8)
                with self.assertRaises(LostRankLease):
                    manager.fence(first)
                manager.release(first)
                manager.fence(later)


class RouteLeaseTests(unittest.TestCase):
    def test_overlapping_finite_pool_skips_native_traversal_and_preserves_witness(self):
        from r32_route_join import traverse
        from r32_route_pool import evaluate
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,geo,board=planted_gold(45,1,5)
        p=SearchProblem(tiles,geo)
        with TemporaryDirectory() as folder,patch("r31_route_geometry.proved_connected",return_value=True):
            path=Path(folder)/"pool.db"
            def run():
                return evaluate(p,[board],range(7),(2,2,3),0,database=path)
            visited=[]
            ready,waited,release=Event(),Event(),Event()
            raw_wait=RankedLeases.wait
            def wait(holder,*args,**kwargs):
                waited.set()
                return raw_wait(holder,*args,**kwargs)
            def blocked(*args,**kwargs):
                visited.append(1)
                ready.set()
                if not release.wait(15):raise RuntimeError("lost finite-pool test peer")
                return traverse(*args,**kwargs)
            with (patch("r32_route_join.traverse",side_effect=blocked),
                  patch.object(RankedLeases,"wait",wait),ThreadPoolExecutor(max_workers=2) as pool):
                first_task=pool.submit(run)
                try:
                    self.assertTrue(ready.wait(15))
                    with closing(sqlite3.connect(path,timeout=1)) as db:
                        db.execute('BEGIN IMMEDIATE')
                        db.rollback()
                    second=pool.submit(run)
                    self.assertTrue(waited.wait(15))
                finally:
                    release.set()
                first=first_task.result()
                self.assertTrue(second.result()["complete"])
                self.assertEqual(second.result()["best"],first["best"])
            self.assertEqual(visited,[1])
            self.assertTrue(first["complete"])
            self.assertIsNotNone(first["best"])
            self.assertEqual(run()["best"],first["best"])

    def test_streamed_page_checks_do_not_hold_writer_and_skip_active_interval(self):
        from r32_route_join import PhysicalPrices
        from r32_route_pool import stream_prefixes
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,geo,board=planted_gold(45,1,5)
        p=SearchProblem(tiles,geo)
        with TemporaryDirectory() as folder,patch("r31_route_geometry.proved_connected",return_value=True):
            path=Path(folder)/"stream.db"
            def run(limit=None):
                return stream_prefixes(p,[board],range(8),(2,2,3),14,path,stop=3,max_pages=limit)
            native=PhysicalPrices.pair_catalog
            attempts=[]
            def blocked(obj,*args):
                with closing(sqlite3.connect(path,timeout=1)) as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.rollback()
                with patch.object(PhysicalPrices,"pair_catalog",side_effect=AssertionError("duplicated catalogue")):
                    partial=run(limit=1)
                    self.assertFalse(partial["range_finished"])
                    self.assertEqual(partial["new_pages"],0)
                    self.assertEqual(partial["next_item"],0)
                attempts.append(1)
                return native(obj,*args)
            with patch.object(PhysicalPrices,"pair_catalog",blocked):
                done=run()
            self.assertEqual(attempts,[1])
            self.assertTrue(done["range_finished"])
            self.assertEqual(done["completed_items"],3)
            self.assertFalse(run()["work"])
