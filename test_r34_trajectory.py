"""Round 34 scoped continuation and compact signed-price regressions."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from r31_sampling import run_seeds
from r34_signed_snapshot import restore,snapshot


class PhaseContinuation(unittest.TestCase):
    def test_non_byte_aligned_price_vectors_and_scope(self):
        from types import SimpleNamespace
        def make(scope):
            return SimpleNamespace(scope=scope,dart_phases=np.zeros(15,np.int8),prices={})
        live=make('phase-a')
        live.prices={i:(np.arange(15)%3==i%3) for i in (1,4,14)}
        state=snapshot(live)
        copied=make('phase-a')
        restore(copied,state)
        self.assertEqual(set(live.prices),set(copied.prices))
        for i,row in live.prices.items():
            np.testing.assert_array_equal(row,copied.prices[i])
        with self.assertRaisesRegex(ValueError,'validation'):
            restore(make('phase-b'),state)
        with self.assertRaisesRegex(ValueError,'validation'):
            restore(make('phase-a'),dict(state,checksum='bad'))
        empty=make('phase-a')
        restore(empty,snapshot(empty))
        self.assertFalse(empty.prices)

    def test_unbound_old_seed_finishes_only_its_own_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs.sqlite'
            def interrupted(seed,state,save):
                save(dict(cursor=4))
                raise RuntimeError('power')
            with self.assertRaisesRegex(RuntimeError,'power'):
                run_seeds(path,'horizon-8',0,1,interrupted,checkpointed=True)
            result=dict(gold=False,score=12,components=2,codes=[])
            def finish(seed,state,save):
                self.assertEqual(state,dict(cursor=4))
                save(dict(cursor=8))
                return result
            run_seeds(path,'horizon-8',0,1,finish,checkpointed=True,trajectory=('stream',8))
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM r34_trajectories').fetchone()[0],0)
            def first(seed,state,save):
                self.assertIsNone(state)
                save(dict(cursor=12))
                return result
            run_seeds(path,'horizon-12',0,1,first,checkpointed=True,trajectory=('stream',12))
            with closing(sqlite3.connect(path)) as conn,conn:
                conn.execute("UPDATE r34_trajectories SET checksum='bad'")
            with self.assertRaisesRegex(ValueError,'checksum'):
                run_seeds(path,'horizon-16',0,1,first,checkpointed=True,trajectory=('stream',16))

    def test_two_horizons_compute_each_round_once_in_either_order(self):
        for first,second in ((16,64),(64,16)):
            with self.subTest(first=first),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'parallel.sqlite'
                entered=threading.Event()
                proceed=threading.Event()
                rounds=[]
                def run(horizon):
                    def worker(seed,state,save):
                        if horizon==first:
                            entered.set()
                            self.assertTrue(proceed.wait(5))
                        cursor=state['cursor'] if state else 0
                        for index in range(cursor+1,horizon+1):
                            rounds.append(index)
                            if index%8==0:save(dict(cursor=index))
                        return dict(gold=False,score=horizon,components=2,codes=[])
                    return run_seeds(path,'horizon-'+str(horizon),0,1,worker,checkpointed=True,
                                     trajectory=('shared',horizon))
                with ThreadPoolExecutor(2) as pool:
                    a=pool.submit(run,first)
                    self.assertTrue(entered.wait(5))
                    b=pool.submit(run,second)
                    # Both horizon jobs own a row before the first producer finishes.
                    import time
                    until=time.monotonic()+5
                    with closing(sqlite3.connect(path)) as conn:
                        while not conn.execute('SELECT 1 FROM r31_jobs WHERE scope=?',
                                               ('horizon-'+str(second),)).fetchone():
                            if time.monotonic()>until:self.fail('second producer never arrived')
                            time.sleep(.01)
                    proceed.set()
                    self.assertEqual([a.result()['best']['score'],b.result()['best']['score']],[first,second])
                self.assertEqual(rounds,list(range(1,65)))

    def test_shared_phase_rows_support_legacy_and_detect_corrupt_or_missing_data(self):
        from r34_signed_snapshot import archive,expand
        from types import SimpleNamespace
        signed=SimpleNamespace(scope='a',dart_phases=np.zeros(15,np.int8),prices={3:np.arange(15)%2==0})
        with closing(sqlite3.connect(':memory:')) as db,db:
            a=dict(cursor=8,dart_prices=snapshot(signed))
            b=dict(cursor=16,dart_prices=snapshot(signed))
            saved_a=archive(db,a)
            saved_b=archive(db,b)
            self.assertEqual(saved_a['dart_price_blob'],saved_b['dart_price_blob'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM r34_price_blobs').fetchone()[0],1)
            self.assertEqual(expand(db,saved_a.copy()),a)
            self.assertEqual(expand(db,a.copy()),a)
            self.assertEqual(expand(db,saved_b.copy()),b)
            signed.prices[4]=np.arange(15)%3==0
            c=archive(db,dict(cursor=24,dart_prices=snapshot(signed)))
            self.assertNotEqual(c['dart_price_blob'],saved_a['dart_price_blob'])
            db.execute("UPDATE r34_price_blobs SET payload='broken'")
            with self.assertRaisesRegex(ValueError,'damaged'):
                expand(db,saved_a.copy())
            db.execute('DELETE FROM r34_price_blobs')
            with self.assertRaisesRegex(ValueError,'Missing'):
                expand(db,c.copy())
