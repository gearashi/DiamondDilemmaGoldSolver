import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from r30_fresh import search
from search_fixtures import planted_gold
from search_problem import SearchProblem


class FreshSearch(unittest.TestCase):
    def test_repeat_and_distinct_seed_recovery(self):
        tiles,mesh,_ = planted_gold(31,1,4)
        problem = SearchProblem(tiles,mesh)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'independent.sqlite'
            out = Path(tmp)/'board.json'
            first = search(problem,path,starts=2,rounds=0,output=out)
            self.assertEqual(first['new_starts'],2)
            self.assertEqual(problem.validate(first['best']['codes'])['matched_edges'],first['best']['score'])
            with patch('r30_fresh._sample_start',side_effect=AssertionError('Repeated seed')):
                second = search(problem,path,starts=2,rounds=0,output=out)
            self.assertEqual(second['best'],first['best'])
            self.assertEqual(second['reused_starts'],2)
            self.assertEqual(json.loads(out.read_text())['codes'],first['best']['codes'])
            with patch('r30_fresh._sample_start',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    search(problem,path,seed=2,rounds=0)
            self.assertEqual(search(problem,path,seed=2,rounds=0)['new_starts'],1)

    def test_extension_matches_cold_and_resumes_committed_batches(self):
        import sqlite3
        from contextlib import closing
        import r30_fresh
        tiles,mesh,_=planted_gold(39,1,4)
        p=SearchProblem(tiles,mesh)
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'continued.sqlite'
            cold=search(p,Path(tmp)/'cold.sqlite',rounds=7,seed=11,batch=2)
            short=search(p,db,rounds=2,seed=11,batch=2)
            self.assertFalse(short['best']['gold'])
            real=r30_fresh.polish_step
            calls=0
            def interrupt(*args):
                nonlocal calls
                calls+=1
                if calls==4:
                    raise RuntimeError("interrupted within a batch")
                return real(*args)
            with patch('r30_fresh._sample_start',side_effect=AssertionError('Sampler redone')),patch('r30_fresh.polish_step',interrupt):
                with self.assertRaisesRegex(RuntimeError,'within a batch'):
                    search(p,db,rounds=7,seed=11,batch=2)
            resumed=search(p,db,rounds=7,seed=11,batch=2)
            self.assertEqual(resumed['best'],cold['best'])
            self.assertEqual(cold['best'],r30_fresh.one_start(r30_fresh.BoardModel(p),11,'cpu',512,64,7))
            self.assertEqual(resumed['scope'],cold['scope'])
            self.assertEqual((resumed['reused_repair_rounds'],resumed['new_repair_rounds']),(4,3))
            self.assertEqual(resumed['new_sampler_starts'],0)
            middle=search(p,db,rounds=3,seed=11,batch=2)
            independent=search(p,Path(tmp)/'three.sqlite',rounds=3,seed=11,batch=2)
            self.assertEqual(middle['best'],independent['best'])
            self.assertEqual((middle['reused_repair_rounds'],middle['new_repair_rounds']),(2,1))
            # A pre-trajectory result has only its best board; the RNG cannot be inferred.
            with closing(sqlite3.connect(db)) as conn,conn:
                conn.execute('DELETE FROM r30_trajectory')
            self.assertEqual(search(p,db,rounds=2,seed=11)['reused_starts'],1)
            legacy=search(p,db,rounds=8,seed=11,batch=2)
            self.assertEqual((legacy['new_sampler_starts'],legacy['new_repair_rounds']),(1,8))

    def test_valid_gold_ends_unlimited_run_and_survives_reopen(self):
        tiles,mesh,board = planted_gold(31,1,4)
        problem = SearchProblem(tiles,mesh,dict(enumerate(board)))
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/'fresh.sqlite'
            first = search(problem,db,starts=0)
            self.assertEqual(first['new_starts'],1)
            self.assertTrue(first['best']['gold'])
            self.assertEqual(first['best']['repair_rounds'],0)
            self.assertEqual(search(problem,db,seed=20,starts=0)['new_starts'],0)
            other=search(problem,db,seed=50,starts=0,rounds=320)
            self.assertTrue(other['best']['gold'])
            self.assertEqual(other['new_sampler_starts'],0)
