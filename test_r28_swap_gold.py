"""One-loop publication, legacy walk migration and concurrent writers."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from io_utils import atomic_json as write_json
from r27_incumbent import BoardModel
from r28_swap_walk import packed,walk
from test_constraint_separator import permutation_ring


def fixture():
    model=BoardModel(permutation_ring([(1,0,2),(1,0,2),(2,1,0),(2,1,0)]))
    return model,[0,3,6,9],[0,6,3,9]


class GoldWalkTests(unittest.TestCase):
    def test_gold_after_width_tied_multiloop_neighbors_and_resume(self):
        model,multiple,gold=fixture()
        self.assertEqual(model.problem.validate(multiple)['line_components'],3)
        self.assertTrue(model.problem.validate(gold)['valid'])
        with tempfile.TemporaryDirectory() as folder:
            db,out=Path(folder)/'walk.db',Path(folder)/'best.json'
            args=dict(goal=4,floor=4,width=1,limit=1)
            result=walk(model,[multiple],db,out,**args)
            self.assertTrue(result['goal_reached'])
            self.assertTrue(result['solved'])
            self.assertEqual(result['expanded_this_run'],1)
            self.assertTrue(model.problem.validate(json.loads(out.read_text())['codes'])['valid'])
            # One ordinary frontier child was admitted before encountering the
            # later Gold tie. Width one must allow the terminal exception.
            with closing(sqlite3.connect(db)) as conn:
                self.assertGreaterEqual(conn.execute('select count(*) from centers').fetchone()[0],3)
            self.assertEqual(walk(model,[],db,out,**args)['expanded_this_run'],0)

    def test_legacy_schema_checks_tied_centers_and_scope(self):
        model,multiple,gold=fixture()
        with tempfile.TemporaryDirectory() as folder:
            db,out=Path(folder)/'walk.db',Path(folder)/'best.json'
            args=dict(goal=4,floor=4,width=1,limit=1)
            binding=hashlib.sha256(json.dumps([1,model.problem.fingerprint,model.problem.domains,4,4,1],
                                             separators=(',',':')).encode()).digest()
            with closing(sqlite3.connect(db)) as conn:
                conn.execute('create table scope(id integer primary key check(id=1),binding blob not null)')
                conn.execute('insert into scope values (1,?)',(binding,))
                conn.execute('create table centers(board blob primary key,score integer,done integer,priority blob) without rowid')
                for values in (multiple,gold):
                    raw=packed(values)
                    conn.execute('insert into centers values (?,4,1,?)',(raw,hashlib.sha256(raw).digest()))
                conn.commit()
            result=walk(model,[],db,out,**args)
            self.assertTrue(result['solved'])
            self.assertEqual(result['expanded_this_run'],0)
            self.assertEqual(json.loads(out.read_text())['codes'],gold)
            with closing(sqlite3.connect(db)) as conn:
                conn.execute('delete from centers where board=?',(packed(gold),))
                conn.execute('insert or replace into perfect_checks values (?,1)',(packed(multiple),))
                conn.commit()
            with self.assertRaisesRegex(ValueError,'Gold witness'):
                walk(model,[],db,out,**args)

    def test_nonterminal_milestone_keeps_edge_semantics(self):
        model,multiple,_=fixture()
        with tempfile.TemporaryDirectory() as folder:
            db,out=Path(folder)/'walk.db',Path(folder)/'best.json'
            result=walk(model,[multiple],db,out,goal=3,floor=2,width=1,limit=1)
            self.assertTrue(result['goal_reached'])
            self.assertFalse(result['solved'])
            self.assertEqual(result['expanded_this_run'],0)

    def test_concurrent_publication_preserves_higher_winner(self):
        model,multiple,gold=fixture()
        # Use a score milestone so the multiple-loop writer publishes immediately.
        with tempfile.TemporaryDirectory() as folder:
            db,out=Path(folder)/'walk.db',Path(folder)/'best.json'
            args=dict(goal=3,floor=2,width=1,limit=1)
            entered=threading.Event()
            release=threading.Event()
            second_started=threading.Event()
            later_published=threading.Event()
            errors=[]
            def guarded_publish(path,record):
                if record['solved']:
                    write_json(path,record)
                    later_published.set()
                else:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError('publisher never released')
                    write_json(path,record)
            def worker(board,notify=None):
                try:
                    if notify:
                        notify.set()
                    walk(model,[board],db,out,**args)
                except BaseException as exc:
                    errors.append(exc)
            with patch('r28_swap_walk.atomic_json',guarded_publish):
                slow=threading.Thread(target=worker,args=(multiple,))
                fast=threading.Thread(target=worker,args=(gold,second_started))
                slow.start()
                try:
                    self.assertTrue(entered.wait(5))
                    fast.start()
                    self.assertTrue(second_started.wait(5))
                    self.assertFalse(later_published.wait(.15))
                finally:
                    release.set()
                    slow.join(10)
                    if fast.ident is not None:
                        fast.join(10)
            self.assertEqual(errors,[])
            self.assertTrue(later_published.is_set())
            self.assertTrue(json.loads(out.read_text())['solved'])


if __name__=='__main__':
    unittest.main()
