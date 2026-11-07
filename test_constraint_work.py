import itertools,json,random,tempfile,unittest
from pathlib import Path
from constraint_work import *


def leaves(rows):
    return set().union(*(set(itertools.product(*[[i for i in range(v.bit_length()) if v>>i&1] for v in row])) for row in rows)) if rows else set()


class WorkTests(unittest.TestCase):
    def test_exact_disjoint_subtraction(self):
        rng=random.Random(2402)
        for _ in range(250):
            a=tuple(rng.randrange(1,16) for _ in range(3));b=tuple(rng.randrange(1,16) for _ in range(3))
            parts=subtract(a,b)
            self.assertTrue(disjoint(parts));self.assertEqual(leaves(parts),leaves([a])-leaves([b]))
            self.assertEqual(leaves(coalesce(parts)),leaves(parts))
    def test_indexed_disjointness_matches_pairwise_enumeration(self):
        rng=random.Random(240203)
        for dimensions in (1,2,3,8):
            for _ in range(120):
                rows=[tuple(rng.randrange(1,32) for _ in range(dimensions)) for _ in range(rng.randrange(18))]
                expected=all(not all(a&b for a,b in zip(left,right)) for i,left in enumerate(rows) for right in rows[:i])
                self.assertEqual(disjoint(rows),expected)
        with self.assertRaises(ValueError):disjoint([(1,2),(1,)])
        with self.assertRaises(ValueError):disjoint([(0,2)])

    def test_large_prefix_frontier_avoids_pairwise_work_and_cancellation_propagates(self):
        rows=[tuple(1<<((number>>axis)&1) for axis in range(12))+(7,)*148 for number in range(4096)]
        ticks=[0]
        def count():ticks[0]+=1
        self.assertTrue(disjoint(rows,check=count))
        self.assertLess(ticks[0],len(rows)*30)
        self.assertFalse(disjoint(rows+[rows[-1]]))
        def stop():raise InterruptedError('test stop')
        with self.assertRaises(InterruptedError):disjoint(rows,check=stop)

    def test_budget_returns_original_coverage(self):
        a=((15,15,15),);parts,done=subtract_many(a,[(1,1,1)],max_boxes=1)
        self.assertFalse(done);self.assertEqual(parts,a)
    def test_checkpoint_pending_union(self):
        cp={'stack':[{'domains':['0x7','0x7'],'cell':0,'choices':[0,1,2],'next':1},
                     {'domains':['0x1','0x7'],'cell':1,'choices':[0,1,2],'next':2},
                     {'domains':['0x1','0x2'],'cell':-1,'choices':[],'next':0}]}
        rows=checkpoint_boxes(cp)
        self.assertTrue(disjoint(rows))
        self.assertEqual(leaves(rows),{(a,b) for a in range(3) for b in range(3)}-{(0,0)})
    def test_frontier_binding_and_corruption(self):
        scope=WorkScope('a','gold',(7,7),(7,7));rows=((1,7),(6,7))
        saved=json.loads(json.dumps(frontier_snapshot(scope,rows)))
        self.assertEqual(restore_frontier(saved,scope),rows)
        with self.assertRaises(ValueError):restore_frontier(saved,WorkScope('a','edges',(7,7),(7,7)))
        bad=json.loads(json.dumps(saved));bad['boxes'][0][0]='0x7'
        with self.assertRaises(ValueError):restore_frontier(bad,scope)
        with self.assertRaises(ValueError):frontier_snapshot(scope,((3,7),(6,7)))
    def test_coverage_restart_and_restricted_roots(self):
        wide=WorkScope('a','gold',(7,7),(7,7));narrow=WorkScope('a','gold',(7,7),(1,7))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'closed.sqlite3'
            with CoverageLedger(path,wide) as db:
                for status,complete in [('timeout',False),('infeasible',False),('solved',True)]:
                    with self.assertRaises(ValueError):db.record_completed((1,7),status=status,complete=complete)
                db.record_completed((1,3),status='infeasible',complete=True,source_scope=narrow)
            with CoverageLedger(path,wide) as db:
                rows,done=db.remaining((wide.root,))
                self.assertTrue(done);self.assertEqual(leaves(rows),leaves([wide.root])-leaves([(1,3)]))
                self.assertEqual(db.exclusions(narrow),((1,3),))
                with self.assertRaises(ValueError):db.exclusions(WorkScope('a','edges',(7,7),(7,7)))
    def test_local_ledger_detects_corruption(self):
        scope=WorkScope('a','gold',(7,7),(7,7))
        with tempfile.TemporaryDirectory() as directory:
            with CoverageLedger(Path(directory)/'closed.sqlite3',scope) as db:
                db.record_completed((1,3),status='infeasible',complete=True)
                db.db.execute("UPDATE closed SET box='[\"0x7\",\"0x7\"]'");db.db.commit()
                with self.assertRaises(ValueError):db.exclusions()

if __name__=='__main__':unittest.main()
