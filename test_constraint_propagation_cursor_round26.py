import copy,json,random,tempfile,unittest
from constraint_propagation_cursor_round26 import BasicCursor,PropagationCursor
from constraint_cpu_archive_round26 import CPUReasoningArchive
from constraint_dfs import _build_support_tables,_ArcFilter
from test_constraint_sat_domains import fixture
from test_constraint_dfs import brute

def basic_oracle(domains,tables):
    domains=list(domains);f=_ArcFilter(tables)
    while all(domains):
        before=domains.copy()
        for a,sa,b,sb in tables.arcs:domains[a]=f(domains[a],domains[b],sa,sb)
        if not all(domains):return False,domains
        owners={};hosts=[[] for _ in domains]
        for cell,mask in enumerate(domains):
            tiles={c//3 for c in range(mask.bit_length()) if mask>>c&1}
            if len(tiles)==1:
                tile=next(iter(tiles))
                if tile in owners:return False,domains
                owners[tile]=cell
            for tile in tiles:hosts[tile].append(cell)
        for tile,cell in owners.items():
            for other in hosts[tile]:
                if other!=cell:domains[other]&=~tables.tile_bits[tile]
        for tile,cells in enumerate(hosts):
            if not cells:return False,domains
            if len(cells)==1:domains[cells[0]]&=tables.tile_bits[tile]
        if domains==before:return True,domains
    return False,domains

class PropagationCursorTests(unittest.TestCase):
    def test_every_microstep_json_restart_keeps_exact_work_and_outcome(self):
        problem,_=fixture();root=[sum(1<<c for c in row) for row in problem.domains]
        tables=_build_support_tables(problem,root,'gold',lambda:None);rng=random.Random(2602);solutions=brute(problem,'edges')
        for _ in range(45):
            row=rng.choice(solutions);domains=[(1<<row[c])|rng.randrange(1<<12) for c in range(4)]
            expected,_=basic_oracle(domains,tables);cold=BasicCursor(domains,tables.arcs,tables.incoming,tables.tile_bits);coldstats={}
            self.assertEqual(cold.advance(_ArcFilter(tables),stats=coldstats),expected)
            actual=BasicCursor(domains,tables.arcs,tables.incoming,tables.tile_bits);stats={};total=0
            while not actual.done:
                calls=0
                def check():
                    nonlocal calls
                    calls+=1
                    if calls==2:raise InterruptedError
                try:actual.advance(_ArcFilter(tables),check,stats=stats)
                except InterruptedError:pass
                actual=BasicCursor.restore(json.loads(json.dumps(actual.snapshot())),tables.arcs,tables.incoming,tables.tile_bits);total+=1
                self.assertLess(total,400)
            self.assertEqual(actual.domains,cold.domains);self.assertEqual(stats,coldstats)
            self.assertEqual([r for r in solutions if all(m>>v&1 for m,v in zip(domains,r))],[r for r in solutions if all(m>>v&1 for m,v in zip(actual.domains,r))])
    def test_interrupted_arc_is_not_consumed_and_strong_stages_commit(self):
        arcs=((0,0,1,0),);incoming=((),(0,));tiles=(7,56)
        cursor=BasicCursor([7,56],arcs,incoming,tiles)
        def fail(*args):raise InterruptedError
        with self.assertRaises(InterruptedError):cursor.advance(fail)
        self.assertEqual(list(cursor.queue),[0]);self.assertTrue(cursor.advance(lambda a,b,s,t:a))
        domains=[7,56];stagecalls=[];cursor=PropagationCursor(domains,(),((),()),tiles,('first','second','third')).attach(domains)
        def first(ds):stagecalls.append('first')
        def second(ds):stagecalls.append('second');ds[0]&=3;raise InterruptedError
        def third(ds):stagecalls.append('third')
        with self.assertRaises(InterruptedError):cursor.advance(lambda *a:a[0],[('first',first),('second',second),('third',third)])
        self.assertEqual(cursor.stage,1);self.assertEqual(domains,[3,56]);self.assertEqual(stagecalls,['first','second'])
        def finish(ds):stagecalls.append('finish')
        restored=PropagationCursor.restore(json.loads(json.dumps(cursor.snapshot())),(),((),()),tiles,('first','second','third')).attach(domains)
        self.assertTrue(restored.advance(lambda *a:a[0],[('first',first),('second',finish),('third',third)]))
        self.assertEqual(stagecalls,['first','second','finish','third','first','finish','third'])
        self.assertEqual(restored.stats['propagation_cursor_opaque_retries'],1)
    def test_archive_exact_scope_attachment_policy_and_invalid_payload(self):
        names=('optional',);cursor=PropagationCursor([7,56],(),((),()),(7,56),names)
        def stop():raise InterruptedError
        with self.assertRaises(InterruptedError):cursor.advance(lambda *a:a[0],[('optional',lambda ds:None)],stop)
        with tempfile.TemporaryDirectory() as directory:
            archive=CPUReasoningArchive(directory,'p','gold',(63,63));self.assertTrue(archive.save_propagation(('frame',),cursor))
            loaded=archive.load_propagation(('frame',),(),((),()),(7,56),names);self.assertIsNotNone(loaded)
            with self.assertRaises(ValueError):loaded.attach([3,56])
            self.assertIsNone(archive.load_propagation(('frame',),(),((),()),(7,56),('changed',)))
            self.assertIsNone(CPUReasoningArchive(directory,'p','gold',(7,56)).load_propagation(('frame',),(),((),()),(7,56),names))
        bad=json.loads(json.dumps(cursor.snapshot()));bad['basic']['queue']=[True]
        with self.assertRaises(ValueError):PropagationCursor.restore(bad,(),((),()),(7,56),names)
    def test_native_partial_call_rebuilds_only_opaque_queue(self):
        arcs=((0,0,1,0),(1,0,0,0));incoming=((1,),(0,));cursor=PropagationCursor([7,56],arcs,incoming,(7,56),())
        called=[]
        def first(ds,queue):called.append(queue);ds[0]=3;raise InterruptedError
        with self.assertRaises(InterruptedError):cursor.advance(lambda *a:a[0],[],native_basic=first)
        cursor=PropagationCursor.restore(json.loads(json.dumps(cursor.snapshot())),arcs,incoming,(7,56),())
        def finish(ds,queue):called.append(queue);self.assertEqual(ds,[3,56]);return True
        stats={};self.assertTrue(cursor.advance(lambda *a:a[0],[],stats=stats,native_basic=finish))
        self.assertEqual(called,[(0,1),(0,1)]);self.assertEqual(stats['propagation_cursor_opaque_retries'],1)

    def test_large_inventory_and_false_strong_return(self):
        n=160;domains=[1<<(3*c) for c in range(n)];cursor=BasicCursor(domains,(),tuple(() for _ in domains),tuple(7<<(3*c) for c in range(n)))
        self.assertTrue(cursor.advance(lambda *a:a[0]));self.assertEqual(cursor.domains,domains)
        strong=PropagationCursor([7,56],(),((),()),(7,56),('reject',))
        self.assertFalse(strong.advance(lambda *a:a[0],[('reject',lambda ds:False)]))
if __name__=='__main__':unittest.main()
