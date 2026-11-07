import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from constraint_reasoning_store_round26 import LocalReasoningStore,digest
from constraint_cpu_archive_round26 import CPUReasoningArchive,dump_state,restore_state
from constraint_lcg import LCGState,LCGStore,LearnedClause
from constraint_lookahead import CompletedProbeCache

class DurableReasoningTests(unittest.TestCase):
    def test_atomic_unchanged_selective_source_scope_and_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.py';b=Path(directory)/'b.py';a.write_text('a');b.write_text('b')
            store=LocalReasoningStore(directory,{'root':[3,3]},max_entries=2)
            self.assertTrue(store.save('a',1,{'done':5},sources=[a]));self.assertTrue(store.save('a',1,{'done':5},sources=[a]))
            self.assertEqual(store.stats['reasoning_cache_writes'],1);self.assertEqual(store.stats['reasoning_cache_unchanged'],1)
            store.save('b',1,{'done':6},sources=[b]);a.write_text('new')
            self.assertIsNone(store.load('a',1,sources=[a]));self.assertEqual(store.load('b',1,sources=[b]),{'done':6})
            self.assertIsNone(LocalReasoningStore(directory,{'root':[1,3]}).load('b',1,sources=[b]))
            store.save('c',1,{},sources=[b]);self.assertLessEqual(len(list(store.directory.glob('*.json'))),2)
            self.assertGreater(store.stats['reasoning_cache_evictions'],0)
            old=store.load('b',1,sources=[b])
            with patch('constraint_reasoning_store_round26.os.replace',side_effect=OSError('denied')):
                self.assertFalse(store.save('b',1,{'new':7},sources=[b]))
            self.assertEqual(store.load('b',1,sources=[b]),old);self.assertFalse(list(store.directory.glob('*.tmp')))

    def test_global_context_eviction_stale_temp_and_unknown_migration(self):
        import os,time
        with tempfile.TemporaryDirectory() as directory:
            source=Path(__file__);stores=[]
            for context in range(5):
                store=LocalReasoningStore(directory,{'context':context},global_max_entries=2);stores.append(store)
                self.assertTrue(store.save('x',0,{'value':context},sources=[source]))
            files=list((Path(directory)/'local-reasoning-v1').glob('*/*.json'));self.assertEqual(len(files),2)
            self.assertEqual(stores[-1].load('x',0,sources=[source]),{'value':4})
            stale=stores[-1].directory/'.reasoning-dead.tmp';stale.write_text('incomplete');os.utime(stale,(time.time()-90000,)*2)
            fresh=stores[-1].directory/'.reasoning-live.tmp';fresh.write_text('current')
            outside=Path(directory)/'unrelated.json';outside.write_text('must stay')
            stores[-1].prune_global();self.assertFalse(stale.exists());self.assertTrue(fresh.exists());self.assertTrue(outside.exists())
            self.assertEqual(stores[-1].stats['reasoning_cache_stale_temporaries'],1)

    def test_stop_exceptions_propagate_before_commit_and_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(__file__);store=LocalReasoningStore(directory,{'root':1});store.save('x',1,{'old':1},sources=[source])
            for error in (InterruptedError,ValueError,OSError,RuntimeError):
                for boundary in (1,2,3):
                    calls=0
                    def stop():
                        nonlocal calls
                        calls+=1
                        if calls==boundary:raise error('stop')
                    with self.assertRaises(error):store.save('x',1,{'new':2},sources=[source],check=stop)
                    self.assertEqual(store.load('x',1,sources=[source]),{'old':1})
                for boundary in (1,2,3):
                    calls=0
                    def stop_read():
                        nonlocal calls
                        calls+=1
                        if calls==boundary:raise error('stop')
                    with self.assertRaises(error):store.load('x',1,sources=[source],check=stop_read)

    def test_corruption_versions_size_and_cap_cold_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(__file__);store=LocalReasoningStore(directory,{'root':1},max_bytes=10000)
            self.assertFalse(store.save('x',0,'x'*10001,sources=[source]))
            store.save('x',0,{},sources=[source]);path=next(store.directory.glob('*.json'))
            data=json.loads(path.read_bytes());data['version']=2;data['digest']=digest({k:v for k,v in data.items() if k!='digest'});path.write_text(json.dumps(data))
            self.assertIsNone(store.load('x',0,sources=[source]));path.write_bytes(b'x'*10001)
            self.assertIsNone(store.load('x',0,sources=[source]))
            path.write_text('{');self.assertIsNone(store.load('x',0,sources=[source]))

    def test_lcg_reason_graph_and_store_survive_restart_and_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=CPUReasoningArchive(directory,'fixture','gold',(3,3,3,3))
            root=LCGState((3,3,3,3),'fixture');child=root.fork(0,1)
            child.record_cell(1,1,[0],'implication');child.record((1,1,1,1),[1],'shared')
            child.failure=[2,3];learned=child.first_uip()
            self.assertTrue(archive.save_reason(('child',1),child));self.assertTrue(archive.save_reason(('root',),root))
            loaded=CPUReasoningArchive(directory,'fixture','gold',(3,3,3,3)).load_reason(('child',1))
            self.assertEqual(loaded.first_uip(),learned);self.assertEqual(json.loads(json.dumps(dump_state(loaded))),json.loads(json.dumps(dump_state(child))))
            loaded_root=archive.load_reason(('root',));self.assertEqual(len(loaded_root.facts),4)
            sibling=loaded_root.fork(0,2);self.assertEqual(sibling.domains,[2,3,3,3]);self.assertEqual(loaded.domains,[1,1,1,1])
            store=LCGStore('fixture','gold',(3,3,3,3));store.learn(learned)
            values=[1,3,3,3];store.prune(values,lambda:None)
            archive.save_lcg(store);restored=archive.load_lcg()
            self.assertEqual(restored.clauses,store.clauses);self.assertEqual(restored.stats,store.stats)
            self.assertEqual(restored._prune_snapshot,store._prune_snapshot)
            for first in (1,2,3):
                a=[first,3,3,3];b=a.copy();store.prune(a,lambda:None);restored.prune(b,lambda:None)
                self.assertEqual(a,b);self.assertEqual(restored.stats,store.stats)
            self.assertIsNone(CPUReasoningArchive(directory,'fixture','edges',(3,3,3,3)).load_lcg())

    def test_malformed_reason_structure_rejected(self):
        state=LCGState((3,3),'fixture').fork(0,1);data=json.loads(json.dumps(dump_state(state)))
        for key,value in [('level',False),('domains',[2,3]),('current',[99,1]),('enabled',1),('facts',[])]:
            bad=copy.deepcopy(data);bad[key]=value
            with self.assertRaises(ValueError,msg=key):restore_state(bad,(3,3),'fixture','gold')
        bad=copy.deepcopy(data);bad['facts'][2][3]=[2]
        with self.assertRaises(ValueError):restore_state(bad,(3,3),'fixture','gold')
        with self.assertRaises(ValueError):restore_state(data,(1,3),'fixture','gold')

    def test_completed_probes_reuse_same_and_narrower_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=CPUReasoningArchive(directory,'fixture','gold',(7,7));cache=CompletedProbeCache();calls=[]
            scope=('fixture','gold',(7,7),(),(('option',True),))
            def probe(domains,cell):calls.append(tuple(domains));domains[1]&=3;return True
            first=[7,7];cache.run(scope,first,None,probe,lambda:None,{})
            archive.save_probes(cache);loaded=archive.load_probes();self.assertIsNotNone(loaded)
            stats={};exact=[7,7];loaded.run(scope,exact,None,probe,lambda:None,stats)
            self.assertEqual(exact,first);self.assertEqual(len(calls),1);self.assertEqual(stats['lookahead_cache_hits'],1)
            narrow=[1,7];loaded.run(scope,narrow,None,probe,lambda:None,stats)
            self.assertEqual(narrow,[1,3]);self.assertEqual(calls[-1],(1,3));self.assertEqual(stats['lookahead_subset_hits'],1)
            loaded.run((*scope[:-1],(('option',False),)),[7,7],None,probe,lambda:None,stats)
            self.assertEqual(len(calls),3)

    def test_real_proof_import_progress_restarts_without_prefix_work(self):
        from constraint_proof_reuse_round26 import clear
        clear()
        from constraint_proof import MaskProofContext
        from search_problem import SearchProblem
        from test_constraint_dfs import simple_tiles,double_face
        p=SearchProblem(simple_tiles(),double_face());root=tuple(sum(1<<code for code in row) for row in p.domains)
        producer=MaskProofContext(p,root)
        try:certificate=producer.prove([(0,1),(1,1)],seconds=2)
        finally:producer.close()
        self.assertIsNotNone(certificate)
        with tempfile.TemporaryDirectory() as directory:
            total_visits=0;seen_saved=0
            for _ in range(300):
                context=MaskProofContext(p,root,indexed=True);context.configure_local_cache(directory)
                try:
                    result=context.verify_incremental(certificate,ticks=1)
                    total_visits+=context.stats['proof_cursor_visits'];seen_saved+=context.stats['proof_cursor_local_loads']
                finally:context.close()
                if result is not None:break
            self.assertIs(result,True);self.assertGreater(seen_saved,0)
            from constraint_proof_reuse_round26 import clear
            clear()  # The comparison is genuinely cold, without shared checked prefixes.
            cold=MaskProofContext(p,root,indexed=True)
            try:
                self.assertTrue(cold.verify_incremental(certificate,ticks=100000))
                self.assertEqual(total_visits,cold.stats['proof_cursor_visits'])
            finally:cold.close()



class TransitiveSourceBindingTests(unittest.TestCase):
    def test_nested_leaf_and_adjacent_native_are_bound(self):
        from unittest.mock import patch
        import constraint_cpu_archive_round26 as module
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'constraint_cpu_archive_round26.py').write_text('import constraint_dfs\n')
            (root/'constraint_dfs.py').write_text('def optional():\n    import leaf\n')
            (root/'leaf.py').write_text('value=1\n')
            (root/'leaf.c').write_text('int value=1;\n')
            with patch.object(module,'__file__',str(root/'constraint_cpu_archive_round26.py')):
                paths=module.CPUReasoningArchive.sources('propagation')
                self.assertEqual({p.name for p in paths},{'constraint_cpu_archive_round26.py','constraint_dfs.py','leaf.py','leaf.c'})
                (root/'leaf.py').write_text('import extra\n')
                (root/'extra.py').write_text('value=2\n')
                self.assertIn(root/'extra.py',module.CPUReasoningArchive.sources('lcg'))

if __name__=='__main__':unittest.main()
