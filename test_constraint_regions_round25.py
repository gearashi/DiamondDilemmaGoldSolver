"""Independent finite regional reuse checks for round25 proposals 14 onward."""
import itertools,random,unittest,json,tempfile,copy
from pathlib import Path
from unittest.mock import patch
from constraint_region_incremental import IncrementalRegionCache
from constraint_region_symbolic import DirectRegionDiagram,join_diagrams
from constraint_regions import HierarchicalRegionPruner
from constraint_separator import _Budget
from test_constraint_separator import ring,permutation_ring
from test_constraint_advanced import domains_of,bits
from test_constraint_dfs import brute


def support_edge_work(diagram):
    pending=[] if diagram.root is None else [diagram.root];seen=set();work=0
    while pending:
        node=pending.pop()
        if node in seen or node in diagram.terminals:continue
        seen.add(node);edges=diagram.nodes[node][1];work+=len(edges);pending.extend(child for code,child in edges)
    return work


def incremental_comparison():
    problem=ring(7);root=domains_of(problem);owner=HierarchicalRegionPruner(problem)
    contexts=[root.copy()]
    for code in problem.domains[0]:
        child=root.copy();child[0]=1<<code;contexts.extend((child,root.copy()))
    cache=IncrementalRegionCache(owner);baseline={'compile_states':0,'compile_transitions':0,'support_edges':0}
    for domains in contexts:
        cold=DirectRegionDiagram(owner,range(7),domains)
        warm=cache.get(range(7),domains)
        assert cold.supports()==warm.supports() and cold.rows==warm.rows
        baseline['compile_states']+=cold.states;baseline['compile_transitions']+=cold.steps
        baseline['support_edges']+=support_edge_work(cold)
    stats=dict(cache.stats)
    baseline['enumeration_and_edge_work']=baseline['compile_transitions']+baseline['support_edges']
    candidate=stats['region_restrict_compile_transitions']+stats['region_restrict_index_edges']+stats['region_restrict_update_edges']+stats['region_restrict_query_edges']
    return {'contexts':len(contexts),'baseline':baseline,'incremental':stats,'incremental_enumeration_and_edge_work':candidate,
            'non_claims':['Copies and source-binding comparisons are additional work; no elapsed-time or whole-Gold solve improvement is asserted.']}


class IncrementalRegionTests(unittest.TestCase):
    def test_restriction_undo_skips_recompilation_on_real_region_relation(self):
        report=incremental_comparison()
        self.assertEqual(report['incremental']['region_restrict_compilations'],1)
        self.assertEqual(report['incremental']['region_restrict_reuses'],14)
        self.assertLess(report['incremental_enumeration_and_edge_work'],report['baseline']['enumeration_and_edge_work'])

    def test_random_complete_histories_match_independent_all_solutions(self):
        rng=random.Random(2514)
        for target in ('gold','edges'):
            problem=permutation_ring([(1,0,2),(0,2,1)]*2);root=domains_of(problem)
            owner=HierarchicalRegionPruner(problem,target);cache=IncrementalRegionCache(owner)
            original=cache.get(range(4),root);answers=brute(problem,target);old_histories=set(original.histories())
            for _ in range(50):
                domains=[bits(code for code in row if rng.random()<.75) for row in problem.domains]
                result=cache.get(range(4),domains)
                expected={codes for codes in answers if all(mask>>code&1 for mask,code in zip(domains,codes))}
                self.assertEqual({codes for codes,signature in result.histories()},expected)
                self.assertEqual(result.rows,len(expected))
                self.assertEqual(result.supports(),tuple(bits(row[c] for row in expected) for c in range(4)))
            self.assertEqual(set(original.histories()),old_histories)
            self.assertEqual(cache.stats['region_restrict_compilations'],1)

    def test_partial_regions_guard_outside_assumptions_and_overlap_views(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3);owner=HierarchicalRegionPruner(problem)
        root=domains_of(problem);cache=IncrementalRegionCache(owner);left=(0,1,2,3);right=(2,3,4,5)
        cache.get(left,root);child=root.copy();child[0]&=~(1<<problem.domains[0][0])
        view=cache.get(left,child);cold=DirectRegionDiagram(owner,left,child)
        self.assertEqual(set(view.histories()),set(cold.histories()))
        self.assertEqual(cache.stats['region_restrict_compilations'],1)
        changed=child.copy();changed[5]&=~(1<<problem.domains[5][0]);cache.get(left,changed)
        self.assertEqual(cache.stats['region_restrict_compilations'],2)
        rhs=cache.get(right,child);expected=join_diagrams(owner,cold,DirectRegionDiagram(owner,right,child),child)[0]
        self.assertEqual(join_diagrams(owner,view,rhs,child)[0],expected)
        self.assertEqual(set(view.histories()),set(cold.histories()))

    def test_stop_caps_and_source_changes_never_publish_partial_supports(self):
        problem=ring(5);owner=HierarchicalRegionPruner(problem);root=domains_of(problem)
        child=root.copy();child[4]=1<<problem.domains[4][0];stopped=0
        for boundary in range(1,35):
            cache=IncrementalRegionCache(owner);cache.get(range(5),root);current=child.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop')
            try:cache.prune(range(5),current,check)
            except InterruptedError:stopped+=1;self.assertEqual(current,child)
            expected=DirectRegionDiagram(owner,range(5),child).supports()
            self.assertEqual(cache.get(range(5),child).supports(),expected)
        self.assertGreater(stopped,10)
        cache=IncrementalRegionCache(owner);current=root.copy()
        self.assertFalse(cache.prune(range(5),current,max_steps=0));self.assertEqual(current,root)
        self.assertFalse(cache.diagrams)
        for invalid in ([-1,*root[1:]],[True,*root[1:]]):
            with self.assertRaises(ValueError):cache.get(range(5),invalid)
        cache.get(range(5),root);problem.tiles[0]['segments']=[[[0,2],[1,10]]]
        with self.assertRaises(ValueError):cache.get(range(5),root)

    def test_actual160_cell_region_preserves_known_gold_witness_and_reuses(self):
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_region_research import RegionalResearchPruner
        data,mesh,witness=planted_gold(25140,size=4);problem=SearchProblem(data,mesh)
        domains=[1<<code for code in witness]
        for cell in range(8):domains[cell]=7<<(3*(witness[cell]//3))
        self.assertEqual(problem.n,160);self.assertTrue(problem.validate(witness)['valid'])
        stage=RegionalResearchPruner(problem,'gold',{'incremental_regions':True})
        first=domains.copy();stage.prune(first)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(first,witness)))
        spent=stage.stats['region_restrict_compile_transitions'];again=domains.copy();stage.prune(again)
        self.assertEqual(again,first);self.assertEqual(stage.stats['region_restrict_compile_transitions'],spent)
        self.assertGreater(stage.stats['region_restrict_reuses'],0)

    def test_actual_research_stage_and_public_dfs_route(self):
        from constraint_round24 import ResearchPruner
        from search_options import normalize_options
        from constraint_dfs import solve
        problem=ring(5);root=domains_of(problem)
        stage=ResearchPruner(problem,'gold',normalize_options({'incremental_regions':True}))
        stage.prune(root,lambda:None,lambda action,reason:action())
        child=root.copy();child[0]=1<<problem.domains[0][0]
        stage.prune(child,lambda:None,lambda action,reason:action())
        self.assertGreater(stage.stats['region_restrict_reuses'],0)
        self.assertEqual(sum(mask.bit_count() for mask in child),17)
        answer=solve(problem,seconds=10,target='gold',options={'incremental_regions':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
        self.assertGreater(answer['stats']['region_restrict_completed'],0)

def session_diagram(session):
    result=object.__new__(DirectRegionDiagram);result.__dict__.update(session.result());return result


def durable_comparison():
    from constraint_region_cursor import PortableRegionSession,LocalRegionCursorStore
    problem=ring(7);owner=HierarchicalRegionPruner(problem);domains=tuple(domains_of(problem));cells=tuple(range(7))
    key=(cells,domains,4096,50000);cold=PortableRegionSession(owner,cells,domains)
    while not cold.advance(256):pass
    baseline=cold.steps;unique=0;previous=None
    with tempfile.TemporaryDirectory() as folder:
        store=LocalRegionCursorStore(folder,owner)
        for _ in range(3):
            fresh=PortableRegionSession(owner,cells,domains);fresh.advance(200);baseline+=fresh.steps
            resumed=store.load(key) or PortableRegionSession(owner,cells,domains)
            before=resumed.steps;resumed.advance(200);unique+=resumed.steps-before;store.save(key,resumed)
        resumed=store.load(key);before=resumed.steps
        while not resumed.advance(256):pass
        unique+=resumed.steps-before;store.save(key,resumed)
        assert session_diagram(resumed).supports()==session_diagram(cold).supports() and resumed.rows==cold.rows
        assert resumed.steps==cold.steps and resumed.ticks==cold.ticks and unique==cold.steps
        return {'process_restarts':3,'baseline_transition_attempts':baseline,'durable_transition_attempts':unique,
                'completed_transitions_repeated':unique-cold.steps,'rows':resumed.rows,'final_ticks':resumed.ticks,
                'store':dict(store.stats),'non_claims':['Trusted local cache only; hashing and JSON reconstruction are additional work.',
                    'An abrupt crash can repeat transitions since the last successful save. No full-board solve or elapsed-time claim.']}


def local_cursor_writer(folder,quantum):
    from constraint_region_cursor import PortableRegionSession,LocalRegionCursorStore
    problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=tuple(domains_of(problem));cells=tuple(range(5))
    session=PortableRegionSession(owner,cells,domains);session.advance(quantum)
    store=LocalRegionCursorStore(folder,owner)
    for _ in range(3):
        if not store.save((cells,domains,4096,50000),session):raise RuntimeError('Concurrent cursor save failed.')


class DurableRegionTests(unittest.TestCase):
    def test_explicit_compiler_matches_old_compiler_and_all_restart_boundaries(self):
        from constraint_region_cursor import PortableRegionSession
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);domains=tuple(domains_of(problem));cells=tuple(range(4))
        checked=0
        for target in ('gold','edges'):
            owner=HierarchicalRegionPruner(problem,target);original=DirectRegionDiagram(owner,cells,domains)
            cold=PortableRegionSession(owner,cells,domains)
            while not cold.advance(256):pass
            self.assertEqual(set(session_diagram(cold).histories()),set(original.histories()))
            self.assertEqual((cold.states,cold.steps,cold.memo_hits),(original.states,original.steps,original.memo_hits))
            for boundary in range(cold.ticks+1):
                session=PortableRegionSession(owner,cells,domains)
                if boundary:session.advance(boundary)
                payload=json.loads(json.dumps(session._local_payload()))
                restored=PortableRegionSession._from_local_payload(owner,payload,cells,domains,4096,50000)
                while not restored.advance(256):pass
                self.assertEqual(set(session_diagram(restored).histories()),set(original.histories()))
                self.assertEqual((restored.states,restored.steps,restored.memo_hits,restored.ticks),(cold.states,cold.steps,cold.memo_hits,cold.ticks))
                checked+=1
        self.assertGreater(checked,100)

    def test_restart_discriminator_repeats_no_completed_compiler_transition(self):
        report=durable_comparison()
        self.assertGreater(report['baseline_transition_attempts'],report['durable_transition_attempts'])
        self.assertEqual(report['completed_transitions_repeated'],0)
        self.assertEqual(report['rows'],5040)

    def test_interrupted_app_stage_loads_exact_cursor_in_new_stage(self):
        from constraint_region_research import RegionalResearchPruner
        problem=ring(7);domains=domains_of(problem);options={'incremental_regions':True,'durable_regions':True}
        with tempfile.TemporaryDirectory() as folder:
            first=RegionalResearchPruner(problem,'gold',options);first.configure_cache_dir(folder);current=domains.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==81:raise InterruptedError('stop during compilation')
            with self.assertRaises(InterruptedError):first.prune(current,check)
            self.assertEqual(current,domains);self.assertGreater(first.stats['region_cursor_writes'],0)
            attempted=first.stats['region_restrict_compile_transitions']
            second=RegionalResearchPruner(problem,'gold',options);second.configure_cache_dir(folder);second.prune(current)
            cold=DirectRegionDiagram(first.owner,range(7),domains)
            self.assertGreater(second.stats['region_cursor_loads'],0)
            self.assertEqual(attempted+second.stats['region_restrict_compile_transitions'],cold.steps)
            self.assertEqual(current,list(cold.supports()))
            third=RegionalResearchPruner(problem,'gold',options);third.configure_cache_dir(folder);again=domains.copy();third.prune(again)
            self.assertEqual(again,current);self.assertEqual(third.stats['region_restrict_compile_transitions'],0)

    def test_corruption_source_caps_and_strict_shapes_recover_cold(self):
        from constraint_region_cursor import PortableRegionSession,LocalRegionCursorStore,encoded
        import hashlib
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=tuple(domains_of(problem));cells=tuple(range(4));key=(cells,domains,4096,50000)
        with tempfile.TemporaryDirectory() as folder:
            store=LocalRegionCursorStore(folder,owner);session=PortableRegionSession(owner,cells,domains);session.advance(20)
            self.assertTrue(store.save(key,session));path=store._path(key);original=json.loads(path.read_text())
            for mutate in (lambda x:x['payload']['cursor'].__setitem__('steps',True),
                           lambda x:x['payload']['cursor'].__setitem__('complete',True),
                           lambda x:x['payload']['cursor']['stack'][0].__setitem__('index',999999),
                           lambda x:x['payload']['cursor'].__setitem__('nodes',[[False,[0,[[0,0]]]]])):
                changed=copy.deepcopy(original);mutate(changed)
                changed['sha256']=hashlib.sha256(encoded(changed['payload'])).hexdigest();path.write_text(json.dumps(changed))
                self.assertIsNone(store.load(key))
            path.write_text('{broken');self.assertIsNone(store.load(key))
            self.assertTrue(store.save(key,session))
            altered=(cells,domains,4096,50001);self.assertIsNone(store.load(altered))
            old_producer=store.producer;store.producer=old_producer+(('changed','hash'),);self.assertIsNone(store.load(key));store.producer=old_producer
            stopped=lambda:(_ for _ in ()).throw(InterruptedError('stop'))
            with self.assertRaises(InterruptedError):store.load(key,stopped)
            cache=IncrementalRegionCache(owner,durable=True);cache.configure_cache_dir(folder)
            path.write_text('broken');got=cache.get(cells,domains)
            self.assertEqual(got.supports(),DirectRegionDiagram(owner,cells,domains).supports())
            self.assertGreater(cache.stats['region_cursor_rejections'],0)

    def test_storage_failure_and_capped_cursor_remain_non_authoritative(self):
        from constraint_region_cursor import LocalRegionCursorStore
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        with tempfile.TemporaryDirectory() as folder:
            first=IncrementalRegionCache(owner,durable=True);first.configure_cache_dir(folder);current=domains.copy()
            self.assertFalse(first.prune(range(5),current,max_steps=0));self.assertEqual(current,domains)
            second=IncrementalRegionCache(owner,durable=True);second.configure_cache_dir(folder)
            self.assertFalse(second.prune(range(5),current,max_steps=0));self.assertEqual(current,domains)
            self.assertGreater(second.stats['region_cursor_limited_hits'],0)
            self.assertEqual(second.stats['region_restrict_compile_transitions'],0)
            third=IncrementalRegionCache(owner,durable=True);third.configure_cache_dir(folder)
            with patch('constraint_region_cursor.atomic_json',side_effect=PermissionError('denied')):
                result=third.get(range(5),domains)
            self.assertEqual(result.rows,120);self.assertGreater(third.stats['region_cursor_write_failures'],0)
            third.configure_cache_dir(None);self.assertIsNone(third.store)

    def test_written_byte_bound_matches_atomic_json_and_strict_cursor_invariants(self):
        from constraint_region_cursor import PortableRegionSession,LocalRegionCursorStore,encoded
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=tuple(domains_of(problem));cells=tuple(range(4));key=(cells,domains,4096,50000)
        session=PortableRegionSession(owner,cells,domains);session.advance(20)
        with tempfile.TemporaryDirectory() as folder:
            store=LocalRegionCursorStore(folder,owner);self.assertTrue(store.save(key,session))
            path=store._path(key);actual=path.stat().st_size
            self.assertEqual(store.stats['region_cursor_saved_bytes'],actual)
            compact=len(encoded(json.loads(path.read_text(encoding='utf-8'))));self.assertGreater(actual,compact)
            bounded=LocalRegionCursorStore(Path(folder)/'bounded',owner,max_bytes=compact+1)
            self.assertFalse(bounded.save(key,session));self.assertFalse(bounded._path(key).exists())
            bounded.max_bytes=actual;self.assertTrue(bounded.save(key,session));self.assertIsNotNone(bounded.load(key))
        payload=json.loads(json.dumps(session._local_payload()))
        payload['steps']=50001
        with self.assertRaises(ValueError):PortableRegionSession._from_local_payload(owner,payload,cells,domains,4096,50000)
        # An awaiting child's code is unfinished work, not a completed edge.
        while not any(frame['stage']=='child' and frame['edges'] for frame in session.stack):session.advance(1)
        payload=json.loads(json.dumps(session._local_payload()))
        frame=next(frame for frame in payload['stack'] if frame['stage']=='child' and frame['edges'])
        frame['edges'].append([frame['pending'],frame['edges'][-1][1]])
        with self.assertRaises(ValueError):PortableRegionSession._from_local_payload(owner,payload,cells,domains,4096,50000)

    def test_unexpected_midtransition_error_cannot_save_or_resume_skipped_branch(self):
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        with tempfile.TemporaryDirectory() as folder:
            cache=IncrementalRegionCache(owner,durable=True);cache.configure_cache_dir(folder)
            with patch('constraint_region_cursor.assignment_signature',side_effect=RuntimeError('unexpected internal fault')):
                with self.assertRaises(RuntimeError):cache.get(range(4),domains)
            self.assertFalse(cache.pending);self.assertFalse(list(Path(folder).rglob('*.json')))
            recovered=cache.get(range(4),domains)
            self.assertEqual(set(recovered.histories()),set(DirectRegionDiagram(owner,range(4),domains).histories()))

    def test_two_process_writers_keep_one_complete_cursor_record(self):
        import multiprocessing
        from constraint_region_cursor import PortableRegionSession,LocalRegionCursorStore
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=tuple(domains_of(problem));cells=tuple(range(5))
        with tempfile.TemporaryDirectory() as folder:
            context=multiprocessing.get_context('spawn')
            workers=[context.Process(target=local_cursor_writer,args=(folder,quantum)) for quantum in (40,120)]
            for worker in workers:worker.start()
            for worker in workers:
                worker.join(20)
                if worker.is_alive():worker.terminate();worker.join();self.fail('Bounded cursor writer did not finish.')
                self.assertEqual(worker.exitcode,0)
            restored=LocalRegionCursorStore(folder,owner).load((cells,domains,4096,50000))
            self.assertIsNotNone(restored);self.assertIn(restored.ticks,(40,120))
            while not restored.advance(256):pass
            cold=DirectRegionDiagram(owner,cells,domains)
            self.assertEqual(set(session_diagram(restored).histories()),set(cold.histories()))
            self.assertEqual(restored.steps,cold.steps)

    def test_actual_public_dfs_forwards_directory_without_extended_regions(self):
        from constraint_dfs import solve
        problem=ring(5)
        with tempfile.TemporaryDirectory() as folder:
            state={'_regional_cache_dir':folder}
            answer=solve(problem,seconds=10,target='gold',learning_state=state,options={
                'durable_regions':True,'native':'python','extended_regions':False,
                'lookahead':False,'pair_lookahead':False,'adaptive':False})
            self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
            self.assertGreater(answer['stats']['region_cursor_writes'],0)
            self.assertTrue(list((Path(folder)/'regional-cursors-v1').glob('*.json')))

if __name__=='__main__':unittest.main()
