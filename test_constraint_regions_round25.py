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

def compact_comparison():
    from constraint_region_compact import CompactFactors,compile_compact
    from constraint_region_elimination import problem_factors
    from search_fixtures import planted_gold
    from search_problem import SearchProblem
    data,mesh,witness=planted_gold(2516,size=4);problem=SearchProblem(data,mesh)
    domains=[7<<(3*(code//3)) for code in witness]
    model=CompactFactors(problem,domains);result=compile_compact(model,max_steps=1000)
    capped=False
    try:problem_factors(problem,domains,max_entries=1000)
    except _Budget:capped=True
    assert capped and result['complete'] and all(mask>>code&1 for mask,code in zip(result['supports'],witness))
    pair_entries=sum(len(model.choices[a])*len(model.choices[b]) for a,sa,b,sb in problem.edges)
    assert len({code//3 for code in witness})==160
    return {'cells':problem.n,'candidate_count':sum(mask.bit_count() for mask in domains),
            'prior_cartesian_entry_cap':1000,'prior_construction_capped':capped,
            'prior_complete_cartesian_entries':pair_entries,'compact':dict(model.stats),'result':{k:v for k,v in result.items() if k!='supports'},
            'known_gold_witness_preserved':True,'non_claims':['The compact model proves seam and inventory feasibility only; Gold connectivity is proposal17.',
               'Work counters cover different operations; no universal total-work or elapsed-time dominance is claimed.']}


class CompactFactorTests(unittest.TestCase):
    def test_full160_cell_direct_model_finishes_before_cartesian_construction_cap(self):
        report=compact_comparison()
        self.assertEqual(report['cells'],160);self.assertTrue(report['result']['complete'])
        self.assertLessEqual(report['compact']['compact_factor_transitions'],1000)
        self.assertGreater(report['prior_complete_cartesian_entries'],1000)

    def test_every_support_and_count_matches_independent_complete_edge_solutions(self):
        from constraint_region_compact import CompactFactors,compile_compact
        from test_constraint_sat_domains import fixture
        problem,_=fixture();answers=brute(problem,'edges');gold=brute(problem,'gold');rng=random.Random(25160)
        for _ in range(80):
            domains=[bits(code for code in row if rng.random()<.75) for row in problem.domains]
            result=compile_compact(CompactFactors(problem,domains))
            expected={row for row in answers if all(mask>>code&1 for mask,code in zip(domains,row))}
            support=tuple(bits(row[c] for row in expected) for c in range(problem.n))
            self.assertEqual(result['supports'],support);self.assertEqual(result['rows'],len(expected))
            self.assertTrue(all(all(mask>>code&1 for mask,code in zip(support,row)) for row in gold
                                if all(mask>>code&1 for mask,code in zip(domains,row))))

    def test_resource_lifetimes_forget_only_after_last_occurrence(self):
        from constraint_region_compact import CompactFactors,compile_compact
        problem=ring(8);domains=[bits(3*tile for tile in range(c//2*2,c//2*2+2)) for c in range(8)]
        model=CompactFactors(problem,domains);result=compile_compact(model)
        self.assertEqual(result['rows'],16);self.assertEqual(result['supports'],tuple(domains))
        self.assertEqual(model.stats['compact_factor_live_resources'],2)
        self.assertEqual(model.stats['compact_factor_states'],13)
        for level in range(1,9):
            self.assertFalse(model.future[level]&sum(1<<tile for tile in range((level//2)*2)))
        # Tile zero's final opportunity is the first cell, so a complete board
        # must use it there even though local seams allow the other choice.
        problem=ring(3);domains=[bits((0,3)),bits((3,6)),bits((3,6))]
        model=CompactFactors(problem,domains);result=compile_compact(model)
        self.assertEqual(result['supports'],(1,bits((3,6)),bits((3,6))));self.assertEqual(result['rows'],2)

    def test_stop_caps_unknown_cache_and_scope_changes_are_safe(self):
        from constraint_region_compact import CompactFactorPruner
        problem=ring(6);owner=HierarchicalRegionPruner(problem);root=domains_of(problem)
        for bound in ({'max_placements':0},{'max_frontier':0},{'max_resources':0},{'max_states':0},{'max_steps':0}):
            stage=CompactFactorPruner(owner);current=root.copy();self.assertFalse(stage.prune(current,**bound))
            self.assertEqual(current,root);self.assertEqual(stage.stats['compact_factor_completed'],0)
            spent=stage.stats['compact_factor_transitions'];stage.prune(current,**bound)
            self.assertEqual(stage.stats['compact_factor_transitions'],spent)
        for boundary in (1,5,20,100,300,600):
            stage=CompactFactorPruner(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop compact model')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
            stage.prune(current);self.assertEqual(current,root)
        stage=CompactFactorPruner(owner);stage.prune(root)
        for bad in (-1,True,1.0):
            with self.assertRaises(ValueError):stage.prune(root,max_steps=bad)
        with self.assertRaises(ValueError):stage.prune([-1,*root[1:]])
        problem.domains[0]=problem.domains[0][1:]
        with self.assertRaises(ValueError):stage.prune(root)

    def test_missing_piece_and_public_stage_route_are_complete(self):
        from constraint_region_compact import CompactFactors,compile_compact
        from constraint_round24 import ResearchPruner
        from search_options import normalize_options
        from constraint_dfs import solve
        problem=ring(4);domains=[bits((0,3,6))]*4
        result=compile_compact(CompactFactors(problem,domains));self.assertEqual(result['rows'],0)
        stage=ResearchPruner(problem,'gold',normalize_options({'compact_factors':True}));stage.prune(domains,lambda:None,lambda action,reason:action())
        self.assertFalse(all(domains));self.assertEqual(stage.stats['compact_factor_completed'],1)
        answer=solve(problem,seconds=10,target='gold',options={'compact_factors':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
        self.assertGreater(answer['stats']['compact_factor_completed'],0)

def gold_compact_comparison():
    from constraint_region_compact import CompactFactors,GoldCompactFactors,compile_compact
    from search_fixtures import planted_gold
    from search_problem import SearchProblem
    from validator import validate_arrangement
    data,mesh,witness=planted_gold(2516,size=4);problem=SearchProblem(data,mesh)
    domains=[7<<(3*(code//3)) for code in witness]
    good=GoldCompactFactors(problem,domains);positive=compile_compact(good,max_steps=1000)
    assert positive['witness'] is not None and problem.validate(positive['witness'])['valid']
    bad=None
    for tile in range(len(data['tiles'])):
        segments=data['tiles'][tile]['segments']
        if len(segments)<2:continue
        (a,b),(c,d)=segments[:2]
        for first,second in (([a,c],[b,d]),([a,d],[b,c])):
            changed=copy.deepcopy(data);changed['tiles'][tile]['segments'][:2]=[first,second]
            report=validate_arrangement(changed,witness,mesh)
            if report['matched_edges']==len(mesh.edges) and report['line_components']>1:
                bad=SearchProblem(changed,mesh);break
        if bad is not None:break
    assert bad is not None
    edges=compile_compact(CompactFactors(bad,domains),max_steps=1000)
    gold_model=GoldCompactFactors(bad,domains);negative=compile_compact(gold_model,max_steps=1000)
    assert edges['rows']==1 and negative['rows']==0
    return {'cells':problem.n,'positive_gold_rows':positive['rows'],'positive_gold':dict(good.stats),
            'rewired_edge_perfect_rows':edges['rows'],'rewired_gold_rows':negative['rows'],
            'rewired_witness_components':bad.validate(witness)['line_components'],'negative_gold':dict(gold_model.stats),
            'non_claims':['Both are controlled160-cell domains; the original unrestricted Gold puzzle remains unsolved.',
                          'All complete diagrams enforce full Gold semantics; work or endpoint caps remain unknown.']}


class GoldCompactFactorTests(unittest.TestCase):
    def test_full160_gold_solution_and_edge_perfect_multiple_loop_rejection(self):
        report=gold_compact_comparison()
        self.assertEqual(report['cells'],160);self.assertEqual(report['positive_gold_rows'],1)
        self.assertEqual(report['rewired_edge_perfect_rows'],1);self.assertEqual(report['rewired_gold_rows'],0)
        self.assertGreater(report['rewired_witness_components'],1)

    def test_all_gold_supports_counts_and_witnesses_equal_independent_enumeration(self):
        from constraint_region_compact import GoldCompactFactors,compile_compact
        from test_constraint_sat_domains import fixture
        problem,_=fixture();answers=brute(problem,'gold');rng=random.Random(2517)
        for _ in range(80):
            domains=[bits(code for code in row if rng.random()<.8) for row in problem.domains]
            result=compile_compact(GoldCompactFactors(problem,domains))
            expected={row for row in answers if all(mask>>code&1 for mask,code in zip(domains,row))}
            self.assertEqual(result['rows'],len(expected))
            self.assertEqual(result['supports'],tuple(bits(row[c] for row in expected) for c in range(problem.n)))
            if expected:self.assertIn(result['witness'],expected)
            else:self.assertIsNone(result['witness'])

    def test_early_cycle_and_final_multiple_cycles_are_distinct_rejections(self):
        from constraint_region_compact import GoldCompactFactors,compile_compact
        from test_constraint_dfs import multiple_loops,double_face,board
        from search_problem import SearchProblem
        problem=SearchProblem(multiple_loops(),double_face());model=GoldCompactFactors(problem,domains_of(problem));result=compile_compact(model)
        self.assertEqual(result['rows'],0);self.assertGreater(model.stats['compact_gold_final_rejections'],0)
        mesh=board(4,[(a,side,a+1,side) for a in (0,2) for side in range(3)])
        problem=SearchProblem([{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)],mesh)
        model=GoldCompactFactors(problem,domains_of(problem));result=compile_compact(model)
        self.assertEqual(result['rows'],0);self.assertGreater(model.stats['compact_gold_premature_cycles'],0)

    def test_caps_stop_widening_and_rewired_source_do_not_leak(self):
        from constraint_region_compact import CompactFactorPruner
        problem=ring(5);owner=HierarchicalRegionPruner(problem);root=domains_of(problem)
        for bound in ({'max_ports':0},{'max_states':0},{'max_steps':0}):
            stage=CompactFactorPruner(owner,gold=True);current=root.copy();stage.prune(current,**bound)
            self.assertEqual(current,root);self.assertIsNone(stage.solution);self.assertEqual(stage.stats['compact_gold_completed'],0)
        for boundary in (1,10,40,100,200,400):
            stage=CompactFactorPruner(owner,gold=True);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop exact Gold factor')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root);self.assertIsNone(stage.solution)
            stage.prune(current);self.assertIsNotNone(stage.solution);self.assertTrue(problem.validate(stage.solution)['valid'])
        stage=CompactFactorPruner(owner,gold=True);restricted=root.copy();restricted[0]=1<<problem.domains[0][0]
        stage.prune(restricted);stage.prune(root);self.assertTrue(problem.validate(stage.solution)['valid'])
        with self.assertRaises(ValueError):stage.prune(root,max_ports=True)
        problem.tiles[0]['segments']=[[[0,2],[1,10]]]
        with self.assertRaises(ValueError):stage.prune(root)

    def test_public_dfs_returns_completed_gold_witness_without_another_branch(self):
        from constraint_dfs import solve
        problem=ring(5)
        answer=solve(problem,seconds=10,target='gold',options={'gold_elimination':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertEqual(answer['nodes'],1)
        self.assertEqual(answer['stats']['regional_solutions_validated'],1)
        self.assertEqual(answer['stats']['leaves'],0)
        self.assertTrue(problem.validate(answer['codes'])['valid'])

    def test_dfs_ignores_out_of_scope_witness_and_rejects_invalid_current_witness(self):
        from constraint_dfs import solve
        from constraint_round24 import ResearchPruner
        problem=ring(4);stale=(0,3,6,9);problem.domains[0]=problem.domains[0][1:]
        options={'gold_elimination':True,'native':'python','lookahead':False,'pair_lookahead':False,'adaptive':False}
        with patch.object(ResearchPruner,'solution',new=property(lambda self:stale)):
            answer=solve(problem,seconds=10,target='gold',options=options)
        self.assertEqual(answer['status'],'solved');self.assertNotEqual(tuple(answer['codes']),stale)
        self.assertIn(answer['codes'][0],problem.domains[0]);self.assertEqual(answer['stats']['regional_solutions_validated'],0)
        problem=ring(4)
        with patch.object(ResearchPruner,'solution',new=property(lambda self:(0,0,0,0))):
            with self.assertRaisesRegex(ValueError,'failed independent validation'):
                solve(problem,seconds=10,target='gold',options=options)

    def test_stage_exposes_validated_witness_and_edge_target_keeps_edge_solutions(self):
        from constraint_region_research import RegionalResearchPruner
        from constraint_dfs import solve
        from test_constraint_dfs import multiple_loops,double_face
        from search_problem import SearchProblem
        problem=ring(5);root=domains_of(problem)
        stage=RegionalResearchPruner(problem,'gold',{'compact_factors':True,'gold_elimination':True});stage.prune(root)
        self.assertTrue(problem.validate(stage.solution)['valid']);self.assertEqual(stage.stats['compact_gold_validations'],1)
        stage.prune(root);self.assertEqual(stage.stats['compact_gold_validations'],1)
        answer=solve(problem,seconds=10,target='gold',options={'gold_elimination':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
        self.assertGreater(answer['stats']['compact_gold_completed'],0)
        bad=SearchProblem(multiple_loops(),double_face())
        edge_answer=solve(bad,seconds=10,target='edges',options={'gold_elimination':True,'native':'python'})
        self.assertEqual(edge_answer['status'],'edge_perfect');self.assertEqual(edge_answer['stats']['compact_gold_queries'],0)

def andor_fixture(positive=False):
    patterns=[(0,1,2)]*8;patterns[4]=(1,2,0) if positive else (1,0,2)
    problem=permutation_ring(patterns)
    domains=[bits((0,12)) if cell in (0,4) else bits((3,6,9)) if cell<4 else bits((15,18,21)) for cell in range(8)]
    return problem,domains


def andor_answers(problem,domains,target='gold'):
    answers=set()
    for codes in itertools.product(*(tuple(i for i in range(3*problem.n) if mask>>i&1) for mask in domains)):
        if len({code//3 for code in codes})!=problem.n:continue
        report=problem.validate(codes)
        if report['valid'] or target=='edges' and report['unique_tiles'] and report['matched_edges']==len(problem.edges):answers.add(codes)
    return answers


def andor_comparison():
    from constraint_region_andor import SeparatorSearch
    problem,domains=andor_fixture();assert not andor_answers(problem,domains)
    report={}
    for name,reuse in (('recompile',False),('reuse',True)):
        stage=SeparatorSearch(HierarchicalRegionPruner(problem),reuse_components=reuse);current=domains.copy()
        stage.prune(current);assert not all(current) and stage.solution is None
        report[name]=dict(stage.stats)
    assert report['reuse']['andor_component_builds']<report['recompile']['andor_component_builds']
    assert report['reuse']['andor_component_transitions']<report['recompile']['andor_component_transitions']
    return {'cells':8,'independently_checked_complete_assignments':72,'gold_completions':0,'configurations':report,
            'non_claims':['The comparison isolates conditional component reuse inside the same automatic separator planner.',
                          'No universal dominance over the whole-board compact engine or runtime improvement is asserted.']}


class AutomaticAndOrTests(unittest.TestCase):
    def test_automatic_conditioning_reuses_exact_component_interfaces(self):
        report=andor_comparison();cold=report['configurations']['recompile'];warm=report['configurations']['reuse']
        self.assertEqual(cold['andor_component_builds'],4);self.assertEqual(warm['andor_component_builds'],2)
        self.assertEqual(warm['andor_component_reuses'],2);self.assertEqual(warm['andor_condition_cases'],4)
        self.assertEqual(warm['andor_completed'],1);self.assertEqual(warm['andor_unknown'],0)

    def test_random_whole_query_outcomes_agree_with_independent_enumeration(self):
        from constraint_region_andor import SeparatorSearch
        rng=random.Random(2518)
        for positive in (False,True):
            problem,base=andor_fixture(positive);engine=SeparatorSearch(HierarchicalRegionPruner(problem))
            all_answers=andor_answers(problem,base)
            for _ in range(30):
                domains=[bits(code for code in range(24) if mask>>code&1 and rng.random()<.85) for mask in base]
                expected={codes for codes in all_answers if all(mask>>code&1 for mask,code in zip(domains,codes))};before=domains.copy()
                engine.prune(domains)
                if not all(domains):self.assertFalse(expected)
                if engine.solution is not None:self.assertIn(engine.solution,expected)
                if all(domains):self.assertEqual(domains,before)

    def test_all160_cells_and_fixed_path_interfaces_participate(self):
        from constraint_region_andor import SeparatorSearch
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        data,mesh,witness=planted_gold(25180,size=4);problem=SearchProblem(data,mesh);owner=HierarchicalRegionPruner(problem)
        second=next(c for c in range(1,problem.n) if c not in owner.neighbors[0]);domains=[1<<code for code in witness]
        for cell in (0,second):domains[cell]=7<<(3*(witness[cell]//3))
        stage=SeparatorSearch(owner);stage.prune(domains)
        self.assertEqual(problem.n,160);self.assertIsNotNone(stage.solution)
        self.assertTrue(problem.validate(stage.solution)['valid']);self.assertEqual(stage.stats['andor_component_builds'],2)

    def test_caps_stop_source_and_query_cache_never_publish_partial_negatives(self):
        from constraint_region_andor import SeparatorSearch
        problem,root=andor_fixture(True);owner=HierarchicalRegionPruner(problem)
        for limits in ({'max_active':0},{'max_separator':0},{'max_cases':0},{'max_states':0},{'max_entries':0},{'max_steps':0}):
            engine=SeparatorSearch(owner);current=root.copy();engine.prune(current,**limits)
            self.assertEqual(current,root);self.assertIsNone(engine.solution);spent=engine.stats['andor_work_ticks']
            engine.prune(current,**limits);self.assertEqual(engine.stats['andor_work_ticks'],spent)
        for boundary in (1,5,20,80,150,300,600,900):
            engine=SeparatorSearch(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop conditional decomposition')
            try:engine.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
            engine.prune(current);self.assertIsNotNone(engine.solution);self.assertTrue(problem.validate(engine.solution)['valid'])
        engine=SeparatorSearch(owner)
        with self.assertRaises(ValueError):engine.prune(root,max_steps=True)
        engine.prune(root);owner.target='edges'
        with self.assertRaises(ValueError):engine.prune(root)

    def test_public_route_uses_separator_witness_without_gold_elimination(self):
        from constraint_dfs import solve
        problem,domains=andor_fixture(True)
        problem.domains=[list(code for code in range(24) if mask>>code&1) for mask in domains]
        answer=solve(problem,seconds=10,target='gold',options={'separator_search':True,'gold_elimination':False,
                     'native':'python','lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
        self.assertEqual(answer['nodes'],1);self.assertEqual(answer['stats']['regional_solutions_validated'],1)
        self.assertGreater(answer['stats']['andor_component_builds'],0)



def selective_comparison():
    from constraint_region_refinement import SelectiveDiagram,SelectiveRefinementPruner,_Work
    from constraint_region_elimination import Factor,refined_diagram_possible,WorkBudget
    from test_constraint_sat_domains import fixture,ordinary_fixpoint,admitted
    choices={v:(0,1) for v in range(12)}
    factors=[Factor((9,10),frozenset(((0,0),(1,1)))),Factor((10,11),frozenset(((0,0),(1,1)))),
             Factor((9,11),frozenset(((0,1),(1,0))))]
    configs={}
    for name,reuse in (('rebuild',False),('reuse',True)):
        model=SelectiveDiagram(factors,choices);work=_Work(50000,lambda:None)
        result,witness=model.possible(reuse_prefix=reuse,work=work)
        assert result is False and witness is None
        configs[name]={'work_ticks':work.steps,**model.stats}
    budget=WorkBudget();answer,old=refined_diagram_possible(factors,choices,budget=budget);assert not answer
    problem,domains=fixture();assert ordinary_fixpoint(problem,domains)==domains
    engine=SelectiveRefinementPruner(HierarchicalRegionPruner(problem));current=domains.copy()
    engine.prune(current,max_candidates=48)
    for target in ('edges','gold'):
        answers=brute(problem,target);assert admitted(answers,domains)==admitted(answers,current)
    assert current==[3460,1737,1177,301]
    previous=engine.stats['selective_dd_transitions'];again=domains.copy();engine.prune(again,max_candidates=48)
    assert again==current and engine.stats['selective_dd_transitions']==previous
    return {'late_factor_contradiction':configs,'uniform_width_baseline':old,
            'puzzle_fixture':{'before':domains,'after':current,'stats':engine.stats},
            'non_claims':['Selective refinement is exact finite-factor negative reasoning; positive factor witnesses are not Gold solutions.',
                          'Transition reduction is fixture-specific and support-cache lookups are reported separately.',
                          'Whole original unrestricted Gold solving and general runtime dominance remain unestablished.']}


class SelectiveRefinementTests(unittest.TestCase):
    def test_selective_counterexample_refinement_reuses_unaffected_prefix(self):
        report=selective_comparison();a=report['late_factor_contradiction']['rebuild'];b=report['late_factor_contradiction']['reuse']
        self.assertEqual(b['selective_dd_reused_layers'],9)
        self.assertEqual(a['selective_dd_transitions'],52);self.assertEqual(b['selective_dd_transitions'],34)
        self.assertLess(b['work_ticks'],a['work_ticks'])
        self.assertEqual(report['puzzle_fixture']['stats']['selective_dd_prunes'],24)

    def test_random_finite_relations_preserve_all_exact_answers(self):
        from constraint_region_refinement import SelectiveDiagram
        from constraint_region_elimination import Factor
        rng=random.Random(2519)
        for _ in range(150):
            n=rng.randrange(2,6);choices={v:tuple(range(rng.randrange(1,4))) for v in range(n)};factors=[]
            for k in range(rng.randrange(2,7)):
                scope=tuple(sorted(rng.sample(range(n),rng.randrange(1,min(n,3)+1))))
                rows=frozenset(row for row in itertools.product(*(choices[v] for v in scope)) if rng.random()<.6)
                factors.append(Factor(scope,rows))
            exact=[row for row in itertools.product(*(choices[v] for v in range(n)))
                   if all(tuple(row[v] for v in f.scope) in f.rows for f in factors)]
            for reuse in (False,True):
                model=SelectiveDiagram(factors,choices);answer,witness=model.possible(reuse_prefix=reuse,max_refinements=n,max_states=4096,max_steps=200000)
                self.assertEqual(answer,bool(exact))
                if answer:self.assertIn(tuple(witness[v] for v in range(n)),exact)
                else:self.assertIsNone(witness)

    def test_actual_puzzle_contexts_keep_every_edge_and_gold_completion(self):
        from constraint_region_refinement import SelectiveRefinementPruner
        from test_constraint_sat_domains import fixture,admitted
        problem,base=fixture();stage=SelectiveRefinementPruner(HierarchicalRegionPruner(problem));rng=random.Random(25191)
        answers={t:brute(problem,t) for t in ('gold','edges')}
        for _ in range(40):
            current=[bits(code for code in row if rng.random()<.8) for row in problem.domains];before=current.copy()
            stage.prune(current,max_candidates=48)
            for rows in answers.values():self.assertEqual(admitted(rows,before),admitted(rows,current))
        current=base.copy();stage.prune(current,max_candidates=48);self.assertEqual(current,[3460,1737,1177,301])

    def test_caps_stop_invalid_inputs_and_scope_widening_remain_safe(self):
        from constraint_region_refinement import SelectiveDiagram,SelectiveRefinementPruner
        from constraint_region_elimination import Factor
        from test_constraint_sat_domains import fixture,admitted
        problem,base=fixture();owner=HierarchicalRegionPruner(problem);answers=brute(problem,'edges')
        for cap in ({'max_steps':0},{'max_states':1,'max_refinements':0}):
            stage=SelectiveRefinementPruner(owner);current=base.copy();stage.prune(current,**cap)
            self.assertEqual(admitted(answers,base),admitted(answers,current))
        for boundary in (1,4,20,70,200,600,1500,5000):
            stage=SelectiveRefinementPruner(owner);current=base.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop refinement')
            try:stage.prune(current,check,max_candidates=48)
            except InterruptedError:self.assertEqual(current,base)
            stage.prune(current,max_candidates=48);self.assertEqual(current,[3460,1737,1177,301])
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):SelectiveRefinementPruner(owner,limit=bad)
            with self.assertRaises(ValueError):SelectiveRefinementPruner(owner).prune(base,max_steps=bad)
        for bad in (True,1.5):
            with self.assertRaises(ValueError):SelectiveDiagram([Factor((0,),frozenset(((bad,),)))],{0:(0,1)})
        stage=SelectiveRefinementPruner(owner);current=base.copy();stage.prune(current)
        wide=domains_of(problem);stage.prune(wide);self.assertEqual(admitted(answers,domains_of(problem)),admitted(answers,wide))
        owner.target='edges'
        with self.assertRaises(ValueError):stage.prune(base)

    def test_full160_projection_and_public_app_route(self):
        from constraint_region_refinement import SelectiveRefinementPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        data,mesh,witness=planted_gold(25190,size=4);problem=SearchProblem(data,mesh)
        domains=[1<<code for code in witness]
        for cell in (0,50,100):domains[cell]=7<<(3*(witness[cell]//3))
        stage=SelectiveRefinementPruner(HierarchicalRegionPruner(problem));stage.prune(domains)
        self.assertEqual(stage.stats['selective_dd_full_queries'],1)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(domains,witness)))
        self.assertTrue(all(mask.bit_count()==1 for mask in domains))
        answer=solve(ring(4),seconds=10,target='gold',options={'selective_refinement':True,'native':'python',
                  'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['selective_dd_model_builds'],0)



def joint_comparison():
    from constraint_region_joint import JointSymbolicPruner
    report={}
    for n in (7,9):
        problem=ring(n);root=domains_of(problem);configs={}
        for name,symbolic in (('inventory_pairs',False),('symbolic',True)):
            stage=JointSymbolicPruner(HierarchicalRegionPruner(problem),symbolic=symbolic);current=root.copy()
            stage.prune(current,max_steps=200000);assert current==root and stage.stats['joint_region_completed']==1
            configs[name]=dict(stage.stats)
        report[str(n)]=configs
    assert report['9']['symbolic']['joint_region_work_ticks']<report['9']['inventory_pairs']['joint_region_work_ticks']
    return {'whole_ring_queries':report,'non_claims':[
        'Both configurations include identical direct regional compilation and exact terminal support projection.',
        'The seven-cell case costs more total counted work with ZDDs; the retained option is off by default.',
        'Only two-region complete-board joins are integrated; unrestricted original-board success is unestablished.']}


class JointSymbolicTests(unittest.TestCase):
    def test_correlated_inventory_families_avoid_cartesian_join(self):
        report=joint_comparison()['whole_ring_queries']['9'];a=report['inventory_pairs'];b=report['symbolic']
        self.assertEqual(a['joint_region_inventory_pairs'],15876);self.assertEqual(b['joint_region_interface_joins'],1)
        self.assertEqual(b['joint_region_support_probes'],252);self.assertLess(b['joint_region_work_ticks'],a['joint_region_work_ticks'])

    def test_exact_gold_and_edge_supports_match_independent_enumeration(self):
        from constraint_region_joint import JointSymbolicPruner
        from test_constraint_sat_domains import fixture
        problem,_=fixture();rng=random.Random(2520)
        for target in ('gold','edges'):
            answers=brute(problem,target);stage=JointSymbolicPruner(HierarchicalRegionPruner(problem,target))
            for _ in range(40):
                current=[bits(code for code in row if rng.random()<.8) for row in problem.domains];before=current.copy()
                expected={row for row in answers if all(mask>>code&1 for mask,code in zip(before,row))}
                stage.prune(current,max_steps=200000)
                if all(before):
                    self.assertEqual(current,[bits(row[c] for row in expected) for c in range(problem.n)])
                    self.assertEqual(stage.stats['joint_region_unknown'],0)
        # Different strand pairings are never pooled merely for equal inventory.
        problem=permutation_ring([(0,1,2),(1,2,0),(1,0,2),(0,1,2),(2,0,1),(0,1,2)])
        root=domains_of(problem);answers=andor_answers(problem,root);stage=JointSymbolicPruner(HierarchicalRegionPruner(problem))
        stage.prune(root,max_steps=200000)
        self.assertEqual(root,[bits(row[c] for row in answers) for c in range(problem.n)])
        self.assertGreater(stage.stats['joint_region_interfaces'],2)

    def test_limits_stop_scope_and_cache_do_not_publish_partial_supports(self):
        from constraint_region_joint import JointSymbolicPruner
        problem=ring(5);owner=HierarchicalRegionPruner(problem);root=domains_of(problem)
        for cap in ({'max_steps':0},{'max_states':0},{'max_entries':0},{'max_nodes':0}):
            stage=JointSymbolicPruner(owner);current=root.copy();stage.prune(current,**cap)
            self.assertEqual(current,root);self.assertEqual(stage.stats['joint_region_unknown'],1)
            used=stage.stats['joint_region_work_ticks'];stage.prune(current,**cap);self.assertEqual(stage.stats['joint_region_work_ticks'],used)
        for boundary in (1,5,20,50,150,400,700):
            stage=JointSymbolicPruner(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop symbolic join')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
            stage.prune(current);self.assertEqual(current,root)
        stage=JointSymbolicPruner(owner);current=root.copy();current[0]=1
        stage.prune(current);wide=root.copy();stage.prune(wide);self.assertEqual(wide,root)
        for bad in (-1,True,1.1):
            with self.assertRaises(ValueError):stage.prune(root,max_nodes=bad)
        owner.target='edges'
        with self.assertRaises(ValueError):stage.prune(root)

    def test_all160_cells_and_public_solver_route(self):
        from constraint_region_joint import JointSymbolicPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        data,mesh,witness=planted_gold(25200,size=4);problem=SearchProblem(data,mesh)
        domains=[1<<code for code in witness]
        for cell in (0,90):domains[cell]=7<<(3*(witness[cell]//3))
        stage=JointSymbolicPruner(HierarchicalRegionPruner(problem));stage.prune(domains)
        self.assertEqual(stage.stats['joint_region_completed'],1)
        self.assertEqual(domains,[1<<code for code in witness])
        answer=solve(ring(5),seconds=10,target='gold',options={'joint_symbolic_regions':True,'native':'python',
                 'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['joint_region_completed'],0)



def algebraic_fixture():
    from test_constraint_dfs import board
    from search_problem import SearchProblem
    positions=(1,4,8,11);perms=[(1,0,2,3),(0,2,1,3),(0,1,3,2)]*2+[(0,1,2,3),(1,0,2,3)]
    data=[{'id':i,'segments':[[[0,positions[j]],[1,positions[3-p[j]]]] for j in range(4)]} for i,p in enumerate(perms)]
    problem=SearchProblem(data,board(8,[(i,1,(i+1)%8,0) for i in range(8)]))
    domains=[bits(3*t for t in (range(6) if c<6 else (6,7))) for c in range(8)]
    return problem,domains


def algebraic_comparison():
    from constraint_region_rank import cut_representatives
    from constraint_region_symbolic import endpoint_matchings,representative_matchings
    from constraint_region_research import RegionalResearchPruner
    rows=tuple(endpoint_matchings(range(8)));old,baseline=representative_matchings(rows);new,algebraic=cut_representatives(rows)
    problem,domains=algebraic_fixture();app={}
    for option in ('representative_connectivity','algebraic_connectivity'):
        stage=RegionalResearchPruner(problem,'gold',{option:True});current=domains.copy();stage.prune(current)
        assert current==domains
        app[option]={k:v for k,v in stage.stats.items() if 'algebraic_' in k or 'rank_' in k}
    assert app['algebraic_connectivity']['algebraic_connectivity_kept']<app['algebraic_connectivity']['algebraic_connectivity_rows']
    return {'all_8_endpoint_matchings':{'baseline':baseline,'cut_basis':algebraic},'actual_app':app,
            'non_claims':['Cut-incidence basis may keep more representatives than the full matching-connectivity matrix.',
                          'The app uses existence-only bounded selected-region reasoning; no per-placement support follows from basis membership.',
                          'Shared regional table construction is outside the displayed basis/join comparison.']}


class AlgebraicConnectivityTests(unittest.TestCase):
    def test_cut_basis_preserves_every_outside_matching_for_random_families(self):
        from constraint_region_rank import cut_representatives
        from constraint_region_symbolic import endpoint_matchings,single_cycle
        rng=random.Random(2521)
        for n in (4,6,8):
            rows=tuple(endpoint_matchings(range(n)))
            matrix={(a,b):single_cycle(a,b) for a in rows for b in rows}
            for _ in range(40):
                family=tuple(row for row in rows if rng.random()<.35);basis,stats=cut_representatives(family)
                self.assertTrue(set(basis)<=set(family))
                for outside in rows:self.assertEqual(any(matrix[row,outside] for row in family),any(matrix[row,outside] for row in basis))
        report=algebraic_comparison();a=report['all_8_endpoint_matchings']['baseline'];b=report['all_8_endpoint_matchings']['cut_basis']
        self.assertLess(b['cut_entries']+b['xors'],a['operations']);self.assertGreater(b['representatives'],a['representatives'])

    def test_actual_seam_inventory_buckets_keep_all_gold_completions(self):
        from constraint_region_research import RegionalResearchPruner
        problem,base=algebraic_fixture();answers=andor_answers(problem,base);self.assertTrue(answers);rng=random.Random(25210)
        stage=RegionalResearchPruner(problem,'gold',{'algebraic_connectivity':True})
        for _ in range(30):
            domains=[bits(code for code in range(24) if mask>>code&1 and rng.random()<.9) for mask in base];before=domains.copy()
            expected={row for row in answers if all(mask>>code&1 for mask,code in zip(before,row))}
            stage.prune(domains)
            self.assertTrue(all(all(mask>>code&1 for mask,code in zip(domains,row)) for row in expected))
            if not all(domains):self.assertFalse(expected)
        self.assertGreater(stage.stats['algebraic_connectivity_cut_entries'],0)

    def test_caps_stop_invalid_rows_and_reused_basis_are_safe(self):
        from constraint_region_rank import cut_representatives,AlgebraicConnectivity
        from constraint_region_research import RegionalResearchPruner
        from constraint_region_symbolic import endpoint_matchings
        rows=tuple(endpoint_matchings(range(8)))
        for limits in ({'max_endpoints':6},{'max_rows':1}):
            with self.assertRaises(_Budget):cut_representatives(rows,**limits)
        for bad in ([((0,True),)], [((0,1.5),)], [((0,1),(1,2))], [((0,1),),((0,2),)]):
            with self.assertRaises(ValueError):cut_representatives(bad)
        problem,domains=algebraic_fixture();stage=RegionalResearchPruner(problem,'gold',{'algebraic_connectivity':True})
        regions,tables=stage._table_query(domains,tuple(range(8)),lambda:None,large_first=True)
        engine=AlgebraicConnectivity(stage.owner);self.assertIsNone(engine.feasible(domains,regions,tables,max_steps=0))
        for boundary in (1,5,30,80,120):
            engine=AlgebraicConnectivity(stage.owner);calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop cut basis')
            try:engine.feasible(domains,regions,tables,check)
            except InterruptedError:pass
            self.assertTrue(engine.feasible(domains,regions,tables))
        engine.feasible(domains,regions,tables);self.assertGreater(engine.stats['algebraic_connectivity_basis_hits'],0)
        stage.owner.target='edges'
        with self.assertRaises(ValueError):engine.feasible(domains,regions,tables)

    def test_public_solver_route_preserves_gold_status(self):
        from constraint_dfs import solve
        answer=solve(ring(5),seconds=10,target='gold',options={'algebraic_connectivity':True,'native':'python',
                 'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['algebraic_connectivity_completed'],0)



def isomorphic_contexts(n):
    contexts=[]
    for active in ((0,2),(0,3),(1,4)):
        tiles=iter(range(2,n));contexts.append([9 if c in active else 1<<(3*next(tiles)) for c in range(n)])
    return contexts


def isomorphic_comparison():
    from constraint_region_isomorphism import IsomorphicResidualPruner,STAT_KEYS
    report={}
    for n in (7,160):
        problem=ring(n);owner=HierarchicalRegionPruner(problem);totals=dict.fromkeys(STAT_KEYS,0);warm=IsomorphicResidualPruner(owner)
        for domains in isomorphic_contexts(n):
            cold=IsomorphicResidualPruner(owner);current=domains.copy();cold.prune(current);assert current==domains
            for key in totals:totals[key]+=cold.stats[key]
            current=domains.copy();warm.prune(current);assert current==domains
        report[str(n)]={'cold':totals,'reuse':dict(warm.stats)}
    assert report['160']['reuse']['isomorphic_hits']==2
    assert report['160']['reuse']['isomorphic_work_ticks']<report['160']['cold']['isomorphic_work_ticks']
    return {'contracted_ring_contexts':report,'non_claims':[
        'A160cell ring is a controlled topology fixture, not the original polyhedron.',
        'Seven-cell graph matching costs more total ticks; the option is default off.',
        'Mappings verify the complete encoded residual; no original piece interchangeability or hash authority is assumed.']}


class ResidualIsomorphismTests(unittest.TestCase):
    def test_verified_nongeometric_residual_reuse_reduces_recompilation(self):
        report=isomorphic_comparison()['contracted_ring_contexts']['160'];cold=report['cold'];warm=report['reuse']
        self.assertEqual(cold['isomorphic_compilations'],3);self.assertEqual(warm['isomorphic_compilations'],1)
        self.assertEqual(warm['isomorphic_verified_mappings'],2);self.assertEqual(warm['isomorphic_hits'],2)
        self.assertLess(warm['isomorphic_transitions'],cold['isomorphic_transitions'])

    def test_arbitrary_scopes_and_pairings_keep_exact_gold_supports(self):
        from constraint_region_isomorphism import IsomorphicResidualPruner
        from test_constraint_sat_domains import fixture
        problem,_=fixture();answers=brute(problem,'gold');rng=random.Random(2522);stage=IsomorphicResidualPruner(HierarchicalRegionPruner(problem))
        for _ in range(80):
            domains=[bits(code for code in row if rng.random()<.6) for row in problem.domains];before=domains.copy()
            expected={row for row in answers if all(mask>>code&1 for mask,code in zip(before,row))}
            stage.prune(domains,max_steps=200000)
            if all(before) and any(mask.bit_count()>1 for mask in before):
                self.assertEqual(domains,[bits(row[c] for row in expected) for c in range(problem.n)])
        self.assertEqual(stage.stats['isomorphic_unknown'],0)

    def test_changed_choice_and_port_relationships_do_not_match(self):
        from constraint_region_isomorphism import ResidualGraph,verified_isomorphism,_Work,STAT_KEYS
        stats=dict.fromkeys(STAT_KEYS,0);work=_Work(20000,lambda:None,stats)
        def graph(edges):
            g=ResidualGraph(32,work)
            for color in (0,0,1,1,2,2):g.node(color)
            for a,b,color in edges:g.edge(a,b,color)
            return g
        left=graph(((0,2,1),(1,3,1),(2,4,2),(3,5,2),(4,5,3)))
        right=graph(((1,3,1),(0,2,1),(3,5,2),(2,4,2),(5,4,3)))
        mapping=verified_isomorphism(left,right,work,stats);self.assertIsNotNone(mapping)
        changed=graph(((0,2,1),(1,3,1),(2,4,2),(3,5,2),(4,5,4)))
        self.assertIsNone(verified_isomorphism(left,changed,work,stats))

    def test_limits_stop_fixed_failures_scope_and_actual160_public_route(self):
        from constraint_region_isomorphism import IsomorphicResidualPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        problem=ring(7);root=isomorphic_contexts(7)[0];owner=HierarchicalRegionPruner(problem)
        for cap in ({'max_vertices':0},{'max_states':0},{'max_steps':0}):
            stage=IsomorphicResidualPruner(owner);current=root.copy();stage.prune(current,**cap);self.assertEqual(current,root)
        for boundary in (1,8,25,80,120,200):
            stage=IsomorphicResidualPruner(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop isomorphism')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
            stage.prune(current);self.assertEqual(current,root)
        stage=IsomorphicResidualPruner(owner);bad=root.copy();bad[4]=bad[5];stage.prune(bad);self.assertFalse(all(bad))
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):stage.prune(root,max_trials=bad)
        data,mesh,witness=planted_gold(25220,size=4);problem=SearchProblem(data,mesh);domains=[1<<code for code in witness]
        for cell in (0,100):domains[cell]=7<<(3*(witness[cell]//3))
        stage=IsomorphicResidualPruner(HierarchicalRegionPruner(problem));stage.prune(domains)
        self.assertEqual(domains,[1<<code for code in witness]);self.assertEqual(stage.stats['isomorphic_completed'],1)
        answer=solve(ring(4),seconds=10,target='gold',options={'residual_isomorphism':True,'native':'python',
                   'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['isomorphic_completed'],0)



def gold_pricing_comparison():
    from constraint_region_pricing import GoldPricingPruner
    from constraint_region_master import priced_inventory_master,check_farkas
    from fractions import Fraction
    problem,domains=andor_fixture();owner=HierarchicalRegionPruner(problem)
    diagrams=[DirectRegionDiagram(owner,tuple(range(a,b)),domains) for a,b in ((0,4),(4,8))]
    inventory=priced_inventory_master(diagrams,problem.n);stage=GoldPricingPruner(owner);current=domains.copy();stage.prune(current)
    assert inventory['status']=='possible' and not all(current) and stage.last_certificate
    certificate=stage.last_certificate;y=tuple(Fraction(a,b) for a,b in certificate['certificate'])
    interface_rows={key:2+problem.n+i for i,key in enumerate(certificate['interface_rows'])};columns=list(certificate['columns']);histories=0
    for region,diagram in enumerate(diagrams):
        for codes,signature in diagram.histories():
            histories+=1;columns.append({region:1,interface_rows[region,signature[1:]]:1,**{2+code//3:1 for code in codes}})
    assert check_farkas(columns,certificate['rhs'],y)
    return {'inventory_only':inventory,'gold_aware':dict(stage.stats),'independent_all_history_certificate_columns':histories,
            'non_claims':['The stronger relaxation proves infeasibility on this eight-cell fixture; inventory-only pricing cannot.',
                          'Exact pricing and a full Farkas replay authorize the negative, not the floating LP status.',
                          'This is a bounded two-region fractional master, not unrestricted branch-and-price or a Gold solution finder.']}


class GoldPricingTests(unittest.TestCase):
    def test_terminal_weighted_pricing_matches_independent_history_scores(self):
        from constraint_region_pricing import exact_interface_price
        from fractions import Fraction
        problem,domains=andor_fixture(True);owner=HierarchicalRegionPruner(problem);diagram=DirectRegionDiagram(owner,tuple(range(4)),domains)
        rng=random.Random(2523)
        for _ in range(30):
            pieces={t:Fraction(rng.randrange(-5,6),rng.randrange(1,5)) for t in range(problem.n)}
            terminal={sig[1:]:Fraction(rng.randrange(-4,5),3) for sig in diagram.terminals.values()}
            score,path,signature,edges=exact_interface_price(diagram,pieces,terminal)
            expected=max((sum(pieces[code//3] for code in codes)+terminal[sig[1:]],codes,sig) for codes,sig in diagram.histories())
            self.assertEqual((score,path,signature),expected)
        for bad in (True,1.5):
            with self.assertRaises(ValueError):exact_interface_price(diagram,{0:bad},{})

    @unittest.skipUnless(__import__('importlib').util.find_spec('ortools'),'optional OR-Tools unavailable')
    def test_gold_master_adds_a_certified_negative_beyond_inventory_master(self):
        report=gold_pricing_comparison();self.assertEqual(report['gold_aware']['gold_pricing_failures'],1)
        self.assertGreater(report['independent_all_history_certificate_columns'],report['gold_aware']['gold_pricing_columns'])

    @unittest.skipUnless(__import__('importlib').util.find_spec('ortools'),'optional OR-Tools unavailable')
    def test_random_domains_preserve_all_independently_enumerated_gold_completions(self):
        from constraint_region_pricing import GoldPricingPruner
        from test_constraint_sat_domains import fixture
        problem,_=fixture();answers=brute(problem,'gold');stage=GoldPricingPruner(HierarchicalRegionPruner(problem));rng=random.Random(25230)
        for _ in range(30):
            domains=[bits(code for code in row if rng.random()<.7) for row in problem.domains];before=domains.copy()
            expected={row for row in answers if all(mask>>code&1 for mask,code in zip(before,row))}
            stage.prune(domains)
            self.assertTrue(all(all(mask>>code&1 for mask,code in zip(domains,row)) for row in expected))
            if not all(domains):self.assertFalse(expected)

    def test_unknown_caps_bad_duals_stop_and_scope_are_non_authoritative(self):
        from constraint_region_pricing import GoldPricingPruner
        problem,root=andor_fixture(True);owner=HierarchicalRegionPruner(problem)
        for limits in ({'max_steps':0},{'max_states':0},{'max_rounds':0},{'max_columns':0},{'time_ms':0}):
            stage=GoldPricingPruner(owner);current=root.copy();stage.prune(current,**limits);self.assertEqual(current,root)
        for result in ({'status':'unavailable'},{'status':'optimal','objective':10,'dual':(1.2,)},
                       {'status':'optimal','objective':10,'dual':(True,)}):
            stage=GoldPricingPruner(owner);current=root.copy()
            with patch('constraint_region_pricing.phase_one',return_value=result):stage.prune(current)
            self.assertEqual(current,root);self.assertIsNone(stage.last_certificate)
        for boundary in (1,10,50,100,200,350):
            stage=GoldPricingPruner(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop Gold pricing')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
        stage=GoldPricingPruner(owner)
        with self.assertRaises(ValueError):stage.prune(root,max_steps=True)
        owner.target='edges'
        with self.assertRaises(ValueError):stage.prune(root)

    @unittest.skipUnless(__import__('importlib').util.find_spec('ortools'),'optional OR-Tools unavailable')
    def test_original160_controlled_domains_and_public_dfs(self):
        from constraint_region_pricing import GoldPricingPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        data,mesh,witness=planted_gold(25231,size=4);problem=SearchProblem(data,mesh);domains=[1<<code for code in witness]
        for cell in (0,100):domains[cell]=7<<(3*(witness[cell]//3))
        stage=GoldPricingPruner(HierarchicalRegionPruner(problem));stage.prune(domains)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(domains,witness)));self.assertEqual(stage.stats['gold_pricing_completed'],1)
        answer=solve(ring(4),seconds=10,target='gold',options={'gold_pricing':True,'native':'python',
                 'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['gold_pricing_completed'],0)



def multichain_fixture(positive=False,matching=None):
    from test_constraint_dfs import board
    from search_problem import SearchProblem
    segments=[[[0,10],[1,2]],[[1,10],[2,2]],[[2,10],[0,2]]] if positive else [[[0,10],[1,2]],[[1,10],[0,2]],[[2,2],[2,10]]]
    if matching is not None:segments=[[[a//2,(2,10)[a%2]],[b//2,(2,10)[b%2]]] for a,b in matching]
    tiles=[{'id':0,'segments':segments}]+[{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(1,4)]
    return SearchProblem(tiles,board(4,[(0,s,s+1,0) for s in range(3)])),[7,8,64,512]


def multichain_comparison():
    from constraint_region_multichain import MultichainRoutingPruner
    from constraint_region_oracles import LabelledRoutingPruner
    report={}
    for positive in (False,True):
        problem,domains=multichain_fixture(positive);answers=andor_answers(problem,domains)
        old=LabelledRoutingPruner(problem);current=domains.copy();old.prune(current);assert current==domains
        stage=MultichainRoutingPruner(HierarchicalRegionPruner(problem));current=domains.copy();stage.prune(current)
        assert bool(all(current))==bool(answers)
        report['positive' if positive else 'negative']={'gold_completions':len(answers),'single_chain':old.stats,'joint':stage.stats}
    return {'three_mandatory_chains':report,'non_claims':[
        'Positive joint linkage may omit unused strands; it is not a complete Gold solution.',
        'Multiple strands from one physical tile are deliberately allowed when their cell/code labels agree.',
        'The discriminator compares deduction strength, not whole-search runtime.']}


class MultichainTests(unittest.TestCase):
    def test_three_chain_joint_contradiction_beyond_single_chain_route(self):
        report=multichain_comparison()['three_mandatory_chains'];negative=report['negative'];positive=report['positive']
        self.assertEqual(negative['single_chain']['labelled_failures'],0);self.assertEqual(negative['joint']['multichain_failures'],1)
        self.assertEqual(positive['gold_completions'],3);self.assertEqual(positive['joint']['multichain_failures'],0)

    def test_all_six_port_pairings_and_domains_match_independent_completions(self):
        from constraint_region_multichain import MultichainRoutingPruner
        from constraint_region_symbolic import endpoint_matchings
        for matching in endpoint_matchings(range(6)):
            problem,base=multichain_fixture(matching=matching);answers=andor_answers(problem,base)
            stage=MultichainRoutingPruner(HierarchicalRegionPruner(problem))
            for mask in (3,5,6,7):
                domains=[mask,*base[1:]];before=domains.copy();stage.prune(domains)
                expected={row for row in answers if mask>>row[0]&1}
                self.assertEqual(bool(all(domains)),bool(expected))
                self.assertTrue(all(all(mask>>code&1 for mask,code in zip(domains,row)) for row in expected))

    def test_global_labels_allow_separate_strands_but_reject_conflicting_routes(self):
        from constraint_region_multichain import JointChainOracle
        from constraint_region_oracles import LabelledRouteOracle
        chains=((0,1),(2,3),(4,5))
        for shared in (False,True):
            graph={}
            def edge(a,b,code):
                labels=((0,code),);graph.setdefault(a,[]).append((b,labels));graph.setdefault(b,[]).append((a,labels))
            for i,pairs in enumerate((((1,2),(3,0)),((3,4),(5,2)),((5,0),(1,4)))):
                for a,b in pairs:edge(a,b,0 if shared else i)
            full={v:list(edges) for v,edges in graph.items()}
            for a,b in chains:full[a].append((b,()));full[b].append((a,()))
            for a,b in chains:self.assertTrue(LabelledRouteOracle(full).possible(a,b,count=2))
            self.assertEqual(JointChainOracle(graph,chains).possible(),shared)

    def test_caps_stop_cache_binding_and_actual160_public_route(self):
        from constraint_region_multichain import MultichainRoutingPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        problem,root=multichain_fixture(False);owner=HierarchicalRegionPruner(problem)
        for limits in ({'max_steps':0},{'max_chains':0},{'max_ports':0},{'max_vertices':0}):
            stage=MultichainRoutingPruner(owner);current=root.copy();stage.prune(current,**limits);self.assertEqual(current,root)
        for boundary in (1,5,20,50,80,90):
            stage=MultichainRoutingPruner(owner);current=root.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop joint linkage')
            try:stage.prune(current,check)
            except InterruptedError:self.assertEqual(current,root)
            stage.prune(current);self.assertFalse(all(current))
        stage=MultichainRoutingPruner(owner);current=root.copy();stage.prune(current);steps=stage.stats['multichain_steps']
        current=root.copy();stage.prune(current);self.assertEqual(stage.stats['multichain_steps'],steps)
        with self.assertRaises(ValueError):stage.prune(root,max_ports=True)
        owner.target='edges'
        with self.assertRaises(ValueError):stage.prune(root)
        data,mesh,witness=planted_gold(2524,size=4);problem=SearchProblem(data,mesh);domains=[1<<code for code in witness]
        cell=next(c for c,code in enumerate(witness) if len(problem.tiles[code//3]['segments'])>=2)
        domains[cell]=7<<(3*(witness[cell]//3));stage=MultichainRoutingPruner(HierarchicalRegionPruner(problem));stage.prune(domains)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(domains,witness)));self.assertEqual(stage.stats['multichain_completed'],1)
        problem,domains=multichain_fixture(True);problem.domains=[list(code for code in range(12) if mask>>code&1) for mask in domains]
        answer=solve(problem,seconds=10,target='gold',options={'multichain_routing':True,'native':'python',
                   'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['multichain_completed'],0)



def certificate_comparison():
    from constraint_region_certificate import CertifiedRegionCache
    problem=ring(7);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);cells=tuple(range(7))
    with tempfile.TemporaryDirectory() as directory:
        first=CertifiedRegionCache(owner);first.configure_cache_dir(directory);current=domains.copy();first.prune(cells,current)
        proof=next((Path(directory)/'regional-certificates-v1').glob('*.json'));artifact=json.loads(proof.read_text(encoding='utf-8'))
        second=CertifiedRegionCache(owner);second.configure_cache_dir(directory);current=domains.copy();second.prune(cells,current)
        assert current==domains and second.stats['region_certificate_loads']==1 and second.stats['region_certificate_compiler_transitions']==0
        return {'cells':7,'complete_histories':artifact['count'],'proof_states':len(artifact['states']),
                'artifact_bytes':proof.stat().st_size,'producer':first.stats,'verified_import':second.stats,
                'non_claims':['The importer checks every semantic transition, including dead branches; it does not trust a checksum or accepted-history list.',
                              'Producer plus first verification costs more than an unchecked compile. Reuse avoids rebuilding an imported search DAG, not all proof-checking work.',
                              'The certificate region is optimistic about outside domains and only claims its exact selected-cell relation.']}


class RegionalCertificateTests(unittest.TestCase):
    def test_complete_import_avoids_recompilation_and_checks_every_choice(self):
        report=certificate_comparison();self.assertEqual(report['complete_histories'],5040)
        self.assertEqual(report['proof_states'],128);self.assertEqual(report['verified_import']['region_certificate_verified_transitions'],889)
        self.assertEqual(report['verified_import']['region_certificate_compiler_transitions'],0)
        self.assertEqual(report['producer']['region_certificate_saved_bytes'],report['artifact_bytes'])
        self.assertEqual(report['producer']['region_certificate_export_transitions'],889)

    def test_independent_full_board_counts_and_supports_match_brute_force(self):
        from constraint_region_certificate import build_certificate,verify_certificate
        from test_constraint_sat_domains import fixture
        problem,_=fixture();rng=random.Random(2525)
        for target in ('gold','edges'):
            owner=HierarchicalRegionPruner(problem,target);answers=brute(problem,target)
            for _ in range(30):
                domains=[bits(code for code in row if rng.random()<.75) for row in problem.domains]
                proof=build_certificate(owner,tuple(range(4)),domains);result=verify_certificate(owner,tuple(range(4)),domains,json.loads(json.dumps(proof)))
                expected={row for row in answers if all(mask>>code&1 for mask,code in zip(domains,row))}
                self.assertEqual(result['count'],len(expected));self.assertEqual(result['supports'],tuple(bits(row[c] for row in expected) for c in range(4)))

    def test_partial_region_relation_is_complete_and_optimistic_outside(self):
        from constraint_region_certificate import build_certificate,verify_certificate,CertificateError
        problem=ring(7);owner=HierarchicalRegionPruner(problem);root=domains_of(problem)
        for cells in ((0,1,2),(0,2,4)):
            domains=root.copy()
            for cell in set(range(7))-set(cells):domains[cell]=1<<(3*cell)
            proof=build_certificate(owner,cells,domains);result=verify_certificate(owner,cells,domains,proof)
            independent=tuple(itertools.permutations(range(7),3))
            self.assertEqual(result['count'],len(independent));self.assertEqual(result['count'],210)
            self.assertEqual(result['supports'],tuple(bits(3*row[i] for row in independent) for i in range(3)))
            # Outside fixed-piece commitments reduce global possibilities, but
            # this selected-region proof deliberately retains all local rows.
            self.assertTrue(all(mask>>(3*cell)&1 for cell,mask in zip(cells,result['supports'])))
            changed=domains.copy();other=next(c for c in range(7) if c not in cells);changed[other]=root[other]
            with self.assertRaises(CertificateError):verify_certificate(owner,cells,changed,proof)

    def test_dead_child_and_omitted_transition_forgery_is_rejected(self):
        from constraint_region_certificate import build_certificate,verify_certificate,CertificateError
        problem,domains=multichain_fixture();owner=HierarchicalRegionPruner(problem);cells=tuple(range(4));proof=json.loads(json.dumps(build_certificate(owner,cells,domains)))
        self.assertEqual(proof['count'],0);self.assertTrue(any(edge[1] is not None for edge in proof['states'][proof['root']]['edges']))
        mutations=[]
        bad=copy.deepcopy(proof);bad['states'][bad['root']]['edges'][0][1]=None;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['states'][bad['root']]['edges'].pop();mutations.append(bad)
        bad=copy.deepcopy(proof);bad['states'][bad['root']]['edges'][0][0]=True;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['count']=1;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['supports'][0]=1;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['root']=True;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['states'][1]['signature'][0]=0;mutations.append(bad)
        bad=copy.deepcopy(proof);bad['states'].append(copy.deepcopy(bad['states'][1]));mutations.append(bad)
        bad=copy.deepcopy(proof);bad['states'][1]['signature']=[[[[[[[[[0]]]]]]]]];mutations.append(bad)
        bad=copy.deepcopy(proof);bad['domains'][0]=3;mutations.append(bad)
        for bad in mutations:
            with self.assertRaises((CertificateError,ValueError,TypeError)):verify_certificate(owner,cells,domains,bad)
        with self.assertRaises(_Budget):build_certificate(owner,cells,domains,max_states=0)
        with self.assertRaises(_Budget):verify_certificate(owner,cells,domains,proof,max_steps=0)
        with self.assertRaises(ValueError):build_certificate(owner,cells,[-1,*domains[1:]])
        with self.assertRaises(ValueError):verify_certificate(owner,cells,domains,proof,max_steps=True)

    def test_untrusted_file_tamper_stop_and_write_failure_are_safe(self):
        from constraint_region_certificate import CertifiedRegionCache,build_certificate,verify_certificate
        problem=ring(5);owner=HierarchicalRegionPruner(problem);root=domains_of(problem);cells=tuple(range(5))
        with tempfile.TemporaryDirectory() as directory:
            stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory);stage.prune(cells,root.copy())
            path=next((Path(directory)/'regional-certificates-v1').glob('*.json'))
            data=json.loads(path.read_text(encoding='utf-8'));data['count']=0;data['supports']=[0]*5;path.write_text(json.dumps(data),encoding='utf-8')
            stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory);current=root.copy();stage.prune(cells,current)
            self.assertEqual(current,root);self.assertEqual(stage.stats['region_certificate_rejected'],1)
            for boundary in (2,10,50,100,200):
                stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory);current=root.copy();calls=0
                def check():
                    nonlocal calls
                    calls+=1
                    if calls==boundary:raise InterruptedError('stop imported proof verification')
                with self.assertRaises(InterruptedError):stage.prune(cells,current,check)
                self.assertEqual(current,root);self.assertEqual(stage.stats['region_certificate_completed'],0)
            stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory);calls=0
            def value_error_stop():
                nonlocal calls
                calls+=1
                if calls==10:raise ValueError('caller cancellation is not certificate corruption')
            with self.assertRaisesRegex(ValueError,'caller cancellation'):stage.prune(cells,root.copy(),value_error_stop)
            self.assertEqual(stage.stats['region_certificate_rejected'],0)
            stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory)
            with patch('constraint_region_certificate.build_certificate',side_effect=AssertionError('must verify imported proof')):stage.prune(cells,root.copy())
            self.assertEqual(stage.stats['region_certificate_loads'],1)
        with tempfile.TemporaryDirectory() as directory:
            stage=CertifiedRegionCache(owner);stage.configure_cache_dir(directory)
            with patch('constraint_region_certificate.atomic_json',side_effect=PermissionError('test denied')):stage.prune(cells,root.copy())
            self.assertEqual(stage.stats['region_certificate_write_failures'],1)

    def test_original160_certificate_and_public_stage(self):
        from constraint_region_certificate import build_certificate,verify_certificate
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_dfs import solve
        data,mesh,witness=planted_gold(25250,size=4);problem=SearchProblem(data,mesh);domains=[1<<code for code in witness];owner=HierarchicalRegionPruner(problem)
        proof=build_certificate(owner,tuple(range(160)),domains);result=verify_certificate(owner,tuple(range(160)),domains,proof)
        self.assertEqual(result['count'],1);self.assertEqual(result['supports'],tuple(domains))
        answer=solve(ring(5),seconds=10,target='gold',options={'region_certificates':True,'native':'python',
                    'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertEqual(answer['status'],'solved');self.assertGreater(answer['stats']['region_certificate_completed'],0)

if __name__=='__main__':unittest.main()
