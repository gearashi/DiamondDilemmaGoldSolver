"""Finite disjoint-queue and real spawn regressions; no production puzzle run."""
import copy
import itertools
import json
import os
import unittest
from unittest.mock import patch
from constraint_dfs import solve as dfs_solve,_checkpoint_digest
from constraint_parallel import solve,partition_checkpoint,_digest,_WorkerCaches,_choose_job,AFFINITY_SLICES
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import multiple_loops,double_face,simple_tiles

OPTIONS={'failure_guided':False,'lookahead':False,'strand_connectivity':False,'multi_seam_cuts':False,'regional_cache':False,'biconnected':False,'separator_regions':False,'backjumping':False,'incremental':False,'adaptive':False,'persistent_learning':False,'native':'python'}

def legacy_root(problem,hints=None):
    checkpoint=dfs_solve(problem,seconds=0,hints=hints,options=OPTIONS)['checkpoint']
    checkpoint.pop('root_symmetry',None)
    checkpoint['digest']=_checkpoint_digest(checkpoint)
    return checkpoint

class CrashProblem(SearchProblem):
    def validate(self,codes):
        os._exit(29)

def checkpoint_emitting_worker(problem,connection,cancel):
    request=connection.recv()
    if request is None:return
    result=dfs_solve(problem,seconds=0,target=request['target'],resume=request['checkpoint'],options=request['options'])
    connection.send({'kind':'progress','id':request['id'],'pid':os.getpid(),'event':{
        'event':'progress','nodes':0,'elapsed_seconds':0,'codes':[-1]*problem.n,
        'is_partial':True,'checkpoint':result['checkpoint']}})
    cancel.wait(3)
    connection.send({'kind':'result','id':request['id'],'pid':os.getpid(),'result':result})
    connection.recv();connection.close()

def forged_knowledge_worker(problem,connection,cancel):
    from constraint_knowledge import KnowledgeStore,_digest as knowledge_digest
    request=connection.recv()
    payload=KnowledgeStore(problem).snapshot()
    payload['records']=[[[0,0],[1,4]]];payload['digest']=knowledge_digest(payload)
    connection.send({'kind':'progress','id':request['id'],'pid':os.getpid(),'event':{
        'event':'progress','nodes':0,'knowledge':payload}})
    cancel.wait(3)
    connection.close()

def claimed_conflict_worker(problem,connection,cancel):
    _conflict_worker(problem,connection,cancel,[[[0,0],[1,3]]])

def valid_conflict_worker(problem,connection,cancel):
    _conflict_worker(problem,connection,cancel,[[[0,0],[1,0]]])

def malformed_conflict_worker(problem,connection,cancel):
    _conflict_worker(problem,connection,cancel,[[[0,0],[1,0]]],malformed=True)

def _conflict_worker(problem,connection,cancel,records,malformed=False):
    from test_constraint_shared_conflicts import snapshot_claims
    request=connection.recv()
    payload=snapshot_claims(problem,records)
    if malformed:payload['digest']='invalid'
    connection.send({'kind':'progress','id':request['id'],'pid':os.getpid(),'event':{
        'event':'progress','nodes':0,'shared_conflicts':payload}})
    cancel.wait(3)
    result=dfs_solve(problem,seconds=0,target=request['target'],resume=request['checkpoint'],options=request['options'])
    connection.send({'kind':'result','id':request['id'],'pid':os.getpid(),'result':result})
    connection.recv();connection.close()

class ParallelTests(unittest.TestCase):
    def setUp(self):self.problem=SearchProblem(multiple_loops(),double_face())
    def test_partition_exact_remaining_choices_and_no_input_mutation(self):
        checkpoint=legacy_root(self.problem);all_codes=hex((1<<6)-1)
        checkpoint['stack']=[{'domains':[all_codes,all_codes],'cell':0,'choices':list(range(6)),'next':2},
                             {'domains':[hex(1<<1),hex(sum(1<<c for c in [3,4,5]))],'cell':1,'choices':[3,4,5],'next':1}]
        checkpoint['digest']=_checkpoint_digest(checkpoint);before=copy.deepcopy(checkpoint)
        parts=partition_checkpoint(self.problem,checkpoint,parts=8,options=OPTIONS)
        self.assertEqual(checkpoint,before)
        for codes in itertools.product(range(6),repeat=2):
            if codes[0]//3==codes[1]//3:continue
            expected=codes[0]>=2 or codes[0]==1 and codes[1] in (4,5)
            owners=sum(all(int(value,16)&(1<<code) for value,code in zip(part['stack'][0]['domains'],codes)) for part in parts)
            self.assertEqual(owners,int(expected),codes)
        self.assertTrue(all(part['nodes']==0 for part in parts))
    def test_pending_symmetry_is_not_split_and_zero_budget_spawns_nothing(self):
        root=dfs_solve(self.problem,seconds=0,options=OPTIONS)['checkpoint']
        self.assertEqual(len(partition_checkpoint(self.problem,root,parts=64,options=OPTIONS)),1)
        with patch('constraint_parallel.mp.get_context',side_effect=AssertionError('No spawn')):
            result=solve(self.problem,seconds=0,workers=4,options=OPTIONS)
        self.assertEqual(result['status'],'timeout');self.assertFalse(result['complete'])
        self.assertEqual(result['parallel']['pending_jobs'],1)
        self.assertEqual(result['checkpoint']['inflight_jobs'],[])
    def test_exact_spawn_enumeration_matches_serial_and_is_disjoint(self):
        expected=[];seen=[]
        serial=dfs_solve(self.problem,seconds=5,options=OPTIONS,progress=lambda e:expected.append(tuple(e['codes'])) if e['event']=='candidate' else None)
        result=solve(self.problem,seconds=10,workers=2,resume=legacy_root(self.problem),options=OPTIONS,
                     progress=lambda e:seen.append(tuple(e['codes'])) if e['event']=='candidate' else None)
        self.assertEqual(serial['status'],'infeasible');self.assertEqual(result['status'],'infeasible',result.get('error'))
        self.assertEqual(set(seen),set(expected));self.assertEqual(len(seen),len(set(seen)));self.assertEqual(len(seen),18)
        self.assertEqual(result['stats']['parallel_workers_started'],2)
        self.assertEqual(result['native_backend'],'python')
    def test_cooperative_stop_and_changed_worker_resume_preserve_leaf_coverage(self):
        seen=[]
        def progress(event):
            if event['event']=='candidate':seen.append(tuple(event['codes']))
        first=solve(self.problem,seconds=10,workers=1,resume=legacy_root(self.problem),options=OPTIONS,
                    progress=progress,stop=lambda:len(seen)>=1)
        self.assertEqual(first['status'],'stopped',first.get('error'))
        checkpoint=json.loads(json.dumps(first['checkpoint']))
        self.assertEqual(checkpoint['inflight_jobs'],[])
        final=solve(self.problem,seconds=10,workers=3,resume=checkpoint,options=OPTIONS,progress=progress)
        self.assertEqual(final['status'],'infeasible',final.get('error'))
        self.assertEqual(len(seen),18);self.assertEqual(len(set(seen)),18)
        self.assertGreaterEqual(final['nodes'],first['nodes'])
        self.assertEqual(final['stats']['parallel_replayed_jobs'],0)
    def test_actual_spawn_planted_gold_independent_validation(self):
        data,board,known=planted_gold(seed=199,size=1)
        problem=SearchProblem(data,board)
        result=solve(problem,seconds=10,workers=2,resume=legacy_root(problem,known),options=OPTIONS)
        self.assertEqual(result['status'],'solved',result.get('error'))
        self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertEqual(result['stats']['parallel_workers_started'],2)
    def test_ready_symmetry_partition_preserves_single_canonical_root(self):
        from constraint_symmetry import canonical_root_domains,proper_rotations
        data,board,known=planted_gold(seed=199,size=1)
        problem=SearchProblem(data,board)
        anchor_cell=next(i for i,code in enumerate(known) if code//3==0)
        rotation=next(t for t in proper_rotations(board) if t.faces[board.cells[anchor_cell].face]==0)
        known=rotation.apply(known)
        checkpoint=dfs_solve(problem,seconds=0,hints=known,options=OPTIONS)['checkpoint']
        domains,metadata=canonical_root_domains(problem)
        checkpoint['root_symmetry'].update(state='ready',reduction=metadata)
        checkpoint['stack'][0]['domains']=[hex(value) for value in domains]
        checkpoint['digest']=_checkpoint_digest(checkpoint)
        children=partition_checkpoint(problem,checkpoint,parts=4,options=OPTIONS)
        self.assertTrue(all(child['root_symmetry']==checkpoint['root_symmetry'] for child in children))
        self.assertEqual(sum(all(int(value,16)&(1<<code) for value,code in zip(child['stack'][0]['domains'],known)) for child in children),1)
        result=solve(problem,seconds=10,workers=2,resume=checkpoint,options=OPTIONS)
        self.assertEqual(result['status'],'solved',result.get('error'))
        self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertEqual(result['root_symmetry'],checkpoint['root_symmetry'])

    def test_worker_crash_preserves_last_committed_job_and_reports_replay(self):
        problem=CrashProblem(simple_tiles(),double_face())
        result=solve(problem,seconds=5,workers=1,resume=legacy_root(problem),options=OPTIONS)
        self.assertEqual(result['status'],'error');self.assertFalse(result['complete'])
        self.assertIn('last committed',result['error'])
        self.assertTrue(result['checkpoint']['jobs'])
        self.assertGreaterEqual(result['stats']['parallel_replayed_jobs'],1)
        safe=SearchProblem(simple_tiles(),double_face())
        resumed=solve(safe,seconds=10,workers=2,resume=json.loads(json.dumps(result['checkpoint'])),options=OPTIONS)
        self.assertEqual(resumed['status'],'solved',resumed.get('error'))
    def test_checkpoint_rejects_corruption_duplicate_overlap_and_region_escape(self):
        first=solve(self.problem,seconds=0,workers=1,resume=legacy_root(self.problem),options=OPTIONS)['checkpoint']
        variants=[]
        bad=copy.deepcopy(first);bad['retired_nodes']+=1;variants.append(bad)
        bad=copy.deepcopy(first);bad['parallel_version']=999;bad['digest']=_digest(bad);variants.append(bad)
        bad=copy.deepcopy(first);bad['parallel_version']=True;bad['digest']=_digest(bad);variants.append(bad)
        bad=copy.deepcopy(first);bad['jobs'].append(copy.deepcopy(bad['jobs'][0]));bad['digest']=_digest(bad);variants.append(bad)
        bad=copy.deepcopy(first);extra=copy.deepcopy(bad['jobs'][0]);extra['id']=1;bad['jobs'].append(extra);bad['next_job_id']=2;bad['digest']=_digest(bad);variants.append(bad)
        bad=copy.deepcopy(first);bad['jobs'][0]['region'][0]='0x1';bad['digest']=_digest(bad);variants.append(bad)
        bad=copy.deepcopy(first);bad['inflight_jobs']=[99];bad['digest']=_digest(bad);variants.append(bad)
        for checkpoint in variants:
            with self.subTest(checkpoint=checkpoint),self.assertRaises(ValueError):solve(self.problem,seconds=0,resume=checkpoint,options=OPTIONS)
    def test_snapshot_owns_data_and_replayed_lease_is_explicit(self):
        first=solve(self.problem,seconds=0,resume=legacy_root(self.problem),options=OPTIONS)
        cp=copy.deepcopy(first['checkpoint']);cp['inflight_jobs']=[cp['jobs'][0]['id']];cp['digest']=_digest(cp)
        second=solve(self.problem,seconds=0,resume=cp,options=OPTIONS)
        self.assertEqual(second['stats']['parallel_replayed_jobs'],1)
        second['checkpoint']['jobs'][0]['region'][0]='0x0'
        self.assertNotEqual(first['checkpoint']['jobs'][0]['region'][0],'0x0')
    def test_committed_short_slice_emits_live_partial_board(self):
        problem=self.problem;events=[]
        def progress(event):
            if event['event'] in ('parallel_progress','parallel_checkpoint'):events.append(event)
        # A zero-budget native slice deterministically returns its unconsumed
        # checkpoint, exercising progress readback without timing assumptions.
        with patch('constraint_parallel.SLICE_SECONDS',0):
            result=solve(problem,seconds=5,workers=1,resume=legacy_root(problem),options=OPTIONS,
                         progress=progress,stop=lambda:bool(events))
        self.assertEqual(result['status'],'stopped',result.get('error'))
        partial=next(event for event in events if event.get('is_partial'))
        self.assertEqual(len(partial['codes']),problem.n)
        self.assertEqual(partial['assigned_tiles'],sum(code>=0 for code in partial['codes']))
        self.assertGreater(partial['parallel']['pending_jobs'],0)
        self.assertEqual(result['checkpoint']['inflight_jobs'],[])

    def test_worker_progress_never_exposes_single_job_checkpoint(self):
        events=[]
        with patch('constraint_parallel._worker',checkpoint_emitting_worker):
            result=solve(self.problem,seconds=5,workers=1,resume=legacy_root(self.problem),options=OPTIONS,
                         progress=events.append,stop=lambda:any(e['event']=='progress' for e in events))
        self.assertEqual(result['status'],'stopped',result.get('error'))
        self.assertTrue(any(e['event']=='progress' for e in events))
        self.assertTrue(all('checkpoint' not in e or e['checkpoint'].get('parallel_version')==1 for e in events))
        self.assertGreaterEqual(len(result['checkpoint']['jobs']),2)
        seen=[]
        final=solve(self.problem,seconds=10,workers=2,resume=result['checkpoint'],options=OPTIONS,
                    progress=lambda e:seen.append(tuple(e['codes'])) if e['event']=='candidate' else None)
        self.assertEqual(final['status'],'infeasible',final.get('error'))
        self.assertEqual(len(seen),18);self.assertEqual(len(set(seen)),18)

    def test_fixed_placement_survives_partition_and_spawn(self):
        problem=SearchProblem(simple_tiles(),double_face(),fixed={0:0})
        checkpoint=legacy_root(problem)
        parts=partition_checkpoint(problem,checkpoint,parts=8,options=OPTIONS)
        self.assertTrue(all(child['stack'][0]['domains'][0]=='0x1' for child in parts))
        result=solve(problem,seconds=5,workers=2,resume=checkpoint,options=OPTIONS)
        self.assertEqual(result['status'],'solved',result.get('error'))
        self.assertEqual(result['codes'][0],0)
        self.assertTrue(problem.validate(result['codes'])['valid'])

    def test_default_options_spawn_independently_validates_gold(self):
        problem=SearchProblem(simple_tiles(),double_face())
        result=solve(problem,seconds=5,workers=2,resume=legacy_root(problem),options={'native':'auto'})
        self.assertEqual(result['status'],'solved',result.get('error'))
        self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertTrue(result['search_options']['strand_connectivity'])
        self.assertNotEqual(result['native_backend'],'not_started')

    def test_unstarted_cancelled_slice_does_not_report_false_mixed_backend(self):
        from constraint_parallel import _record_backend
        backends,reasons=set(),set()
        _record_backend({'native_backend':'compiled','native_fallback_reason':None,
                         'stats':{'native_calls':227,'python_arcs_checked':0}},backends,reasons)
        _record_backend({'native_backend':'python','native_fallback_reason':'Propagation has not started.',
                         'stats':{'native_calls':227,'python_arcs_checked':0}},backends,reasons)
        self.assertEqual(backends,{'compiled'});self.assertEqual(reasons,set())
        _record_backend({'native_backend':'not_started','native_fallback_reason':'No work'},backends,reasons)
        self.assertEqual(backends,{'compiled'});self.assertEqual(reasons,set())
        # A real Python fallback still contributes to a genuinely mixed run.
        _record_backend({'native_backend':'python','native_fallback_reason':'Extension unavailable.'},backends,reasons)
        self.assertEqual(backends,{'compiled','python'})
        self.assertEqual(reasons,{'Extension unavailable.'})

    def test_worker_lru_is_bounded_and_private_but_support_is_reused(self):
        from constraint_dfs import _build_support_tables,_NogoodStore
        problem=self.problem;caches=_WorkerCaches(problem,limit=2)
        initial=tuple(sum(1<<code for code in row) for row in problem.domains)
        table=_build_support_tables(problem,initial,'gold',lambda:None)
        def request(job):return {'id':job,'region_key':str(job),'target':'gold','options':OPTIONS,'checkpoint':{}}
        one=caches.get(request(1));one['_support']=table
        one['_nogoods']=_NogoodStore(problem.fingerprint,'gold',initial_domains=initial)
        one['_runtime']=object();caches.remember(request(1),one)
        two=caches.get(request(2));self.assertIs(two['_support'],table)
        self.assertIsNot(two['_nogoods'],one['_nogoods']);self.assertNotIn('_runtime',two)
        self.assertIs(caches.get(request(1)),one)
        caches.get(request(3));self.assertEqual(len(caches.jobs),2)
        self.assertEqual(caches.stats['job_cache_evictions'],1)
        fresh=caches.get(request(2));self.assertIsNot(fresh,two)
        self.assertNotIn('_runtime',fresh);self.assertGreater(caches.stats['support_reuses'],0)
        changed=request(4);changed['checkpoint']['root_symmetry']={'state':'different'}
        self.assertNotIn('_support',caches.get(changed))
        problem.domains[0]=[0]
        with self.assertRaisesRegex(ValueError,'domains changed'):caches.get(request(1))
    def test_affinity_has_bounded_rotation_and_no_lost_jobs(self):
        queue=[1,2,3];owners={};last={};chosen=[]
        for _ in range(12):
            job,hit=_choose_job(queue,0,owners,last);chosen.append(job);queue.append(job)
        self.assertEqual(chosen[:AFFINITY_SLICES],[1]*AFFINITY_SLICES)
        self.assertNotEqual(chosen[AFFINITY_SLICES],1)
        self.assertEqual(sorted(queue),[1,2,3]);self.assertEqual(set(chosen),{1,2,3})
        self.assertLessEqual(max(len(list(group)) for _,group in itertools.groupby(chosen)),AFFINITY_SLICES)
    def test_warm_cache_hits_survive_actual_spawn_slices_and_stop(self):
        events=[]
        with patch('constraint_parallel.SLICE_SECONDS',0):
            result=solve(self.problem,seconds=8,workers=1,resume=legacy_root(self.problem),options=OPTIONS,
                         progress=events.append,stop=lambda:sum(e['event']=='parallel_progress' for e in events)>=6)
        self.assertEqual(result['status'],'stopped',result.get('error'))
        self.assertGreaterEqual(result['stats']['parallel_job_cache_hits'],3)
        self.assertGreaterEqual(result['stats']['parallel_affinity_hits'],3)
        self.assertEqual(result['checkpoint']['inflight_jobs'],[])
    def test_verified_global_knowledge_survives_spawn_and_restart(self):
        from constraint_knowledge import KnowledgeStore,load
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        options={**OPTIONS,'persistent_learning':True};events=[]
        result=solve(problem,seconds=8,workers=2,resume=legacy_root(problem),options=options,
                     learning_state={'_knowledge':store},progress=events.append)
        self.assertEqual(result['status'],'solved',result.get('error'))
        self.assertTrue(problem.validate(result['codes'])['valid'])
        restored=load(problem,json.loads(json.dumps(result['knowledge'])))
        self.assertGreaterEqual(len(restored),1)
        self.assertGreaterEqual(result['stats']['shared_loop_witnesses'],1)
        self.assertEqual(result['stats']['loop_witnesses'],len(result['knowledge']['records']))
        again=solve(problem,seconds=8,workers=1,resume=legacy_root(problem),options=options,
                    learning_state={'_knowledge':restored})
        self.assertEqual(again['status'],'solved',again.get('error'))
        self.assertTrue(problem.validate(again['codes'])['valid'])
        for event in events:
            if 'knowledge' in event:load(problem,event['knowledge'])

    def test_invalid_worker_witness_keeps_the_whole_durable_queue(self):
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem();options={**OPTIONS,'persistent_learning':True}
        with patch('constraint_parallel._worker',forged_knowledge_worker):
            result=solve(problem,seconds=5,workers=1,resume=legacy_root(problem),options=options)
        self.assertEqual(result['status'],'error');self.assertFalse(result['complete'])
        self.assertIn('closed proper',result['error'])
        self.assertEqual(result['knowledge']['records'],[])
        self.assertGreaterEqual(len(result['checkpoint']['jobs']),2)
        final=solve(problem,seconds=8,workers=2,resume=result['checkpoint'],options=options)
        self.assertEqual(final['status'],'solved',final.get('error'))
        self.assertTrue(problem.validate(final['codes'])['valid'])

    def test_invalid_controls_fail_before_process_start(self):
        for options in ({'workers':0},{'workers':65},{'workers':True},{'seconds':-1},{'seconds':float('inf')},{'target':'wrong'}):
            with self.subTest(options=options),self.assertRaises(ValueError):solve(self.problem,**options)

class ParallelSharedConflictTests(unittest.TestCase):
    def setUp(self):
        self.problem=SearchProblem(simple_tiles(),double_face())
        self.options={**OPTIONS,'shared_conflicts':True}
    def test_disabled_feature_allocates_no_shared_store_and_filters_raw_payloads(self):
        from constraint_parallel import _progress_fields,_add_stats
        with patch('constraint_parallel.VerifiedConflictStore',side_effect=AssertionError('Disabled')):
            result=solve(self.problem,seconds=0,options=OPTIONS)
        self.assertNotIn('shared_conflicts',result)
        self.assertNotIn('shared_conflicts',_progress_fields({'shared_conflicts':{'unverified':True},'codes':[0,3]}))
        retired={};_add_stats(retired,{'shared_conflicts_entries':3,'shared_conflicts_pending':4,
                                    'shared_conflicts_clauses':50,'shared_conflicts_calls':2,
                                    'hierarchical_subtable_states':20,'hierarchical_subtable_support_slots':40,
                                    'hierarchical_cached_summaries':25},retiring=True)
        self.assertEqual(retired,{'shared_conflicts_calls':2})
    def test_wrong_original_scope_is_rejected_before_spawn(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        store=VerifiedConflictStore(self.problem,'edges')
        try:
            with self.assertRaisesRegex(ValueError,'original puzzle'):
                solve(self.problem,seconds=0,options=self.options,learning_state={'_shared_conflicts':store})
        finally:store.close()
    def test_worker_reuses_global_verifier_but_not_private_job_learning(self):
        import constraint_parallel as parallel
        from test_constraint_shared_conflicts import snapshot_claims,finish
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional SAT package unavailable')
        requests=[{'id':job,'region_key':str(job),'target':'gold','options':self.options,
                   'checkpoint':{},'seconds':.1,'shared_conflicts':snapshot_claims(self.problem,[[[0,0],[1,0]]]) if job==1 else None}
                  for job in (1,2)]
        class Connection:
            def __init__(self):self.items=iter([*requests,None]);self.sent=[]
            def recv(self):return next(self.items)
            def send(self,item):self.sent.append(item)
            def close(self):pass
        class Cancel:
            def is_set(self):return False
        connection=Connection();states=[];stores=[]
        def fake_solve(problem,**kwargs):
            state=kwargs['learning_state'];states.append(state);store=state['_shared_conflicts'];stores.append(store)
            self.assertNotIn('_private_test',state);state['_private_test']=object()
            finish(store)
            return {'status':'timeout','nodes':0,'stats':{}}
        with patch.object(parallel,'dfs_solve',side_effect=fake_solve):parallel._worker(self.problem,connection,Cancel())
        self.assertEqual(len(states),2);self.assertIsNot(states[0],states[1]);self.assertIs(stores[0],stores[1])
        self.assertEqual(len(stores[0]),1)
        self.assertTrue(all(message['result']['shared_conflicts']['records'] for message in connection.sent))
    def test_spawn_rejects_satisfiable_worker_claim_and_never_forwards_it(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional SAT package unavailable')
        store=VerifiedConflictStore(self.problem);events=[]
        try:
            with patch('constraint_parallel._worker',claimed_conflict_worker):
                result=solve(self.problem,seconds=5,workers=1,resume=legacy_root(self.problem),options=self.options,
                    learning_state={'_shared_conflicts':store},progress=events.append,
                    stop=lambda:store.stats['shared_conflicts_rejected']>0)
            self.assertEqual(result['status'],'stopped',result.get('error'))
            self.assertEqual(result['shared_conflicts']['records'],[])
            self.assertTrue(all(not e.get('shared_conflicts',{}).get('records') for e in events))
            self.assertGreaterEqual(len(result['checkpoint']['jobs']),2)
            self.assertEqual(result['stats']['shared_conflicts_rejected'],1)
            resumed=solve(self.problem,seconds=5,workers=2,resume=result['checkpoint'],options=self.options)
            self.assertEqual(resumed['status'],'solved',resumed.get('error'))
            self.assertTrue(self.problem.validate(resumed['codes'])['valid'])
        finally:store.close()
    def test_spawn_coordinator_reproves_valid_claim_and_restart_reverifies(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional SAT package unavailable')
        store=VerifiedConflictStore(self.problem);events=[]
        try:
            with patch('constraint_parallel._worker',valid_conflict_worker):
                result=solve(self.problem,seconds=5,workers=1,resume=legacy_root(self.problem),options=self.options,
                    learning_state={'_shared_conflicts':store},progress=events.append,stop=lambda:len(store)>0)
            self.assertEqual(result['status'],'stopped',result.get('error'))
            self.assertEqual(len(result['shared_conflicts']['records']),1)
            self.assertGreater(result['stats']['shared_conflicts_calls'],0)
            # The first forwarded progress snapshot cannot contain the just
            # received worker claim; verification happens on a later pass.
            first=next(e for e in events if e['event']=='progress')
            self.assertEqual(first['shared_conflicts']['records'],[])
            restored=VerifiedConflictStore(self.problem)
            try:
                restored.import_snapshot(json.loads(json.dumps(result['shared_conflicts'])))
                self.assertEqual(len(restored),0);self.assertEqual(restored.pending_count,1)
                finish(restored);self.assertEqual(len(restored),1)
                final=solve(self.problem,seconds=5,workers=2,resume=result['checkpoint'],options=self.options,
                            learning_state={'_shared_conflicts':restored})
                self.assertEqual(final['status'],'solved',final.get('error'))
                self.assertTrue(self.problem.validate(final['codes'])['valid'])
                self.assertEqual(final['stats']['shared_conflicts_entries'],len(restored))
            finally:restored.close()
        finally:store.close()
    def test_malformed_worker_snapshot_errors_without_losing_queue(self):
        with patch('constraint_parallel._worker',malformed_conflict_worker):
            result=solve(self.problem,seconds=5,workers=1,resume=legacy_root(self.problem),options=self.options)
        self.assertEqual(result['status'],'error');self.assertFalse(result['complete'])
        self.assertIn('integrity',result['error']);self.assertEqual(result['shared_conflicts']['records'],[])
        self.assertGreaterEqual(len(result['checkpoint']['jobs']),2)
        final=solve(self.problem,seconds=5,workers=2,resume=result['checkpoint'],options=self.options)
        self.assertEqual(final['status'],'solved',final.get('error'))

if __name__=='__main__':unittest.main()
