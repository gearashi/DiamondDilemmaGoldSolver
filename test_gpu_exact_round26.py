"""Independent finite and lifecycle checks for round26 GPU exact reasoning."""
import base64,copy,hashlib,itertools,os,unittest
from unittest.mock import patch
import numpy as np
from gpu_reasoning_round26 import SUPPORT_COUNTERS,validate_support_policy
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import solve
from test_gpu_domain_round24 import enumerate_engine
from test_gpu_domain_round25 import independent_solutions,precise_reason_fixture,word_boundary_fixture,planted160_fixture,fork_gold_fixture
from test_constraint_dfs import simple_tiles,tetrahedron_tiles,double_face
from search_problem import SearchProblem

class SupportWitnessHostTests(unittest.TestCase):
    def test_bounded_configuration_rejects_before_native_import(self):
        for value in (0,1,None,'yes'):
            with self.assertRaises(ValueError):validate_support_policy(value,False,None)
        for slots in (0,1,7,9,16384,True):
            with self.assertRaises(ValueError):validate_support_policy(True,False,slots)
        self.assertEqual(validate_support_policy(True,False,8),(True,8))
        self.assertEqual(validate_support_policy(False,True,None),(True,1024))
        with self.assertRaises(ValueError):validate_support_policy(False,True,8)
        with self.assertRaisesRegex(ValueError,'budget'):
            DomainCUDA(SearchProblem(*tetrahedron_tiles()),support_reuse=True,max_memory_bytes=1)
    def test_stop_without_device_and_boolean_runner_scope(self):
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(SearchProblem(*tetrahedron_tiles()),stop=lambda:True,options={'gpu_support_reuse':True})
        factory.assert_not_called();self.assertEqual(result['status'],'stopped')
        self.assertTrue(all(result['stats'][key]==0 for key in SUPPORT_COUNTERS))
        with self.assertRaises(ValueError):solve(SearchProblem(*tetrahedron_tiles()),options={'gpu_support_reuse':1})

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class SupportWitnessCUDATests(unittest.TestCase):
    def test_complete_solution_sets_and_work_saved_across_existing_policies(self):
        problem=SearchProblem(*tetrahedron_tiles());expected=independent_solutions(problem,'gold');saved=0;hits=0
        for joint,event,precise in itertools.product((False,True),repeat=3):
            records=[]
            for reuse in (False,True):
                engine=DomainCUDA(problem,target='gold',joint_support=joint,event_propagation=event,precise_reasons=precise,support_reuse=reuse,support_metrics=True)
                try:
                    found=enumerate_engine(engine,31);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected));records.append(engine.stats())
                finally:engine.close()
            self.assertEqual(records[0]['decisions'],records[1]['decisions']);self.assertEqual(records[0]['domain_removals'],records[1]['domain_removals'])
            saved+=records[0]['support_word_scans']-records[1]['support_word_scans'];hits+=records[1]['support_cache_hits']
        self.assertGreater(saved,0);self.assertGreater(hits,0)
    def test_cache_eviction_word_boundaries_and_learning_disabled(self):
        evictions=0
        for problem in (precise_reason_fixture(),word_boundary_fixture()):
            expected=independent_solutions(problem,'edges')
            for slots in (8,1024):
                engine=DomainCUDA(problem,target='edges',support_reuse=True,support_slots=slots,joint_support=True,learning=False,incremental_strands=False)
                try:
                    found=enumerate_engine(engine,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                    evictions+=engine.stats()['support_cache_evictions']
                finally:engine.close()
        self.assertGreater(evictions,0)
    def test_each_primitive_resume_retains_native_witnesses_and_remaining_work(self):
        problem=SearchProblem(simple_tiles(),double_face());expected=independent_solutions(problem,'gold');seen=set()
        options=dict(support_reuse=True,support_slots=8,joint_support=True,event_propagation=True,precise_reasons=True)
        engine=DomainCUDA(problem,**options)
        try:
            for tick in range(1000):
                state=engine.step(1)
                for item in engine.candidates():seen.add(tuple(item['codes']));engine.acknowledge([item['lane']])
                snapshot=engine.checkpoint();other=DomainCUDA(problem,**options)
                try:other.restore(snapshot);np.testing.assert_array_equal(other.state.get(),snapshot['state']);self.assertEqual(engine.remaining_domain_cubes(),other.remaining_domain_cubes())
                except BaseException:other.close();raise
                engine.close();engine=other
                if all(state==2):break
            else:self.fail('Tiny resumed enumeration stalled')
            self.assertEqual(seen,expected)
        finally:engine.close()
    def test_malformed_cache_indices_rejected_and_valid_false_witness_rechecked(self):
        problem=SearchProblem(*tetrahedron_tiles());engine=DomainCUDA(problem,support_reuse=True)
        try:
            for _ in range(8):engine.step(1)
            saved=engine.checkpoint()
            for field,value in [('support_keys',4*problem.n*engine.nc+1),('support_values',engine.nc)]:
                bad=copy.deepcopy(saved);bad['state'][engine.offsets[field],0]=value;bad['state_sha256']=hashlib.sha256(memoryview(bad['state']).cast('B')).hexdigest()
                with self.assertRaisesRegex(ValueError,'support witness'):engine.restore(bad)
            # In-range cache corruption is never proof authority: every hit is
            # checked against immutable tables/current domains before success.
            altered=copy.deepcopy(saved);altered['state'][engine.offsets['support_values']:engine.offsets['support_values']+3*engine.support_slots]=0
            altered['state_sha256']=hashlib.sha256(memoryview(altered['state']).cast('B')).hexdigest();engine.restore(altered)
            self.assertEqual(set(enumerate_engine(engine,31)),independent_solutions(problem,'gold'))
        finally:engine.close()
    def test_full160_fork_rebase_and_lane_replacement_preserve_scope(self):
        problem=fork_gold_fixture();parent=DomainCUDA(problem,target='gold',support_reuse=True,joint_support=True,event_propagation=True)
        child=None
        try:
            for _ in range(1000):
                _,done=parent.propagate_root(17)
                if done:break
            self.assertTrue(done)
            from gpu_domain_runner import split_cubes
            cubes=split_cubes(parent.remaining_domain_cubes(),2);child=parent.fork_root(cubes)
            self.assertTrue(child.support_reuse);self.assertEqual(set(enumerate_engine(child,31)),independent_solutions(problem,'gold'))
        finally:
            parent.close()
            if child is not None:child.close()
        problem=planted160_fixture();engine=DomainCUDA(problem,support_reuse=True,joint_support=True,event_propagation=True)
        try:
            for _ in range(2000):
                states=engine.step(97)
                if engine.candidates():break
            self.assertTrue(engine.candidates());self.assertTrue(engine.candidates()[0]['validation']['valid'])
        finally:engine.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class WatchedClauseCUDATests(unittest.TestCase):
    def test_complete_solutions_and_saved_clause_literal_work(self):
        saved=0
        for problem in (SearchProblem(*tetrahedron_tiles()),precise_reason_fixture()):
            for event in (False,True):
                records=[];expected=independent_solutions(problem,'gold')
                for watched in (False,True):
                    engine=DomainCUDA(problem,event_propagation=event,precise_reasons=True,support_reuse=True,watched_clauses=watched,clause_metrics=True)
                    try:
                        found=enumerate_engine(engine,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected));records.append(engine.stats())
                    finally:engine.close()
                saved+=records[0]['clause_literal_checks']-records[1]['clause_literal_checks']
        self.assertGreater(saved,0)
    def test_each_primitive_checkpoint_watch_index_and_ring_lifecycle(self):
        problem=precise_reason_fixture();engine=DomainCUDA(problem,watched_clauses=True,event_propagation=True,precise_reasons=True)
        try:
            phases=set()
            for tick in range(4000):
                state=engine.step(1);saved=engine.checkpoint();phases.add(int(saved['state'][6,0]))
                replacement=DomainCUDA(problem,watched_clauses=True,event_propagation=True,precise_reasons=True)
                try:replacement.restore(saved);np.testing.assert_array_equal(replacement.state.get(),saved['state'])
                except BaseException:replacement.close();raise
                engine.close();engine=replacement
                if all(state==2):break
            else:self.fail('Watched finite enumeration stalled')
            self.assertTrue({0,1,2,3}<=phases);self.assertEqual(engine.candidates(),[])
            self.assertGreater(engine.stats()['clause_rescans'],0)
        finally:engine.close()
    def test_imported_original_root_clauses_and_fork_rebase(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        problem=SearchProblem(*tetrahedron_tiles());expected=independent_solutions(problem,'gold')
        store=VerifiedConflictStore(problem,'gold');engine=DomainCUDA(problem,watched_clauses=True,event_propagation=True,support_reuse=True)
        try:
            # Two placements of the same physical piece are an original-root
            # inventory conflict, independently established by the native SAT verifier.
            self.assertTrue(store.offer(((0,0),(1,1))))
            for _ in range(20):
                store.advance(lambda:None,seconds=.05,attempts=1,conflicts=1024)
                if store.records():break
            self.assertTrue(store.records());engine.step(7)
            self.assertGreater(engine.import_verified_clauses(store)['inserted'],0)
            self.assertEqual(set(enumerate_engine(engine,31)),expected)
        finally:store.close();engine.close()
    def test_malformed_consumer_index_rejected(self):
        problem=SearchProblem(*tetrahedron_tiles());engine=DomainCUDA(problem,watched_clauses=True)
        try:
            saved=engine.checkpoint();saved['state'][engine.offsets['watch_cells'],0]=1
            saved['state_sha256']=hashlib.sha256(memoryview(saved['state']).cast('B')).hexdigest()
            with self.assertRaisesRegex(ValueError,'consumer index'):engine.restore(saved)
        finally:engine.close()


def inventory_fixture(kind='filter'):
    from test_constraint_dfs import board
    n=6;problem=SearchProblem([{'id':str(i),'segments':[[[0,6],[1,6]]]} for i in range(n)],board(n,[(cell,1,(cell+1)%n,0) for cell in range(n)]))
    if kind=='failure':problem.domains=[[0,3] if cell%2==0 else [6,9,12,15] for cell in range(n)]
    elif kind=='components':problem.domains=[[0,3]]*2+[[6,9]]*2+[[12,15]]*2
    else:problem.domains=[[0,3],[0,3,6],[6,9],[0,3],[12,15],[12,15]]
    return problem

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class GlobalMatchingCUDATests(unittest.TestCase):
    def test_global_hall_and_unsupported_edges_preserve_all_independent_answers(self):
        for kind in ('failure','filter','components'):
            problem=inventory_fixture(kind);expected=independent_solutions(problem,'edges');records=[]
            for enabled in (False,True):
                engine=DomainCUDA(problem,target='edges',global_matching=enabled,joint_support=True,event_propagation=True,watched_clauses=True,precise_reasons=True,support_reuse=True)
                try:
                    found=enumerate_engine(engine,71);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected));records.append(engine.stats())
                finally:engine.close()
            if kind=='failure':
                self.assertEqual(records[1]['decisions'],0);self.assertGreater(records[0]['decisions'],0);self.assertGreater(records[1]['matching_hall_failures'],0)
            if kind=='filter':self.assertGreaterEqual(records[1]['matching_exclusions'],3)
            if kind=='components':self.assertLess(records[1]['matching_rows_rebuilt'],problem.n*records[1]['matching_runs'])
    def test_existing_seams_gold_and_disabled_learning_matrix(self):
        for problem in (SearchProblem(*tetrahedron_tiles()),precise_reason_fixture(),word_boundary_fixture()):
            for target in ('edges','gold'):
                expected=independent_solutions(problem,target)
                for event,learning in itertools.product((False,True),repeat=2):
                    engine=DomainCUDA(problem,target=target,global_matching=True,event_propagation=event,learning=learning,joint_support=True,support_reuse=True,watched_clauses=True)
                    try:
                        found=enumerate_engine(engine,71);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                    finally:engine.close()
    def test_every_matching_primitive_resume_and_remaining_solution_ownership(self):
        problem=inventory_fixture();expected=independent_solutions(problem,'edges');seen=set();stages=set()
        options=dict(target='edges',global_matching=True,event_propagation=True,precise_reasons=True,watched_clauses=True)
        engine=DomainCUDA(problem,**options);other=DomainCUDA(problem,**options)
        try:
            for tick in range(6000):
                states=engine.step(1)
                for item in engine.candidates():seen.add(tuple(item['codes']));engine.acknowledge([item['lane']])
                saved=engine.checkpoint()
                if saved['state'][6,0]==5:stages.add(int(saved['state'][engine.offsets['matching_meta'],0]))
                other.restore(saved);np.testing.assert_array_equal(other.state.get(),saved['state'])
                cubes=other.remaining_domain_cubes();remaining={row for row in expected if any(all(mask&(1<<code) for mask,code in zip(cube,row)) for cube in cubes)}
                self.assertEqual(remaining,expected-seen);engine,other=other,engine
                if all(states==2):break
            else:self.fail('Incremental matching primitive run stalled')
            self.assertEqual(seen,expected);self.assertEqual(stages,set(range(7)))
        finally:engine.close();other.close()
    def test_full160_controlled_witness_and_invalid_matching_state(self):
        problem=planted160_fixture();engine=DomainCUDA(problem,global_matching=True,event_propagation=True,watched_clauses=True)
        try:
            for tick in range(1000):
                engine.step(257)
                if engine.candidates():break
            self.assertTrue(engine.candidates());self.assertTrue(engine.candidates()[0]['validation']['valid'])
            saved=engine.checkpoint()
            for field,value in [('matching_cells',engine.n),('matching_components',engine.n),('matching_meta',7)]:
                bad=copy.deepcopy(saved);bad['state'][engine.offsets[field],0]=value;bad['state_sha256']=hashlib.sha256(memoryview(bad['state']).cast('B')).hexdigest()
                with self.assertRaisesRegex(ValueError,'matching'):engine.restore(bad)
        finally:engine.close()


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class CooperativeCUDATests(unittest.TestCase):
    def test_scalar_warp_exact_state_and_all_answers_with_full_reasoning(self):
        from gpu_domain_runner import split_cubes
        for problem,target in ((SearchProblem(*tetrahedron_tiles()),'gold'),(word_boundary_fixture(),'edges'),(inventory_fixture(),'edges')):
            roots=[tuple(sum(1<<code for code in row) for row in problem.domains)];cubes=split_cubes(roots,4)
            options=dict(target=target,event_propagation=True,precise_reasons=True,global_matching=True,watched_clauses=True,support_reuse=True)
            left=DomainCUDA(problem,cubes,**options);right=DomainCUDA(problem,cubes,cooperative=True,**options);seen=set()
            try:
                self.assertTrue(right.state.flags.f_contiguous)
                compare=np.ones(left.rows,bool);compare[left.offsets['support_counters']+6:left.offsets['support_counters']+8]=False
                for tick in range(5000):
                    a=left.step(7);b=right.step(7);np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(left.state.get()[compare],right.state.get()[compare])
                    pending=left.candidates();self.assertEqual([x['codes'] for x in pending],[x['codes'] for x in right.candidates()])
                    if pending:
                        seen.update(tuple(x['codes']) for x in pending);ids=[x['lane'] for x in pending];left.acknowledge(ids);right.acknowledge(ids)
                    if tick%11==0:
                        saved=right.checkpoint();right.restore(saved);np.testing.assert_array_equal(saved['state'],right.state.get())
                    if all(a==2):break
                else:self.fail('Cooperative finite enumeration stalled')
                self.assertEqual(seen,independent_solutions(problem,target))
            finally:left.close();right.close()
    def test_fork_rebase_replacement_and_factored_resume(self):
        from gpu_domain_runner import split_cubes
        from gpu_checkpoint_codec_round25 import encode_native,decode_native
        problem=inventory_fixture('components')
        for cooperative in (False,True):
            options=dict(target='edges',event_propagation=True,global_matching=True,watched_clauses=True,support_reuse=True,cooperative=cooperative)
            parent=DomainCUDA(problem,**options);child=None;replacement=None
            try:
                for _ in range(1000):
                    _,done=parent.propagate_root(17)
                    if done:break
                self.assertTrue(done);cubes=split_cubes(parent.remaining_domain_cubes(),4);child=parent.fork_root(cubes)
                child.step(23);before=child.state.get();child.replace_lanes({0:cubes[0]})
                np.testing.assert_array_equal(before[:,1:],child.state.get()[:,1:])
                self.assertEqual(int(child.state[child.offsets['support_keys']:child.offsets['support_keys']+child.support_slots,0].sum()),0)
                self.assertTrue(np.all(child.state[child.offsets['watch_positions']:child.offsets['watch_positions']+64,0].get()==0xffffffff))
                checkpoint=child.checkpoint();encoded,blobs,_=encode_native(child,checkpoint)
                for column in encoded['columns']:column['data']=base64.b64encode(blobs[column['sha256']]).decode('ascii')
                replacement=DomainCUDA(problem,cubes,**options);replacement.restore(decode_native(encoded,replacement))
                self.assertEqual(set(enumerate_engine(replacement,71)),independent_solutions(problem,'edges'))
            finally:
                parent.close()
                if child is not None:child.close()
                if replacement is not None:replacement.close()
    def test_full160_thread_warp_equal_gold(self):
        problem=planted160_fixture();records=[]
        for cooperative in (False,True):
            engine=DomainCUDA(problem,global_matching=True,event_propagation=True,support_reuse=True,watched_clauses=True,cooperative=cooperative)
            try:
                for _ in range(1000):
                    engine.step(257)
                    if engine.candidates():break
                self.assertTrue(engine.candidates()[0]['validation']['valid']);records.append(engine.candidates()[0]['codes'])
            finally:engine.close()
        self.assertEqual(*records)


class StructuralProducerHostTests(unittest.TestCase):
    def test_real_hall_certificate_and_conservative_bounded_unknown(self):
        from gpu_deduction_round26 import produce
        from constraint_deduction_chain_round26 import verify_chain
        problem=inventory_fixture('failure');root=[sum(1<<v for v in row) for row in problem.domains]
        certificate,stats=produce(problem,'edges',root,[],[])
        self.assertEqual(verify_chain(problem,certificate,target='edges'),());self.assertEqual(stats['gpu_proof_certificates'],1)
        certificate,stats=produce(problem,'edges',root,[],[],max_work=1)
        self.assertIsNone(certificate);self.assertEqual(stats['gpu_proof_cap_hits'],1)
        problem=inventory_fixture('components');root=[sum(1<<v for v in row) for row in problem.domains]
        certificate,stats=produce(problem,'edges',root,[],[0])
        self.assertIsNone(certificate);self.assertEqual(stats['gpu_proof_unknown'],1)
    def test_stop_does_not_publish_or_commit_dedup(self):
        from gpu_shared_round25 import GPUClauseExchange,SharingStopped
        from types import SimpleNamespace
        problem=inventory_fixture('failure');root=[sum(1<<v for v in row) for row in problem.domains]
        trace={'root':root,'decisions':[],'trail':[],'identity':'0'*64,'next':0}
        engine=SimpleNamespace(proof_transport=True,deduction_input=lambda start:trace)
        exchange=GPUClauseExchange(problem,'edges');calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==3:raise SharingStopped()
        try:
            with self.assertRaises(SharingStopped):exchange._structural(engine,stop)
            self.assertEqual(exchange.proof_seen,[]);self.assertEqual(exchange.store.records(),[])
            exchange._structural(engine,lambda:None);self.assertEqual(exchange.store.records(),[[]])
            saved=exchange.snapshot();other=GPUClauseExchange(problem,'edges',resume=saved)
            try:
                with patch('gpu_deduction_round26.produce',side_effect=AssertionError('Repeated completed proof work')):other._structural(engine,lambda:None)
                self.assertEqual(other.store.records(),[[]]);self.assertEqual(other.counters['gpu_proof_attempts'],saved['counters']['gpu_proof_attempts'])
                other._structural(engine,lambda:None);self.assertEqual(other.counters['gpu_proof_duplicate_skips'],1)
            finally:other.close()
        finally:exchange.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class StructuralProofCUDATests(unittest.TestCase):
    def test_native_hall_proof_admits_without_sat_queries(self):
        from gpu_shared_round25 import GPUClauseExchange
        problem=inventory_fixture('failure');engine=DomainCUDA(problem,target='edges',global_matching=True,proof_transport=True);exchange=GPUClauseExchange(problem,'edges')
        try:
            for _ in range(1000):
                state=engine.step(17)
                if all(state==2):break
            with patch.object(exchange,'_verify',side_effect=AssertionError('Structural proof must not query SAT')):
                self.assertTrue(exchange.sync(engine))
            self.assertEqual(exchange.store.records(),[[]]);self.assertEqual(exchange.counters['gpu_proof_admitted'],1)
            self.assertGreater(exchange.counters['gpu_proof_checked_hall_placements'],0)
        finally:engine.close();exchange.close()
    def test_seam_hall_gold_failed_traces_preserve_every_actual_solution(self):
        from gpu_deduction_round26 import produce
        from constraint_deduction_chain_round26 import verify_chain
        from test_constraint_dfs import multiple_loops
        certificates=0;proper_loops=0;backjump_saved=False
        for problem in (SearchProblem(*tetrahedron_tiles()),precise_reason_fixture(),SearchProblem(multiple_loops(),double_face())):
            expected=independent_solutions(problem,'gold');engine=DomainCUDA(problem,global_matching=problem.n!=6,event_propagation=True,proof_transport=True,precise_reasons=True)
            try:
                for _ in range(12000):
                    states=engine.step(1)
                    trace=engine.deduction_input() if not engine.delivered else None
                    if trace:
                        certificate,_=produce(problem,'gold',trace['root'],trace['decisions'],trace['trail'])
                        if certificate is not None:
                            guards=verify_chain(problem,certificate);certificates+=1;proper_loops+=int(certificate['terminal']['rule']=='loop')
                            self.assertFalse(any(all(mask&(1<<answer[c]) for c,mask in guards) for answer in expected))
                        saved=engine.checkpoint()
                        if saved['state'][engine.offsets['proof_meta'],0]>saved['state'][1,0]:
                            from gpu_checkpoint_codec_round25 import encode_native,decode_native
                            spec,blobs,_=encode_native(engine,saved)
                            for column in spec['columns']:column['data']=base64.b64encode(blobs[column['sha256']]).decode('ascii')
                            engine.restore(decode_native(spec,engine));self.assertEqual(trace['identity'],engine.deduction_input()['identity']);backjump_saved=True
                    pending=engine.candidates()
                    if pending:engine.acknowledge([item['lane'] for item in pending])
                    if all(states==2):break
                else:self.fail('Proof-producing finite search stalled')
            finally:engine.close()
        self.assertGreater(certificates,0);self.assertGreater(proper_loops,0);self.assertTrue(backjump_saved)
    def test_verified_insertion_at_every_matching_stage_waits_for_clause_pass(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        problem=inventory_fixture('components');store=VerifiedConflictStore(problem,'edges')
        # This native-independent Hall certificate proves a duplicate physical tile.
        from gpu_deduction_round26 import produce
        root=[sum(1<<v for v in row) for row in problem.domains]
        certificate,_=produce(problem,'edges',root,[(0,1),(1,1)],[]);self.assertGreater(store.admit_deduction_chain(certificate),0)
        try:
            for watched in (False,True):
                for stage in range(7):
                    engine=DomainCUDA(problem,target='edges',global_matching=True,watched_clauses=watched,event_propagation=True)
                    try:
                        for _ in range(2000):
                            engine.propagate_root(1)
                            if int(engine.state[6,0])==5 and int(engine.state[engine.offsets['matching_meta'],0])==stage:break
                        else:self.fail('Matching stage not reached')
                        self.assertGreater(engine.import_verified_clauses(store)['inserted'],0)
                        for _ in range(2000):
                            _,done=engine.propagate_root(17)
                            if done:break
                        self.assertTrue(done);saved=engine.checkpoint();engine.restore(saved)
                        self.assertEqual(set(enumerate_engine(engine,71)),independent_solutions(problem,'edges'))
                    finally:engine.close()
        finally:store.close()


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round26 exact tests')
class ExactLifecycleCUDATests(unittest.TestCase):
    def test_original_proofs_fill_and_overwrite_actual_watch_ring(self):
        from gpu_deduction_round26 import produce
        from constraint_shared_conflicts import VerifiedConflictStore
        problem=SearchProblem(*tetrahedron_tiles());root=[sum(1<<v for v in row) for row in problem.domains]
        claims=[[(a,1<<(3*tile+ra)),(b,1<<(3*tile+rb))] for a in range(4) for b in range(a+1,4) for tile in range(4) for ra in range(3) for rb in range(3)][:64]
        engine=DomainCUDA(problem,watched_clauses=True,event_propagation=True,global_matching=True,cooperative=True)
        try:
            for start in range(0,64,8):
                store=VerifiedConflictStore(problem,'gold')
                try:
                    for claim in claims[start:start+8]:
                        cert,_=produce(problem,'gold',root,claim,[]);self.assertIsNotNone(cert);self.assertEqual(store.admit_deduction_chain(cert),1)
                    self.assertEqual(engine.import_verified_clauses(store)['inserted'],8)
                    for _ in range(2000):
                        _,done=engine.propagate_root(31)
                        if done:break
                    self.assertTrue(done);engine.restore(engine.checkpoint())
                finally:store.close()
            self.assertEqual(int(engine.state[8,0]),32);self.assertEqual(int(engine.state[19,0]),0)
            self.assertEqual(set(enumerate_engine(engine,71)),independent_solutions(problem,'gold'))
        finally:engine.close()
    def test_policy_restart_preserves_owned_frontier_and_saved_payload(self):
        from gpu_domain_runner import pack_native,BASE_COUNTERS
        from constraint_work import WorkScope,frontier_snapshot,digest
        problem=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(problem)
        options={'gpu_support_reuse':True,'gpu_watched_clauses':True,'gpu_global_matching':True,'gpu_cooperative':True,'gpu_proof_transport':True}
        engine=DomainCUDA(problem,support_reuse=True,watched_clauses=True,global_matching=True,cooperative=True,proof_transport=True)
        try:
            engine.step(13);payload={'version':1,'scope':scope.payload(),'queue':[],'native':pack_native(engine),'totals':dict.fromkeys(BASE_COUNTERS,0),
                'completed_lanes':0,'active_stats':engine.stats(),'frontier':frontier_snapshot(scope,engine.remaining_domain_cubes())}
        finally:engine.close()
        payload['digest']=digest(payload);before=copy.deepcopy(payload)
        for changed in ({},options):
            result=solve(problem,seconds=10,replicas=1,resume=payload,options=changed)
            self.assertEqual(result['status'],'solved');self.assertTrue(result['validation']['valid']);self.assertEqual(payload,before)
            self.assertEqual(result['stats']['gpu_domain_portable_restart'],int(not changed))

class ExactLifecycleHostTests(unittest.TestCase):
    def test_policy_memory_and_unsupported_backend_fail_before_device(self):
        from gpu_reasoning_round26 import memory_estimate
        info=memory_estimate(160,512,joint_support=True,event_propagation=True,precise_reasons=True,support_reuse=True,watched_clauses=True,global_matching=True,proof_transport=True)
        self.assertEqual(info['state_bytes'],512*info['bytes_per_lane']);self.assertLess(info['state_bytes'],512*1024*1024)
        problem=SearchProblem(*tetrahedron_tiles())
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            for backend in ('webgpu','metal','rocm'):
                with self.assertRaisesRegex(ValueError,'CUDA'):solve(problem,backend=backend,options={'gpu_global_matching':True})
        factory.assert_not_called()
    def test_malformed_producer_inputs_are_bounded_rejections(self):
        from gpu_deduction_round26 import produce
        problem=inventory_fixture();root=[sum(1<<v for v in row) for row in problem.domains]
        for changed in ([-1]*problem.n,[True]*problem.n,root[:-1]):
            with self.assertRaises(ValueError):produce(problem,'edges',changed,[],[])
        with self.assertRaises(ValueError):produce(problem,'edges',root,[],[-1])


class ProofCacheHostTests(unittest.TestCase):
    def fixture(self):
        from gpu_deduction_round26 import produce
        problem=SearchProblem(*tetrahedron_tiles());root=[sum(1<<v for v in row) for row in problem.domains];root[2]=1<<6
        certificate,_=produce(problem,'gold',root,[(0,1),(1,1)],[]);self.assertIsNotNone(certificate)
        return problem,certificate
    def test_guard_generalization_and_durable_recheck_avoid_sat(self):
        from gpu_proof_cache_round26 import ProofCache
        from gpu_shared_round25 import COUNTERS
        from constraint_shared_conflicts import VerifiedConflictStore
        from constraint_deduction_chain_round26 import verify_chain
        problem,certificate=self.fixture();stats=dict.fromkeys(COUNTERS,0);store=VerifiedConflictStore(problem,'gold');cache=ProofCache(problem,'gold',stats)
        try:
            cache.offer('1'*64,certificate)
            while cache.pending():cache.advance(store,max_trials=1)
            self.assertEqual(verify_chain(problem,cache.records[0]['certificate']),((0,1),(1,1)))
            self.assertEqual(store.records(),[[[0,0],[1,0]]]);self.assertGreater(stats['gpu_proof_minimize_successes'],0)
            saved=cache.snapshot();other=VerifiedConflictStore(problem,'gold');other.import_snapshot(store.snapshot());restored=ProofCache(problem,'gold',stats,resume=saved)
            try:
                self.assertFalse(other.records());restored.advance(other)
                self.assertEqual(other.records(),store.records());self.assertEqual(other.stats['shared_conflicts_calls'],0)
                self.assertGreater(other.stats['shared_conflicts_pending_reused'],0)
            finally:other.close()
        finally:store.close()
    def test_stop_keeps_last_checked_certificate_and_pending_trial(self):
        from gpu_proof_cache_round26 import ProofCache
        from gpu_shared_round25 import COUNTERS,SharingStopped
        from constraint_shared_conflicts import VerifiedConflictStore
        problem,certificate=self.fixture();stats=dict.fromkeys(COUNTERS,0);store=VerifiedConflictStore(problem,'gold');cache=ProofCache(problem,'gold',stats)
        try:
            cache.offer('2'*64,certificate)
            def stop():
                if stats['gpu_proof_minimize_trials']:raise SharingStopped()
            with self.assertRaises(SharingStopped):cache.advance(store,stop)
            self.assertEqual(cache.records[0]['certificate'],certificate);self.assertEqual(len(store.records()[0]),3)
            cursor=cache.records[0]['cursor'];self.assertEqual(cursor,3)
            saved=cache.snapshot();restored=ProofCache(problem,'gold',stats,resume=saved)
            while restored.pending():restored.advance(store)
            self.assertEqual(store.records(),[[[0,0],[1,0]]]);self.assertEqual(cache.records[0]['cursor'],cursor)
        finally:store.close()
    def test_cache_eviction_and_corrupt_cursor_never_supply_authority(self):
        from gpu_proof_cache_round26 import ProofCache,MAX_RECORDS
        from gpu_shared_round25 import COUNTERS
        from constraint_work import digest
        problem,certificate=self.fixture();stats=dict.fromkeys(COUNTERS,0);cache=ProofCache(problem,'gold',stats)
        for number in range(MAX_RECORDS+2):cache.offer(f'{number:064x}',certificate)
        self.assertEqual(len(cache.records),MAX_RECORDS);self.assertEqual(stats['gpu_proof_cache_evictions'],2)
        saved=cache.snapshot();saved['records'][0]['cursor']=True;saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
        with self.assertRaises(ValueError):ProofCache(problem,'gold',stats,resume=saved)

if __name__=='__main__':unittest.main()
