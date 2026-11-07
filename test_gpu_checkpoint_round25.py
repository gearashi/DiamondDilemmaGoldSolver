"""Factored checkpoint integrity, transaction boundaries and CUDA continuation."""
import base64,copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from constraint_work import WorkScope,frontier_snapshot,digest
from gpu_checkpoint_codec_round25 import encode_native,decode_native,inline_native,_ranges,sha
from gpu_checkpoint_round25 import CheckpointStore,StoreCapacity,StoreFailure,resolve_checkpoint,checkpoint_frontier,is_reference
from gpu_domain_round24 import DomainCUDA
from test_gpu_domain_round25 import joint_hall_fixture,precise_reason_fixture,fork_gold_fixture,planted160_fixture,independent_solutions
from test_gpu_domain_round24 import enumerate_engine

def fixture():
    problem=joint_hall_fixture();scope=WorkScope.for_problem(problem,'edges');n=4;fields=[('meta',29),('domain',4),('trail',48),('reason',4),
        ('frame_cell',4),('frame_code',4),('frame_remaining',4),('frame_trail',4),('frame_union',4),('frame_assign',4),('frame_reason',16),('frame_failure',4),
        ('board',4),('owner',4),('assign_log',4),('parent',64),('size',64),('union_child',64),('union_parent',64),('union_size',64),
        ('clause_length',32),('clause_cell',128),('clause_code',128)]
    offsets={};cursor=0
    for name,count in fields:offsets[name]=cursor;cursor+=count
    engine=SimpleNamespace(n=n,words=1,reason_words=1,rows=cursor,lanes=2,offsets=offsets,schema='codec-unit',
        binding={'fingerprint':problem.fingerprint,'target':'edges','original':[hex(x) for x in scope.original]})
    data=np.arange(cursor*2,dtype=np.uint32).reshape(cursor,2)+10;data[1]=2;data[2]=3;data[9]=2;data[10]=3
    data[offsets['clause_length']:offsets['clause_length']+32]=np.arange(32,dtype=np.uint32)[:,None]%3
    snapshot={'schema':engine.schema,'binding':engine.binding,'cubes':[[hex(x) for x in scope.root]]*2,'rows':cursor,'delivered':[],'state':data,'state_sha256':sha(data.tobytes())}
    return engine,scope,snapshot

def payload(scope,native,epoch=1):
    value={'version':1,'scope':scope.payload(),'native':native,'queue':[],'totals':{'decisions':epoch},'completed_lanes':epoch,
           'active_stats':{},'frontier':frontier_snapshot(scope,[scope.root]),'epoch':epoch,'lane_pool':{'test_owner_generation':epoch}}
    value['digest']=digest(value);return value

class CheckpointCodecTests(unittest.TestCase):
    def test_live_ranges_roundtrip_and_unchanged_chunk_reuse(self):
        engine,scope,snapshot=fixture();cache={};spec,blobs,metrics=encode_native(engine,snapshot,cache)
        decoded=decode_native(inline_native(spec,blobs),engine)
        for lane in range(engine.lanes):
            mask=np.zeros(engine.rows,bool)
            for start,length in _ranges(engine,snapshot['state'],lane):mask[start:start+length]=True
            np.testing.assert_array_equal(decoded['state'][mask,lane],snapshot['state'][mask,lane]);self.assertFalse(np.any(decoded['state'][~mask,lane]))
        again,more,nextmetrics=encode_native(engine,decoded,cache)
        self.assertEqual(spec,again);self.assertEqual(blobs,more);self.assertEqual(nextmetrics['reused_chunks'],2)
        self.assertGreater(metrics['omitted_bytes'],0)
    def test_malformed_chunk_length_shape_and_compression_are_rejected(self):
        engine,scope,snapshot=fixture();spec,blobs,_=encode_native(engine,snapshot);inline=inline_native(spec,blobs)
        for mutate in (lambda x:x['shape'].__setitem__(0,2**31),lambda x:x['columns'][0].__setitem__('packed_bytes',1),lambda x:x['columns'][0].__setitem__('data','!bad')):
            bad=copy.deepcopy(inline);mutate(bad)
            with self.assertRaises(ValueError):decode_native(bad,engine)
        snapshot['state'][engine.offsets['clause_length'],0]=5
        with self.assertRaisesRegex(ValueError,'clause'):encode_native(engine,snapshot)
    def test_live_bounds_are_explicit_errors(self):
        engine,scope,snapshot=fixture();snapshot['state'][1,0]=999
        with self.assertRaises(ValueError):encode_native(engine,snapshot)

class CheckpointStoreTests(unittest.TestCase):
    def generations(self,directory):
        engine,scope,snapshot=fixture();store=CheckpointStore(directory,scope.payload());spec,blobs,_=encode_native(engine,snapshot,store.cache)
        first=store.write(payload(scope,spec,1),blobs);snapshot['state'][11,0]+=1
        spec2,blobs2,_=encode_native(engine,snapshot,store.cache);second=store.write(payload(scope,spec2,2),blobs2)
        return engine,scope,store,first,second,spec2,blobs2
    def test_delayed_old_consumer_and_metadata_only_frontier(self):
        with tempfile.TemporaryDirectory() as folder:
            engine,scope,store,first,second,_,_=self.generations(folder)
            with patch('gpu_checkpoint_codec_round25.zlib.decompressobj',side_effect=AssertionError('Metadata must not decompress')):
                self.assertEqual(checkpoint_frontier(first,folder),frontier_snapshot(scope,[scope.root]))
                old=resolve_checkpoint(first,folder);new=resolve_checkpoint(second,folder)
            self.assertEqual(old['epoch'],1);self.assertEqual(new['epoch'],2)
            self.assertEqual(new['lane_pool']['test_owner_generation'],2)
            self.assertEqual(int(decode_native(new['native'],engine)['state'][11,0])-int(decode_native(old['native'],engine)['state'][11,0]),1)
    def test_fresh_run_ignores_old_head_and_resume_chains_only_named_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            _,scope,_,first,second,spec,blobs=self.generations(folder)
            fresh=CheckpointStore(folder,scope.payload());third=fresh.write(payload(scope,spec,3),blobs)
            self.assertIsNone(third['previous']);self.assertEqual(resolve_checkpoint(second,folder)['epoch'],2)
            old=resolve_checkpoint(first,folder);resumed=CheckpointStore(folder,scope.payload(),previous=old['checkpoint_origin'])
            fourth=resumed.write(payload(scope,spec,4),blobs)
            self.assertEqual(fourth['previous'],first['generation']);self.assertNotEqual(fourth['previous'],third['generation'])
    def test_capacity_preserves_all_old_references_and_scopes(self):
        with tempfile.TemporaryDirectory() as folder:
            engine,scope,store,first,second,spec,blobs=self.generations(folder);before=sorted(p.name for p in (Path(folder)/'chunks').iterdir())
            store.max_bytes=store._usage()[0]+4096
            with self.assertRaises(StoreCapacity):store.write(payload(scope,spec,3),blobs)
            self.assertEqual(resolve_checkpoint(first,folder)['epoch'],1);self.assertEqual(resolve_checkpoint(second,folder)['epoch'],2)
            self.assertEqual(before,sorted(p.name for p in (Path(folder)/'chunks').iterdir()))
            with tempfile.TemporaryDirectory() as other:
                CheckpointStore(other,scope.payload())
                with self.assertRaisesRegex(ValueError,'lineage'):resolve_checkpoint(first,other)
    def test_failure_after_manifest_before_head_keeps_previous_generation(self):
        import gpu_checkpoint_round25 as module
        with tempfile.TemporaryDirectory() as folder:
            engine,scope,store,first,second,spec,blobs=self.generations(folder);original=module._atomic;old_head=(Path(folder)/'HEAD.json').read_bytes()
            def failing(path,raw):
                if path.name=='HEAD.json':raise OSError('injected journal publication failure')
                return original(path,raw)
            with patch.object(module,'_atomic',failing),self.assertRaises(StoreFailure):store.write(payload(scope,spec,3),blobs)
            self.assertEqual((Path(folder)/'HEAD.json').read_bytes(),old_head);self.assertEqual(resolve_checkpoint(second,folder)['epoch'],2)
    def test_corrupt_current_chunk_requires_explicit_whole_generation_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            engine,scope,store,first,second,spec,blobs=self.generations(folder)
            old=resolve_checkpoint(first,folder);old_keys={x['sha256'] for x in old['native']['columns']}
            unique=next(x['sha256'] for x in spec['columns'] if x['sha256'] not in old_keys)
            path=Path(folder)/'chunks'/(unique+'.bin');path.write_bytes(b'bad')
            with self.assertRaises(ValueError):resolve_checkpoint(second,folder)
            recovered=resolve_checkpoint(second,folder,allow_recovery=True)
            self.assertEqual(recovered['epoch'],1);self.assertEqual(recovered['lane_pool']['test_owner_generation'],1)
            self.assertEqual(recovered['checkpoint_recovery']['generations_rolled_back'],1)
            self.assertEqual(recovered['native'],old['native'])
    def test_intact_manifest_reference_mismatch_never_recovers_elsewhere(self):
        with tempfile.TemporaryDirectory() as folder:
            _,scope,_,first,second,spec,blobs=self.generations(folder)
            fresh=CheckpointStore(folder,scope.payload());third=fresh.write(payload(scope,spec,3),blobs)
            bad=copy.deepcopy(second);bad['previous']=third['generation'];bad['digest']=digest({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaisesRegex(ValueError,'reference metadata'):resolve_checkpoint(bad,folder,allow_recovery=True)
    def test_missing_manifest_can_use_only_reference_named_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,first,second,_,_=self.generations(folder)
            (Path(folder)/'manifests'/(second['generation']+'.json')).unlink()
            with self.assertRaises(ValueError):resolve_checkpoint(second,folder)
            self.assertEqual(resolve_checkpoint(second,folder,allow_recovery=True)['epoch'],1)
            bad=copy.deepcopy(second);bad['previous']='0'*64;bad['digest']=digest({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaises(ValueError):resolve_checkpoint(bad,folder,allow_recovery=True)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA factored checkpoint tests')
class CheckpointCUDATests(unittest.TestCase):
    def roundtrip(self,engine):
        snapshot=engine.checkpoint();spec,blobs,_=encode_native(engine,snapshot);decoded=decode_native(inline_native(spec,blobs),engine)
        replacement=DomainCUDA(engine.problem,engine.cubes,target=engine.target,learning=engine.learning,incremental_strands=engine.incremental_strands,
            joint_support=engine.joint_support,event_propagation=engine.event_propagation,precise_reasons=engine.precise_reasons,
            reason_slots=engine.reason_slots if engine.precise_reasons else None)
        try:replacement.restore(decoded)
        except BaseException:replacement.close();raise
        return replacement
    def test_every_phase_canonical_state_and_continuation_match(self):
        problem=joint_hall_fixture();engine=DomainCUDA(problem,target='edges',event_propagation=True,precise_reasons=True);checked_phases=set();seen=set()
        try:
            for tick in range(2000):
                phase=int(engine.state[6,0].get());other=self.roundtrip(engine);checked_phases.add(phase)
                try:
                    self.assertEqual(engine.stats(),other.stats());self.assertEqual(engine.remaining_domain_cubes(),other.remaining_domain_cubes())
                    engine.step(1);other.step(1)
                    left,_,_=encode_native(engine,engine.checkpoint());right,_,_=encode_native(other,other.checkpoint());self.assertEqual(left,right)
                finally:other.close()
                for item in engine.candidates():
                    answer=tuple(item['codes']);self.assertNotIn(answer,seen);seen.add(answer);engine.acknowledge([item['lane']])
                if engine.stats()['exhausted']:break
            else:self.fail('Bounded complete primitive checkpoint traversal did not finish')
            self.assertEqual(seen,independent_solutions(problem,'edges'));self.assertTrue(checked_phases>={0,1,2,3,4})
        finally:engine.close()
    def test_backjump_live_state_with_refill_and_verified_ring_is_preserved(self):
        from gpu_lane_pool_round25 import LanePool
        from gpu_domain_runner import split_cubes
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        problem=precise_reason_fixture();scope=WorkScope.for_problem(problem,'edges');pool=LanePool(scope,split_cubes([scope.root],12))
        engine=DomainCUDA(problem,pool.take_batch(4),target='edges',event_propagation=True,precise_reasons=True);checked=set()
        try:
            for _ in range(2000):
                states=engine.step(1);meta=engine.state[:7].get()
                marks=set(int(x) for x in meta[6])
                if 3 in marks and 'failure' not in checked:
                    replacement=self.roundtrip(engine)
                    try:self.assertEqual(set(enumerate_engine(replacement,17)),set())
                    finally:replacement.close()
                    checked.add('failure')
                if any(states==2):
                    pool.retire(engine,states)
                    if pool.refill(engine):
                        replacement=self.roundtrip(engine)
                        try:LanePool.restore(scope,pool.queue,replacement,pool.snapshot())
                        finally:replacement.close()
                        checked.add('refill')
                if checked=={'failure','refill'}:break
            self.assertEqual(checked,{'failure','refill'})
        finally:engine.close()
        from test_constraint_sat_assist import HAS_SAT
        if HAS_SAT:
            problem=joint_hall_fixture();store=VerifiedConflictStore(problem,'edges');store.offer([(0,0)]);finish(store)
            engine=DomainCUDA(problem,target='edges',event_propagation=True,precise_reasons=True)
            try:
                engine.import_verified_clauses(store);replacement=self.roundtrip(engine)
                try:self.assertEqual(set(enumerate_engine(replacement,17)),independent_solutions(problem,'edges'))
                finally:replacement.close()
            finally:engine.close();store.close()
    def test_stop_after_donation_roundtrips_entire_owner_generation(self):
        from gpu_domain_runner import solve
        from gpu_lane_pool_round25 import LanePool
        problem=precise_reason_fixture();flag=[False];donate=LanePool.donate;step=DomainCUDA.step
        def stop_after(pool,engine):
            value=donate(pool,engine)
            if value is not None:flag[0]=True
            return value
        options={'gpu_incremental_checkpoint':True,'gpu_lane_balance':True,'gpu_propagated_forks':True,'gpu_event_propagation':True,'gpu_precise_reasons':True}
        with tempfile.TemporaryDirectory() as folder:
            learning={'_gpu_checkpoint_dir':folder}
            with patch.object(LanePool,'donate',stop_after),patch.object(DomainCUDA,'step',lambda engine,work:step(engine,1)):
                first=solve(problem,target='edges',seconds=30,replicas=4,options=options,learning_state=learning,stop=lambda:flag[0])
            self.assertEqual(first['status'],'stopped');self.assertTrue(is_reference(first['checkpoint']))
            saved=resolve_checkpoint(first['checkpoint'],folder);self.assertEqual(saved['lane_pool']['stats']['gpu_lane_owner_groups'],1)
            second=solve(problem,target='edges',seconds=30,replicas=4,options=options,learning_state=learning,resume=first['checkpoint'])
            self.assertEqual(second['status'],'infeasible');self.assertEqual(second['stats']['gpu_lane_owners_completed'],1)
    def test_deep160_active_frames_and_acknowledgement_preserve_remaining_volume(self):
        import math
        from test_constraint_dfs import board
        from search_problem import SearchProblem
        n=160;geometry=board(n,[(a,side,a+1,side) for a in range(0,n,2) for side in range(3)])
        tiles=[{'id':i,'segments':[[[side,2],[side,10]] for side in range(3)]} for i in range(n)]
        problem=SearchProblem(tiles,geometry);problem.domains=[list(range(3*c,3*c+3)) for c in range(n)]
        engine=DomainCUDA(problem,target='edges',event_propagation=True,precise_reasons=True);seen=0
        try:
            for _ in range(2000):
                engine.step(4096)
                if engine.candidates():
                    self.assertEqual(int(engine.state[1,0].get()),160)
                    seen+=1;engine.acknowledge([0]);replacement=self.roundtrip(engine);engine.close();engine=replacement
                    volume=sum(math.prod(value.bit_count() for value in row) for row in engine.remaining_domain_cubes())
                    self.assertEqual(volume,3**160-seen)
                    if seen==2:break
            self.assertEqual(seen,2)
        finally:engine.close()
    def test_full160_one_host_copy_runner_and_off_policy_resume(self):
        from gpu_domain_runner import solve
        problem=planted160_fixture();options={'gpu_incremental_checkpoint':True,'gpu_lane_balance':True,'gpu_propagated_forks':True,'gpu_event_propagation':True,'gpu_precise_reasons':True}
        with tempfile.TemporaryDirectory() as folder:
            learning={'_gpu_checkpoint_dir':folder};result=solve(problem,seconds=30,replicas=4,options=options,learning_state=learning)
            self.assertEqual(result['status'],'solved');self.assertEqual(result['stats']['gpu_checkpoint_host_copies'],1);self.assertTrue(is_reference(result['checkpoint']))
            restored=resolve_checkpoint(result['checkpoint'],folder);self.assertEqual(restored['checkpoint_storage']['gpu_checkpoint_host_copies'],1)
            resumed=solve(problem,seconds=30,replicas=4,resume=result['checkpoint'],options={**options,'gpu_incremental_checkpoint':False},learning_state=learning)
            self.assertEqual(resumed['codes'],result['codes']);self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],0)
    def test_runner_cap_fallback_keeps_current_exact_work(self):
        from gpu_domain_runner import solve
        import gpu_checkpoint_round25 as module
        problem=precise_reason_fixture();original=module.CheckpointStore
        def capped(directory,scope,**kwargs):return original(directory,scope,max_bytes=8192,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch.object(module,'CheckpointStore',capped):
            result=solve(problem,seconds=30,target='edges',replicas=4,options={'gpu_incremental_checkpoint':True},learning_state={'_gpu_checkpoint_dir':folder})
            self.assertTrue(result['complete']);self.assertEqual(result['status'],'infeasible')
            # An exhausted native-free generation can fit this cap; force a
            # live checkpoint through immediate cancellation after one dispatch.
            flag=[False];step=DomainCUDA.step
            def once(engine,work):value=step(engine,1);flag[0]=True;return value
            with tempfile.TemporaryDirectory() as other,patch.object(DomainCUDA,'step',once):
                stopped=solve(problem,seconds=30,target='edges',replicas=4,options={'gpu_incremental_checkpoint':True},learning_state={'_gpu_checkpoint_dir':other},stop=lambda:flag[0])
            self.assertEqual(stopped['status'],'stopped');self.assertFalse(is_reference(stopped['checkpoint']));self.assertEqual(stopped['gpu_checkpoint_status'],'legacy_fallback')
            resumed=solve(problem,seconds=30,target='edges',replicas=4,resume=stopped['checkpoint'])
            self.assertEqual(resumed['status'],'infeasible')

if __name__=='__main__':unittest.main(verbosity=2)
