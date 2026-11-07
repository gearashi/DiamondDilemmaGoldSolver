"""CPU-only structural checks for trusted CUDA checkpoint restoration."""
import hashlib
import unittest
import numpy as np
from gpu_domain_round24 import DomainCUDA,META
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face


class HostCheckpointTests(unittest.TestCase):
    def engine(self):
        # Build the same fixed state layout without importing CuPy or launching
        # a kernel. restore's last copy uses a deliberately trivial device sink.
        result=DomainCUDA.__new__(DomainCUDA);result.problem=SearchProblem(simple_tiles(),double_face())
        result.n=2;result.nc=6;result.words=1;result.reason_words=1;result.lanes=1;result.target='gold'
        result.learning=True;result.incremental_strands=True
        result.binding={'fingerprint':result.problem.fingerprint,'target':'gold','original':result._original(),
                        'learning':True,'incremental_strands':True}
        result.cubes=(tuple(int(v,16) for v in result.binding['original']),)
        fields=[('meta',META),('domain',2),('trail',12),('reason',2),('frame_cell',2),('frame_code',2),
                ('frame_remaining',2),('frame_trail',2),('frame_union',2),('frame_assign',2),('frame_reason',4),
                ('frame_failure',2),('board',2),('owner',2),('assign_log',2),('parent',32),('size',32),
                ('union_child',32),('union_parent',32),('union_size',32),('clause_length',32),('clause_cell',64),('clause_code',64)]
        result.offsets={};cursor=0
        for name,count in fields:result.offsets[name]=cursor;cursor+=count
        result.rows=cursor;result.state=result._initial_state();result.cp=type('Sink',(),{'asarray':staticmethod(np.asarray)})
        result.delivered=[]
        return result

    def checkpoint(self,engine):
        state=engine.state.copy()
        return {'schema':engine.SCHEMA,'binding':engine.binding,'cubes':[[hex(v) for v in cube] for cube in engine.cubes],
                'state':state,'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'delivered':[],'rows':engine.rows}

    def rehash(self,snapshot):snapshot['state_sha256']=hashlib.sha256(snapshot['state'].tobytes()).hexdigest()

    def test_valid_initial_state_round_trips(self):
        engine=self.engine();snapshot=self.checkpoint(engine);engine.restore(snapshot)
        self.assertTrue(np.array_equal(snapshot['state'],engine.state))

    def test_rehashed_parent_cycle_is_rejected_before_device_execution(self):
        engine=self.engine();snapshot=self.checkpoint(engine);start=engine.offsets['parent']
        snapshot['state'][start,0]=1;snapshot['state'][start+1,0]=0;self.rehash(snapshot)
        with self.assertRaisesRegex(ValueError,'Cyclic'):engine.restore(snapshot)

    def test_rehashed_domain_padding_and_trail_mismatch_are_rejected(self):
        for kind in ('padding','unrecorded_removal','duplicate'):
            engine=self.engine();snapshot=self.checkpoint(engine);state=snapshot['state']
            if kind=='padding':state[engine.offsets['domain'],0]|=np.uint32(1<<31)
            elif kind=='unrecorded_removal':state[engine.offsets['domain'],0]&=np.uint32(~1&0xffffffff)
            else:
                state[2,0]=2;state[engine.offsets['trail'],0]=0;state[engine.offsets['trail']+1,0]=0
            self.rehash(snapshot)
            with self.assertRaises(ValueError):engine.restore(snapshot)

    def test_rehashed_union_and_assignment_log_indices_are_rejected(self):
        engine=self.engine();snapshot=self.checkpoint(engine);state=snapshot['state'];state[10,0]=1
        state[engine.offsets['union_child'],0]=100;self.rehash(snapshot)
        with self.assertRaisesRegex(ValueError,'rollback'):engine.restore(snapshot)
        snapshot=self.checkpoint(engine);snapshot['state'][9,0]=1;snapshot['state'][engine.offsets['assign_log'],0]=100;self.rehash(snapshot)
        with self.assertRaisesRegex(ValueError,'assignment'):engine.restore(snapshot)

if __name__=='__main__':unittest.main()
