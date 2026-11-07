"""Host-only contracts for paired cooperative timing; no device workload."""
import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import benchmark_gpu_round26 as benchmark

class _Array:
    def __init__(self,array):self.array=array
    def get(self):return self.array.copy()

class FakeDomain:
    opened=[];closed=[];mismatch=False
    def __init__(self,problem,cubes,*,cooperative=False,**kwargs):
        self.lanes=len(cubes);self.rows=43;self.offsets={'meta':0,'support_counters':29};self.cooperative=cooperative
        self.state=_Array(np.zeros((self.rows,self.lanes),dtype=np.uint32));self.device='host-only mock';self.state_bytes=self.rows*self.lanes*4
        type(self).opened.append(self)
    def step(self,work):
        self.state.array[11]+=work*(2 if self.cooperative and type(self).mismatch else 1)
        self.state.array[35]+=work*(2 if self.cooperative else 1)
        return self.state.array[0].copy()
    def stats(self,*,snapshot):return {'decisions':int(snapshot[11].sum()),'support_word_scans':int(snapshot[35].sum()),'lanes':self.lanes}
    def candidates(self):return []
    def close(self):type(self).closed.append(self)

class CooperativeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        FakeDomain.opened=[];FakeDomain.closed=[];FakeDomain.mismatch=False
        from search_fixtures import planted_gold
        data=planted_gold(2614,size=4)[0];encoded=json.dumps(data).encode();original=Path.read_bytes
        def read(path):
            if path.name=='tiles.json' and path.parent.name=='data':return encoded
            return original(path)
        fixture=patch.object(Path,'read_bytes',read);fixture.start();self.addCleanup(fixture.stop)

    def test_alternating_pairs_equal_work_and_charge_wider_scans(self):
        with patch('gpu_domain_round24.DomainCUDA',FakeDomain):rows=benchmark.cooperative_rows(repeats=3,dispatches=2,quantum=3,replicas=1)
        self.assertEqual(len(rows),6);self.assertEqual(len(FakeDomain.opened),len(FakeDomain.closed))
        for row in rows:
            pair=row['ordered_pair'];self.assertEqual([v['cooperative'] for v in pair],[False,True] if row['repetition']%2==0 else [True,False])
            self.assertEqual(pair[0]['final_logical_sha256'],pair[1]['final_logical_sha256'])
            byflag={entry['cooperative']:entry for entry in pair}
            self.assertGreater(byflag[True]['stats']['support_word_scans'],byflag[False]['stats']['support_word_scans'])
            for entry in pair:
                self.assertEqual(entry['active_lane_dispatches'],2);self.assertEqual(len(entry['dispatch_seconds']),2)
                self.assertEqual(entry['dispatches'],2);self.assertEqual(entry['quantum'],3)
            self.assertEqual(len(row['warmup']),2 if row['repetition']==0 else 0)
    def test_mismatched_logical_work_rejects_and_closes_all_owners(self):
        FakeDomain.mismatch=True
        with patch('gpu_domain_round24.DomainCUDA',FakeDomain):
            with self.assertRaisesRegex(RuntimeError,'logical state'):benchmark.cooperative_rows(repeats=3,dispatches=2,quantum=1,replicas=1)
        self.assertEqual(len(FakeDomain.opened),len(FakeDomain.closed))
