"""Incremental classic checkpoint ownership, old stack tails and finite native controls."""
import json
import os
import unittest
from types import SimpleNamespace

import numpy as np

from classic_checkpoint_ranges import CheckpointRanges, FRAME_FIELDS, _frame_windows, _attained_bounds
from dfs_gpu import DFS, fill_order
from systematic_jobs import JobLedger
from test_replica_resize import host_dfs, frontier
from validator import orientation_masks, validate_arrangement


def compare(test, gpu, reader):
    expected = gpu.snapshot_checkpoint()
    actual = reader.capture()
    test.assertEqual(actual.keys(), expected.keys())
    for name, value in actual.items():
        test.assertEqual(value.dtype, expected[name].dtype, name)
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
        if isinstance(value, np.ndarray):
            test.assertFalse(value.flags.writeable, name)
    return actual


class HostTests(unittest.TestCase):
    def test_immutable_saved_tails_and_install_retire_between_snapshots(self):
        gpu = host_dfs(3)
        rows,lengths = frontier(5)
        ledger = JobLedger.empty(5,3)
        ledger.refill(gpu,rows,lengths)
        archive = gpu.snapshot_checkpoint()
        for name in FRAME_FIELDS:
            archive[name] = archive[name].copy()
            archive[name][87] = [17,63,451]
        gpu.restore_checkpoint(archive)
        reader = CheckpointRanges(gpu,ledger)
        first = compare(self,gpu,reader)
        for lane in (0,1):
            gpu.reject_current_subtree(lane,gpu.boards[:,lane].get().tolist())
        ledger.refill(gpu,rows,lengths)
        # Reassigned and already retired. The observed lane state alone
        # could hide both intervening installations and the reset tail.
        for lane in (0,1):
            gpu.reject_current_subtree(lane,gpu.boards[:,lane].get().tolist())
        ledger.retire(gpu.states.get())
        second = compare(self,gpu,reader)
        for name in FRAME_FIELDS:
            np.testing.assert_array_equal(first[name][87],[17,63,451])
            np.testing.assert_array_equal(second[name][87],[65535 if name=='cursors' else 0]*2+[451])
        gpu.reject_current_subtree(2,gpu.boards[:,2].get().tolist())
        ledger.retire(gpu.states.get())
        compare(self,gpu,reader)
        compare(self,gpu,reader)
        self.assertEqual((reader.maps,reader.read_bytes),(0,0))

    def test_every_valid_depth_floor_and_installed_window(self):
        pairs = np.array([(f,m) for f in range(161) for m in range(f,161)])
        floor,high = pairs.T
        for reset in (np.zeros(len(pairs),bool), np.arange(len(pairs))%3==0):
            ids = np.arange(len(pairs))
            covered = np.zeros((160,len(ids)),bool)
            for a,b,subset in _frame_windows(ids,floor,high,reset):
                self.assertFalse(covered[a:b,subset].any())
                covered[a:b,subset] = True
            expected = (np.arange(160)[:,None]>=floor)&(np.arange(160)[:,None]<=high)
            self.assertFalse(np.any((expected|reset)&~covered))

    def test_placement_bounds_cover_every_short_walk_including_unbilled_rewinds(self):
        import itertools
        # Starts, filled-rank counters, and final depths are all the proof can use.
        # Include ordinary increments/decrements, a same-rank candidate, and a
        # multi-rank host/Gold rewind that consumes no placement.
        cases=[]
        for start in range(5):
            for path in itertools.product(range(4),repeat=5):
                d=start
                placed=0
                touched=[]
                for move in path:
                    if move==0:
                        touched.extend((d,d+1))
                        d+=1
                        placed+=1
                    elif move==1:
                        d=max(0,d-1)
                    elif move==2:
                        touched.append(d)
                    else:
                        d//=2
                if touched:
                    cases.append((start,d,placed,min(touched),max(touched)))
        start,final,placed,low,top=np.array(cases,dtype=np.int16).T
        n=len(cases)
        counters=np.zeros((3,n),np.uint64)
        prior={'depths':start,'counters':counters}
        now={'depths':final,'counters':np.stack((np.ones(n),placed,np.zeros(n))).astype(np.uint64),
             'floors':np.zeros(n,np.int16),'maxdepths':np.full(n,160,np.int16)}
        lanes,begin,end=_attained_bounds(prior,now,np.arange(n),np.zeros(n,bool),True)
        self.assertEqual(len(lanes),n)
        self.assertTrue(np.all(begin<=low))
        self.assertTrue(np.all(end>=top))
        _,begin,end=_attained_bounds(prior,now,np.arange(n),np.zeros(n,bool),False)
        np.testing.assert_array_equal(begin,now['floors'])
        np.testing.assert_array_equal(end,now['maxdepths'])

    def test_backend_without_range_support_preserves_its_capture_contract(self):
        gpu = SimpleNamespace(snapshot_checkpoint=lambda: {'native':'full'})
        self.assertEqual(CheckpointRanges(gpu,None).capture(),{'native':'full'})


class NativeControls:
    def pool(self,data,board,n,**options):
        args = (np.array(orientation_masks(data),np.uint16),board.neighbor_cells,board.neighbor_sides)
        if self.web:
            import wgpu
            from dfs_webgpu import WebGPUDFS
            api = os.environ.get('DIAMOND_RANGE_API')
            if api:
                options['adapter'] = next(a for a in wgpu.gpu.enumerate_adapters_sync()
                    if a.info['backend_type']==api and a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
            gpu = WebGPUDFS(*args,n=n,prefixes=[],allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1',**options)
            self.addCleanup(gpu.close)
        else:
            gpu = DFS(*args,n=n,prefixes=[],**options)
        return gpu

    def test_scatter_stale_restored_frames_counters_and_cold_resume(self):
        from test_gpu_gold import gold_fixture
        data,board = gold_fixture()
        n = 512
        ids = np.arange(0,n,16)
        gpu = self.pool(data,board,n)
        gpu.load_prefixes([[int(i)] for i in range(len(ids))],lanes=ids)
        payload = gpu.snapshot_checkpoint()
        for name in FRAME_FIELDS:
            payload[name]=payload[name].copy()
            payload[name][135,:]=np.arange(n,dtype=np.uint16)+19
        payload['counters'] = np.full((3,n),2**32-2,np.uint64)
        gpu.restore_checkpoint(payload)
        ledger = JobLedger.empty(len(ids),n)
        ledger.cursor = len(ids)
        ledger.ids[ids] = np.arange(len(ids))
        ledger.validate(gpu.states.get())
        reader = CheckpointRanges(gpu,ledger)
        old = compare(self,gpu,reader)
        gpu.step(64)
        advanced = compare(self,gpu,reader)
        full = 816*n*4 if self.web else sum(old[k].nbytes for k in gpu.DEVICE_FIELDS)
        self.assertLess(reader.read_bytes,full//4)
        self.assertLessEqual(reader.maps,6 if self.web else 12)
        for name in ('order','roots','prefix_codes'):
            self.assertIs(old[name],advanced[name])
        if self.web:
            self.assertTrue({'scatter','gather'} <= gpu._column_copies.pipelines.keys())
        np.testing.assert_array_equal(old['counters'],payload['counters'])
        self.assertGreater(advanced['counters'][0,ids].max(),2**32)
        other = self.pool(data,board,n)
        other.restore_checkpoint(advanced)
        gpu.step(3)
        other.step(3)
        after = compare(self,gpu,reader)
        fresh = other.snapshot_checkpoint()
        for name in fresh:
            np.testing.assert_array_equal(after[name],fresh[name],err_msg=name)

    def test_variable_height_frame_packets_with_real_field_values(self):
        from test_gpu_gold import gold_fixture
        data,board=gold_fixture()
        gpu=self.pool(data,board,1024)
        ids=np.arange(1024)
        # Each root gets a different artificial historical high-water mark. The
        # entire frame storage has a distinct deterministic pattern, so a bad
        # offset, lane map, row stripe or high bit cannot match by coincidence.
        ledger=JobLedger.empty(0,len(ids))
        reader=CheckpointRanges(gpu,ledger)
        windows=[(1,13,ids),(13,41,ids[::3]),(43,115,ids[17:199]),(147,160,ids[5::13])]
        source=[]
        for i,name in enumerate(FRAME_FIELDS):
            value=(np.arange(160*len(ids),dtype=np.uint32).reshape(160,-1)*31+7*i).astype(np.uint16)
            getattr(gpu,name)[:,:]=gpu.cp.asarray(value)
            source.append(value)
        for raw,value in zip(reader._frames(windows),source):
            want=np.concatenate([value[a:b,cols].ravel() for a,b,cols in windows])
            np.testing.assert_array_equal(raw,want)
        self.assertEqual(reader.maps,1)
        self.assertEqual(reader.read_bytes,8*sum((b-a)*len(cols) for a,b,cols in windows))

    def test_legacy_gold_rewind_to_exhaustion(self):
        from test_gpu_gold import gold_fixture
        data,board=gold_fixture(False)
        edge=self.pool(data,board,4)
        rows,lengths=frontier(4)
        ledger=JobLedger.empty(4,4)
        ledger.refill(edge,rows,lengths)
        edge.step(8)
        old=edge.snapshot_checkpoint()
        old['dfs_version']=np.int32(2)
        del old['gold_fingerprint']
        gpu=self.pool(data,board,4,gold_data=data)
        gpu.restore_checkpoint(old)
        reader=CheckpointRanges(gpu,ledger)
        compare(self,gpu,reader)
        for _ in range(64):
            gpu.step(16)
            ledger.retire(gpu.states.get())
            compare(self,gpu,reader)
            if ledger.completed==4:
                break
        self.assertEqual(ledger.completed,4)
        compare(self,gpu,reader)
        self.assertEqual(reader.maps,0)

    def test_assisted_ownership_pending_gold_and_acknowledgement(self):
        from test_gpu_reasoning import varied_gold_fixture
        data,board=varied_gold_fixture()
        order=fill_order(board.neighbor_cells)
        gpu=self.pool(data,board,2,order=order,gold_data=data,
                      gpu_features={'dynamic_mrv':True,'forced':True,'cpu_endgame':True,'cpu_probe':True})
        rows=np.full((1,160),-1,np.int16)
        rows[0,:157]=3*order[:157]
        ledger=JobLedger.empty(1,2)
        ledger.refill(gpu,rows,np.array([157],np.int16))
        reader=CheckpointRanges(gpu,ledger)
        first=compare(self,gpu,reader)
        gpu.cpu_assist.interval=1
        gpu.cpu_assist.seconds=1e-9
        for _ in range(3):
            gpu.step(1)
            compare(self,gpu,reader)
            if gpu.cpu_assist.entries:
                break
        self.assertTrue(gpu.cpu_assist.entries)
        self.assertFalse(json.loads(str(first['cpu_assist_state']))['entries'])
        entry=next(iter(gpu.cpu_assist.entries.values()))
        entry['slice_seconds']=2
        for _ in range(16):
            gpu.step(1)
            captured=compare(self,gpu,reader)
            if captured['states'][0]==1:
                break
        self.assertEqual(captured['states'][0],1)
        self.assertTrue(validate_arrangement(data,captured['boards'][:,0].tolist(),board)['valid'])
        self.assertGreater(json.loads(str(captured['cpu_assist_state']))['stats']['cpu_endgame_solutions'],0)
        gpu.acknowledge([0])
        compare(self,gpu,reader)
        self.assertEqual(captured['states'][0],1)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires CUDA device')
class CudaTests(NativeControls,unittest.TestCase):
    web = False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires WebGPU device')
class WebTests(NativeControls,unittest.TestCase):
    web = True
