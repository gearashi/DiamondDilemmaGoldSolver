"""Actual CUDA regressions for perfect-candidate publication and acknowledgement."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gpu_engine import GPU
from geometry import build_board
from test_gpu import fixture


class PendingCandidateTests(unittest.TestCase):
    def flat_gpu(self, n=8):
        geometry=build_board()
        gpu=GPU(np.zeros((480,3),np.uint16),geometry.neighbor_cells,geometry.neighbor_sides,n=n,seed=912)
        gpu.initialize()
        return gpu

    def snapshot(self,gpu):
        return {name:getattr(gpu,name).get().copy() for name in
                ('boards','bestboards','scores','bestscores','positions','rng','counters','pending')}

    def assert_snapshot(self,gpu,snapshot):
        for name,expected in snapshot.items():
            np.testing.assert_array_equal(getattr(gpu,name).get(),expected,err_msg=name)

    def test_initial_perfect_boards_stay_frozen(self):
        gpu=self.flat_gpu()
        self.assertEqual(gpu.pending_indices().tolist(),list(range(gpu.n)))
        before=self.snapshot(gpu)
        for _ in range(3):
            gpu.step(128)
        self.assert_snapshot(gpu,before)
        self.assertEqual(gpu.reseed(fraction=1).tolist(),[])
        self.assert_snapshot(gpu,before)
        gpu.verify_device_scores()

    def test_acknowledgement_releases_only_selected_replicas(self):
        gpu=self.flat_gpu()
        before=self.snapshot(gpu)
        self.assertEqual(gpu.acknowledge([0,2,2]),2)
        self.assertEqual(gpu.acknowledge([0,2]),0)
        gpu.step(128)
        after=self.snapshot(gpu)
        frozen=np.array([1,3,4,5,6,7])
        np.testing.assert_array_equal(after['boards'][:,frozen],before['boards'][:,frozen])
        np.testing.assert_array_equal(after['rng'][frozen],before['rng'][frozen])
        np.testing.assert_array_equal(after['counters'][:,frozen],before['counters'][:,frozen])
        self.assertTrue(np.all(after['pending']==1))
        for index in (0,2):
            self.assertFalse(np.array_equal(after['bestboards'][:,index],before['bestboards'][:,index]))
            self.assertGreater(int(after['counters'][0,index]),0)
            self.assertLessEqual(int(after['counters'][0,index]),128)
        gpu.step(128)
        self.assert_snapshot(gpu,after)
        gpu.verify_device_scores()

    def test_new_perfect_candidate_stops_the_kernel(self):
        masks,neighbors,sides,reference=fixture()
        damaged=reference.copy();damaged[0]+=1
        gpu=GPU(masks,neighbors,sides,n=8,seed=125)
        gpu.initialize(damaged,perturbations=0)
        self.assertFalse(np.any(gpu.pending.get()))
        gpu.temps.fill(0)
        gpu.guidance.fill(1)
        for _ in range(40):
            gpu.step(32)
            if np.all(gpu.pending.get()):break
        self.assertTrue(np.all(gpu.pending.get()),gpu.scores.get())
        self.assertTrue(np.all(gpu.bestscores.get()==240))
        frozen=self.snapshot(gpu)
        gpu.step(128)
        self.assert_snapshot(gpu,frozen)
        gpu.verify_device_scores()

    def test_noop_does_not_republish_identical_checked_board(self):
        masks,neighbors,sides,reference=fixture()
        gpu=GPU(masks,neighbors,sides,n=8,seed=714)
        gpu.initialize(reference,perturbations=0)
        gpu.acknowledge(np.arange(gpu.n))
        gpu.temps.fill(0)
        gpu.step(128)
        # This planted board has no accepted different edge-perfect neighbor.
        # Accepted no-ops must not halt the replica after every CPU acknowledgement.
        self.assertFalse(np.any(gpu.pending.get()))
        np.testing.assert_array_equal(gpu.counters.get()[0],np.full(gpu.n,128))
        gpu.verify_device_scores()

    def test_pending_survives_checkpoint_and_legacy_upgrade(self):
        masks,neighbors,sides,reference=fixture()
        gpu=GPU(masks,neighbors,sides,n=8,seed=342)
        gpu.initialize(reference,perturbations=0)
        gpu.acknowledge([1,3])
        with tempfile.TemporaryDirectory(prefix='diamond-pending-') as directory:
            path=Path(directory)/'state.npz';gpu.checkpoint(path)
            other=GPU(masks,neighbors,sides,n=8,seed=9)
            self.assertTrue(other.resume(path))
            before=self.snapshot(gpu)
            self.assert_snapshot(other,before)
            gpu.step(32);other.step(32)
            self.assert_snapshot(other,self.snapshot(gpu))
            with np.load(path,allow_pickle=False) as archive:
                arrays={key:archive[key].copy() for key in archive.files}
            legacy={key:value.copy() for key,value in arrays.items() if key!='pending'}
            legacy['version']=np.int32(1)
            damaged=reference.copy();damaged[0]+=1
            legacy['boards'][:,0]=damaged
            legacy['scores'][0]=gpu.score_cpu(damaged)[0]
            old=Path(directory)/'legacy.npz';np.savez_compressed(old,**legacy)
            self.assertTrue(other.resume(old))
            self.assertTrue(np.all(other.pending.get()==1))
            np.testing.assert_array_equal(other.boards.get(),other.bestboards.get())
            other.verify_device_scores()
            arrays['pending'][0]=2
            bad=Path(directory)/'bad.npz';np.savez_compressed(bad,**arrays)
            with self.assertRaisesRegex(ValueError,'pending'):
                other.resume(bad)

    def test_reseed_preserves_unacknowledged_candidates(self):
        gpu=self.flat_gpu()
        before=self.snapshot(gpu)
        gpu.acknowledge([3,5])
        selected=gpu.reseed(fraction=1)
        self.assertEqual(set(selected),{3,5})
        frozen=np.array([0,1,2,4,6,7])
        np.testing.assert_array_equal(gpu.boards.get()[:,frozen],before['boards'][:,frozen])
        np.testing.assert_array_equal(gpu.bestboards.get()[:,frozen],before['bestboards'][:,frozen])
        self.assertTrue(np.all(gpu.pending.get()==1))
        gpu.verify_device_scores()

    def test_invalid_acknowledgements_rejected(self):
        gpu=self.flat_gpu()
        for indices in ([0.0],[True],[-1],[gpu.n],[[0]],0):
            with self.assertRaises(ValueError):gpu.acknowledge(indices)
        self.assertEqual(gpu.acknowledge([]),0)
        self.assertEqual(gpu.pending_indices().size,gpu.n)

if __name__=='__main__':
    unittest.main(verbosity=2)
