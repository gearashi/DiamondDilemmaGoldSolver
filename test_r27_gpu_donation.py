"""CPU-executed ownership regressions at native CUDA donation boundaries."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from gpu_domain_round24 import DomainCUDA
from gpu_lane_pool_round25 import LanePool
from gpu_reasoning_round26 import domain_layout
from r27_gpu_snapshots import PropagatedPrefix, donation_prefixes


class HostArray(np.ndarray):
    def get(self):
        return self.copy()


def host_engine():
    engine = DomainCUDA.__new__(DomainCUDA)
    engine.n,engine.nc,engine.words,engine.lanes = 3,9,1,2
    engine.incremental_strands = True
    engine.binding = {'original':[hex(511)]*3}
    engine.cubes = ((1,8,64),(2,8,64))
    engine.offsets = {}
    engine.rows = 0
    for field,length in domain_layout(3):
        engine.offsets[field] = engine.rows
        engine.rows += length
    engine.delivered = []
    engine.state = engine._initial_state().view(HostArray)
    engine.cp = SimpleNamespace(asarray=np.asarray,cuda=SimpleNamespace(get_current_stream=lambda:Mock()))
    return engine


class DonationBoundaries(unittest.TestCase):
    def test_pending_candidate_survives_direct_batch_and_pool_donation(self):
        engine = host_engine()
        engine.state[0] = [0,1]
        before = engine.state.copy()
        original = engine.cubes
        with self.assertRaisesRegex(ValueError,'unacknowledged'):
            engine.replace_lanes({0:original[0],1:original[1]})
        self.assertEqual(engine.cubes,original)
        np.testing.assert_array_equal(engine.state,before)
        pool = LanePool(None,[])
        pool.lanes = [None,0]
        self.assertIsNone(pool.donate(engine))
        with self.assertRaisesRegex(ValueError,'active donor'):
            donation_prefixes(engine,engine.state[:,1],())
        np.testing.assert_array_equal(engine.state,before)

    def test_failed_copy_restores_all_old_native_columns_and_host_roots(self):
        engine = host_engine()
        class InterruptedCopy(HostArray):
            failed = False
            def __setitem__(self,key,value):
                if not self.failed:
                    self.failed = True
                    # Simulate one column reaching the device before the second
                    # one faults. The next assignment is the rollback operation.
                    selected_rows,lanes = key
                    super().__setitem__((selected_rows,lanes[0]),value[:,0])
                    raise RuntimeError('injected partial CUDA transfer')
                super().__setitem__(key,value)
        engine.state = engine.state.view(InterruptedCopy)
        previous = np.array(engine.state)
        original = engine.cubes
        with self.assertRaisesRegex(RuntimeError,'partial CUDA transfer'):
            engine.replace_lanes({0:(1,8,128),1:(2,8,128)})
        np.testing.assert_array_equal(engine.state,previous)
        self.assertEqual(engine.cubes,original)

    def test_partially_propagated_prefix_and_ancestor_sibling_replay(self):
        engine = host_engine()
        o = engine.offsets
        col = engine.state[:,0]
        # Frame branches the third cell on codes 6 or 7. The first two cells
        # are assigned during propagation of the selected child, after its mark.
        col[1],col[2],col[6],col[9],col[10] = 1,1,1,2,1
        col[o['trail']] = 2*engine.nc+7
        col[o['frame_cell']] = 2
        col[o['frame_remaining']] = 1<<7
        col[o['frame_assign']] = 1
        col[o['frame_union']] = 0
        col[o['assign_log']:o['assign_log']+2] = (0,1)
        col[o['board']:o['board']+2] = (0,3)
        col[o['union_child']] = 16
        col[o['union_parent']] = 0
        parts = engine.remaining_domain_cubes([0],lane_column=col)
        self.assertEqual(parts,[(1,8,64),(1,8,128)])
        marks = donation_prefixes(engine,col,parts)
        self.assertEqual(marks,[(2,1),(1,0)])
        prefix = PropagatedPrefix.capture(engine,col)
        before = engine.state.copy()
        for invalid in ([(3,0),(1,0)],[(2,1),(1,1)],[(2,1),(-1,0)]):
            with self.assertRaises(ValueError):
                engine.replace_lanes(dict(enumerate(parts)),prefixes=(prefix,invalid))
            np.testing.assert_array_equal(engine.state,before)
        col[6] = 3
        self.assertEqual(engine.remaining_domain_cubes([0],lane_column=col),[parts[1]])
        self.assertEqual(donation_prefixes(engine,col,[parts[1]]),[(1,0)])
        col[6] = 1
        with self.assertRaisesRegex(ValueError,'source CUDA engine'):
            prefix.install(host_engine(),before[:,0],parts[0],2,1)
        engine.replace_lanes(dict(enumerate(parts)),prefixes=(prefix,marks))
        self.assertEqual(engine.state[9].tolist(),[2,1])
        self.assertEqual(engine.state[10].tolist(),[1,0])
        self.assertEqual(engine.state[o['parent']+16].tolist(),[0,16])
        self.assertEqual(engine.state[o['owner']+1].tolist(),[1,0xffffffff])

