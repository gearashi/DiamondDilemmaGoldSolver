"""Ownership and resume tests independent of CUDA and the real runtime."""
import unittest
import numpy as np
from systematic_jobs import JobLedger

class Array:
    def __init__(self, values): self.values=np.asarray(values, np.uint8)
    def get(self): return self.values.copy()

class GPU:
    def __init__(self, n): self.states=Array([3]*n); self.seen=[]
    def load_prefixes(self, prefixes, lanes):
        for prefix, lane in zip(prefixes, lanes):
            self.seen.append(tuple(prefix)); self.states.values[lane]=0

class LedgerTests(unittest.TestCase):
    def test_disjoint_refill_resume_covers_each_job_once(self):
        prefixes=np.arange(21,dtype=np.int16).reshape(7,3)
        lengths=np.full(7,3)
        gpu=GPU(3); ledger=JobLedger.empty(7,3)
        ledger.refill(gpu,prefixes,lengths)
        self.assertEqual(ledger.cursor,3)
        gpu.states.values[[0,2]]=2
        ledger.refill(gpu,prefixes,lengths)
        ledger=JobLedger.restore(ledger.fields(),gpu.states.get())
        self.assertEqual(ledger.completed,2)
        gpu.states.values[:]=2
        ledger.refill(gpu,prefixes,lengths)
        self.assertEqual(ledger.cursor,7)
        gpu.states.values[:]=2
        ledger.refill(gpu,prefixes,lengths)
        self.assertEqual(ledger.completed,7)
        self.assertEqual(sorted(gpu.seen),[tuple(row) for row in prefixes])
        self.assertEqual(len(gpu.seen),len(set(gpu.seen)))

    def test_pending_board_keeps_job_owned(self):
        ledger=JobLedger(4,np.array([0,1],np.int64),2,0)
        ledger.retire([1,0])
        self.assertEqual(ledger.completed,0)
        self.assertEqual(ledger.ids.tolist(),[0,1])

    def test_invalid_checkpoint_rejected(self):
        cases=[JobLedger(4,np.array([0,0],np.int64),2,0),
               JobLedger(4,np.array([0,2],np.int64),2,0),
               JobLedger(4,np.array([0,1],np.int64),2,1),
               JobLedger(4,np.array([0,-1],np.int64),2,0)]
        for ledger in cases:
            with self.subTest(ids=ledger.ids), self.assertRaises(ValueError): ledger.validate([0,0])

    def test_ownership_must_match_fixed_prefix(self):
        prefixes=np.array([[0,-1],[3,6],[9,-1]],np.int16)
        lengths=np.array([1,2,1],np.int16)
        ledger=JobLedger(3,np.array([0,2],np.int64),3,1)
        ledger.validate_prefixes(prefixes[[0,2]],np.array([1,1]),prefixes,lengths)
        with self.assertRaisesRegex(ValueError,'persistent job ID'):
            ledger.validate_prefixes(prefixes[[2,0]],np.array([1,1]),prefixes,lengths)

    def test_failed_load_does_not_advance_ledger(self):
        class Failing(GPU):
            def load_prefixes(self,*args,**kwargs): raise RuntimeError('load failed')
        ledger=JobLedger.empty(2,2)
        with self.assertRaises(RuntimeError): ledger.refill(Failing(2),np.array([[0],[1]]),np.ones(2,dtype=int))
        self.assertEqual(ledger.cursor,0)
        self.assertEqual(ledger.ids.tolist(),[-1,-1])

if __name__=='__main__': unittest.main()
