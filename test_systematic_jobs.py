"""Ownership and resume tests independent of CUDA and the real runtime."""
import unittest
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
import systematic_jobs as jobs
from test_replica_resize import host_dfs, frontier, snapshot, owned_states
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


class OwnershipCacheTests(unittest.TestCase):
    def setUp(self):
        self.gate = patch.object(jobs, 'OWNERSHIP_CACHE_MIN_LANES', 1)
        self.gate.start();self.addCleanup(self.gate.stop)
    def ledger(self):
        value=JobLedger(10,np.arange(4,dtype=np.int64),4,0)
        self.assertTrue(value.validate(np.zeros(4,np.uint8)))
        self.assertTrue(value.validate(np.zeros(4,np.uint8)))
        return value
    def test_duplicate_in_place_and_replacement_after_cache_hits(self):
        for replacement in (False,True):
            value=self.ledger()
            if replacement:value.ids=value.ids.copy()
            value.ids[2]=value.ids[1]
            with self.subTest(replacement=replacement),self.assertRaisesRegex(ValueError,'same job'):
                value.validate(np.zeros(4,np.uint8))
    def test_reorder_and_restore_validate_exact_ownership(self):
        value=self.ledger();value.ids=value.ids[::-1].copy()
        self.assertTrue(value.validate(np.zeros(4,np.uint8)))
        saved=value.fields()
        self.assertFalse(any('cache' in name or 'validated' in name for name in saved))
        restored=JobLedger.restore(saved,np.zeros(4,np.uint8))
        np.testing.assert_array_equal(restored.ids,value.ids)
        restored.ids[0]=restored.ids[-1]
        with self.assertRaisesRegex(ValueError,'same job'):restored.validate(np.zeros(4,np.uint8))
    def test_all_live_checks_remain_after_cache_hits(self):
        for change,states in ((lambda x:setattr(x,'completed',1),[0]*4),
                              (lambda x:setattr(x,'cursor',3),[0]*4),
                              (lambda x:setattr(x,'total',3),[0]*4),
                              (lambda x:setattr(x,'retired_counters',np.zeros(2,np.uint64)),[0]*4),
                              (lambda x:None,[0,0,3,0]),
                              (lambda x:x.ids.__setitem__(0,-2),[0]*4),
                              (lambda x:x.ids.__setitem__(0,4),[0]*4)):
            value=self.ledger();change(value)
            with self.assertRaises(ValueError):value.validate(states)
    def test_paused_overlap_rechecked_with_unchanged_active_ids(self):
        value=self.ledger()
        value.paused=SimpleNamespace(owned={1},max_id=1,count=1)
        with self.assertRaisesRegex(ValueError,'same job'):value.validate([0]*4)
    def test_private_copy_is_frozen_and_does_not_alias_ids(self):
        value=self.ledger();cached=value._validated_unique_ids
        self.assertFalse(cached.flags.writeable)
        self.assertFalse(np.shares_memory(cached,value.ids))
        with patch.object(jobs.np,'unique',side_effect=AssertionError('unexpected repeated sort')):
            self.assertTrue(value.validate([0]*4))
        value.ids[0]=value.ids[1]
        np.testing.assert_array_equal(cached,np.arange(4))
        with self.assertRaises(ValueError):value.validate([0]*4)
    def test_small_pool_keeps_original_path(self):
        with patch.object(jobs,'OWNERSHIP_CACHE_MIN_LANES',32768):
            value=self.ledger();self.assertIsNone(value._validated_unique_ids)
            value.ids[0]=value.ids[1]
            with self.assertRaises(ValueError):value.validate([0]*4)
    def test_sparse_large_pool_uses_active_count(self):
        with patch.object(jobs,'OWNERSHIP_CACHE_MIN_LANES',4096):
            ids=np.full(32768,-1,np.int64);ids[:128]=np.arange(128)
            flags=np.full(32768,3,np.uint8);flags[:128]=0
            value=JobLedger(128,ids,128,0)
            value.validate(flags);value.validate(flags)
            self.assertIsNone(value._validated_unique_ids)
            value.ids[1]=0
            with self.assertRaises(ValueError):value.validate(flags)
    def test_retire_refill_shrink_grow_and_resume_after_cache_hits(self):
        prefixes,lengths=frontier(8);gpu=host_dfs(4)
        value=JobLedger.empty(8,4);value.refill(gpu,prefixes,lengths)
        value.validate(gpu.states.get());value.validate(gpu.states.get())
        gpu.states.value[1]=2;gpu.boards.value[:,1]=-1;gpu.used.value[:,1]=0;gpu.depths.value[1]=0
        value.retire(gpu.states.get());self.assertEqual(value.completed,1)
        value.refill(gpu,prefixes,lengths);self.assertEqual(value.ids.tolist(),[0,4,2,3])
        before=owned_states(gpu,value);saved=snapshot(gpu,value)
        small=host_dfs(2);resized=JobLedger.resume(small,saved,prefixes,lengths)
        self.assertEqual(resized.paused_count,2)
        resized.validate(small.states.get());resized.validate(small.states.get())
        big=host_dfs(6);grown=JobLedger.resume(big,snapshot(small,resized),prefixes,lengths)
        after=owned_states(big,grown);self.assertEqual(set(before),set(after))
        for job in before:
            for name in before[job]:np.testing.assert_array_equal(before[job][name],after[job][name])
        grown.refill(big,prefixes,lengths);self.assertEqual(grown.cursor,7)
        self.assertEqual(len(set(grown.ids)),6)

class OwnershipCacheTurnoverTests(unittest.TestCase):
    def setUp(self):
        gate=patch.object(jobs,'OWNERSHIP_CACHE_MIN_LANES',1)
        gate.start();self.addCleanup(gate.stop)

    def churn(self):
        value=JobLedger(100,np.arange(4,dtype=np.int64),4,0)
        value.validate([0]*4)
        for _ in range(jobs.OWNERSHIP_CACHE_CHURN_LIMIT):
            value.ids=np.roll(value.ids,1)
            value.validate([0]*4)
        self.assertIsNone(value._validated_unique_ids)
        self.assertEqual(value._ownership_cache_cooldown,jobs.OWNERSHIP_CACHE_COOLDOWN)
        return value

    def test_turnover_uses_exact_checks_without_copy_and_recovers(self):
        value=self.churn()
        for _ in range(jobs.OWNERSHIP_CACHE_COOLDOWN):
            value.ids=np.roll(value.ids,1)
            value.validate([0]*4)
            self.assertIsNone(value._validated_unique_ids)
        value.validate([0]*4)
        self.assertIsNotNone(value._validated_unique_ids)
        for _ in range(jobs.OWNERSHIP_CACHE_MIN_HITS):value.validate([0]*4)
        value.ids=np.roll(value.ids,1)
        value.validate([0]*4)
        self.assertEqual(value._ownership_cache_churn,0)
        with patch.object(jobs.np,'unique',side_effect=AssertionError('stable cache failed')):
            value.validate([0]*4)

    def test_duplicate_during_cooldown_and_probe_are_rejected(self):
        for remaining in (0,jobs.OWNERSHIP_CACHE_COOLDOWN):
            value=self.churn()
            for _ in range(jobs.OWNERSHIP_CACHE_COOLDOWN-remaining):value.validate([0]*4)
            value.ids[2]=value.ids[1]
            with self.assertRaisesRegex(ValueError,'same job'):value.validate([0]*4)

    def test_cooldown_keeps_state_and_overlap_validation(self):
        value=self.churn()
        with self.assertRaisesRegex(ValueError,'lane state'):value.validate([0,0,0,3])
        value.paused=SimpleNamespace(owned={1},max_id=1,count=1)
        with self.assertRaisesRegex(ValueError,'same job'):value.validate([0]*4)

    def test_policy_is_ephemeral_and_small_pool_clears_certificate(self):
        value=self.churn()
        restored=JobLedger.restore(value.fields(),[0]*4)
        self.assertEqual(restored._ownership_cache_cooldown,0)
        with patch.object(jobs,'OWNERSHIP_CACHE_MIN_LANES',10):
            restored.validate([0]*4)
        self.assertIsNone(restored._validated_unique_ids)
        self.assertEqual(restored._ownership_cache_churn,0)

if __name__=='__main__': unittest.main()
