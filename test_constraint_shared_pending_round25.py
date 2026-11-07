"""Pending verifier transport is optional work, never imported proof authority."""
import copy,itertools,json,unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch
import constraint_shared_conflicts as shared
from test_constraint_separator import ring
from test_constraint_dfs import brute
from test_constraint_shared_conflicts import finish
from test_constraint_sat_assist import HAS_SAT

class UnknownOracle:
    def __init__(self):self.stats={'sat_assist_calls':0,'sat_assist_clauses':0};self.last_outcome='unknown';self.called=False
    def prepare(self,*args,**kwargs):return True
    def query(self,*args,**kwargs):self.called=True;self.stats['sat_assist_calls']+=1;return None
    def close(self):pass

def signed(payload):
    payload['digest']=shared._digest(payload);return payload

def state(store):
    return (list(store._pending.items()),list(store._discarded.items()),store.records(),dict(store.stats),store._snapshot_json)

class PendingConflictTests(unittest.TestCase):
    def store(self,problem=None):
        store=shared.VerifiedConflictStore(problem or ring(4));self.addCleanup(store.close);return store
    def unknown(self,store):
        store._oracle=UnknownOracle()
        with patch.object(shared,'time',SimpleNamespace(monotonic=lambda:0)):
            self.assertEqual(store.advance(seconds=.01,attempts=1),0)
        return store._oracle.stats['sat_assist_calls']
    def test_unknown_budget_survives_json_restarts_and_final_tombstone(self):
        problem=ring(4);store=self.store(problem);store.offer([(0,0),(1,0)]);calls=0
        for attempt in range(shared.MAX_ATTEMPTS):
            calls+=self.unknown(store);payload=json.loads(json.dumps(store.pending_snapshot()))
            self.assertEqual(store.records(),[]);self.assertEqual(store.snapshot()['records'],[])
            if attempt+1<shared.MAX_ATTEMPTS:self.assertEqual(payload['pending'][0]['attempts'],attempt+1)
            else:self.assertEqual(len(payload['discarded']),1)
            store=self.store(problem);store.import_pending(payload)
        self.assertEqual(calls,shared.MAX_ATTEMPTS);self.assertEqual(store.pending_count,0)
        self.assertFalse(store.offer([(0,0),(1,0)]));self.assertEqual(self.unknown(store),0)
        self.assertEqual(store.records(),[])
    def test_merge_uses_max_attempts_and_discarded_dominates_old_pending(self):
        store=self.store();store.offer([(0,0)]);old=store.pending_snapshot()
        newer=copy.deepcopy(old);newer['pending'][0]['attempts']=2;signed(newer)
        self.assertEqual(store.import_pending(newer),0);store.import_pending(old)
        self.assertEqual(list(store._pending.values()),[2])
        terminal=copy.deepcopy(old);terminal['pending']=[];terminal['discarded']=[[[0,0]]];signed(terminal)
        store.import_pending(terminal);store.import_pending(old)
        self.assertEqual(store.pending_count,0);self.assertFalse(store.offer([(0,0)]));self.assertEqual(store.records(),[])
    def test_cancelled_query_and_preparation_do_not_charge_attempts(self):
        store=self.store();store.offer([]);before=store.pending_snapshot();oracle=UnknownOracle();store._oracle=oracle
        def stop():
            if oracle.called:raise InterruptedError('cancelled verifier')
        with patch.object(shared,'time',SimpleNamespace(monotonic=lambda:0)),self.assertRaises(InterruptedError):
            store.advance(stop,seconds=.01,attempts=1)
        self.assertEqual(store.pending_snapshot(),before);self.assertEqual(store.records(),[])
        receiver=self.store();receiver.import_pending(before);self.assertEqual(list(receiver._pending.values()),[0])
        def absent():raise ImportError('optional verifier missing')
        missing=shared.VerifiedConflictStore(ring(4),factory=absent);self.addCleanup(missing.close);missing.import_pending(before)
        self.assertEqual(missing.advance(seconds=.01),0);self.assertEqual(list(missing._pending.values()),[0]);self.assertEqual(missing.records(),[])
    def test_malformed_rows_bindings_and_digests_are_atomic(self):
        store=self.store();store.offer([(3,0)]);store._remember_discard(((2,0),));store.snapshot()
        donor=self.store();donor.offer([(0,0)]);payload=donor.pending_snapshot();before=state(store)
        mutations=[lambda p:p.update(shared_conflicts_pending_version=True),lambda p:p.update(fingerprint='other'),
            lambda p:p.update(domains_sha256='other'),lambda p:p.update(target='edges'),lambda p:p.update(digest='wrong'),
            lambda p:p.update(extra=[]),lambda p:p.update(pending={}),lambda p:p.update(discarded={}),
            lambda p:p['pending'][0].update(extra=0),lambda p:p['pending'].append(copy.deepcopy(p['pending'][0])),
            lambda p:p.update(discarded=[[[0,0]]]),lambda p:p.update(discarded=[[],[]]),
            lambda p:p['pending'].append({'placements':[[True,0]],'attempts':0}),
            lambda p:p['pending'][0].update(placements=[[0,False]]),
            lambda p:p['pending'][0].update(placements=[[0,0],[0,3]]),
            lambda p:p['pending'][0].update(placements=[[0,999]])]
        mutations.extend(lambda p,value=value:p['pending'][0].update(attempts=value) for value in (True,False,-1,3,1.0,'1',None))
        for mutate in mutations:
            broken=copy.deepcopy(payload);mutate(broken)
            if broken.get('digest')!='wrong':signed(broken)
            with self.subTest(payload=broken),self.assertRaises(ValueError):store.import_pending(broken)
            self.assertEqual(state(store),before)
        with patch.object(shared,'MAX_PENDING_BYTES',10),self.assertRaises(ValueError):store.import_pending(payload)
        self.assertEqual(state(store),before)
    def test_record_literal_caps_and_live_root_mutation_are_rejected(self):
        problem=ring(34);store=self.store(problem);payload=store.pending_snapshot()
        for key in ('pending','discarded'):
            broken=copy.deepcopy(payload);broken[key]=([{'placements':[],'attempts':0}]*129 if key=='pending' else [[]]*129);signed(broken)
            with self.assertRaisesRegex(ValueError,'record limit'):store.import_pending(broken)
            broken=copy.deepcopy(payload);row=[[cell,0] for cell in range(33)]
            broken[key]=([{'placements':row,'attempts':0}] if key=='pending' else [row]);signed(broken)
            with self.assertRaisesRegex(ValueError,'literal limit'):store.import_pending(broken)
        problem.domains[0].pop()
        with self.assertRaisesRegex(ValueError,'changed'):store.import_pending(payload)
        with self.assertRaisesRegex(ValueError,'changed'):store.pending_snapshot()
    def test_full_worst_shape_fits_separate_cap_and_snapshot_is_owned(self):
        problem=ring(160);store=self.store(problem)
        # 256 different32-literal claims; all cells/codes have three digits.
        records=[tuple([(128,300+3*(index%60))]+[(cell,300+3*(index//60)) for cell in range(129,160)]) for index in range(256)]
        self.assertEqual(len(set(records)),256)
        for record in records[:128]:store.offer(record)
        for record in records[128:]:store._remember_discard(record)
        payload=store.pending_snapshot();self.assertEqual(len(payload['pending']),128);self.assertEqual(len(payload['discarded']),128)
        size=len((json.dumps(payload,indent=2)+'\n').replace('\n','\r\n').encode('utf-8'))
        self.assertGreater(size,shared.MAX_BYTES);self.assertLessEqual(size,shared.MAX_PENDING_BYTES)
        receiver=self.store(problem);self.assertEqual(receiver.import_pending(json.loads(json.dumps(payload))),128)
        self.assertEqual(receiver.pending_snapshot(),payload);self.assertEqual(receiver.snapshot()['records'],[])
        payload['pending'][0]['attempts']=2;payload['discarded'].clear()
        self.assertEqual(list(store._pending.values()),[0]*128);self.assertEqual(len(store._discarded),128)
        before=state(store)
        with patch.object(shared,'MAX_PENDING_BYTES',10),self.assertRaisesRegex(ValueError,'byte limit'):store.pending_snapshot()
        self.assertEqual(state(store),before)
    def test_merge_caps_are_fifo_and_do_not_change_verified_snapshot(self):
        problem=ring(34);receiver=self.store(problem);donor=self.store(problem)
        records=list(itertools.islice(itertools.combinations(range(34),2),256))
        for a,b in records[:128]:receiver.offer([(a,0),(b,0)])
        for a,b in records[128:]:donor.offer([(a,0),(b,0)])
        before=receiver.snapshot();self.assertEqual(receiver.import_pending(donor.pending_snapshot()),128)
        self.assertEqual(receiver.pending_count,128);self.assertEqual(receiver.snapshot(),before)
        self.assertEqual(list(receiver._pending),list(donor._pending));self.assertEqual(receiver.stats['shared_conflicts_pending_dropped'],128)
        # Tombstone eviction bounds storage, not permanent suppression claims.
        for a,b in records:receiver._remember_discard(((a,0),(b,0)))
        self.assertEqual(len(receiver._discarded),128)
        self.assertTrue(receiver.offer([(records[0][0],0),(records[0][1],0)]))
    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_imported_empty_claim_never_prunes_and_verified_api_stays_separate(self):
        problem=ring(4);donor=self.store(problem);donor.offer([]);donor.offer([(0,0),(1,0)])
        receiver=self.store(problem);receiver.import_pending(donor.pending_snapshot())
        self.assertEqual(receiver.records(),[]);self.assertEqual(receiver.snapshot()['records'],[])
        finish(receiver);self.assertTrue(brute(problem,'gold'))
        self.assertEqual(receiver.records(),[[[0,0],[1,0]]])
        self.assertIn((),receiver._discarded)
        for row in brute(problem,'gold'):
            self.assertFalse(all(row[cell]==code for cell,code in receiver.records()[0]))
        before=receiver.snapshot();receiver.import_pending(donor.pending_snapshot());self.assertEqual(receiver.snapshot(),before)
        self.assertEqual(receiver.pending_count,0)
        imported=self.store(problem);imported.import_snapshot(before)
        self.assertEqual(imported.records(),[]);self.assertEqual(imported.pending_count,1)

if __name__=='__main__':unittest.main()
