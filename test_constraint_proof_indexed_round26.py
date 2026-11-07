import copy,itertools,json,random,unittest
from constraint_proof import RUPChecker,BoundedDPLL,_hash
from constraint_proof_indexed_round26 import IndexedRUPChecker


def complete(checker,ticks=7):
    for _ in range(100000):
        result=checker.advance(ticks=ticks)
        if checker.finished:return result
    raise AssertionError('Finite checker did not finish.')


def satisfied(cnf,bits):return all(any(bits[abs(lit)-1]==(lit>0) for lit in row) for row in cnf)

class IndexedProofTests(unittest.TestCase):
    def test_random_completed_proofs_match_independent_truth_enumeration(self):
        rng=random.Random(2604)
        for _ in range(100):
            n=rng.randrange(1,6);cnf=[]
            for _ in range(rng.randrange(1,20)):
                cnf.append(tuple(rng.choice((-1,1))*v for v in rng.sample(range(1,n+1),rng.randrange(1,min(n,3)+1))))
            expected=any(satisfied(cnf,b) for b in itertools.product((False,True),repeat=n))
            producer=BoundedDPLL(cnf);answer=producer.solve()
            self.assertEqual(answer,expected)
            if not expected:
                # An existing empty axiom needs no producer emission; the
                # imported-proof contract still requires an explicit empty step.
                if not producer.proof:producer.proof=['0']
                checker=IndexedRUPChecker(cnf,producer.proof)
                self.assertTrue(complete(checker));self.assertTrue(RUPChecker(cnf).verify(producer.proof))
                hints=checker.generated_hints
                second=IndexedRUPChecker(cnf,producer.proof,hints=hints)
                self.assertTrue(complete(second))
            else:self.assertFalse(complete(IndexedRUPChecker(cnf,['0'])))

    def test_every_microstep_roundtrip_preserves_watch_order_and_work(self):
        cnf=[(1,2,3),(-1,2,3),(1,-2,3),(-1,-2,3),(1,2,-3),(-1,2,-3),(1,-2,-3),(-1,-2,-3)]
        producer=BoundedDPLL(cnf);self.assertFalse(producer.solve());proof=producer.proof
        uninterrupted=IndexedRUPChecker(cnf,proof,context={'root':'fixture'})
        restored=IndexedRUPChecker(cnf,proof,context={'root':'fixture'});phases=set()
        for _ in range(10000):
            phases.add(restored.phase)
            uninterrupted.advance(ticks=1);restored.advance(ticks=1)
            saved=restored.snapshot();restored=IndexedRUPChecker.restore_trusted_local(cnf,proof,json.loads(json.dumps(saved)),context={'root':'fixture'})
            self.assertEqual(uninterrupted.snapshot(),restored.snapshot())
            if restored.finished:break
        self.assertTrue(restored.result);self.assertTrue({'build','proof','units','watch'}<=phases)
        self.assertGreater(restored.stats['proof_watch_moves'],0)

    def test_hints_are_checked_not_authority_and_fallback_is_sound(self):
        cnf=[(1,2),(1,-2),(-1,2),(-1,-2)];proof=['1 0','-1 0','0']
        first=IndexedRUPChecker(cnf,proof);self.assertTrue(complete(first))
        second=IndexedRUPChecker(cnf,proof,hints=first.generated_hints);self.assertTrue(complete(second))
        self.assertGreater(second.stats['proof_hint_visits'],0)
        for hints in ([[999],[],[]],):self.assertFalse(complete(IndexedRUPChecker(cnf,proof,hints=hints)))
        self.assertFalse(complete(IndexedRUPChecker([(1,2)],['0'],hints=[[0]])))
        self.assertTrue(complete(IndexedRUPChecker(cnf,proof,hints=[[],[],[]])))

    def test_tautology_deletion_empty_and_bad_lines(self):
        for cnf,proof,expected in [([(1,),(-1,)],['1 -1 0','d 1 0','0'],True),([],['1 -1 0'],False),([()],['0'],True),([(1,)],['0'],False),([()],['bad 0'],False),([()],['1 0 0'],False)]:
            checker=IndexedRUPChecker(cnf,proof);self.assertEqual(complete(checker),expected)

    def test_stop_is_before_mutation_and_limit_is_unknown(self):
        c=IndexedRUPChecker([(1,),(-1,)],['0']);before=c.snapshot()['state'];calls=[0]
        def stop():calls[0]+=1;raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):c.advance(stop,ticks=1)
        after=c.snapshot()['state'];after['stats']['proof_resumptions']=before['stats']['proof_resumptions']
        self.assertEqual(before,after);self.assertTrue(complete(c))
        capped=IndexedRUPChecker([(1,),(-1,)],['0'],max_visits=0)
        self.assertIsNone(complete(capped));self.assertTrue(capped.limited)
        restored=IndexedRUPChecker.restore_trusted_local([(1,),(-1,)],['0'],capped.snapshot(),max_visits=0)
        self.assertIsNone(restored.advance());self.assertTrue(restored.limited)

    def test_snapshot_binding_and_structure_fail_closed(self):
        cnf=[(1,2),(-1,2),(1,-2),(-1,-2)];proof=['1 0','0'];c=IndexedRUPChecker(cnf,proof,context={'root':1});c.advance(ticks=8)
        saved=c.snapshot()
        for params in [(cnf,proof,{'context':{'root':2}}),(cnf,['0'],{'context':{'root':1}}),(cnf[:-1],proof,{'context':{'root':1}}),(cnf,proof,{'context':{'root':1},'max_visits':10})]:
            with self.assertRaises(ValueError):IndexedRUPChecker.restore_trusted_local(params[0],params[1],saved,**params[2])
        for key,value in [('proof_cursor',999),('positions',[]),('watchers',[]),('event_cursor',999),('finished',1),('result',1),('seen_empty',True)]:
            bad=copy.deepcopy(saved);bad['state'][key]=value;bad['digest']=_hash({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaises(ValueError,msg=key):IndexedRUPChecker.restore_trusted_local(cnf,proof,bad,context={'root':1})
        with self.assertRaises(ValueError):IndexedRUPChecker.restore_trusted_local(cnf,proof,{'proof_format':'diamond-rup-mask-v1'})

    def test_mutation_error_poison_blocks_snapshot_and_resume(self):
        c=IndexedRUPChecker([(1,),(-1,)],['0'])
        def crash():raise OSError('mutation failure')
        c._step=crash
        with self.assertRaises(OSError):c.advance(ticks=1)
        with self.assertRaises(RuntimeError):c.snapshot()
        with self.assertRaises(RuntimeError):c.advance()

    def test_real_mask_store_import_uses_indexed_checker(self):
        from constraint_proof import MaskProofContext
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        from search_problem import SearchProblem
        from test_constraint_dfs import simple_tiles,double_face
        p=SearchProblem(simple_tiles(),double_face());root=tuple(sum(1<<code for code in row) for row in p.domains)
        producer=MaskProofContext(p,root)
        try:certificate=producer.prove([(0,1),(1,1)],seconds=2)
        finally:producer.close()
        self.assertIsNotNone(certificate)
        store=VerifiedMaskConflictStore(p,proof_indexing=True)
        try:
            self.assertTrue(store.offer_certificate(certificate))
            for _ in range(100):
                store.advance(seconds=1)
                if len(store):break
            self.assertEqual(len(store),1)
            self.assertGreater(store.stats['mask_proof_index_literals'],0)
            self.assertEqual(store.stats['mask_proofs_verified'],1)
            old=store._proof_context;store.configure_policy(proof_transport=True,resumable_proof_checking=True,proof_indexing=False)
            self.assertFalse(old._verification_cursors);self.assertEqual(len(store),1)
        finally:store.close()

    def test_clause_index_avoids_rescanning_unrelated_formula(self):
        result=compare_indexed_checking();self.assertLess(result['indexed']['proof_clause_visits'],result['scan']['proof_clause_visits'])
        self.assertGreater(result['indexed']['proof_index_build_literals'],0)
        self.assertEqual(result['indexed']['proof_steps'],result['scan']['proof_steps'])


def compare_indexed_checking():
    cnf=[(v,v+1) for v in range(20,420)]+[(1,2),(1,-2),(-1,2),(-1,-2)]
    proof=['1 0','-1 0','0'];scan=RUPChecker(cnf);indexed=IndexedRUPChecker(cnf,proof)
    if not scan.verify(proof) or complete(indexed) is not True:raise AssertionError('Independent checker comparison disagrees.')
    hinted=IndexedRUPChecker(cnf,proof,hints=indexed.generated_hints)
    if complete(hinted) is not True:raise AssertionError('Derived hints failed independent replay.')
    return {'scan':scan.stats,'indexed':indexed.stats,'hinted':hinted.stats,'base_clauses':len(cnf),'proof_lines':len(proof)}

if __name__=='__main__':unittest.main()
