import copy,itertools,unittest,subprocess,sys,os
from constraint_proof import RUPChecker,MaskProofContext,BoundedDPLL,ProofLimit
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,brute
from test_constraint_sat_assist import HAS_SAT


class RUPTests(unittest.TestCase):
    def test_independent_rup_and_malformed_proof(self):
        clauses=((1,2),(1,-2),(-1,2),(-1,-2))
        self.assertTrue(RUPChecker(clauses).verify(['1 0','0']))
        self.assertFalse(RUPChecker(((1,2),)).verify(['1 0','0']))
        self.assertFalse(RUPChecker(((1,),)).verify([]))
        self.assertTrue(RUPChecker(((1,),(-1,))).verify(['0']))
        with self.assertRaises(ValueError):RUPChecker(clauses).verify(['1'])
    def test_transported_mask_proof_needs_no_native_resolve(self):
        p=SearchProblem(simple_tiles(),double_face());initial=tuple(sum(1<<c for c in row) for row in p.domains)
        first=MaskProofContext(p,initial);second=MaskProofContext(p,initial)
        try:
            proof=first.prove([(0,1),(1,1)],seconds=2)
            self.assertIsNotNone(proof);self.assertTrue(second.verify(proof))
            self.assertEqual(second.stats['proof_native_calls'],0)
            self.assertFalse(any(row[0]==row[1]==0 for row in brute(p,'gold')))
            bad=copy.deepcopy(proof);bad['record']=[[0,hex(initial[0])]]
            self.assertFalse(second.verify(bad))
            bad=copy.deepcopy(proof);bad['target']='edges';self.assertFalse(second.verify(bad))
            self.assertIsNone(first.prove([],seconds=2))
        finally:first.close();second.close()

class BoundedProducerTests(unittest.TestCase):
    def test_complete_proofs_match_all_three_variable_cnfs(self):
        import random
        rng=random.Random(240005)
        for _ in range(150):
            clauses=[]
            for k in range(7):
                selected=rng.sample(range(1,4),rng.randrange(1,4))
                clauses.append(tuple(v*rng.choice((-1,1)) for v in selected))
            expected=any(all(any(assignment[abs(lit)-1]==(lit>0) for lit in clause) for clause in clauses)
                         for assignment in itertools.product((False,True),repeat=3))
            producer=BoundedDPLL(clauses);answer=producer.solve()
            self.assertEqual(answer,expected)
            if not answer:self.assertTrue(RUPChecker(clauses).verify(producer.proof or ['0']))

    def test_resource_cap_does_not_return_a_certificate(self):
        clauses=((1,2),(1,-2),(-1,2),(-1,-2))
        with self.assertRaises(ProofLimit):BoundedDPLL(clauses,max_decisions=0).solve()

    def test_pure_proof_child_exits_cleanly(self):
        source="from constraint_proof import BoundedDPLL,RUPChecker; c=((1,2),(1,-2),(-1,2),(-1,-2)); p=BoundedDPLL(c); assert p.solve() is False; assert RUPChecker(c).verify(p.proof); print('verified')"
        child=subprocess.run([sys.executable,'-B','-c',source],cwd=os.path.dirname(__file__),capture_output=True,text=True,timeout=20)
        self.assertEqual(child.returncode,0,(child.stdout,child.stderr));self.assertIn('verified',child.stdout)

    def test_binding_and_malformed_proofs_are_rejected(self):
        p=SearchProblem(simple_tiles(),double_face());initial=[sum(1<<c for c in row) for row in p.domains]
        context=MaskProofContext(p,initial)
        try:
            proof=context.prove([(0,1),(1,1)],seconds=2);self.assertIsNotNone(proof)
            for replacement in (None,['x 0'],['1'],['1 0']*5000):
                broken=copy.deepcopy(proof);broken['proof']=replacement
                self.assertFalse(context.verify(broken))
            p.domains[0]=p.domains[0][1:]
            with self.assertRaises(ValueError):context.verify(proof)
        finally:context.close()


class MaskProofTransportTests(unittest.TestCase):
    def stores(self):
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        p=SearchProblem(simple_tiles(),double_face())
        def absent():raise ImportError('native solver intentionally absent')
        first=VerifiedMaskConflictStore(p,proof_transport=True,factory=absent)
        second=VerifiedMaskConflictStore(p,proof_transport=True,factory=absent)
        self.addCleanup(first.close);self.addCleanup(second.close)
        return p,first,second

    def test_generated_certificate_imports_without_native_sat(self):
        p,first,second=self.stores();first.offer(((0,1),(1,1)))
        self.assertEqual(first.advance(seconds=1),1)
        self.assertEqual(first.stats['mask_proofs_generated'],1)
        payload=first.snapshot();self.assertEqual(payload['mask_conflicts_version'],2)
        self.assertTrue(payload['certificates'])
        second.import_snapshot(payload)
        self.assertEqual(len(second),0)
        self.assertEqual(second.advance(seconds=1),1)
        self.assertEqual(second.stats['mask_conflicts_calls'],0)
        self.assertEqual(second.stats['mask_proofs_verified'],1)
        self.assertIsNone(second._oracle)

    def test_corrupt_proof_cannot_prune_or_skip_ordinary_verification(self):
        p,first,second=self.stores();first.offer(((0,1),(1,1)));first.advance(seconds=1)
        certificate=copy.deepcopy(first.snapshot()['certificates'][0]);certificate['proof']=['not-an-integer 0']
        self.assertTrue(second.offer_certificate(certificate))
        second.advance(seconds=1)
        self.assertGreater(second.stats['mask_proofs_rejected'],0)
        # Pure DPLL may independently re-prove the real claim. The corrupt
        # bytes themselves were rejected; a satisfiable forged scope cannot.
        third=type(second)(p,proof_transport=True,factory=lambda: (_ for _ in ()).throw(ImportError()))
        self.addCleanup(third.close)
        certificate['record']=[];self.assertTrue(third.offer_certificate(certificate))
        third.advance(seconds=1);self.assertEqual(len(third),0)

if __name__=='__main__':unittest.main()
