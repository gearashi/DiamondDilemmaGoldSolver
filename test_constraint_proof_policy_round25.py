"""Warm proof-policy switches preserve authority and change subsequent work."""
import copy
import unittest
from unittest.mock import patch

from constraint_dfs import solve, _solve_core
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_mixed_proof import MixedProofContext
from constraint_proof import MaskProofContext
from search_options import BOOLEAN_OPTIONS, normalize_options
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, multiple_loops, double_face
from test_constraint_sat_assist import GoldFixtureOracle


class StorePolicyTests(unittest.TestCase):
    def fixture(self,*,loops=False,**options):
        problem=SearchProblem(multiple_loops() if loops else simple_tiles(),double_face())
        store=VerifiedMaskConflictStore(problem,factory=lambda:GoldFixtureOracle(problem),**options)
        self.addCleanup(store.close)
        return problem,store

    def certificate(self,problem):
        initial=tuple(sum(1<<code for code in row) for row in problem.domains)
        producer=MaskProofContext(problem,initial);self.addCleanup(producer.close)
        certificate=producer.prove([(0,1),(1,1)],seconds=2)
        self.assertIsNotNone(certificate)
        return producer,certificate

    def test_plain_mixed_plain_preserves_verified_records_oracle_and_counters(self):
        _,store=self.fixture()
        store.offer(((0,1),(1,1)))
        self.assertEqual(store.advance(seconds=1),1)
        oracle=store._oracle;records=store.records();verified=store.stats['mask_conflicts_verified']
        self.assertTrue(store.configure_policy(mixed_proof_dag=True))
        self.assertIs(store._oracle,oracle)
        self.assertIsInstance(store._proofs(),MixedProofContext)
        self.assertEqual(store.records(),records)
        store.advance(seconds=1)
        total=store.stats['mixed_proof_nodes_checked']
        self.assertGreater(total,0)
        self.assertTrue(store.configure_policy())
        self.assertFalse(store.mixed_proof_dag)
        self.assertFalse(store.resumable_proof_checking)
        self.assertFalse(store.proof_transport)
        self.assertIsNone(store._proof_context)
        self.assertIs(store._oracle,oracle)
        self.assertEqual(store.records(),records)
        self.assertEqual(store.stats['mask_conflicts_verified'],verified)
        self.assertEqual(store.stats['mixed_proof_nodes_checked'],total)
        domain=[1,1];self.assertTrue(store.prune(domain));self.assertIn(0,domain)
        self.assertEqual(store.snapshot()['mask_conflicts_version'],1)
        self.assertEqual(store.advance(seconds=1),0)
        self.assertIsNone(store._proof_context)
        self.assertFalse(store.configure_policy())

    def test_enabling_mixed_reconsiders_edge_sat_claim_without_trusting_it(self):
        _,store=self.fixture(loops=True)
        record=((0,1),(1,8))
        store.offer(record);self.assertEqual(store.advance(seconds=1),0)
        self.assertIn(record,store._discarded)
        self.assertFalse(store.records())
        oracle=store._oracle
        store.configure_policy(mixed_proof_dag=True)
        self.assertIn(record,store._pending)
        self.assertFalse(store.records())
        self.assertEqual(store.advance(seconds=1),1)
        self.assertIs(store._oracle,oracle)
        self.assertIn(record,store.records())
        self.assertGreater(store.stats['mask_proofs_generated'],0)

    def test_resumable_toggle_closes_cursor_preserves_claim_and_accumulates_stats(self):
        problem,store=self.fixture(proof_transport=True,resumable_proof_checking=True)
        producer,certificate=self.certificate(problem)
        certificate['proof']=[' '.join(map(str,(*producer.base[0],0)))]*600+certificate['proof']
        self.assertTrue(store.offer_certificate(certificate))
        self.assertEqual(store.advance(seconds=1),0)
        old=store._proof_context;self.assertTrue(old._verification_cursors)
        before=store.stats['mask_proof_clause_visits'];self.assertGreater(before,0)
        self.assertFalse(store.records())
        store.configure_policy(mixed_proof_dag=True)
        self.assertFalse(old._verification_cursors)
        self.assertIsNone(store._proof_context)
        self.assertTrue(store._certificate_pending)
        self.assertFalse(store.records())
        self.assertEqual(store.advance(seconds=1),0)
        after=store.stats['mask_proof_clause_visits'];self.assertGreater(after,before)
        self.assertIsInstance(store._proof_context,MixedProofContext)
        store.configure_policy(proof_transport=True,resumable_proof_checking=False)
        self.assertFalse(store.resumable_proof_checking)
        self.assertFalse(store.mixed_proof_dag)
        self.assertFalse(store.records())
        self.assertEqual(store.advance(seconds=1),1)
        self.assertIs(type(store._proof_context),MaskProofContext)
        self.assertGreaterEqual(store.stats['mask_proof_clause_visits'],after)
        self.assertEqual(store.stats['mask_conflicts_verified'],1)
        store.close()
        self.assertGreaterEqual(store.stats['mask_proof_clause_visits'],after)

    def test_unverified_or_forged_certificates_never_gain_authority_from_toggle(self):
        problem,store=self.fixture(proof_transport=True,resumable_proof_checking=True)
        _,certificate=self.certificate(problem)
        forged=copy.deepcopy(certificate);forged['record']=[];forged['proof']=['0']
        self.assertTrue(store.offer_certificate(forged))
        for options in ({},{'resumable_proof_checking':True},{'mixed_proof_dag':True}):
            store.configure_policy(**options)
            self.assertFalse(store.records())
        store.advance(seconds=1)
        self.assertFalse(store.records())
        self.assertEqual(store.stats['mask_proofs_verified'],0)
        for kwargs in ({'mixed_proof_dag':'yes'},{'resumable_proof_checking':1}):
            with self.assertRaises(ValueError):store.configure_policy(**kwargs)

    def test_disabling_mixed_preserves_pending_claim_but_drops_incompatible_bytes(self):
        problem,store=self.fixture(loops=True,mixed_proof_dag=True)
        source=MixedProofContext(problem,store.initial);self.addCleanup(source.close)
        certificate=source.prove(((0,1),(1,8)),seconds=2)
        self.assertTrue(store.offer_certificate(certificate))
        store.configure_policy(proof_transport=True)
        self.assertFalse(store._certificate_pending)
        self.assertEqual(store.pending_count,1)
        self.assertFalse(store.records())
        store.advance(seconds=1)
        self.assertFalse(store.records())
        self.assertNotIsInstance(store._proof_context,MixedProofContext)


class WarmRunnerPolicyTests(unittest.TestCase):
    @staticmethod
    def options(**extra):
        return {**dict.fromkeys(BOOLEAN_OPTIONS,False),'native':'python','smart_learning':True,**extra}

    def test_actual_warm_dfs_reconfigures_both_original_and_job_stores(self):
        problem=SearchProblem(simple_tiles(),double_face())
        plain=self.options();base=_solve_core(problem,seconds=0,options=plain)['checkpoint'];state={}
        answer=solve(problem,seconds=0,resume=base,learning_state=state,options=plain)
        self.assertEqual(answer['status'],'timeout')
        stores=(state['_mask_conflicts'],state['_runtime'].local_mask_conflicts)
        for store in stores:self.addCleanup(store.close)
        configurations=({'mixed_proof_dag':True},{'proof_transport':True},
                        {'resumable_proof_checking':True},{})
        for configuration in configurations:
            options=self.options(**configuration)
            answer=solve(problem,seconds=0,resume=base,learning_state=state,options=options)
            self.assertEqual(answer['status'],'timeout')
            self.assertIs(state['_mask_conflicts'],stores[0])
            self.assertIs(state['_runtime'].local_mask_conflicts,stores[1])
            expected=normalize_options(options)
            for store in stores:
                self.assertEqual(store.mixed_proof_dag,expected['mixed_proof_dag'])
                self.assertEqual(store.resumable_proof_checking,expected['resumable_proof_checking'])
                self.assertEqual(store.proof_transport,expected['proof_transport'])
                if store.proof_transport:
                    self.assertIs(type(store._proofs()),MixedProofContext if store.mixed_proof_dag else MaskProofContext)
                else:self.assertIsNone(store._proof_context)

    def test_actual_warm_parallel_parent_updates_existing_policy_without_workers(self):
        from constraint_parallel import solve as parallel_solve
        problem=SearchProblem(simple_tiles(),double_face());state={}
        parallel_solve(problem,seconds=0,learning_state=state,options=self.options())
        store=state['_mask_conflicts'];self.addCleanup(store.close)
        for configuration in ({'mixed_proof_dag':True},{'resumable_proof_checking':True},{}):
            options=self.options(**configuration)
            parallel_solve(problem,seconds=0,learning_state=state,options=options)
            expected=normalize_options(options)
            self.assertIs(state['_mask_conflicts'],store)
            self.assertEqual(store.mixed_proof_dag,expected['mixed_proof_dag'])
            self.assertEqual(store.resumable_proof_checking,expected['resumable_proof_checking'])

    def test_hybrid_repair_store_receives_policy_before_exact_dfs(self):
        import gpu_sampling
        problem=SearchProblem(multiple_loops(),double_face())
        options=normalize_options(self.options(mixed_proof_dag=True))
        for existing in (False,True):
            state={}
            if existing:
                state['_mask_conflicts']=VerifiedMaskConflictStore(problem)
                self.addCleanup(state['_mask_conflicts'].close)
            original=state.get('_mask_conflicts')
            def exact(*args,**kwargs):
                store=state['_mask_conflicts']
                if original is not None:self.assertIs(store,original)
                self.assertTrue(store.mixed_proof_dag)
                self.assertTrue(store.resumable_proof_checking)
                return {'status':'timeout','stats':{},'elapsed_seconds':0,'complete':False}
            with patch('gpu_sampling.sample',return_value={'top_samples':[{'codes':[0,3]}]}), \
                 patch('constraint_lns.repair',return_value={'experimental_repair':[]}), \
                 patch('constraint_dfs.solve',side_effect=exact):
                result=gpu_sampling.solve(problem,seconds=1,learning_state=state,options=options)
            self.assertEqual(result['status'],'timeout')
            if original is None:self.addCleanup(state['_mask_conflicts'].close)


if __name__=='__main__':unittest.main()
