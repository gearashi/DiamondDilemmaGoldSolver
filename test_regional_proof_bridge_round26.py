"""Independent regional certificate transport into original-root mask learning."""
import copy,json,unittest
from unittest.mock import patch
from constraint_regions import HierarchicalRegionPruner
from constraint_region_proof_round26 import build_core
from constraint_region_compiler_round26 import _digest
from constraint_regional_proof_bridge_round26 import RegionalProofQueue,verify_reason
from constraint_deduction_chain_round26 import verify_chain,ChainError
from constraint_mask_conflicts import VerifiedMaskConflictStore
from test_constraint_separator import ring
from test_constraint_advanced import domains_of
from test_constraint_dfs import brute,multiple_loops,double_face
from search_problem import SearchProblem


def chain(problem,domains,data,target='gold'):
    return {'format':'diamond-deduction-chain-v1','fingerprint':problem.fingerprint,'target':target,
      'original':[hex(v) for v in domains_of(problem)],'root':[hex(v) for v in domains],
      'decisions':[],'events':[],'terminal':{'rule':'regional','data':data}}

class RegionalProofBridgeTests(unittest.TestCase):
    def core(self):
        p=ring(4);domains=domains_of(p);domains[0]=domains[1]=1
        record=build_core(HierarchicalRegionPruner(p,'gold'),(0,1),domains,generalize=0)
        self.assertIsNotNone(record);return p,domains,record
    def test_full_dsu_certificate_admits_guard_without_sat_and_preserves_all_solutions(self):
        p,domains,record=self.core();claim=chain(p,domains,{'kind':'regional_core','certificate':record})
        stats={};guards=verify_chain(p,claim,stats=stats)
        self.assertTrue(guards);self.assertGreater(stats['chain_regional_proof_rows'],0)
        answers=brute(p,'gold');self.assertTrue(answers)
        self.assertFalse(any(all((mask>>board[cell])&1 for cell,mask in guards) for board in answers))
        store=VerifiedMaskConflictStore(p,'gold')
        self.assertGreater(store.admit_deduction_chain(claim),0)
        self.assertEqual(store.stats.get('mask_conflicts_verifier_queries',0),0)
    def test_rehashed_missing_dead_child_and_wider_assumptions_are_rejected(self):
        p,domains,record=self.core();bad=copy.deepcopy(record)
        bad['certificate']['states'][0]['edges']=[];bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises((ChainError,ValueError)):verify_chain(p,chain(p,domains,{'kind':'regional_core','certificate':bad}))
        with self.assertRaises((ChainError,ValueError)):verify_chain(p,chain(p,domains_of(p),{'kind':'regional_core','certificate':record}))
    def test_actual_connectivity_reason_is_gold_only_and_scope_bound(self):
        from constraint_connectivity_round26 import ConnectivityObligations
        p=SearchProblem(multiple_loops(),double_face());before=domains_of(p)
        stage=ConnectivityObligations(HierarchicalRegionPruner(p,'gold'));current=before.copy();stage.prune(current)
        self.assertFalse(all(current));self.assertTrue(stage.reasons)
        valid=[]
        for reason in stage.reasons:
            data={'kind':'connectivity','certificate':reason}
            if verify_reason(p,reason['domains'],data):valid.append(data)
        self.assertTrue(valid)
        verify_chain(p,chain(p,valid[0]['certificate']['domains'],valid[0]))
        with self.assertRaises((ChainError,ValueError)):verify_chain(p,chain(p,before,valid[0],target='edges'),target='edges')
    def test_fresh_semantic_replay_all_origins_never_trusts_a_cursor(self):
        p,domains,_=self.core()
        for kind in ('compiler','overlap','recursive','integer'):
            with self.subTest(origin=kind):
                queue=RegionalProofQueue(p,'gold');queue.capture(kind,domains,[0]*p.n)
                records=list(queue.drain());self.assertEqual(len(records),1)
                self.assertEqual(records[0]['terminal']['data']['certificate']['origin'],kind)
                self.assertTrue(verify_chain(p,records[0]));self.assertGreater(queue.stats['regional_chain_build_transitions'],0)
        fake={'kind':'regional_replay','certificate':{'origin':'integer','proof':{'complete':True,'feasible':False}}}
        with self.assertRaises((ChainError,ValueError)):verify_chain(p,chain(p,domains,fake))
    def test_candidate_conflict_replays_under_explicit_singleton_guard(self):
        p=ring(4);before=domains_of(p);before[0]=1;after=before.copy();after[1]&=~1
        queue=RegionalProofQueue(p,'gold');queue.capture('recursive',before,after)
        record=list(queue.drain())[0];guards=verify_chain(p,record)
        self.assertIn((0,1),guards);self.assertIn((1,1),guards)
        for answer in brute(p,'gold'):self.assertFalse(all(mask>>answer[cell]&1 for cell,mask in guards))
    def test_caps_and_cancellation_do_not_publish_or_consume_pending_proof(self):
        p,domains,_=self.core();queue=RegionalProofQueue(p,'gold');queue.capture('integer',domains,[0]*p.n)
        def stopped():raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):list(queue.drain(stopped))
        self.assertEqual(len(queue.pending),1)
        records=list(queue.drain(max_states=1,max_steps=1));self.assertEqual(records,[]);self.assertGreater(queue.stats['regional_chain_caps'],0)
        queue=RegionalProofQueue(p,'gold');queue.capture('integer',domains,[0]*p.n)
        iterator=queue.drain();record=next(iterator);before=queue.stats['regional_chain_build_transitions'];iterator.close()
        self.assertEqual(len(queue.pending),1)
        self.assertEqual(list(queue.drain()),[record]);self.assertEqual(queue.stats['regional_chain_build_transitions'],before)
        self.assertFalse(queue.pending)
    def test_callback_value_error_and_live_scope_mutation_propagate(self):
        p,domains,record=self.core();data={'kind':'regional_core','certificate':record}
        def stop():raise ValueError('user stop')
        with self.assertRaisesRegex(ValueError,'user stop'):verify_reason(p,domains,data,stop)
        queue=RegionalProofQueue(p,'gold');queue.capture('integer',domains,[0]*p.n)
        p.domains[0]=p.domains[0][1:]
        with self.assertRaisesRegex(ValueError,'scope changed'):list(queue.drain())
    def test_production_stage_queue_is_bounded_and_all_stage_paths_transport(self):
        from constraint_regions_round26 import RegionalWorkstreams
        for option in ('regional_compiler','recursive_regions','integer_regional_master','regional_proof_reuse','connectivity_obligations'):
            with self.subTest(option=option):
                p,domains,_=self.core();stage=RegionalWorkstreams(p,'gold',{option:True,'smart_learning':True})
                for _ in range(100):
                    stage.prune(domains)
                    if not all(domains):break
                self.assertFalse(all(domains));claims=list(stage.drain_deduction_chains())
                self.assertTrue(claims)
                for record in claims:verify_chain(p,record)
                self.assertLessEqual(len(stage.proof_queue.pending),16)

    def test_public_dfs_store_admission_and_stop_resume_preserve_exact_result(self):
        from constraint_dfs import solve
        from constraint_connectivity import ConnectivityPruner
        from constraint_regions_round26 import KEYS
        from search_options import BOOLEAN_OPTIONS
        p=SearchProblem(multiple_loops(),double_face());expected=brute(p,'gold');self.assertFalse(expected)
        options={key:False for key in BOOLEAN_OPTIONS};options.update({key:True for key in KEYS});options.update(smart_learning=True,native='python')
        real=VerifiedMaskConflictStore.admit_deduction_chain;calls=[];cancel=[False]
        def interrupt(store,claim,check=lambda:None,**kwargs):
            calls.append(claim);cancel[0]=True;return real(store,claim,check,**kwargs)
        owners={}
        with patch.object(ConnectivityPruner,'prune',return_value=False),patch.object(VerifiedMaskConflictStore,'admit_deduction_chain',interrupt):
            first=solve(p,seconds=10,options=options,learning_state=owners,stop=lambda:cancel[0])
        self.assertTrue(calls);self.assertEqual(first['status'],'stopped');self.assertFalse(first['complete'])
        cancel[0]=False;admitted=[]
        def admit(store,claim,check=lambda:None,**kwargs):
            answer=real(store,claim,check,**kwargs);admitted.append(answer);return answer
        with patch.object(ConnectivityPruner,'prune',return_value=False),patch.object(VerifiedMaskConflictStore,'admit_deduction_chain',admit):
            tail=solve(p,seconds=10,options=options,learning_state=owners,resume=first['checkpoint'])
        self.assertEqual(tail['status'],'infeasible');self.assertTrue(tail['complete'])
        # The completed contradictory domain update is already in the exact
        # checkpoint, so resume may finish before optional proof admission.
        # A fresh public run separately exercises actual store publication.
        with patch.object(ConnectivityPruner,'prune',return_value=False),patch.object(VerifiedMaskConflictStore,'admit_deduction_chain',admit):
            fresh=solve(p,seconds=10,options=options)
        self.assertEqual(fresh['status'],'infeasible');self.assertTrue(admitted)
        self.assertGreater(fresh['stats']['chain_regional_proof_checks'],0)
