"""Completed residual proofs are reused through the public exact DFS API."""
import unittest
from constraint_dfs import solve,_solve_core,_checkpoint_digest
from constraint_parallel import _root_copy,partition_checkpoint
from search_options import BOOLEAN_OPTIONS
from test_constraint_separator import ring


class ResidualSearchIntegrationTests(unittest.TestCase):
    def setup_problem(self):
        problem=ring(7)
        options={**dict.fromkeys(BOOLEAN_OPTIONS,False),'residual_reuse':True,'native':'python'}
        base=_solve_core(problem,seconds=0,options=options)['checkpoint']
        a=[(1<<0)|(1<<3)]*3+[1<<9,1<<12,1<<15,1<<18]
        b=a.copy();b[3],b[4]=b[4],b[3]
        return problem,options,base,a,b

    def test_equivalent_completed_subtree_reuses_proof_before_propagation(self):
        problem,options,base,a,b=self.setup_problem();state={}
        first=solve(problem,seconds=5,resume=_root_copy(base,a),options=options,learning_state=state)
        self.assertEqual(first['status'],'infeasible')
        self.assertGreater(first['stats']['residual_closed'],0)
        cache=state['_runtime'].residual
        self.assertEqual(cache.key(a),cache.key(b))
        second=solve(problem,seconds=5,resume=_root_copy(base,a),options=options,learning_state=state)
        self.assertEqual(second['status'],'infeasible')
        self.assertEqual(second['stats']['residual_hits'],1)
        self.assertEqual(second['stats']['propagations'],0)
        self.assertEqual(second['stats']['residual_queries'],1)
        # A job-local empty clause cannot be carried into a different root.
        with self.assertRaises(ValueError):
            solve(problem,seconds=5,resume=base,options=options,learning_state=state)
        with self.assertRaises(ValueError):
            solve(problem,seconds=5,resume=_root_copy(base,b),options=options,learning_state=state)
        third=solve(problem,seconds=5,resume=base,options=options)
        self.assertEqual(third['status'],'solved')
        self.assertTrue(problem.validate(third['codes'])['valid'])

    def test_zero_budget_and_stop_never_remember_unsearched_subtree(self):
        for stopped in (False,True):
            problem,options,base,a,b=self.setup_problem();state={}
            result=solve(problem,seconds=5 if stopped else 0,resume=_root_copy(base,a),
                options=options,learning_state=state,stop=lambda:stopped)
            self.assertIn(result['status'],('timeout','stopped'))
            if '_runtime' in state:self.assertEqual(state['_runtime'].residual.closed,{})
            # A cold completed run is still necessary after cancellation.
            second=solve(problem,seconds=5,resume=_root_copy(base,a),options=options,learning_state=state)
            self.assertEqual(second['status'],'infeasible')
            self.assertEqual(second['stats']['residual_hits'],0)
            self.assertGreater(second['stats']['propagations'],0)

    def test_scope_and_target_are_bound_to_runtime(self):
        problem,options,base,a,b=self.setup_problem();state={}
        solve(problem,seconds=5,resume=_root_copy(base,a),options=options,learning_state=state)
        self.assertEqual(state['_runtime'].residual.root_scope,
                         tuple(a))
        with self.assertRaises(ValueError):
            solve(problem,seconds=5,target='edges',options=options,learning_state=state)


    def test_legacy_restricted_root_gains_stable_scope_without_losing_pending_work(self):
        problem,options,base,a,b=self.setup_problem()
        child=_root_copy(base,a);child.pop('owned_root_domains');child['digest']=_checkpoint_digest(child)
        zero=solve(problem,seconds=0,resume=child,options=options)['checkpoint']
        self.assertEqual(zero['stack'],child['stack'])
        self.assertEqual(zero['owned_root_domains'],child['stack'][0]['domains'])
        state={}
        self.assertEqual(solve(problem,seconds=5,resume=zero,options=options,learning_state=state)['status'],'infeasible')
        self.assertEqual(state['_nogoods'].initial_domains,tuple(a))
        with self.assertRaises(ValueError):solve(problem,seconds=5,resume=base,options=options,learning_state=state)

    def test_explicit_owned_scope_survives_narrower_propagated_first_frame(self):
        problem,options,base,a,b=self.setup_problem();state={}
        # Both checkpoints own the same complete root; their first frames can
        # differ due to completed propagation. This test performs no search in
        # the hand-narrowed snapshot, only validates the stable scope binding.
        solve(problem,seconds=0,resume=base,options=options,learning_state=state)
        narrowed=dict(base);narrowed['stack']=[{'domains':[hex(v) for v in a],'cell':-1,'choices':[],'next':0}]
        narrowed['digest']=_checkpoint_digest(narrowed)
        result=solve(problem,seconds=0,resume=narrowed,options=options,learning_state=state)
        self.assertEqual(result['checkpoint']['owned_root_domains'],base['owned_root_domains'])

    def test_split_jobs_and_native_search_keep_child_root_assumptions(self):
        problem,options,base,a,b=self.setup_problem()
        for native_search in (False,True):
            selected={**options,'native':'auto','native_search':native_search,'smart_learning':True}
            child=_root_copy(base,a);state={}
            result=solve(problem,seconds=5,resume=child,options=selected,learning_state=state)
            self.assertEqual(result['status'],'infeasible')
            self.assertEqual(state['_nogoods'].initial_domains,tuple(a))
            self.assertEqual(state['_runtime'].local_mask_conflicts.initial,tuple(a))
            with self.assertRaises(ValueError):solve(problem,seconds=5,resume=base,options=selected,learning_state=state)
        for child in partition_checkpoint(problem,base,parts=3,options=options):
            self.assertEqual(child['owned_root_domains'],child['stack'][0]['domains'])
            self.assertNotIn('local_mask_conflicts',child)
            solve(problem,seconds=0,resume=child,options=options)


    def test_parent_mask_claims_restrict_to_child_as_unverified_candidates(self):
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        from test_constraint_sat_domains import fixture
        from constraint_parallel import _restricted_mask_claims
        problem,_=fixture();options={'smart_learning':True,'native':'python'}
        parent=_solve_core(problem,seconds=0,options=options)['checkpoint']
        source=VerifiedMaskConflictStore(problem);self.addCleanup(source.close)
        source.offer([(0,3),(1,3)]);payload=source.snapshot()
        parent['local_mask_conflicts']=payload;parent['digest']=_checkpoint_digest(parent)
        domains=list(source.initial);domains[0]=1
        child=_root_copy(parent,domains);state={}
        self.assertEqual(child['local_mask_conflicts']['records'],[[[1,'0x3']]])
        solve(problem,seconds=0,resume=child,options=options,learning_state=state)
        store=state['_runtime'].local_mask_conflicts;self.addCleanup(store.close)
        self.assertEqual(store.initial,tuple(domains));self.assertFalse(store.records())
        self.assertEqual(store.pending_count,1)
        with self.assertRaises(ValueError):_restricted_mask_claims(child['local_mask_conflicts'],list(source.initial))
        impossible=domains.copy();impossible[1]=1<<6
        self.assertEqual(_restricted_mask_claims(payload,impossible)['records'],[])
        # Even a version2 payload never relabels parent-root certificates.
        from constraint_mask_conflicts import _digest
        proof=dict(payload,mask_conflicts_version=2,certificates=[{'untrusted':'parent claim'}]);proof['digest']=_digest(proof)
        self.assertEqual(_restricted_mask_claims(proof,domains)['certificates'],[])

if __name__=='__main__':unittest.main()
