import itertools,json,unittest,time
from unittest.mock import patch
from constraint_sat_session import PersistentSatSession,solve
from constraint_work import WorkScope,restore_frontier,frontier_snapshot,digest
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,tetrahedron_tiles,brute
from test_constraint_sat_assist import HAS_SAT

@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class PersistentSessionTests(unittest.TestCase):
    def test_complete_targets_match_independent_enumeration(self):
        for p in [SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),SearchProblem(*tetrahedron_tiles())]:
            for target in ('gold','edges'):
                expected=brute(p,target);answer=solve(p,seconds=5,target=target)
                self.assertTrue(answer['complete'],answer)
                if expected:self.assertIn(tuple(answer['codes']),expected)
                else:self.assertEqual(answer['status'],'infeasible')
    def test_empty_slice_does_not_consume_frontier(self):
        p=SearchProblem(simple_tiles(),double_face());scope=WorkScope.for_problem(p)
        session=PersistentSatSession(p,scope)
        try:
            first=session.advance(seconds=0)
            self.assertEqual(first['status'],'timeout')
            self.assertEqual(restore_frontier(first['checkpoint']['frontier'],scope),(scope.root,))
            resumed=PersistentSatSession.restore(p,json.loads(json.dumps(first['checkpoint'])))
            try:self.assertEqual(resumed.advance(seconds=5)['status'],'solved')
            finally:resumed.close()
        finally:session.close()
    def test_completed_cubes_are_absent_after_restart(self):
        p=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(p)
        rows=[(1<<a,1<<b) for a in range(6) for b in range(6)]
        session=PersistentSatSession(p,scope,rows)
        stop=[False]
        try:
            first=session.advance(seconds=5,stop=lambda:stop[0],progress=lambda event:stop.__setitem__(0,True))
            self.assertEqual(first['status'],'stopped');self.assertEqual(len(session.boxes),35)
            saved=json.loads(json.dumps(first['checkpoint']));restored=PersistentSatSession.restore(p,saved)
            try:
                self.assertEqual(len(restored.boxes),35)
                self.assertEqual(restored.advance(seconds=5)['status'],'infeasible')
                self.assertEqual(restored.stats['session_cubes_closed'],36)
            finally:restored.close()
        finally:session.close()
    def test_warm_slices_build_model_once(self):
        p=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(p)
        session=PersistentSatSession(p,scope,[(1<<a,1<<b) for a in range(6) for b in range(6)])
        try:
            for _ in range(40):
                stop=[False]
                answer=session.advance(seconds=5,stop=lambda:stop[0],progress=lambda event:stop.__setitem__(0,True))
                if answer['complete']:break
            self.assertTrue(answer['complete']);self.assertEqual(session.stats['session_model_builds'],1)
            self.assertGreater(session.stats['session_warm_slices'],1)
        finally:session.close()
    def test_incompatible_objective_rejected(self):
        p=SearchProblem(simple_tiles(),double_face());s=PersistentSatSession(p,WorkScope.for_problem(p))
        try:
            with self.assertRaises(ValueError):PersistentSatSession.restore(p,s.snapshot(),'edges')
        finally:s.close()

class SessionScopeTests(unittest.TestCase):
    def test_scope_checks_original_domains_not_only_fingerprint(self):
        p=SearchProblem(simple_tiles(),double_face());scope=WorkScope.for_problem(p)
        other=SearchProblem(simple_tiles(),double_face());other.domains[0]=[0]
        self.assertEqual(p.fingerprint,other.fingerprint)
        with self.assertRaises(ValueError):PersistentSatSession(other,scope)
        session=PersistentSatSession(p,scope)
        try:
            with self.assertRaises(ValueError):solve(other,seconds=0,learning_state={'_sat_session':session})
        finally:session.close()

    def test_edge_target_refuses_valid_gold_only_cut(self):
        p=SearchProblem(multiple_loops(),double_face());session=PersistentSatSession(p,WorkScope.for_problem(p,'edges'))
        try:
            payload=session.snapshot();payload['gold_cuts']=[p.gold_nogood([0,3])]
            payload['digest']=digest({k:v for k,v in payload.items() if k!='digest'})
            with self.assertRaisesRegex(ValueError,'Gold-only'):PersistentSatSession.restore(p,payload,'edges')
        finally:session.close()

    def test_warm_session_can_replace_native_state_with_portable_frontier(self):
        p=SearchProblem(simple_tiles(),double_face());scope=WorkScope.for_problem(p);state={}
        solve(p,seconds=0,learning_state=state);prior=state['_sat_session']
        payload=frontier_snapshot(scope,[(1,scope.root[1])])
        try:
            answer=solve(p,seconds=0,learning_state=state,resume=payload)
            self.assertIsNot(state['_sat_session'],prior)
            self.assertEqual(answer['checkpoint']['frontier']['boxes'],payload['boxes'])
            # Bad incoming state must not destroy a reusable warm session.
            current=state['_sat_session'];bad=dict(payload,digest='bad')
            with self.assertRaises(ValueError):solve(p,seconds=0,learning_state=state,resume=bad)
            self.assertIs(state['_sat_session'],current)
        finally:state['_sat_session'].close()

    def test_completed_false_survives_stop_unknown_does_not(self):
        from constraint_sat_session import SessionInterrupted
        class FakeNative:
            def __init__(self,answer):self.answer=answer
            def clear_interrupt(self):pass
            def conf_budget(self,n):pass
            def interrupt(self):pass
            def solve_limited(self,**kwargs):time.sleep(.04);return self.answer
            def accum_stats(self):return {'decisions':7}
        p=SearchProblem(simple_tiles(),double_face());session=PersistentSatSession(p,WorkScope.for_problem(p))
        def stop():raise SessionInterrupted('stopped')
        try:
            session.selector=1;session.solver=FakeNative(False)
            self.assertIs(session._solve_once(stop,time.monotonic()+2),False)
            self.assertEqual(session.stats['session_native_calls'],1)
            session.solver=FakeNative(None)
            with self.assertRaises(SessionInterrupted):session._solve_once(stop,time.monotonic()+2)
            session.solver=FakeNative(False)
            with self.assertRaisesRegex(RuntimeError,'callback'):
                session._solve_once(lambda:(_ for _ in ()).throw(RuntimeError('callback')),time.monotonic()+2)
        finally:session.solver=None;session.close()

    def test_progress_contract_contains_phase_and_real_checkpoint(self):
        p=SearchProblem(simple_tiles(),double_face());events=[]
        def advance(session,**kwargs):
            event=session.result('searching',time.monotonic());kwargs['progress'](event)
            return session.result('timeout',time.monotonic())
        with patch.object(PersistentSatSession,'advance',advance):solve(p,seconds=0,progress=events.append)
        self.assertEqual(events[0]['phase'],'searching')
        self.assertEqual(events[0]['checkpoint']['schema'],'diamond-sat-session-v1')
        self.assertNotEqual(events[0].get('event'),'checkpoint')

if __name__=='__main__':unittest.main()
