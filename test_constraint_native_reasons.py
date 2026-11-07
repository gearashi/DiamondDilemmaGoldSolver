"""Native reason logs checked against independent complete-board enumeration."""
import itertools
import random
import unittest
from unittest.mock import patch
from dataclasses import replace
from constraint_lcg import LCGStore
from constraint_native_learning import replay_fast_failure, verify_native_reasons, explain_fast_failure
from constraint_native_search import FastFailure, NativeSearch
from test_constraint_native_search import NativeSearchTests, callback, frame, puzzle_tables
from test_constraint_native_learning import oracle, violates


def compare_native_explanations():
    rng=random.Random(11);n=4;initial=[4095]*n
    patterns=[[rng.randrange(2) for _ in range(3)] for _ in range(12)]
    edges=[(0,0,1,0),(1,1,2,0),(2,1,3,0),(3,1,0,1)]
    tables=puzzle_tables(n,patterns,edges);expected=oracle(initial,patterns,edges)
    reports={}
    for enabled in (False,True):
        engine=NativeSearch(n);engine.configure_propagation(tables)
        engine.configure_explanations(enabled)
        store=LCGStore('finite','edges',initial);leaves=[];checks={};clauses=[]
        basic=callback(leaves=leaves,distinct=False)
        def process(domains,decisions):
            store.prune(domains,lambda:None)
            if not all(domains):return {'action':'reject'}
            return basic(domains,decisions)
        def event(name,payload):
            if name!='fast_reject':return
            explain=explain_fast_failure if enabled else replay_fast_failure
            clause=explain(payload['native_failure'],initial=initial,tables=tables,
                           fingerprint='finite',target='edges',stats=checks)
            if clause is not None:
                clauses.append(clause);store.learn(clause)
                pending=store.asserting_frontier(engine.snapshot(),clause,require_jump=True)
                if pending is not None:return {'replace_stack':pending}
        try:
            result=engine.run([frame(initial)],process,on_event=event,capture_fast_reject=True)
        finally:engine.close()
        assert result['status']=='exhausted'
        assert set(leaves)==expected and len(leaves)==len(expected)
        assert all(not violates(board,clause) for clause in clauses for board in expected)
        reports['recorded' if enabled else 'replayed']={
            'solutions':len(leaves),'nodes':result['stats']['native_search_nodes'],
            'checker_work':checks.get('native_learning_checks',0)+checks.get('native_reason_checks',0),
            'stats':checks,'native_stats':result['stats']}
    return reports


class ReasonCheckerTests(unittest.TestCase):
    def test_requested_native_explanations_require_available_native_search(self):
        from constraint_dfs import solve
        from search_problem import SearchProblem
        from test_constraint_dfs import tetrahedron_tiles
        with patch('constraint_native_search.NativeSearch',side_effect=RuntimeError('missing test library')):
            with self.assertRaisesRegex(RuntimeError,'Compiled search unavailable'):
                solve(SearchProblem(*tetrahedron_tiles()),seconds=5,options={'native_explanations':True})

    def test_duplicate_piece_reason_omits_unrelated_decision(self):
        failure=FastFailure((3,6,448),((0,3),(2,448),(1,6)),((4,0,1,0,0,0,0),))
        tables=puzzle_tables(3,[[0]*3]*9,[]);stats={}
        clause=verify_native_reasons(failure,initial=[511]*3,tables=tables,fingerprint='finite',stats=stats)
        self.assertEqual(clause.literals,((0,3),(1,6)))
        self.assertEqual(stats['native_reason_checks'],1)
        self.assertEqual(stats['native_learning_replays'],0)
        for board in oracle([511]*3,[[0]*3]*9,[]):self.assertFalse(violates(board,clause))
        for forged in (((4,0,0,0,0,0,0),),((4,0,1,0,0,1,0),),((9,0,1,0,0,0,0),)):
            self.assertIsNone(verify_native_reasons(replace(failure,reasons=forged),initial=[511]*3,
                                                    tables=tables,fingerprint='finite'))

    def test_unexplained_carried_domains_are_never_assumptions(self):
        tables=puzzle_tables(2,[[0]*3]*6,[])
        failure=FastFailure((3,6),(),((4,0,1,0,0,0,0),))
        self.assertIsNone(verify_native_reasons(failure,initial=[63,63],tables=tables,fingerprint='finite'))
        self.assertIsNone(explain_fast_failure(failure,initial=[63,63],tables=tables,fingerprint='finite'))
        self.assertEqual(verify_native_reasons(failure,initial=[3,6],tables=tables,fingerprint='finite').literals,())

    def test_wrong_arc_support_and_bounded_checks(self):
        tables=puzzle_tables(2,[[0]*3]*6,[(0,0,1,0)])
        failure=FastFailure((63,63),(),((1,0,1,0,0,0,63),))
        self.assertIsNone(verify_native_reasons(failure,initial=[63,63],tables=tables,fingerprint='finite'))
        duplicate=FastFailure((3,6),(),((4,0,1,0,0,0,0),))
        self.assertIsNone(verify_native_reasons(duplicate,initial=[3,6],tables=tables,fingerprint='finite',max_work=0))
        def cancel():raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):
            verify_native_reasons(duplicate,initial=[3,6],tables=tables,fingerprint='finite',check=cancel)


class NativeReasonTests(unittest.TestCase):
    setUpClass=classmethod(NativeSearchTests.setUpClass.__func__)
    engine=NativeSearchTests.engine

    def test_complete_search_keeps_all_solutions_with_less_checked_work(self):
        r=compare_native_explanations()
        self.assertEqual(r['recorded']['solutions'],152)
        self.assertGreater(r['recorded']['stats']['native_reason_clauses'],0)
        self.assertLess(r['recorded']['checker_work'],r['replayed']['checker_work'])

    def test_native_failure_keeps_trace_after_pop_and_frontier_replacement(self):
        engine=self.engine(3);tables=puzzle_tables(3,[[0]*3]*9,[])
        engine.configure_propagation(tables);engine.configure_explanations(True)
        seen=[]
        def event(name,payload):
            if name=='fast_reject':
                failure=payload['native_failure'];seen.append(failure)
                self.assertEqual(failure.reasons[-1][0],4)
                clause=verify_native_reasons(failure,initial=[3,6,448],tables=tables,fingerprint='finite')
                self.assertEqual(clause.literals,())
        result=engine.run([frame([3,6,448])],lambda *_:self.fail('native rejection expected'),
                          on_event=event,capture_fast_reject=True)
        self.assertEqual(result['status'],'exhausted');self.assertEqual(len(seen),1)
        self.assertEqual(result['stack'],[])
        engine._replace_frontier([frame([7,56,448])])
        self.assertEqual(engine.lib.dd_search_reason_count(engine._handle),-1)

    def test_random_native_traces_never_exclude_an_independent_solution(self):
        rng=random.Random(250004);proved=0
        for _ in range(25):
            n=3;patterns=[[rng.randrange(3) for _ in range(3)] for _ in range(9)]
            edges=[(0,0,1,1),(1,2,2,0),(2,1,0,2)]
            tables=puzzle_tables(n,patterns,edges);initial=[511]*3;expected=oracle(initial,patterns,edges)
            engine=NativeSearch(n);engine.configure_propagation(tables);engine.configure_explanations(True)
            leaves=[]
            def event(name,payload):
                nonlocal proved
                if name=='fast_reject':
                    clause=verify_native_reasons(payload['native_failure'],initial=initial,tables=tables,
                                                 fingerprint='random',target='edges')
                    if clause is not None:
                        proved+=1
                        self.assertTrue(all(not violates(board,clause) for board in expected))
            try:
                result=engine.run([frame(initial)],callback(leaves=leaves,distinct=False),
                                  on_event=event,capture_fast_reject=True)
                self.assertEqual(set(leaves),expected);self.assertEqual(result['status'],'exhausted')
            finally:engine.close()
        self.assertGreater(proved,10)

    def test_enabled_log_interruption_and_resume_preserve_all_completions(self):
        rng=random.Random(250044);n=3;initial=[511]*3
        patterns=[[rng.randrange(2) for _ in range(3)] for _ in range(9)]
        edges=[(0,0,1,1),(1,2,2,0),(2,1,0,2)]
        tables=puzzle_tables(n,patterns,edges);expected=oracle(initial,patterns,edges)
        for stop_at in range(1,21):
            engine=NativeSearch(n);engine.configure_propagation(tables);engine.configure_explanations(True)
            leaves=[];polls=[0];stopped=[False]
            def check():
                polls[0]+=1
                if polls[0]==stop_at and not stopped[0]:
                    stopped[0]=True;raise InterruptedError('stop inside native proof path')
            def event(name,payload):
                if name=='fast_reject':
                    clause=verify_native_reasons(payload['native_failure'],initial=initial,tables=tables,
                                                 fingerprint='stop',target='edges')
                    if clause is not None:
                        self.assertTrue(all(not violates(board,clause) for board in expected))
            try:
                try:result=engine.run([frame(initial)],callback(leaves=leaves,distinct=False),check,
                                      event,capture_fast_reject=True)
                except InterruptedError:
                    result=engine.run(None,callback(leaves=leaves,distinct=False),on_event=event,capture_fast_reject=True)
                self.assertEqual(result['status'],'exhausted')
                self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
            finally:engine.close()

    def test_committed_reject_stays_consumed_when_reason_consumer_stops(self):
        engine=self.engine(3);tables=puzzle_tables(3,[[0]*3]*9,[])
        engine.configure_propagation(tables);engine.configure_explanations(True)
        seen=[]
        def event(name,payload):
            self.assertEqual(name,'fast_reject');seen.append(payload['native_failure'])
            raise InterruptedError('after complete rejection')
        with self.assertRaises(InterruptedError):
            engine.run([frame([3,6,448])],lambda *_:self.fail('native failure'),
                       on_event=event,capture_fast_reject=True)
        self.assertEqual(engine.snapshot(),[])
        again=engine.run(None,lambda *_:self.fail('consumed node replayed'),on_event=event,capture_fast_reject=True)
        self.assertEqual(again['status'],'exhausted');self.assertEqual(len(seen),1)

    def test_native_log_cap_uses_checked_replay_fallback(self):
        engine=self.engine(3);tables=puzzle_tables(3,[[0]*3]*9,[]);stats={}
        engine.configure_propagation(tables);engine.configure_explanations(True,max_records=0)
        def event(name,payload):
            self.assertEqual(name,'fast_reject');failure=payload['native_failure']
            self.assertIsNone(failure.reasons)
            clause=explain_fast_failure(failure,initial=[3,6,448],tables=tables,fingerprint='cap',stats=stats)
            self.assertEqual(clause.literals,())
        result=engine.run([frame([3,6,448])],lambda *_:self.fail('native failure'),on_event=event,capture_fast_reject=True)
        self.assertEqual(result['status'],'exhausted')
        self.assertEqual(stats['native_reason_fallbacks'],1);self.assertEqual(stats['native_learning_replays'],1)


if __name__=='__main__':unittest.main()
