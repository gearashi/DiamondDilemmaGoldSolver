"""Independent native failure proofs, exact pending coverage and Stop contracts."""
import itertools
import random
import unittest
from unittest.mock import patch

from constraint_lcg import LCGStore
from constraint_native_learning import replay_fast_failure
from constraint_native_search import FastFailure
import test_constraint_native_search as support
from test_constraint_native_search import callback, frame, puzzle_tables, values


def oracle(initial,patterns,edges):
    return {codes for codes in itertools.product(*(values(mask) for mask in initial))
            if len({code//3 for code in codes})==len(initial)
            and all(patterns[codes[a]][sa]==patterns[codes[b]][sb] for a,sa,b,sb in edges)}


def violates(codes,clause):
    return all(mask & (1<<codes[cell]) for cell,mask in clause.literals)


class ReplayTests(unittest.TestCase):
    def test_duplicate_tile_explanation_omits_irrelevant_mask_decision(self):
        initial=[511]*3;tables=puzzle_tables(3,[[0]*3]*9,[])
        failure=FastFailure((3,6,448),((0,3),(2,448),(1,6)))
        stats={};clause=replay_fast_failure(failure,initial=initial,tables=tables,fingerprint='tiny',stats=stats)
        self.assertEqual(clause.literals,((0,3),(1,6)))
        self.assertNotIn(2,dict(clause.literals))
        self.assertTrue(stats['native_learning_checks'])
        self.assertEqual(stats['native_learning_clauses'],1)
        for codes in oracle(initial,[[0]*3]*9,[]):self.assertFalse(violates(codes,clause))

    def test_restricted_root_cannot_become_false_global_unsat(self):
        tables=puzzle_tables(2,[[0]*3]*6,[])
        failure=FastFailure((3,6),())
        self.assertIsNone(replay_fast_failure(failure,initial=[63,63],tables=tables,fingerprint='tiny'))
        local=replay_fast_failure(failure,initial=[3,6],tables=tables,fingerprint='tiny')
        self.assertEqual(local.literals,())
        self.assertTrue(oracle([63,63],[[0]*3]*6,[]))
        with self.assertRaises(ValueError):LCGStore('tiny','gold',[63,63]).learn(local)
        with self.assertRaises(ValueError):LCGStore('tiny','edges',[3,6]).learn(local)
        with self.assertRaises(ValueError):LCGStore('other','gold',[3,6]).learn(local)

    def test_caps_and_stop_publish_nothing(self):
        tables=puzzle_tables(3,[[0]*3]*9,[]);failure=FastFailure((3,6,511),((0,3),(1,6)))
        for budget in (0,1):
            stats={}
            self.assertIsNone(replay_fast_failure(failure,initial=[511]*3,tables=tables,fingerprint='tiny',max_work=budget,stats=stats))
            self.assertLessEqual(stats['native_learning_checks'],budget)
            self.assertEqual(stats['native_learning_clauses'],0)
        for stop_at in (1,2,3):
            checks=[0];stats={}
            def stop():
                checks[0]+=1
                if checks[0]==stop_at:raise InterruptedError('replay cancelled')
            with self.assertRaises(InterruptedError):
                replay_fast_failure(failure,initial=[511]*3,tables=tables,fingerprint='tiny',check=stop,stats=stats)
            self.assertEqual(stats['native_learning_clauses'],0)

    def test_random_completed_clauses_against_full_independent_oracle(self):
        rng=random.Random(802404);learned=0
        for trial in range(120):
            n=3;initial=[511]*n
            patterns=[[rng.randrange(3) for _ in range(3)] for _ in range(9)]
            edges=[(0,0,1,1),(1,2,2,0),(2,1,0,2)]
            tables=puzzle_tables(n,patterns,edges)
            decisions=tuple((cell,rng.randrange(1,512)) for cell in rng.sample(range(n),rng.randrange(1,4)))
            current=initial.copy()
            for cell,mask in decisions:current[cell]&=mask
            # Extra carried restrictions are intentionally not proofs.
            failure=FastFailure(tuple(mask & -mask for mask in current),decisions)
            clause=replay_fast_failure(failure,initial=initial,tables=tables,fingerprint='tiny')
            if clause is not None:
                learned+=1
                for codes in oracle(initial,patterns,edges):
                    self.assertFalse(violates(codes,clause),(trial,codes,clause))
        self.assertGreater(learned,20)

    def test_malformed_domains_and_decisions_rejected(self):
        tables=puzzle_tables(2,[[0]*3]*6,[])
        for failure,initial in ((FastFailure((0,63),()),[63,63]),
                                (FastFailure((64,63),()),[63,63]),
                                (FastFailure((63,63),()),[64,63]),
                                (FastFailure((63,63),((0,1),)),[63,63]),
                                (FastFailure((63,63),((2,1),)),[63,63])):
            with self.assertRaises(ValueError):replay_fast_failure(failure,initial=initial,tables=tables,fingerprint='tiny')


class NativeReplayTests(unittest.TestCase):
    setUpClass=classmethod(support.NativeSearchTests.setUpClass.__func__)
    engine=support.NativeSearchTests.engine

    def test_complete_enumeration_learns_and_reduces_native_nodes(self):
        rng=random.Random(11);n=4;initial=[4095]*n
        patterns=[[rng.randrange(2) for _ in range(3)] for _ in range(12)]
        edges=[(0,0,1,0),(1,1,2,0),(2,1,3,0),(3,1,0,1)]
        tables=puzzle_tables(n,patterns,edges);expected=oracle(initial,patterns,edges)
        results=[];all_clauses=[];replay_stats={}
        for enabled in (False,True):
            engine=self.engine(n);engine.configure_propagation(tables)
            store=LCGStore('finite','edges',initial);leaves=[]
            basic=callback(leaves=leaves,distinct=False)
            def process(domains,decisions):
                if enabled:
                    store.prune(domains,lambda:None)
                    if not all(domains):return {'action':'reject'}
                return basic(domains,decisions)
            def event(name,payload):
                if enabled and name=='fast_reject':
                    clause=replay_fast_failure(payload['native_failure'],initial=initial,tables=tables,
                        fingerprint='finite',target='edges',stats=replay_stats)
                    if clause is not None:
                        all_clauses.append(clause);store.learn(clause)
                        replacement=store.asserting_frontier(engine.snapshot(),clause,require_jump=True)
                        if replacement is not None:return {'replace_stack':replacement}
            result=engine.run([frame(initial)],process,on_event=event,capture_fast_reject=enabled)
            results.append(result)
            self.assertEqual(result['status'],'exhausted')
            self.assertEqual(set(leaves),expected)
            self.assertEqual(len(leaves),len(expected))
        self.assertEqual(len(expected),152)
        self.assertLess(results[1]['stats']['native_search_nodes'],results[0]['stats']['native_search_nodes'])
        self.assertLess(results[1]['stats']['native_search_transitions'],results[0]['stats']['native_search_transitions'])
        self.assertGreater(replay_stats['native_learning_checks'],0)
        self.assertTrue(all_clauses)
        for clause in all_clauses:
            for codes in expected:self.assertFalse(violates(codes,clause))

    def test_full_solver_native_root_conflict_reaches_lcg(self):
        from types import SimpleNamespace
        from constraint_dfs import solve
        from search_options import BOOLEAN_OPTIONS
        from search_problem import SearchProblem
        from test_constraint_dfs import brute
        tiles=[{'id':0,'segments':[[[2,6],[1,6]],[[0,2],[0,10]]]},
               {'id':1,'segments':[[[1,6],[2,6]],[[0,2],[0,10]]]},
               {'id':2,'segments':[[[2,6],[1,10]],[[0,6],[1,2]]]},
               {'id':3,'segments':[[[0,6],[1,6]]]}]
        edges=((0,0,1,0),(0,1,2,0),(0,2,3,0),(1,1,2,1),(1,2,3,1),(2,2,3,2))
        problem=SearchProblem(tiles,SimpleNamespace(cells=tuple(range(4)),edges=edges))
        options={**dict.fromkeys(BOOLEAN_OPTIONS,False),'native':'compiled','native_search':True,
                 'lcg':True,'backjumping':True}
        self.assertEqual(brute(problem,'edges'),[])
        result=solve(problem,target='edges',seconds=5,options=options)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['nodes'],1)
        self.assertEqual(result['stats']['native_learning_clauses'],1)
        self.assertEqual(result['stats']['lcg_learned'],1)
        self.assertGreater(result['stats']['native_learning_checks'],0)
        options['lcg']=False
        with patch('constraint_dfs.replay_fast_failure',side_effect=AssertionError('LCG disabled')):
            result=solve(problem,target='edges',seconds=5,options=options)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['stats']['native_learning_replays'],0)

    def test_empty_stack_root_capture_and_replay_stop_after_commit(self):
        engine=self.engine(2);tables=puzzle_tables(2,[[0]*3]*6,[]);engine.configure_propagation(tables)
        captures=[]
        def event(name,payload):
            self.assertEqual(name,'fast_reject');captures.append(payload['native_failure'])
            def stop():raise InterruptedError('inside independent replay')
            replay_fast_failure(payload['native_failure'],initial=[3,6],tables=tables,fingerprint='tiny',check=stop)
        with self.assertRaises(InterruptedError) as caught:
            engine.run([frame([3,6])],lambda *args:self.fail('native must reject'),on_event=event,capture_fast_reject=True)
        self.assertEqual(captures,[FastFailure((3,6),())])
        self.assertEqual(caught.exception.native_search_state['stack'],[])
        self.assertEqual(caught.exception.native_search_state['status'],'exhausted')
        self.assertEqual(engine.stats()['native_search_nodes'],1)

    def test_nested_grouped_mask_choice_capture_and_resume(self):
        engine=self.engine(3);tables=puzzle_tables(3,[[0]*3]*9,[]);engine.configure_propagation(tables)
        frames=[{'domains':[511]*3,'cell':0,'choices':[3,508],'next':1},
                {'domains':[3,511,511],'cell':1,'choices':[6,505],'next':1},frame([3,6,511])]
        captures=[]
        def event(name,payload):
            if name=='fast_reject':captures.append(payload['native_failure']);raise InterruptedError('after committed fail')
        with self.assertRaises(InterruptedError) as caught:
            engine.run(frames,callback(distinct=False),on_event=event,capture_fast_reject=True)
        self.assertEqual(captures,[FastFailure((3,6,511),((0,3),(1,6)))])
        self.assertEqual(caught.exception.native_search_state['stack'],frames[:-1])
        self.assertEqual(engine.snapshot(),frames[:-1])
        leaves=[];engine.run(None,callback(leaves=leaves,distinct=False))
        self.assertEqual(set(leaves),oracle([511]*3,[[0]*3]*9,[]))
        self.assertEqual(len(leaves),len(set(leaves)))

    def test_replacement_root_capture_is_updated_and_not_global_proof(self):
        engine=self.engine(2);tables=puzzle_tables(2,[[0]*3]*6,[]);engine.configure_propagation(tables)
        captures=[];replaced=[False]
        def event(name,payload):
            if not replaced[0]:
                replaced[0]=True
                return {'replace_stack':[frame([3,6])]}
            if name=='fast_reject':
                captures.append(payload['native_failure'])
                self.assertIsNone(replay_fast_failure(payload['native_failure'],initial=[63]*2,tables=tables,fingerprint='tiny'))
                raise InterruptedError('after replacement rejection')
        with self.assertRaises(InterruptedError) as caught:
            engine.run([frame([63]*2)],callback(distinct=False),on_event=event,capture_fast_reject=True)
        self.assertEqual(captures,[FastFailure((3,6),())])
        self.assertEqual(caught.exception.native_search_state['stack'],[])
        self.assertEqual(engine.stats()['native_search_nodes'],2)

    def test_stop_after_frontier_replacement_keeps_replacement_unconsumed(self):
        engine=self.engine(2);tables=puzzle_tables(2,[[0]*3]*6,[]);engine.configure_propagation(tables)
        replaced=[False]
        def event(name,payload):
            if not replaced[0]:
                replaced[0]=True;return {'replace_stack':[frame([3,6])]}
        def check():
            if replaced[0]:raise InterruptedError('replacement committed')
        with self.assertRaises(InterruptedError) as caught:
            engine.run([frame([63]*2)],callback(distinct=False),check,event,capture_fast_reject=True)
        self.assertEqual(caught.exception.native_search_state['stack'],[frame([3,6])])
        captures=[]
        engine.run(None,lambda *args:self.fail('must fail natively'),on_event=lambda name,payload:captures.append(payload['native_failure']),capture_fast_reject=True)
        self.assertEqual(captures,[FastFailure((3,6),())])

if __name__=='__main__':unittest.main()
