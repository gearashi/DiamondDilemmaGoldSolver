"""Exact finite oracles for dependency explanations and DFS learning."""
import json,random,unittest
from unittest.mock import patch
from constraint_reasons import ReasonState
import constraint_dfs as dfs
from search_problem import SearchProblem
from search_options import BOOLEAN_OPTIONS
from test_constraint_dfs import tetrahedron_tiles,brute

class ReasonsTests(unittest.TestCase):
    def test_dependency_closure_discards_unrelated_decision(self):
        root=ReasonState([7,56,448])
        a=root.fork(0,0);b=a.fork(1,3)
        narrowed=b.domains.copy();narrowed[2]=64
        b.record(narrowed,[0],'seam support')
        narrowed[2]=0;b.record(narrowed,[2],'seam contradiction')
        self.assertEqual(b.contradiction(),{(0,0)})
        self.assertTrue(b.validate_graph())
        self.assertEqual(root.domains,[7,56,448])
        # Descendant is discarded before the next sibling; its tail is not kept.
        del b
        sibling=a.fork(1,4)
        self.assertEqual(sibling.causes([1]),{(1,4)})
        self.assertTrue(sibling.validate_graph())
        self.assertEqual(a.causes([0]),{(0,0)})

    def test_direct_rule_and_clause_keep_only_their_antecedents(self):
        root=ReasonState([7,56,448,3584]);trace=root.fork(0,0).fork(1,3)
        trace.record_cell(2,64,[0],'seam')
        self.assertEqual(trace.causes([2]),{(0,0)})
        store=dfs._NogoodStore('test','gold');stats=dict.fromkeys(dfs.CONFLICT_STAT_KEYS,0)
        store.learn([(2,6),(3,9)],stats);domains=trace.domains.copy()
        def reason(c,v,deps):
            self.assertEqual(domains[c],v);trace.record_cell(c,v,deps,'learned clause')
        store.prune(domains,stats,lambda:None,reason=reason)
        self.assertEqual(trace.causes([3]),{(0,0)})
        self.assertEqual(domains[3],3584&~512);self.assertTrue(trace.validate_graph())
        with self.assertRaises(ValueError):trace.record_cell(3,3584,[],'widen')

    def test_resume_and_rebase_keep_all_needed_assumptions(self):
        active=[(0,0),(1,3)]
        state=ReasonState([1,8,64],active,initial=[7,56,448])
        self.assertEqual(state.causes([2]),set(active))
        state.rebase(active)
        self.assertTrue(state.validate_graph())
        with self.assertRaises(ValueError):state.record([7,8,64],[0],'invalid widening')

    def test_every_learned_clause_preserves_independent_tiny_solutions(self):
        rng=random.Random(61129);learned_count=0
        original=dfs._NogoodStore.learn
        for case in range(24):
            tiles,board=tetrahedron_tiles();rng.shuffle(tiles)
            for tile in tiles:
                r=rng.randrange(3)
                tile['segments']=[[[((s+r)%3),p] for s,p in segment] for segment in tile['segments']]
            if case%3==1:tiles[0]['segments'][0][0][1]=1
            problem=SearchProblem(tiles,board)
            if case%4==2:problem.domains[case%4]=rng.sample(problem.domains[case%4],4)
            for target in ('gold','edges'):
                expected=brute(problem,target);clauses=[]
                def learn(store,placements,stats):
                    material=tuple(placements);clauses.append(material)
                    return original(store,material,stats)
                opts={key:False for key in BOOLEAN_OPTIONS}
                opts.update(reason_tracking=True,backjumping=True,native='python')
                with patch.object(dfs._NogoodStore,'learn',learn):
                    result=dfs.solve(problem,seconds=5,target=target,options=opts,seed=case)
                self.assertTrue(result['complete'],result)
                if expected:self.assertIn(tuple(result['codes']),expected)
                else:self.assertEqual(result['status'],'infeasible')
                for clause in clauses:
                    self.assertFalse(any(all(codes[c]==v for c,v in clause) for codes in expected),(case,clause))
                learned_count+=len(clauses)
        self.assertGreater(learned_count,0)

    def test_stop_resume_never_serializes_reason_graph(self):
        problem=SearchProblem(*tetrahedron_tiles());calls=[0]
        def stop():calls[0]+=1;return calls[0]>25
        first=dfs.solve(problem,seconds=5,stop=stop,options={'reason_tracking':True,'native':'python'})
        self.assertEqual(first['status'],'stopped')
        payload=json.dumps(first['checkpoint']);self.assertNotIn('_reasons',payload)
        result=dfs.solve(problem,seconds=5,resume=json.loads(payload),options={'reason_tracking':True})
        self.assertEqual(result['status'],'solved')


    def test_multilevel_backjump_discards_only_dead_fact_tail(self):
        root=ReasonState([7,56,448,3584])
        a=root.fork(0,0);b=a.fork(1,3);leaf=b.fork(2,6)
        narrowed=leaf.domains.copy();narrowed[3]=512
        leaf.record(narrowed,[0,2],'dependent reduction')
        stack=[{'trace':root},{'trace':a},{'trace':b}]
        self.assertEqual(dfs._jump_conflict(stack,[(0,0),(1,3),(2,6)],[(0,0)]),2)
        self.assertEqual(len(stack),1)
        del leaf,b,a
        sibling=root.fork(0,1)
        self.assertIs(sibling.facts,root.facts)
        self.assertEqual(sibling.causes([0]),{(0,1)})
        self.assertFalse(sibling.causes([1,2,3]))
        self.assertEqual(len(sibling.facts),5)
        self.assertTrue(sibling.validate_graph())
        child=sibling.fork(1,4)
        self.assertEqual(child.causes([0,1]),{(0,1),(1,4)})
        self.assertTrue(child.validate_graph())

    def test_bulk_outputs_use_pre_call_antecedents_and_preserve_prior_causes(self):
        root=ReasonState([7,56,448,3584]);trace=root.fork(0,0).fork(1,3)
        narrowed=trace.domains.copy();narrowed[2]=64
        trace.record(narrowed,[0],'first premise')
        narrowed=trace.domains.copy();narrowed[2]=0;narrowed[3]=512
        trace.record(narrowed,[1,2],'partial bulk rule')
        self.assertEqual(trace.contradiction(),{(0,0),(1,3)})
        self.assertEqual(trace.causes([3]),{(0,0),(1,3)})
        self.assertTrue(trace.validate_graph())
        # A conservative rebase after lookahead must not silently drop a cause.
        trace.rebase([(0,0),(1,3),(3,9)])
        self.assertEqual(trace.contradiction(),{(0,0),(1,3),(3,9)})

    def test_learned_failure_stop_resume_with_shared_runtime_preserves_other_gold(self):
        from test_smart_dfs import mixed_loop_problem
        problem=mixed_loop_problem();solutions=brute(problem,'gold');cache={};captured=[]
        original=dfs._NogoodStore.learn
        def learn(store,placements,stats):
            clause=tuple(placements);captured.append(clause)
            return original(store,clause,stats)
        options={key:False for key in BOOLEAN_OPTIONS}
        options.update(reason_tracking=True,backjumping=True,native='python')
        with patch.object(dfs._NogoodStore,'learn',learn):
            first=dfs.solve(problem,seconds=5,options=options,hints=[0,3],learning_state=cache,stop=lambda:bool(captured))
        self.assertEqual(first['status'],'stopped');self.assertTrue(captured)
        for clause in captured:
            self.assertFalse(any(all(row[c]==v for c,v in clause) for row in solutions))
        payload=json.loads(json.dumps(first['checkpoint']))
        self.assertTrue(payload['stack']);self.assertNotIn('_reasons',json.dumps(payload))
        resumed=dfs.solve(problem,seconds=5,options=options,resume=payload,learning_state=cache)
        self.assertEqual(resumed['status'],'solved');self.assertIn(tuple(resumed['codes']),solutions)

    def test_cancellation_after_a_tracked_reduction_reconstructs_safe_reasons(self):
        problem=SearchProblem(*tetrahedron_tiles());original=ReasonState.record;direct_original=ReasonState.record_cell;reduced=[]
        def record(trace,domains,dependencies,rule):
            count=original(trace,domains,dependencies,rule)
            if count:reduced.append((rule,count))
            return count
        def direct(trace,cell,value,dependencies,rule):
            count=direct_original(trace,cell,value,dependencies,rule)
            if count:reduced.append((rule,count))
            return count
        options={key:False for key in BOOLEAN_OPTIONS}
        options.update(reason_tracking=True,backjumping=True,native='python')
        with patch.object(ReasonState,'record',record), patch.object(ReasonState,'record_cell',direct):
            first=dfs.solve(problem,seconds=5,options=options,stop=lambda:bool(reduced))
        self.assertTrue(reduced);self.assertEqual(first['status'],'stopped')
        resumed=dfs.solve(problem,seconds=5,options=options,resume=json.loads(json.dumps(first['checkpoint'])))
        self.assertEqual(resumed['status'],'solved');self.assertTrue(problem.validate(resumed['codes'])['valid'])

if __name__=='__main__':unittest.main()
