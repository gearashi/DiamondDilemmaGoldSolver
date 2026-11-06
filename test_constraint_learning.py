"""Bounded conflict proofs and nonchronological stack truncation."""
import copy
import itertools
import json
import random
import unittest
from unittest.mock import patch
import constraint_dfs as dfs
from search_problem import SearchProblem
from test_constraint_dfs import brute, tetrahedron_tiles, simple_tiles, double_face, multiple_loops, board
from search_options import BOOLEAN_OPTIONS


class ConflictExplanationTests(unittest.TestCase):
    def test_deletions_require_a_completed_contradiction(self):
        assumptions=[(0,0),(1,3),(2,6),(3,9)]
        calls=[]
        def prove(items):
            calls.append(tuple(items))
            return (0,0) in items and (2,6) in items
        yielded=list(dfs._minimize_conflict(assumptions,prove))
        self.assertEqual(yielded[-1],((0,0),(2,6)))
        self.assertEqual(len(calls),4)
        for clause in yielded:self.assertTrue(prove(clause))

    def test_budget_raises_without_yielding_an_unproved_reduction(self):
        def inconclusive(_):raise TimeoutError()
        iterator=dfs._minimize_conflict([(0,0),(1,3)],inconclusive)
        with self.assertRaises(TimeoutError):next(iterator)

    def test_jump_keeps_responsible_sibling_cursor_and_all_earlier_choices(self):
        decisions=[(i,3*i) for i in range(5)]
        stack=[{'cell':i,'choices':[3*i,3*i+1,3*i+2],'next':1,'domains':[7<<3*j for j in range(5)]}
               for i in range(5)]
        original=copy.deepcopy(stack)
        self.assertEqual(dfs._jump_conflict(stack,decisions,[(0,0),(2,6)]),2)
        self.assertEqual(stack,original[:3])
        self.assertEqual(stack[-1]['choices'][stack[-1]['next']:],[7,8])
        self.assertEqual(dfs._jump_conflict(stack,decisions,[(4,13)]),0)
        self.assertEqual(dfs._jump_conflict(stack,decisions,[]),3)
        self.assertEqual(stack,[])

    def test_every_removed_subtree_implies_its_conflict(self):
        # Exhaust every nonempty subset of a six-decision conjunction. For
        # each discarded continuation, vary all decisions deeper than the last
        # conflict literal independently; every completion still has conflict.
        decisions=[(i,3*i) for i in range(6)]
        for mask in range(1,64):
            clause=[literal for i,literal in enumerate(decisions) if mask>>i&1]
            stack=[{'id':i} for i in range(6)]
            dfs._jump_conflict(stack,decisions,clause)
            keep=len(stack)
            for tail in itertools.product(range(3),repeat=6-keep):
                codes=[3*i for i in range(keep)]+[3*(keep+j)+r for j,r in enumerate(tail)]
                self.assertTrue(all(codes[cell]==code for cell,code in clause))


class LearningIntegrationTests(unittest.TestCase):
    def test_all_new_defaults_agree_with_independent_small_enumeration(self):
        rng=random.Random(85219)
        for case in range(12):
            tiles,board=tetrahedron_tiles()
            rng.shuffle(tiles)
            for tile in tiles:
                r=rng.randrange(3)
                tile['segments']=[[[((s+r)%3),p] for s,p in segment] for segment in tile['segments']]
            if case%2:tiles[0]['segments'][0][0][1] %= 10;tiles[0]['segments'][0][0][1] += 1
            problem=SearchProblem(tiles,board)
            if case%3==1:problem.domains[case%4]=[rng.randrange(12)]
            for target in ('gold','edges'):
                expected=brute(problem,target)
                result=dfs.solve(problem,seconds=5,target=target,seed=case)
                self.assertTrue(result['complete'],result)
                if expected:self.assertIn(tuple(result['codes']),expected)
                else:self.assertEqual(result['status'],'infeasible')

    def backjump_fixture(self):
        problem=SearchProblem([dict(tile,id=i) for i,tile in enumerate(multiple_loops()*2)],
                              board(4,[(0,s,1,s) for s in range(3)]+[(2,s,3,s) for s in range(3)]))
        problem.domains=[[3*i+r for r in range(3)] for i in range(4)]
        return problem

    def test_actual_tile_conflict_backjumps_and_resumes_without_duplicate_leaves(self):
        # Suppress only early necessary loop filters to expose rejected actual
        # Gold leaves. The original independent validator still rejects them;
        # no fabricated failure predicate or learned clause is supplied.
        problem=self.backjump_fixture()
        self.assertEqual(brute(problem,'gold'),[])
        options={key:False for key in BOOLEAN_OPTIONS}
        options.update(backjumping=True,native='python',persistent_learning=True)
        seen=[];resume=None;cache={}
        with patch.object(dfs.ConnectivityPruner,'prune',return_value=False), \
             patch.object(problem,'partial_loop_impossible',return_value=False), \
             patch.object(dfs.time,'monotonic',return_value=1.):
            for _ in range(100):
                before=len(seen)
                result=dfs.solve(problem,seconds=5,options=options,seed=4,resume=resume,learning_state=cache,
                                 stop=lambda:len(seen)>before,
                                 progress=lambda event:seen.append(tuple(event['codes'])) if event['event']=='candidate' else None)
                if result['complete']:break
                self.assertEqual(result['status'],'stopped')
                resume=json.loads(json.dumps(result['checkpoint']))
            else:self.fail('Tiny exact tree did not finish')
        self.assertEqual(result['status'],'infeasible')
        self.assertGreater(result['stats']['backjumps'],0)
        self.assertGreater(result['stats']['explanation_literals_removed'],0)
        self.assertEqual(len(seen),len(set(seen)))
        self.assertGreater(result['stats']['loop_witnesses'],0)

    def test_projected_scheduler_preserves_search_and_reduces_matching_calls(self):
        problem=self.backjump_fixture()
        options={key:False for key in BOOLEAN_OPTIONS};options['native']='python'
        runs=[]
        for incremental in (False,True):
            seen=[]
            with patch.object(dfs.ConnectivityPruner,'prune',return_value=False), \
                 patch.object(problem,'partial_loop_impossible',return_value=False), \
                 patch.object(dfs.time,'monotonic',return_value=1.):
                result=dfs.solve(problem,seconds=5,options={**options,'incremental':incremental},seed=4,
                                 progress=lambda event:seen.append(tuple(event['codes'])) if event['event']=='candidate' else None)
            runs.append((result,seen))
        self.assertEqual(runs[0][1],runs[1][1])
        self.assertEqual(runs[0][0]['nodes'],runs[1][0]['nodes'])
        self.assertEqual(runs[1][0]['status'],'infeasible')
        self.assertLess(runs[1][0]['stats']['hall_checks'],runs[0][0]['stats']['hall_checks'])
        self.assertGreater(runs[1][0]['stats']['propagation_unchanged_skips'],0)

    def test_shared_knowledge_requires_typed_validated_binding(self):
        from constraint_knowledge import KnowledgeStore
        problem=SearchProblem(simple_tiles(),double_face())
        knowledge=KnowledgeStore(problem)
        result=dfs.solve(problem,seconds=5,learning_state={'_knowledge':knowledge})
        self.assertEqual(result['status'],'solved')
        self.assertIn('knowledge',result)
        with self.assertRaises(ValueError):
            dfs.solve(problem,seconds=5,learning_state={'_knowledge':knowledge.snapshot()})
        with self.assertRaises(ValueError):
            dfs.solve(problem,seconds=5,target='edges',learning_state={'_knowledge':knowledge})
        problem.domains[0]=problem.domains[0][:1]
        with self.assertRaises(ValueError):
            dfs.solve(problem,seconds=5,learning_state={'_knowledge':knowledge})

    def test_stop_during_explanation_leaves_a_valid_committed_stack(self):
        problem=self.backjump_fixture();requested=[False]
        options={key:False for key in BOOLEAN_OPTIONS};options.update(backjumping=True,native='python')
        original=dfs._minimize_conflict
        def interrupt(*args,**kwargs):
            requested[0]=True
            yield from original(*args,**kwargs)
        with patch.object(dfs.ConnectivityPruner,'prune',return_value=False), \
             patch.object(problem,'partial_loop_impossible',return_value=False), \
             patch.object(dfs,'_minimize_conflict',interrupt):
            first=dfs.solve(problem,seconds=5,options=options,seed=4,stop=lambda:requested[0])
        self.assertEqual(first['status'],'stopped')
        self.assertGreater(first['stats']['explanation_aborted'],0)
        resumed=dfs.solve(problem,seconds=5,resume=json.loads(json.dumps(first['checkpoint'])))
        self.assertEqual(resumed['status'],'infeasible')

    def test_scheduler_options_and_legacy_checkpoint_resume(self):
        problem=SearchProblem(simple_tiles(),double_face())
        first=dfs.solve(problem,seconds=0)
        cp=json.loads(json.dumps(first['checkpoint']))
        for incremental in (False,True):
            for adaptive in (False,True):
                result=dfs.solve(problem,seconds=5,resume=cp,options={'incremental':incremental,'adaptive':adaptive})
                self.assertEqual(result['status'],'solved')
                self.assertTrue(problem.validate(result['codes'])['valid'])




class PrunerSliceTableTests(unittest.TestCase):
    def test_stop_inside_construction_then_resume_and_rebind_next_slice(self):
        problem=SearchProblem(*tetrahedron_tiles());cache={};requested=[False]
        original=dfs.Pruner._build_clusters
        def stop_build(helper):
            requested[0]=True
            helper.check()
            return original(helper)
        options={key:False for key in BOOLEAN_OPTIONS};options['native']='python'
        with patch.object(dfs.Pruner,'_build_clusters',stop_build):
            first=dfs.solve(problem,seconds=5,options=options,stop=lambda:requested[0],learning_state=cache)
        self.assertEqual(first['status'],'stopped');self.assertIsNone(cache['_runtime'].pruner_tables)
        first=dfs.solve(problem,seconds=5,options=options,resume=first['checkpoint'],learning_state=cache)
        self.assertEqual(first['status'],'solved');tables=cache['_runtime'].pruner_tables
        self.assertEqual(first['stats']['pruner_table_builds'],1)
        # A new slice creates a fresh Pruner and a new native selector. The old
        # slice's check closure still sees requested=True and must not survive.
        second=dfs.solve(problem,seconds=5,options={**options,'native':'auto'},learning_state=cache)
        self.assertEqual(second['status'],'solved');self.assertIs(cache['_runtime'].pruner_tables,tables)
        self.assertEqual(second['stats']['pruner_table_reuses'],1)
        self.assertIn(tuple(second['codes']),brute(problem,'gold'))

    def test_completed_lazy_profiles_survive_stop_after_construction(self):
        problem=SearchProblem(*tetrahedron_tiles());cache={};requested=[False]
        original=dfs.Pruner._build_stars
        def finish_build(helper):
            profiles=original(helper);requested[0]=True;return profiles
        options={key:False for key in BOOLEAN_OPTIONS};options['native']='python'
        with patch.object(dfs.Pruner,'_build_stars',finish_build):
            first=dfs.solve(problem,seconds=5,options=options,stop=lambda:requested[0],learning_state=cache)
        self.assertEqual(first['status'],'stopped')
        tables=cache['_runtime'].pruner_tables;self.assertIsNotNone(tables.stars)
        with patch.object(dfs.Pruner,'_build_stars',side_effect=AssertionError('Completed profiles rebuilt')):
            result=dfs.solve(problem,seconds=5,options=options,resume=first['checkpoint'],learning_state=cache)
        self.assertEqual(result['status'],'solved');self.assertIs(cache['_runtime'].pruner_tables,tables)

    def test_resumed_table_counters_accumulate_without_serializing_tables(self):
        problem=SearchProblem(*tetrahedron_tiles());cache={};requested=[False]
        original=dfs.Pruner.prune
        def stop_after(helper,domains):
            changed=original(helper,domains);requested[0]=True;helper.check();return changed
        options={key:False for key in BOOLEAN_OPTIONS};options['native']='python'
        with patch.object(dfs.Pruner,'prune',stop_after):
            first=dfs.solve(problem,seconds=5,options=options,stop=lambda:requested[0],learning_state=cache)
        self.assertEqual(first['status'],'stopped');tables=cache['_runtime'].pruner_tables
        checkpoint=json.loads(json.dumps(first['checkpoint']))
        self.assertNotIn('pruner_tables',json.dumps(checkpoint))
        second=dfs.solve(problem,seconds=5,options=options,resume=checkpoint,learning_state=cache)
        self.assertEqual(second['status'],'solved');self.assertIs(cache['_runtime'].pruner_tables,tables)
        self.assertEqual(second['stats']['pruner_table_builds'],1)
        self.assertEqual(second['stats']['pruner_table_reuses'],1)


if __name__=='__main__':unittest.main()
