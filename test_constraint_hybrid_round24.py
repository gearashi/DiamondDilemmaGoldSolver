"""Independent finite checks for round24 branching and constructive search."""
import itertools
import random
import unittest
from constraint_branching_round24 import RoutingBranching,CompletionOrdering,LearnedOrdering
from constraint_repair_round24 import core_guided_repair,loop_merge_trade,fragment_crossover,GoldRepairOracle,scoped_repair_failure
from constraint_lns import _Control
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,tetrahedron_tiles,brute,multiple_loops
from test_dependencies import requires_ortools

def masks(problem):return [sum(1<<code for code in row) for row in problem.domains]

class BranchingRound24Tests(unittest.TestCase):
    def test_dual_partition_preserves_every_independent_completion(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board);domains=masks(problem)
        domains[0]&=~7;domains[1]&=~7
        plan=RoutingBranching(problem).choose(domains,0,lambda:None,{},dual_piece=True)
        self.assertIsNotNone(plan);self.assertEqual(plan.patterns[0],0)
        self.assertEqual(plan.masks[0]&plan.masks[1],0)
        self.assertEqual(plan.masks[0]|plan.masks[1],domains[plan.cell])
        for answer in brute(problem,'gold'):
            if all(mask&(1<<code) for mask,code in zip(domains,answer)):
                self.assertEqual(sum(bool(mask&(1<<answer[plan.cell])) for mask in plan.masks),1)
    def test_strand_partition_uses_partner_rather_than_seam_mask(self):
        tiles=[{'id':i,'segments':[[[0,2],[1,2]],[[0,10],[1,10]],[[2,2],[2,10]]]} for i in range(2)]
        problem=SearchProblem(tiles,double_face());domains=[1,masks(problem)[1]&~7]
        plan=RoutingBranching(problem).choose(domains,1,lambda:None,{},strand=True)
        self.assertIsNotNone(plan);self.assertEqual(plan.side,-2)
        self.assertEqual(sum(plan.masks),domains[plan.cell]);self.assertEqual(len(set(plan.patterns)),len(plan.patterns))
        self.assertEqual(len({tuple(problem.masks[code]) for code in range(3,6)}),1)
    def test_completed_counts_match_independent_local_enumeration(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board);domains=masks(problem);stats={}
        ordering=CompletionOrdering(problem,max_cells=4,max_choices=12,max_states=1000000)
        counts=ordering.counts(0,domains,list(range(12)),lambda:None,stats)
        gold=brute(problem,'edges')
        self.assertEqual(counts,{code:sum(answer[0]==code for answer in gold) for code in range(12)})
    def test_capped_count_orders_every_value_without_exclusion(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board);domains=masks(problem)
        choices=list(range(12));stats={}
        got=CompletionOrdering(problem,max_states=1).order(0,domains,choices,{},list(range(12)),lambda:None,stats)
        self.assertEqual(sorted(got),choices);self.assertGreater(stats['completion_order_capped'],0)
    def test_learned_scores_only_reorder_values_and_validate_coefficients(self):
        problem=SearchProblem(simple_tiles(),double_face());domains=masks(problem)
        model=LearnedOrdering(problem,[0,1,-2,.1,0]);choices=list(range(6))
        ordered=model.order(0,domains,choices,{5:0},list(range(6)),lambda:None,{})
        self.assertEqual(ordered[0],5);self.assertEqual(sorted(ordered),choices)
        with self.assertRaises(ValueError):LearnedOrdering(problem,[0]*4)
        examples=[{'features':[1,float(i),0,0,0],'work':i*i} for i in range(1,6)]
        fitted=LearnedOrdering.fit(examples)
        self.assertEqual(fitted['examples'],5);self.assertGreater(fitted['weights'][1],0)

class RepairRound24Tests(unittest.TestCase):
    @requires_ortools
    def test_core_repair_minimum_distance_matches_complete_enumeration(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board);seed=[0,4,6,9]
        solutions=brute(problem,'gold');expected=min(sum(a!=b for a,b in zip(seed,answer)) for answer in solutions)
        result=core_guided_repair(problem,seed,seconds=5)
        self.assertEqual(result['status'],'solved',result);self.assertEqual(result['minimum_changes'],expected)
        self.assertTrue(problem.validate(result['best_board'])['valid']);self.assertGreater(len(result['cores']),0)
    def test_core_repair_rejects_seed_that_changes_fixed_cells(self):
        problem=SearchProblem(simple_tiles(),double_face(),fixed={0:0})
        with self.assertRaisesRegex(ValueError,'fixed placement'):
            core_guided_repair(problem,[3,0],seconds=5)

    @requires_ortools
    def test_scoped_failure_does_not_exclude_successful_sibling(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board)
        oracle=GoldRepairOracle(problem,_Control(5,None))
        bad=oracle.query([(0,1)]);good=oracle.query([(0,0)])
        self.assertEqual(bad['status'],'infeasible');self.assertEqual(good['status'],'solved')
        for answer in brute(problem,'gold'):
            for core in bad['core'],:
                self.assertFalse(all(answer[cell]==code for cell,code in core))
        self.assertTrue(oracle.query([(0,1),(1,3)]).get('cached'))
    @requires_ortools
    def test_crossover_jointly_handles_fragments_and_inventory(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board)
        result=fragment_crossover(problem,[[1,3,6,9],[0,4,6,9]],seconds=5,max_fragment=1)
        self.assertEqual(result['status'],'solved',result);self.assertTrue(problem.validate(result['best_board'])['valid'])
        self.assertEqual(result['sources'],[0,1])
    def test_trade_merges_loops_and_preserves_fixed_boundary(self):
        tiles=[{'id':i,'segments':[[[0,2],[1,2]],[[0,10],[1,10]],[[2,2],[2,10]]]} for i in range(2)]
        problem=SearchProblem(tiles,double_face(),fixed={0:0})
        original=[0,3];self.assertEqual(problem.validate(original)['line_components'],3)
        result=loop_merge_trade(problem,original,[1],seconds=2)
        self.assertEqual(result['status'],'solved');self.assertEqual(result['best_board'][0],0)
        self.assertGreater(result['stats']['trade_improvements'],0)
    def test_failed_and_interrupted_trades_are_local_only(self):
        problem=SearchProblem(multiple_loops(),double_face())
        result=loop_merge_trade(problem,[0,3],[0],seconds=2)
        self.assertTrue(result['complete_local_query']);self.assertFalse(result['validation']['valid'])
        capped=loop_merge_trade(problem,[0,3],[0,1],seconds=2,max_states=1)
        self.assertFalse(capped['complete_local_query']);self.assertEqual(capped['status'],'unknown')
    @requires_ortools
    def test_hybrid_repair_claims_require_independent_verification(self):
        from unittest.mock import patch
        import gpu_sampling
        try:import pysat
        except ImportError:self.skipTest('python-sat unavailable')
        problem=SearchProblem(*tetrahedron_tiles());seed=[1,3,6,9];learning={}
        payload={'kind':'gold_repair_assumption_cores','fingerprint':problem.fingerprint,'domains':problem.domains,
                 'cores':[[[0,1]],[[0,0]]]}
        repair={'status':'unknown','best_board':seed,'experimental_repair':[{'exclusions':payload}]}
        def exact(*args,**kwargs):
            store=kwargs['learning_state']['_mask_conflicts']
            self.assertEqual(len(store),0);self.assertEqual(store.pending_count,2)
            store.advance(seconds=1,attempts=4,conflicts=10000)
            self.assertEqual(store.stats['mask_conflicts_verified'],1)
            self.assertEqual(store.stats['mask_conflicts_rejected'],1)
            return {'status':'timeout','complete':False,'nodes':0,'stats':{},'elapsed_seconds':0}
        try:
            with patch('gpu_sampling.sample',return_value={'top_samples':[]}),patch('constraint_lns.repair',return_value=repair),patch('constraint_dfs.solve',side_effect=exact):
                result=gpu_sampling.solve(problem,seconds=5,seed_board=seed,learning_state=learning,options={'repair_learning':True,'smart_learning':True})
            self.assertEqual(result['local_repair']['claims_queued_for_independent_proof'],2)
        finally:
            if learning.get('_mask_conflicts'):learning['_mask_conflicts'].close()

    def test_cancelled_core_repair_does_not_build_model(self):
        problem=SearchProblem(simple_tiles(),double_face())
        result=core_guided_repair(problem,[1,3],seconds=0)
        self.assertEqual(result['status'],'timeout')

if __name__=='__main__':unittest.main()
