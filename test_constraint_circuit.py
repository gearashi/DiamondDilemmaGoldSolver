"""Independent finite solution-set checks for direct optional-port circuits."""
import copy
import itertools
import random
import threading
import unittest
from unittest.mock import patch
from ortools.sat.python import cp_model
from constraint_circuit import build_circuit_model,CircuitLimit
from constraint_models import solve
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_models import pair_problem,brute
from test_constraint_dfs import tetrahedron_tiles


def enumerate_model(problem,*,group_supports=True,break_direction=True):
    built=build_circuit_model(problem,problem.domains,group_supports=group_supports,break_direction=break_direction)
    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):super().__init__();self.boards=set();self.count=0
        def on_solution_callback(self):
            self.boards.add(tuple(self.value(code) for code in built.codes));self.count+=1
    collector=Collector();solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1;solver.parameters.enumerate_all_solutions=True
    solver.parameters.max_time_in_seconds=5
    status=solver.solve(built.model,collector)
    if status not in (cp_model.OPTIMAL,cp_model.INFEASIBLE):raise AssertionError('Finite circuit enumeration did not complete')
    return collector,built


class CircuitTests(unittest.TestCase):
    def test_complete_tiny_solution_sets_match_independent_validator(self):
        fixtures=[pair_problem(),pair_problem(multiple_loops=True),pair_problem(boundary=True),pair_problem(fixed={0:0})]
        # Symmetric endpoint masks with different internal pairings distinguish
        # a valid single Gold loop from edge-perfect multiple components.
        from test_constraint_knowledge import mixed_problem
        fixtures.append(mixed_problem())
        rng=random.Random(60912)
        for case in range(6):
            tiles,board=tetrahedron_tiles();rng.shuffle(tiles)
            for tile in tiles:
                rotation=rng.randrange(3)
                tile['segments']=[[[((side+rotation)%3),p] for side,p in segment] for segment in tile['segments']]
            if case%2:tiles[0]['segments'][0][0][1]=tiles[0]['segments'][0][0][1]%11+1
            fixtures.append(SearchProblem(tiles,board))
        for index,problem in enumerate(fixtures):
            with self.subTest(fixture=index):
                found,built=enumerate_model(problem)
                self.assertEqual(found.boards,brute(problem,'gold'))
                self.assertEqual(found.count,len(found.boards))
                self.assertEqual(built.model.validate(),'')

    def test_grouping_and_direction_symmetry_preserve_exact_board_set(self):
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem();expected=brute(problem,'gold')
        base,_=enumerate_model(problem,group_supports=False,break_direction=False)
        optimized,built=enumerate_model(problem)
        self.assertEqual(base.boards,expected);self.assertEqual(optimized.boards,expected)
        self.assertEqual(base.count,2*len(expected));self.assertEqual(optimized.count,len(expected))
        self.assertGreater(built.stats['circuit_reused_groups']+built.stats['circuit_complement_groups'],0)

    def test_direct_gold_proves_multiloop_infeasible_without_candidate_cuts(self):
        result=solve(pair_problem(multiple_loops=True),engine='cp-sat-circuit',seconds=5)
        self.assertEqual(result['status'],'infeasible');self.assertTrue(result['complete'])
        self.assertEqual(result['stats']['gold_cuts'],0)
        self.assertEqual(result['stats']['candidates_checked'],0)
        self.assertEqual(result['stats']['connectivity_encoding'],'direct_optional_endpoint_circuit')

    def test_edges_target_explicitly_omits_single_loop_constraint(self):
        problem=pair_problem(multiple_loops=True)
        with patch('constraint_circuit.build_circuit_model',side_effect=AssertionError('Edges must not need Gold circuit')):
            result=solve(problem,engine='cp-sat-circuit',seconds=5,target='edges')
        self.assertEqual(result['status'],'edge_perfect')
        self.assertEqual(result['stats']['connectivity_encoding'],'not_required_for_edges')

    def test_known_planted_gold_certificates_at_10_40_and_160_cells(self):
        for size in (1,2,4):
            data,board,reference=planted_gold(928+size,size,0)
            problem=SearchProblem(data,board)
            # Ten-cell model remains unrestricted. Larger fixtures retain both
            # the certified orientation and an alternate rotation of each
            # physical piece, exercising optional ports at full topology size.
            if size>1:problem.domains=[[code,3*(code//3)+(code+1)%3] for code in reference]
            result=solve(problem,engine='cp-sat-circuit',seconds=10,hints=reference)
            self.assertEqual(result['status'],'solved',result)
            self.assertTrue(problem.validate(result['codes'])['valid'])
            self.assertEqual(result['stats']['gold_cuts'],0)
            self.assertEqual(result['stats']['circuit_direction_symmetry'],True)

    def test_cooperative_cancellation_inside_builder_never_returns_partial_model(self):
        problem=pair_problem();calls=[0]
        def cancel():
            calls[0]+=1
            if calls[0]==9:raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):build_circuit_model(problem,problem.domains,check=cancel)
        event=threading.Event()
        result=solve(problem,engine='cp-sat-circuit',seconds=5,stop=event,
                     progress=lambda value:event.set() if value['phase']=='building_model' else None)
        self.assertEqual(result['status'],'stopped');self.assertFalse(result['complete']);self.assertEqual(result['nodes'],0)

    def test_model_caps_are_explicitly_unsupported_not_infeasible(self):
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem()
        for limits in ({'max_cells':1},{'max_placements':1},{'max_arcs':1},{'max_link_terms':0}):
            with self.subTest(limits=limits),self.assertRaises(CircuitLimit):
                build_circuit_model(problem,problem.domains,**limits)
        with patch('constraint_circuit.build_circuit_model',side_effect=CircuitLimit('test model cap')):
            result=solve(problem,engine='cp-sat-circuit',seconds=5)
        self.assertEqual(result['status'],'unsupported');self.assertFalse(result['complete'])
        self.assertIn('test model cap',result['reason'])

    def test_build_interruption_and_native_resume_rebuild_remain_incomplete(self):
        problem=pair_problem();first=solve(problem,engine='cp-sat-circuit',seconds=0)
        self.assertEqual(first['status'],'timeout');self.assertFalse(first['complete'])
        second=solve(problem,engine='cp-sat-circuit',seconds=5,resume=copy.deepcopy(first['checkpoint']))
        self.assertEqual(second['status'],'solved');self.assertTrue(problem.validate(second['codes'])['valid'])
        self.assertEqual(second['stats']['resumption_mode'],'rebuild_direct_circuit_with_revalidated_gold_cuts')

    def test_builder_cannot_widen_fixed_or_narrowed_problem_domains(self):
        fixed=pair_problem(fixed={0:0})
        rows=copy.deepcopy(fixed.domains);rows[0].append(1)
        with self.assertRaisesRegex(ValueError,'subsets'):build_circuit_model(fixed,rows)
        narrowed=pair_problem();narrowed.domains[0].remove(0)
        rows=copy.deepcopy(narrowed.domains);rows[0].append(0)
        with self.assertRaisesRegex(ValueError,'subsets'):build_circuit_model(narrowed,rows)

    def test_actual_crossings_are_separate_segment_edges(self):
        # All endpoints and same-side U turns remain explicit; crossing artwork
        # introduces no central node. Two disjoint drawn connections cannot be
        # made one loop by a geometric intersection in the tile interior.
        problem=pair_problem(multiple_loops=True)
        built=build_circuit_model(problem,[[0],[3]])
        self.assertEqual(built.stats['circuit_nodes'],8)
        self.assertEqual(built.stats['circuit_segment_edges'],4)
        solver=cp_model.CpSolver();self.assertEqual(solver.solve(built.model),cp_model.INFEASIBLE)


if __name__=='__main__':unittest.main()
