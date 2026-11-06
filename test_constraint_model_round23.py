"""Independent solution sets and model/rollback limits for experimental engines."""
import itertools,random,unittest
from unittest.mock import patch
from constraint_models import solve
from constraint_sat_connectivity import add_reachability,SATConnectivityLimit,gold_propagator
from search_problem import SearchProblem
from test_constraint_dfs import brute,simple_tiles,double_face,multiple_loops,tetrahedron_tiles
from test_constraint_pair_lookahead import turn_problem
from test_constraint_sat_domains import fixture
from test_constraint_sat_assist import HAS_SAT
try:import ortools;HAS_CP=True
except ImportError:HAS_CP=False

SAT_ENGINES=('sat-propagator','sat-reachability','sat-channel','sat-network')

@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class ExperimentalSatModels(unittest.TestCase):
    def test_complete_finite_gold_and_edge_results_match_enumeration(self):
        problems=[SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                  SearchProblem(*tetrahedron_tiles()),turn_problem(False),turn_problem(True)]
        for problem in problems:
            for target in ('gold','edges'):
                expected=brute(problem,target)
                for engine in SAT_ENGINES:
                    with self.subTest(n=problem.n,target=target,engine=engine,solutions=len(expected)):
                        answer=solve(problem,engine=engine,target=target,seconds=20)
                        self.assertTrue(answer['complete'],answer)
                        if expected:self.assertIn(tuple(answer['codes']),expected)
                        else:self.assertEqual(answer['status'],'infeasible')

    def test_every_edge_perfect_fixture_board_classified_by_direct_connectivity(self):
        problem,_=fixture();edge=brute(problem,'edges');gold=set(brute(problem,'gold'));original=problem.domains
        self.assertEqual((len(edge),len(gold)),(72,32))
        try:
            for row in edge:
                problem.domains=[[code] for code in row]
                for engine in ('sat-propagator','sat-reachability'):
                    answer=solve(problem,engine=engine,seconds=5)
                    self.assertEqual(answer['status'],'solved' if row in gold else 'infeasible',(row,engine,answer))
        finally:problem.domains=original

    def test_partial_external_conflicts_avoid_complete_bad_board_candidates(self):
        problem=turn_problem(False)
        baseline=solve(problem,engine='sat',seconds=20)
        improved=solve(problem,engine='sat-propagator',seconds=20)
        self.assertEqual(baseline['status'],improved['status']);self.assertEqual(improved['status'],'infeasible')
        self.assertGreater(improved['stats']['sat_external_partial_conflicts'],0)
        self.assertEqual(improved['stats']['candidates_checked'],0)
        self.assertGreater(baseline['stats']['candidates_checked'],0)

    def test_partial_propagator_backtracks_and_keeps_fixed_observations(self):
        problem=SearchProblem(simple_tiles(),double_face());variables=[];counter=0
        for row in problem.domains:
            ids={}
            for code in row:counter+=1;ids[code]=counter
            variables.append(ids)
        external=gold_propagator(problem,variables)
        external.on_new_level();external.on_assignment(1);external.on_new_level();external.on_assignment(2)
        external.on_assignment(-1,True);external.on_backtrack(0)
        self.assertEqual(external.assignments,{1:-1})
        self.assertFalse(external.has_clause())

    def test_limits_and_stop_are_inconclusive_and_resume_keeps_objective(self):
        problem=SearchProblem(simple_tiles(),double_face())
        for engine in SAT_ENGINES:
            stopped=solve(problem,engine=engine,seconds=10,stop=lambda:True)
            self.assertEqual(stopped['status'],'stopped');self.assertFalse(stopped['complete'])
            resumed=solve(problem,engine=engine,seconds=5,resume=stopped['checkpoint'])
            self.assertTrue(resumed['validation']['valid'])
        with patch('constraint_sat_connectivity.add_reachability',side_effect=SATConnectivityLimit('test cap')):
            answer=solve(problem,engine='sat-reachability',seconds=5)
        self.assertEqual(answer['status'],'unsupported');self.assertFalse(answer['complete'])

    def test_gate_port_cap_raises_before_misreporting_infeasibility(self):
        problem=SearchProblem(simple_tiles(),double_face());next_id=[0];clauses=[]
        def fresh():next_id[0]+=1;return next_id[0]
        variables=[{code:fresh() for code in row} for row in problem.domains]
        with self.assertRaises(SATConnectivityLimit):
            add_reachability(problem,problem.domains,variables,fresh,clauses.append,max_ports=1)


@unittest.skipUnless(HAS_CP,'optional ortools missing')
class ExperimentalRegionalModel(unittest.TestCase):
    def test_gold_and_edge_objectives_preserved(self):
        for problem in (SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                        SearchProblem(*tetrahedron_tiles()),turn_problem(False),turn_problem(True)):
            for target in ('gold','edges'):
                expected=brute(problem,target)
                answer=solve(problem,engine='cp-sat-regions',target=target,seconds=20)
                self.assertTrue(answer['complete'],answer)
                if expected:self.assertIn(tuple(answer['codes']),expected)
                else:self.assertEqual(answer['status'],'infeasible')
                self.assertGreater(answer['stats']['regional_model_regions'],0)

    def test_bounded_table_fallback_keeps_every_original_solution(self):
        from ortools.sat.python import cp_model
        from constraint_region_models import add_regional_alternatives
        problem=SearchProblem(*tetrahedron_tiles());expected=brute(problem,'gold')
        model=cp_model.CpModel()
        codes=[model.new_int_var_from_domain(cp_model.Domain.from_values(row),f'c{c}') for c,row in enumerate(problem.domains)]
        stats=add_regional_alternatives(problem,problem.domains,model,codes,max_rows=1,max_steps=2)
        self.assertEqual(stats['regional_model_regions'],problem.n)
        self.assertGreater(stats['regional_model_budget_fallbacks'],0)
        # The fallback creates exact single-cell tables, leaving all assignments
        # in the input product available; existing global rules handle them.
        for row in expected:
            clone=model.clone()
            for cell,code in enumerate(row):clone.add(codes[cell]==code)
            solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
            self.assertIn(solver.solve(clone),(cp_model.OPTIMAL,cp_model.FEASIBLE))
