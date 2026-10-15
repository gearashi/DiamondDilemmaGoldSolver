"""Independent finite solution sets for exact fixed-chain contraction."""
import copy
import random
import unittest

from test_dependencies import cp_model, requires_ortools
from constraint_circuit import build_circuit_model, CircuitLimit
from constraint_residual import _rotation_tables
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_models import pair_problem, brute
from test_constraint_dfs import tetrahedron_tiles, board
from test_constraint_knowledge import mixed_problem


def symmetric_planted(seed=107, size=4):
    """Independent Gold fixture with equal masks but differing internal paths."""
    data,geometry,reference=planted_gold(seed,size,1000)
    # Each seam has two ports. Order-preserving relabeling to {2,10} preserves
    # p -> 12-p, every original segment connection, and the certified loop.
    for tile in data['tiles']:
        positions={side:sorted({p for segment in tile['segments'] for s,p in segment if s==side})
                   for side in range(3)}
        assert all(len(row)==2 for row in positions.values())
        tile['segments']=[[[s,2 if p==positions[s][0] else 10] for s,p in segment]
                          for segment in tile['segments']]
    problem=SearchProblem(data,geometry)
    assert problem.validate(reference)['valid']
    return problem,reference


def enumerate_domains(problem, rows, *, contract_fixed=True, break_direction=True):
    built=build_circuit_model(problem,rows,contract_fixed=contract_fixed,break_direction=break_direction)
    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):super().__init__();self.boards=set();self.count=0
        def on_solution_callback(self):
            self.boards.add(tuple(int(self.value(code)) for code in built.codes));self.count+=1
    collector=Collector();solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1;solver.parameters.enumerate_all_solutions=True
    solver.parameters.max_time_in_seconds=5
    status=solver.solve(built.model,collector)
    if status not in (cp_model.OPTIMAL,cp_model.INFEASIBLE):
        raise AssertionError('Residual finite enumeration was inconclusive')
    if built.model.validate():raise AssertionError(built.model.validate())
    return collector,built


class ResidualCircuitTests(unittest.TestCase):
    @requires_ortools
    def test_finite_domain_solution_sets_match_full_model_and_validator(self):
        tiles,geometry=tetrahedron_tiles()
        fixtures=[pair_problem(),pair_problem(multiple_loops=True),pair_problem(boundary=True),
                  pair_problem(fixed={0:0}),mixed_problem(),SearchProblem(tiles,geometry)]
        rng=random.Random(109783)
        for index,problem in enumerate(fixtures):
            valid=brute(problem,'gold');original=copy.deepcopy(problem.domains)
            for case in range(18):
                if case==0:rows=copy.deepcopy(problem.domains)
                elif case==1:rows=[[],*copy.deepcopy(problem.domains[1:])]
                else:
                    rows=[([rng.choice(row)] if row and rng.random()<.45 else
                           [code for code in row if rng.random()<.6]) for row in problem.domains]
                expected={codes for codes in valid if all(code in rows[cell] for cell,code in enumerate(codes))}
                with self.subTest(fixture=index,case=case):
                    contracted,_=enumerate_domains(problem,rows)
                    full,_=enumerate_domains(problem,rows,contract_fixed=False)
                    self.assertEqual(contracted.boards,expected)
                    self.assertEqual(contracted.boards,full.boards)
                    self.assertEqual(problem.domains,original)

    @requires_ortools
    def test_fixed_anchor_and_variable_anchor_preserve_direction_and_parallel_paths(self):
        problem=mixed_problem()
        for fixed_code in (0,3):
            rows=[[fixed_code],list(range(6))]
            expected={codes for codes in brute(problem,'gold') if codes[0]==fixed_code}
            unrestricted,_=enumerate_domains(problem,rows,break_direction=False)
            contracted,built=enumerate_domains(problem,rows)
            self.assertEqual(contracted.boards,expected)
            self.assertEqual(unrestricted.boards,expected)
            self.assertEqual(contracted.count,len(expected))
            self.assertEqual(unrestricted.count,2*len(expected))
            self.assertEqual(built.stats['circuit_fixed_cells'],1)
            self.assertEqual(built.stats['circuit_contracted_ports'],6)
            self.assertEqual(built.stats['circuit_contracted_paths'],3)
            self.assertEqual(built.stats['circuit_nodes'],6)
            self.assertEqual(built.stats['circuit_fixed_inventory_removals'],3)

    @requires_ortools
    def test_all_fixed_gold_needs_no_circuit_but_full_codes_are_reconstructed(self):
        problem=pair_problem();reference=next(iter(brute(problem,'gold')))
        found,built=enumerate_domains(problem,[[code] for code in reference])
        self.assertEqual(found.boards,{reference})
        self.assertEqual(built.stats['circuit_nodes'],0)
        self.assertEqual(built.stats['circuit_arcs'],0)
        self.assertEqual(built.stats['circuit_fixed_cycles'],1)
        self.assertEqual([code.name for code in built.codes],['code_0','code_1'])
        self.assertTrue(problem.validate(list(next(iter(found.boards))))['valid'])

    @requires_ortools
    def test_closed_proper_fixed_loop_is_not_an_empty_solved_residual(self):
        data=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        geometry=board(4,[(a,side,a+1,side) for a in (0,2) for side in range(3)])
        problem=SearchProblem(data,geometry)
        rows=[[0],[3],[6,7,8,9,10,11],[6,7,8,9,10,11]]
        found,built=enumerate_domains(problem,rows)
        self.assertFalse(found.boards)
        self.assertEqual(built.stats['circuit_residual_rejection'],'closed_proper_fixed_loop')
        self.assertGreater(built.stats['circuit_fixed_cycles'],0)
        all_fixed=pair_problem(multiple_loops=True)
        found,built=enumerate_domains(all_fixed,[[0],[3]])
        self.assertFalse(found.boards);self.assertEqual(built.stats['circuit_fixed_cycles'],2)

    @requires_ortools
    def test_duplicate_fixed_inventory_empty_domains_and_fixed_seam_mismatch(self):
        problem=pair_problem()
        for rows in ([[0],[0]],[[],[3]],[[0],[4]]):
            found,built=enumerate_domains(problem,rows)
            expected={codes for codes in brute(problem,'gold') if all(c in rows[i] for i,c in enumerate(codes))}
            self.assertEqual(found.boards,expected)
            if not expected:self.assertIn('circuit_residual_rejection',built.stats)

    @requires_ortools
    def test_narrowed_160_cell_symmetric_fixture_contracts_without_deciding_open_cells(self):
        problem,reference=symmetric_planted()
        open_cells=list(range(0,160,20));pool=[3*(reference[cell]//3)+r for cell in open_cells for r in range(3)]
        rows=[[code] for code in reference]
        for cell in open_cells:rows[cell]=pool
        built=build_circuit_model(problem,rows,contract_fixed=True)
        self.assertEqual(built.stats['circuit_fixed_cells'],152)
        self.assertEqual(built.stats['circuit_residual_placements'],192)
        self.assertEqual(built.stats['circuit_nodes'],48)
        self.assertEqual(built.stats['circuit_contracted_ports'],912)
        self.assertGreater(built.stats['circuit_contracted_paths'],0)
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=5
        status=solver.solve(built.model)
        self.assertIn(status,(cp_model.OPTIMAL,cp_model.FEASIBLE))
        codes=[int(solver.value(code)) for code in built.codes]
        self.assertTrue(problem.validate(codes)['valid'])
        self.assertTrue(all(code in rows[cell] for cell,code in enumerate(codes)))

    @requires_ortools
    def test_limits_and_cancellation_are_inconclusive(self):
        problem=mixed_problem();rows=[[0],list(range(6))]
        for limits in ({'max_cells':1},{'max_placements':1},{'max_arcs':1}):
            with self.subTest(limits=limits),self.assertRaises(CircuitLimit):
                build_circuit_model(problem,rows,contract_fixed=True,**limits)
        calls=[0]
        def check():
            calls[0]+=1
            if calls[0]==12:raise InterruptedError('stop residual construction')
        with self.assertRaises(InterruptedError):
            build_circuit_model(problem,rows,contract_fixed=True,check=check)
        with self.assertRaises(ValueError):build_circuit_model(problem,rows,contract_fixed=1)

    def test_rotation_table_cache_binds_actual_geometry_and_returns_immutable_tables(self):
        problem=mixed_problem()
        key=tuple(tile['segments'] for tile in problem.tiles)
        first=_rotation_tables(key)
        self.assertIs(first,_rotation_tables(key))
        changed=list(key);changed[0]=(((0,3),(1,3)),)
        second=_rotation_tables(tuple(changed))
        self.assertNotEqual(first,second)
        self.assertEqual(first,_rotation_tables(key))
        self.assertIsInstance(first,tuple);self.assertIsInstance(first[0],tuple)


if __name__=='__main__':unittest.main()
