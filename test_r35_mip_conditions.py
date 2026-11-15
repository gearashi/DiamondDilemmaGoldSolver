"""Check native spherical and physical mate restrictions against finite direct search."""
import unittest
from importlib.util import find_spec
from math import inf
from pathlib import Path
from types import SimpleNamespace

from r35_loss_witness import physical_price
from r35_signed_conditions import SignedConditions
from test_r33_gold_refine import fixture


class DirectInventoryTests(unittest.TestCase):
    def test_all_gauged_phases_on_planted_boards(self):
        for split, uniform, count, minimum in (
            (False, False, 3, 0), (True, True, 59, 4), (True, False, 0, None)
        ):
            problem, reference, groups, required, _ = fixture(split, uniform)
            full = (1 << problem.n) - 1
            rules = SignedConditions(problem, euler=True, mate_cost=True).for_loss(reference, groups, required, 30)
            prices = []
            for phase in range(0, full + 1, 2):
                ordinary = physical_price(problem.tiles, reference, groups, required, phase)
                if ordinary is None:
                    continue
                accepted = rules.accepts(phase)
                self.assertEqual(accepted, rules.accepts(full ^ phase))
                if accepted:
                    prices.append(rules.price(phase, ordinary[0]))
            self.assertEqual((len(prices), min(prices, default=None)), (count, minimum))

    def test_bundled_cost33_phases_recheck_under_both_gold_scopes(self):
        from bench_r35_loss_boundary344 import problem_scope
        from r35_loss_bound import REFERENCE_PHASES
        from verify_r35_contact_wide import read_rows
        tiles = next((p for p in map(Path, ('data/tiles.json','../data/tiles.json')) if p.is_file()), None)
        if tiles is None:
            self.skipTest('Original prepared tiles are unavailable')
        problem, ref, initial, req = problem_scope(tiles)
        directory = Path(__file__).resolve().parent / 'docs'
        groups = [*initial[:13], *read_rows(directory/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
                  *read_rows(directory/'R35_EDGE_CONTACTS.json','edge_perfect_cores')]
        rules = SignedConditions(problem,euler=True,mate_cost=True).for_loss(ref,groups,req,40)
        data = ((0xcc57eae4e99109f8b073557e8c590b9202465766,34),
                (0xcc778d82e99c517987f6b57ec15837c21207cf06,37))
        for value, coupled_cost in data:
            self.assertIn(value,REFERENCE_PHASES)
            self.assertEqual(physical_price(problem.tiles,ref,groups,req,value)[0],33)
            self.assertTrue(rules.accepts(value))
            self.assertEqual(rules.price(value,33),coupled_cost)


@unittest.skipUnless(find_spec('ortools'), 'Native signed integer tests need optional OR-Tools')
class NativeConditionsTests(unittest.TestCase):
    def test_raw_euler_with_negative_coefficients_gaps_and_unreachable_totals(self):
        from ortools.linear_solver import pywraplp
        from r35_mip_conditions import euler_row

        for allowed in ((-2, 0, 2), (0,), (-4, 2), (1,), (-8, -4, 0, 4)):
            for bits in range(8):
                engine = pywraplp.Solver.CreateSolver('SCIP')
                phase = [engine.BoolVar('') for _ in range(3)]
                rows = [((2,), (-2,)), ((-1,), (1,)), ((1,), (3,))]
                direct = sum(row[(bits >> i) & 1][0] for i, row in enumerate(rows))
                for i, flag in enumerate(phase):
                    engine.Add(flag == ((bits >> i) & 1))
                euler_row(engine, phase, SimpleNamespace(rows=rows, euler=set(allowed)))
                self.assertEqual(engine.Solve() == engine.OPTIMAL, direct in allowed)

    def test_conditional_native_matches_full_plant_enumeration(self):
        from bench_r35_mip_conditions import factory
        from r35_contact_mip import decide

        for split, uniform, cap, expected in (
            (False, False, 0, True), (True, True, 3, False),
            (True, True, 4, True), (True, False, 15, False)
        ):
            problem, reference, groups, required, outside = fixture(split, uniform)
            for mode in ('whole','lazy','short','warm','rules','small','families','price'):
                with self.subTest(split=split, uniform=uniform, cap=cap, mode=mode):
                    conditions = SignedConditions(problem, euler=True, mate_cost=True)
                    conditions = conditions.for_loss(reference, groups, required, cap)
                    result = decide(problem, reference, groups, required, outside, cap, 0,
                                    conditions=conditions, condition_factory=factory(mode), first_feasible=True)
                    if not expected:
                        self.assertEqual(result['status'], 'INFEASIBLE')
                        continue
                    self.assertEqual(result['status'], 'INTEGER_RELAXATION_FEASIBLE')
                    phase = int(result['phase'],16)
                    score = physical_price(problem.tiles,reference,groups,required,phase)
                    self.assertIsNotNone(score)
                    self.assertTrue(conditions.accepts(phase))
                    self.assertEqual(result['loss_cost'],conditions.price(phase,score[0]))
                    self.assertLessEqual(result['loss_cost'],cap)

    def test_real_matching_obstruction_generalizes_across_a_partial_phase(self):
        from bench_r35_loss_boundary344 import problem_scope
        from ortools.linear_solver import pywraplp
        from r35_mip_conditions import Conditions, _violates
        from r35_signed_mate_support import matching

        tiles = next((p for p in map(Path, ('data/tiles.json', '../data/tiles.json')) if p.is_file()), None)
        if tiles is None:
            self.skipTest('Original prepared tiles are unavailable')
        problem, _, _, _ = problem_scope(tiles)
        rules = SignedConditions(problem, mates=True)
        engine = pywraplp.Solver.CreateSolver('SCIP')
        phase = [engine.BoolVar('') for _ in range(problem.n)]
        engine.Add(phase[0] == 0)
        native = Conditions(rules,engine,phase,{},startup='rules')
        bad = 0xce55aaa6e89109fca073547e8c580ba32e0e5fa2
        self.assertFalse(any(_violates(row,bad) for row in rules.initial))
        self.assertTrue(native.reject(bad,21,inf))
        conflict = next(row for row in rules.rejecting if _violates(row,bad))
        self.assertEqual(len(conflict),10)
        pins = {abs(v)-1:int(v<0) for v in conflict}
        family = next((s,e) for s,e,_,_ in rules.families if not matching(s,e,pins))
        for tile in pins:
            self.assertTrue(matching(*family,{k:v for k,v in pins.items() if k!=tile}))
        for tile,value in pins.items():
            engine.Add(phase[tile] == value)
        self.assertEqual(engine.Solve(),engine.INFEASIBLE)
        self.assertTrue(rules.accepts(0xd237aba7e91101e88041413e8c589902021e5126))
