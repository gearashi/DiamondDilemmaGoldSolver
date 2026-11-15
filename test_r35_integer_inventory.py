"""Exhaustive raw-endpoint oracles for partial SAT, finite integer images and complement symmetry."""
from importlib.util import find_spec
from itertools import product
from math import inf
from unittest import TestCase, skipUnless

from bench_r35_loss_proof import physical_price
from r35_integer_phases import EnumeratedPatternFilter, IntegerPhaseImages
from test_r33_gold_refine import fixture


class FiniteInventoryTests(TestCase):
    @staticmethod
    def prices(problem, ref, groups, req, outside):
        costs = {}
        for phase in range(1 << problem.n):
            got = physical_price(problem.tiles, ref, groups, req, phase)
            other = physical_price(problem.tiles, ref, groups, req, phase ^ ((1<<problem.n)-1))
            if got != other:
                raise AssertionError('Globally complementary colors have different physical prices.')
            if got:
                price, failed = got
                value = sum(1 << i for i,bond in enumerate(outside) if bond in failed)
                costs[value] = min(price, costs.get(value, inf))
        return costs

    def test_direct_exact_every_pattern_boundary_and_valid_witness(self):
        for split, uniform in ((True,True), (False,False), (False,True)):
            p, ref, groups, req, outside = fixture(split, uniform)
            costs = self.prices(p, ref, groups, req, outside)
            image = IntegerPhaseImages.compile(p, groups, req, outside)
            count = 1 << len(outside)
            self.assertLessEqual(count, 1 << 16)
            for value in range(count):
                indices = frozenset(i for i in range(len(outside)) if value>>i & 1)
                cost, phase, support = image.minimum(indices)
                self.assertEqual(cost, costs.get(value, inf))
                if phase is None:
                    self.assertTrue(all((value & support) != (other & support) for other in costs))
                else:
                    actual, mismatches = physical_price(p.tiles, ref, groups, req, phase)
                    self.assertEqual(actual,cost)
                    self.assertEqual(indices,{i for i,bond in enumerate(outside) if bond in mismatches})
            with self.assertRaises(TimeoutError):
                image.minimum((),0)
            check = EnumeratedPatternFilter(image,outside,0)
            with self.assertRaises(TimeoutError):
                check((),0)
            self.assertLessEqual(1,len(costs))

    @skipUnless(find_spec('pysat'), 'Optional SAT backend')
    def test_partial_and_complete_sat_vs_all_raw_phases_all_budget_boundaries(self):
        from r35_integer_sat import InventorySAT
        p,ref,groups,req,outside = fixture(True,True)
        costs = self.prices(p, ref, groups, req, outside)
        for excess in (False, True):
            for cap in range(max(costs.values())+2):
                check=InventorySAT(p, groups, req, outside, cap, excess=excess)
                try:
                    for states in product((-1,0,1), repeat=len(outside)):
                        mask = sum(1<<i for i,x in enumerate(states) if x>=0)
                        value = sum(1<<i for i,x in enumerate(states) if x==1)
                        expected = any(k & mask == value and cost<=cap for k,cost in costs.items())
                        self.assertEqual(check.solve(mask,value),expected,(cap,states))
                finally:
                    check.close()

    @skipUnless(find_spec('pysat'), 'Optional SAT backend')
    def test_planted_valid_loop_at_zero_and_unrestricted_adverse_case(self):
        from r35_integer_sat import InventorySAT
        p,ref,groups,req,outside = fixture(False,False)
        self.assertTrue(p.validate([3*i for i in range(p.n)])['valid'])
        baseline=InventorySAT(p,groups,req,outside,0)
        try:
            self.assertTrue(baseline.solve(0,0))
            phase=sum(1<<(x-1) for x in baseline.solver.get_model() if 0<x<=p.n)
            self.assertEqual(physical_price(p.tiles,ref,groups,req,phase)[0],0)
            self.assertFalse(baseline.solve(1,1))
        finally:
            baseline.close()
