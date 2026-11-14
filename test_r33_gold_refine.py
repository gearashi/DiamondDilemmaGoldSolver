"""Direct checks of the signed integer refinement and pattern-assumption scope."""
from collections import Counter
from importlib.util import find_spec
from itertools import product
from math import inf
from unittest import TestCase
from unittest.mock import patch

from r27_structure_fixtures import planted
from r29_gold_phase import EndpointCuts, _reciprocal
from r32_gold_bonds import inventory
from r32_gold_loss import reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerPatternFilter
import unittest


def fixture(split, uniform):
    p, _ = planted(1, 3, 1, split, uniform)
    codes = [3*i for i in range(p.n)]
    ref, _, groups = reference_groups(p, codes)
    packed, required, outside = OutsideSyndromes.layout(ref, groups, inventory(p).required)
    return p, ref, packed, required, outside


def exact_costs(problem, reference, groups, required, outside):
    """Independent enumeration over every tile phase, using actual signed-label counts."""
    sides = EndpointCuts(problem).sides
    costs = {}
    for phases in product((0, 1), repeat=problem.n):
        labels = [(mask, color ^ phases[i//3]) for i, (mask, color) in enumerate(sides)]
        counts = Counter(labels)
        valid = True
        for pair, count in counts.items():
            other = _reciprocal(*pair)
            if (other == pair and count % 2) or counts[other] != count:
                valid = False
                break
        if not valid:
            continue
        failed = {pair for pair in reference | required if labels[pair[0]] != _reciprocal(*labels[pair[1]])}
        if failed & required or any(not (group-required) for group in groups):
            continue
        # Disjoint groups need one actual broken bond even when every phase agrees.
        price = len(failed) + sum(not bool(group & failed) for group in groups)
        signature = frozenset(i for i, bond in enumerate(outside) if bond in failed)
        costs[signature] = min(costs.get(signature, price), price)
    return costs


@unittest.skipUnless(find_spec('ortools'), 'Integer refinement needs optional OR-Tools')
class IntegerRefinementTests(TestCase):
    def test_every_outside_pattern_matches_exact_integer_inventory(self):
        p, ref, groups, req, out = fixture(True, True)
        costs = exact_costs(p, ref, groups, req, out)
        self.assertEqual(len(out), 6)
        patterns = [frozenset(i for i, bit in enumerate(row) if bit)
                    for row in product((0, 1), repeat=len(out))]
        for cap in (5, 8):
            check = IntegerPatternFilter(p, ref, groups, req, out, cap)
            results = [check(pattern, inf) for pattern in patterns]
            expected = [costs.get(pattern, inf) <= cap for pattern in patterns]
            self.assertEqual(expected, results)
            self.assertIn(True, results)
            self.assertIn(False, results)
            self.assertEqual(check(patterns[0], inf), results[0])
            self.assertEqual(list(check.model.proto.assumptions), [])

    def test_zero_loss_planted_gold_and_required_reference_bonds(self):
        p, ref, groups, req, out = fixture(False, False)
        self.assertTrue(p.validate([3*i for i in range(p.n)])['valid'])
        self.assertTrue(req & ref)
        check = IntegerPatternFilter(p, ref, groups, req, out, 0)
        self.assertTrue(check(set(), inf))
        self.assertFalse(check({0}, inf))
        self.assertTrue(check(set(), inf))

    def test_unknown_discards_assumptions_and_remains_open(self):
        p, ref, groups, req, out = fixture(False, False)
        check = IntegerPatternFilter(p, ref, groups, req, out, 0)
        with self.assertRaises(TimeoutError):
            check(set(), 0)
        with patch.object(check.solver, 'solve', return_value=check.cp.UNKNOWN):
            with self.assertRaises(TimeoutError):
                check({2}, inf)
        self.assertEqual(list(check.model.proto.assumptions), [])
        self.assertTrue(check(set(), inf))
        with self.assertRaises(ValueError):
            check({len(out)}, inf)
        with self.assertRaises(ValueError):
            IntegerPatternFilter(p, ref, groups, req, out[:-1], 0)
