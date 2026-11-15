"""Exhaustively compare the production slack encoding with subset enumeration."""
from collections import Counter
from random import Random
import unittest

from r35_contact_slack import exact_budget_clauses, excluded_losses


class WeightedBudgetTests(unittest.TestCase):
    def test_every_loss_subset_and_integer_cap(self):
        try:
            from pysat.solvers import Solver
        except ImportError:
            self.skipTest("The optional native SAT package is unavailable.")
        rng = Random(3581)
        labels = [(2*i, 2*i+1) for i in range(6)]
        ids = {bond: i+1 for i, bond in enumerate(labels)}
        required = {(20, 21)}
        for _ in range(12):
            groups = [(set(rng.sample(labels, rng.randrange(1, 6))), rng.randrange(1, 5))
                      for _ in range(rng.randrange(2, 7))]
            degrees = Counter()
            for group, weight in groups:
                for bond in group:
                    degrees[bond] += weight
            certificate = {
                'maximum': max(degrees.values()), 'weight_sum': sum(w for _, w in groups),
                'weighted_cores': [
                    {'bonds': sorted(group | required), 'weight': weight} for group, weight in groups
                ],
            }
            at_least_one = [[ids[bond] for bond in group] for group, _ in groups]
            for cap in range(7):
                extra = exact_budget_clauses(certificate, required, labels, ids, cap, len(labels))
                single, pairs = excluded_losses(labels, required, cap, certificate)
                with Solver(name='g4', bootstrap_with=[*at_least_one, *extra]) as solver:
                    for mask in range(64):
                        lost = {bond for i, bond in enumerate(labels) if mask & (1 << i)}
                        expected = len(lost) <= cap and all(group & lost for group, _ in groups)
                        pins = [n if b in lost else -n for b, n in ids.items()]
                        self.assertEqual(solver.solve(assumptions=pins), expected, (cap, mask, groups))
                        if expected:
                            self.assertFalse(single & lost)
                            self.assertFalse(any(a in lost and b in lost for a, b in pairs))


if __name__ == '__main__':
    unittest.main()
