"""Independent finite cardinality certificates for precise Hall reasons."""
from itertools import product,permutations
import unittest
from constraint_explanations import removal_witness,failure_witness
from constraint_pruning import Pruner
from test_constraint_pruning import blank_problem,tile_domain

class HallWitnessTests(unittest.TestCase):
    def test_every_three_by_three_graph_has_only_valid_certificates(self):
        for rows in product(range(8),repeat=3):
            solutions=[p for p in permutations(range(3)) if all(row&(1<<t) for row,t in zip(rows,p))]
            failed=failure_witness(rows)
            if not solutions:
                self.assertIsNotNone(failed,rows)
                union=0
                for c in failed:union|=rows[c]
                self.assertLess(union.bit_count(),len(failed))
                continue
            self.assertIsNone(failed)
            for cell in range(3):
                for tile in range(3):
                    if not rows[cell]&(1<<tile):continue
                    witness=removal_witness(rows,solutions[0],cell,tile)
                    supported=any(p[cell]==tile for p in solutions)
                    if supported:self.assertIsNone(witness)
                    else:
                        self.assertIsNotNone(witness,(rows,cell,tile))
                        self.assertNotIn(cell,witness)
                        union=0
                        for c in witness:union|=rows[c]
                        self.assertEqual(union.bit_count(),len(witness));self.assertTrue(union&(1<<tile))
                        # Release ALL other restrictions. This subset alone
                        # still excludes forcing this cell to the removed tile.
                        self.assertFalse(any(p[cell]==tile and all(rows[c]&(1<<p[c]) for c in witness) for p in permutations(range(3))))
    def test_bounds_stale_matching_and_cancellation_never_make_a_claim(self):
        self.assertIsNone(removal_witness([3,3,7],[0,1,2],2,0,limit=0))
        self.assertIsNone(removal_witness([2,3,7],[0,1,2],2,0))
        self.assertIsNone(failure_witness([1,1,7],limit=0))
        def cancel():raise InterruptedError()
        with self.assertRaises(InterruptedError):failure_witness([1,1,7],cancel)
        with self.assertRaises(InterruptedError):removal_witness([3,3,7],[0,1,2],2,0,cancel)
    def test_live_hall_callbacks_are_ordered_and_tight(self):
        p=Pruner(blank_problem(4),native='python',parity=False)
        domains=[tile_domain(x) for x in ((0,1),(0,1),(0,1,2,3),(2,3))];events=[]
        def event(c,value,deps):
            self.assertEqual(domains[c],value);events.append((c,value,deps))
        p.on_narrow=event;p.prune_hall(domains)
        self.assertEqual(events,[(2,tile_domain((2,3)),(0,1))])
        self.assertEqual(p.stats['hall_explanation_witnesses'],2)

if __name__=='__main__':unittest.main()
