"""The CPU before GPU dispatch may retire only provably empty physical inventories."""
from itertools import permutations
from unittest import TestCase
from unittest.mock import patch
import random

from gpu_domain_runner import inventory_coverage,inventory_screen,solve,split_cubes
from search_fixtures import planted_gold
from search_problem import SearchProblem


def crossing_box(feasible):
    pairs=[[0,2],[0,3],[0,4],[1,2],[1,3],[1,4,5] if feasible else [1,4]]
    return tuple(sum(1<<(3*t) for t in row) for row in pairs)+((1<<24)-1,)*2


class HostHallTests(TestCase):
    def test_higher_order_deficit_and_positive_portable_frontier(self):
        from constraint_work import WorkScope,frontier_snapshot,restore_frontier
        from test_gpu_domain_round25 import planted160_fixture,independent_solutions
        from itertools import combinations
        bad=crossing_box(False)
        good=crossing_box(True)
        live,closed=inventory_screen([bad,good],8)
        self.assertEqual(live,[good])
        self.assertEqual(closed,[(bad,tuple(range(6)))])
        for a,b in combinations(bad[:6],2):
            union=a|b
            self.assertGreaterEqual(union.bit_count(),sum(not row&~union for row in bad))
        p=planted160_fixture()
        scope=WorkScope.for_problem(p)
        boxes=split_cubes([scope.root],128)
        portable=frontier_snapshot(scope,boxes)
        restored=restore_frontier(portable,scope)
        live,closed=inventory_screen(restored,p.n)
        solutions=independent_solutions(p,'gold')
        self.assertTrue(solutions)
        self.assertEqual(closed,[])
        for codes in solutions:
            self.assertEqual(sum(all(row[i] & 1<<c for i,c in enumerate(codes)) for row in live),1)

    def test_rotated_domains_and_random_boxes_match_physical_oracle(self):
        rng=random.Random(343)
        n=5
        boxes=[(3,4,8,64,512),
               tuple(7<<(3*i) for i in range(n))]
        for _ in range(256):
            a=rng.randrange(1,1<<(3*n))
            b=rng.randrange(1,1<<(3*n))
            boxes.append(tuple(rng.choice((a,b,7,5,2)) for _ in range(n)))
        live,closed=inventory_screen(boxes,n)
        self.assertIn(boxes[1],live)
        self.assertIn(boxes[0],[row for row,_ in closed])
        for row in boxes:
            possible=any(all(row[i] & (7<<(3*d)) for i,d in enumerate(order))
                         for order in permutations(range(n)))
            self.assertEqual(row in live,possible)

    def test_measured_full_root_and_early_resume(self):
        n=160
        rows=split_cubes([((1<<(3*n))-1,)*n],128)
        live,closed=inventory_screen(rows,n)
        self.assertEqual((len(live),len(closed)),(81,47))
        self.assertEqual(set(live)|{row for row,_ in closed},set(rows))
        tiles,geometry,codes=planted_gold(43,1,4)
        problem=SearchProblem(tiles,geometry)
        problem.domains=[[c] for c in codes]
        problem.domains[0]=[0,1]
        problem.domains[1]=[2]
        run=dict(seconds=5,backend='cuda',replicas=1,options={'gpu_global_matching':True})
        with patch('gpu_domain_round24.DomainCUDA',side_effect=AssertionError("Allocated a certified-empty lane")):
            first=solve(problem,**run)
            again=solve(problem,resume=first['checkpoint'],**run)
        self.assertEqual((first['status'],again['status']),('infeasible','infeasible'))
        self.assertEqual(first['checkpoint']['host_hall_closed'],1)
        self.assertEqual(again['stats']['gpu_host_hall_closed'],1)

    def test_ledger_combines_original_root_cuts_and_reopens_identical_work(self):
        from constraint_work import CoverageLedger, WorkScope
        from pathlib import Path
        from tempfile import TemporaryDirectory
        n = 160
        root = ((1 << (3 * n)) - 1,) * n
        scope = WorkScope('full-physical-inventory', 'gold', root, root)
        boxes = split_cubes([root], 128)
        live, cuts = inventory_screen(boxes, n)
        merged = list(inventory_coverage(cuts, root))
        self.assertEqual((len(cuts), len(merged)), (47, 1))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coverage.sqlite'
            with CoverageLedger(path, scope) as ledger:
                for box, cells in merged:
                    ledger.record_completed(box, status='infeasible', complete=True,
                                            evidence={'kind': 'physical_inventory_hall', 'cells': list(cells)})
            with CoverageLedger(path, scope) as ledger:
                left, complete = ledger.remaining(boxes)
                self.assertTrue(complete)
                self.assertEqual(list(left), live)
                self.assertEqual(len(ledger.exclusions()), 1)

    def test_lifted_cuts_preserve_every_physical_injection_in_restricted_roots(self):
        from constraint_work import contains
        from itertools import product

        # These two failed boxes have a feasible hull. Each tile can use distinct
        # rotations, and the third cell has a genuine restriction in the root.
        root = (63, 63, 448)
        failed = [(1, 4, 64), (8, 32, 64)]
        live, cuts = inventory_screen(failed, len(root))
        self.assertFalse(live)
        proofs = list(inventory_coverage(cuts, root))
        self.assertEqual(len(proofs), 2)
        for box, cells in proofs:
            self.assertTrue(contains(root, box))
            available = {v // 3 for i in cells for v in range(9) if box[i] & (1 << v)}
            self.assertLess(len(available), len(cells))
        self.assertTrue(all(any(contains(box, row) for box, _ in proofs) for row in failed))
        for row in product(*(tuple(v for v in range(9) if choices & (1 << v)) for choices in root)):
            if len({v // 3 for v in row}) == len(root):
                self.assertFalse(any(all(mask & (1 << v) for mask, v in zip(box, row)) for box, _ in proofs))
        self.assertEqual(list(inventory_coverage([], root)), [])
        self.assertEqual(len(list(inventory_coverage(cuts * 2, root))), 2)

    def test_solver_publishes_group_once_and_second_start_recovers_without_gpu(self):
        from constraint_work import CoverageLedger, WorkScope
        from pathlib import Path
        from tempfile import TemporaryDirectory

        tiles, geometry, _ = planted_gold(43, 1, 4)
        problem = SearchProblem(tiles, geometry)
        problem.domains = [[0, 1, 2, 3, 4, 5] for _ in range(problem.n)]
        scope = WorkScope.for_problem(problem, 'gold')
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / 'coverage.sqlite'
            args = dict(backend='cuda', replicas=8, learning_state={'_coverage_path': str(file)},
                        options={'gpu_global_matching': True, 'shared_coverage': True})
            with patch('gpu_domain_round24.DomainCUDA', side_effect=AssertionError('Excluded queue allocated GPU')):
                first = solve(problem, **args)
                second = solve(problem, **args)
            self.assertEqual([first['status'], second['status']], ['infeasible', 'infeasible'])
            self.assertEqual(first['checkpoint']['host_hall_closed'], 8)
            self.assertEqual(second['stats']['gpu_host_hall_closed'], 0)
            with CoverageLedger(file, scope) as ledger:
                self.assertEqual(tuple(ledger.exclusions()), (scope.root,))
