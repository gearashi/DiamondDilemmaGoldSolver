"""The CPU before GPU dispatch may retire only provably empty physical inventories."""
from itertools import permutations
from unittest import TestCase
from unittest.mock import patch
import random

from gpu_domain_runner import inventory_screen,solve,split_cubes
from search_fixtures import planted_gold
from search_problem import SearchProblem


class HostHallTests(TestCase):
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
        for row,_ in closed:
            self.assertFalse(any(all(row[i] & (7<<(3*d)) for i,d in enumerate(order))
                                 for order in permutations(range(n))))

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
