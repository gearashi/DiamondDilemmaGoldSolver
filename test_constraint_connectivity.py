"""Independent Gold certificates and finite soundness for seam-graph cuts."""
from itertools import product
import random
import unittest
from constraint_connectivity import ConnectivityPruner
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import board, double_face, multiple_loops, simple_tiles, brute


def bits(rows):return [sum(1<<code for code in row) for row in rows]


class ConnectivityTests(unittest.TestCase):
    def test_rejects_disconnected_gold_graph_before_any_tile_is_fixed(self):
        geometry=board(4,[(0,s,1,s) for s in range(3)]+[(2,s,3,s) for s in range(3)])
        tiles=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        problem=SearchProblem(tiles,geometry);domains=bits(problem.domains)
        self.assertTrue(all(domain.bit_count()>1 for domain in domains))
        pruning=ConnectivityPruner(problem)
        self.assertTrue(pruning.prune(domains));self.assertIn(0,domains)
        self.assertEqual(pruning.stats['connectivity_failures'],1)
        self.assertEqual(brute(problem,'gold'),[])
        self.assertTrue(brute(problem,'edges'))

    def test_parallel_seams_are_not_incorrectly_classed_as_bridges(self):
        problem=SearchProblem(simple_tiles(),double_face())
        domains=bits(problem.domains);pruning=ConnectivityPruner(problem)
        pruning.prune(domains)
        self.assertEqual(pruning.stats['connectivity_bridges'],0)
        self.assertTrue(all(domain & (1<<code) for domain,code in zip(domains,[0,3])))

    def test_bridge_needs_positive_even_crossings_and_preserves_gold(self):
        geometry=board(4,[(0,0,1,0),(1,1,2,1),(2,0,3,0)])
        tiles=[{'id':0,'segments':[[[0,2],[0,10]]]},
               {'id':1,'segments':[[[0,10],[1,3]],[[0,2],[1,9]]]},
               {'id':2,'segments':[[[1,9],[0,4]],[[1,3],[0,8]]]},
               {'id':3,'segments':[[[0,8],[0,4]]]}]
        problem=SearchProblem(tiles,geometry);reference=[0,3,6,9]
        self.assertTrue(problem.validate(reference)['valid'])
        # Broader domains intentionally include rotations with a zero endpoint
        # count at an essential bridge. The checker need not assume prior AC.
        domains=[(1<<(3*problem.n))-1]*problem.n
        pruning=ConnectivityPruner(problem)
        self.assertTrue(pruning.prune(domains))
        self.assertEqual(pruning.stats['connectivity_bridges'],3)
        self.assertTrue(all(domain & (1<<code) for domain,code in zip(domains,reference)))
        self.assertGreater(pruning.stats['connectivity_prunes'],0)

    def test_small_domains_never_remove_an_independently_valid_gold_solution(self):
        rng=random.Random(173)
        problem=SearchProblem(simple_tiles(),double_face());solutions=brute(problem,'gold')
        for _ in range(180):
            domains=[rng.randrange(1,1<<(3*problem.n)) for _ in range(problem.n)]
            expected=[codes for codes in solutions if all(domain&(1<<code) for domain,code in zip(domains,codes))]
            ConnectivityPruner(problem).prune(domains)
            for codes in expected:
                self.assertTrue(all(domain&(1<<code) for domain,code in zip(domains,codes)))

    def test_large_planted_gold_survives_narrowed_domains(self):
        rng=random.Random(311)
        for size in (1,2,4):
            data,geometry,reference=planted_gold(197,size);problem=SearchProblem(data,geometry)
            for _ in range(20):
                domains=[(1<<code) if rng.random()<.6 else ((1<<(3*problem.n))-1) for code in reference]
                pruning=ConnectivityPruner(problem)
                pruning.prune(domains)
                self.assertTrue(all(domain&(1<<code) for domain,code in zip(domains,reference)))

    def test_stop_before_completed_graph_does_not_delete_domains(self):
        problem=SearchProblem(simple_tiles(),double_face());domains=bits(problem.domains);before=domains.copy()
        def check():raise InterruptedError()
        with self.assertRaises(InterruptedError):ConnectivityPruner(problem).prune(domains,check)
        self.assertEqual(domains,before)


if __name__=='__main__':unittest.main()
