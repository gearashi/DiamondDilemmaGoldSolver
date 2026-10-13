"""Finite independent checks for exact bounded separator/frontier summaries."""
from math import factorial
import random
import unittest
from constraint_separator import SeparatorPruner
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import board,double_face,simple_tiles,multiple_loops,tetrahedron_tiles,brute
from test_constraint_advanced import bits,domains_of,chain


def ring(n):
    mesh=board(n,[(i,1,(i+1)%n,0) for i in range(n)])
    data=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(n)]
    return SearchProblem(data,mesh)


def permutation_ring(permutations):
    n=len(permutations);positions=(2,6,10)
    data=[{'id':i,'segments':[[[0,positions[j]],[1,positions[2-p[j]]]] for j in range(3)]}
          for i,p in enumerate(permutations)]
    return SearchProblem(data,board(n,[(i,1,(i+1)%n,0) for i in range(n)]))


class SeparatorTests(unittest.TestCase):
    def test_complete_small_regions_equal_independent_gold_and_edge_enumeration(self):
        tetra,mesh=tetrahedron_tiles()
        fixtures=[SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                  SearchProblem(tetra,mesh),ring(4)]
        for problem in fixtures:
            for target in ('gold','edges'):
                with self.subTest(n=problem.n,target=target):
                    solutions=brute(problem,target);domains=domains_of(problem)
                    helper=SeparatorPruner(problem,target)
                    helper.prune_region(domains,range(problem.n))
                    self.assertEqual(helper.last_region['completions'],len(solutions))
                    if solutions:
                        self.assertEqual(domains,[bits(row[c] for row in solutions) for c in range(problem.n)])
                    else:
                        self.assertIn(0,domains)

    def test_eight_cell_open_inventory_dp_merges_many_histories_exactly(self):
        problem=ring(9);domains=domains_of(problem);original=domains.copy()
        helper=SeparatorPruner(problem)
        self.assertFalse(helper.prune_region(domains,range(8)))
        self.assertEqual(domains,original)
        self.assertEqual(helper.last_region['completions'],factorial(9))
        self.assertEqual(helper.last_region['summary_count'],9)
        self.assertLess(helper.last_region['states'],1000)
        self.assertGreater(helper.last_region['merges'],1000)
        self.assertFalse(helper.last_region['outside_pairing_assumptions'])
        result=next(iter(helper.cache.values()))
        self.assertTrue(all(used.bit_count()==8 and cycles==0 and len(pairing)==1
                            for used,masks,pairing,cycles in result['summaries']))
        # This is an open inventory: nine tiles compete for eight local cells.
        self.assertEqual(len({used for used,_,_,_ in result['summaries']}),9)
        helper.prune_region(original.copy(),range(8))
        self.assertEqual(helper.stats['separator_cache_hits'],1)

    def test_pairing_is_part_of_summary_key_not_only_boundary_masks(self):
        problem=permutation_ring([(1,0,2),(0,2,1),(0,1,2)])
        domains=[bits([0,3]),bits([0,3]),bits([6])]
        helper=SeparatorPruner(problem,'edges');helper.prune_region(domains,(0,1))
        summaries=next(iter(helper.cache.values()))['summaries']
        self.assertEqual(len(summaries),2)
        self.assertEqual(summaries[0][0],summaries[1][0])
        self.assertEqual(summaries[0][1],summaries[1][1])
        self.assertNotEqual(summaries[0][2],summaries[1][2])
        self.assertEqual(helper.last_region['completions'],2)

    def test_cycles_are_distinguished_and_improper_cycles_rejected_only_for_gold(self):
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        helper=SeparatorPruner(problem,'edges');domains=domains_of(problem)
        helper.prune_region(domains,range(4))
        summaries=next(iter(helper.cache.values()))['summaries']
        self.assertEqual({summary[3] for summary in summaries},{1,3})
        self.assertEqual(helper.last_region['completions'],24)
        gold=SeparatorPruner(problem);gold.prune_region(domains_of(problem),range(4))
        self.assertEqual(gold.last_region['completions'],len(brute(problem,'gold')))
        self.assertGreater(gold.last_region['completions'],0)
        self.assertLess(gold.last_region['completions'],24)

    def test_boundary_outside_domain_and_tile_inventory_change_the_cache_key(self):
        problem=ring(4);helper=SeparatorPruner(problem)
        broad=domains_of(problem);helper.prune_region(broad.copy(),(0,1,2))
        for outside in (0,9):
            actual=broad.copy();actual[3]=bits([outside])
            expected=actual.copy()
            helper.prune_region(actual,(0,1,2));SeparatorPruner(problem).prune_region(expected,(0,1,2))
            self.assertEqual(actual,expected)
            self.assertTrue(all(not domain&bits([outside]) for domain in actual[:3]))
        self.assertEqual(helper.stats['separator_cache_hits'],0)
        changed=broad.copy();changed[0]=bits([0,3]);helper.prune_region(changed,(0,1,2))
        self.assertEqual(helper.stats['separator_cache_entries'],4)

    def test_outside_pairings_are_optimistic_but_internal_closed_loops_are_not(self):
        problem,reference=chain()
        altered=[{'id':t['id'],'segments':t['segments']} for t in problem.tiles]
        altered[2]['segments']=[[[1,9],[1,3]],[[0,4],[0,8]]]
        other=SearchProblem(altered,problem.board)
        self.assertTrue(problem.validate(reference)['valid']);self.assertFalse(other.validate(reference)['valid'])
        results=[]
        for p in (problem,other):
            domains=[bits(range(6))]*2+[bits([6]),bits([9])]
            h=SeparatorPruner(p);h.prune_region(domains,(0,1));results.append(domains)
            self.assertEqual(h.last_region['completions'],1)
        self.assertEqual(*results)
        h=SeparatorPruner(other);domains=list(map(lambda c:1<<c,reference))
        self.assertTrue(h.prune_region(domains,range(4)));self.assertIn(0,domains)

    def test_parallel_seams_and_boundary_empty_masks_are_all_enforced(self):
        problem=SearchProblem(simple_tiles(),double_face())
        helper=SeparatorPruner(problem,'edges');domains=domains_of(problem)
        helper.prune_region(domains,(0,1))
        self.assertEqual(helper.last_region['completions'],len(brute(problem,'edges')))
        problem,reference=chain();helper=SeparatorPruner(problem)
        domains=[bits(range(6))]*2+[bits([6]),bits([9])]
        helper.prune_region(domains,(0,1))
        self.assertEqual(domains[:2],[bits([0]),bits([3])])
        self.assertTrue(all(d&(1<<c) for d,c in zip(domains,reference)))

    def test_state_or_transition_budget_never_publishes_partial_support(self):
        for attribute in ('max_states','max_transitions'):
            problem=ring(6);helper=SeparatorPruner(problem)
            setattr(helper,attribute,2)
            domains=domains_of(problem);before=domains.copy()
            self.assertFalse(helper.prune_region(domains,range(5)))
            self.assertEqual(domains,before);self.assertFalse(helper.cache)
            self.assertEqual(helper.stats['separator_budget_skips'],1)
            setattr(helper,attribute,20000)
            helper.prune_region(domains,range(5))
            self.assertEqual(helper.last_region['completions'],factorial(6))

    def test_cancelled_partial_dp_publishes_no_result_and_retains_domains(self):
        problem=ring(6);helper=SeparatorPruner(problem)
        domains=domains_of(problem);before=domains.copy();calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==12:raise InterruptedError
        with self.assertRaises(InterruptedError):helper.prune_region(domains,range(5),check)
        self.assertEqual(domains,before);self.assertFalse(helper.cache)
        helper.prune_region(domains,range(5))
        self.assertEqual(helper.last_region['completions'],factorial(6))

    def test_default_discovery_includes_larger_open_inventory_regions(self):
        problem=ring(9);helper=SeparatorPruner(problem);domains=domains_of(problem)
        helper.prune(domains)
        self.assertGreaterEqual(len(helper.last_region['cells']),5)
        self.assertGreater(helper.stats['separator_checks'],0)
        self.assertGreater(helper.stats['separator_merges'],0)

    def test_random_domains_preserve_all_independent_tiny_solutions(self):
        rng=random.Random(58241)
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        for target in ('edges','gold'):
            solutions=brute(problem,target);helper=SeparatorPruner(problem,target)
            for trial in range(35):
                domains=[bits(c for c in row if rng.random()<.65) for row in problem.domains]
                if not all(domains):continue
                expected=[s for s in solutions if all(d&(1<<c) for d,c in zip(domains,s))]
                helper.prune_region(domains,range(4))
                for solution in expected:self.assertTrue(all(d&(1<<c) for d,c in zip(domains,solution)))

    def test_bounded_large_planted_regions_preserve_known_gold(self):
        rng=random.Random(4118)
        for size in (1,2,4):
            data,mesh,reference=planted_gold(218,size);problem=SearchProblem(data,mesh)
            domains=[bits([c,rng.randrange(3*problem.n)]) for c in reference]
            helper=SeparatorPruner(problem);helper.prune(domains)
            self.assertTrue(all(d&(1<<c) for d,c in zip(domains,reference)))


if __name__=='__main__':unittest.main()
