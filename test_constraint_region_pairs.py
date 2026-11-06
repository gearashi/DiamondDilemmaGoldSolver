"""Independent exhaustive checks of disjoint regional-table support filtering."""
import random
import unittest
from unittest.mock import patch
from constraint_regions import HierarchicalRegionPruner
from test_constraint_advanced import bits,domains_of
from test_constraint_dfs import brute,tetrahedron_tiles
from test_constraint_separator import ring,permutation_ring
from search_problem import SearchProblem


def fixture():
    return permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)]),[585,585,65,585]


def independent_fixpoint(helper,domains,left,right):
    while all(domains):
        before=domains.copy()
        helper.prune_region(domains,left);helper.prune_region(domains,right)
        if domains==before:return domains
    return domains


class RegionalPairTests(unittest.TestCase):
    def test_pair_filters_beyond_independent_fixpoint_and_preserves_all_solutions(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem)
        before=independent_fixpoint(helper,original.copy(),(0,1),(2,3))
        self.assertEqual(before,original)
        actual=before.copy();self.assertTrue(helper.prune_pair(actual,(0,1),(2,3)))
        expected=[s for s in brute(problem,'gold') if all(d&(1<<c) for d,c in zip(original,s))]
        self.assertEqual(len(expected),4)
        self.assertEqual(actual,[bits(s[c] for s in expected) for c in range(4)])
        self.assertEqual(sum(a.bit_count()-b.bit_count() for a,b in zip(before,actual)),6)
        self.assertEqual(helper.last_pair['summaries_removed'],6)
        self.assertNotIn('table',helper.last_region)

    def test_full_pair_union_equals_independent_gold_and_edge_enumeration(self):
        rng=random.Random(221003)
        data,mesh=tetrahedron_tiles()
        for problem in (fixture()[0],ring(4),SearchProblem(data,mesh)):
            for target in ('gold','edges'):
                solutions=brute(problem,target);helper=HierarchicalRegionPruner(problem,target)
                for _ in range(24):
                    original=[bits(c for c in row if rng.random()<.75) for row in problem.domains]
                    if not all(original):continue
                    expected=[s for s in solutions if all(d&(1<<c) for d,c in zip(original,s))]
                    actual=original.copy();helper.prune_pair(actual,(0,1),(2,3))
                    if expected:self.assertEqual(actual,[bits(s[c] for s in expected) for c in range(4)])
                    else:self.assertIn(0,actual)

    def test_edge_target_does_not_discard_multiple_loops(self):
        problem,original=fixture();actual=original.copy();helper=HierarchicalRegionPruner(problem,'edges')
        helper.prune_pair(actual,(0,1),(2,3))
        edges=[s for s in brute(problem,'edges') if all(d&(1<<c) for d,c in zip(original,s))]
        gold=[s for s in edges if problem.validate(s)['valid']]
        self.assertGreater(len(edges),len(gold))
        self.assertEqual(actual,original)
        self.assertTrue(all(all(d&(1<<c) for d,c in zip(actual,s)) for s in edges))

    def test_noncontiguous_disjoint_regions_and_parallel_seams(self):
        data,mesh=tetrahedron_tiles();problem=SearchProblem(data,mesh)
        for target in ('gold','edges'):
            expected=brute(problem,target);helper=HierarchicalRegionPruner(problem,target)
            actual=domains_of(problem);helper.prune_pair(actual,(0,2),(1,3))
            self.assertTrue(expected)
            self.assertEqual(actual,[bits(s[c] for s in expected) for c in range(4)])

    def test_reused_tables_bind_boundary_inventory_and_orientation_domains(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem)
        broad=domains_of(problem)
        cases=[broad,[bits([0,3]),*broad[1:]],[*broad[:4],bits([12]),bits([15])],
               [*broad[:5],bits([1])],broad]
        for original in cases:
            actual,expected=original.copy(),original.copy()
            helper.prune_pair(actual,(0,1),(2,3))
            fresh=HierarchicalRegionPruner(problem);fresh.prune_pair(expected,(0,1),(2,3))
            self.assertEqual(actual,expected)
        self.assertGreater(helper.stats['hierarchical_cache_hits'],0)

    def test_overlap_is_rejected_and_never_publishes(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem)
        actual=original.copy();self.assertFalse(helper.prune_pair(actual,(0,1),(1,2)))
        self.assertEqual(actual,original);self.assertFalse(helper.cache)

    def test_pair_limits_never_publish_partial_support_unions(self):
        problem,original=fixture()
        for limit in (0,1,20,44):
            helper=HierarchicalRegionPruner(problem);helper.max_pair_transitions=limit
            actual=original.copy();self.assertFalse(helper.prune_pair(actual,(0,1),(2,3)))
            self.assertEqual(actual,original);self.assertIsNone(helper.last_pair)
            self.assertEqual(helper.stats['hierarchical_pair_budget_skips'],1)
            helper.max_pair_transitions=20000
            self.assertTrue(helper.prune_pair(actual,(0,1),(2,3)))

    def test_cancelled_join_never_publishes_and_complete_deductions_are_sound(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem)
        helper._completed_table(original,(0,1),lambda:None)
        helper._completed_table(original,(2,3),lambda:None)
        solutions=[s for s in brute(problem,'gold') if all(d&(1<<c) for d,c in zip(original,s))]
        interrupted=0
        for limit in range(1,35):
            actual=original.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==limit:raise InterruptedError
            try:helper.prune_pair(actual,(0,1),(2,3),check)
            except InterruptedError:
                interrupted+=1
                if helper.last_pair is None:self.assertEqual(actual,original)
            self.assertTrue(all(all(d&(1<<c) for d,c in zip(actual,s)) for s in solutions))
        self.assertGreater(interrupted,10)

    def test_pair_callback_includes_all_domain_dependencies(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem);actual=original.copy();events=[]
        def narrowed(cell,value,deps):
            self.assertEqual(actual[cell],value);events.append((cell,value,deps))
        helper.on_narrow=narrowed;helper.prune_pair(actual,(0,1),(2,3))
        self.assertTrue(events);self.assertTrue(all(d==tuple(range(4)) for _,_,d in events))

    def test_default_prune_selects_pair_beyond_single_query_budget(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem)
        with patch.object(helper,'_regions',return_value=[(0,1),(2,3)]):
            actual=original.copy();self.assertTrue(helper.prune(actual))
        self.assertEqual(helper.stats['hierarchical_pair_checks'],1)
        self.assertEqual(actual,[65,520,65,520])
        # A larger region covering the whole board cannot hide eligible pairs.
        problem=ring(18);actual=[bits([3*c,3*((c+1)%18)]) for c in range(18)]
        helper=HierarchicalRegionPruner(problem)
        helper.max_queries=0
        with patch.object(helper,'_regions',return_value=[tuple(range(16)),tuple(range(9)),tuple(range(9,18))]):
            helper.prune(actual)
        self.assertEqual(helper.stats['hierarchical_pair_checks'],1)
        self.assertTrue(all(d&(1<<(3*c)) for c,d in enumerate(actual)))

    def test_two_sixteen_cell_tables_do_not_construct_a_thirty_two_cell_parent(self):
        problem=ring(32);helper=HierarchicalRegionPruner(problem)
        actual=[bits([3*c,3*((c+1)%32)]) for c in range(32)]
        with patch.object(helper,'_solve',wraps=helper._solve) as solve:
            helper.prune_pair(actual,range(16),range(16,32))
        self.assertIsNotNone(helper.last_pair)
        self.assertEqual([len(call.args[1][0]) for call in solve.call_args_list],[16,16])
        self.assertTrue(all(len(key[0])<=16 for key in helper.cache))
        for solution in ([3*c for c in range(32)],[3*((c+1)%32) for c in range(32)]):
            self.assertTrue(problem.validate(solution)['valid'])
            self.assertTrue(all(d&(1<<c) for d,c in zip(actual,solution)))

    def test_query_payloads_and_subtables_share_support_storage_cap(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem);helper.max_cached_support_slots=45
        for left,right in (((0,1),(2,3)),((1,2),(3,4)),((0,1,2),(3,4,5))):
            helper.prune_pair(domains_of(problem),left,right)
            slots=sum(len(k[0])*r['summary_count'] for k,r in helper.cache.items())
            slots+=sum(len(k[1])*len(t) for k,t in helper.subtables.items())
            self.assertLessEqual(slots,45)
            self.assertEqual(slots,helper.stats['hierarchical_cached_support_slots'])
            for key,result in helper.cache.items():
                union=[0]*len(key[0])
                for support,count in result['table'].values():
                    self.assertGreater(count,0);union=[a|b for a,b in zip(union,support)]
                self.assertEqual(tuple(union),result['supports'])

if __name__=='__main__':unittest.main()
