"""Reopened timing-only candidates: exact-work reuse, independently checked."""
import random
import unittest
from constraint_pruning import Pruner
from constraint_parity import ParityPruner
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import brute
from test_constraint_pruning import PersistentNeighborhoodTests, unique_tetrahedron

class DirtyStarTests(unittest.TestCase):
    def fixture(self):
        small=unique_tetrahedron()
        return small,PersistentNeighborhoodTests.padded(small)

    def test_random_widening_and_narrowing_matches_independent_star_enumeration(self):
        small,problem=self.fixture();rng=random.Random(53017)
        helper=Pruner(problem,regions=[],parity=False,dirty_neighborhood=True)
        valid=brute(small,'edges')
        for case in range(160):
            initial=[sum(1<<c for c in range(12) if rng.random()<(.8 if case%4 else 1)) for _ in range(4)]
            initial += [1<<(3*c) for c in range(4,32)]
            limit=(None,1,4,12)[case%4]
            expected=PersistentNeighborhoodTests.oracle(problem,initial,limit)
            for _ in range(2):
                actual=initial.copy();helper.prune_neighborhood(actual,max_domain_values=limit)
                self.assertEqual(actual,expected)
            for solution in valid:
                if all(initial[c]>>v&1 for c,v in enumerate(solution)):
                    self.assertTrue(all(actual[c]>>v&1 for c,v in enumerate(solution)))
        self.assertLessEqual(len(helper._dirty_supports),problem.n)

    def test_certificate_survives_global_cache_eviction_and_avoids_repeated_checks(self):
        _,problem=self.fixture();domain=[4095]*4+[1<<(3*c) for c in range(4,32)]
        helpers=[Pruner(problem,regions=[],parity=False,dirty_neighborhood=flag) for flag in (False,True)]
        values=[]
        for helper in helpers:
            helper.prune_neighborhood(domain.copy())
            helper._neighborhood_cache.clear()  # Same logical effect as unrelated key eviction.
            before=helper.stats['neighborhood_checks'];actual=domain.copy()
            helper.prune_neighborhood(actual);values.append((actual,helper.stats['neighborhood_checks']-before))
        self.assertEqual(values[0][0],values[1][0]);self.assertGreater(values[0][1],0)
        self.assertEqual(values[1][1],0);self.assertEqual(helpers[1].stats['neighborhood_signature_skips'],4)

    def test_ineligible_changes_remain_dirty_until_complete(self):
        _,problem=self.fixture();helper=Pruner(problem,regions=[],parity=False,dirty_neighborhood=True)
        wide=[4095]*4+[1<<(3*c) for c in range(4,32)]
        helper.prune_neighborhood(wide.copy());narrow=wide.copy();narrow[1]=1
        helper.prune_neighborhood(narrow.copy(),max_domain_values=1)
        actual=narrow.copy();helper.prune_neighborhood(actual)
        self.assertEqual(actual,PersistentNeighborhoodTests.oracle(problem,narrow))
        actual=wide.copy();helper.prune_neighborhood(actual)
        self.assertEqual(actual,PersistentNeighborhoodTests.oracle(problem,wide))

    def test_cancelled_diff_or_star_never_publishes_incomplete_proof(self):
        small,problem=self.fixture();wide=[4095]*4+[1<<(3*c) for c in range(4,32)]
        solutions=brute(small,'edges')
        for budget in range(1,18):
            helper=Pruner(problem,regions=[],parity=False,dirty_neighborhood=True)
            helper.prune_neighborhood(wide.copy(),max_domain_values=1)
            count=[0]
            def check():
                count[0]+=1
                if count[0]==budget:raise InterruptedError('bounded test')
            helper.check=check;actual=wide.copy()
            try:helper.prune_neighborhood(actual)
            except InterruptedError:pass
            self.assertTrue(all(all(actual[c]>>v&1 for c,v in enumerate(solution)) for solution in solutions))
            helper.check=lambda:None;expected=PersistentNeighborhoodTests.oracle(problem,actual)
            helper.prune_neighborhood(actual);self.assertEqual(actual,expected)
            changed=wide.copy();changed[0]=1;count[0]=0;helper.check=check
            try:helper.prune_neighborhood(changed)
            except InterruptedError:pass
            helper.check=lambda:None;actual=wide.copy();helper.prune_neighborhood(actual)
            self.assertEqual(actual,PersistentNeighborhoodTests.oracle(problem,wide))

    def test_immutable_tables_never_share_local_certificates(self):
        _,problem=self.fixture();helper=Pruner(problem,regions=[],parity=False,dirty_neighborhood=True)
        helper.prune_neighborhood([4095]*4+[1<<(3*c) for c in range(4,32)])
        warm=Pruner(problem,regions=[],parity=False,tables=helper.tables,dirty_neighborhood=True)
        self.assertIs(warm._stars,helper._stars);self.assertIsNone(warm._dirty_previous)
        self.assertEqual(warm._dirty_supports,[])

class ParityRootReuseTests(unittest.TestCase):
    def test_all_sizes_multiple_domain_widths_preserve_gold_and_exact_filtering(self):
        for size in (1,2,4):
            for seed in (54018,54019):
                data,board,reference=planted_gold(seed,size);problem=SearchProblem(data,board);rng=random.Random(seed)
                helpers=[ParityPruner(problem,root_reuse=flag) for flag in (False,True)]
                for width in (0,1,3,6,12):
                    domains=[sum(1<<c for c in ({reference[cell]}|set(rng.sample(row,min(width,len(row)))))) for cell,row in enumerate(problem.domains)]
                    actual=[domains.copy(),domains.copy()]
                    results=[h.prune(d) for h,d in zip(helpers,actual)]
                    self.assertEqual(results[0],results[1]);self.assertEqual(actual[0],actual[1])
                    self.assertTrue(all(actual[1][cell]>>c&1 for cell,c in enumerate(reference)))
                self.assertGreater(helpers[1].stats['parity_root_queries_avoided'],0)
                self.assertLess(helpers[1].stats['parity_root_queries'],helpers[0].stats['parity_root_queries'])

    def test_cancellation_during_materialization_changes_no_domains(self):
        data,board,reference=planted_gold(54018,4);problem=SearchProblem(data,board)
        helper=ParityPruner(problem,root_reuse=True);domains=[1<<c for c in reference];before=domains.copy();calls=[0]
        def check():
            calls[0]+=1
            if calls[0]==len(problem.edges)+2:raise InterruptedError('during roots')
        with self.assertRaises(InterruptedError):helper.prune(domains,check)
        self.assertEqual(domains,before);self.assertEqual(helper.stats['parity_root_queries'],0)
        self.assertFalse(helper.prune(domains));self.assertGreater(helper.stats['parity_root_queries_avoided'],0)

    def test_pruner_and_reused_tables_forward_both_controls(self):
        data,board,reference=planted_gold(54018,1);problem=SearchProblem(data,board)
        first=Pruner(problem,dirty_neighborhood=True,parity_root_reuse=True)
        warm=Pruner(problem,tables=first.tables,dirty_neighborhood=True,parity_root_reuse=True)
        self.assertTrue(warm.parity.root_reuse);self.assertTrue(warm.dirty_neighborhood)
        warm.prune([1<<c for c in reference]);self.assertGreater(warm.stats['parity_root_queries_avoided'],0)

if __name__=='__main__':unittest.main()
