"""Independent bounded checks of separator-tree summary composition."""
from math import factorial
import random
import unittest
from constraint_regions import HierarchicalRegionPruner
from constraint_separator import SeparatorPruner
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import simple_tiles,double_face,multiple_loops,tetrahedron_tiles,brute
from test_constraint_separator import ring,permutation_ring
from test_constraint_advanced import bits,domains_of,chain


class RegionalPartitionTests(unittest.TestCase):
    @staticmethod
    def reference(helper,cells,domains,check=lambda:None):
        # Deliberately use edge scans and explicit sets, independent of masks.
        from math import log2
        members=set(cells);cuts=set();minimum=max(1,len(cells)//3)
        frontier=lambda part:sum((a in part)!=(b in part) for a,_,b,_ in helper.problem.edges)
        for seed in cells:
            check();left={seed}
            while len(left)<=len(cells)-minimum:
                right=members-left
                if len(left)>=minimum:
                    a,b=tuple(sorted(left)),tuple(sorted(right));cuts.add(min((a,b),(b,a)))
                choices=(set().union(*(helper.neighbors[c] for c in left))&members)-left
                if not choices:choices=right
                if not choices:break
                cell=min(choices,key=lambda c:(frontier(left|{c}),domains[c].bit_count(),c))
                left.add(cell)
        def score(pair):
            left,right=map(set,pair)
            cut=sum((a in left and b in right) or (b in left and a in right) for a,_,b,_ in helper.problem.edges)
            return (not(helper._connected(left) and helper._connected(right)),max(frontier(left),frontier(right)),
                    cut,abs(len(left)-len(right)),max(sum(log2(domains[c].bit_count()) for c in part) for part in pair),pair)
        return min(cuts,key=score)

    def test_every_small_graph_subset_and_parallel_seam_matches_scan_order(self):
        from types import SimpleNamespace
        comparisons=0
        for n in range(2,5):
            pairs=[(a,b) for a in range(n) for b in range(a+1,n)]
            for graph in range(1<<len(pairs)):
                edges=[pair for i,pair in enumerate(pairs) if graph>>i&1]
                for extra in ([],[(c,c) for c in range(n)]+edges[:1]):
                    # Topology-only fixture also checks self-loop cancellation;
                    # public SearchProblem already rejects self-loop boards.
                    problem=SimpleNamespace(n=n,edges=tuple((a,0,b,0) for a,b in edges+extra),
                        masks=[(0,0,0)]*(3*n),tiles=[{'segments':[[[0,1],[1,1]]]}]*n)
                    helper=HierarchicalRegionPruner(problem)
                    for subset in range(1<<n):
                        cells=tuple(c for c in range(n) if subset>>c&1)
                        if len(cells)<2:continue
                        for domains in ([1]*n,[(1<<(n-c))-1 for c in range(n)]):
                            self.assertEqual(helper._partition(cells,domains,lambda:None),self.reference(helper,cells,domains))
                            comparisons+=1
        self.assertGreater(comparisons,2500)

    def test_finite_query_results_and_work_equal_scan_reference(self):
        from unittest.mock import patch
        rng=random.Random(140224)
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        cases=[]
        for _ in range(24):
            row=[bits(c for c in choices if rng.random()<.8) for choices in problem.domains]
            if all(row):cases.append((problem,row,tuple(range(4))))
        data,board,known=planted_gold(140225,4);large=SearchProblem(data,board)
        for face in (0,4,9):
            cells=tuple(range(16*face,16*(face+1)));row=[1<<c for c in known]
            for cell in cells:row[cell]|=bits(rng.sample(range(3*large.n),3))
            cases.append((large,row,cells))
        for target in ('gold','edges'):
            for problem,row,cells in cases:
                actual=HierarchicalRegionPruner(problem,target);reference=HierarchicalRegionPruner(problem,target)
                for original in (row,row):
                    a,b=original.copy(),original.copy();changed=actual.prune_region(a,cells)
                    with patch.object(reference,'_partition',lambda c,d,k:self.reference(reference,c,d,k)):
                        expected=reference.prune_region(b,cells)
                    self.assertEqual((changed,a,actual.last_region,actual.stats),(expected,b,reference.last_region,reference.stats))
                    self.assertEqual(actual.cache,reference.cache)
                    self.assertEqual(actual.subtables,reference.subtables)

    def test_partition_stop_and_budget_keep_exact_publication_boundaries(self):
        problem=ring(10);row=domains_of(problem)
        for limit in range(1,11):
            helper=HierarchicalRegionPruner(problem);calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==limit:raise InterruptedError
            with self.assertRaises(InterruptedError):helper._partition(tuple(range(10)),row,stop)
            self.assertFalse(helper.cache);self.assertFalse(helper.subtables)
            self.assertEqual(helper._partition(tuple(range(10)),row,lambda:None),self.reference(helper,tuple(range(10)),row))
        for attribute in ('max_states','max_transitions','max_created'):
            from unittest.mock import patch
            a=HierarchicalRegionPruner(problem);b=HierarchicalRegionPruner(problem)
            setattr(a,attribute,2);setattr(b,attribute,2);da=row.copy();db=row.copy()
            result=a.prune_region(da,range(10))
            with patch.object(b,'_partition',lambda c,d,k:self.reference(b,c,d,k)):
                expected=b.prune_region(db,range(10))
            self.assertEqual((result,da,a.stats,a.cache,a.subtables),(expected,db,b.stats,b.cache,b.subtables))


class HierarchicalTests(unittest.TestCase):
    def test_exact_small_counts_and_supports_match_independent_full_enumeration(self):
        data,mesh=tetrahedron_tiles()
        for problem in (SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                        SearchProblem(data,mesh),ring(4)):
            for target in ('gold','edges'):
                with self.subTest(n=problem.n,target=target):
                    expected=brute(problem,target);domains=domains_of(problem)
                    helper=HierarchicalRegionPruner(problem,target);helper.prune_region(domains,range(problem.n))
                    self.assertEqual(helper.last_region['completions'],len(expected))
                    if expected:self.assertEqual(domains,[bits(s[c] for s in expected) for c in range(problem.n)])
                    else:self.assertIn(0,domains)

    def test_twelve_cell_hierarchy_composes_factorial_histories(self):
        problem=ring(12);helper=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        before=domains.copy();self.assertFalse(helper.prune_region(domains,range(12)))
        result=helper.last_region
        self.assertEqual(domains,before);self.assertEqual(result['completions'],factorial(12))
        self.assertEqual(result['summary_count'],1);self.assertEqual(len(result['decomposition']),11)
        self.assertGreater(result['joins'],1000);self.assertGreater(result['merges'],1000)
        self.assertLess(result['states'],10000);self.assertLessEqual(result['peak_frontier_edges'],2)
        helper.prune_region(before.copy(),range(12));self.assertEqual(helper.stats['hierarchical_cache_hits'],1)
        signature=next(iter(helper.cache.values()))['summaries'][0]
        self.assertEqual(signature[0],(1<<12)-1);self.assertEqual(signature[1:],((),(),1))

    def test_larger_open_inventory_and_boundary_domain_exclusion(self):
        problem=ring(13);helper=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        helper.prune_region(domains,range(12));result=helper.last_region
        self.assertIsNotNone(result);self.assertEqual(result['completions'],factorial(13))
        self.assertEqual(result['summary_count'],13)
        self.assertTrue(all(s[0].bit_count()==12 and len(s[2])==1 and s[3]==0 for s in next(iter(helper.cache.values()))['summaries']))
        changed=domains_of(problem);changed[12]=bits([36]);helper.prune_region(changed,range(12))
        self.assertTrue(all(not d&bits([36]) for d in changed[:12]))
        self.assertEqual(helper.last_region['completions'],factorial(12))
        self.assertEqual(helper.stats['hierarchical_cache_hits'],0)

    def test_pairing_cycle_and_inventory_state_matches_sequential_frontier_oracle(self):
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        rng=random.Random(631107)
        for target in ('gold','edges'):
            for trial in range(22):
                domains=[bits(c for c in row if rng.random()<.8) for row in problem.domains]
                if not all(domains):continue
                cells=range(3 if trial%2 else 4)
                a,b=domains.copy(),domains.copy();tree=HierarchicalRegionPruner(problem,target);line=SeparatorPruner(problem,target)
                tree.prune_region(a,cells);line.prune_region(b,cells)
                self.assertEqual(tree.last_region['completions'],line.last_region['completions'])
                self.assertEqual(next(iter(tree.cache.values()))['summaries'],next(iter(line.cache.values()))['summaries'])
                if tree.last_region['completions']:self.assertEqual(a,b)
                else:self.assertIn(0,a)

    def test_all_gold_solutions_survive_random_narrowed_domains(self):
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        rng=random.Random(31581)
        for target in ('gold','edges'):
            solutions=brute(problem,target);helper=HierarchicalRegionPruner(problem,target)
            for _ in range(30):
                domains=[bits(c for c in row if rng.random()<.65) for row in problem.domains]
                if not all(domains):continue
                surviving=[s for s in solutions if all(d&(1<<c) for d,c in zip(domains,s))]
                helper.prune_region(domains,range(4))
                for solution in surviving:self.assertTrue(all(d&(1<<c) for d,c in zip(domains,solution)))

    def test_limits_and_mid_join_cancellation_never_publish_partial_tables(self):
        for attribute in ('max_states','max_transitions','max_created'):
            helper=HierarchicalRegionPruner(ring(10));setattr(helper,attribute,2)
            domains=domains_of(helper.problem);before=domains.copy()
            self.assertFalse(helper.prune_region(domains,range(10)))
            self.assertEqual(domains,before);self.assertFalse(helper.cache)
            self.assertEqual(helper.stats['hierarchical_budget_skips'],1)
        helper=HierarchicalRegionPruner(ring(10));domains=domains_of(helper.problem);before=domains.copy();calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==130:raise InterruptedError
        with self.assertRaises(InterruptedError):helper.prune_region(domains,range(10),check)
        self.assertFalse(helper.cache);self.assertEqual(domains,before)
        helper.prune_region(domains,range(10));self.assertEqual(helper.last_region['completions'],factorial(10))

    def test_local_relaxation_does_not_assume_outside_pairing(self):
        problem,reference=chain();data=[{'id':t['id'],'segments':t['segments']} for t in problem.tiles]
        data[2]['segments']=[[[1,9],[1,3]],[[0,4],[0,8]]];other=SearchProblem(data,problem.board)
        self.assertTrue(problem.validate(reference)['valid']);self.assertFalse(other.validate(reference)['valid'])
        outputs=[]
        for p in (problem,other):
            domains=[bits(range(6))]*2+[bits([6]),bits([9])]
            helper=HierarchicalRegionPruner(p);helper.prune_region(domains,(0,1));outputs.append(domains)
        self.assertEqual(outputs[0],outputs[1])
        domains=[1<<c for c in reference];helper=HierarchicalRegionPruner(other);helper.prune_region(domains,range(4))
        self.assertIn(0,domains)

    def test_query_key_includes_region_domains_and_boundary_inventory(self):
        problem=ring(5);helper=HierarchicalRegionPruner(problem);broad=domains_of(problem)
        for changed in (broad,[bits([0,3]),*broad[1:]],[*broad[:4],bits([12])],broad):
            actual,expected=changed.copy(),changed.copy()
            helper.prune_region(actual,(0,1,2,3));fresh=HierarchicalRegionPruner(problem);fresh.prune_region(expected,(0,1,2,3))
            self.assertEqual(actual,expected)
        self.assertEqual(helper.stats['hierarchical_cache_hits'],1)
        self.assertEqual(helper.stats['hierarchical_cache_entries'],3)

    def test_face_and_ring_discovery_select_genuinely_larger_regions(self):
        problem=ring(12);domains=[bits([3*c,3*((c+1)%12)]) for c in range(12)]
        helper=HierarchicalRegionPruner(problem);helper.prune(domains)
        self.assertGreater(len(helper.last_region['cells']),8)
        self.assertTrue(all(d&(1<<(3*c)) for c,d in enumerate(domains)))
        data,board,reference=planted_gold(381,4);problem=SearchProblem(data,board)
        domains=[bits([c]) for c in reference]
        domains[0]|=bits([reference[1]])
        helper=HierarchicalRegionPruner(problem);regions=helper._regions(domains,lambda:None)
        self.assertIn(tuple(range(16)),regions)
        helper.prune_region(domains,range(16));self.assertIsNotNone(helper.last_region)
        self.assertTrue(all(d&(1<<c) for d,c in zip(domains,reference)))
        self.assertEqual(helper.last_region['completions'],1)

    def test_planted_gold_face_survives_extra_candidates_and_hierarchical_joins(self):
        rng=random.Random(4417)
        data,board,reference=planted_gold(719,4);problem=SearchProblem(data,board)
        for face in (0,4,9):
            domains=[bits([c]) for c in reference]
            for cell in range(16*face,16*(face+1)):
                domains[cell]|=bits([rng.randrange(480),rng.randrange(480)])
            helper=HierarchicalRegionPruner(problem);helper.prune_region(domains,range(16*face,16*(face+1)))
            self.assertIsNotNone(helper.last_region)
            self.assertTrue(all(d&(1<<c) for d,c in zip(domains,reference)))
            self.assertGreater(helper.last_region['joins'],0)


class RegionalSubtableTests(unittest.TestCase):
    def test_overlapping_parents_reuse_children_with_canonical_support_positions(self):
        problem=ring(8);helper=HierarchicalRegionPruner(problem);original=domains_of(problem)
        for cells in ((0,1,2,3),(0,1,2,3,4),(0,1,2,3,4,5)):
            actual,expected=original.copy(),original.copy()
            helper.prune_region(actual,cells)
            cold=HierarchicalRegionPruner(problem);cold.subtable_cache_limit=0
            cold.prune_region(expected,cells)
            self.assertEqual(actual,expected)
            self.assertEqual(helper.last_region['completions'],cold.last_region['completions'])
            self.assertEqual(helper.cache[helper._key(original,cells)]['summaries'],
                             cold.cache[cold._key(original,cells)]['summaries'])
        self.assertGreater(helper.stats['hierarchical_subtable_hits'],0)
        self.assertEqual(helper.stats['hierarchical_cache_hits'],0)
        for key,table in helper.subtables.items():
            node=key[1]
            for signature,(support,count) in table.items():
                self.assertEqual(len(support),len(node));self.assertGreater(count,0)
                self.assertEqual(signature[0].bit_count(),len(node))
                self.assertTrue(all(value&~original[cell]==0 for cell,value in zip(node,support)))

    def test_reused_child_supports_are_remapped_when_parent_cell_offsets_change(self):
        problem=ring(8);helper=HierarchicalRegionPruner(problem)
        original=[bits([3*c,3*((c+1)%8)]) for c in range(8)]
        queries=((0,1,2,3),(0,1,2,3,4),(0,1,2,3,4,5),(1,2,3,4,5),(2,3,4,5,6))
        for cells in queries:
            actual,expected=original.copy(),original.copy()
            helper.prune_region(actual,cells)
            cold=HierarchicalRegionPruner(problem);cold.subtable_cache_limit=0
            cold.prune_region(expected,cells)
            self.assertEqual(actual,expected)
            self.assertEqual(helper.last_region['completions'],cold.last_region['completions'])
            self.assertEqual(helper.cache[helper._key(original,cells)]['supports'],
                             cold.cache[cold._key(original,cells)]['supports'])
        self.assertGreater(helper.stats['hierarchical_subtable_hits'],5)

    def test_nonadjacent_parent_inventory_is_a_dependency_and_irrelevant_tiles_are_omitted(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        node=(0,1);domains[0]=bits([0]);domains[1]=bits([3])
        key=lambda boundary:helper._subtable_key(node,domains,boundary,lambda:None)
        # Cell 3 is not adjacent to this child, but may lose every unused tile.
        self.assertNotEqual(key(((3,bits([0])),)),key(((3,bits([0,3])),)))
        # Either outside tile 4 or 5 always remains available to cell 3.
        self.assertEqual(key(()),key(((3,bits([0,12])),)))
        self.assertEqual(key(((3,bits([0,12])),)),key(((4,bits([3,15])),)))
        # Distinct orientations of the same nonadjacent tile are equivalent.
        self.assertEqual(key(((3,bits([0])),)),key(((3,bits([1])),)))
        # An adjacent outside cell must retain orientation correlations.
        self.assertNotEqual(key(((2,bits([3])),)),key(((2,bits([4])),)))
        gold=key(());helper.target='edges';self.assertNotEqual(gold,key(()))

    def test_changed_outside_restrictions_never_reuse_incompatible_subtables(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem);original=domains_of(problem)
        cases=[]
        for outside in (bits([0]),bits([15]),bits([0,3]),original[5]):
            row=original.copy();row[5]=outside;cases.append(row)
        cases.extend([original.copy(),[bits([0,3]),*original[1:]]])
        for row in cases:
            for cells in ((0,1,2,3),(0,1,2,3,4)):
                actual,expected=row.copy(),row.copy()
                helper.prune_region(actual,cells)
                cold=HierarchicalRegionPruner(problem);cold.subtable_cache_limit=0
                cold.prune_region(expected,cells)
                self.assertEqual(actual,expected)
                self.assertEqual(helper.last_region['completions'],cold.last_region['completions'])
        self.assertGreater(helper.stats['hierarchical_subtable_hits'],0)

    def test_cached_count_and_pairing_supports_match_sequential_oracle_under_changing_parents(self):
        problem=permutation_ring([(1,0,2),(0,2,1),(1,0,2),(0,2,1)])
        rng=random.Random(771304)
        for target in ('gold','edges'):
            helper=HierarchicalRegionPruner(problem,target)
            for _ in range(18):
                original=[bits(c for c in row if rng.random()<.85) for row in problem.domains]
                if not all(original):continue
                for cells in ((0,1),(0,1,2),(1,2,3),(0,1,2,3)):
                    actual,expected=original.copy(),original.copy()
                    helper.prune_region(actual,cells)
                    oracle=SeparatorPruner(problem,target);oracle.prune_region(expected,cells)
                    self.assertEqual(helper.last_region['completions'],oracle.last_region['completions'])
                    self.assertEqual(helper.cache[helper._key(original,cells)]['summaries'],
                                     oracle.cache[oracle._key(original,cells)]['summaries'])
                    if helper.last_region['completions']:self.assertEqual(actual,expected)
                    else:self.assertIn(0,actual)

    def test_combined_cache_and_support_storage_stay_bounded_with_lru_eviction(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem);original=domains_of(problem)
        helper.max_cached_summaries=25;helper.max_cached_support_slots=30
        helper.subtable_cache_limit=5;helper.cache_limit=2
        for cells in ((0,1,2),(1,2,3),(0,1,2,3),(2,3,4),(0,1,2,3,4)):
            domains=original.copy();helper.prune_region(domains,cells)
            states=sum(len(table) for table in helper.subtables.values())
            slots=sum(len(key[1])*len(table) for key,table in helper.subtables.items())
            query_states=sum(value['summary_count'] for value in helper.cache.values())
            self.assertLessEqual(states+query_states,25);self.assertLessEqual(slots,30)
            self.assertLessEqual(len(helper.subtables),5);self.assertLessEqual(len(helper.cache),2)
            self.assertEqual(helper.stats['hierarchical_cached_summaries'],states+query_states)
            self.assertEqual(helper.stats['hierarchical_subtable_states'],states)
            self.assertEqual(helper.stats['hierarchical_subtable_support_slots'],slots)
        self.assertGreater(helper.stats['hierarchical_subtable_evictions'],0)

    def test_completed_children_remain_valid_after_parent_budget_or_cancellation(self):
        problem=ring(6);original=domains_of(problem)
        for cancel in (False,True):
            helper=HierarchicalRegionPruner(problem);helper.max_transitions=35
            row=original.copy()
            def check():
                if cancel and len(helper.subtables)>=2:raise InterruptedError
            if cancel:
                with self.assertRaises(InterruptedError):helper.prune_region(row,range(6),check)
            else:self.assertFalse(helper.prune_region(row,range(6),check))
            self.assertEqual(row,original);self.assertFalse(helper.cache)
            self.assertTrue(helper.subtables)
            for key,table in list(helper.subtables.items()):
                node=key[1]
                oracle=SeparatorPruner(problem)
                expected=oracle._solve(original,(node,tuple(original[c] for c in node),()),lambda:None)
                self.assertEqual(tuple(sorted(table)),expected['summaries'])
                self.assertEqual(sum(count for support,count in table.values()),expected['completions'])
                union=tuple(0 for _ in node)
                for support,count in table.values():union=tuple(a|b for a,b in zip(union,support))
                self.assertEqual(union,expected['supports'])
            helper.max_transitions=200000;helper.prune_region(row,range(6))
            self.assertEqual(helper.last_region['completions'],factorial(6))
            self.assertGreater(helper.stats['hierarchical_subtable_hits'],0)

    def test_narrow_callback_reports_all_query_dependencies_after_domain_change(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        domains[5]=bits([15]);events=[]
        def narrowed(cell,value,dependencies):
            self.assertEqual(domains[cell],value)
            events.append((cell,value,dependencies))
        helper.on_narrow=narrowed;helper.prune_region(domains,range(5))
        self.assertTrue(events)
        self.assertTrue(all(dependencies==tuple(range(6)) for _,_,dependencies in events))
        original=domains_of(problem);original[5]=bits([15]);domains=original.copy()
        def interrupted(cell,value,dependencies):
            self.assertEqual(domains[cell],value);raise InterruptedError
        helper.on_narrow=interrupted
        with self.assertRaises(InterruptedError):helper.prune_region(domains,range(5))
        self.assertNotEqual(domains,original)
        self.assertTrue(all(new&~old==0 for old,new in zip(original,domains)))
        helper.on_narrow=None;helper.prune_region(domains,range(5))
        self.assertEqual(helper.last_region['completions'],factorial(5))


if __name__=='__main__':unittest.main()
