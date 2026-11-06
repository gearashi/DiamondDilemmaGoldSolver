"""Independent finite checks for R23 regional reasoning and storage."""
import json
import random
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch
from constraint_regions import HierarchicalRegionPruner
from constraint_region_network import ExactRegionDiagram,RegionNetwork,assignment_signature
from constraint_region_cuts import MultiPortCutPruner,EndpointSupplyPruner
from constraint_region_store import RegionalTableStore
from test_constraint_advanced import bits,domains_of,chain
from test_constraint_dfs import brute,tetrahedron_tiles
from test_constraint_region_pairs import fixture
from test_constraint_separator import ring,permutation_ring
from search_problem import SearchProblem
from search_fixtures import planted_gold


def survivors(solutions,domains):
    return [s for s in solutions if all(d&(1<<c) for d,c in zip(domains,s))]


def expected_supports(solutions,n):
    return [bits(s[c] for s in solutions) for c in range(n)]


class RegionNetworkTests(unittest.TestCase):
    def test_full_three_region_join_matches_independent_enumeration(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3)
        rng=random.Random(230021)
        for target in ('gold','edges'):
            solutions=brute(problem,target)
            for _ in range(20):
                original=[bits(c for c in row if rng.random()<.65) for row in problem.domains]
                if not all(original):continue
                expected=survivors(solutions,original)
                helper=HierarchicalRegionPruner(problem,target);actual=original.copy()
                helper.prune_network(actual,((0,1),(2,3),(4,5)))
                self.assertTrue(helper._network.last['complete'])
                if expected:self.assertEqual(actual,expected_supports(expected,6))
                else:self.assertIn(0,actual)

    def test_live_summary_fixpoint_prunes_beyond_repeated_pair_domain_filtering(self):
        problem=ring(8);regions=((0,1),(2,3),(4,5),(6,7))
        original=[262216,266240,33281,33281,2392136,33288,2101248,32769]
        baseline=original.copy();helper=HierarchicalRegionPruner(problem)
        for _ in range(8):
            before=baseline.copy()
            for left,right in combinations(regions,2):helper.prune_pair(baseline,left,right)
            if baseline==before:break
        self.assertEqual(baseline,original)
        helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
        helper._network.triples=False;helper._network.inventories=False;helper._network.outside_paths=False
        actual=baseline.copy();helper.prune_network(actual,regions)
        self.assertEqual(sum(a.bit_count()-b.bit_count() for a,b in zip(baseline,actual)),1)
        # An independent distinct-piece DFS is exhaustive on this one-strand
        # ring: every permitted permutation is exactly one Gold loop.
        solutions=[]
        def enumerate_rows(cell,used,codes):
            if cell==8:solutions.append(tuple(codes));return
            for code in range(0,24,3):
                if original[cell]&(1<<code) and code//3 not in used:
                    enumerate_rows(cell+1,used|{code//3},[*codes,code])
        enumerate_rows(0,set(),[])
        self.assertTrue(solutions)
        self.assertTrue(all(problem.validate(s)['valid'] for s in solutions))
        self.assertEqual(survivors(solutions,actual),solutions)
        self.assertEqual(helper._network.last['summaries_removed'],6)
        self.assertGreater(helper._network.stats['region_network_rounds'],1)

    def test_three_region_joint_topology_filters_beyond_pairwise_consistency(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3)
        original=[4680,37448,4672,33352,4680,33353];regions=((0,1),(2,3),(4,5))
        results=[]
        for triples in (False,True):
            helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
            helper._network.triples=triples;helper._network.inventories=False;helper._network.outside_paths=False
            actual=original.copy();helper.prune_network(actual,regions);results.append(actual)
        self.assertEqual(results[0],original)
        self.assertEqual(results[1],original[:-1]+[1])
        solutions=survivors(brute(problem,'gold'),original)
        self.assertTrue(solutions);self.assertEqual(survivors(solutions,results[1]),solutions)

    def test_global_inventory_stage_rejects_pairwise_supported_shortage(self):
        problem=ring(8);regions=((0,1),(2,3),(4,5),(6,7))
        original=[2359809,262209,2101249,2363457,262145,2097153,295424,2134592]
        results=[]
        for inventories in (False,True):
            helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
            helper._network.triples=False;helper._network.inventories=inventories;helper._network.outside_paths=False
            actual=original.copy();helper.prune_network(actual,regions);results.append(actual)
        self.assertTrue(all(results[0]));self.assertIn(0,results[1])
        # Independent exact distinct-piece assignment oracle.
        def possible(cell,used):
            if cell==8:return True
            return any(possible(cell+1,used|{code//3}) for code in range(0,24,3)
                if original[cell]&(1<<code) and code//3 not in used)
        self.assertFalse(possible(0,set()))

    def test_network_scopes_are_exact_and_tables_not_mutated(self):
        problem,original=fixture();helper=HierarchicalRegionPruner(problem)
        regions=((0,1),(2,3))
        for cells in regions:helper._completed_table(original,cells,lambda:None)
        before={k:dict(v['table']) for k,v in helper.cache.items()}
        actual=original.copy();helper.prune_network(actual,regions)
        self.assertEqual(before,{k:dict(v['table']) for k,v in helper.cache.items()})
        helper.prune_network(original.copy(),regions)
        self.assertEqual(helper._network.stats['region_network_cache_hits'],1)
        changed=original.copy();changed[0]=bits([0])
        a,b=changed.copy(),changed.copy();helper.prune_network(a,regions)
        HierarchicalRegionPruner(problem).prune_network(b,regions)
        self.assertEqual(a,b)

    def test_fixed_outside_strands_add_information_without_changing_local_table(self):
        problem,reference=chain()
        data=[{'id':t['id'],'segments':t['segments']} for t in problem.tiles]
        data[2]['segments']=[[[1,9],[1,3]],[[0,4],[0,8]]]
        other=SearchProblem(data,problem.board)
        original=[bits(range(6))]*2+[bits([6]),bits([9])]
        local=original.copy();helper=HierarchicalRegionPruner(other)
        helper.prune_region(local,(0,1));self.assertTrue(all(local))
        actual=original.copy();helper.prune_network(actual,((0,1),))
        self.assertIn(0,actual)
        self.assertGreater(helper._network.stats['region_outside_path_rejections'],0)
        valid=original.copy();HierarchicalRegionPruner(problem).prune_network(valid,((0,1),))
        self.assertTrue(all(d&(1<<c) for d,c in zip(valid,reference)))

    def test_budget_and_cancellation_never_apply_incomplete_stage(self):
        problem,original=fixture();solutions=survivors(brute(problem,'gold'),original)
        for limit in (0,1,4,12,24,48):
            helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
            helper._network.max_steps=limit;actual=original.copy()
            helper.prune_network(actual,((0,1),(2,3)))
            self.assertEqual(survivors(solutions,actual),solutions)
            if not helper._network.last['complete']:self.assertFalse(helper._network.cache)
        for limit in range(1,30):
            helper=HierarchicalRegionPruner(problem);actual=original.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==limit:raise InterruptedError
            try:helper.prune_network(actual,((0,1),(2,3)),check)
            except InterruptedError:pass
            self.assertEqual(survivors(solutions,actual),solutions)

    def test_whole_inventory_allocation_detects_joint_shortage(self):
        # Pairwise-disjoint inventories need not admit a simultaneous allocation.
        problem=ring(6);helper=HierarchicalRegionPruner(problem)
        domains=[bits([0,3,6,9])]*6
        helper._network=RegionNetwork(helper);helper._network.triples=False
        helper._network.outside_paths=False
        actual=domains.copy();helper.prune_network(actual,((0,1),(2,3),(4,5)))
        self.assertIn(0,actual)
        self.assertGreater(helper._network.stats['region_inventory_checks'],0)

    def test_component_composition_matches_complete_enumeration(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3)
        rng=random.Random(230032)
        for target in ('gold','edges'):
            solutions=brute(problem,target)
            for _ in range(12):
                original=[bits(c for c in row if rng.random()<.6) for row in problem.domains]
                if not all(original):continue
                expected=survivors(solutions,original);actual=original.copy()
                helper=HierarchicalRegionPruner(problem,target)
                helper.prune_components(actual,((0,1),(2,3),(4,5)))
                if expected:self.assertEqual(actual,expected_supports(expected,6))
                else:self.assertIn(0,actual)

    def test_overlap_uses_exact_shared_assignments_and_matches_complete_oracle(self):
        data,mesh=tetrahedron_tiles()
        fixtures=(fixture()[0],SearchProblem(data,mesh))
        rng=random.Random(230023)
        for problem in fixtures:
            for target in ('gold','edges'):
                solutions=brute(problem,target)
                for _ in range(24):
                    original=[bits(c for c in row if rng.random()<.7) for row in problem.domains]
                    if not all(original):continue
                    expected=survivors(solutions,original);actual=original.copy()
                    helper=HierarchicalRegionPruner(problem,target)
                    helper.prune_overlap(actual,(0,1,2),(1,2,3))
                    if expected:self.assertEqual(actual,expected_supports(expected,4))
                    else:self.assertIn(0,actual)

    def test_overlap_adds_support_filtering_beyond_independent_region_fixpoint(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        original=[584,585,577,521];baseline=original.copy();helper=HierarchicalRegionPruner(problem)
        for _ in range(8):
            previous=baseline.copy();helper.prune_region(baseline,(0,1,2));helper.prune_region(baseline,(1,2,3))
            if baseline==previous:break
        self.assertEqual(baseline,original)
        actual=baseline.copy();helper.prune_overlap(actual,(0,1,2),(1,2,3))
        self.assertEqual(actual,[72,584,513,521])
        solutions=survivors(brute(problem,'gold'),original)
        self.assertEqual(len(solutions),3);self.assertEqual(survivors(solutions,actual),solutions)

    def test_four_component_join_adds_pruning_beyond_triples_and_inventory(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*4);regions=((0,1),(2,3),(4,5),(6,7))
        original=[262144,294920,2359360,2392577,262217,266761,65,512]
        helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
        helper._network.outside_paths=False;baseline=original.copy();helper.prune_network(baseline,regions)
        self.assertEqual(baseline,[262144,32776,2097216,2129921,73,4096,65,512])
        actual=baseline.copy();helper.prune_components(actual,regions)
        self.assertEqual(sum(a.bit_count()-b.bit_count() for a,b in zip(baseline,actual)),1)
        from itertools import product
        from constraint_pruning import values
        solutions=[row for row in product(*(tuple(values(d)) for d in baseline))
                   if len({c//3 for c in row})==8 and problem.validate(row)['valid']]
        self.assertTrue(solutions);self.assertEqual(survivors(solutions,actual),solutions)

    def test_overlap_budget_and_diagram_limits_leave_inputs_unchanged(self):
        problem,original=fixture()
        for attribute,limit in (('max_steps',0),('max_rows',1)):
            helper=HierarchicalRegionPruner(problem);helper._network=RegionNetwork(helper)
            setattr(helper._network,attribute,limit);actual=original.copy()
            self.assertFalse(helper.prune_overlap(actual,(0,1,2),(1,2,3)))
            self.assertEqual(actual,original)

    def test_diagram_roundtrip_and_shared_suffix_compression(self):
        problem=ring(6);helper=HierarchicalRegionPruner(problem)
        diagram=ExactRegionDiagram(helper,range(6),domains_of(problem),lambda:None)
        expected=set(brute(problem,'gold'))
        self.assertEqual({codes for codes,_ in diagram.histories()},expected)
        self.assertEqual(diagram.rows,720)
        self.assertLess(len(diagram.nodes),100)
        self.assertEqual(len(diagram.terminals),1)


class RegionWidthStorageTests(unittest.TestCase):
    def test_width_admission_solves_more_than_sixteen_cells(self):
        problem=ring(40);helper=HierarchicalRegionPruner(problem).enable_advanced()
        original=[bits([3*c,3*((c+1)%40)]) for c in range(40)]
        actual=original.copy();helper.prune_region(actual,range(40))
        self.assertIsNotNone(helper.last_region)
        self.assertEqual(helper.last_region['completions'],2)
        self.assertEqual(actual,original)
        self.assertLessEqual(helper.last_region['peak_frontier_edges'],2)
        self.assertTrue(helper._plan_cache)
        # Reuse plans independently of tables, preserving exact result.
        helper.cache.clear();helper.subtables.clear();helper._subtable_index.clear()
        helper._query_cached_summaries=helper._query_support_slots=0
        helper._subtable_states=helper._subtable_support_slots=0
        helper.prune_region(original.copy(),range(40))
        self.assertGreater(helper.extended_stats.get('region_plan_hits',0),0)

    def test_width_budget_is_unknown_and_never_partial_exclusion(self):
        problem=ring(20);helper=HierarchicalRegionPruner(problem).enable_advanced()
        helper.max_frontier_edges=1
        original=[bits([3*c,3*((c+1)%20)]) for c in range(20)]
        actual=original.copy();self.assertFalse(helper.prune_region(actual,range(20)))
        self.assertEqual(actual,original);self.assertFalse(helper.cache)

    def test_conflict_activity_changes_region_ranking_only(self):
        problem=ring(24);helper=HierarchicalRegionPruner(problem).enable_advanced()
        domains=[bits([3*c,3*((c+1)%24)]) for c in range(24)]
        before=helper._regions(domains,lambda:None)
        helper.note_conflict((17,18,19))
        after=helper._regions(domains,lambda:None)
        self.assertEqual(set(before),set(after))
        self.assertGreaterEqual(sum(c in after[0] for c in (17,18,19)),sum(c in before[0] for c in (17,18,19)))

    def test_persistent_tables_roundtrip_identity_and_corruption(self):
        problem,original=fixture()
        with tempfile.TemporaryDirectory() as folder:
            helper=HierarchicalRegionPruner(problem).enable_advanced(folder)
            actual=original.copy();helper.prune_region(actual,(0,1))
            self.assertGreater(helper._disk_store.writes,0)
            other=HierarchicalRegionPruner(problem).enable_advanced(folder)
            with patch.object(other,'_solve',side_effect=AssertionError('Should reuse completed disk table')):
                expected=original.copy();other.prune_region(expected,(0,1))
            self.assertEqual(actual,expected);self.assertEqual(other._disk_store.hits,1)
            changed=original.copy();changed[0]=bits([0])
            self.assertIsNone(other._disk_store.load(other._key(changed,(0,1)),lambda:None))
            edge=HierarchicalRegionPruner(problem,'edges').enable_advanced(folder)
            self.assertIsNone(edge._disk_store.load(edge._key(original,(0,1)),lambda:None))
            path=next(Path(folder).glob('*.json'));record=json.loads(path.read_text())
            record['payload']['rows']=[];path.write_text(json.dumps(record))
            third=HierarchicalRegionPruner(problem).enable_advanced(folder)
            self.assertIsNone(third._disk_store.load(third._key(original,(0,1)),lambda:None))
            self.assertEqual(third._disk_store.rejections,1)

    def test_full_planted_board_survives_all_advanced_paths(self):
        data,board,solution=planted_gold(230051,4);problem=SearchProblem(data,board)
        original=[1<<c for c in solution]
        for cell in (0,5,16,31,47,68,93,122,159):
            original[cell]|=1<<solution[(cell+1)%160]
        helper=HierarchicalRegionPruner(problem).enable_advanced()
        helper.max_frontier_edges=64
        actual=original.copy();helper.prune_region(actual,range(160))
        self.assertIsNotNone(helper.last_region)
        self.assertTrue(all(d&(1<<c) for d,c in zip(actual,solution)))
        self.assertEqual(helper.last_region['completions'],1)


class RegionCutTests(unittest.TestCase):
    def test_multi_port_condition_preserves_every_cycle_of_small_graphs(self):
        # Every n-cycle is an independent known witness.  Adding optional
        # graph edges/vertices can only relax the necessary condition.
        rng=random.Random(230026)
        for n in range(3,10):
            for _ in range(15):
                graph={i:{(i-1)%n,(i+1)%n} for i in range(n)}
                for a,b in combinations(range(n),2):
                    if rng.random()<.25:graph[a].add(b);graph[b].add(a)
                for k in (0,1,2,3):
                    for removed in combinations(range(n),min(k,n)):
                        self.assertFalse(MultiPortCutPruner.violates_cut(graph,set(range(n)),removed))

    def test_two_vertex_separator_rejects_three_mandatory_branches(self):
        graph={0:{2,3,4},1:{2,3,4},2:{0,1},3:{0,1},4:{0,1}}
        self.assertTrue(MultiPortCutPruner.violates_cut(graph,{2,3,4},(0,1)))
        self.assertFalse(MultiPortCutPruner.violates_cut(graph,{2,3},(0,1)))
        # Optional-only third branch must not become a required component.
        self.assertFalse(MultiPortCutPruner.violates_cut(graph,{2},(0,1)))

    def test_real_pruners_preserve_independently_enumerated_gold_solutions(self):
        rng=random.Random(230052);data,mesh=tetrahedron_tiles()
        for problem in (fixture()[0],SearchProblem(data,mesh),ring(4)):
            solutions=brute(problem,'gold')
            for _ in range(30):
                original=[bits(c for c in row if rng.random()<.7) for row in problem.domains]
                if not all(original):continue
                expected=survivors(solutions,original)
                actual=original.copy();MultiPortCutPruner(problem).prune(actual)
                self.assertEqual(survivors(expected,actual),expected)
                for region in ((0,1),(0,2),(0,1,2)):
                    actual=original.copy();EndpointSupplyPruner(problem).prune_region(actual,region)
                    self.assertEqual(survivors(expected,actual),expected)

    def test_joint_endpoint_supply_counts_each_physical_piece_only_once(self):
        from test_constraint_dfs import board
        edges=[(0,0,1,0),(0,1,2,0),(0,2,3,0),(1,1,4,0),(1,2,5,0)]
        tiles=[{'id':0,'segments':[[[s,2],[s,10]] for s in range(3)]}]
        tiles.extend({'id':i,'segments':[[[0,2],[0,10]]]} for i in range(1,6))
        problem=SearchProblem(tiles,board(6,edges))
        domains=[bits([0,3])]*2+[bits([3*c]) for c in range(2,6)]
        helper=EndpointSupplyPruner(problem)
        self.assertTrue(helper.impossible(domains,(0,1)))
        actual=domains.copy();self.assertTrue(helper.prune_region(actual,(0,1)))
        self.assertIn(0,actual)
        # Both regional cells would need the sole high-capacity physical tile.
        # Exhaust every allowed distinct-piece assignment independently.
        from itertools import product
        from constraint_pruning import values
        self.assertFalse(any(problem.validate(row)['valid'] for row in product(*(tuple(values(d)) for d in domains))
                             if len({c//3 for c in row})==6))

    def test_planted_full_board_survives_cut_checks(self):
        data,board,solution=planted_gold(230026,4);problem=SearchProblem(data,board)
        domains=[1<<c for c in solution]
        helper=MultiPortCutPruner(problem);helper.max_cuts=24
        self.assertFalse(helper.prune(domains.copy()))
        supply=EndpointSupplyPruner(problem)
        for region in (range(16),range(48),range(1,160,2)):
            self.assertFalse(supply.prune_region(domains.copy(),region))


if __name__=='__main__':unittest.main()
