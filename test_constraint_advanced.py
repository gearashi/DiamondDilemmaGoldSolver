"""Independent finite checks for strand, correlated-cut and regional filters."""
from itertools import product, permutations
import random
import unittest

from constraint_advanced import AdvancedPruner, STAT_KEYS, cyclic_edge_blocks, _port_supports
from constraint_connectivity import ConnectivityPruner
from constraint_pruning import values
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import board, double_face, simple_tiles, multiple_loops, tetrahedron_tiles, brute


def bits(codes):
    return sum(1 << c for c in set(codes))


def domains_of(problem):
    return list(map(bits, problem.domains))


def chain(extra=False):
    edges = [(0, 0, 1, 0), (1, 1, 2, 1), (2, 0, 3, 0)]
    if extra:
        edges.append((0, 1, 2, 2))
    tiles = [{'id': 0, 'segments': [[[0, 2], [0, 10]]]},
             {'id': 1, 'segments': [[[0, 10], [1, 3]], [[0, 2], [1, 9]]]},
             {'id': 2, 'segments': [[[1, 9], [0, 4]], [[1, 3], [0, 8]]]},
             {'id': 3, 'segments': [[[0, 8], [0, 4]]]}]
    return SearchProblem(tiles, board(4, edges)), [0, 3, 6, 9]


class StrandTests(unittest.TestCase):
    def test_one_pass_port_supports_match_independent_endpoint_membership(self):
        rng = random.Random(812923)
        # Include an empty orientation collection, empty segment lists, repeated
        # pairs/endpoints and the full 480-code public-puzzle capacity.
        for count in (0, 1, 2, 6, 30, 480):
            for case in range(10):
                segments = tuple(tuple((rng.randrange(33), rng.randrange(33))
                                       for _ in range(rng.randrange(5)))
                                 for _ in range(count))
                if count and case % 2:
                    segments = (((0, 0), (0, 32), (0, 32)), *segments[1:])
                expected = tuple(sum(1 << code for code, pairs in enumerate(segments)
                                     if port in {v for pair in pairs for v in pair})
                                 for port in range(33))
                actual = _port_supports(segments)
                self.assertEqual(actual, expected)
                self.assertIsInstance(actual, tuple)

    def test_native_forest_cache_is_lazy_exact_and_interruption_safe(self):
        from unittest.mock import Mock
        problem = SearchProblem(simple_tiles(), double_face())
        helper = AdvancedPruner(problem, options={'native': 'python'})
        reference = AdvancedPruner(problem, options={'native': 'python'})
        graph = Mock(); workspace = graph.prepare_local_forests.return_value
        helper.native_graph = graph
        def calculate(domain, check):
            result = reference._local_forest(domain, lambda: None)
            check()
            return result
        workspace.calculate.side_effect = calculate
        domain = (1 << (3*problem.n))-1
        def cancel():
            raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper._local_forest(domain, cancel)
        self.assertFalse(helper._local_ports)
        result = helper._local_forest(domain, lambda: None)
        self.assertEqual(result, reference._local_forest(domain, lambda: None))
        graph.prepare_local_forests.assert_called_once()
        self.assertEqual(workspace.calculate.call_count, 2)
        self.assertIs(helper._local_forest(domain, lambda: None), result)
        self.assertEqual(workspace.calculate.call_count, 2)
        # Widening and siblings use exact keys rather than inherited hints.
        for other in (1, 8, domain, 3):
            self.assertEqual(helper._local_forest(other, lambda: None),
                             reference._local_forest(other, lambda: None))

    def test_optional_disconnected_port_components_are_not_mandatory(self):
        # Three disjoint optimistic loops are alternatives, not three required
        # loops: a tile chooses a single rotated U on any one of the seams.
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(2)]
        problem = SearchProblem(tiles, double_face())
        domains = domains_of(problem); before = domains.copy()
        helper = AdvancedPruner(problem)
        self.assertFalse(helper.prune_strands(domains))
        self.assertEqual(domains, before)
        self.assertTrue(brute(problem, 'gold'))

    def test_every_strand_of_one_code_must_share_an_eligible_component(self):
        problem = SearchProblem(multiple_loops(), double_face())
        domains = domains_of(problem)
        # Cell-level possible-seam connectivity sees a connected graph.
        self.assertFalse(ConnectivityPruner(problem).prune(domains))
        helper = AdvancedPruner(problem)
        self.assertTrue(helper.prune_strands(domains))
        self.assertIn(0, domains)
        self.assertEqual(helper.stats['strand_failures'], 1)
        self.assertFalse(brute(problem, 'gold'))
        self.assertTrue(brute(problem, 'edges'))

    def test_cell_component_choices_must_have_a_common_host(self):
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(4)]
        geometry = board(4, [(0, s, 1, s) for s in range(3)] + [(2, s, 3, s) for s in range(3)])
        problem = SearchProblem(tiles, geometry)
        domains = domains_of(problem)
        self.assertTrue(AdvancedPruner(problem).prune_strands(domains))
        self.assertIn(0, domains)

    def test_gold_only_and_exact_domain_cache_under_backtracking(self):
        problem = SearchProblem(simple_tiles(), double_face())
        helper = AdvancedPruner(problem)
        broad = domains_of(problem)
        for domains in ([bits([0]), bits([3])], [bits([0]), bits([4])], broad,
                        [bits([1]), bits([4])], broad):
            actual, expected = domains.copy(), domains.copy()
            helper.prune_strands(actual)
            AdvancedPruner(problem).prune_strands(expected)
            self.assertEqual(actual, expected)
        edge_helper = AdvancedPruner(problem, target='edges')
        invalid = [bits([0]), bits([4])]
        self.assertFalse(edge_helper.prune_strands(invalid))
        self.assertEqual(edge_helper.stats['strand_checks'], 0)

    def test_early_union_stop_matches_full_independent_port_graph(self):
        data, geometry, _ = planted_gold(28, 1)
        problem = SearchProblem(data, geometry); helper = AdvancedPruner(problem)
        rng = random.Random(8102)
        for _ in range(100):
            domain = rng.randrange(1, 1 << (3*problem.n))
            forest, used = helper._local_forest(domain, lambda: None)
            adjacency = [set() for _ in range(33)]
            for code in values(domain):
                for a, b in helper.segments[code]:
                    adjacency[a].add(b); adjacency[b].add(a)
            expected = []
            remaining = {p for p, neighbors in enumerate(adjacency) if neighbors}
            while remaining:
                root = min(remaining); reached = {root}; work = [root]
                while work:
                    for other in adjacency[work.pop()]-reached:
                        reached.add(other); work.append(other)
                expected.append(frozenset(reached)); remaining -= reached
            actual_adj = [set() for _ in range(33)]
            for a, b in forest:
                actual_adj[a].add(b); actual_adj[b].add(a)
            actual = []
            remaining = set(values(used))
            while remaining:
                root = min(remaining); reached = {root}; work = [root]
                while work:
                    for other in actual_adj[work.pop()]-reached:
                        reached.add(other); work.append(other)
                actual.append(frozenset(reached)); remaining -= reached
            self.assertEqual(set(actual), set(expected))

    def test_cancel_before_completed_connectivity_publishes_no_result(self):
        problem = SearchProblem(simple_tiles(), double_face())
        helper = AdvancedPruner(problem); domains = domains_of(problem); before = domains.copy()
        count = 0
        def check():
            nonlocal count
            count += 1
            if count == 4:
                raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper.prune_strands(domains, check)
        self.assertEqual(domains, before)
        self.assertEqual(helper._strand_cache, {})
        helper.prune_strands(domains)
        self.assertTrue(all(domain & (1 << c) for domain, c in zip(domains, [0, 3])))


class BiconnectedTests(unittest.TestCase):
    def test_articulation_blocks_and_parallel_edges(self):
        # Two triangles share one optional articulation port; the connecting
        # tail edge cannot lie on any cycle. Parallel edges form a valid block.
        edges=[(0,1),(1,2),(2,0),(0,3),(3,4),(4,0),(4,5),(6,7),(6,7)]
        blocks=cyclic_edge_blocks(8,edges)
        self.assertEqual(len(set(blocks[:3])),1);self.assertGreaterEqual(blocks[0],0)
        self.assertEqual(len(set(blocks[3:6])),1);self.assertNotEqual(blocks[0],blocks[3])
        self.assertEqual(blocks[6],-1);self.assertEqual(blocks[7],blocks[8]);self.assertGreaterEqual(blocks[7],0)

    def test_optional_blocks_are_alternatives_not_required_components(self):
        tiles=[{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(2)]
        problem=SearchProblem(tiles,double_face());domains=domains_of(problem)
        before=domains.copy();helper=AdvancedPruner(problem)
        self.assertFalse(helper.prune_biconnected(domains));self.assertEqual(domains,before)

    def test_all_candidate_segments_and_their_seams_must_fit_one_block(self):
        problem=SearchProblem(multiple_loops(),double_face());domains=domains_of(problem)
        helper=AdvancedPruner(problem);self.assertTrue(helper.prune_biconnected(domains))
        self.assertIn(0,domains);self.assertEqual(helper.stats['biconnected_failures'],1)
        domains=domains_of(problem);before=domains.copy()
        self.assertFalse(AdvancedPruner(problem,'edges').prune_biconnected(domains));self.assertEqual(before,domains)

    def test_optional_port_absent_neighbor_and_interruption_cache_safety(self):
        problem=SearchProblem(simple_tiles(),double_face());helper=AdvancedPruner(problem)
        broad=domains_of(problem)
        for source in ([bits([0]),bits([4])],broad,[bits([0]),bits([3])],broad):
            actual,expected=source.copy(),source.copy()
            helper.prune_biconnected(actual);AdvancedPruner(problem).prune_biconnected(expected)
            self.assertEqual(actual,expected)
        helper=AdvancedPruner(problem);calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==4:raise InterruptedError
        with self.assertRaises(InterruptedError):helper.prune_biconnected(broad,check)
        self.assertFalse(helper._biconnected_cache);self.assertEqual(broad,domains_of(problem))

    def test_native_certificate_cache_discards_interrupted_results(self):
        from unittest.mock import Mock
        helper = AdvancedPruner(SearchProblem(simple_tiles(), double_face()),
                                options={'native': 'python'})
        graph = Mock(); workspace = graph.prepare_local_forests.return_value
        helper.native_graph = graph
        result = (((0, 1),), 3)
        def calculate(domain, check):
            check()
            return result
        workspace.certificate.side_effect = calculate
        def cancel():
            raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper._local_block_certificate(1, cancel)
        self.assertFalse(helper._local_certificates)
        self.assertIs(helper._local_block_certificate(1, lambda: None), result)
        self.assertIs(helper._local_block_certificate(1, lambda: None), result)
        self.assertEqual(workspace.certificate.call_count, 2)
        graph.prepare_local_forests.assert_called_once()

    def test_sparse_local_certificates_equal_full_port_union_filtering(self):
        rng=random.Random(81817)
        for size in (1,2):
            data,mesh,reference=planted_gold(54,size);problem=SearchProblem(data,mesh)
            compact=AdvancedPruner(problem);full=AdvancedPruner(problem)
            if full.native_graph is not None:full.native_graph.supports_biconnected=False
            def uncompressed(domain,check):
                pairs=tuple(sorted({tuple(sorted(pair)) for code in values(domain) for pair in full.segments[code]}))
                ports=sum(1<<port for port in {p for pair in pairs for p in pair})
                return pairs,ports
            full._local_block_certificate=uncompressed
            for trial in range(20):
                original=[bits([code,*(c for c in range(3*problem.n) if rng.random()<(.12 if trial%2 else .8))]) for code in reference]
                left,right=original.copy(),original.copy()
                compact.prune_biconnected(left);full.prune_biconnected(right)
                self.assertEqual(left,right)
                self.assertTrue(all(d&(1<<c) for d,c in zip(left,reference)))

    def test_block_algorithm_agrees_with_independent_simple_cycle_enumeration(self):
        rng=random.Random(615)
        for _ in range(18):
            n=6;edges=[(a,b) for a in range(n) for b in range(a+1,n) if rng.random()<.4]
            graph=[[] for _ in range(n)]
            for eid,(a,b) in enumerate(edges):graph[a].append((b,eid));graph[b].append((a,eid))
            groups=[{i} for i in range(len(edges))];cyclic=set()
            def visit(start,node,vertices,path):
                for other,eid in graph[node]:
                    if other==start and len(vertices)>=3:
                        cycle=set(path+[eid]);cyclic.update(cycle)
                        merged=set().union(*(group for group in groups if group&cycle))
                        for group in groups:
                            if group&merged:group.update(merged)
                    elif other>start and other not in vertices:
                        visit(start,other,vertices|{other},path+[eid])
            for start in range(n):visit(start,start,{start},[])
            actual=cyclic_edge_blocks(n,edges)
            for a in range(len(edges)):
                self.assertEqual(actual[a]>=0,a in cyclic)
                for b in range(len(edges)):
                    if a in cyclic and b in cyclic:self.assertEqual(actual[a]==actual[b],b in groups[a])

    def test_finite_solutions_and_larger_gold_certificate_survive(self):
        tetra,mesh=tetrahedron_tiles();rng=random.Random(9182)
        problem=SearchProblem(tetra,mesh);solutions=brute(problem,'gold')
        helper=AdvancedPruner(problem)
        for _ in range(80):
            domains=[rng.randrange(1,1<<12) for _ in range(4)]
            expected=[s for s in solutions if all(d&(1<<c) for d,c in zip(domains,s))]
            helper.prune_biconnected(domains)
            for solution in expected:self.assertTrue(all(d&(1<<c) for d,c in zip(domains,solution)))
        data,geometry,reference=planted_gold(127,4);problem=SearchProblem(data,geometry)
        domains=[bits([c,rng.randrange(480),rng.randrange(480)]) for c in reference]
        AdvancedPruner(problem).prune_biconnected(domains)
        self.assertTrue(all(d&(1<<c) for d,c in zip(domains,reference)))


class CutTests(unittest.TestCase):
    def test_correlated_two_exit_choice_adds_to_independent_bridge_filter(self):
        problem, reference = chain(extra=True)
        self.assertTrue(problem.validate(reference)['valid'])
        domains = [(1 << 12)-1]*4
        ConnectivityPruner(problem).prune(domains)
        # Code 9 uses only cell 2's essential seam to cell 3. Both exits of
        # region {0,1} would be empty together, although neither is a bridge.
        self.assertTrue(domains[2] & (1 << 9))
        helper = AdvancedPruner(problem)
        self.assertTrue(helper.prune_cuts(domains))
        self.assertFalse(domains[2] & (1 << 9))
        self.assertTrue(all(domain & (1 << c) for domain, c in zip(domains, reference)))
        self.assertGreater(helper.stats['cut_prunes'], 0)

    def test_parallel_seams_are_distinct_correlated_variables(self):
        problem = SearchProblem(simple_tiles(), double_face())
        helper = AdvancedPruner(problem)
        # Direct table solves the legitimate singleton-cell boundary, including
        # three geometrically different seams to the same neighboring cell.
        original = domains_of(problem)
        support = dict(helper._cut_support(original, (0, 1, 2), lambda: None))
        solutions = brute(problem, 'gold')
        self.assertTrue(solutions)
        for codes in solutions:
            self.assertTrue(all(support[c] & (1 << code) for c, code in enumerate(codes)))
        for cell in range(2):
            for code in values(support[cell]):
                self.assertTrue(any(row[cell] == code for row in solutions))

    def test_budget_skip_and_edges_target_do_not_exclude_candidates(self):
        problem, _ = chain(extra=True)
        helper = AdvancedPruner(problem); helper.cut_vector_limit = 0
        domains = [(1 << 12)-1]*4; before = domains.copy()
        self.assertFalse(helper.prune_cuts(domains))
        self.assertEqual(before, domains)
        self.assertFalse(helper._cut_cache)
        helper = AdvancedPruner(problem, target='edges')
        self.assertFalse(helper.prune_cuts(domains))
        self.assertEqual(helper.stats['cut_checks'], 0)

    def test_cancelled_cut_does_not_cache_incomplete_support(self):
        problem, _ = chain(extra=True)
        helper = AdvancedPruner(problem); domains = [(1 << 12)-1]*4
        count = 0
        def check():
            nonlocal count
            count += 1
            if count == 5:
                raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper._cut_support(domains, (1, 3), check)
        self.assertFalse(helper._cut_cache)


class RegionalTests(unittest.TestCase):
    def fixture(self):
        problem, reference = chain()
        domains = [bits(range(6)), bits(range(6)), bits([6]), bits([9])]
        return problem, reference, domains

    def test_entire_local_problem_is_enumerated_and_reused(self):
        problem, reference, original = self.fixture()
        helper = AdvancedPruner(problem)
        domains = original.copy()
        self.assertTrue(helper.prune_region(domains, (0, 1)))
        self.assertEqual(domains[:2], [bits([0]), bits([3])])
        self.assertEqual(helper.stats['regional_assignments'], 1)
        self.assertEqual(helper.last_region['assignments'], 1)
        self.assertTrue(helper.last_region['complete'])
        self.assertFalse(helper.last_region['outside_pairing_assumptions'])
        revisit = original.copy()
        self.assertTrue(helper.prune_region(revisit, (0, 1)))
        self.assertEqual(revisit, domains)
        self.assertEqual(helper.stats['regional_cache_hits'], 1)
        self.assertEqual(helper.stats['regional_cache_entries'], 1)
        self.assertEqual(helper.stats['regional_assignments'], 1)
        self.assertTrue(problem.validate(reference)['valid'])

    def test_boundary_domain_changes_and_inventory_changes_miss_cache(self):
        problem, _, original = self.fixture(); helper = AdvancedPruner(problem)
        helper.prune_region(original.copy(), (0, 1))
        cases = [original[:2]+[bits([7]), bits([9])],
                 [bits([0, 1, 2, 6, 7, 8])]*2+original[2:],
                 [bits([0, 3]), original[1], original[2], original[3]]]
        for domains in cases:
            actual, expected = domains.copy(), domains.copy()
            helper.prune_region(actual, (0, 1))
            AdvancedPruner(problem).prune_region(expected, (0, 1))
            self.assertEqual(actual, expected)
        self.assertEqual(helper.stats['regional_cache_hits'], 0)
        self.assertEqual(helper.stats['regional_cache_entries'], 4)

    def test_remote_domains_do_not_affect_optimistic_local_cache(self):
        problem, _, original = self.fixture(); helper = AdvancedPruner(problem)
        first = original.copy(); helper.prune_region(first, (0, 1))
        # Cell 3 is not adjacent to region {0,1}; its domain cannot influence
        # this deliberately optimistic local problem, even if globally invalid.
        changed_remote = original[:3]+[bits([10])]
        helper.prune_region(changed_remote, (0, 1))
        self.assertEqual(first[:2], changed_remote[:2])
        self.assertEqual(helper.stats['regional_cache_hits'], 1)

    def test_unknown_inventory_is_not_assumed_closed(self):
        problem, _, original = self.fixture(); helper = AdvancedPruner(problem)
        original[0] |= bits([6])
        before = original.copy()
        self.assertFalse(helper.prune_region(original, (0, 1)))
        self.assertEqual(original, before)
        self.assertEqual(helper.stats['regional_checks'], 0)

    def test_budget_after_a_completion_does_not_publish_partial_support(self):
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(2)]
        problem = SearchProblem(tiles, double_face()); helper = AdvancedPruner(problem)
        helper.region_node_limit = 3  # First leaf fits; next prefix exceeds bound.
        domains = domains_of(problem); before = domains.copy()
        self.assertFalse(helper.prune_region(domains, (0, 1)))
        self.assertEqual(domains, before)
        self.assertEqual(helper.stats['regional_budget_skips'], 1)
        self.assertEqual(helper.stats['regional_assignments'], 0)
        self.assertFalse(helper._regional_cache)
        helper.region_node_limit = 4096
        helper.prune_region(domains, (0, 1))
        self.assertEqual(helper.last_region['assignments'], 6)

    def test_cancel_after_a_completion_publishes_nothing(self):
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(2)]
        problem = SearchProblem(tiles, double_face()); helper = AdvancedPruner(problem)
        original = domains_of(problem); before = original.copy()
        count = 0
        def check():
            nonlocal count
            count += 1
            if count == 4:
                raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper.prune_region(original, (0, 1), check)
        self.assertEqual(original, before)
        self.assertFalse(helper._regional_cache)
        self.assertEqual(helper.stats['regional_assignments'], 0)

    def test_outside_pairings_are_not_an_assumption(self):
        problem = SearchProblem(multiple_loops(), double_face())
        domains = domains_of(problem)
        helper = AdvancedPruner(problem)
        helper.prune_region(domains, (0, 1))
        # Edge-perfect multiple loops remain locally admissible. This cache
        # performs no whole-board Gold certification and records that explicitly.
        self.assertTrue(helper.last_region['assignments'])
        self.assertFalse(helper.last_region['outside_pairing_assumptions'])
        self.assertTrue(all(domains))
        self.assertFalse(brute(problem, 'gold'))

    def test_changed_segment_pairings_are_deliberately_optimistic(self):
        first, reference = chain()
        changed = [{'id': tile['id'], 'segments': tile['segments']} for tile in first.tiles]
        changed[2]['segments'] = [[[1, 9], [1, 3]], [[0, 4], [0, 8]]]
        second = SearchProblem(changed, first.board)
        self.assertEqual(first.masks.tolist(), second.masks.tolist())
        self.assertTrue(first.validate(reference)['valid'])
        self.assertFalse(second.validate(reference)['valid'])
        original = [bits(range(6))]*2+[bits([6]), bits([9])]
        outputs = []
        for problem in (first, second):
            helper = AdvancedPruner(problem); domains = original.copy()
            helper.prune_region(domains, (0, 1)); outputs.append(domains)
            self.assertFalse(helper.last_region['outside_pairing_assumptions'])
        self.assertEqual(outputs[0], outputs[1])

    def test_four_cell_region_supports_match_complete_independent_enumeration(self):
        tiles, geometry = tetrahedron_tiles()
        problem = SearchProblem(tiles, geometry)
        solutions = brute(problem, 'edges')
        domains = domains_of(problem); helper = AdvancedPruner(problem, target='edges')
        helper.prune_region(domains, range(4))
        self.assertEqual(helper.last_region['assignments'], len(solutions))
        self.assertEqual(domains, [bits(row[c] for row in solutions) for c in range(4)])

    def test_parallel_internal_seams_all_checked_and_counts_match_bruteforce(self):
        problem = SearchProblem(simple_tiles(), double_face())
        domains = domains_of(problem); helper = AdvancedPruner(problem, target='edges')
        helper.prune_region(domains, (0, 1))
        solutions = brute(problem, 'edges')
        self.assertEqual(helper.last_region['assignments'], len(solutions))
        self.assertEqual(domains, [bits(row[c] for row in solutions) for c in range(2)])

    def test_default_discovery_actually_solves_closed_regions(self):
        problem, _, original = self.fixture()
        helper = AdvancedPruner(problem, options={'strand_connectivity': False, 'multi_seam_cuts': False})
        self.assertTrue(helper.prune(original))
        self.assertGreater(helper.stats['regional_checks'], 0)
        self.assertGreater(helper.stats['regional_assignments'], 0)
        self.assertGreater(helper.stats['regional_prunes'], 0)


class FiniteSoundnessTests(unittest.TestCase):
    def test_random_domain_subsets_preserve_every_independent_tiny_solution(self):
        tetra, mesh = tetrahedron_tiles()
        fixtures = [(simple_tiles(), double_face()), (multiple_loops(), double_face()), (tetra, mesh)]
        rng = random.Random(20261012)
        for tiles, geometry in fixtures:
            problem = SearchProblem(tiles, geometry)
            for target in ('edges', 'gold'):
                solutions = brute(problem, target)
                helper = AdvancedPruner(problem, target)
                for trial in range(80):
                    original = [rng.randrange(1, 1 << (3*problem.n)) for _ in range(problem.n)]
                    expected = [row for row in solutions if all(d & (1 << c) for d, c in zip(original, row))]
                    domains = original.copy()
                    for _ in range(4):
                        if not helper.prune(domains) or not all(domains):
                            break
                    for row in expected:
                        self.assertTrue(all(d & (1 << c) for d, c in zip(domains, row)), (target, trial, original, row))

    def test_planted_gold_survives_narrowed_full_size_domains(self):
        rng = random.Random(2431)
        for size in (1, 2, 4):
            data, geometry, reference = planted_gold(173, size)
            problem = SearchProblem(data, geometry)
            helper = AdvancedPruner(problem)
            for trial in range(3):
                domains = [bits([code, *(rng.randrange(3*problem.n) for _ in range(2))])
                           for code in reference]
                helper.prune(domains)
                self.assertTrue(all(d & (1 << c) for d, c in zip(domains, reference)), (size, trial))

    def test_enabled_options_stats_and_cancellation_after_deletion(self):
        problem, reference, domains = RegionalTests().fixture()
        helper = AdvancedPruner(problem, options={'strand_connectivity': False, 'multi_seam_cuts': False, 'regional_cache': False, 'biconnected': False, 'separator_regions': False, 'hierarchical_regions': False})
        before = domains.copy()
        self.assertFalse(helper.prune(domains))
        self.assertEqual(before, domains)
        self.assertEqual(set(helper.stats), set(STAT_KEYS))
        helper = AdvancedPruner(problem)
        original = domains.copy()
        # Interruption after a sound first deletion may retain that deletion.
        def check():
            if domains != original:
                raise InterruptedError
        with self.assertRaises(InterruptedError):
            helper.prune_region(domains, (0, 1), check)
        self.assertTrue(all(d & (1 << c) for d, c in zip(domains, reference)))
        helper.prune_region(domains, (0, 1))
        self.assertTrue(all(d & (1 << c) for d, c in zip(domains, reference)))


if __name__ == '__main__':
    unittest.main()
