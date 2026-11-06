"""Independent finite checks for strand, correlated-cut and regional filters."""
from itertools import product, permutations
import random
import unittest

from constraint_advanced import AdvancedPruner, STAT_KEYS, cyclic_edge_blocks, _port_supports, small_cut_boundaries, _small_matching_supports, _complete_matching, _complete_matching_supports
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


class ConditionalCutMatchingTests(unittest.TestCase):
    @staticmethod
    def fixture(mixed=True):
        _, geometry = tetrahedron_tiles()
        tiles = []
        for index in range(4):
            p, q = (3, 9) if mixed and index >= 2 else (2, 10)
            tiles.append({'id': index, 'segments': [
                [[0, p], [1, p]], [[1, q], [2, p]], [[2, q], [0, q]]]})
        return SearchProblem(tiles, geometry)

    @staticmethod
    def oracle(rows):
        result = [0]*len(rows); found = False
        def visit(cell, used, selected):
            nonlocal found
            if cell == len(rows):
                found = True
                for index, tile in enumerate(selected):result[index] |= 1 << tile
                return
            for tile in range(max((row.bit_length() for row in rows), default=0)):
                if rows[cell] & (1 << tile) and tile not in used:
                    visit(cell+1, used | {tile}, selected+[tile])
        visit(0, set(), [])
        return tuple(result) if found else None

    def test_subset_supports_equal_exhaustive_injective_assignments(self):
        rng = random.Random(202610251)
        for n in range(7):
            for _ in range(24):
                rows = [rng.randrange(1 << min(n+1, 7)) for _ in range(n)]
                self.assertEqual(_small_matching_supports(rows), self.oracle(rows), rows)
        self.assertEqual(_small_matching_supports([1 << 159, (1 << 159) | (1 << 157)]),
                         (1 << 159, 1 << 157))
        self.assertEqual(_small_matching_supports([3, 3, 15]), (3, 3, 12))
        with self.assertRaises(ValueError):_small_matching_supports([255]*7)

    def test_conditional_collision_survives_seams_and_broad_hall(self):
        from constraint_pruning import Pruner
        from validator import reverse_mask
        problem = self.fixture(); domains = domains_of(problem)
        self.assertFalse(Pruner(problem, native='python').prune_hall(domains))
        self.assertEqual(domains, domains_of(problem))
        # Every ordinary directed seam has a different-piece support. All four
        # cells can use all four physical tiles before choosing seam patterns.
        for a, sa, b, sb in problem.edges:
            for left, ls, right, rs in ((a, sa, b, sb), (b, sb, a, sa)):
                for code in values(domains[left]):
                    self.assertTrue(any(other//3 != code//3 and
                        problem.masks[code, ls] == reverse_mask(int(problem.masks[other, rs]))
                        for other in values(domains[right])))
        # Each center orientation also has a positive-even, mask-compatible
        # assignment on its three-edge boundary when piece reuse is permitted.
        # Thus the older mask-only cut union would retain every orientation.
        for code in values(domains[0]):
            for a, sa, b, sb in problem.edges[:3]:
                self.assertEqual(a, 0)
                self.assertTrue(any(problem.masks[code, sa] == reverse_mask(int(problem.masks[other, sb]))
                                    for other in values(domains[b])))
            self.assertEqual(sum(int(mask).bit_count() for mask in problem.masks[code]), 6)
        helper = AdvancedPruner(problem, options={'native': 'python'})
        support = dict(helper._cut_support(domains, (0, 1, 2), lambda: None))
        self.assertTrue(all(value == 0 for value in support.values()))
        self.assertGreater(helper.stats['cut_matching_failures'], 0)
        self.assertFalse(brute(problem, 'gold'))
        self.assertTrue(helper.prune_cuts(domains));self.assertIn(0, domains)

    def test_every_valid_gold_solution_survives_conditional_matching(self):
        problem = self.fixture(False); solutions = brute(problem, 'gold')
        self.assertTrue(solutions)
        rng = random.Random(202610252)
        helper = AdvancedPruner(problem, options={'native': 'python'})
        for _ in range(48):
            domains = [rng.randrange(1, 1 << 12) for _ in range(4)]
            expected = [row for row in solutions if all(domain & (1 << code) for domain, code in zip(domains, row))]
            helper.prune_cuts(domains)
            for row in expected:self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, row)))
        domains = domains_of(problem)
        self.assertFalse(AdvancedPruner(problem, target='edges').prune_cuts(domains))
        self.assertEqual(domains, domains_of(problem))

    def test_matching_interruption_never_publishes_partial_cut_support(self):
        from unittest.mock import patch
        problem = self.fixture(); domains = domains_of(problem); before = domains.copy()
        helper = AdvancedPruner(problem, options={'native': 'python'})
        def cancelled(rows, check):
            def stop():raise InterruptedError
            return _small_matching_supports(rows, stop)
        with patch('constraint_advanced._small_matching_supports', side_effect=cancelled):
            with self.assertRaises(InterruptedError):helper._cut_support(domains, (0, 1, 2), lambda: None)
        self.assertFalse(helper._cut_cache);self.assertEqual(domains, before)
        self.assertTrue(all(value == 0 for _, value in helper._cut_support(domains, (0, 1, 2), lambda: None)))

    def test_matching_cache_preserves_siblings_widening_and_outside_inventory(self):
        problem = self.fixture(False); helper = AdvancedPruner(problem, options={'native': 'python'})
        broad = domains_of(problem)
        sources = [broad, [bits([0, 3]), bits([3, 6]), bits([0, 6]), broad[3]],
                   [bits([0]), bits([0]), bits([6]), broad[3]], broad]
        for domains in sources:
            actual = helper._cut_support(domains, (0, 1), lambda: None)
            expected = AdvancedPruner(problem, options={'native': 'python'})._cut_support(domains, (0, 1), lambda: None)
            self.assertEqual(actual, expected)
        # Cell3 is outside these seams but its inventory now participates in
        # each global conditional matching, so it must invalidate the cut key.
        first = helper._cut_support(broad, (0, 1), lambda: None)
        changed = broad.copy(); changed[3] = bits([0])
        self.assertIsNot(helper._cut_support(changed, (0, 1), lambda: None), first)
        self.assertIs(helper._cut_support(broad, (0, 1), lambda: None), first)
        # Outside orientation changes preserving physical tiles remain a valid
        # exact cache hit: the inventory helper makes no outside seam claim.
        rotated = broad.copy();rotated[3] = bits([1, 4, 7, 10])
        self.assertIs(helper._cut_support(rotated, (0, 1), lambda: None), first)


class ConditionalGlobalInventoryTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # A fixed fixture selected by seed 21001; the two-seam conditional
        # inventories expose a Hall deficit invisible in their broad union.
        tiles = [
            {'id': 0, 'segments': [[[0, 2], [1, 10]], [[0, 10], [1, 2]]]},
            {'id': 1, 'segments': [[[2, 10], [2, 2]]]},
            {'id': 2, 'segments': [[[0, 10], [1, 10]], [[2, 10], [1, 2]], [[2, 2], [0, 2]]]},
            {'id': 3, 'segments': [[[0, 2], [0, 10]]]}]
        problem = SearchProblem(tiles, board(4, [(0, 0, 1, 0), (1, 1, 2, 0),
                                               (2, 1, 3, 0), (3, 1, 0, 1)]))
        return problem, [4026, 2035, 4093, 4008]

    @staticmethod
    def cut_oracle(problem, domains):
        from validator import reverse_mask
        return [codes for codes in product(*(values(domain) for domain in domains))
                if len({code//3 for code in codes}) == problem.n
                and all(problem.masks[codes[a], sa] == reverse_mask(int(problem.masks[codes[b], sb]))
                        for a, sa, b, sb in problem.edges[:2])
                and sum(int(problem.masks[codes[a], sa]).bit_count()
                        for a, sa, _, _ in problem.edges[:2]) in (2, 4, 6)]

    def test_complete_matching_matches_independent_enumeration_and_rechecks_hints(self):
        rng = random.Random(21002)
        for n in range(7):
            hint = None
            for _ in range(30):
                rows = tuple(rng.randrange(1 << n) for _ in range(n))
                expected = ConditionalCutMatchingTests.oracle(rows)
                actual = _complete_matching(rows, hint=hint)
                self.assertEqual(actual is None, expected is None, rows)
                if actual is not None:
                    self.assertEqual(len(actual), n);self.assertEqual(len(set(actual)), n)
                    self.assertTrue(all(row & (1 << tile) for row, tile in zip(rows, actual)))
                    hint = actual
        self.assertEqual(_complete_matching([1 << 159, (1 << 157) | (1 << 159)]), (159, 157))
        for rows in ((1, 1), (0, 2), (3, 3)):
            result = _complete_matching(rows, hint=(1, 0))
            self.assertEqual(result is None, ConditionalCutMatchingTests.oracle(rows) is None)

    def test_global_conditioning_prunes_beyond_local_cut_and_unconditioned_hall(self):
        from constraint_pruning import Pruner
        from unittest.mock import patch
        problem, domains = self.fixture();original = domains.copy()
        self.assertFalse(Pruner(problem, native='python').prune_hall(domains))
        # Replacing only whole-board feasibility by True reproduces the old
        # local conditional filter. Its union is already a parent fixpoint.
        with patch('constraint_advanced._complete_matching', return_value=(0, 1, 2, 3)), \
             patch('constraint_advanced._matching_supports', side_effect=lambda rows, *args: rows):
            baseline = AdvancedPruner(problem, options={'native': 'python'})._cut_support(domains, (0, 1), lambda: None)
        self.assertEqual(baseline[:3], tuple(enumerate(domains[:3])))
        helper = AdvancedPruner(problem, options={'native': 'python'})
        supported = helper._cut_support(domains, (0, 1), lambda: None)
        self.assertEqual(supported[:3], ((0, 4026), (1, 1507), (2, 981)))
        self.assertEqual(sum((domains[c] & ~keep).bit_count() for c, keep in supported), 6)
        self.assertEqual(helper.stats['cut_global_matching_failures'], 1)
        completions = self.cut_oracle(problem, domains);self.assertTrue(completions)
        for codes in completions:
            self.assertTrue(all(keep & (1 << codes[c]) for c, keep in supported))
        self.assertEqual(domains, original)

    def test_cut_cache_tracks_outside_inventory_changes_and_backtracking(self):
        problem, original = self.fixture();helper = AdvancedPruner(problem, options={'native': 'python'})
        first = helper._cut_support(original, (0, 1), lambda: None)
        for outside in (4095, 1, 7, 4008, 4095, 4008):
            domains = original.copy();domains[3] = outside
            actual = helper._cut_support(domains, (0, 1), lambda: None)
            expected = AdvancedPruner(problem, options={'native': 'python'})._cut_support(domains, (0, 1), lambda: None)
            self.assertEqual(actual, expected)
            for codes in self.cut_oracle(problem, domains):
                self.assertTrue(all(keep & (1 << codes[c]) for c, keep in actual))
        self.assertIs(helper._cut_support(original, (0, 1), lambda: None), first)

    def test_partial_budget_keeps_completed_inventory_facts_but_no_cut_union(self):
        problem, domains = self.fixture();before = domains.copy()
        helper = AdvancedPruner(problem, options={'native': 'python'});helper.cut_matching_work_limit = 20
        self.assertIsNone(helper._cut_support(domains, (0, 1), lambda: None))
        self.assertFalse(helper._cut_cache);self.assertTrue(helper._cut_inventory_cache)
        self.assertEqual(domains, before);self.assertEqual(helper.stats['cut_global_matching_steps'], 20)
        self.assertEqual(helper.stats['cut_global_matching_budget_skips'], 1)
        helper.cut_matching_work_limit = 200000
        result = helper._cut_support(domains, (0, 1), lambda: None)
        self.assertEqual(result, AdvancedPruner(problem, options={'native': 'python'})._cut_support(domains, (0, 1), lambda: None))
        self.assertGreater(helper.stats['cut_global_matching_cache_hits'], 0)

    def test_stop_during_global_matching_never_publishes_incomplete_support(self):
        from unittest.mock import patch
        problem, domains = self.fixture();helper = AdvancedPruner(problem, options={'native': 'python'})
        original = _complete_matching;calls = [0]
        def interrupted(rows, check, hint):
            calls[0] += 1
            if calls[0] == 2:
                def cancel():raise InterruptedError('Stop during conditional matching')
                return original(rows, cancel, hint)
            return original(rows, check, hint)
        with patch('constraint_advanced._complete_matching', side_effect=interrupted):
            with self.assertRaises(InterruptedError):helper._cut_support(domains, (0, 1), lambda: None)
        self.assertEqual(calls[0], 2);self.assertFalse(helper._cut_cache)
        self.assertEqual(len(helper._cut_inventory_cache), 1)
        self.assertEqual(domains, self.fixture()[1])
        self.assertEqual(helper._cut_support(domains, (0, 1), lambda: None),
                         AdvancedPruner(problem, options={'native': 'python'})._cut_support(domains, (0, 1), lambda: None))

    def test_every_gold_solution_survives_and_edge_target_remains_unchanged(self):
        problem = ConditionalCutMatchingTests.fixture(False)
        solutions = brute(problem, 'gold');self.assertEqual(len(solutions), 768)
        helper = AdvancedPruner(problem, options={'native': 'python'})
        rng = random.Random(21003)
        for _ in range(32):
            domains = [rng.randrange(1, 1 << 12) for _ in range(4)]
            expected = [row for row in solutions if all(domain & (1 << code) for domain, code in zip(domains, row))]
            helper.prune_cuts(domains)
            for row in expected:self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, row)))
        domains = domains_of(problem);before = domains.copy()
        self.assertFalse(AdvancedPruner(problem, target='edges').prune_cuts(domains))
        self.assertEqual(domains, before)


class ConditionalGlobalSupportTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # Deterministic seed22001 sample2, already a fixed point of R21's cut
        # union followed by unconditioned Hall filtering. Cell3 is outside.
        problem, _ = ConditionalGlobalInventoryTests.fixture()
        return problem, [2116, 240, 343, 3189]

    def test_full_matching_supports_equal_independent_enumeration(self):
        rng = random.Random(22002);hint=None
        for n in range(8):
            for _ in range(40):
                rows=tuple(rng.randrange(1 << min(n+1,8)) for _ in range(n))
                supported,matching=_complete_matching_supports(rows,hint=hint)
                self.assertEqual(supported,ConditionalCutMatchingTests.oracle(rows),rows)
                if matching is not None:hint=matching
        for n in range(4):
            for rows in product(range(1 << n),repeat=n):
                self.assertEqual(_complete_matching_supports(rows)[0],ConditionalCutMatchingTests.oracle(rows),rows)
        rows=(1 << 159,(1 << 157)|(1 << 159))
        self.assertEqual(_complete_matching_supports(rows)[0],(1 << 159,1 << 157))

    def test_support_filter_reaches_outside_cell_beyond_old_cut_and_hall(self):
        from unittest.mock import patch
        from constraint_pruning import Pruner
        problem,domains=self.fixture();before=domains.copy()
        self.assertFalse(Pruner(problem,native='python').prune_hall(domains))
        # Disable only SCC supports, retaining exact whole-board feasibility.
        with patch('constraint_advanced._matching_supports',side_effect=lambda rows,*args:rows):
            old=dict(AdvancedPruner(problem,options={'native':'python'})._cut_support(domains,(0,1),lambda:None))
        self.assertEqual([d&old[c] for c,d in enumerate(domains)],domains)
        helper=AdvancedPruner(problem,options={'native':'python'})
        support=dict(helper._cut_support(domains,(0,1),lambda:None))
        after=[d&support[c] for c,d in enumerate(domains)]
        self.assertEqual(after,[2116,240,343,3141]);self.assertEqual(domains,before)
        completions=ConditionalGlobalInventoryTests.cut_oracle(problem,domains)
        self.assertEqual(len(completions),12)
        exact=[0]*4
        for codes in completions:
            for cell,code in enumerate(codes):exact[cell] |= 1 << code
        self.assertEqual(after,exact)
        self.assertGreater(helper.stats['cut_global_matching_support_prunes'],0)

    def test_cached_outside_tile_supports_follow_current_rotations(self):
        from constraint_pruning import _tile_choices
        problem,domains=self.fixture();helper=AdvancedPruner(problem,options={'native':'python'})
        original=helper._cut_support(domains,(0,1),lambda:None)
        for rotation in range(3):
            current=domains.copy();current[3]=bits(3*t+rotation for t in values(_tile_choices(domains[3])))
            cached=helper._cut_support(current,(0,1),lambda:None)
            fresh=AdvancedPruner(problem,options={'native':'python'})._cut_support(current,(0,1),lambda:None)
            self.assertIs(cached,original);self.assertEqual(cached,fresh)
            after=[d&dict(cached)[c] for c,d in enumerate(current)]
            for codes in ConditionalGlobalInventoryTests.cut_oracle(problem,current):
                self.assertTrue(all(d&(1<<code) for d,code in zip(after,codes)))
            self.assertTrue(all((after[3]>>code)&1==0 for code in range(12) if code%3!=rotation))

    def test_support_budget_retains_old_deductions_without_caching_partial_scc(self):
        from unittest.mock import patch
        problem,domains=ConditionalGlobalInventoryTests.fixture()
        with patch('constraint_advanced._matching_supports',side_effect=lambda rows,*args:rows):
            old=AdvancedPruner(problem,options={'native':'python'})._cut_support(domains,(0,1),lambda:None)
        for limit in (0,1,2,5):
            helper=AdvancedPruner(problem,options={'native':'python'});helper.cut_support_work_limit=limit
            result=helper._cut_support(domains,(0,1),lambda:None)
            self.assertEqual(result,old);self.assertFalse(helper._cut_cache)
            self.assertGreater(helper.stats['cut_global_support_budget_skips'],0)
            self.assertEqual(helper.stats['cut_global_matching_budget_skips'],0)
            # Failed matchings may be cached; interrupted support tables may not.
            self.assertTrue(all(value is None for value in helper._cut_inventory_cache.values()))
            helper.cut_support_work_limit=200000
            self.assertEqual(helper._cut_support(domains,(0,1),lambda:None),
                             AdvancedPruner(problem,options={'native':'python'})._cut_support(domains,(0,1),lambda:None))

    def test_stop_inside_scc_publishes_no_incomplete_table(self):
        from unittest.mock import patch
        import constraint_advanced
        problem,domains=self.fixture();original=constraint_advanced._matching_supports
        helper=AdvancedPruner(problem,options={'native':'python'});before=domains.copy()
        def stopped(rows,matching,check):
            def stop():raise InterruptedError('Stop during SCC support filtering')
            return original(rows,matching,stop)
        with patch('constraint_advanced._matching_supports',side_effect=stopped):
            with self.assertRaises(InterruptedError):helper._cut_support(domains,(0,1),lambda:None)
        self.assertFalse(helper._cut_cache);self.assertFalse(helper._cut_inventory_cache)
        self.assertEqual(domains,before)
        self.assertEqual(helper._cut_support(domains,(0,1),lambda:None),
                         AdvancedPruner(problem,options={'native':'python'})._cut_support(domains,(0,1),lambda:None))


class DynamicCutTests(unittest.TestCase):
    @staticmethod
    def two_blocks():
        # Two K4-minus-one-edge components joined by two seams. Adjacent-pair
        # regions each have four exits, so the old static filter misses this.
        pairs = [(a, b) for base in (0, 4) for a in range(base, base+4)
                 for b in range(a+1, base+4) if (a, b) != (base, base+1)]
        pairs += [(0, 4), (1, 5)]
        sides = [0]*8; edges = []
        for a, b in pairs:
            edges.append((a, sides[a], b, sides[b])); sides[a] += 1; sides[b] += 1
        tiles = [{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(8)]
        problem = SearchProblem(tiles, board(8, edges))
        return problem, [7 << (3*cell) for cell in range(8)]

    def test_new_four_cell_bottleneck_removes_branches_and_preserves_all_gold_solutions(self):
        problem, original = self.two_blocks()
        helper = AdvancedPruner(problem, options={'native': 'python'})
        possible = (True,)*len(problem.edges)
        self.assertNotIn((10, 11), [boundary for _, boundary in helper.cut_regions])
        self.assertIn((10, 11), small_cut_boundaries(problem.n, problem.edges, possible))
        # Independently enumerate every orientation combination with one
        # physical tile per cell, then validate full loops with the validator.
        solutions = [tuple(3*c+r for c, r in enumerate(rotations))
                     for rotations in product(range(3), repeat=8)
                     if problem.validate([3*c+r for c, r in enumerate(rotations)])['valid']]
        self.assertTrue(solutions)
        domains = original.copy()
        self.assertTrue(helper.prune_cuts(domains))
        self.assertGreater(helper.stats['dynamic_cut_prunes'], 0)
        self.assertGreater(helper.stats['dynamic_cut_checks'], 0)
        before = 1; after = 1
        for old, new in zip(original, domains):
            before *= old.bit_count(); after *= new.bit_count()
        self.assertLess(after, before)
        for solution in solutions:
            self.assertTrue(all(d & (1 << code) for d, code in zip(domains, solution)))
        self.assertEqual([row for row in solutions if all(d & (1 << code) for d, code in zip(domains, row))], solutions)

    def test_discovered_cuts_match_independent_vertex_subset_boundaries(self):
        rng = random.Random(194853)
        for n in range(2, 9):
            for case in range(20):
                # The chain keeps the graph connected; extra parallel edges
                # exercise edge identity independently of neighbor identity.
                pairs = [(i-1, i) for i in range(1, n)]
                pairs += [tuple(rng.sample(range(n), 2)) for _ in range(rng.randrange(2*n))]
                edges = tuple((a, 0, b, 0) for a, b in pairs)
                possible = (True,)*len(edges)
                expected = set()
                for region in range(1, 1 << (n-1)):
                    cut = tuple(e for e, (a, _, b, _) in enumerate(edges)
                                if bool(region & (1 << a)) != bool(region & (1 << b)))
                    if len(cut) in (2, 3): expected.add(cut)
                actual = set(small_cut_boundaries(n, edges, possible, limit=10000))
                self.assertLessEqual(actual, expected)
                self.assertLessEqual({cut for cut in expected if len(cut) == 2}, actual)

    def test_exact_topology_cache_backtracking_and_interruption(self):
        problem, original = self.two_blocks()
        helper = AdvancedPruner(problem, options={'native': 'python'})
        helper.prune_cuts(original.copy())
        expected = original.copy(); helper.prune_cuts(expected)
        self.assertGreater(helper.stats['dynamic_cut_cache_hits'], 0)
        for cell in range(8):
            domains = original.copy(); domains[cell] &= 1 << (3*cell)
            actual = domains.copy(); fresh = domains.copy()
            helper.prune_cuts(actual)
            AdvancedPruner(problem, options={'native': 'python'}).prune_cuts(fresh)
            self.assertEqual(actual, fresh)
        helper = AdvancedPruner(problem, options={'native': 'python'})
        helper.cut_regions = ()
        domains = original.copy(); calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 4: raise InterruptedError
        with self.assertRaises(InterruptedError): helper.prune_cuts(domains, cancel)
        self.assertEqual(domains, original)
        self.assertFalse(helper._dynamic_cut_cache)
        self.assertLessEqual(len(small_cut_boundaries(problem.n, problem.edges,
                                                     (True,)*len(problem.edges), limit=2)), 2)

    def test_disabled_seams_and_disconnected_graphs_never_invent_boundaries(self):
        problem, _ = self.two_blocks()
        possible = tuple(e not in (1, 4, 10) for e in range(len(problem.edges)))
        expected = set()
        for region in range(1, 1 << (problem.n-1)):
            cut = tuple(e for e, (a, _, b, _) in enumerate(problem.edges) if possible[e]
                        and bool(region & (1 << a)) != bool(region & (1 << b)))
            if len(cut) in (2, 3): expected.add(cut)
        self.assertLessEqual(set(small_cut_boundaries(problem.n, problem.edges, possible)), expected)


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
