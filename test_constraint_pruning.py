"""Finite, independent soundness checks for reusable mathematical pruning."""
from collections import Counter
from itertools import permutations, product
import random
from types import SimpleNamespace
import unittest

import numpy as np

from constraint_pruning import Pruner, STAT_KEYS, boundary_minimum, _tile_choices, _orientation_choices, _distinct_supports
from geometry import build_board
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import board, double_face, simple_tiles, tetrahedron_tiles, brute
from validator import reverse_mask


def bits(codes):
    return sum(1 << code for code in set(codes))


def tile_domain(tiles):
    return bits(3*tile+rotation for tile in tiles for rotation in range(3))


def blank_problem(n):
    # Pure AllDifferent fixtures need no gold or geometry assumptions.
    return SimpleNamespace(n=n, board=board(n, []), masks=np.zeros((3*n, 3), dtype=np.uint16),
                           neighbors=np.full((n, 3), -1), sides=np.full((n, 3), -1))


def masks_to_tiles(rows):
    result = []
    for tile, masks in enumerate(rows):
        segments = []
        for side, mask in enumerate(masks):
            points = [[side, position+1] for position in range(11) if mask & (1 << position)]
            if len(points) % 2:
                raise ValueError('Fixture edges must have even endpoint counts.')
            segments.extend([points[index:index+2] for index in range(0, len(points), 2)])
        result.append({'id': tile, 'segments': segments})
    return result


def unique_tetrahedron():
    _, geometry = tetrahedron_tiles()
    masks = [3, 5, 9, 17, 33, 65]
    rows = [[0]*3 for _ in range(4)]
    for (a, sa, b, sb), mask in zip(geometry.edges, masks):
        rows[a][sa], rows[b][sb] = mask, reverse_mask(mask)
    return SearchProblem(masks_to_tiles(rows), geometry)


def supported_assignments(domains, n):
    return [order for order in permutations(range(n))
            if all(domains[cell] & (7 << (3*tile)) for cell, tile in enumerate(order))]


class InventoryTests(unittest.TestCase):
    def test_bound_counts_empty_and_self_reversed_signatures(self):
        self.assertEqual(boundary_minimum({0: 3}), 1)
        self.assertEqual(boundary_minimum({32: 3}), 1)
        self.assertEqual(boundary_minimum({1: 5, 1024: 2, 0: 3, 32: 2}), 4)
        self.assertEqual(boundary_minimum({1: 2, 1024: 2, 0: 4}), 0)

    def test_bound_matches_independent_relaxed_pair_enumeration(self):
        def minimum_unpaired(items):
            if not items:
                return 0
            result = 1+minimum_unpaired(items[1:])
            for index in range(1, len(items)):
                if items[index] == reverse_mask(items[0]):
                    result = min(result, minimum_unpaired(items[1:index]+items[index+1:]))
            return result
        for counts in product(range(3), repeat=4):
            inventory = dict(zip((0, 1, 1024, 32), counts))
            items = tuple(mask for mask, count in inventory.items() for _ in range(count))
            self.assertEqual(boundary_minimum(inventory), minimum_unpaired(items))

    def test_macroface_geometry_has_twelve_boundary_edges(self):
        geometry = build_board()
        fake = blank_problem(160)
        fake.board = geometry
        fake.neighbors = geometry.neighbor_cells
        fake.sides = geometry.neighbor_sides
        pruner = Pruner(fake, parity=False)
        face_regions = [(cells, boundary, size) for cells, boundary, size in pruner.regions if len(cells) == 16]
        self.assertEqual(len(face_regions), 10)
        self.assertTrue(all(size == 12 for _, _, size in face_regions))
        self.assertEqual(pruner.regions[-1][2], 0)

    def test_fixed_inventory_rejects_excess_boundary_requirement(self):
        _, geometry = tetrahedron_tiles()
        first, second = [3, 5, 9], [17, 33, 65]
        rows = [first, second, list(map(reverse_mask, first)), list(map(reverse_mask, second))]
        problem = SearchProblem(masks_to_tiles(rows), geometry)
        pruner = Pruner(problem, regions=[(0, 1)])
        domains = [tile_domain((0, 1))]*2+[tile_domain((2, 3))]*2
        self.assertEqual(pruner.regions[0][2], 4)
        self.assertTrue(pruner.prune_regions(domains))
        self.assertIn(0, domains)
        self.assertEqual(pruner.stats['region_failures'], 1)

    def test_unknown_inventory_is_not_assumed_fixed(self):
        problem = unique_tetrahedron()
        pruner = Pruner(problem, regions=[(0, 1)])
        domains = [tile_domain(range(4)) for _ in range(4)]
        before = domains.copy()
        self.assertFalse(pruner.prune_regions(domains))
        self.assertEqual(domains, before)
        self.assertEqual(pruner.stats['region_checks'], 0)

    def test_saturated_boundary_inventory_excludes_wrong_rotations(self):
        problem = unique_tetrahedron()
        pruner = Pruner(problem, regions=[(0, 1)])
        domains = [tile_domain((0, 1))]*2+[tile_domain((2, 3))]*2
        # The only internal-compatible pair is physical side zero of tiles 0/1;
        # every other signature is required on the four exterior patch edges.
        self.assertTrue(pruner.prune_regions(domains))
        self.assertEqual(domains[:2], [bits((0, 3)), bits((0, 3))])
        self.assertGreater(pruner.stats['region_prunes'], 0)

    def test_bad_regions_are_rejected(self):
        for regions in ([[]], [[-1]], [[2]], [[True]]):
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                Pruner(blank_problem(2), regions=regions)


class MatchingTests(unittest.TestCase):
    def test_hall_set_prunes_without_singleton_cells_or_tiles(self):
        pruner = Pruner(blank_problem(5), regions=[])
        domains = [tile_domain((0, 1)), tile_domain((0, 1))]+[tile_domain(range(5))]*3
        self.assertTrue(pruner.prune_hall(domains))
        self.assertEqual(domains[:2], [tile_domain((0, 1))]*2)
        self.assertEqual(domains[2:], [tile_domain((2, 3, 4))]*3)
        self.assertEqual(pruner.stats['hall_prunes'], 18)

    def test_hall_failure_and_rotations_do_not_create_extra_tiles(self):
        for domains in ([tile_domain((0, 1))]*3+[tile_domain((2, 3))],
                        [bits((0, 1)), bits((2,))]):
            with self.subTest(domains=domains):
                pruner = Pruner(blank_problem(len(domains)), regions=[])
                self.assertTrue(pruner.prune_hall(domains))
                self.assertIn(0, domains)
                self.assertEqual(pruner.stats['hall_failures'], 1)

    def test_every_three_cell_domain_graph_matches_permutation_oracle(self):
        pruner = Pruner(blank_problem(3), regions=[])
        for graph in product(range(1, 8), repeat=3):
            domains = [tile_domain(tile for tile in range(3) if mask & (1 << tile)) for mask in graph]
            before = domains.copy()
            expected = supported_assignments(before, 3)
            pruner.prune_hall(domains)
            if not expected:
                self.assertIn(0, domains)
            else:
                for cell in range(3):
                    allowed = {order[cell] for order in expected}
                    self.assertEqual(domains[cell], before[cell] & tile_domain(allowed))

    def test_matching_preserves_allowed_rotations_exactly(self):
        rng = random.Random(4091)
        pruner = Pruner(blank_problem(4), regions=[])
        for _ in range(80):
            domains = [bits(code for code in range(12) if rng.random() < .45) for _ in range(4)]
            if 0 in domains:
                continue
            before = domains.copy()
            expected = supported_assignments(before, 4)
            pruner.prune_hall(domains)
            if not expected:
                self.assertIn(0, domains)
            else:
                self.assertEqual(domains, [before[cell] & tile_domain({order[cell] for order in expected}) for cell in range(4)])


class ClusterTests(unittest.TestCase):
    def test_unique_bonds_determine_all_cluster_embeddings(self):
        problem = unique_tetrahedron()
        pruner = Pruner(problem)
        self.assertEqual(len(pruner.bonds), 6)
        self.assertEqual(len(pruner.clusters), 1)
        self.assertEqual(pruner.stats['forced_bonds'], 6)
        self.assertEqual(pruner.stats['forced_clusters'], 1)
        expected = set(brute(problem, 'edges'))
        embeddings = {tuple(code for _, code in embedding) for embedding in pruner.clusters[0][2]}
        self.assertEqual(embeddings, expected)
        domains = [bits(problem.domains[cell]) for cell in range(4)]
        domains[0] = 1 << 0
        self.assertTrue(pruner.prune_clusters(domains))
        self.assertEqual(domains, [1 << (3*cell) for cell in range(4)])

    def test_forced_self_partner_on_closed_board_is_impossible(self):
        data = [{'id': 0, 'segments': [[[0, 2], [1, 10]]]},
                {'id': 1, 'segments': [[[0, 6], [1, 6]]]}]
        problem = SearchProblem(data, double_face())
        self.assertFalse(brute(problem, 'edges'))
        pruner = Pruner(problem)
        domains = [bits(row) for row in problem.domains]
        self.assertTrue(pruner.prune_clusters(domains))
        self.assertIn(0, domains)

    def test_open_board_does_not_force_boundary_edges_together(self):
        tiles = [{'id': tile, 'segments': [[[0, 2], [0, 10]]]} for tile in range(2)]
        problem = SearchProblem(tiles, board(2, [(0, 0, 1, 0)]))
        pruner = Pruner(problem)
        self.assertEqual(pruner.bonds, ())
        self.assertEqual(pruner.clusters, ())
        domains = [bits(row) for row in problem.domains]
        pruner.prune(domains)
        for solution in brute(problem, 'edges'):
            self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, solution)))


class NeighborhoodTests(unittest.TestCase):
    def test_bit_gather_matches_per_orientation_projection(self):
        rng = random.Random(1441)
        domains = list(range(4096)) + [rng.getrandbits(480) for _ in range(200)]
        domains += [1 << code for code in range(480)]
        for domain in domains:
            expected = 0
            for code in range(domain.bit_length()):
                if domain & (1 << code):
                    expected |= 1 << (code//3)
            self.assertEqual(_tile_choices(domain), expected)

    def test_three_neighbor_hall_matches_independent_injective_choices(self):
        for count in range(4):
            for choices in product(range(8), repeat=count):
                expected = any(all(choices[cell] & (1 << tile) for cell, tile in enumerate(order))
                               for order in permutations(range(3), count))
                self.assertEqual(_distinct_supports(choices), expected)

    def test_detects_star_conflict_missed_by_binary_arcs_and_global_hall(self):
        # Triangular-prism adjacency: every cell has three DISTINCT neighbors.
        # Three tiles of each signature color give pairwise seam support and a
        # global perfect tile matching, but a tile has only two distinct partners.
        pairs = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5))
        degrees, edges = [0]*6, []
        for a, b in pairs:
            edges.append((a, degrees[a], b, degrees[b]))
            degrees[a] += 1
            degrees[b] += 1
        problem = SearchProblem(masks_to_tiles([[1025]*3]*3+[[514]*3]*3), board(6, edges))
        domains = [bits(row) for row in problem.domains]
        for cell in range(6):
            for code in problem.domains[cell]:
                for side in range(3):
                    neighbor, other_side = problem.neighbors[cell, side], problem.sides[cell, side]
                    self.assertTrue(any(other//3 != code//3 and
                                        problem.masks[other, other_side] == reverse_mask(int(problem.masks[code, side]))
                                        for other in problem.domains[neighbor]))
        pruner = Pruner(problem, regions=[])
        self.assertFalse(pruner.prune_hall(domains))
        before = domains.copy()
        self.assertFalse(pruner.prune_neighborhood(domains, max_domain_values=5))
        self.assertEqual(domains, before)
        self.assertTrue(pruner.prune_neighborhood(domains))
        self.assertIn(0, domains)
        self.assertEqual(pruner.stats['neighborhood_failures'], 1)

    def test_parallel_seams_need_one_neighbor_orientation(self):
        problem = SearchProblem(masks_to_tiles([[3, 5, 9], [reverse_mask(m) for m in (3, 9, 5)]]), double_face())
        self.assertFalse(brute(problem, 'edges'))
        domains = [bits(row) for row in problem.domains]
        pruner = Pruner(problem, regions=[])
        self.assertTrue(pruner.prune_neighborhood(domains))
        self.assertIn(0, domains)

    def test_parallel_neighbor_is_counted_once_and_valid_boards_survive(self):
        problem = SearchProblem(simple_tiles(), double_face())
        expected = brute(problem, 'edges')
        self.assertTrue(expected)
        pruner = Pruner(problem, regions=[])
        for codes in expected:
            domains = [1 << code for code in codes]
            self.assertFalse(pruner.prune_neighborhood(domains))
            self.assertEqual(domains, [1 << code for code in codes])

    def test_open_cells_have_no_artificial_neighbors(self):
        pruner = Pruner(blank_problem(4), regions=[])
        domains = [tile_domain(range(4))]*4
        self.assertFalse(pruner.prune_neighborhood(domains))
        self.assertEqual(pruner.stats['neighborhood_checks'], 0)


class PersistentNeighborhoodTests(unittest.TestCase):
    @staticmethod
    def padded(problem):
        # Preserve an independently enumerable small star and its physical tile
        # identities while exercising the larger-board persistent-cache path.
        padded = blank_problem(32)
        padded.masks[:3*problem.n] = problem.masks
        padded.neighbors[:problem.n] = problem.neighbors
        padded.sides[:problem.n] = problem.sides
        return padded

    @staticmethod
    def oracle(problem, domains, limit=None):
        from constraint_pruning import values
        snapshot, expected = domains.copy(), domains.copy()
        for cell, domain in enumerate(snapshot):
            if limit is not None and domain.bit_count() > limit:
                continue
            neighbors = sorted({int(nb) for nb in problem.neighbors[cell] if nb >= 0})
            if not neighbors:
                continue
            keep = 0
            for code in values(domain):
                candidates = []
                for neighbor in neighbors:
                    candidates.append([other for other in values(snapshot[neighbor])
                        if other//3 != code//3 and all(
                            int(problem.masks[other, problem.sides[cell, side]]) ==
                            reverse_mask(int(problem.masks[code, side]))
                            for side, nb in enumerate(problem.neighbors[cell]) if nb == neighbor)])
                if any(len({other//3 for other in choice}) == len(neighbors)
                       for choice in product(*candidates)):
                    keep |= 1 << code
            expected[cell] &= keep
            if not expected[cell]:
                break
        return expected

    def test_exact_keys_match_independent_stars_under_narrowing_widening_and_rotation(self):
        fixtures = [unique_tetrahedron(), SearchProblem(simple_tiles(), double_face()),
            SearchProblem(masks_to_tiles([[3, 5, 9], [reverse_mask(m) for m in (3, 9, 5)]]), double_face())]
        rng = random.Random(918304)
        for small in fixtures:
            problem = self.padded(small); helper = Pruner(problem, parity=False, regions=[])
            valid = brute(small, 'edges')
            for case in range(160):
                initial = [bits(code for code in range(3*small.n) if rng.random() < (.8 if case%4 else 1))
                           for _ in range(small.n)] + [1 << (3*cell) for cell in range(small.n, 32)]
                expected = self.oracle(problem, initial)
                for repeat in range(2):
                    actual = initial.copy(); before = helper.stats['neighborhood_checks']
                    self.assertEqual(helper.prune_neighborhood(actual), initial != expected)
                    self.assertEqual(actual, expected)
                    if repeat:
                        self.assertEqual(helper.stats['neighborhood_checks'], before)
                for completion in valid:
                    if all(initial[cell] & (1 << code) for cell, code in enumerate(completion)):
                        self.assertTrue(all(actual[cell] & (1 << code) for cell, code in enumerate(completion)))
                self.assertLessEqual(len(helper._neighborhood_cache), 512)

    def test_changed_neighbor_and_parallel_seam_profile_cannot_reuse_wrong_support(self):
        small = SearchProblem(simple_tiles(), double_face())
        problem = self.padded(small); helper = Pruner(problem, parity=False, regions=[])
        tail = [1 << (3*cell) for cell in range(2,32)]
        for left,right in ((1,8),(1,16),(1,8),(7,56),(2,8),(7,56)):
            initial=[left,right]+tail;actual=initial.copy()
            helper.prune_neighborhood(actual)
            self.assertEqual(actual,self.oracle(problem,initial))
        # Identical domain integers on another puzzle with different seam
        # profiles must start from a private empty cache even with reused tables.
        warm=Pruner(problem,parity=False,regions=[],tables=helper.tables)
        self.assertEqual(warm._neighborhood_cache,{})
        self.assertIsNot(warm._neighborhood_cache,helper._neighborhood_cache)
        self.assertIs(warm._stars,helper._stars)
        different=self.padded(SearchProblem(masks_to_tiles([[3,5,9],[reverse_mask(m) for m in (3,9,5)]]),double_face()))
        other=Pruner(different,parity=False,regions=[]);initial=[7,56]+tail;actual=initial.copy()
        other.prune_neighborhood(actual)
        self.assertEqual(actual,self.oracle(different,initial))

    def test_eviction_is_bounded_and_recomputes_old_keys(self):
        problem=blank_problem(32)
        problem.neighbors[0,0]=1;problem.sides[0,0]=0
        problem.neighbors[1,0]=0;problem.sides[1,0]=0
        helper=Pruner(problem,parity=False,regions=[])
        tail=[1 << (3*cell) for cell in range(2,32)]
        first=[1,8]+tail;helper.prune_neighborhood(first.copy())
        oldest=next(iter(helper._neighborhood_cache))
        for index in range(2,602):
            initial=[1,index << 3]+tail;actual=initial.copy()
            helper.prune_neighborhood(actual)
            self.assertEqual(actual,self.oracle(problem,initial))
            self.assertLessEqual(len(helper._neighborhood_cache),512)
        self.assertEqual(len(helper._neighborhood_cache),512)
        self.assertNotIn(oldest,helper._neighborhood_cache)
        before=helper.stats['neighborhood_checks'];actual=first.copy()
        helper.prune_neighborhood(actual)
        self.assertGreater(helper.stats['neighborhood_checks'],before)
        self.assertEqual(actual,self.oracle(problem,first))

    def test_cancellation_retains_only_completed_stars_and_resume_is_exact(self):
        small=unique_tetrahedron();problem=self.padded(small)
        initial=[4095]*4+[1 << (3*cell) for cell in range(4,32)]
        for stop_after_completed in (False,True):
            helper=Pruner(problem,parity=False,regions=[])
            helper.prune_neighborhood(initial.copy(),max_domain_values=1)  # Build immutable profiles only.
            calls=[0]
            def stop():
                calls[0]+=1
                if (bool(helper._neighborhood_cache) if stop_after_completed else calls[0]==2):
                    raise InterruptedError('star query')
            helper.check=stop;actual=initial.copy()
            with self.assertRaises(InterruptedError):helper.prune_neighborhood(actual)
            self.assertEqual(len(helper._neighborhood_cache),int(stop_after_completed))
            for completion in brute(small,'edges'):
                self.assertTrue(all(actual[cell] & (1 << code) for cell,code in enumerate(completion)))
            helper.check=lambda:None;expected=self.oracle(problem,actual)
            helper.prune_neighborhood(actual)
            self.assertEqual(actual,expected)
            widened=initial.copy();helper.prune_neighborhood(widened)
            self.assertEqual(widened,self.oracle(problem,initial))

    def test_tiny_gate_and_eligible_limit_preserve_private_cache_rules(self):
        small=unique_tetrahedron();helper=Pruner(small,parity=False)
        for _ in range(2):helper.prune_neighborhood([4095]*4)
        self.assertEqual(helper._neighborhood_cache,{})
        large=Pruner(self.padded(small),parity=False,regions=[])
        initial=[4095]*4+[1 << (3*cell) for cell in range(4,32)]
        self.assertFalse(large.prune_neighborhood(initial.copy(),max_domain_values=1))
        self.assertEqual(large._neighborhood_cache,{})
        large.prune_neighborhood(initial.copy())
        self.assertTrue(large._neighborhood_cache)
        # Eligibility is checked before cache lookup; unrestricted cached work
        # must not alter a later restricted invocation.
        actual=initial.copy();self.assertFalse(large.prune_neighborhood(actual,max_domain_values=1))
        self.assertEqual(actual,initial)


class ReuseTests(unittest.TestCase):
    def test_orientation_scatter_and_gather_are_exact(self):
        rng = random.Random(7781)
        masks = list(range(4096)) + [rng.getrandbits(160) for _ in range(100)]
        for tiles in masks:
            expected = sum(7 << (3*tile) for tile in range(tiles.bit_length()) if tiles & (1 << tile))
            self.assertEqual(_orientation_choices(tiles), expected)
            self.assertEqual(_tile_choices(expected), tiles)
        self.assertEqual(_tile_choices.cache_info().maxsize, 8192)
        self.assertEqual(_orientation_choices.cache_info().maxsize, 8192)

    def test_warm_matching_repairs_edges_after_backtracking_and_failure(self):
        pruner = Pruner(blank_problem(4), regions=[])
        rng = random.Random(4281)
        cases = [[tile_domain(range(4))]*4,
                 [tile_domain((1,)), tile_domain((0,)), tile_domain((3,)), tile_domain((2,))],
                 [tile_domain((0, 1))]*3+[tile_domain((2, 3))]]
        cases += [[bits(code for code in range(12) if rng.random() < .5) for _ in range(4)] for _ in range(80)]
        cases += cases[:2]
        for before in cases:
            if not all(before):
                continue
            expected = supported_assignments(before, 4)
            domains = before.copy()
            pruner.prune_hall(domains)
            if not expected:
                self.assertIn(0, domains)
            else:
                self.assertEqual(domains, [before[cell] & tile_domain({order[cell] for order in expected}) for cell in range(4)])
            self.assertLessEqual(len(pruner._hall_cache), 16)

    def test_hall_cache_ignores_rotation_changes_but_preserves_them(self):
        pruner = Pruner(blank_problem(4), regions=[])
        wide = [tile_domain((0, 1))]*2+[tile_domain(range(4))]*2
        pruner.prune_hall(wide.copy())
        narrow = [bits(3*tile+(cell+tile)%3 for tile in range(4) if wide[cell] & (7 << (3*tile)))
                  for cell in range(4)]
        expected = supported_assignments(narrow, 4)
        before = narrow.copy()
        pruner.prune_hall(narrow)
        self.assertEqual(narrow, [before[cell] & tile_domain({order[cell] for order in expected}) for cell in range(4)])
        self.assertEqual(pruner.stats['hall_checks'], 2)

    def test_cluster_reuse_matches_fresh_filter_after_expansion_and_siblings(self):
        rows = [[3, 1025, 1025], [reverse_mask(3), 1025, 1025],
                [5, 1025, 1025], [reverse_mask(5), 1025, 1025]]
        topology = board(4, [(a, side, a+1, side) for a in (0, 2) for side in range(3)])
        problem = SearchProblem(masks_to_tiles(rows), topology)
        solutions = brute(problem, 'edges')
        self.assertTrue(solutions)
        reused = Pruner(problem, regions=[])
        self.assertEqual(len(reused.clusters), 2)
        rng = random.Random(991)
        for trial in range(60):
            witness = rng.choice(solutions)
            initial = [bits([witness[cell]]+[code for code in row if rng.random() < (1 if trial % 5 == 0 else .4)])
                       for cell, row in enumerate(problem.domains)]
            expected = initial.copy()
            Pruner(problem, regions=[]).prune_clusters(expected)
            actual = initial.copy()
            reused.prune_clusters(actual)
            self.assertEqual(actual, expected)
            self.assertTrue(all(actual[cell] & (1 << code) for cell, code in enumerate(witness)))
            self.assertTrue(all(len(cache) <= 8 for cache in reused._cluster_cache))

    def test_cached_reductions_are_safe_if_interrupted_then_reused(self):
        before = [tile_domain((0, 1))]*2+[tile_domain(range(5))]*3
        domains = before.copy()
        armed = [False]
        def check():
            if armed[0] and domains != before:
                raise InterruptedError('after cached deletion')
        pruner = Pruner(blank_problem(5), check, regions=[])
        expected = before.copy()
        pruner.prune_hall(expected)
        armed[0] = True
        with self.assertRaises(InterruptedError):
            pruner.prune_hall(domains)
        for order in supported_assignments(before, 5):
            self.assertTrue(all(domain & (7 << (3*tile)) for domain, tile in zip(domains, order)))
        armed[0] = False
        pruner.prune_hall(domains)
        self.assertEqual(domains, expected)
        expanded = [tile_domain(range(5))]*5
        self.assertFalse(pruner.prune_hall(expanded))
        self.assertEqual(expanded, [tile_domain(range(5))]*5)


class CombinedSoundnessTests(unittest.TestCase):
    def test_restricted_tiny_domains_preserve_every_valid_edge_completion(self):
        data, geometry = tetrahedron_tiles()
        fixtures = [SearchProblem(simple_tiles(), double_face()), SearchProblem(data, geometry), unique_tetrahedron()]
        rng = random.Random(55801)
        for problem in fixtures:
            expected = brute(problem, 'edges')
            self.assertTrue(expected)
            pruner = Pruner(problem, regions=[(0,), (0, 1), tuple(range(problem.n))])
            for _ in range(24):
                witness = rng.choice(expected)
                domains = [bits([witness[cell]]+[code for code in row if rng.random() < .5]) for cell, row in enumerate(problem.domains)]
                retained = [codes for codes in expected if all(domain & (1 << code) for domain, code in zip(domains, codes))]
                for _ in range(4):
                    if not pruner.prune(domains):
                        break
                self.assertNotIn(0, domains)
                for codes in retained:
                    self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, codes)))

    def test_planted_gold_certificates_survive_small_and_full_board_filtering(self):
        for seed, size in ((17, 1), (29, 2), (31, 4)):
            with self.subTest(seed=seed, size=size):
                data, geometry, reference = planted_gold(seed, size)
                problem = SearchProblem(data, geometry)
                self.assertTrue(problem.validate(reference)['valid'])
                pruner = Pruner(problem)
                rng = random.Random(seed+100)
                domains = [bits([reference[cell]]+[code for code in row if rng.random() < .2]) for cell, row in enumerate(problem.domains)]
                for _ in range(3):
                    pruner.prune(domains)
                    self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, reference)))
                singleton = [1 << code for code in reference]
                self.assertFalse(pruner.prune(singleton))
                self.assertEqual(singleton, [1 << code for code in reference])

    def test_cooperative_interruption_after_a_deletion_remains_sound(self):
        domains = [tile_domain((0, 1)), tile_domain((0, 1))]+[tile_domain(range(5))]*3
        before = domains.copy()
        expected = supported_assignments(before, 5)
        armed = [False]
        class Interrupted(Exception):
            pass
        def check():
            if armed[0] and domains != before:
                raise Interrupted()
        pruner = Pruner(blank_problem(5), check, regions=[])
        armed[0] = True
        with self.assertRaises(Interrupted):
            pruner.prune(domains)
        self.assertNotEqual(domains, before)
        for order in expected:
            self.assertTrue(all(domain & (7 << (3*tile)) for domain, tile in zip(domains, order)))
        armed[0] = False
        pruner.prune(domains)
        self.assertEqual(domains[2:], [tile_domain((2, 3, 4))]*3)
        self.assertEqual(set(pruner.stats), set(STAT_KEYS))
        self.assertTrue(all(type(value) is int and value >= 0 for value in pruner.stats.values()))

    def test_constructor_and_prune_honor_checks_and_validate_domains(self):
        class Interrupted(Exception):
            pass
        def check():
            raise Interrupted()
        with self.assertRaises(Interrupted):
            Pruner(blank_problem(2), check)
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                Pruner(blank_problem(2), neighborhood_limit=limit)
        pruner = Pruner(blank_problem(2))
        for domains in ([1], [-1, 1], [1 << 6, 1], [True, 1]):
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                pruner.prune(domains)
        self.assertFalse(pruner.prune([0, 1]))




class ClusterBitsetTests(unittest.TestCase):
    def test_transpose_matches_independent_embedding_sets_after_narrowing_and_widening(self):
        rows = [[3, 1025, 1025], [reverse_mask(3), 1025, 1025],
                [5, 1025, 1025], [reverse_mask(5), 1025, 1025]]
        topology = board(4, [(a, side, a+1, side) for a in (0, 2) for side in range(3)])
        problem = SearchProblem(masks_to_tiles(rows), topology)
        helper = Pruner(problem, regions=[])
        self.assertEqual(len(helper.clusters), 2)
        self.assertTrue(any(len(embedding) < problem.n for cluster in helper.clusters for embedding in cluster[2]))
        rng = random.Random(710312)
        for case in range(400):
            initial = [bits(code for code in range(12) if rng.random() < (1 if case % 11 == 0 else .6))
                       for _ in range(4)]
            expected, calls = initial.copy(), []
            checks = failures = removed = 0
            for members, _, embeddings in helper.clusters:
                checks += 1
                viable = [dict(embedding) for embedding in embeddings
                          if all(expected[cell] & (1 << code) for cell, code in embedding)]
                if not viable:
                    calls.append((0, 0)); removed += expected[0].bit_count()
                    expected[0] = 0; failures = 1
                    break
                # Independent set enumeration: an unrelated tile is supported
                # only by an embedding that leaves this particular cell free.
                allowed = [bits(code for code in range(12) if any(
                    placement.get(cell) == code or (cell not in placement and code//3 not in members)
                    for placement in viable)) for cell in range(4)]
                for cell, keep in enumerate(allowed):
                    new = expected[cell] & keep
                    calls.append((cell, new)); removed += (expected[cell] ^ new).bit_count()
                    expected[cell] = new
                    if not new:
                        failures = 1
                        break
                if failures:
                    break
            actual, actual_calls = initial.copy(), []
            original = helper._narrow
            def record(domains, cell, allowed, kind, dependencies=None):
                actual_calls.append((cell, domains[cell] & allowed))
                return original(domains, cell, allowed, kind, dependencies)
            helper._narrow = record
            before = helper.stats.copy()
            try:
                changed = helper.prune_clusters(actual)
            finally:
                helper._narrow = original
            self.assertEqual(actual, expected)
            self.assertEqual(changed, initial != expected)
            self.assertEqual(actual_calls, calls)
            self.assertEqual(helper.stats['cluster_checks']-before['cluster_checks'], checks)
            self.assertEqual(helper.stats['cluster_failures']-before['cluster_failures'], failures)
            self.assertEqual(helper.stats['cluster_prunes']-before['cluster_prunes'], removed)
            self.assertTrue(all(len(cache) <= 8 for cache in helper._cluster_cache))

    def test_empty_embeddings_and_impossible_singletons_fail_without_caching(self):
        from unittest.mock import patch
        with patch.object(Pruner, '_build_clusters', return_value=((), (((0,), 7, ()),))):
            helper = Pruner(blank_problem(1), parity=False)
        domains = [1]
        self.assertTrue(helper.prune_clusters(domains))
        self.assertEqual(domains, [0]); self.assertEqual(helper._cluster_cache, [{}])
        self.assertEqual(helper.stats['cluster_failures'], 1)
        problem = unique_tetrahedron(); helper = Pruner(problem)
        reference = brute(problem, 'edges')[0]
        domains = [1 << code for code in reference]
        domains[0] = 1 << reference[1]
        self.assertTrue(helper.prune_clusters(domains))
        self.assertEqual(domains[0], 0)
        self.assertTrue(all(not cache for cache in helper._cluster_cache))

    def test_interrupted_support_construction_publishes_no_tables(self):
        from dataclasses import replace
        from unittest.mock import patch
        problem = unique_tetrahedron(); complete = Pruner(problem)
        for tables in (None, replace(complete.tables, cluster_supports=None)):
            partial = Pruner.__new__(Pruner)
            with patch.object(Pruner, '_build_cluster_supports', side_effect=InterruptedError('index build')):
                with self.assertRaises(InterruptedError):
                    partial.__init__(problem, tables=tables)
            self.assertFalse(hasattr(partial, 'tables'))
        rebuilt = Pruner(problem, tables=replace(complete.tables, cluster_supports=None))
        self.assertEqual(rebuilt.tables.cluster_supports, complete.tables.cluster_supports)
        self.assertIs(rebuilt._cluster_supports, rebuilt.tables.cluster_supports)

    def test_supports_are_immutable_shared_and_bounded_by_finite_embeddings(self):
        problem = unique_tetrahedron(); helper = Pruner(problem)
        warm = Pruner(problem, tables=helper.tables)
        self.assertIs(warm.tables, helper.tables)
        self.assertIs(warm._cluster_supports, helper._cluster_supports)
        self.assertIsNot(warm._cluster_cache, helper._cluster_cache)
        with self.assertRaises(TypeError):
            helper.tables.cluster_supports[0][1][0][1][0] = (0, 0)
        for (_, _, embeddings), (all_embeddings, rows) in zip(helper.clusters, helper._cluster_supports):
            self.assertEqual(all_embeddings, (1 << len(embeddings))-1)
            self.assertEqual(len(rows), problem.n)
            for absent, supports, full in rows:
                self.assertEqual(absent & ~all_embeddings, 0)
                self.assertEqual(full, bits(bit.bit_length()-1 for bit, _ in supports))
                self.assertTrue(all(mask & ~all_embeddings == 0 for _, mask in supports))
                self.assertLessEqual(len(supports), len(embeddings))

    def test_interrupted_query_does_not_cache_incomplete_supports(self):
        problem = unique_tetrahedron(); helper = Pruner(problem)
        initial = [tile_domain(range(4))]*4; domains = initial.copy(); calls = [0]
        def check():
            calls[0] += 1
            if calls[0] == 3:
                raise InterruptedError('during supports')
        helper.check = check
        with self.assertRaises(InterruptedError):
            helper.prune_clusters(domains)
        self.assertEqual(domains, initial)
        self.assertTrue(all(not cache for cache in helper._cluster_cache))
        helper.check = lambda: None
        expected = initial.copy()
        expected_changed = Pruner(problem).prune_clusters(expected)
        self.assertEqual(helper.prune_clusters(domains), expected_changed)
        self.assertEqual(domains, expected)


class ImmutablePrunerTableTests(unittest.TestCase):
    def test_warm_tables_keep_callbacks_stats_and_mutable_caches_private(self):
        from dataclasses import FrozenInstanceError
        from unittest.mock import patch
        problem=unique_tetrahedron()
        first=Pruner(problem,regions=[(0,1)],native='python')
        tables=first.tables
        with self.assertRaises(FrozenInstanceError):tables.bonds=()
        first.check=lambda:(_ for _ in ()).throw(AssertionError('stale check'))
        first.prune_hall=lambda domains:(_ for _ in ()).throw(AssertionError('stale wrapper'))
        first.on_narrow=lambda *args:(_ for _ in ()).throw(AssertionError('stale reason callback'))
        first._hall_cache[('bad',)]=()
        first.regions[0][1].clear()
        selected=[]
        with patch('constraint_pruning.select_hall',side_effect=lambda mode:(selected.append(mode) or None,'test')):
            warm=Pruner(problem,regions=[(0,1)],tables=tables,native='auto')
        self.assertEqual(selected,['auto']);self.assertIs(warm.tables,tables)
        self.assertIsNot(warm.stats,first.stats);self.assertEqual(warm._hall_cache,{})
        self.assertIsNone(warm._matching);self.assertIsNone(warm.on_narrow)
        self.assertTrue(warm.regions[0][1]);self.assertEqual(warm.stats['pruner_table_reuses'],1)
        self.assertEqual(warm.stats['cluster_embeddings'],0)
        domains=[tile_domain(range(4))]*4
        warm.prune(domains)
        self.assertNotIn(0,domains)

    def test_exact_binding_rejects_changed_masks_topology_and_regions(self):
        import copy
        problem=unique_tetrahedron();tables=Pruner(problem).tables
        for field in ('masks','neighbors','sides'):
            altered=copy.deepcopy(problem)
            getattr(altered,field)[0,0]^=1
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Pruner tables differ'):
                Pruner(altered,tables=tables)
        with self.assertRaisesRegex(ValueError,'Pruner tables differ'):
            Pruner(problem,tables=tables,regions=[(0,1)])
        with self.assertRaisesRegex(ValueError,'Pruner tables differ'):
            Pruner(problem,tables=tables,parity=False)
        with self.assertRaises(ValueError):Pruner(problem,tables={})
        # No search domain contributes to a puzzle-only construction table.
        problem.domains[0]=[0]
        self.assertIs(Pruner(problem,tables=tables).tables,tables)

    def test_cold_and_warm_filters_preserve_tiny_enumerated_solutions(self):
        problem=unique_tetrahedron();helper=Pruner(problem)
        helper.prune_neighborhood([tile_domain(range(4))]*4)
        tables=helper.tables;self.assertIsNotNone(tables.stars)
        legal=brute(problem,'edges');self.assertTrue(legal)
        rng=random.Random(83061)
        for case in range(24):
            domains=[bits(code for code in range(12) if rng.randrange(3)) for _ in range(4)]
            if case%2==0:
                witness=legal[case%len(legal)]
                domains=[value|(1<<code) for value,code in zip(domains,witness)]
            expected=[codes for codes in legal if all(value&(1<<code) for value,code in zip(domains,codes))]
            outputs=[]
            for cached in (None,tables):
                helper=Pruner(problem,tables=cached);narrowed=domains.copy()
                for _ in range(20):
                    if not helper.prune(narrowed) or 0 in narrowed:break
                else:self.fail('Tiny filtering did not reach a fixed point')
                for codes in expected:self.assertTrue(all(value&(1<<code) for value,code in zip(narrowed,codes)))
                outputs.append(narrowed)
            self.assertEqual(*outputs)

    def test_lazy_profiles_publish_only_after_completion_and_keep_per_pass_state_private(self):
        problem=unique_tetrahedron();helper=Pruner(problem);cold=helper.tables
        domains=[tile_domain(range(4))]*4
        helper.prune_neighborhood(domains)
        self.assertIsNone(cold.stars);self.assertIsNotNone(helper.tables.stars)
        self.assertIs(helper._stars,helper.tables.stars)
        warm=Pruner(problem,tables=helper.tables)
        self.assertIs(warm._stars,helper._stars)
        # Narrow, rotate, then widen: all per-pass domain supports are fresh.
        for current in ([1,8,64,512],[2,16,128,1024],[tile_domain(range(4))]*4):
            first=current.copy();second=current.copy()
            warm.prune_neighborhood(first);Pruner(problem).prune_neighborhood(second)
            self.assertEqual(first,second)
        with self.assertRaises(TypeError):helper.tables.stars[0][0][1][0]=0

    def test_stop_before_during_and_after_lazy_build_preserves_publication_boundary(self):
        problem=unique_tetrahedron()
        for stop_at in (1,3):
            helper=Pruner(problem);old=helper.tables;calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('lazy build stop')
            helper.check=check
            with self.assertRaises(InterruptedError):helper.prune_neighborhood([tile_domain(range(4))]*4)
            self.assertIs(helper.tables,old);self.assertIsNone(helper._stars)
        helper=Pruner(problem);old=helper.tables;original=helper._build_stars;stop=[False]
        def completed_then_stop():
            profiles=original();stop[0]=True;return profiles
        helper._build_stars=completed_then_stop
        helper.check=lambda:(_ for _ in ()).throw(InterruptedError('after build')) if stop[0] else None
        with self.assertRaises(InterruptedError):helper.prune_neighborhood([tile_domain(range(4))]*4)
        self.assertIsNot(helper.tables,old);self.assertIsNotNone(helper.tables.stars)
        rebound=Pruner(problem,tables=helper.tables)
        self.assertIs(rebound._stars,helper.tables.stars)

    def test_interrupted_build_never_exports_partial_tables(self):
        from unittest.mock import patch
        problem=unique_tetrahedron();partial=Pruner.__new__(Pruner)
        with patch.object(Pruner,'_build_clusters',side_effect=InterruptedError('stop')):
            with self.assertRaises(InterruptedError):partial.__init__(problem)
        self.assertFalse(hasattr(partial,'tables'))
        complete=Pruner(problem)
        checks=[0]
        def stop():
            checks[0]+=1
            if checks[0]==3:raise InterruptedError('warm validation stop')
        with self.assertRaises(InterruptedError):Pruner(problem,stop,tables=complete.tables)
        self.assertEqual(Pruner(problem,tables=complete.tables).tables,complete.tables)


if __name__ == '__main__':
    unittest.main()
