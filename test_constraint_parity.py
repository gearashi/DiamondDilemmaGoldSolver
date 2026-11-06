"""Finite preservation tests for the sphere parity propagator."""
import random
import unittest
from constraint_parity import ParityPruner
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, double_face

class ParityTests(unittest.TestCase):
    def test_actual_odd_length_gold_loop_survives(self):
        from geometry import build_board
        board = build_board(1)
        # A five-cycle around the upper apex, joined to a doubled path through
        # the lower cells: one explicit Euler walk with 15 seam crossings.
        walk = [0,1,2,3,4,0,5,6,7,8,9,8,7,6,5,0]
        occurrences, links = {}, []
        for a,b in zip(walk,walk[1:]):
            sa = board.neighbor_cells[a].index(b)
            sb = board.neighbor_cells[b].index(a)
            key = tuple(sorted((a,b)))
            p = (3,7)[occurrences.get(key,0)]
            occurrences[key] = occurrences.get(key,0)+1
            pa = p if a < b else 12-p
            links.append(((a,sa,pa),(b,sb,12-pa)))
        tiles = [{'id':str(cell),'segments':[]} for cell in range(10)]
        for i, link in enumerate(links):
            cell,side,p = link[1]
            following = links[(i+1)%len(links)][0]
            self.assertEqual(following[0],cell)
            tiles[cell]['segments'].append([[side,p],list(following[1:])])
        problem = SearchProblem(tiles,board)
        reference = [3*cell for cell in range(10)]
        self.assertEqual(problem.total_segments,15)
        self.assertTrue(problem.validate(reference)['valid'])
        pruner = ParityPruner(problem)
        domains = [1 << code for code in reference]
        self.assertFalse(pruner.prune(domains))
        self.assertEqual(domains,[1 << code for code in reference])

    def test_custom_topology_is_disabled(self):
        problem = SearchProblem(simple_tiles(), double_face())
        pruner = ParityPruner(problem)
        self.assertFalse(pruner.enabled)
        domains = [sum(1 << code for code in row) for row in problem.domains]
        before = domains.copy()
        self.assertFalse(pruner.prune(domains))
        self.assertEqual(domains, before)

    def test_valid_gold_certificates_survive_random_domain_restrictions(self):
        for size in (1, 2, 4):
            for seed in (21, 319):
                data, board, reference = planted_gold(seed, size)
                problem = SearchProblem(data, board)
                pruner = ParityPruner(problem)
                self.assertTrue(pruner.enabled)
                self.assertTrue(problem.validate(reference)['valid'])
                rng = random.Random(seed)
                for trial in range(12):
                    domains = []
                    for cell, row in enumerate(problem.domains):
                        kept = {reference[cell]}
                        kept.update(rng.sample(row, min(len(row), trial)))
                        domains.append(sum(1 << code for code in kept))
                    while pruner.prune(domains):
                        self.assertTrue(all(domains))
                    self.assertTrue(all(domains[cell] & (1 << code) for cell, code in enumerate(reference)))

    def test_impossible_odd_crossing_total_is_detected_on_even_edge_domains(self):
        # Real tiles each have one path with two odd sides. Restricting them to
        # all-even oriented masks in a controlled fake mask table makes the
        # apex equation contradictory. This isolates the propagator invariant.
        data, board, reference = planted_gold(41, 1)
        problem = SearchProblem(data, board)
        problem.total_segments = 1
        problem.masks[:] = 0
        pruner = ParityPruner(problem)
        domains = [sum(1 << code for code in row) for row in problem.domains]
        self.assertTrue(pruner.prune(domains))
        self.assertIn(0, domains)
        self.assertEqual(pruner.stats['parity_failures'], 1)

    def test_forced_vertex_relations_remove_incompatible_rotations(self):
        data, board, reference = planted_gold(77, 1)
        problem = SearchProblem(data, board)
        # The fixed reference is all-even at every seam. A second orientation
        # with two odd sides is incompatible with that connected zero parity.
        cell = 0
        wrong = 3 * (reference[cell] // 3) + (reference[cell] + 1) % 3
        problem.masks[wrong, 0] = 1
        problem.masks[wrong, 1] = 1
        problem.masks[wrong, 2] = 0
        pruner = ParityPruner(problem)
        domains = [1 << code for code in reference]
        domains[cell] |= 1 << wrong
        self.assertTrue(pruner.prune(domains))
        self.assertEqual(domains[cell], 1 << reference[cell])

    def test_interruptions_only_leave_sound_deletions(self):
        data, board, reference = planted_gold(71, 1)
        problem = SearchProblem(data, board)
        domains = [1 << code for code in reference]
        before = domains.copy()
        def stop(): raise InterruptedError('fixture')
        with self.assertRaises(InterruptedError):
            ParityPruner(problem).prune(domains, stop)
        self.assertEqual(domains, before)



class ImmutableParityTableTests(unittest.TestCase):
    def test_warm_parity_rebuilds_private_stats_and_validates_all_inputs(self):
        import copy
        data,geometry,codes=planted_gold(71,1)
        problem=SearchProblem(data,geometry);first=ParityPruner(problem)
        self.assertTrue(first.enabled)
        first.stats['parity_checks']=99
        first.prune=lambda *args:(_ for _ in ()).throw(AssertionError('old wrapper'))
        warm=ParityPruner(problem,tables=first.tables)
        self.assertEqual(warm.stats['parity_checks'],0)
        domains=[1<<code for code in codes];self.assertFalse(warm.prune(domains))
        self.assertEqual(domains,[1<<code for code in codes])
        for field in ('masks','edges','total_segments','board'):
            altered=copy.deepcopy(problem)
            if field=='masks':altered.masks[0,0]^=1
            elif field=='edges':altered.edges=altered.edges[:-1]
            elif field=='total_segments':altered.total_segments+=1
            else:
                from dataclasses import replace
                altered.board=replace(altered.board,subdivision=2)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Parity tables differ'):
                ParityPruner(altered,tables=first.tables)

    def test_disabled_custom_parity_can_reuse_only_disabled_tables(self):
        problem=SearchProblem(simple_tiles(),double_face());cold=ParityPruner(problem)
        self.assertFalse(cold.enabled)
        warm=ParityPruner(problem,tables=cold.tables)
        self.assertFalse(warm.enabled)
        self.assertFalse(warm.prune([7,56]))
        data,geometry,_=planted_gold(71,1)
        with self.assertRaises(ValueError):ParityPruner(SearchProblem(data,geometry),tables=cold.tables)
        # A malformed/custom Board disables parity before requiring search
        # arrays, preserving the old constructor's lightweight mock behavior.
        from dataclasses import replace
        from types import SimpleNamespace
        custom=SimpleNamespace(board=replace(geometry,subdivision=2))
        disabled=ParityPruner(custom)
        self.assertFalse(disabled.enabled)
        self.assertFalse(ParityPruner(custom,tables=disabled.tables).enabled)


if __name__ == '__main__': unittest.main()
