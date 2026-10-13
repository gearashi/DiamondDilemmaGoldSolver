"""Independent graph traversal checks for cached partial Gold loop detection."""
from itertools import product
import random
import unittest

from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import simple_tiles, multiple_loops, double_face, tetrahedron_tiles, board


def graph_components(problem, codes):
    # Independent adjacency-list traversal: no production tables or DSU.
    endpoints, graph, owners = {}, {}, {}
    for cell, code in enumerate(codes):
        if code < 0:
            continue
        tile, rotation = divmod(code, 3)
        for index, segment in enumerate(problem.tiles[tile]['segments']):
            node = (cell, index)
            graph[node], owners[node] = [], cell
            for side, position in segment:
                endpoints[cell, (side+rotation)%3, position] = node
    for a, sa, b, sb in problem.edges:
        for position in range(1,12):
            left, right = endpoints.get((a,sa,position)), endpoints.get((b,sb,12-position))
            if left is not None and right is not None:
                graph[left].append(right); graph[right].append(left)
    seen, result = set(), []
    for node in graph:
        if node in seen:
            continue
        pending, component = [node], []
        seen.add(node)
        while pending:
            current = pending.pop(); component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor); pending.append(neighbor)
        if all(len(graph[item]) == 2 for item in component):
            result.append((len(component), sorted({owners[item] for item in component})))
    return result


class CachedConnectivityTests(unittest.TestCase):
    def test_mask_reversal_lookup_matches_exact_low_bit_convention(self):
        import numpy as np
        from validator import reverse_mask
        for value in range(2048):
            expected=int(format(value,'011b')[::-1],2)
            self.assertEqual(reverse_mask(value),expected)
            self.assertEqual(reverse_mask(np.uint16(value)),expected)
            self.assertEqual(reverse_mask(value+2048),expected)
            self.assertEqual(reverse_mask(value-2048),expected)
            self.assertEqual(reverse_mask(reverse_mask(value)),value)

    def test_all_tiny_partial_assignments_agree_with_graph_traversal(self):
        tetra, geometry = tetrahedron_tiles()
        for problem in (SearchProblem(simple_tiles(), double_face()),
                        SearchProblem(multiple_loops(), double_face()),
                        SearchProblem(tetra, geometry)):
            # Repeated physical IDs are included: this graph helper must not
            # conflate cell-local segment identities, even on an invalid board.
            for codes in product(range(-1, 3*problem.n), repeat=problem.n):
                self.assertEqual(problem._closed_components(codes), graph_components(problem,codes))

    def test_seeded_larger_partial_and_mismatched_boards(self):
        rng = random.Random(711)
        for size in (1,2,4):
            data, geometry, reference = planted_gold(197,size)
            problem = SearchProblem(data,geometry)
            for trial in range(80):
                if trial%2:
                    codes=reference.copy()
                else:
                    pieces=list(range(problem.n));rng.shuffle(pieces)
                    codes=[3*piece+rng.randrange(3) for piece in pieces]
                codes=[code if rng.random()<.72 else -1 for code in codes]
                expected=graph_components(problem,codes)
                self.assertEqual(problem._closed_components(codes),expected)
                self.assertEqual(problem.partial_loop_impossible(codes),any(size<problem.total_segments for size,_ in expected))
                proper=next((cells for size,cells in expected if size<problem.total_segments),None)
                cut=[(cell,codes[cell]) for cell in proper] if proper is not None else [(cell,code) for cell,code in enumerate(codes) if code>=0]
                self.assertEqual(problem.gold_nogood(codes),cut)

    def test_cached_results_cannot_be_changed_by_a_caller(self):
        problem=SearchProblem(multiple_loops(),double_face())
        first=problem._closed_components([0,3]);expected=graph_components(problem,[0,3])
        first[0][1].clear();first.clear()
        self.assertEqual(problem._closed_components([0,3]),expected)
        self.assertEqual(problem._closed_components([-1,3]),[])
        self.assertEqual(problem._closed_components([0,3]),expected)

    def test_seam_cache_eviction_does_not_change_answers(self):
        problem=SearchProblem(multiple_loops(),double_face())
        problem._seam_cache_limit=3
        for codes in product(range(-1,6),repeat=2):
            self.assertEqual(problem._closed_components(codes),graph_components(problem,codes))
            self.assertLessEqual(len(problem._seam_joins),3)

    def test_boundaries_and_invalid_codes(self):
        tiles=[{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(2)]
        problem=SearchProblem(tiles,board(2,[(0,0,1,0)]))
        self.assertEqual(problem._closed_components([0,3]),[(2,[0,1])])
        for codes in ([],[0],[0,True],[0,6],[-2,3],[0,3.0]):
            with self.assertRaises(ValueError):problem._closed_components(codes)


if __name__=='__main__':unittest.main()
