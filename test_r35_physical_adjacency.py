"""Independent controls for target-mesh tile adjacency propagation."""
from collections import Counter
from itertools import combinations, product
from random import Random
from types import SimpleNamespace
import unittest

from r27_structure_filters import Work
from r27_structure_fixtures import planted
from r35_physical_adjacency import _local_supported, filter_neighbors, three_connected
from test_r28_geometry import matchings


def neighbor_options(n, graph):
    return {(a,b) for a,b in combinations(range(3*n),2) if b//3 in graph[a//3]}


def separator_oracle(graph):
    for size in (0,1,2):
        for removed in combinations(range(len(graph)),size):
            rest=set(range(len(graph)))-set(removed)
            reached={next(iter(rest))}
            frontier=list(reached)
            while frontier:
                v=frontier.pop()
                for w in (graph[v]&rest)-reached:
                    reached.add(w)
                    frontier.append(w)
            if reached!=rest:
                return False
    return True


class AdjacencyTests(unittest.TestCase):
    def test_owner_support_by_exhausting_neighbor_triples(self):
        for left,right in product(range(16),repeat=2):
            rows=[{0},{i for i in range(4) if left>>i&1},{i for i in range(4) if right>>i&1}]
            for candidate,limit in product(range(4),(1,2,3)):
                feasible=any(max(Counter((candidate,a,b)).values())<=limit for a,b in product(rows[1],rows[2]))
                self.assertEqual(_local_supported(rows,0,3*candidate,limit),feasible)

    def test_all_small_graphs_by_independent_two_vertex_deletions(self):
        edges=list(combinations(range(5),2))
        for mask in range(1<<len(edges)):
            graph=[set() for _ in range(5)]
            for i,(a,b) in enumerate(edges):
                if mask>>i&1:
                    graph[a].add(b)
                    graph[b].add(a)
            self.assertEqual(three_connected(graph,Work(100000)),separator_oracle(graph))

    def test_complete_and_partial_tetrahedron_sewings(self):
        tetra=[set(range(4))-{i} for i in range(4)]
        p=SimpleNamespace(n=4,neighbors=tetra,edges=[(a,0,b,0) for a,b in combinations(range(4),2)])
        options=neighbor_options(4,tetra)
        valid=[]
        for sewing in matchings(tuple(range(12))):
            selected=set(sewing)
            if not selected<=options:
                continue
            multiplicity=Counter((a//3,b//3) for a,b in selected)
            accepted=all(count==1 for count in multiplicity.values())
            self.assertEqual(bool(filter_neighbors(p,selected,selected)[0]),accepted)
            if accepted:
                valid.append(selected)
        self.assertEqual(len(valid),6**4)
        rng=Random(35)
        for _ in range(80):
            sample=rng.choice(valid)
            restricted=sample|{edge for edge in options if rng.random()<.12}
            required=set(rng.sample(sorted(sample),2))
            attainable=set().union(*(s for s in valid if required<=s<=restricted))
            supported,_=filter_neighbors(p,restricted,required,connect=True)
            self.assertLessEqual(attainable,supported)

    def test_separator_beyond_distinct_neighbors_and_unknown(self):
        p,_=planted(1,3,41)
        sides=[{0,1,2,3,4,5},{0,1,6,7,8,9}]
        graph=[set.union(*(group for group in sides if a in group))-{a} for a in range(10)]
        options=neighbor_options(10,graph)
        self.assertEqual(filter_neighbors(p,options,())[0],options)
        self.assertEqual(filter_neighbors(p,options,(),connect=True)[1].get('tile_separator'),1)
        live,stats=filter_neighbors(p,options,(),connect=True,work_limit=1)
        self.assertEqual(live,options)
        self.assertEqual(stats.get('separator_unknown'),1)
        weak=SimpleNamespace(n=10,neighbors=graph,edges=[(a,0,b,0) for a in range(10) for b in graph[a] if a<b])
        self.assertEqual(filter_neighbors(weak,options,(),connect=True)[0],options)

    def test_multiple_seams_and_open_mesh(self):
        sewing={(0,3),(1,4),(2,5)}
        p=SimpleNamespace(n=2,neighbors=((1,1,1),(0,0,0)),edges=[(0,s,1,s) for s in range(3)])
        self.assertEqual(filter_neighbors(p,sewing,sewing,connect=True)[0],sewing)
        p.neighbors=((1,1,-1),(0,0,-1))
        self.assertEqual(filter_neighbors(p,sewing,sewing)[1],{'applicable':False})


if __name__=='__main__':
    unittest.main()
