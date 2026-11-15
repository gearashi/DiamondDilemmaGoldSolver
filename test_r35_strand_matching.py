"""Independent exhaustive matching, contraction and Tutte-obstruction controls."""
from functools import lru_cache
from itertools import combinations
from random import Random
import unittest

from r27_structure_filters import Work,separator_ok,strand_graph
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r32_gold_bonds import inventory
from bench_r35_strand_oracle import check_sewing,matching_condition,physical_graph,verify_obstruction


def brute(graph):
    @lru_cache(None)
    def pair(vertices):
        if not vertices:
            return True
        a=vertices[0]
        return any(pair(tuple(v for v in vertices[1:] if v!=b)) for b in vertices[1:] if b in graph[a])
    if len(graph)%2==0:
        return pair(tuple(range(len(graph))))
    return all(pair(tuple(v for v in range(len(graph)) if v!=i)) for i in range(len(graph)))


class StrandTests(unittest.TestCase):
    def test_all_graphs_through_six_and_held_out_sparse_even_odd(self):
        for n in (3,4,5,6,7,8,9):
            edges=list(combinations(range(n),2))
            rng=Random(n)
            masks=range(1<<len(edges)) if n<=6 else [rng.getrandbits(len(edges)) for _ in range(700)]
            for mask in masks:
                g=[set() for _ in range(n)]
                for j,(u,v) in enumerate(edges):
                    if mask>>j&1:
                        g[u].add(v);g[v].add(u)
                certificate,result=matching_condition(g,Work(100000))
                expected=brute(g)
                self.assertEqual(result=='passes',expected,(n,mask))
                if certificate is not None:
                    self.assertTrue(verify_obstruction(g,certificate))

    def test_odd_even_obstructions_survive_published_separator_and_limits(self):
        for m in (4,5):
            n=3+m
            g=[set(range(3,n)) if i<3 else set(range(3)) for i in range(n)]
            self.assertTrue(separator_ok(g,Work(100000)))
            certificate,result=matching_condition(g,Work(100000))
            self.assertEqual(result,'obstruction')
            self.assertTrue(verify_obstruction(g,certificate))
            self.assertFalse(verify_obstruction(g,{'vertex':certificate['vertex'],'barrier':list(range(n))}))
            self.assertEqual(matching_condition(g,Work(1)),(None,'unknown'))
        self.assertEqual(matching_condition([set(),set()],Work(100)),(None,'trivial'))

    def test_planted_single_loop_and_contracted_open_port_graph(self):
        for seed in (7,35,103):
            p,_=planted(1,3,seed)
            rules=inventory(p)
            rows=[set() for _ in range(3*p.n)]
            for a,b in rules.possible:
                rows[a].add(b);rows[b].add(a)
            g,proper=physical_graph(p,rules.possible,[])
            self.assertFalse(proper)
            self.assertEqual(g,[row-{i} for i,row in enumerate(strand_graph(p,rows))])
            seams=physical_bonds(p,[3*t for t in range(p.n)])
            rng=Random(seed)
            rng.shuffle(seams)
            for i in range(len(seams)+1):
                record=check_sewing(p,rules.possible,seams[:i],Work(100000))
                self.assertIn(record['result'],('passes','trivial'))
            self.assertEqual(record['vertices'],1)
        q,_=planted(1,3,35,split=True)
        complete=physical_bonds(q,[3*t for t in range(q.n)])
        self.assertTrue(physical_graph(q,complete,complete)[1])


if __name__=='__main__':
    unittest.main()
