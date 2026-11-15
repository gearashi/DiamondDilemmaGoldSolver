"""Compare conditional mate support to each vertex-labelled perfect matching."""
from itertools import combinations
from random import Random
from importlib.util import find_spec
import unittest

from r35_mate_correlations import conditional_matches


def direct(nodes,edges):
    if not nodes:
        yield frozenset();return
    u=min(nodes)
    for v in nodes-{u}:
        if (u,v) not in edges:continue
        for row in direct(nodes-{u,v},edges):yield row|{(u,v)}


class ConditionalMatchesTests(unittest.TestCase):
    def test_every_conditional_support_on_sparse_dense_bipartite_or_odd_cycles(self):
        rng=Random(35)
        for n in (4,6,8,10):
            nodes=set(range(n))
            for seed in range(35):
                edges={e for e in combinations(range(n),2) if rng.random()<.47 and (seed%3 or e[0]%2!=e[1]%2)}
                sols=list(direct(nodes,edges));table=conditional_matches(nodes,edges)
                self.assertTrue(table['complete'])
                self.assertEqual(bool(sols),table['feasible'])
                if not sols:continue
                must=set.intersection(*map(set,sols));possible=set.union(*map(set,sols))
                for pair in possible-must:
                    active=[s for s in sols if pair in s]
                    together=set.intersection(*map(set,active))-{pair}-must
                    absent={e for e in possible-set.union(*map(set,active)) if not set(e)&set(pair)}
                    self.assertEqual(set(table['bundles'].get(pair,())),together)
                    self.assertEqual(set(table['incompatible'].get(pair,())),absent)
        limited=conditional_matches(range(6),set(combinations(range(6),2)),max_states=1)
        self.assertFalse(limited['complete'])
        self.assertNotIn('bundles',limited)

    @unittest.skipUnless(find_spec('ortools'),'geographic CP search requires the Python 3.12 profile')
    def test_public_option_under_sparse_and_projected_matching(self):
        from r27_structure_fixtures import planted
        from r30_gold import search
        p,ds=planted(1,3,71)
        for projection in (False,True):
            out=search(p,domains=ds,geography=True,project_blanks=projection,mate_correlations=True,
                       max_rounds=48,seconds=8)
            self.assertEqual(out['status'],'GOLD')
            self.assertTrue(p.validate(out['codes'])['valid'])
        with self.assertRaises(ValueError):search(p,matching=False,mate_correlations=True)

if __name__=='__main__':unittest.main()
