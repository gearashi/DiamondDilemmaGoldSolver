"""Check labelled projections and multiplicities against all binary phase assignments."""
from collections import Counter
from itertools import product
from random import Random
from types import SimpleNamespace
import unittest

from r35_phase_frontier import compile_projection


def enumerate_assignments(interface,pins,boundary,relations,euler):
    counts=Counter()
    for phases in product((0,1),repeat=len(interface.rows)):
        if any(phases[t]!=v for t,v in pins.items()):continue
        if any(phases[u]^phases[v]!=v0 for u,v,v0 in relations):continue
        totals=[sum(interface.rows[t][z][j] for t,z in enumerate(phases)) for j in range(len(interface.mod))]
        if any((v%2 if mod else v) for mod,v in zip(interface.mod,totals[:-1])):continue
        if euler and totals[-1] not in interface.euler:continue
        counts[sum(phases[t]<<i for i,t in enumerate(boundary))]+=1
    return counts


class PhaseFrontierTests(unittest.TestCase):
    def test_exhaustive_random_integer_and_parity_signatures(self):
        rng=Random(235)
        for _ in range(400):
            n=rng.randrange(2,9);dims=rng.randrange(1,6)
            rows=[tuple(tuple(rng.randrange(-1,2) for j in range(dims)) for z in (0,1)) for t in range(n)]
            x=SimpleNamespace(rows=rows,mod=[rng.randrange(2) for j in range(dims-1)]+[False],
                              euler=set(rng.sample(range(-4,5),3)))
            pins={t:rng.randrange(2) for t in rng.sample(range(n),rng.randrange(3))}
            boundary=tuple(rng.sample(range(n),rng.randrange(n+1)))
            rel=[(rng.randrange(n),rng.randrange(n),rng.randrange(2)) for k in range(rng.randrange(5))]
            euler=bool(rng.randrange(2))
            want=enumerate_assignments(x,pins,boundary,rel,euler)
            for split in (False,True):
                got=compile_projection(x,pins,boundary,relations=rel,euler=euler,multiplicities=True,split=split)
                self.assertEqual(got['status'],'COMPLETE')
                self.assertEqual(dict(want),got['counts'])

    def test_interchangeable_reversed_and_zero_columns_and_budget(self):
        x=SimpleNamespace(rows=[((0,0),(1,1))]*4+[((1,1),(0,0))]*4+[((-4,0),(-4,0))],
                          mod=[False,False],euler={4})
        bound=(3,6)
        out=compile_projection(x,{},bound,multiplicities=True)
        self.assertEqual(out['counts'],enumerate_assignments(x,{},bound,[],True))
        self.assertGreater(out['interchangeable_internal'],0)
        for budget in [dict(max_states=1),dict(max_transitions=1)]:
            fit=compile_projection(x,{},bound,**budget)
            self.assertEqual(fit['status'],'UNKNOWN')
            self.assertNotIn('patterns',fit)

if __name__=='__main__':unittest.main()
