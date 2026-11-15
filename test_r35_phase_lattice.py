"""Exhaust binary assignments and replay integer modular witnesses independently."""
from itertools import product
from random import Random
from types import SimpleNamespace
import unittest

from r35_phase_lattice import PhaseColumns,price,verify


class PhaseLatticeTests(unittest.TestCase):
    def test_higher_modulus_and_mod_two_slack(self):
        interface=SimpleNamespace(rows=[((1,0),(5,0)),((0,0),(2,0))],mod=[False,False],euler={0})
        cols=PhaseColumns(interface,relations=[(0,1,1)])
        outcome=price(cols)
        self.assertEqual(outcome['status'],'INFEASIBLE')
        self.assertEqual(outcome['proofs'][0]['separator']['modulus'],2)
        wide=PhaseColumns(SimpleNamespace(rows=[((2,0),(6,0))],mod=[False,False],euler={0}))
        outcome=price(wide)
        self.assertEqual(outcome['proofs'][0]['separator']['modulus'],4)
        joint=PhaseColumns(SimpleNamespace(rows=[((0,-1,0),(1,0,0)),((0,0,0),(1,-2,0))],
                                           mod=[False,False,False],euler={0}))
        proof=price(joint)['proofs'][0]
        self.assertEqual(proof['separator']['modulus'],3)
        self.assertTrue(verify(joint.vectors(),proof['target'],proof['separator']))
        wide.mod[0]=True
        self.assertEqual(price(wide)['status'],'PASSES')
        self.assertFalse(verify([{0:4}],[-2],{'weights':[[0,1]],'modulus':2}))

    def test_all_assignments_small_varied_incidence_and_pins(self):
        rng=Random(35);rejects=0
        for _ in range(300):
            n=rng.randrange(2,8)
            rows=[tuple(tuple(rng.randint(-2,2) for k in range(4)) for z in (0,1)) for t in range(n)]
            mod=[True,False,rng.choice([False,True]),False]
            goal=set(rng.sample(range(-4,5),3))
            interface=SimpleNamespace(rows=rows,mod=mod,euler=goal)
            rel=[(rng.randrange(n),rng.randrange(n),rng.randrange(2)) for _ in range(rng.randrange(4))]
            pins={0:rng.randrange(2)} if rng.randrange(2) else {}
            cols=PhaseColumns(interface,rel,pins)
            feasible=False
            for phases in product((0,1),repeat=n):
                if any(phases[t]!=z for t,z in pins.items()) or any(phases[u]^phases[v]!=z for u,v,z in rel):continue
                counts=[sum(rows[t][z][j] for t,z in enumerate(phases)) for j in range(4)]
                if counts[-1] in goal and all((v%2 if m else v)==0 for v,m in zip(counts[:-1],mod)):
                    feasible=True;break
            result=price(cols)
            if result['status'] in ('INFEASIBLE','XOR_CONFLICT'):
                self.assertFalse(feasible)
                rejects+=1
                for row in result.get('proofs',[]):
                    self.assertTrue(verify(cols.vectors(),row['target'],row['separator']))
        self.assertGreater(rejects,100)


if __name__=='__main__':
    unittest.main()
