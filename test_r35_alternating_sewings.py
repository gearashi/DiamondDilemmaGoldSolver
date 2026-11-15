"""Direct enumeration of whole alternating cycles with phase and geographic controls."""
from importlib.util import find_spec
from itertools import product
import unittest

from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r35_alternating_sewings import Family,common_phases
from bench_r35_alternating_sewings import partner,oracle
from bench_r35_physical_tour import family


@unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
class AlternatingTests(unittest.TestCase):
    def test_long_cycles_and_every_offspring(self):
        for seed in (35,71,103,157,251):
            p,_=planted(1,10,seed,uniform_ports=True)
            phase,_,_,true=family(p)
            a=dict(phases=phase,pairs=sorted(true))
            b=partner(p,a,(3,4,3),seed)
            common,state=common_phases(p,a,b)
            self.assertEqual(state,'COMPATIBLE')
            f=Family(p,a,b,common)
            expected=oracle(p,f,1<<len(f.groups))
            for memo in (False,True):
                r=f.solve(memo=memo)
                self.assertTrue(r['complete'])
                self.assertEqual(r['connected_nonblank'],expected)
            union={tuple(edge) for parent in (a,b) for edge in parent['pairs']}
            direct=set()
            def pair(left,pairs):
                if not left:
                    direct.add(frozenset(pairs));return
                x=min(left)
                for y in left-{x}:
                    edge=min(x,y),max(x,y)
                    if edge in union:pair(left-{x,y},pairs+[edge])
            pair(set(range(3*p.n)),[])
            represented={frozenset(f.pairs(x,y)) for x,y in product(range(1<<len(f.groups)),range(1<<len(f.blanks)))}
            self.assertEqual(direct,represented)
            golden=f.solve(geography=True,domains=tuple(1<<(3*i) for i in range(p.n)))
            self.assertTrue(p.validate(golden['gold'])['valid'])
            self.assertFalse(f.solve(max_nodes=0)['complete'])

    def test_rejects_wrong_phase_or_nonreciprocal_parent(self):
        p,d=planted(1,6,35,uniform_ports=True)
        phases,_,_,true=family(p)
        a=dict(phases=phases,pairs=sorted(true))
        b=partner(p,a,(3,))
        malformed=dict(b,pairs=b['pairs'][:-1])
        with self.assertRaises(ValueError):common_phases(p,a,malformed)
        corrupt=phases.copy();corrupt[a['pairs'][0][0]//3]^=1
        with self.assertRaises(ValueError):Family(p,a,b,corrupt)
        complementary=dict(a,phases=[1-v for v in phases])
        common,status=common_phases(p,a,complementary)
        self.assertEqual(status,'COMPATIBLE')
        self.assertTrue(p.validate(Family(p,a,complementary,common).solve(geography=True)['gold'])['valid'])


if __name__=='__main__':
    unittest.main()
