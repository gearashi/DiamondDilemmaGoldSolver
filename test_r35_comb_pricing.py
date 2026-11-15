"""Exhaustive two-strand comb prices and independent Hamiltonian inequality checks."""
from collections import Counter
from importlib.util import find_spec
from itertools import combinations,permutations
from random import Random
import unittest

from r35_comb_pricing import candidate_handles,price_teeth,rows,larger_rows


def prism(size):
    weights={}
    for offset in (0,size):
        for u in range(size):
            weights[tuple(sorted((offset+u,offset+(u+1)%size)))]=.5
    weights.update({(u,u+size):1 for u in range(size)})
    return weights


def direct(weights,part):
    return sum(w for (u,v),w in weights.items() if (u in part)!=(v in part))


@unittest.skipUnless(find_spec("ortools"),"requires OR-Tools")
class CombTests(unittest.TestCase):
    def test_every_fixed_handle_cardinality_against_enumeration(self):
        rng=Random(35)
        for size in (6,7,8,9,10):
            for _ in range(24):
                weights={e:rng.randrange(1,15)/10 for e in combinations(range(size),2)}
                h=set(rng.sample(range(size),size//2))
                boundary=2.01
                expected={}
                cross=[e for e in weights if (e[0] in h)!=(e[1] in h)]
                for t in range(3,min(len(h),size-len(h))+1,2):
                    score=max(sum(2*weights[e]-1 for e in teeth) for teeth in combinations(cross,t)
                              if len({v for edge in teeth for v in edge})==2*t)
                    if score+1>boundary:
                        expected[t]=round(score,7)
                actual={len(teeth):round(sum(2*weights[e]-1 for e in teeth),7)
                        for teeth in price_teeth(h,weights,boundary,Counter())}
                self.assertEqual(expected,actual)

    def test_odd_five_tooth_prism_survives_all_smaller_pricing(self):
        n=10;w=prism(5)
        best=Counter()
        allrows=[]
        for mask in range(1,(1<<n)-1):
            h={v for v in range(n) if mask>>v&1}
            boundary=direct(w,h)
            self.assertGreaterEqual(boundary,2)
            for teeth in price_teeth(h,w,boundary,Counter()):
                lhs=boundary+sum(direct(w,set(e)) for e in teeth)
                self.assertLess(lhs,3*len(teeth)+1)
                best[len(teeth)]+=1
                allrows.append((h,teeth))
        self.assertEqual(best,{5:2})
        self.assertTrue(list(rows(n,w,[set(range(5))],Counter())))
        larger=list(larger_rows(n,w,[set(range(5))],Counter()))
        self.assertTrue(larger)
        for h,teeth in larger:
            self.assertLess(direct(w,h)+sum(direct(w,set(t)) for t in teeth),3*len(teeth)+1)
        self.assertTrue(all(len(t)==5 for _,t in allrows))
        h,teeth=allrows[0]
        coeff={e:direct({e:1},h)+sum(direct({e:1},set(t)) for t in teeth)
               for e in combinations(range(n),2)}
        for cycle in permutations(range(1,n)):
            value=sum(coeff[min(a,b),max(a,b)] for a,b in zip((0,)+cycle,cycle+(0,)))
            self.assertGreaterEqual(value,16)

    def test_hamiltonian_inequality_materialization_every_small_tour(self):
        h={0,1,2}
        teeth=((0,3),(1,4),(2,5))
        for cycle in permutations(range(1,7)):
            weights={tuple(sorted(edge)):1 for edge in zip((0,)+cycle,cycle+(0,))}
            self.assertGreaterEqual(direct(weights,h)+sum(direct(weights,set(e)) for e in teeth),10)
        self.assertTrue(list(rows(6,prism(3),[h],Counter())))
        self.assertEqual(candidate_handles(6,{},maximum=0),[])


if __name__=='__main__':
    unittest.main()
