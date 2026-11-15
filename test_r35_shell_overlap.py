from collections import Counter
from itertools import permutations,combinations
from math import comb
import unittest
import numpy as np

from r35_shell_overlap import can_meet,cycle_type,distance,intersection,patterns


class ShellTests(unittest.TestCase):
    def test_every_small_class_pair_including_unequal_sizes_and_fixed_extra_labels(self):
        for n in range(3,8):
            perms=list(permutations(range(n)))
            representatives={cycle_type(p):p for p in perms}
            for delta,rel in representatives.items():
                inv=np.argsort(rel)
                reference=Counter((cycle_type(p),cycle_type(inv[list(p)])) for p in perms)
                for a in representatives:
                    if not a:continue
                    for b in representatives:
                        if b:self.assertEqual(intersection(delta,a,b,n),reference[a,b],(n,delta,a,b))
        # A third eligible cell can connect two otherwise disjoint pair choices.
        self.assertEqual(intersection((3,),(2,),(2,),3),3)
        self.assertEqual(intersection((3,),(3,),(3,),3),1)
        self.assertGreater(intersection((3,),(3,),(3,),6),1)

    def test_stabilizer_weights_and_separation_beyond_parity_or_distance(self):
        for delta in ((2,2,3),(3,3,3),(2,2,2,2,2),(2,5),(7,)):
            for k in range(2,8):
                sums=Counter()
                for subset,weight in patterns(delta,k,100):sums[len(subset)]+=weight
                self.assertEqual(sums,Counter({i:comb(sum(delta),i) for i in range(min(k,sum(delta))+1)}))
        # Enumerate a class-specific gap for which both old comparisons pass.
        chosen=[(d,a,b) for d in ((5,),(2,4),(3,3),(2,2),(2,2,2,2))
                for a in ((2,2),(3,),(4,),(2,4)) for b in ((2,),(3,),(2,2),(4,))
                if can_meet(d,a,b) and not intersection(d,a,b,8)]
        self.assertTrue(chosen)


    def test_true_union_of_multiple_copies_counts_extra_labels_and_retains_budget_fallback(self):
        from r35_shell_overlap import union_count
        f=(2,2);n=8
        relations=[{0:1,1:2,2:0},{1:3,3:4,4:1},{0:1,1:2,2:0}]
        expected=0
        inverses=[np.argsort([a.get(i,i) for i in range(n)]) for a in relations]
        for order in permutations(range(n)):
            if cycle_type(order)==f and any(cycle_type(inv[list(order)])==f for inv in inverses):expected+=1
        self.assertEqual(union_count(f,relations,n),expected)
        self.assertEqual(union_count(f,relations[:1],n),intersection((3,),f,f,n))
        self.assertIsNone(union_count(f,relations,n,limit=1))
        self.assertEqual(union_count(f,[],n),0)

if __name__=='__main__':unittest.main()
