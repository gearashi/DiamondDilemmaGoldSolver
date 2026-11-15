"""Mixed physical-turn word oracles on all short cycles and rooted domain restrictions."""
from itertools import product
from random import Random
import unittest

from r27_structure_fixtures import planted
from r27_structure_filters import domain_bits
from r28_geo_sewing import PosePacking
from r30_gold import search
from r31_geographic import physical_bonds
from r35_turn_words import TurnWords


def sewing(cycle):
    return {tuple(sorted((3*t+b,3*u+c))) for (t,_,b),(u,c,_) in zip(cycle,cycle[1:]+cycle[:1])}


class TurnTests(unittest.TestCase):
    def test_every_mixed_word_to_seven_vs_named_cell_pose_enumerator(self):
        p,_=planted(1,3,23)
        words=TurnWords(p)
        poses=PosePacking(p)
        for size in range(3,8):
            for sign in product((1,2),repeat=size):
                cycle=tuple((t,t%3,(t+turn)%3) for t,turn in enumerate(sign))
                edges=tuple(sorted(sewing(cycle)))
                expected=bool(poses.poses(tuple(range(size)),edges))
                self.assertEqual(words.possible(cycle),expected,sign)
                reverse=tuple((t,b,a) for t,a,b in reversed(cycle))
                self.assertEqual(words.possible(reverse),expected)
                self.assertEqual(words.possible(cycle[1:]+cycle[:1]),expected)
                if len(set(sign))>1:
                    self.assertEqual(bool(words.conflicts(edges)),not expected)
                    for trial in edges[:2]:
                        excluded=words.conditional(set(edges),set(edges)-{trial})
                        self.assertEqual(excluded,{trial} if not expected else set())
        self.assertGreater(words.stats['word_reuses'],500)

    def test_anchored_domains_and_historical_corner_scope(self):
        p,_=planted(1,3,33)
        true_map=physical_bonds(p,[3*t for t in range(p.n)])
        pinned=tuple(1<<(3*c) for c in range(p.n))
        known=TurnWords(p,pinned)
        self.assertFalse(known.conflicts(true_map))
        poses=PosePacking(p,pinned)
        rng=Random(35)
        for _ in range(250):
            cells=rng.sample(range(p.n),rng.randrange(3,9))
            cycle=tuple((tile,0,rng.choice((1,2))) for tile in cells)
            expected=bool(poses.poses(tuple(sorted(cells)),tuple(sorted(sewing(cycle)))))
            self.assertEqual(known.possible(cycle),expected)
        constant=sewing(tuple((t,0,1) for t in range(3)))
        self.assertEqual(known.conflicts(constant),[])
        other=TurnWords(p,max_steps=0)
        self.assertEqual(other.conflicts(true_map),[])
        self.assertTrue(other.stats['traversal_limit'])
        with self.assertRaises(ValueError):
            search(p,domains=domain_bits(p),turn_length=5)
        with self.assertRaises(ValueError):
            search(p,domains=domain_bits(p),geography=True,turn_length=17)


if __name__=='__main__':
    unittest.main()
