"""Independently enumerate labeled pose choices and count each attained reference pair once."""
import itertools
import random
import unittest

from r35_alternative_retention import prune_alternatives, projected_dual, pose_charges


def physical_rows(domains):
    n=len(domains)
    for permutation in itertools.permutations(range(n)):
        rotations=[[3*t+r for r in range(3) if domains[c] & (1<<(3*t+r))] for c,t in enumerate(permutation)]
        yield from itertools.product(*rotations)


class AlternativeRetentionTests(unittest.TestCase):
    def test_checked_bound_preserves_every_labeled_witness(self):
        rng=random.Random(738)
        for trial in range(160):
            n=4
            ds=[sum(1<<v for v in range(3*n) if rng.random()<.56) for _ in range(n)]
            pairs={}
            for a,b in itertools.combinations(range(n),2):
                for side in (0,1):
                    rows=[(c,v) for c,m in enumerate(ds) for v in range(3*n)
                          if v//3 in (a,b) and m>>v&1 and rng.randrange(5)==0]
                    if rows: pairs[3*a+side,3*b+side]=rows
            valid=[]
            for board in physical_rows(ds):
                chosen=set(enumerate(board))
                valid.append((len([q for q in pairs.values() if not chosen.isdisjoint(q)]),board))
            for slack in (1,2,3):
                live=ds.copy();out=prune_alternatives(live,pairs,slack)
                if valid:
                    self.assertLessEqual(out['extra_minimum'],min(k for k,_ in valid))
                for k,b in valid:
                    if k<=slack:
                        self.assertFalse(out['over'],(trial,slack,b))
                        self.assertTrue(all(d&(1<<v) for d,v in zip(live,b)),(trial,slack,b))

    def test_duplicate_endpoints_and_saturated_disjunction(self):
        # Both ends of one retained pair may imply it. Either of two physical
        # alternatives is allowed; saturation cannot forbid both named pairs.
        ds=[(1<<0)|(1<<3),(1<<6)|(1<<9),(1<<0)|(1<<3),(1<<6)|(1<<9)]
        pairs={(0,6):[(0,0),(1,6),(0,0)],(3,9):[(0,3),(1,9)]}
        out=prune_alternatives(ds,pairs,1)
        self.assertFalse(out['over']);self.assertEqual(out['extra_minimum'],1)
        self.assertTrue(all(ds[c]&(1<<v) for c,v in enumerate((0,6,3,9))))
        self.assertTrue(all(ds[c]&(1<<v) for c,v in enumerate((3,9,0,6))))

    def test_weighted_contested_tile_and_unpriced_positive(self):
        n=8
        ds=[(1<<0)|(1<<(3*(c+1))) if c<4 else sum(1<<(3*t) for t in range(n)) for c in range(n)]
        # Six named endpoints per pair are unnecessary; only the charged
        # tile poses are present. Each paid site then forces two real labels.
        optional={}
        for c in range(4):
            code=3*(c+1)
            for side in (0,1):optional[code+side,18+side]=[(c,code)]
        price,_=pose_charges(optional);dual=projected_dual(ds,price)
        self.assertEqual(dual['lower'],12)
        self.assertTrue(prune_alternatives(ds.copy(),optional,5)['over'])
        self.assertFalse(prune_alternatives(ds.copy(),optional,6)['over'])


if __name__=='__main__':unittest.main()
