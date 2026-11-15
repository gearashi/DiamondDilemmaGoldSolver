"""Exercise exact physical cardinalities, multi-pose owner semantics and bound identity."""
from itertools import permutations,product
from random import Random
import unittest
from unittest.mock import patch
from pathlib import Path

from r27_structure_fixtures import planted
from r28_exchange import read_json
from r30_gold import search
from r31_geographic import physical_bonds
from r35_geographic_loss import retained_bound,certified_reference
from search_problem import SearchProblem


class GeographicLossTests(unittest.TestCase):
    def test_saturation_on_all_edge_perfect_residual_boards(self):
        rng=Random(35)
        for seed in (4,31,70):
            p,ds=planted(1,3,seed,uniform_ports=True)
            original=[3*t for t in range(p.n)]
            choices=[i for i,m in enumerate(ds) if m.bit_count()>1]
            source=set(physical_bonds(p,original)); valid=[]
            for perm in permutations(choices):
                for rots in product(range(3),repeat=len(choices)):
                    codes=original.copy()
                    for c,t,r in zip(choices,perm,rots):codes[c]=3*t+r
                    pairs=set(physical_bonds(p,codes))
                    if len(pairs)==len(p.edges):valid.append((codes,pairs))
            for _ in range(24):
                reference=set(rng.sample(sorted(source),rng.randint(1,len(source))))
                limit=rng.randrange(len(reference)+1)
                live=list(ds);res=retained_bound(p,live,reference,limit)
                for codes,pairs in valid:
                    if len(reference&pairs)<=limit:
                        self.assertFalse(res['over'])
                        self.assertTrue(all(mask>>code&1 for mask,code in zip(live,codes)))
                        self.assertFalse(set(res['forbidden'])&pairs)
        self.assertIsNone(certified_reference(p))
        self.assertTrue(search(p,geography=True,reference_bound=True,channel=True,domains=ds)['status']=='GOLD')
        with self.assertRaises(ValueError):search(p,reference_bound=True)

    def test_original_identity_and_changed_conditions(self):
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if not path:self.skipTest('Original pieces unavailable.')
        p=SearchProblem(read_json(path));ref,limit=certified_reference(p)
        self.assertEqual((len(ref),limit),(207,187))
        codes=read_json('docs/R29_INCUMBENT.json')['codes'];wrong=codes.copy()
        wrong[0],wrong[1]=wrong[1],wrong[0]
        self.assertIsNone(certified_reference(p,wrong))
        with patch('r35_geographic_loss.read_json',side_effect=FileNotFoundError):
            self.assertIsNone(certified_reference(p))
        with patch('r35_geographic_loss.REFERENCE_SCOPE','unverified'):
            self.assertIsNone(certified_reference(p))
        with patch('r35_geographic_loss.OutsideSyndromes.layout',return_value=([],set(),[])):
            self.assertIsNone(certified_reference(p))
        out=search(p,bonds=sorted(ref)[:188],geography=True,reference_bound=True)
        self.assertEqual((out['status'],out['cp_calls']),('INFEASIBLE',0))

if __name__=='__main__':unittest.main()
