"""Exercise translated Gold repair proofs, partial catalogues and domain changes."""
import json
from itertools import combinations,permutations,product
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from constraint_symmetry import proper_rotations
from r35_repair_seeds import archive_rows, certified_distance, proof_sources
from search_problem import SearchProblem
from test_constraint_dfs import brute,tetrahedron_tiles
from test_r35_pose_transfer_authority import planted_on_codes

DOCS=Path(__file__).resolve().parent/'docs'


def recovered(p,seed):
    result=set()
    state=SimpleNamespace(p=p,codes=seed,check=lambda:None,
                          add=lambda row,_kind:result.add(frozenset(row)))
    archive_rows(state)
    return result


class RepairRotationTests(TestCase):
    def test_every_rotated_literal_and_domain_adjustment_on_finite_gold(self):
        p=SearchProblem(*tetrahedron_tiles());gold=brute(p,'gold')
        ref=(0,5,7,11)
        rows=[subset for k in range(1,5) for subset in combinations(range(4),k)
              if all(any(ref[c]!=g[c] for c in subset) for g in gold)]
        lower=min(sum(x!=y for x,y in zip(ref,g)) for g in gold)
        champion=min(gold,key=lambda g:sum(x!=y for x,y in zip(ref,g)))
        catalog={'codes':ref,'hit':[c for c in range(4) if champion[c]!=ref[c]],'rows':rows,'lower_bound':lower}
        images=[]
        for cells in permutations(range(4)):
            for offsets in product(range(3),repeat=4):
                if all(p.neighbors[cells[a],(s+offsets[a])%3]==cells[b]
                       and p.sides[cells[a],(s+offsets[a])%3]==(t+offsets[b])%3 for a,s,b,t in p.edges):
                    images.append(tuple((cells[c],3*(ref[c]//3)+(ref[c]%3+offsets[c])%3) for c in range(4)))
        self.assertGreater(len(images),1)
        cases=[list(g) for g in gold]+[list(ref),[0,3,6,9]]
        for image in images:
            cases.append([code for _,code in sorted(image)])
        with patch('r35_repair_seeds.active_catalog',return_value=catalog),patch('r35_repair_seeds.catalog_images',return_value=images):
            for restricted in (False,True):
                if restricted:p.domains[0]=[gold[0][0]]
                positive=[g for g in gold if all(x in d for x,d in zip(g,p.domains))]
                self.assertTrue(positive)
                for seed in cases:
                    actual=min(sum(x!=y for x,y in zip(seed,g)) for g in positive)
                    self.assertLessEqual(certified_distance(p,seed)[0],actual)
                    for row in recovered(p,seed):
                        self.assertTrue(all(any(seed[c]!=g[c] for c in row) for g in positive))

    def test_changed_original_artwork_cannot_inherit_saved_distance_or_rows(self):
        ref=json.loads((DOCS/'R29_INCUMBENT.json').read_bytes())['codes']
        tiles,board=planted_on_codes(ref)
        p=SearchProblem(tiles,board,physical_gold_support=True)
        for r in proper_rotations(board):
            seed=r.apply(ref)
            self.assertTrue(p.validate(seed)['valid'])
            self.assertEqual(certified_distance(p,seed),(0,None))
            self.assertFalse(recovered(p,seed))
        self.assertTrue({'constraint_symmetry.py','geometry.py','r35_pose_departures.py','r35_anchor_rules.py','r35_claim_catalogue.py'}<=set(proof_sources()))

    def test_original_alt_full_domain_and_fixed_reference_pose(self):
        source=next((f for f in (DOCS.parent/'data/tiles.json',DOCS.parent.parent/'data/tiles.json') if f.is_file()),None)
        if source is None:self.skipTest('Original prepared puzzle unavailable')
        data=json.loads(source.read_bytes());p=SearchProblem(data,physical_gold_support=True)
        ref=json.loads((DOCS/'R29_INCUMBENT.json').read_bytes())['codes']
        alt=json.loads((DOCS/'R29_GOLD_RESULTS.json').read_bytes())['residual_reference']
        r=proper_rotations(p.board)[7];seed=r.apply(alt);center=r.apply(ref)
        self.assertEqual(certified_distance(p,seed)[0],93)
        self.assertEqual(len(recovered(p,seed)),4059)
        changed=[d for d,(a,b) in enumerate(zip(center,seed)) if a!=b]
        self.assertEqual(len(changed),2)
        for lo,d in enumerate(changed,1):
            p.domains[d]=[center[d]]
            self.assertEqual(certified_distance(p,seed)[0],93+2*lo)
        p.neighbors[0,0]=-1
        self.assertEqual(certified_distance(p,seed),(0,None))
        self.assertFalse(recovered(p,seed))
