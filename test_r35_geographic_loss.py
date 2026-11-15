"""Exercise exact physical cardinalities, multi-pose owner semantics and bound identity."""
from importlib.util import find_spec
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
    def test_transferred_cap_preserves_every_complete_physical_matching(self):
        from r35_geographic_loss import overlap_lower_bound
        def pairings(sides):
            if not sides:
                yield frozenset()
                return
            a,*rest=sides
            for b in rest:
                for tail in pairings([v for v in rest if v != b]):
                    yield tail | {(a,b)}
        all_boards=list(pairings(range(8)))
        random=Random(89351)
        for _ in range(500):
            source,target=random.sample(all_boards,2)
            limit=random.randrange(len(source)+1)
            required=set(random.sample(sorted(random.choice(all_boards)),random.randrange(3)))
            forbidden=set(random.sample(sorted(target),random.randrange(3)))
            actual=[len(target-board) for board in all_boards if len(board&source)<=limit and
                    required<=board and board.isdisjoint(forbidden)]
            lower=overlap_lower_bound(source,limit,target,required,forbidden)
            if lower is None:
                self.assertFalse(actual)
            elif actual:
                self.assertLessEqual(lower,min(actual))
        self.assertIsNone(overlap_lower_bound(set(),1,set(),[(0,1),(0,2)]))

    def test_published_centers_have_distinct_limits_and_cache_scope(self):
        from tempfile import TemporaryDirectory
        from r32_gold_loss import geographic, reference_groups
        from r35_geographic_loss import geographic_loss_cutoff
        docs=Path(__file__).resolve().parent/'docs'
        data=next((p for p in (docs.parent/'data/tiles.json',docs.parent.parent/'data/tiles.json') if p.is_file()),None)
        if data is None:self.skipTest('Original prepared artwork unavailable')
        p=SearchProblem(read_json(data),physical_gold_support=True)
        codes=read_json(docs/'R29_GOLD_RESULTS.json')['residual_reference']
        target,cert,groups=reference_groups(p,codes)
        with TemporaryDirectory() as folder:
            db=Path(folder)/'scopes.sqlite'
            first=geographic(p,target,[],cert.rows,[],0,29,None,5,db)
            self.assertEqual(first['certified_reference_overlap']['minimum_lost'],30)
            self.assertEqual(first['cp_calls'],0)
            again=geographic(p,target,[],cert.rows,[],0,29,None,5,db)
            self.assertTrue(again['reused'])
            self.assertIsNone(geographic_loss_cutoff(p,target,30,[],[]))
            with self.assertRaises(ValueError):
                geographic(p,target,[],[[(0,0)]],[],0,28,None,5,db)
        p.masks[0,0]^=1
        self.assertIsNone(geographic_loss_cutoff(p,target,29,[],[]))

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

    @unittest.skipUnless(find_spec('ortools'),'geographic CP search requires the Python 3.12 profile')
    def test_public_geographic_bound_and_channel(self):
        p,ds=planted(1,3,70,uniform_ports=True)
        self.assertEqual(search(p,geography=True,reference_bound=True,channel=True,domains=ds)['status'],'GOLD')
        with self.assertRaises(ValueError):search(p,reference_bound=True)

    def test_saturated_certificate_finishes_before_loading_cp(self):
        p,ds=planted(1,3,70,uniform_ports=True)
        pair=next(iter(physical_bonds(p,[3*t for t in range(p.n)])))
        with patch('r35_geographic_loss.certified_reference',return_value=({pair},0)):
            out=search(p,domains=ds,bonds=[pair],geography=True,reference_bound=True)
        self.assertEqual((out['status'],out['cp_calls']),('INFEASIBLE',0))

    def test_original_identity_and_changed_conditions(self):
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if not path:self.skipTest('Original pieces unavailable.')
        p=SearchProblem(read_json(path));ref,limit=certified_reference(p)
        self.assertEqual((len(ref),limit),(207,175))
        from unittest.mock import patch
        with patch('r35_contact_certificate.checked_contact_bound',side_effect=AssertionError('unnecessary repeat')):
            self.assertEqual(certified_reference(p),(ref,limit))
        original_read=Path.read_bytes
        def hide(*names):
            return lambda path: b'{}' if path.name in names else original_read(path)
        with patch.object(Path,'read_bytes',hide('R35_CONTACT_WIDE_PROOF.json')):
            self.assertEqual(certified_reference(p),(ref,limit))
        with patch.object(Path,'read_bytes',hide('R35_SPHERE_CONTACT_PROOF.json')):
            self.assertEqual(certified_reference(p),(ref,176))
        with patch.object(Path,'read_bytes',hide('R35_CONTACT_WIDE_PROOF.json','R35_SPHERE_CONTACT_PROOF.json')):
            self.assertEqual(certified_reference(p),(ref,180))
        self.assertEqual(certified_reference(p),(ref,limit))
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
