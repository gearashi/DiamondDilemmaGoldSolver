"""Input audit: original image identity, independent endpoints, seam inventory.

Passing establishes finite consistency of the encoding with recovered source
endpoints. Interior segment pairings still require their own image audit.
"""
import hashlib
import json
from pathlib import Path
import unittest
from collections import Counter
import numpy as np
from PIL import Image
from audit_endpoints import load_gold,evidence,inventory
from validator import normalize_tiles,orientation_masks

ROOT=Path(__file__).resolve().parent

class SourceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'data'/'tiles.json').read_text(encoding='utf-8-sig'))
        cls.audit=json.loads((ROOT/'data'/'endpoint_audit.json').read_text(encoding='utf-8-sig'))

    def test_original_source_hashes(self):
        for name,expected in self.data['source_sha256'].items():
            actual=hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
            self.assertEqual(actual,expected,name)
            self.assertEqual(actual,self.audit['source_sha256'][name],name)

    def test_tile_family_and_ids(self):
        tiles=normalize_tiles(self.data)
        self.assertEqual(len(tiles),160)
        self.assertEqual(Counter(t['group'] for t in self.data['tiles']),{'silver':32,'red':48,'blue':80})
        self.assertEqual([t['number'] for t in self.data['tiles']],list(range(1,161)))
        self.assertEqual(sum(len(t['segments']) for t in tiles),365)

    def test_independent_endpoint_agreement(self):
        expected={t['id']:{tuple(p) for p in t['endpoints']} for t in self.audit['tiles']}
        mismatches=[]
        for tile in normalize_tiles(self.data):
            actual={point for segment in tile['segments'] for point in segment}
            if actual!=expected[tile['id']]:
                mismatches.append({'id':tile['id'],'unexpected':sorted(actual-expected[tile['id']]),'missing':sorted(expected[tile['id']]-actual)})
        self.assertEqual(mismatches,[])

    def test_independent_endpoint_pixel_reproduction(self):
        arrays,ids,hashes=load_gold(ROOT/'data')
        actual=evidence(arrays,self.audit['parameters'])
        expected=np.array([[p[2] for p in t['endpoint_evidence']] for t in self.audit['tiles']])
        self.assertEqual(ids,[t['id'] for t in self.audit['tiles']])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=.000051)
        self.assertTrue(np.array_equal(actual>=75,actual>=105))
        self.assertEqual(int((actual>=90).sum()),730)
        self.assertTrue(np.all((actual>=90).sum(axis=1)%2==0))

    def test_global_edge_inventory(self):
        masks=orientation_masks(self.data)[::3]
        report=inventory(masks)
        self.assertEqual(report['violations'],[])
        self.assertEqual(report['maximum_possible_matched_edges'],240)

    def test_overlay_dimensions(self):
        for group in ('silver','red','blue'):
            with Image.open(ROOT/'data'/f'tiles{group}.gif') as source:
                with Image.open(ROOT/'data'/f'extraction_{group}.png') as overlay:
                    self.assertEqual(overlay.size,(source.width*3,source.height*3))

if __name__=='__main__':
    unittest.main()
