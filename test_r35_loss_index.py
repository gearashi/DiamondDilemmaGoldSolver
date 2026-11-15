from unittest import TestCase
from math import inf
from test_r33_gold_lookup import fixture
from r32_gold_syndrome import OutsideSyndromes


class LossCatalogueTests(TestCase):
    def test_cross_layer_catalogue_reuse_preserves_ordinal_scopes_and_prices(self):
        problem, ref, groups, oracle = fixture()
        baseline = OutsideSyndromes(problem,ref,groups,oracle.required,reuse_suffix_tables=False)
        candidate = OutsideSyndromes(problem,ref,groups,oracle.required)
        for cap,errors in ((6,2),(7,4),(10,3),(7,4),(11,0),(12,1),(9,4)):
            want=baseline._build_suffix_table(cap,errors)
            actual=candidate._build_suffix_table(cap,errors)
            self.assertEqual((want.order_key,want.prices,want.buckets,want.split_buckets,want.packed_tails),
                             (actual.order_key,actual.prices,actual.buckets,actual.split_buckets,actual.packed_tails))
            self.assertEqual(candidate.indexed(cap,errors,inf,start=0,stop=1)['status'],
                             baseline.indexed(cap,errors,inf,start=0,stop=1)['status'])
        self.assertLess(candidate.index_work['catalogue_rows'],baseline.index_work['catalogue_rows'])
        self.assertLessEqual(len(candidate._catalogues),3)
