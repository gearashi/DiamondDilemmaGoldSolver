from collections import Counter
from itertools import combinations
from math import comb, inf
from tempfile import TemporaryDirectory
from pathlib import Path
import sqlite3
from contextlib import closing
from unittest import TestCase
from unittest.mock import patch

from r32_gold import run_syndrome
from r35_loss_witness import checked_hits, save_witness, physical_price
from r35_suffix_reach import SuffixReachability
from test_r33_gold_refine import fixture
from test_r33_gold_lookup import fixture as indexed_fixture


class FrontierTests(TestCase):
    def test_exact_rank_skips_at_every_window_and_deadline(self):
        columns = (1,2,4,0,1,2,4,3,5,6,1)
        reach = SuffixReachability(7,columns,4)
        for errors in (3,4,5):
            prefixes = list(combinations(range(len(columns)-2), errors-2))
            viable = set()
            for rank, prefix in enumerate(prefixes):
                for tail in combinations(range(prefix[-1]+1,len(columns)),2):
                    value = 7
                    for i in (*prefix,*tail):
                        value ^= columns[i]
                    if value == 0:
                        viable.add(rank)
            for start in range(0,len(prefixes),7):
                end = min(start+7,len(prefixes))
                work = Counter()
                actual = list(reach.ranks(errors,start,end,inf,work))
                self.assertEqual([(i,prefixes[i]) for i in range(start,end) if i in viable],actual)
                self.assertEqual(end-start,len(actual)+work['unreachable_prefixes'])
                self.assertEqual(next(reach.ranks(errors,start,end,0,Counter())),(start,None))

    def test_bucket_core_masks_match_naive_pattern_for_every_owned_tail(self):
        p,ref,groups,q = indexed_fixture()
        table=q._build_suffix_table(9,4)
        n=len(q.outside)
        for key,bucket in table.buckets.items():
            for prefix_tuple in [(),(0,),(1,3)]:
                prefix=sum(1<<i for i in prefix_tuple)
                last=max(prefix_tuple,default=-1)
                for mask in (0,3,5,(1<<n)-1):
                    for value in range(mask+1):
                        if value & ~mask:
                            continue
                        blocked=q._core_ordinals(table,key,bucket,prefix,last,2,[(mask,value)])
                        for j,tail in enumerate(bucket):
                            if tail and tail[0] <= last:
                                continue
                            pattern=prefix|sum(1<<i for i in tail)
                            self.assertEqual(bool(blocked>>j & 1),pattern&mask == value)

    def test_positive_caps_modes_and_foreign_or_damaged_phase(self):
        p,ref,groups,req,out=fixture(False,False)
        phase=next(z for z in range(1<<p.n) if (res:=physical_price(p.tiles,ref,groups,req,z)) and res[0]==0)
        with closing(sqlite3.connect(':memory:')) as db:
            save_witness(db,'same',phase)
            self.assertEqual(checked_hits(db,'other',p,ref,groups,req,out,0,True),{})
            self.assertIn(0,checked_hits(db,'same',p,ref,groups,req,out,0,False))
            self.assertIn(0,checked_hits(db,'same',p,ref,groups,req,out,3,True))
            db.execute("UPDATE r35_loss_witnesses SET phase='zzz'")
            self.assertEqual(checked_hits(db,'same',p,ref,groups,req,out,100,True),{})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'loss.sqlite'
            actual=run_syndrome(p,ref,groups,0,[0],path,0,refine_integer=True)
            self.assertEqual(actual['status'],'INTEGER_RELAXATION_FEASIBLE')
            with patch('r32_gold.OutsideSyndromes.__init__',side_effect=AssertionError):
                reuse=run_syndrome(p,ref,groups,3,[0],path,0)
            self.assertTrue(reuse['witness_revalidated'])
            self.assertEqual(reuse['loss_cost'],0)
