from itertools import product
from math import inf
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase

from r32_gold import run_syndrome
from r32_gold_syndrome import OutsideSyndromes
from r35_loss_witness import physical_price
from r35_overlap_loss import OverlapPattern
from test_r33_gold_refine import fixture


class GroupPricingTests(TestCase):
    def test_all_images_match_exhaustive_labelled_phases(self):
        p,ref,groups,req,outside=fixture(True,True)
        bonds=sorted(set().union(*groups)-req)
        extra=frozenset([outside[0],bonds[0]])
        allgroups=[*groups,extra]
        for integer in (False,True):
            costs={}
            for phase in range(1<<p.n):
                result=physical_price(p.tiles,ref,allgroups,req,phase,integer=integer)
                if result:
                    amount,missing=result
                    value=frozenset(i for i,b in enumerate(outside) if b in missing)
                    costs[value]=min(costs.get(value,inf),amount)
            for cap in range(min(costs.values())-1,max(costs.values())+2):
                check=OverlapPattern(p,groups,allgroups,req,outside,cap,integer,Counter())
                try:
                    for mask in product((0,1),repeat=len(outside)):
                        value=frozenset(i for i,x in enumerate(mask) if x)
                        self.assertEqual(check(value,inf),costs.get(value,inf)<=cap)
                        if check.last_witness is not None:
                            amount,phase=check.last_witness
                            self.assertEqual(physical_price(p.tiles,ref,allgroups,req,phase,integer=integer)[0],amount)
                finally:
                    check.close()

    def test_positive_checkpoint_cannot_omit_a_new_physical_departure_group(self):
        p,ref,groups,req,out=fixture(False,False)
        # The zero-loss planted cube has no genuine conflict groups. A
        # synthetic contradiction is used solely to exercise scope isolation.
        overlapping=[frozenset([out[0],out[1]]),frozenset([out[1],out[2]])]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'pool.sqlite'
            for refined in (False,True):
                weak=run_syndrome(p,ref,groups,0,[0],path,0,refine_integer=refined)
                self.assertTrue(weak['status'].endswith('RELAXATION_FEASIBLE'))
                strong=run_syndrome(p,ref,overlapping,0,[0],path,0,refine_integer=refined,all_groups=True)
                self.assertEqual(strong['status'],'INFEASIBLE')
                self.assertTrue(run_syndrome(p,ref,groups,0,[0],path,0,refine_integer=refined)['reused'])
