"""Exact local row equivalence, closed-only persistence, and both directions of budget dominance."""
from itertools import combinations,product
from random import Random
import sqlite3
import unittest
from unittest.mock import patch

from r27_incumbent import BoardModel
from r27_structure_fixtures import planted
from r28_exchange import ExchangeBound
from r35_defect_cores import CoverCores,canonical
from r35_defect_inventory import for_bound
from bench_r35_defect_cores import direct_rows


class DefectReuseTests(unittest.TestCase):
    def test_incremental_clauses_match_raw_piece_sides_and_pool_masks(self):
        rng=Random(35)
        for seed in (34,91,122):
            p,_=planted(1,3,seed);m=BoardModel(p)
            board=[3*t for t in rng.sample(range(m.n),m.n)];bound=ExchangeBound(m,board);inv=for_bound(bound)
            for k in range(60):
                chosen=rng.sample(range(m.n),rng.randrange(2,6));group=(tuple(chosen),)
                pool=rng.sample(range(m.n),rng.randrange(m.n+1))
                self.assertEqual(inv.rows(group,pool),direct_rows(bound,group,pool,2,14))
                before=inv.work['stars_built'];inv.rows(group,reversed(pool))
                self.assertEqual(inv.work['stars_built'],before)

    def test_subsumption_with_multiple_masks_two_axes_and_budget_frontier(self):
        cache=CoverCores()
        cache.add([((1,2),3),((1,2),12)],2,3)
        self.assertTrue(cache.covered([((1,),1),((2,),4)],1,2))
        self.assertFalse(cache.covered([((1,),1)],1,2))
        for counts in [(3,2),(1,4)]:self.assertFalse(cache.covered([((1,2),3),((1,2),12)],*counts))
        cache.add([((1,2),3),((1,2),12)],3,2)
        self.assertEqual(len(cache.cores),2)
        cache.add([((1,2),3),((1,2),12)],3,3)
        self.assertEqual(len(cache.cores),1)
        self.assertEqual(canonical([((1,2),3),((1,2),1),((1,2),4)]),(((1,2),1),((1,2),4)))

    def test_durable_native_replay_rejects_consistent_digest_forgery_and_capped_result(self):
        with sqlite3.connect(':memory:') as db:
            cache=CoverCores(db,'original')
            clauses=[((0,),0),((1,),0)];cache.add(clauses,0,1,certified=True)
            self.assertFalse(CoverCores(db,'other').covered(clauses,0,1))
            with patch('r35_defect_cores.native_negative',return_value=False):
                self.assertFalse(CoverCores(db,'original').covered(clauses,0,1))
            cold=CoverCores(db,'original')
            self.assertTrue(cold.covered(clauses,0,1));self.assertTrue(cold.covered(clauses,0,1))
            self.assertEqual(cold.work['cold_verifications'],1)
            false=[((3,),1)];cache.add(false,1,1,certified=True)
            self.assertFalse(CoverCores(db,'original').covered(false,1,1))

if __name__=='__main__':unittest.main()
