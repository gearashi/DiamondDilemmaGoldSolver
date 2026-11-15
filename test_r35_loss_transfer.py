from collections import Counter
from importlib.util import find_spec
from math import inf
from unittest import TestCase,skipUnless

from r35_loss_transfer import TransferHints, projections
from r35_loss_witness import physical_price
from test_r33_gold_refine import fixture


class TransferTests(TestCase):
    def test_projection_reorders_physical_pairs_and_refuses_missing_or_conflicting_rows(self):
        out=[(3,7),(6,10),(5,8),(0,4)]
        record=dict(agree=[[0,4],[5,8],[100,110]],disagree=[[3,7]])
        self.assertEqual(projections(out,[record]),{(13,1)})
        self.assertEqual(projections(out[:2],[record]),set())
        self.assertEqual(projections(out,[dict(agree=out,disagree=[out[0]])]),set())

    @skipUnless(find_spec('pysat'), 'Optional SAT backend')
    def test_planted_true_forged_hint_and_replayed_negative(self):
        p,ref,groups,req,out=fixture(False,False)
        phases=[]
        for z in range(1<<p.n):
            got=physical_price(p.tiles,ref,[],req,z)
            if got is not None:
                phases.append(sum(1<<i for i,b in enumerate(out) if b in got[1]))
        self.assertIn(0,phases)
        full=(1<<len(out))-1
        missing=next(v for v in range(1<<len(out)) if v not in phases)
        rows=[dict(agree=[b for i,b in enumerate(out) if not value>>i&1],
                   disagree=[b for i,b in enumerate(out) if value>>i&1]) for value in (0,missing)]
        hint=TransferHints(p,req,out,rows)
        try:
            work=Counter()
            self.assertIsNone(hint.matched(0,inf,work))
            m,bits=hint.matched(missing,inf,work)
            self.assertTrue(all((v&m)!=bits for v in phases))
            self.assertEqual(work['physical_hint_replays'],2)
            self.assertEqual(work['physical_hint_proofs'],1)
        finally:
            hint.close()

    @skipUnless(find_spec('pysat'), 'Optional SAT backend')
    def test_new_runtime_hints_are_shared_by_puzzle_and_reproved_without_a_source_cap(self):
        import sqlite3
        from contextlib import closing
        from r33_gold_refine import IntegerCoreStore
        p,ref,groups,req,out=fixture(False,False)
        with closing(sqlite3.connect(':memory:')) as db:
            donor=IntegerCoreStore(db,'donor-budget')
            donor.remember_physical(p.fingerprint,out,1,0)
            other=IntegerCoreStore(db,'destination-budget')
            self.assertEqual(list(other.physical('different-puzzle')),[])
            projected=TransferHints(p,req,tuple(reversed(out)),[],store=other)
            try:
                self.assertEqual(projected.todo,{(1<<(len(out)-1),0)})
                self.assertIsNone(projected.matched(0,inf,Counter()))
            finally:
                projected.close()
