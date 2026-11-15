"""Public proposal screening, independent finite domains and actual continuation integrity."""
from copy import deepcopy
from itertools import combinations
from unittest import TestCase
from unittest.mock import patch

from constraint_repair_round24 import core_guided_repair
from constraint_repair_resume_round25 import _digest
from r35_repair_propagation import ProposedRepair
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,brute
from test_dependencies import requires_ortools


class ProposalTests(TestCase):
    def setUp(self):
        self.p=SearchProblem(*tetrahedron_tiles())
        self.gold=brute(self.p,'gold')

    def test_full_propagation_matches_every_consistent_partial_goal(self):
        calls=0
        for seed in ([0,4,6,9],[0,5,7,11],[3,0,6,9],self.gold[0]):
            preview=ProposedRepair(self.p,seed)
            for k in range(5):
                for fixed in combinations(range(4),k):
                    legal=[board for board in self.gold if all(board[c]==seed[c] for c in fixed)]
                    repaired,row=preview.test(fixed,lambda:None)
                    if repaired is not None:self.assertIn(tuple(repaired),list(map(tuple,legal)))
                    if row is not None:
                        self.assertTrue(set(row)<=set(fixed))
                        self.assertEqual(legal,[])
                        self.assertFalse(any(all(board[c]==seed[c] for c in row) for board in self.gold))
                        for cell in row:self.assertIsNot(preview.verdict(set(row)-{cell}),False)
                    calls+=1
        self.assertEqual(calls,64)

    def test_known_gold_needs_neither_master_solve_nor_gold_oracle(self):
        for durable in (False,True):
            with patch('constraint_repair_round24.GoldRepairOracle',side_effect=AssertionError('native Gold allocation')), \
                 patch('constraint_repair_round24._minimum_hitting_set',side_effect=AssertionError('native master')):
                row=core_guided_repair(self.p,self.gold[0],durable=durable,seconds=10)
            self.assertEqual(row['status'],'solved')
            self.assertEqual(row['minimum_changes'],0)
            self.assertEqual(row['stats']['oracle_queries'],0)
            self.assertTrue(self.p.validate(row['best_board'])['valid'])

    def interrupted(self):
        seed=[0,4,6,9]
        marker=[]
        original=ProposedRepair.test
        def probe(engine,held,check):
            ans=original(engine,held,check)
            if ans[1] is not None:marker.append(ans[1])
            return ans
        with patch('r35_repair_seeds.repair_rows',return_value=()),patch.object(ProposedRepair,'test',probe):
            row=core_guided_repair(self.p,seed,durable=True,seconds=10,stop=lambda:bool(marker))
        self.assertTrue(marker)
        self.assertEqual(row['status'],'stopped')
        self.assertEqual(row['propagated_cores'],marker)
        self.assertFalse(row['exclusions']['cores'])
        self.assertEqual(row['stats']['repair_native_attempts'],0)
        return seed,row['checkpoint']

    def test_preserved_negative_is_rechecked_and_rehashed_invalid_claim_is_rejected(self):
        seed,saved=self.interrupted()
        for damaged in ('false_row','missing','illicit_pending'):
            forged=deepcopy(saved)
            if damaged=='false_row':forged['state']['propagated_cores']=[[]]
            if damaged=='missing':forged['state'].pop('propagated_cores')
            if damaged=='illicit_pending':
                forged['state']['phase']='query'
                forged['state']['pending']=[]
            forged['digest']=_digest({k:v for k,v in forged.items() if k!='digest'})
            with self.assertRaisesRegex(ValueError,'continuation'):
                core_guided_repair(self.p,seed,resume=forged,seconds=0)
        other=SearchProblem(*tetrahedron_tiles())
        visits=[]
        verdict=ProposedRepair.verdict
        def count(engine,row):
            visits.append(tuple(row))
            return verdict(engine,row)
        for index in range(2):
            with patch.object(ProposedRepair,'verdict',count):
                again=core_guided_repair(other,seed,resume=saved,seconds=0)
            self.assertEqual(len(visits),len(saved['state']['propagated_cores']))
            self.assertEqual(again['propagated_cores'],saved['state']['propagated_cores'])
            self.assertFalse(again['exclusions']['cores'])
            self.assertEqual(again['stats']['repair_native_attempts'],0)
            saved=again['checkpoint']

    @requires_ortools
    def test_public_resumed_solution_after_a_propagated_negative(self):
        seed,saved=self.interrupted()
        outcome=core_guided_repair(self.p,seed,resume=saved,seconds=10)
        self.assertEqual(outcome['minimum_changes'],min(sum(a!=b for a,b in zip(seed,g)) for g in self.gold))
        self.assertEqual(outcome['stats']['repair_native_attempts'],0)
        self.assertTrue(outcome['propagated_cores'])
        self.assertFalse(outcome['exclusions']['cores'])

    def test_interrupting_new_preflight_preserves_proposal_as_screen_not_native_retry(self):
        seed=[0,4,6,9]
        inside=[]
        def stop_inside(_engine,_held,check):
            inside.append(True)
            check()
            raise AssertionError('cancelled callback returned')
        with patch('r35_repair_seeds.repair_rows',return_value=()),patch.object(ProposedRepair,'test',stop_inside):
            row=core_guided_repair(self.p,seed,durable=True,stop=lambda:bool(inside),seconds=10)
        self.assertEqual(row['checkpoint']['state']['phase'],'screen')
        self.assertEqual(row['checkpoint']['state']['pending'],[])
        self.assertFalse(row['propagated_cores'])
        self.assertEqual(row['stats']['repair_native_attempts'],0)
        untouched=core_guided_repair(self.p,seed,resume=row['checkpoint'],seconds=0)
        self.assertEqual(untouched['checkpoint']['state']['phase'],'screen')
        self.assertEqual(untouched['stats']['repair_completed_masters'],1)
