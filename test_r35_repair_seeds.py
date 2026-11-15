"""Check Gold repair obligations and native master behavior on independently enumerated boards."""
from copy import deepcopy
from itertools import permutations,product
from unittest import TestCase,main
from unittest.mock import patch

from constraint_repair_round24 import core_guided_repair
from r35_repair_seeds import master_rows,repair_rows,seed_rows
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,double_face,simple_tiles,multiple_loops,brute
from test_dependencies import requires_ortools


class SeedProofTests(TestCase):
    def test_every_tetrahedral_seed_and_direct_gold_solution(self):
        p=SearchProblem(*tetrahedron_tiles())
        gold=brute(p,'gold')
        self.assertTrue(gold)
        checked=0
        for tiles in permutations(range(4)):
            for rotations in product(range(3),repeat=4):
                seed=[3*t+r for t,r in zip(tiles,rotations)]
                rows,_=seed_rows(p,seed,anchor_radius=2,max_domain=3*p.n)
                for goal in gold:
                    changes={c for c,(x,y) in enumerate(zip(seed,goal)) if x!=y}
                    self.assertTrue(all(row & changes for row in rows),(seed,goal,rows))
                checked+=1
        self.assertEqual(checked,1944)

    def test_parallel_board_seams_and_changed_domains_recheck_rows(self):
        p=SearchProblem(simple_tiles(),double_face());gold=brute(p,'gold')
        for tiles in permutations(range(2)):
            for turns in product(range(3),repeat=2):
                seed=[3*t+r for t,r in zip(tiles,turns)]
                rows=repair_rows(p,seed)
                for board in gold:
                    self.assertTrue(all(any(seed[c]!=board[c] for c in row) for row in rows))
        seed=list(gold[0]);self.assertFalse(repair_rows(p,seed))
        p.domains[0]=[v for v in p.domains[0] if v!=seed[0]]
        changed=repair_rows(p,seed)
        self.assertIn(frozenset({0}),changed)
        p.domains[0]=[seed[0]]
        self.assertFalse(repair_rows(p,seed))

    def test_wholly_fixed_proper_circuit_is_a_structural_negative_without_oracle_calls(self):
        seed=[0,3]
        p=SearchProblem(multiple_loops(),double_face(),fixed=dict(enumerate(seed)))
        self.assertFalse(brute(p,'gold'))
        with patch('constraint_repair_round24.GoldRepairOracle',side_effect=AssertionError('unnecessary native oracle')):
            for durable in (False,True):
                row=core_guided_repair(p,seed,seconds=2,durable=durable)
                self.assertEqual(row['status'],'infeasible')
                self.assertEqual(row['reason'],'proved_changed_cell_conflict')
                self.assertFalse(row['exclusions']['cores'])
                self.assertFalse(row['cores'])
                if durable:
                    again=core_guided_repair(p,seed,seconds=0,resume=row['checkpoint'])
                    self.assertEqual(again['status'],'infeasible')
                    self.assertEqual(again['stats']['repair_native_attempts'],0)
                    self.assertEqual(again['stats']['repair_completed_queries'],0)

    def test_transferred_distance_with_fixed_domains_and_catalog_downgrade(self):
        from r35_repair_seeds import certified_distance,certified_choice,checked_catalog
        p=SearchProblem(*tetrahedron_tiles());gold=brute(p,'gold')
        ref=[0,5,7,11];distance=min(sum(a!=b for a,b in zip(ref,g)) for g in gold)
        self.assertEqual(distance,3)
        with patch('r35_repair_seeds.checked_catalog',return_value={'codes':ref,'hit':[0,1,2]}):
            for rotations in product(range(3),repeat=4):
                seed=[3*t+v for t,v in enumerate(rotations)]
                lower,_=certified_distance(p,seed)
                self.assertLessEqual(lower,min(sum(a!=b for a,b in zip(seed,g)) for g in gold))
            other=ref.copy();other[0]=1
            p.domains[0]=[ref[0]]
            self.assertEqual(certified_distance(p,other)[0],4)
        self.assertIsNone(certified_choice(3,{0,1,2},[{3}],range(4)))
        self.assertIsNone(certified_choice(3,{0,1,2},[{0}],{0,1}))
        self.assertEqual(certified_choice(3,{0,1,2},[{2},{0,1}],range(4)),{0,1,2})
        with patch('r35_repair_seeds.on_original_mesh',return_value=True),patch('pathlib.Path.read_bytes',return_value=b'{}'):
            self.assertIsNone(checked_catalog(p))
            self.assertEqual(certified_distance(p,ref),(0,None))

    def test_available_master_capacity_keeps_all_real_native_cores(self):
        earned=[[0,1],[2,3],[4]]
        derived=(frozenset({0,1,2}),frozenset({2,4,8}),frozenset({7,8}),frozenset({10,11}))
        self.assertEqual(master_rows(derived,earned,4),[*map(frozenset,earned),frozenset({7,8})])
        self.assertEqual(master_rows(derived,earned,3),list(map(frozenset,earned)))

    @requires_ortools
    def test_native_public_repairs_agree_and_save_actual_queries(self):
        p=SearchProblem(*tetrahedron_tiles());gold=brute(p,'gold')
        for seed in ([0,4,6,9],[0,5,7,11],[3,0,6,9],[6,11,5,1]):
            distance=min(sum(a!=b for a,b in zip(seed,solution)) for solution in gold)
            for durable in (False,True):
                base=core_guided_repair(p,seed,seconds=5,durable=durable,seeded=False)
                new=core_guided_repair(p,seed,seconds=5,durable=durable)
                self.assertEqual(base['minimum_changes'],distance)
                self.assertEqual(new['minimum_changes'],distance)
                self.assertTrue(p.validate(new['best_board'])['valid'])
                expected=(self.assertLess if seed!=[6,11,5,1] else self.assertLessEqual)
                expected(new['stats']['oracle_queries'],base['stats']['oracle_queries'])
                if durable:
                    restored=core_guided_repair(p,seed,seconds=0,resume=new['checkpoint'])
                    self.assertEqual(restored['stats']['repair_native_attempts'],new['stats']['repair_native_attempts'])
                    with self.assertRaisesRegex(ValueError,'scope, policy or source'):
                        core_guided_repair(p,seed,resume=new['checkpoint'],seeded=False)

    @requires_ortools
    def test_actual_v115_pending_query_replays_before_new_seeded_master(self):
        import json
        from pathlib import Path
        record=json.loads((Path(__file__).parent/'docs/R35_REPAIR_LEGACY.json').read_bytes())
        p=SearchProblem(*tetrahedron_tiles());seed=record['seed']
        for saved in (record['saved'],{**record['saved'],**record['crlf']}):
            self.assertEqual(saved['state']['pending'],[])
            self.assertEqual(saved['state']['stats']['repair_native_attempts'],0)
            restored=core_guided_repair(p,seed,resume=saved,seconds=5)
            self.assertEqual(restored['minimum_changes'],1)
            self.assertEqual(restored['stats']['repair_completed_queries'],2)
            self.assertEqual(restored['stats']['repair_completed_masters'],2)
            self.assertEqual(len(restored['exclusions']['cores']),1)
            with self.assertRaisesRegex(ValueError,'scope, policy or source'):
                core_guided_repair(p,[1,3,6,9],resume=saved,seconds=1)

    @requires_ortools
    def test_retained_native_pool_solves_earned_rows_only_export_native_cores(self):
        from constraint_repair_gold_round26 import GoldRepairPool
        p=SearchProblem(*tetrahedron_tiles());seed=[0,4,6,9]
        pool=GoldRepairPool(p)
        row=core_guided_repair(p,seed,seconds=5,retained_service=pool)
        self.assertEqual(row['status'],'solved')
        self.assertEqual(row['minimum_changes'],1)
        self.assertFalse(row['cores'])
        self.assertFalse(row['exclusions']['cores'])
        self.assertGreater(row['derived_rows'],0)
        self.assertEqual(row['stats']['repair_completed_queries'],1)


if __name__=='__main__':main()
