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
        with patch('r35_repair_seeds.active_catalog',return_value={'codes':ref,'hit':[0,1,2],'lower_bound':2}):
            lower,hint=certified_distance(p,ref)
            self.assertEqual((lower,hint),(2,{0,1,2}))
            self.assertIsNone(certified_choice(lower,hint,[],range(4)))
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

    def test_archived_v115_terminal_consumers_reopen_twice_without_native_work(self):
        import json
        from pathlib import Path
        from constraint_repair_round24 import fragment_crossover,loop_merge_trade,scoped_repair_failure
        archived=json.loads((Path(__file__).parent/'docs/R35_REPAIR_LEGACY_CONSUMERS.json').read_bytes())
        p=SearchProblem(*tetrahedron_tiles());seed=[0,4,6,9]
        bad=SearchProblem(multiple_loops(),double_face())
        calls={
            'core_solved': lambda saved: core_guided_repair(p,seed,durable=True,seconds=0,resume=saved),
            'core_impossible': lambda saved: core_guided_repair(bad,[0,3],durable=True,seconds=0,resume=saved),
            'scope': lambda saved: scoped_repair_failure(p,seed,[0,1],durable=True,seconds=0,resume=saved),
            'fragment': lambda saved: fragment_crossover(p,[seed,[1,3,6,9]],max_fragment=1,durable=True,seconds=0,resume=saved),
            'trade': lambda saved: loop_merge_trade(bad,[0,3],[0,1],durable=True,seconds=0,resume=saved)}
        with patch('constraint_repair_resume_round25._oracle',side_effect=AssertionError('repeated native work')):
            for name,call in calls.items():
                record=archived['lf'][name]
                for saved in (record,{**record,**archived['crlf'][name]}):
                    counters=saved['state']['stats']
                    for _ in range(2):
                        answer=call(saved)
                        self.assertEqual(answer['status'],record['state']['status'])
                        self.assertEqual(answer['best_board'],record['state']['best_board'])
                        self.assertTrue(answer['continuation_complete'])
                        for key in ('repair_completed_queries','repair_completed_masters','repair_native_attempts','oracle_branches'):
                            self.assertEqual(answer['stats'][key],counters[key])
                        saved=answer['checkpoint']
        with self.assertRaisesRegex(ValueError,'scope, policy or source'):
            fragment_crossover(p,[seed,[1,3,6,9]],max_fragment=2,durable=True,seconds=0,resume=archived['lf']['fragment'])
        p.domains[0]=p.domains[0][1:]
        with self.assertRaisesRegex(ValueError,'scope, policy or source'):
            calls['core_solved'](archived['lf']['core_solved'])

    @requires_ortools
    def test_archived_pool_reuses_both_master_and_verified_gold(self):
        import json
        from pathlib import Path
        from importlib.metadata import version
        from constraint_lns import _Control
        from constraint_repair_gold_round26 import GoldRepairPool
        from ortools.sat.python import cp_model
        archived=json.loads((Path(__file__).parent/'docs/R35_REPAIR_LEGACY_CONSUMERS.json').read_bytes())
        p=SearchProblem(*tetrahedron_tiles())
        record=archived['lf']['pool']
        for saved in (record,{**record,**archived['crlf']['pool']}):
            if version('ortools')!=archived['ortools']:
                with self.assertRaisesRegex(ValueError,'identity|scope'):
                    GoldRepairPool(p,resume=saved)
                continue
            for _ in range(2):
                pool=GoldRepairPool(p,resume=saved)
                original=pool.stats
                with patch.object(pool,'_native_oracle',side_effect=AssertionError('unexpected native oracle')):
                    chosen,branches=pool.minimum_hitting_set([[0,1],[1,2]],list(range(4)),
                        _Control(5,None),cp_model,seed=[0,4,6,9])
                    self.assertEqual((chosen,branches),({1},0))
                    known=record['results'][0]
                    answer=pool.oracle(_Control(5,None)).query(known['assumptions'])
                    self.assertEqual(answer['status'],'solved')
                    self.assertEqual(answer['codes'],known['codes'])
                for key in ('gold_pool_master_calls','gold_pool_native_calls','gold_pool_master_build_attempts'):
                    self.assertEqual(pool.stats[key],original[key])
                self.assertEqual(pool.stats['gold_pool_master_completed_hits'],original['gold_pool_master_completed_hits']+1)
                saved=pool.snapshot()

    def test_each_reviewed_repair_producer_and_new_proof_source_fail_closed_if_edited(self):
        import json
        from pathlib import Path
        from r27_incumbent_compat import _variants
        archived=json.loads((Path(__file__).parent/'docs/R35_REPAIR_LEGACY_CONSUMERS.json').read_bytes())
        p=SearchProblem(*tetrahedron_tiles())
        seed=[0,4,6,9];old=archived['lf']['core_solved']
        def open_record(record):
            return core_guided_repair(p,seed,seconds=0,durable=True,resume=record)
        fresh=open_record(old)['checkpoint']
        reader=Path.read_bytes
        sources=('constraint_repair_round24.py','constraint_repair_resume_round25.py','constraint_repair_gold_round26.py','r35_repair_seeds.py')
        for name in sources:
            with self.subTest(name=name),patch.object(Path,'read_bytes',lambda path: reader(path)+(b'# incompatible' if path.name==name else b'')):
                _variants.cache_clear()
                records=[fresh] if name.startswith('r35_') else [old,{**old,**archived['crlf']['core_solved']},fresh]
                for record in records:
                    with self.assertRaisesRegex(ValueError,'scope, policy or source'):
                        open_record(record)
        _variants.cache_clear()

    def test_refined_scope_rechecks_the_suggested_hit_against_all_later_cuts(self):
        import hashlib
        import json
        from pathlib import Path
        from r35_repair_seeds import active_catalog,certified_choice,regional_catalog
        root=Path(__file__).resolve().parent/'docs'
        original=json.loads((root/'R35_REPAIR_PLACEMENTS.json').read_bytes())
        refined=json.loads((root/'R35_REPAIR_REFINED.json').read_bytes())
        self.assertEqual(refined['base_sha256'],hashlib.sha256((root/'R35_REPAIR_PLACEMENTS.json').read_bytes().strip()).hexdigest())
        hit=set(refined['hit'])
        effective=original['rows']+refined['extra']
        rows=list(map(frozenset,effective))
        self.assertEqual(len(hit),93)
        self.assertEqual(certified_choice(93,hit,rows,range(160)),hit)
        self.assertIsNone(certified_choice(93,hit,rows+list(map(frozenset,refined['additional'])),range(160)))
        with patch('r35_repair_seeds.checked_catalog',return_value=original):
            actual=active_catalog(None)
            self.assertEqual(actual['rows'],effective+refined['additional']+regional_catalog()['small']+regional_catalog()['larger'])
            self.assertEqual(actual['lower_bound'],regional_catalog()['proved_lower'])
            self.assertTrue(all(set(actual['hit']).intersection(row) for row in actual['rows']))
            with patch('r35_repair_seeds.regional_catalog',return_value=None):
                weaker=active_catalog(None)
                self.assertEqual(weaker['rows'],effective+refined['additional'])
                self.assertEqual(weaker['hit'],refined['hit'])
            with patch.object(Path,'read_bytes',return_value=b'{}'):
                self.assertEqual(active_catalog(None),original)
            with patch.object(Path,'read_bytes',side_effect=FileNotFoundError):
                self.assertEqual(active_catalog(None),original)

    def test_projected_path_conflicts_transfer_only_matching_placements(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from r35_repair_seeds import archive_rows,regional_catalog
        root=Path(__file__).resolve().parent/'docs'
        original=json.loads((root/'R35_REPAIR_PLACEMENTS.json').read_bytes())
        refined=json.loads((root/'R35_REPAIR_REFINED.json').read_bytes())
        proof=json.loads((root/'R35_REPAIR_CROSSOVER.json').read_bytes())
        expected=proof['subsets']['undominated']
        claims=regional_catalog();path_rows=claims['small']+claims['larger']
        self.assertLessEqual({tuple(x['cells']) for x in expected},{tuple(x) for x in path_rows})
        old=original['rows']+refined['extra']+refined['additional']
        for rec in expected:
            mask=int(rec['previous_master_hit'],16)
            hit={c for c in range(160) if mask>>c&1}
            self.assertTrue(all(hit.intersection(row) for row in old))
            self.assertFalse(hit.intersection(rec['cells']))
        def transfer(codes):
            rows=set()
            state=SimpleNamespace(p=None,codes=codes,check=lambda:None,
                                  add=lambda cells,_kind:rows.add(tuple(cells)))
            with patch('r35_repair_seeds.checked_catalog',return_value=original):
                archive_rows(state)
            return rows
        reference=original['codes']
        for changed in ((),(0,),(103,),(129,130)):
            seed=reference.copy()
            for c in changed:
                seed[c]=3*(seed[c]//3)+(seed[c]%3+1)%3
            transferred=transfer(seed)
            for row in path_rows:
                self.assertEqual(tuple(row) in transferred,not set(changed).intersection(row))

    def test_repair_cut_generation_accepts_only_independent_pose_conflicts(self):
        from bench_r35_repair_refinement import local_cuts
        class Checking:
            def __init__(self,solutions):self.solutions=solutions;self.calls=0
            def replay(self,seed,row):
                self.calls+=1
                return all(any(seed[c]!=g[c] for c in row) for g in self.solutions)
        p=SearchProblem(*tetrahedron_tiles())
        gold=brute(p,'gold');proof=Checking(gold)
        self.assertFalse(local_cuts(p,gold[0],[],proof))
        bad=[0,5,7,11]
        cuts=local_cuts(p,bad,[],proof)
        self.assertGreater(proof.calls,0)
        self.assertTrue(all(all(any(bad[c]!=g[c] for c in row) for g in gold) for row in cuts))
        with self.assertRaisesRegex(ValueError,'independent|Independent'):
            local_cuts(p,bad,[],type('Reject',(),{'replay':lambda *_:False})())
        with self.assertRaises(ValueError):local_cuts(p,bad,[3,3],proof)


if __name__=='__main__':main()
