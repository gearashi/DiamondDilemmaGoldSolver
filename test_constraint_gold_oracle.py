"""Finite, independent Gold subtree oracle and interruption checks."""
import random
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ortools.sat.python import cp_model
import constraint_gold_oracle as oracle_module
from constraint_gold_oracle import GoldOracle
from constraint_circuit import CircuitLimit, build_circuit_model
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_models import pair_problem, brute
from test_constraint_dfs import tetrahedron_tiles


def bitsets(problem):
    return [sum(1 << code for code in set(row)) for row in problem.domains]


def fake_solver(status, codes=()):
    return SimpleNamespace(parameters=SimpleNamespace(), num_branches=0,
                           solve=lambda model: status,
                           value=lambda variable: codes[int(variable.name.split('_')[1])],
                           status_name=lambda answer: cp_model.CpSolver().status_name(answer),
                           stop_search=lambda: None)


class GoldOracleTests(unittest.TestCase):
    def test_finite_narrowed_domains_match_independent_enumeration(self):
        rng = random.Random(91287)
        fixtures = [pair_problem(), pair_problem(multiple_loops=True),
                    pair_problem(boundary=True), pair_problem(fixed={0: 0})]
        tiles, board = tetrahedron_tiles()
        fixtures.append(SearchProblem(tiles, board))
        for index, problem in enumerate(fixtures):
            expected = brute(problem, 'gold')
            initial = bitsets(problem)
            oracle = GoldOracle(problem, initial)
            for trial in range(8):
                if trial == 0:
                    domains = initial
                elif trial == 1:
                    domains = [0, *initial[1:]]
                else:
                    domains = [sum(1 << code for code in row if rng.random() < .7)
                               for row in problem.domains]
                possible = {codes for codes in expected if all(
                    domains[cell] & (1 << code) for cell, code in enumerate(codes))}
                oracle.begin_slice()
                result = oracle.query(domains, seconds=2)
                with self.subTest(fixture=index, trial=trial):
                    if possible:
                        self.assertEqual(result.status, 'solved')
                        self.assertIn(result.codes, possible)
                        self.assertTrue(result.validation['valid'])
                    else:
                        self.assertEqual(result.status, 'infeasible')

    def test_planted_closed_ten_cell_board_completes_narrowed_gold(self):
        data, board, reference = planted_gold(seed=91, size=1, extra_edges=3)
        problem = SearchProblem(data, board)
        domains = [1 << code for code in reference]
        for cell in range(0, problem.n, 3):
            domains[cell] |= 1 << ((reference[cell]//3)*3+(reference[cell]+1)%3)
        oracle = GoldOracle(problem, bitsets(problem))
        self.assertTrue(oracle.should_query(domains))
        result = oracle.query(domains, seconds=3)
        self.assertEqual(result.status, 'solved')
        self.assertTrue(problem.validate(list(result.codes))['valid'])

    def test_exact_infeasible_cache_does_not_cover_wider_domains(self):
        problem = pair_problem()
        initial = bitsets(problem)
        oracle = GoldOracle(problem, initial)
        impossible = [1, 1]
        self.assertEqual(oracle.query(impossible, seconds=2).status, 'infeasible')
        with patch.object(oracle_module, 'build_circuit_model', side_effect=AssertionError):
            self.assertEqual(oracle.query(impossible, seconds=0).reason, 'exact_domain_cache')
        oracle.begin_slice()
        result = oracle.query(initial, seconds=2)
        self.assertEqual(result.status, 'solved')
        self.assertEqual(oracle.stats['gold_oracle_cache_hits'], 1)
        self.assertEqual(len(oracle.cache), 1)

    def test_lru_eviction_and_new_oracle_binding_are_safe(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem), cache_limit=2)
        for code in range(3):
            oracle.begin_slice()
            self.assertEqual(oracle.query([1 << code]*2, seconds=2).status, 'infeasible')
        self.assertEqual(len(oracle.cache), 2)
        self.assertNotIn((1, 1), oracle.cache)
        other = GoldOracle(pair_problem(multiple_loops=True), bitsets(problem))
        self.assertFalse(other.cache)
        self.assertNotEqual(oracle.binding, other.binding)

    def test_cancellation_preempts_even_a_cached_answer(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem))
        oracle.query([1, 1], seconds=2)
        def stopped(): raise InterruptedError('user stopped')
        with self.assertRaises(InterruptedError): oracle.query([1, 1], stopped)
        self.assertEqual(oracle.stats['gold_oracle_cache_hits'], 0)

    def test_zero_budget_unknown_and_unknown_native_answer_not_cached(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem))
        self.assertEqual(oracle.query(bitsets(problem), seconds=0).status, 'unknown')
        self.assertFalse(oracle.cache)
        oracle.begin_slice()
        native = fake_solver(cp_model.UNKNOWN)
        native.status_name = lambda status: 'UNKNOWN'
        with patch.object(cp_model, 'CpSolver', return_value=native):
            result = oracle.query(bitsets(problem), seconds=2)
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(oracle.cache)

    def test_construction_time_is_inside_budget_and_no_solve_is_started(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem))
        clock = [0.0]
        original = oracle_module.build_circuit_model
        def build(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 1.0
            return result
        native = fake_solver(cp_model.INFEASIBLE)
        with patch.object(oracle_module.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(oracle_module, 'build_circuit_model', side_effect=build), \
             patch.object(cp_model, 'CpSolver', return_value=native) as factory:
            result = oracle.query(bitsets(problem), seconds=.5)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.reason, 'local_deadline')
        factory.assert_not_called()
        self.assertFalse(oracle.cache)

    def test_dependency_load_expiry_does_not_build_a_model(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem))
        clock = [0.0]
        def load():
            clock[0] = 1.0
            return cp_model
        with patch.object(oracle_module.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(oracle_module, '_load_cp_model', side_effect=load), \
             patch.object(oracle_module, 'build_circuit_model') as builder:
            result = oracle.query(bitsets(problem), seconds=.5)
        self.assertEqual(result.status, 'unknown')
        builder.assert_not_called()
        self.assertFalse(oracle.cache)

    def test_native_exception_is_not_hidden_and_watcher_is_joined(self):
        problem = pair_problem()
        oracle = GoldOracle(problem, bitsets(problem))
        native = fake_solver(cp_model.UNKNOWN)
        def fail(model): raise RuntimeError('native fixture failure')
        native.solve = fail
        with patch.object(cp_model, 'CpSolver', return_value=native), \
             self.assertRaisesRegex(RuntimeError, 'native fixture failure'):
            oracle.query(bitsets(problem), seconds=2)
        self.assertFalse(any(t.name == 'diamond-gold-oracle-stop' for t in threading.enumerate()))
        self.assertFalse(oracle.cache)

    def test_native_stop_and_local_timeout_never_become_infeasible(self):
        problem = pair_problem()
        for reason in ('stop', 'timeout'):
            oracle = GoldOracle(problem, bitsets(problem))
            entered, interrupted = threading.Event(), threading.Event()
            native = fake_solver(cp_model.INFEASIBLE)
            native.stop_search = interrupted.set
            def solve(model):
                entered.set()
                if not interrupted.wait(2):
                    raise AssertionError('Native stop was not delivered')
                return cp_model.INFEASIBLE
            native.solve = solve
            def check():
                if reason == 'stop' and entered.is_set(): raise InterruptedError('user stopped')
            def clock(): return 1.0 if reason == 'timeout' and entered.is_set() else 0.0
            with self.subTest(reason=reason), patch.object(cp_model, 'CpSolver', return_value=native), \
                 patch.object(oracle_module.time, 'monotonic', side_effect=clock):
                if reason == 'stop':
                    with self.assertRaises(InterruptedError): oracle.query(bitsets(problem), check, seconds=.5)
                else:
                    result = oracle.query(bitsets(problem), check, seconds=.5)
                    self.assertEqual(result.status, 'unknown')
            self.assertTrue(interrupted.is_set())
            self.assertFalse(oracle.cache)

    def test_model_limits_and_missing_dependency_skip_without_a_proof(self):
        problem = pair_problem()
        for path, error, reason in [('build_circuit_model', CircuitLimit('fixture cap'), 'fixture cap'),
                                    ('_load_cp_model', ImportError('missing'), 'ortools_unavailable')]:
            oracle = GoldOracle(problem, bitsets(problem))
            with patch.object(oracle_module, path, side_effect=error):
                result = oracle.query(bitsets(problem), seconds=2)
            self.assertEqual(result.status, 'skip')
            self.assertEqual(result.reason, reason)
            self.assertFalse(oracle.cache)

    def test_native_invalid_model_and_invalid_witnesses_raise(self):
        for kind in ('model', 'inventory', 'domain', 'multiple_loops'):
            problem = pair_problem(multiple_loops=kind == 'multiple_loops')
            initial = bitsets(problem)
            candidate = next(iter(brute(problem, 'edges')))
            if kind == 'inventory': candidate = (0, 0)
            domains = initial.copy()
            if kind == 'domain': domains[0] &= ~(1 << candidate[0])
            native = fake_solver(cp_model.MODEL_INVALID if kind == 'model' else cp_model.OPTIMAL, candidate)
            native.status_name = lambda status: 'MODEL_INVALID' if status == cp_model.MODEL_INVALID else 'OPTIMAL'
            oracle = GoldOracle(problem, initial)
            with self.subTest(kind=kind), patch.object(cp_model, 'CpSolver', return_value=native), \
                 self.assertRaises(RuntimeError):
                oracle.query(domains, seconds=2)
            self.assertFalse(oracle.cache)

    def test_adaptive_cooldown_caps_attempts_and_resets_per_slice(self):
        problem = pair_problem()
        domains = bitsets(problem)
        oracle = GoldOracle(problem, domains, max_attempts=2)
        self.assertTrue(oracle.should_query(domains))
        oracle.query(domains, seconds=0)
        self.assertFalse(oracle.should_query(domains, node=127))
        self.assertTrue(oracle.should_query(domains, node=128))
        self.assertEqual(oracle.query(domains, seconds=0, node=127).status, 'skip')
        oracle.query(domains, seconds=0, node=128)
        self.assertFalse(oracle.should_query(domains, node=100000))
        oracle.begin_slice()
        self.assertTrue(oracle.should_query(domains))
        self.assertEqual(oracle.attempts, 0)

    def test_large_or_edge_target_domain_is_skipped_and_bad_input_rejected(self):
        problem = pair_problem()
        initial = bitsets(problem)
        for kwargs in ({'max_open': 1}, {'max_placements': 2}, {'target': 'edges'}):
            oracle = GoldOracle(problem, initial, **kwargs)
            self.assertFalse(oracle.should_query(initial))
            self.assertEqual(oracle.query(initial, seconds=2).status, 'skip')
        oracle = GoldOracle(problem, initial)
        for domains in ([True, 1], [-1, 1], [1], [1 << 6, 1]):
            with self.assertRaises(ValueError): oracle.query(domains)
        for seconds in (True, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): oracle.query(initial, seconds=seconds)
        with self.assertRaises(ValueError): oracle.should_query(initial, node=-1)
        with self.assertRaises(ValueError): GoldOracle(problem, [True, 1])
        with self.assertRaises(ValueError): GoldOracle(problem, initial, max_attempts=0)

    def test_cached_proof_rejects_mutated_problem_even_if_fingerprint_is_unchanged(self):
        for field in ('domains', 'tiles', 'edges', 'neighbors', 'masks', 'fixed', 'board_edges'):
            problem = pair_problem()
            oracle = GoldOracle(problem, bitsets(problem))
            oracle.query([1, 1], seconds=2)
            if field == 'domains': problem.domains[0] = problem.domains[0][1:]
            elif field == 'tiles': problem.tiles[0]['segments'] = (((0,3),(1,4)),)
            elif field == 'edges': problem.edges = problem.edges[:-1]
            elif field == 'neighbors': problem.neighbors[0, 0] = -1
            elif field == 'masks': problem.masks[0, 0] ^= 1
            elif field == 'fixed': problem.fixed[0] = 0
            elif field == 'board_edges': problem.board.edges = problem.board.edges[:-1]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'structure changed'):
                oracle.query([1, 1], seconds=2)


    def test_conclusive_unsat_subsumption_only_covers_narrower_domains(self):
        problem = pair_problem();initial = bitsets(problem)
        oracle = GoldOracle(problem, initial)
        self.assertEqual(oracle.query([7, 7], seconds=2).status, 'infeasible')
        self.assertTrue(oracle.should_query([1, 2]))
        with patch.object(oracle_module, 'build_circuit_model', side_effect=AssertionError('A proven subset needs no new model')):
            result = oracle.query([1, 2], seconds=0)
        self.assertEqual(result.status, 'infeasible')
        self.assertEqual(result.reason, 'narrower_domain_cache')
        self.assertEqual(oracle.stats['gold_oracle_subsumption_hits'], 1)
        for domains in ([7, 56], initial):
            oracle.begin_slice()
            result = oracle.query(domains, seconds=2)
            self.assertEqual(result.status, 'solved')
            self.assertTrue(result.validation['valid'])

    def test_unsat_root_cache_covers_all_narrower_domains_without_relaxing_binding(self):
        problem = pair_problem(multiple_loops=True);initial = bitsets(problem)
        oracle = GoldOracle(problem, initial)
        self.assertEqual(oracle.query(initial, seconds=2).status, 'infeasible')
        with patch.object(oracle_module, 'build_circuit_model', side_effect=AssertionError):
            self.assertEqual(oracle.query([1, 8], seconds=0).reason, 'narrower_domain_cache')
        problem.tiles[0]['segments'] = (((0, 3), (1, 4)),)
        with self.assertRaisesRegex(ValueError, 'structure changed'):
            oracle.query([1, 8], seconds=0)

    def test_subsumption_check_observes_cancellation_and_midcheck_mutation(self):
        for mutate in (False, True):
            problem = pair_problem();oracle = GoldOracle(problem, bitsets(problem))
            oracle.query([7, 7], seconds=2);calls = [0]
            def check():
                calls[0] += 1
                if calls[0] == 2:
                    if mutate: problem.fixed[0] = 0
                    else: raise InterruptedError('stop cached proof scan')
            with self.subTest(mutate=mutate), self.assertRaises(ValueError if mutate else InterruptedError):
                oracle.query([1, 2], check, seconds=0)
            self.assertEqual(oracle.stats['gold_oracle_cache_hits'], 0)

    def test_full_and_contracted_oracles_reconstruct_identical_valid_domain_witnesses(self):
        from test_constraint_knowledge import mixed_problem
        problem = mixed_problem();domains = [1, 56]
        valid = {codes for codes in brute(problem, 'gold') if all(domains[i] & (1 << code) for i, code in enumerate(codes))}
        for contract in (False, True):
            oracle = GoldOracle(problem, bitsets(problem), contract_fixed=contract)
            result = oracle.query(domains, seconds=2)
            self.assertEqual(result.status, 'solved');self.assertIn(result.codes, valid)
            if contract:
                self.assertEqual(oracle.stats['gold_oracle_contracted_ports'], 6)
                self.assertEqual(oracle.stats['gold_oracle_contracted_paths'], 3)
        with self.assertRaises(ValueError):GoldOracle(problem, bitsets(problem), contract_fixed=1)


class GoldTemplateTests(unittest.TestCase):
    @staticmethod
    def boards(built):
        class Collector(cp_model.CpSolverSolutionCallback):
            def __init__(self): super().__init__();self.boards=set()
            def on_solution_callback(self):
                self.boards.add(tuple(int(self.value(code)) for code in built.codes))
        result=Collector();solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=1;solver.parameters.enumerate_all_solutions=True
        solver.parameters.max_time_in_seconds=5
        status=solver.solve(built.model,result)
        if status not in (cp_model.OPTIMAL,cp_model.INFEASIBLE):
            raise AssertionError('Finite template enumeration did not complete')
        if built.model.validate():raise AssertionError(built.model.validate())
        return result.boards

    def test_finite_cloned_siblings_match_fresh_models_and_independent_validator(self):
        from test_constraint_knowledge import mixed_problem
        tiles,board=tetrahedron_tiles()
        fixtures=[pair_problem(),pair_problem(multiple_loops=True),pair_problem(boundary=True),
                  pair_problem(fixed={0:0}),mixed_problem(),SearchProblem(tiles,board)]
        rng=random.Random(220281)
        for index,problem in enumerate(fixtures):
            valid=brute(problem,'gold');initial=tuple(bitsets(problem))
            for contract in (False,True):
                oracle=GoldOracle(problem,initial,contract_fixed=contract)
                original=build_circuit_model(problem,problem.domains,contract_fixed=contract)
                oracle._remember_template(initial,original,lambda:None)
                cached=oracle.templates[initial];before=str(cached.model.proto)
                for case in range(18):
                    if case==0:key=initial
                    elif case==1:key=(0,*initial[1:])
                    else:key=tuple(sum(1<<code for code in row if rng.random()<.6) for row in problem.domains)
                    expected={codes for codes in valid if all(key[cell]&(1<<code) for cell,code in enumerate(codes))}
                    reused=oracle._template_model(key,lambda:None)
                    rows=[tuple(oracle_module._codes(mask)) for mask in key]
                    fresh=build_circuit_model(problem,rows,contract_fixed=contract)
                    with self.subTest(fixture=index,contract=contract,case=case):
                        self.assertEqual(self.boards(reused),expected)
                        self.assertEqual(self.boards(reused),self.boards(fresh))
                        self.assertEqual(str(cached.model.proto),before)
                self.assertEqual(oracle.stats['gold_oracle_template_hits'],18)

    def test_contracted_fixed_chain_template_can_fix_more_tiles_or_exclude_its_anchor(self):
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem();source=(1,56)
        oracle=GoldOracle(problem,bitsets(problem))
        built=build_circuit_model(problem,[[0],[3,4,5]],contract_fixed=True)
        self.assertEqual(built.stats['circuit_contracted_paths'],3)
        oracle._remember_template(source,built,lambda:None)
        expected=brute(problem,'gold')
        for key in ((1,8),(1,16),(1,32),(0,56)):
            self.assertEqual(self.boards(oracle._template_model(key,lambda:None)),
                             {codes for codes in expected if all(key[i]&(1<<code) for i,code in enumerate(codes))})

    def test_template_is_reused_after_unknown_but_unknown_is_not_an_unsat_proof(self):
        problem=pair_problem();initial=bitsets(problem);oracle=GoldOracle(problem,initial)
        native=fake_solver(cp_model.UNKNOWN)
        native.status_name=lambda status:'UNKNOWN'
        with patch.object(cp_model,'CpSolver',return_value=native):
            self.assertEqual(oracle.query(initial,seconds=2).status,'unknown')
        self.assertFalse(oracle.cache);self.assertEqual(len(oracle.templates),1)
        oracle.begin_slice()
        with patch.object(oracle_module,'build_circuit_model',side_effect=AssertionError('Must reuse completed encoding')):
            result=oracle.query([7,56],seconds=2)
        self.assertEqual(result.status,'solved');self.assertTrue(result.validation['valid'])
        self.assertEqual(oracle.stats['gold_oracle_template_hits'],1)

    def test_native_model_mutation_does_not_change_private_template(self):
        problem=pair_problem();initial=bitsets(problem);oracle=GoldOracle(problem,initial)
        native=fake_solver(cp_model.UNKNOWN);native.status_name=lambda status:'UNKNOWN'
        def solve(model):model.add_bool_or([]);return cp_model.UNKNOWN
        native.solve=solve
        with patch.object(cp_model,'CpSolver',return_value=native):oracle.query(initial,seconds=2)
        oracle.begin_slice();self.assertEqual(oracle.query(initial,seconds=2).status,'solved')
        oracle.begin_slice();self.assertEqual(oracle.query([7,56],seconds=2).status,'solved')

    def test_templates_never_widen_domains_and_reject_mutated_geometry(self):
        problem=pair_problem();oracle=GoldOracle(problem,bitsets(problem))
        oracle.query([7,56],seconds=2)
        self.assertIsNone(oracle._template_model(tuple(bitsets(problem)),lambda:None))
        before=oracle.stats['gold_oracle_template_hits']
        problem.tiles[0]['segments']=(((0,3),(1,3)),)
        oracle.begin_slice()
        with self.assertRaisesRegex(ValueError,'structure changed'):oracle.query([7,56],seconds=2)
        self.assertEqual(before,oracle.stats['gold_oracle_template_hits'])

    def test_template_entry_and_representation_byte_limits(self):
        problem=pair_problem();initial=bitsets(problem)
        for kwargs in ({'template_limit':0},{'template_byte_limit':0},{'template_byte_limit':1}):
            oracle=GoldOracle(problem,initial,**kwargs)
            self.assertEqual(oracle.query(initial,seconds=2).status,'solved')
            self.assertFalse(oracle.templates);self.assertEqual(oracle.template_bytes,0)
        oracle=GoldOracle(problem,initial,template_limit=1)
        oracle.query([7,56],seconds=2);first=next(iter(oracle.templates))
        oracle.begin_slice();oracle.query([56,7],seconds=2)
        self.assertEqual(len(oracle.templates),1);self.assertNotIn(first,oracle.templates)
        self.assertEqual(oracle.stats['gold_oracle_template_evictions'],1)
        self.assertLessEqual(oracle.template_bytes,oracle.template_byte_limit)
        self.assertEqual(oracle.template_bytes,sum(value.charge for value in oracle.templates.values()))
        for name in ('template_limit','template_byte_limit'):
            for bad in (-1,True,1.5):
                with self.assertRaises(ValueError):GoldOracle(problem,initial,**{name:bad})

    def test_template_scan_clone_restriction_and_admission_stop_are_transactional(self):
        from test_constraint_knowledge import mixed_problem
        problem=mixed_problem();initial=tuple(bitsets(problem));key=(7,56)
        # Interrupt every check point: only a fully returned clone may be used.
        baseline=GoldOracle(problem,initial);built=build_circuit_model(problem,problem.domains,contract_fixed=True)
        baseline._remember_template(initial,built,lambda:None)
        count=[0]
        def count_check():count[0]+=1
        baseline._template_model(key,count_check)
        for stop_at in range(1,count[0]+1):
            before=str(baseline.templates[initial].model.proto);calls=[0]
            def stop():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('stop clone')
            hits=baseline.stats['gold_oracle_template_hits']
            with self.assertRaises(InterruptedError):baseline._template_model(key,stop)
            self.assertEqual(str(baseline.templates[initial].model.proto),before)
            self.assertEqual(baseline.stats['gold_oracle_template_hits'],hits)
        count[0]=0;fresh=GoldOracle(problem,initial);fresh._remember_template(initial,built,count_check)
        for stop_at in range(1,count[0]+1):
            oracle=GoldOracle(problem,initial);calls=[0]
            def stop():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('stop admission')
            with self.assertRaises(InterruptedError):oracle._remember_template(initial,built,stop)
            self.assertFalse(oracle.templates);self.assertEqual(oracle.template_bytes,0)

    def test_broader_conclusive_proof_discards_redundant_narrower_proofs(self):
        problem=pair_problem(multiple_loops=True);oracle=GoldOracle(problem,bitsets(problem))
        for key in ([1,8],[7,56],bitsets(problem)):
            oracle.begin_slice();self.assertEqual(oracle.query(key,seconds=2).status,'infeasible')
            self.assertEqual(len(oracle.cache),1)
        self.assertEqual(oracle.stats['gold_oracle_proofs_subsumed'],2)
        self.assertEqual(next(iter(oracle.cache)),tuple(bitsets(problem)))

    def test_proof_compaction_cancellation_or_mutation_cannot_admit_unfinished_proof(self):
        for mutate in (False,True):
            problem=pair_problem(multiple_loops=True);oracle=GoldOracle(problem,bitsets(problem))
            oracle.query([1,8],seconds=2);before=dict(oracle.cache);calls=[0]
            def stop():
                calls[0]+=1
                if calls[0]==2:
                    if mutate:problem.fixed[0]=0
                    else:raise InterruptedError('stop proof compaction')
            with self.assertRaises(ValueError if mutate else InterruptedError):
                oracle._remember_proof(tuple(bitsets(problem)),stop)
            self.assertEqual(oracle.cache,before)


if __name__ == '__main__':
    unittest.main()
