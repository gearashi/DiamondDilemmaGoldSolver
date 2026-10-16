"""Independent finite solution oracles for first-UIP mask clause learning."""
import itertools
import random
import unittest
from dataclasses import FrozenInstanceError
from copy import deepcopy

from constraint_lcg import LCGState, LCGStore, LearnedClause

FINITE_ORACLE_SUMMARY = {}


def satisfies(row, clause):
    # Deliberately evaluates assigned values, not domain implication code.
    return not all(row[cell] in [v for v in range(3) if mask & (1 << v)]
                   for cell, mask in clause)


class LazyClauseTests(unittest.TestCase):
    def state(self, domains=(3, 3, 3, 3), **kwargs):
        return LCGState(domains, 'fixture', **kwargs)

    def store(self, domains=(3, 3, 3, 3), **kwargs):
        return LCGStore('fixture', 'gold', domains, **kwargs)

    def diamond(self):
        root = self.state()
        child = root.fork(0, 1)
        child.record_cell(1, 1, [0], 'a implies b')
        child.record_cell(2, 1, [1], 'b implies c')
        child.record_cell(3, 1, [1], 'b implies d')
        return root, child

    @staticmethod
    def pending(frames):
        answer=set()
        for frame in frames:
            values=frame['domains'].copy()
            if frame['cell']>=0:
                mask=0
                for choice in frame['choices'][frame['next']:]:mask|=choice
                values[frame['cell']]&=mask
            boards=set(itertools.product(*[[v for v in range(3) if mask>>v&1] for mask in values]))
            if answer & boards:raise AssertionError('Pending rectangles overlap')
            answer |= boards
        return answer

    def test_asserting_jump_keeps_all_nonconflicting_mask_siblings(self):
        store=self.store([7]*3)
        # The first two alternatives are two-value masks, not placements.
        frames=[{'domains':[7,7,7],'cell':0,'choices':[3,4],'next':1},
                {'domains':[3,7,7],'cell':1,'choices':[3,4],'next':1},
                {'domains':[3,3,7],'cell':2,'choices':[1,2,4],'next':1}]
        before=deepcopy(frames)
        learned=LearnedClause(store.binding,((0,3),(1,3)),(1,3),2,1,0)
        replacement=store.asserting_frontier(frames,learned)
        self.assertIsNotNone(replacement)
        self.assertEqual(len(replacement),2)
        self.assertEqual(replacement[-1],{'domains':[3,4,7],'cell':-1,'choices':[],'next':0})
        self.assertEqual(self.pending(replacement),{row for row in self.pending(frames) if satisfies(row,learned.literals)})
        self.assertEqual(frames,before)
        self.assertEqual(store.stats['lcg_asserting_jumps'],1)

    def test_assertion_prunes_partial_uip_mask_without_replaying_completed_choices(self):
        store=self.store([7]*3)
        frames=[{'domains':[7,7,7],'cell':0,'choices':[1,6],'next':1},
                {'domains':[1,7,7],'cell':1,'choices':[1,6],'next':1}]
        learned=LearnedClause(store.binding,((0,1),(2,3)),(2,3),2,1,0)
        replacement=store.asserting_frontier(frames,learned)
        self.assertEqual(replacement[-1]['domains'],[1,6,4])
        self.assertEqual(self.pending(replacement),{row for row in self.pending(frames) if satisfies(row,learned.literals)})
        self.assertEqual(store.stats['lcg_asserted_values'],2)

    def test_native_cost_gate_defers_unit_without_dropping_learned_clause(self):
        store=self.store([7]*3)
        frames=[{'domains':[7,7,7],'cell':0,'choices':[1,6],'next':1},
                {'domains':[1,7,7],'cell':1,'choices':[1,6],'next':1}]
        learned=LearnedClause(store.binding,((0,1),(2,3)),(2,3),2,1,0)
        store.learn(learned)
        self.assertIsNone(store.asserting_frontier(frames,learned,require_jump=True))
        values=[1,6,7];store.prune(values,lambda:None)
        self.assertEqual(values,[1,6,4])
        self.assertEqual(store.stats['lcg_assertions'],0)

    def test_asserting_jump_never_discards_a_deeper_unproved_sibling(self):
        store=self.store([7]*3)
        frames=[{'domains':[7,7,7],'cell':0,'choices':[3,4],'next':1},
                {'domains':[3,7,7],'cell':1,'choices':[1,6],'next':1},
                {'domains':[3,1,7],'cell':2,'choices':[1,6],'next':1}]
        # Deeper values 1/2 of cell2 are not forbidden, so jumping to level1
        # would lose them. The top frame can still assert safely in place.
        learned=LearnedClause(store.binding,((0,3),(2,3)),(2,3),2,1,0)
        replacement=store.asserting_frontier(frames,learned)
        self.assertEqual(len(replacement),3)
        self.assertEqual(replacement[-1]['domains'],[3,1,4])
        self.assertTrue(self.pending(replacement)<=self.pending(frames))
        self.assertTrue({row for row in self.pending(frames) if satisfies(row,learned.literals)} <= self.pending(replacement))
        self.assertEqual(store.stats['lcg_asserting_jumps'],0)

    def test_finite_assertion_frontiers_preserve_every_allowed_pending_board(self):
        # Exhaustive mask-valued UIP/non-UIP clauses over a real DFS frontier.
        # Treat each nogood as the independently specified constraint; the
        # transformation must neither add boards nor remove any satisfying it.
        families=0
        for a,b,c in itertools.product(range(1,8),repeat=3):
            store=self.store([7]*3)
            frames=[{'domains':[7,7,7],'cell':0,'choices':[3,4],'next':1},
                    {'domains':[3,7,7],'cell':1,'choices':[1,6],'next':1},
                    {'domains':[3,1,7],'cell':2,'choices':[1,6],'next':1}]
            clause=((0,a),(1,b),(2,c));learned=LearnedClause(store.binding,clause,(2,c),3,1,0)
            before=self.pending(frames);replacement=store.asserting_frontier(frames,learned)
            if replacement is not None:
                after=self.pending(replacement)
                self.assertLessEqual(after,before)
                self.assertLessEqual({row for row in before if satisfies(row,clause)},after)
            families+=1
        FINITE_ORACLE_SUMMARY['asserting_frontier_families']=families

    def test_exhausted_rectangles_do_not_trigger_a_useless_native_rebuild(self):
        store=self.store([7]*2)
        frames=[{'domains':[7,7],'cell':0,'choices':[1,6],'next':1},
                {'domains':[1,7],'cell':1,'choices':[1,6],'next':2}]
        learned=LearnedClause(store.binding,((0,1),),(0,1),1,0,0)
        self.assertIsNone(store.asserting_frontier(frames,learned))
        self.assertEqual(store.stats['lcg_asserting_jumps'],0)

    def test_empty_root_proof_consumes_all_pending_rectangles(self):
        store=self.store([7]*2)
        frames=[{'domains':[7,7],'cell':0,'choices':[3,4],'next':1},
                {'domains':[3,7],'cell':-1,'choices':[],'next':0}]
        learned=LearnedClause(store.binding,(),None,0,0,0)
        self.assertEqual(store.asserting_frontier(frames,learned),[])
        self.assertEqual(store.stats['lcg_asserting_levels'],2)

    def test_assertion_requires_same_process_local_binding(self):
        store=self.store([7]*2)
        learned=LearnedClause(('other','gold',(7,7)),((0,3),),(0,3),1,0,0)
        with self.assertRaises(ValueError):store.asserting_frontier([],learned)
        with self.assertRaises(ValueError):store.asserting_frontier([],learned.__dict__)

    def test_first_uip_retains_intermediate_fact_not_decision_flattening(self):
        _, child = self.diamond()
        result = child.first_uip([2, 3])  # c=0 AND d=0 is a constraint conflict.
        self.assertEqual(result.literals, ((1, 1),))
        self.assertEqual(result.uip, (1, 1))
        self.assertEqual(result.level, 1)
        self.assertEqual(result.backjump_level, 0)
        self.assertGreaterEqual(result.resolutions, 2)
        self.assertNotEqual(result.literals, ((0, 1),))
        clauses = (((0, 1), (1, 2)), ((1, 1), (2, 2)),
                   ((1, 1), (3, 2)), ((2, 1), (3, 1)))
        models = [row for row in itertools.product(range(2), repeat=4)
                  if all(satisfies(row, c) for c in clauses)]
        self.assertTrue(models)
        self.assertTrue(all(satisfies(row, result.literals) for row in models))
        store = self.store()
        self.assertTrue(store.learn(result))
        values = [3]*4
        store.prune(values, lambda: None)
        self.assertEqual(values, [3, 2, 3, 3])
        self.assertTrue(child.validate_graph())

    def test_mask_decisions_and_unknown_literal_remove_whole_pattern(self):
        root = self.state([7, 7])
        child = root.fork(0, 3)  # Two orientations share one connection pattern.
        child.record_cell(1, 6, [0], 'pattern support')
        result = child.first_uip([0, 1])
        self.assertEqual(result.literals, ((0, 3),))
        store = self.store([7, 7])
        store.learn(result)
        values = [7, 7]
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values, [4, 7])
        sibling = root.fork(0, 4)
        self.assertEqual(sibling.domains, [4, 7])
        self.assertTrue(sibling.validate_graph())

    def test_empty_fact_is_resolved_even_if_it_is_only_current_level_fact(self):
        child = self.state([3, 3]).fork(0, 1)
        child.record_cell(1, 0, [0], 'sound failed support')
        result = child.first_uip()
        self.assertEqual(result.literals, ((0, 1),))
        self.assertTrue(all(mask for _, mask in result.literals))
        self.assertIsNone(self.state([3, 3]).first_uip())

    def test_lower_decision_level_and_exact_backjump_level(self):
        child = self.state([3]*5).fork(0, 1).fork(1, 1)
        child.record_cell(2, 1, [1], 'second branch implies c')
        child.record_cell(3, 1, [2], 'c implies d')
        child.record_cell(4, 1, [2], 'c implies e')
        result = child.first_uip([0, 3, 4])
        self.assertEqual(result.literals, ((0, 1), (2, 1)))
        self.assertEqual(result.uip, (2, 1))
        self.assertEqual((result.level, result.backjump_level), (2, 1))

    def test_reconstruct_uses_actual_levels_and_all_path_restrictions(self):
        rebased = LCGState.reconstruct([7]*4, [3, 1, 2, 4], [(0, 3), (1, 1)],
                                       fingerprint='fixture')
        self.assertEqual(rebased.level, 2)
        self.assertEqual(rebased.facts[rebased.current[2]].level, 2)
        result = rebased.first_uip([2, 3])
        self.assertEqual(result.literals, ((0, 3), (1, 1)))
        self.assertEqual((result.level, result.backjump_level), (2, 1))
        self.assertTrue(rebased.validate_graph())
        with self.assertRaises(ValueError):
            LCGState.reconstruct([3]*2, [3, 3], [(0, 1)], fingerprint='fixture')

    def test_bulk_outputs_capture_pre_call_facts_and_validate_narrowing(self):
        child = self.state().fork(0, 1).fork(1, 2)
        before = child.current.copy()
        count = child.record([1, 2, 1, 1], [0], 'simultaneous output')
        self.assertEqual(count, 2)
        a, b = (child.facts[child.current[c]] for c in (2, 3))
        self.assertEqual(a.parents, (before[0], before[2]))
        self.assertEqual(b.parents, (before[0], before[3]))
        self.assertEqual((a.level, b.level), (1, 1))
        with self.assertRaises(ValueError): child.record_cell(2, 3, [], 'widen')
        with self.assertRaises(ValueError): child.record([1, 2, 3, 1], [], 'widen')
        self.assertTrue(child.validate_graph())

    def test_bounded_fact_memory_loses_only_learning_and_siblings_recover(self):
        root = self.state([7, 7], max_facts=4)
        a = root.fork(0, 3)
        a.record_cell(1, 3, [0], 'rule')
        a.record_cell(1, 1, [0], 'later rule exceeds cap')
        self.assertFalse(a.enabled)
        self.assertEqual(a.domains, [3, 1])
        self.assertIsNone(a.first_uip([0, 1]))
        self.assertEqual(len(a.facts), 4)
        self.assertEqual(a.budget_skips, 1)
        del a
        b = root.fork(0, 4)
        self.assertTrue(b.enabled)
        self.assertEqual(len(b.facts), 3)
        self.assertEqual(root.domains, [7, 7])

    def test_binding_rejects_cross_target_root_and_serialized_proof(self):
        _, child = self.diamond()
        result = child.first_uip([2, 3])
        for store in (LCGStore('other', 'gold', [3]*4),
                      LCGStore('fixture', 'edges', [3]*4), self.store([7]*4)):
            with self.assertRaises(ValueError): store.learn(result)
        store = self.store()
        with self.assertRaises(ValueError): store.learn(result.__dict__)
        self.assertFalse(store.matches('fixture', 'gold', [7]*4))
        with self.assertRaises(ValueError): store.prune([7]*4, lambda: None)

    def test_subsumption_masks_and_bounded_eviction(self):
        store = self.store([7]*3, limit=2, literal_limit=3)
        def learned(lits): return LearnedClause(store.binding, tuple(lits), None, 0, 0, 0)
        store.learn(learned([(0, 1), (1, 1)]))
        store.learn(learned([(0, 3), (1, 1)]))
        self.assertEqual(store.clauses, [((0, 3), (1, 1))])
        self.assertFalse(store.learn(learned([(0, 1), (1, 1), (2, 1)])))
        store.learn(learned([(2, 1)]))
        store.learn(learned([(1, 2)]))
        self.assertLessEqual(len(store), 2)
        self.assertLessEqual(sum(map(len, store.clauses)), 3)
        self.assertGreater(store.stats['lcg_evictions'], 0)
        self.assertFalse(store.learn(learned([(0, 0)])))
        self.assertTrue(store.learn(learned([])))
        self.assertEqual(store.clauses, [()])
        domains = [7]*3
        self.assertTrue(store.prune(domains, lambda: None))
        self.assertEqual(domains[0], 0)

    def test_pruning_reasons_record_sound_narrowing_and_cancellation(self):
        _, child = self.diamond()
        store = self.store()
        store.learn(child.first_uip([2, 3]))
        values = [3]*4
        trace = self.state()
        def reason(c, value, deps): trace.record_cell(c, value, deps, 'learned nogood')
        class Stopped(Exception): pass
        def stop(): raise Stopped()
        with self.assertRaises(Stopped): store.prune(values, stop, reason)
        self.assertEqual(values, [3]*4)
        store.prune(values, lambda: None, reason)
        self.assertEqual(values, trace.domains)
        self.assertEqual(trace.facts[trace.current[1]].level, 0)
        self.assertTrue(trace.validate_graph())

    def test_bulk_reason_is_shared_immutable_and_unchanged_calls_are_lazy(self):
        child = self.state([7]*160).fork(0, 3)
        before = tuple(child.current)
        outputs = [3]*160
        self.assertEqual(child.record(outputs, range(160), 'global rule'), 159)
        reasons = [child.facts[child.current[c]].parents for c in range(1, 160)]
        self.assertEqual(len({id(reason.shared) for reason in reasons}), 1)
        self.assertEqual(reasons[0].shared.refs, before)
        with self.assertRaises(FrozenInstanceError): reasons[0].shared.level = 7
        child.record_cell(0, 1, [], 'later narrowing')
        self.assertEqual(reasons[0].shared.refs, before)
        self.assertTrue(child.validate_graph())
        def must_not_read():
            raise AssertionError('Unchanged bulk output must not construct a reason.')
            yield 0
        self.assertEqual(child.record(child.domains, must_not_read(), 'no output'), 0)

    def test_shared_reasons_survive_sibling_truncation_and_bounded_recording(self):
        root = self.state([7]*8, max_facts=16)
        child = root.fork(0, 3)
        child.record([3]*8, range(8), 'root path implies restrictions')
        self.assertTrue(child.validate_graph())
        child.record_cell(1, 1, [0], 'fact cap')
        self.assertFalse(child.enabled)
        sibling = root.fork(0, 4)
        sibling.record([4]*8, range(8), 'sibling restrictions')
        self.assertTrue(sibling.enabled)
        self.assertTrue(sibling.validate_graph())
        self.assertEqual(sibling.first_uip([1, 2]).literals, ((0, 4),))
        self.assertEqual(root.domains, [7]*8)

    def test_rebased_lazy_graph_accepts_completed_prefilter_but_keeps_whole_path(self):
        trace = LCGState.reconstruct([7]*6, [3, 1, 2, 4, 2, 4],
                                     [(0, 3), (1, 1)], fingerprint='fixture')
        refs = [trace.facts[trace.current[c]].parents for c in range(2, 6)]
        self.assertEqual(len({id(p.shared) for p in refs}), 1)
        self.assertEqual(trace.first_uip([4, 5]).literals, ((0, 3), (1, 1)))
        self.assertTrue(trace.validate_graph())

    def test_indexed_clauses_revisit_late_effects_widening_and_new_learning(self):
        store = self.store([3]*4)
        def learn(lits): store.learn(LearnedClause(store.binding, tuple(lits), None, 0, 0, 0))
        learn([(0, 1), (1, 1)])
        learn([(1, 2), (2, 1)])
        values = [3]*4
        self.assertFalse(store.prune(values, lambda: None))
        checks = store.stats['lcg_checks']
        self.assertFalse(store.prune(values, lambda: None))
        self.assertEqual(store.stats['lcg_checks'], checks)
        values[3] = 1  # An unrelated cell need not wake either clause.
        self.assertFalse(store.prune(values, lambda: None))
        self.assertEqual(store.stats['lcg_checks'], checks)
        values[2] = 1
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values, [3, 1, 1, 1])
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values, [2, 1, 1, 1])
        values[:] = [1, 3, 3, 3]  # Backtracking/widening is also an event.
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values, [1, 2, 2, 3])
        learn([(3, 1)])
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values[3], 2)
        self.assertGreater(store.stats['lcg_clause_skips'], 0)

    def test_indexed_pruning_repeated_fresh_input_reapplies_all_units(self):
        store = self.store([3]*4)
        store.learn(LearnedClause(store.binding, ((0, 1),), None, 0, 0, 0))
        store.learn(LearnedClause(store.binding, ((1, 1), (2, 2)), None, 0, 0, 0))
        for _ in range(5):
            values = [3, 1, 3, 3]
            self.assertTrue(store.prune(values, lambda: None))
            self.assertEqual(values, [2, 1, 1, 3])

    def test_indexed_pruning_interrupt_does_not_mark_unvisited_clauses_complete(self):
        store = self.store([3]*4)
        for cell in range(4):
            store.learn(LearnedClause(store.binding, ((cell, 1),), None, 0, 0, 0))
        class Stopped(Exception): pass
        def reason(cell, value, deps):
            if cell == 1: raise Stopped()
        values = [3]*4
        with self.assertRaises(Stopped): store.prune(values, lambda: None, reason)
        self.assertEqual(values, [2, 2, 3, 3])
        self.assertTrue(store.prune(values, lambda: None))
        self.assertEqual(values, [2]*4)
        self.assertFalse(store.prune(values, lambda: None))

    def test_indexed_clause_fixpoints_match_independent_full_scan(self):
        rng = random.Random(17363)
        for case in range(16):
            store = self.store([7]*5)
            for _ in range(14):
                cells = rng.sample(range(5), rng.randrange(1, 4))
                clause = tuple((c, rng.randrange(1, 7)) for c in cells)
                store.learn(LearnedClause(store.binding, clause, None, 0, 0, 0))
            def full_scan(values):
                while True:
                    before = values.copy()
                    for clause in store.clauses:
                        unknown = [(c, m) for c, m in clause if values[c] & ~m]
                        if any(not values[c] & m for c, m in clause): continue
                        if not unknown: values[0] = 0; return values
                        if len(unknown) == 1:
                            c, m = unknown[0]; values[c] &= ~m
                    if values == before: return values
            for _ in range(30):
                values = [rng.randrange(1, 8) for _ in range(5)]
                expected = full_scan(values.copy())
                for _ in range(20):
                    old = values.copy()
                    store.prune(values, lambda: None)
                    if not all(values) or values == old: break
                self.assertEqual(bool(all(values)), bool(all(expected)), (case, values, expected))
                if all(values): self.assertEqual(values, expected)

    def test_lazy_bulk_clauses_preserve_every_independent_finite_solution(self):
        rng = random.Random(19402)
        total_learned = total_solutions = 0
        for case in range(20):
            source = [tuple((c, rng.randrange(1, 7)) for c in rng.sample(range(4), rng.randrange(1, 4)))
                      for _ in range(10)]
            expected = {row for row in itertools.product(range(3), repeat=4)
                        if all(satisfies(row, clause) for clause in source)}
            total_solutions += len(expected)
            store = self.store([7]*4)
            found = set()
            def visit(trace):
                nonlocal total_learned
                while True:
                    values = trace.domains.copy()
                    before = values.copy()
                    for clause in source:
                        if any(not values[c] & m for c, m in clause): continue
                        unknown = [(c, m) for c, m in clause if values[c] & ~m]
                        if not unknown: values[0] = 0; break
                        if len(unknown) == 1:
                            c, m = unknown[0]; values[c] &= ~m
                    trace.record(values, range(4), 'sound bulk input-rule fixpoint')
                    if all(values):
                        store.prune(values, lambda: None,
                                    lambda c, v, deps: trace.record_cell(c, v, deps, 'learned unit'))
                    if not all(values):
                        learned = trace.first_uip()
                        self.assertIsNotNone(learned)
                        self.assertTrue(all(satisfies(row, learned.literals) for row in expected))
                        store.learn(learned)
                        total_learned += 1
                        return
                    if values == before: break
                self.assertTrue(trace.validate_graph())
                if all(v.bit_count() == 1 for v in values):
                    found.add(tuple(v.bit_length()-1 for v in values)); return
                cell = next(c for c, v in enumerate(values) if v.bit_count() > 1)
                for bit in (1, 2, 4):
                    if values[cell] & bit: visit(trace.fork(cell, bit))
            visit(self.state([7]*4))
            self.assertEqual(found, expected, case)
        self.assertGreater(total_learned, 0)
        FINITE_ORACLE_SUMMARY.update(lazy_families=20, lazy_assignments_examined=1620,
                                    lazy_learned_clauses_validated=total_learned,
                                    lazy_satisfying_assignments=total_solutions)

    def test_every_learned_clause_preserves_exhaustive_finite_solutions(self):
        rng = random.Random(911307)
        learned_count = 0
        intermediate_count = 0
        solutions_count = 0
        for case in range(33):
            n = 4 + case % 2
            source = []
            for _ in range(9 + case % 5):
                cells = rng.sample(range(n), rng.randrange(1, 4))
                source.append(tuple((c, rng.randrange(1, 7)) for c in cells))
            if case == 32:
                source = [((0, 1), (1, 6)), ((1, 1), (2, 6)),
                          ((1, 1), (3, 6)), ((2, 1), (3, 1))]
            expected = {row for row in itertools.product(range(3), repeat=n)
                        if all(satisfies(row, clause) for clause in source)}
            solutions_count += len(expected)
            store = self.store([7]*n)
            found = set()
            def conflict(trace, cells):
                nonlocal learned_count, intermediate_count
                result = trace.first_uip(cells)
                self.assertIsNotNone(result)
                self.assertTrue(all(satisfies(row, result.literals) for row in expected),
                                (case, result, expected))
                if result.uip and not any(f.decision and (f.cell, f.mask) == result.uip
                                          for f in trace.facts):
                    intermediate_count += 1
                store.learn(result)
                learned_count += 1
            def visit(trace):
                # Independent reference unit propagation for the finite input
                # constraints; the LCG store sees only its own proved clauses.
                while True:
                    old = tuple(trace.domains)
                    for clause in source:
                        unknown = []
                        inactive = False
                        for c, mask in clause:
                            assigned_values = [v for v in range(3) if trace.domains[c] & (1 << v)]
                            bad_values = [v for v in assigned_values if mask & (1 << v)]
                            if not bad_values:
                                inactive = True
                                break
                            if len(bad_values) != len(assigned_values): unknown.append((c, mask))
                        if inactive: continue
                        if not unknown:
                            conflict(trace, [c for c, _ in clause])
                            return
                        if len(unknown) == 1:
                            c, mask = unknown[0]
                            trace.record_cell(c, trace.domains[c] & ~mask,
                                              [k for k, _ in clause if k != c], 'finite constraint')
                    domains = trace.domains.copy()
                    store.prune(domains, lambda: None,
                                lambda c, v, deps: trace.record_cell(c, v, deps, 'learned finite clause'))
                    if not all(domains):
                        conflict(trace, [c for c, v in enumerate(domains) if not v])
                        return
                    if tuple(domains) == old: break
                if all(v.bit_count() == 1 for v in trace.domains):
                    found.add(tuple(v.bit_length()-1 for v in trace.domains))
                    return
                cell = min(c for c, v in enumerate(trace.domains) if v.bit_count() > 1)
                for value in range(3):
                    if trace.domains[cell] & (1 << value):
                        visit(trace.fork(cell, 1 << value))
            visit(self.state([7]*n))
            self.assertEqual(found, expected, case)
            self.assertLessEqual(sum(map(len, store.clauses)), store.literal_limit)
        self.assertGreater(learned_count, 0)
        self.assertGreater(intermediate_count, 0)
        FINITE_ORACLE_SUMMARY.update(families=33, assignments_examined=5265,
                                    learned_clauses_validated=learned_count,
                                    intermediate_uips=intermediate_count,
                                    satisfying_assignments=solutions_count)


    def test_cached_subsumption_metadata_matches_independent_forbidden_boards(self):
        # Set containment of explicitly enumerated forbidden assignments is an
        # independent oracle for mask-nogood subsumption, including eviction.
        rng=random.Random(121503);n=4;boards=tuple(itertools.product(range(3),repeat=n))
        store=self.store([7]*n,limit=5,literal_limit=12);expected=[]
        def forbidden(clause):
            return {row for row in boards if not satisfies(row,clause)}
        for event in range(400):
            literals=[(cell,rng.randrange(1,7)) for cell in rng.sample(range(n),rng.randrange(2,5))]
            if event%19==0:literals.append(literals[0])
            if event%23==0:literals.append((literals[0][0],7))
            allowed={}
            for cell,mask in literals:
                choices={v for v in range(3) if mask&(1<<v)}
                allowed[cell]=allowed.get(cell,set(range(3)))&choices
            clause=tuple(sorted((cell,sum(1<<v for v in choices))
                                for cell,choices in allowed.items() if len(choices)<3))
            rejected=any(not choices for choices in allowed.values())
            covered=forbidden(clause)
            if not rejected and not any(covered<=forbidden(old) for old in expected):
                expected=[old for old in expected if not forbidden(old)<=covered]
                expected.append(clause)
                while len(expected)>store.limit or sum(map(len,expected))>store.literal_limit:
                    victim=max(range(len(expected)),key=lambda i:(len(expected[i]),-i))
                    expected.pop(victim)
                accepted=clause in expected
            else:accepted=False
            actual=store.learn(LearnedClause(store.binding,tuple(literals),None,0,0,0))
            self.assertEqual(actual,accepted,event)
            self.assertEqual(store.clauses,expected,event)
            self.assertEqual(set(store._clause_data),set(expected) if len(expected)>=4 else set())
            self.assertLessEqual(sum(len(mapping) for _,mapping in store._clause_data.values()),store.literal_limit)
            for current,(cells,mapping) in store._clause_data.items():
                self.assertEqual(cells,sum(1<<cell for cell,_ in current))
                self.assertEqual(mapping,dict(current))
        self.assertGreater(store.stats['lcg_evictions'],0)
        self.assertGreater(store.stats['lcg_subsumed'],0)

    def test_subsumption_metadata_handles_high_cells_and_empty_clause(self):
        store=self.store([7]*160)
        for clause in (((159,1),(0,1)),((0,3),(159,1)),((0,4),(159,1))):
            self.assertTrue(store.learn(LearnedClause(store.binding,clause,None,0,0,0)))
        self.assertEqual(store.clauses,[((0,3),(159,1)),((0,4),(159,1))])
        self.assertFalse(store._clause_data)  # Preserve the tiny-store fast path.
        for clause in (((1,1),(158,1)),((2,1),(157,1))):
            self.assertTrue(store.learn(LearnedClause(store.binding,clause,None,0,0,0)))
        self.assertEqual(len(store._clause_data),4)
        self.assertTrue(store.learn(LearnedClause(store.binding,((159,1),),None,0,0,0)))
        self.assertEqual(len(store.clauses),3)
        self.assertFalse(store._clause_data)
        self.assertTrue(store.learn(LearnedClause(store.binding,(),None,0,0,0)))
        self.assertEqual(store.clauses,[()])
        self.assertEqual(store._clause_data,{})
        self.assertFalse(store.learn(LearnedClause(store.binding,((159,1),),None,0,0,0)))
        self.assertEqual(store._clause_data,{})

if __name__ == '__main__': unittest.main()
