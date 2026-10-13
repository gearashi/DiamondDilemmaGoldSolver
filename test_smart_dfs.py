"""Finite soundness checks for strong DFS pruning and trusted conflict learning."""
from itertools import product
import copy
import json
import random
import unittest
from unittest.mock import patch

import constraint_dfs as dfs
from constraint_pruning import STAT_KEYS as PRUNER_STAT_KEYS
from search_problem import SearchProblem
from test_constraint_dfs import (board, brute, double_face, multiple_loops,
                                 simple_tiles, tetrahedron_tiles)


LEGACY_OPTIONS = dict(failure_guided=False,lookahead=False,strand_connectivity=False,
                      multi_seam_cuts=False,regional_cache=False,biconnected=False,separator_regions=False,
                      backjumping=False,incremental=False,adaptive=False,persistent_learning=False)


def legacy_solve(*args,**kwargs):
    # Earlier tests deliberately isolate their original deductions/leaf order.
    # Round-four tests below exercise new features explicitly and together.
    kwargs['options']={**LEGACY_OPTIONS,**(kwargs.get('options') or {})}
    return dfs.solve(*args,**kwargs)


def mixed_loop_problem():
    # Equal edge masks allow every rotation. Equal relative rotations produce
    # three loops; the other relative rotations produce a single Gold loop.
    segments = [[[0, 2], [0, 10]], [[1, 2], [2, 2]], [[1, 10], [2, 10]]]
    return SearchProblem([{'id': name, 'segments': copy.deepcopy(segments)}
                          for name in ('A', 'B')], double_face())


def conflict_stats():
    return dict.fromkeys(dfs.CONFLICT_STAT_KEYS, 0)


class NogoodLogicTests(unittest.TestCase):
    def test_unit_filter_never_removes_a_satisfying_completion(self):
        clauses = [((0, 0), (1, 3)), ((1, 4), (2, 6)), ((0, 2), (2, 8))]
        store, stats = dfs._NogoodStore('fixture', 'gold'), conflict_stats()
        for clause in clauses:
            store.learn(clause, stats)
        # Every nonempty subset of three rotations at each of three cells.
        for subsets in product(range(1, 8), repeat=3):
            original = [subset << (3*cell) for cell, subset in enumerate(subsets)]
            domains = original.copy()
            choices = [[3*cell+r for r in range(3) if subset & (1 << r)]
                       for cell, subset in enumerate(subsets)]
            legal = [codes for codes in product(*choices)
                     if not any(all(codes[cell] == code for cell, code in clause)
                                for clause in clauses)]
            while store.prune(domains, stats, lambda: None) and all(domains):
                pass
            for codes in legal:
                self.assertTrue(all(domain & (1 << code) for domain, code in zip(domains, codes)),
                                (subsets, codes, domains))
            if not all(domains):
                self.assertEqual(legal, [])
        self.assertGreater(stats['conflict_prunes'], 0)
        self.assertGreater(stats['conflict_hits'], 0)

    def test_irrelevant_assignments_do_not_defeat_a_proven_loop_conflict(self):
        geometry = board(4, [(0, s, 1, s) for s in range(3)]
                            + [(2, s, 3, s) for s in range(3)])
        tiles = [{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(4)]
        problem = SearchProblem(tiles, geometry)
        support = problem.gold_nogood([0, 3, -1, -1])
        self.assertEqual(support, [(0, 0), (1, 3)])
        store, stats = dfs._NogoodStore(problem.fingerprint, 'gold'), conflict_stats()
        store.learn(support, stats)
        for irrelevant in ([1 << 6, 1 << 9], [7 << 6, 7 << 9], [7 << 9, 7 << 6]):
            domains = [1, (1 << 3) | (1 << 4), *irrelevant]
            store.prune(domains, stats, lambda: None)
            self.assertEqual(domains[1], 1 << 4)
            self.assertEqual(domains[2:], irrelevant)

    def test_subsumption_and_bounded_eviction_only_weaken_the_cache(self):
        store, stats = dfs._NogoodStore('fixture', 'gold', limit=2), conflict_stats()
        store.learn([(0, 0), (1, 3), (2, 6)], stats)
        store.learn([(0, 0), (1, 3)], stats)
        store.learn([(0, 0), (1, 3), (2, 7)], stats)
        self.assertEqual(store.clauses, [frozenset(((0, 0), (1, 3)))])
        store.learn([(0, 1), (1, 4)], stats)
        store.learn([(0, 2), (1, 5)], stats)
        self.assertEqual(len(store.clauses), 2)
        self.assertEqual(stats['conflict_evictions'], 1)
        self.assertGreaterEqual(stats['conflicts_subsumed'], 2)
        # An evicted conflict is allowed through to ordinary DFS, not labelled
        # solved or infeasible by the cache.
        domains = [1 << 0, 1 << 3]
        self.assertFalse(store.prune(domains, stats, lambda: None))
        self.assertEqual(domains, [1, 8])


class SmartDFSIntegrationTests(unittest.TestCase):
    def test_randomized_tiny_puzzles_agree_with_independent_bruteforce(self):
        rng = random.Random(8127)
        for case in range(8):
            tiles, geometry = tetrahedron_tiles()
            rng.shuffle(tiles)
            for tile in tiles:
                rotation = rng.randrange(3)
                tile['segments'] = [[[(side+rotation) % 3, p] for side, p in segment]
                                    for segment in tile['segments']]
            if case % 2:
                endpoint = tiles[0]['segments'][0][0]
                endpoint[1] = endpoint[1] % 11 + 1
            fixed = {0: rng.randrange(12)} if case % 3 == 2 else None
            problem = SearchProblem(tiles, geometry, fixed)
            for target in ('gold', 'edges'):
                with self.subTest(case=case, target=target):
                    expected = brute(problem, target)
                    result = legacy_solve(problem, seconds=5, seed=case, target=target)
                    self.assertTrue(result['complete'], result)
                    if expected:
                        self.assertIn(tuple(result['codes']), expected)
                    else:
                        self.assertEqual(result['status'], 'infeasible')
                    self.assertTrue(set(PRUNER_STAT_KEYS) <= result['stats'].keys())
                    self.assertTrue(all(type(value) is int and value >= 0
                                        for value in result['stats'].values()))

    def test_failed_branch_learning_does_not_exclude_other_gold_solutions(self):
        problem, cache, seen = mixed_loop_problem(), {}, []
        bad = [0, 3]
        self.assertFalse(problem.validate(bad)['valid'])
        first = legacy_solve(problem, seconds=5, hints=bad, learning_state=cache,
                          stop=lambda: bool(seen),
                          progress=lambda event: seen.append(event['codes'])
                          if event['event'] == 'candidate' else None)
        self.assertEqual(first['status'], 'stopped')
        self.assertEqual(seen, [bad])
        self.assertGreater(first['stats']['conflicts_learned'], 0)
        self.assertNotIn(frozenset(), cache['_nogoods'].clauses)
        # Re-enter from the root with a different order: the failed conjunction
        # remains forbidden, while another Gold arrangement is still reachable.
        second_seen = []
        second = legacy_solve(problem, seconds=5, hints=bad, seed=29, learning_state=cache,
                           progress=lambda event: second_seen.append(event['codes'])
                           if event['event'] == 'candidate' else None)
        self.assertEqual(second['status'], 'solved')
        self.assertTrue(problem.validate(second['codes'])['valid'])
        self.assertNotIn(bad, second_seen)
        self.assertGreater(second['stats']['conflict_prunes'], 0)

    def test_multiple_slices_reuse_cache_or_safely_relearn_after_restart(self):
        for retain_cache in (False, True):
            with self.subTest(retain_cache=retain_cache):
                problem, cache, seen = mixed_loop_problem(), {}, []
                first = legacy_solve(problem, seconds=5, hints=[0, 3], learning_state=cache,
                                  stop=lambda: bool(seen),
                                  progress=lambda event: seen.append(tuple(event['codes']))
                                  if event['event'] == 'candidate' else None)
                checkpoint = json.loads(json.dumps(first['checkpoint']))
                self.assertFalse(any('learn' in key or 'nogood' in key for key in checkpoint))
                final = legacy_solve(problem, seconds=5, resume=checkpoint,
                                  learning_state=cache if retain_cache else {},
                                  progress=lambda event: seen.append(tuple(event['codes']))
                                  if event['event'] == 'candidate' else None)
                self.assertEqual(final['status'], 'solved')
                self.assertEqual(len(seen), len(set(seen)))
                self.assertGreaterEqual(final['stats']['conflicts_learned'],
                                        first['stats']['conflicts_learned'])
                self.assertTrue(problem.validate(final['codes'])['valid'])

    def test_stop_inside_strong_pruning_retains_sound_domains_without_learning(self):
        tiles, geometry = tetrahedron_tiles()
        problem, cache = SearchProblem(tiles, geometry), {}
        real = dfs.Pruner
        requested = [False]
        class InterruptingPruner(real):
            def prune(self, domains):
                changed = super().prune(domains)
                requested[0] = True
                self.check()
                return changed
        with patch.object(dfs, 'Pruner', InterruptingPruner):
            first = legacy_solve(problem, seconds=5, stop=lambda: requested[0], learning_state=cache)
        self.assertEqual(first['status'], 'stopped')
        self.assertFalse(first['complete'])
        self.assertEqual(first['stats']['conflicts_learned'], 0)
        self.assertEqual(cache['_nogoods'].clauses, [])
        final = legacy_solve(problem, seconds=5, resume=json.loads(json.dumps(first['checkpoint'])),
                          learning_state=cache)
        self.assertEqual(final['status'], 'solved')
        self.assertTrue(problem.validate(final['codes'])['valid'])

    def test_timeout_and_requested_stop_never_learn_an_unproved_root_failure(self):
        problem = mixed_loop_problem()
        for options in ({'seconds': 0}, {'seconds': 5, 'stop': lambda: True}):
            cache = {}
            result = legacy_solve(problem, learning_state=cache, **options)
            self.assertFalse(result['complete'])
            self.assertEqual(result['stats']['conflicts_learned'], 0)
            self.assertEqual(cache, {})  # Symmetry setup has not bound a cache yet.
            self.assertEqual(legacy_solve(problem, seconds=5, learning_state=cache)['status'], 'solved')

    def test_learning_cache_rejects_cross_problem_target_or_serialized_values(self):
        problem, cache = mixed_loop_problem(), {}
        legacy_solve(problem, seconds=5, learning_state=cache)
        other = SearchProblem(simple_tiles(), double_face())
        for options in ({'problem': other, 'learning_state': cache},
                        {'problem': problem, 'learning_state': cache, 'target': 'edges'},
                        {'problem': problem, 'learning_state': {'_nogoods': []}},
                        {'problem': problem, 'learning_state': []}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                legacy_solve(seconds=0, **options)

    def test_learning_cache_rejects_changed_effective_domains(self):
        problem, cache = mixed_loop_problem(), {}
        original = copy.deepcopy(problem.domains)
        # Restrict the same model to a known multiloop board. Exhaustion learns
        # the empty clause only for this restricted effective search problem.
        problem.domains = [[0], [3]]
        restricted = legacy_solve(problem, seconds=5, learning_state=cache)
        self.assertEqual(restricted['status'], 'infeasible')
        self.assertIn(frozenset(), cache['_nogoods'].clauses)
        problem.domains = original
        with self.assertRaisesRegex(ValueError, 'effective domains'):
            legacy_solve(problem, seconds=5, learning_state=cache)
        result = legacy_solve(problem, seconds=5, learning_state={})
        self.assertEqual(result['status'], 'solved')
        self.assertTrue(problem.validate(result['codes'])['valid'])

    def test_legacy_v2_checkpoint_missing_new_counters_resumes(self):
        problem = SearchProblem(multiple_loops(), double_face())
        seen = []
        first = legacy_solve(problem, seconds=5, stop=lambda: bool(seen),
                          progress=lambda event: seen.append(tuple(event['codes']))
                          if event['event'] == 'candidate' else None)
        checkpoint = json.loads(json.dumps(first['checkpoint']))
        checkpoint['stats'] = {key: checkpoint['stats'][key] for key in dfs.BASE_STAT_KEYS}
        checkpoint.pop('root_symmetry', None)  # Actual metadata-free v2 format.
        checkpoint['digest'] = dfs._checkpoint_digest(checkpoint)
        self.assertTrue(any(frame['cell'] >= 0 for frame in checkpoint['stack']))
        final = legacy_solve(problem, seconds=5, resume=checkpoint,
                          progress=lambda event: seen.append(tuple(event['codes']))
                          if event['event'] == 'candidate' else None)
        self.assertEqual(final['status'], 'infeasible')
        self.assertEqual(len(seen), 18)
        self.assertEqual(len(set(seen)), 18)
        self.assertEqual(final['stats']['leaves'], 18)
        self.assertTrue(set(PRUNER_STAT_KEYS) <= final['stats'].keys())

    def test_checkpoint_never_trusts_added_serialized_conflicts(self):
        problem = SearchProblem(simple_tiles(), double_face())
        checkpoint = legacy_solve(problem, seconds=0)['checkpoint']
        checkpoint['learned_clauses'] = [[]]  # A fabricated claim of global UNSAT.
        checkpoint['digest'] = dfs._checkpoint_digest(checkpoint)
        result = legacy_solve(problem, seconds=5, resume=checkpoint)
        self.assertEqual(result['status'], 'solved')


class PropagationOptimizationTests(unittest.TestCase):
    def test_grouped_arc_filter_matches_per_code_support_exhaustively(self):
        for problem in (mixed_loop_problem(), SearchProblem(simple_tiles(), double_face())):
            initial = tuple(sum(1 << code for code in row) for row in problem.domains)
            tables = dfs._build_support_tables(problem, initial, 'gold', lambda: None)
            for sa, sb in product(range(3), repeat=2):
                for source, neighbor in product(range(1, 1 << (3*problem.n)), repeat=2):
                    expected = sum(1 << code for code in range(3*problem.n)
                                   if source & (1 << code) and any(
                                       neighbor & (1 << other) and code//3 != other//3
                                       and int(problem.masks[code,sa]) == dfs.reverse_mask(problem.masks[other,sb])
                                       for other in range(3*problem.n)))
                    actual = dfs._filter_arc(source, neighbor, tables.groups[sa][sb],
                                             tables.compatible[sa][sb], tables.tile_bits)
                    self.assertEqual(actual, expected, (sa,sb,source,neighbor))

    def test_multiple_rotations_of_one_neighbor_tile_do_not_support_itself(self):
        problem = mixed_loop_problem()
        tables = dfs._build_support_tables(problem, (63,63), 'gold', lambda: None)
        self.assertEqual(dfs._filter_arc(63, (1 << 3) | (1 << 4), tables.groups[0][0],
                                         tables.compatible[0][0], tables.tile_bits), 7)
        self.assertEqual(dfs._filter_arc(63, (1 << 0) | (1 << 3), tables.groups[0][0],
                                         tables.compatible[0][0], tables.tile_bits), 63)

    def test_grouped_solver_preserves_reference_leaf_order(self):
        def scan(domain, neighbor, groups, compatible, tile_bits):
            return sum(1 << code for code in dfs._values(domain) if compatible[code] & neighbor)
        problem = SearchProblem(multiple_loops(), double_face())
        for seed in (0, 14, 71):
            observed = []
            for use_reference in (True, False):
                seen = []
                def progress(event):
                    if event['event'] == 'candidate':
                        seen.append(tuple(event['codes']))
                if use_reference:
                    with patch.object(dfs, '_filter_arc', scan):
                        result = legacy_solve(problem, seconds=5, seed=seed, progress=progress)
                else:
                    result = legacy_solve(problem, seconds=5, seed=seed, progress=progress)
                self.assertEqual(result['status'], 'infeasible')
                self.assertEqual(len(seen), 18)
                observed.append(seen)
            self.assertEqual(*observed)

    def test_immutable_support_cache_is_reused_across_slices(self):
        from dataclasses import FrozenInstanceError
        problem, cache, seen = mixed_loop_problem(), {}, []
        first = legacy_solve(problem, seconds=5, hints=[0,3], learning_state=cache,
                          stop=lambda: bool(seen), progress=lambda event: seen.append(event)
                          if event['event'] == 'candidate' else None)
        tables = cache['_support']
        with self.assertRaises(FrozenInstanceError):
            tables.target = 'edges'
        with patch.object(dfs, '_build_support_tables', side_effect=AssertionError('Reuse immutable tables')):
            final = legacy_solve(problem, seconds=5, resume=first['checkpoint'], learning_state=cache)
        self.assertIs(cache['_support'], tables)
        self.assertEqual(final['status'], 'solved')
        self.assertTrue(problem.validate(final['codes'])['valid'])
        cache['_support'] = {'serialized': 'untrusted'}
        with self.assertRaisesRegex(ValueError, 'support cache'):
            legacy_solve(problem, seconds=0, learning_state=cache)

    def test_interrupted_support_setup_never_caches_an_incomplete_table(self):
        problem, cache, requested = mixed_loop_problem(), {}, [False]
        real_build = dfs._build_support_tables
        def interrupted(problem, initial, target, check):
            def cancel():
                requested[0] = True
                check()
            return real_build(problem, initial, target, cancel)
        with patch.object(dfs, '_build_support_tables', interrupted):
            first = legacy_solve(problem, seconds=5, learning_state=cache, stop=lambda:requested[0])
        self.assertEqual(first['status'], 'stopped')
        self.assertNotIn('_support', cache)
        final = legacy_solve(problem, seconds=5, learning_state=cache, resume=first['checkpoint'])
        self.assertEqual(final['status'], 'solved')


class RoundThreeOptimizationTests(unittest.TestCase):
    def test_connectivity_is_gold_only_and_can_reject_before_branching(self):
        geometry=board(4,[(0,s,1,s) for s in range(3)]+[(2,s,3,s) for s in range(3)])
        tiles=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        problem=SearchProblem(tiles,geometry)
        gold=legacy_solve(problem,seconds=5,target='gold')
        self.assertEqual(gold['status'],'infeasible')
        self.assertEqual(gold['nodes'],1)
        self.assertEqual(gold['stats']['connectivity_failures'],1)
        self.assertEqual(gold['stats']['leaves'],0)
        with patch.object(dfs,'ConnectivityPruner',side_effect=AssertionError('Gold rule in edge search')):
            edge=legacy_solve(problem,seconds=5,target='edges')
        self.assertEqual(edge['status'],'edge_perfect')
        self.assertTrue(all(edge['stats'][key]==0 for key in dfs.CONNECTIVITY_STAT_KEYS))

    def test_interrupted_connectivity_pass_resumes_with_cumulative_counters(self):
        problem=mixed_loop_problem();requested=[False]
        real=dfs.ConnectivityPruner
        class InterruptedConnectivity(real):
            def prune(self,domains,check):
                changed=super().prune(domains,check)
                requested[0]=True
                check()
                return changed
        with patch.object(dfs,'ConnectivityPruner',InterruptedConnectivity):
            first=legacy_solve(problem,seconds=5,stop=lambda:requested[0])
        self.assertEqual(first['status'],'stopped')
        self.assertEqual(first['stats']['conflicts_learned'],0)
        self.assertGreater(first['stats']['connectivity_checks'],0)
        saved=json.loads(json.dumps(first['checkpoint']))
        final=legacy_solve(problem,seconds=5,resume=saved)
        self.assertEqual(final['status'],'solved')
        self.assertTrue(problem.validate(final['codes'])['valid'])
        self.assertGreater(final['stats']['connectivity_checks'],first['stats']['connectivity_checks'])
        # Legacy checkpoints missing all new connectivity work counters remain usable.
        for key in dfs.CONNECTIVITY_STAT_KEYS:saved['stats'].pop(key)
        saved['digest']=dfs._checkpoint_digest(saved)
        self.assertEqual(legacy_solve(problem,seconds=5,resume=saved)['status'],'solved')

    def test_gathered_tile_ids_match_rotation_enumeration(self):
        rng = random.Random(90317)
        domains = list(range(4096)) + [rng.getrandbits(480) for _ in range(100)]
        domains += [1 << code for code in range(480)]
        for domain in domains:
            self.assertEqual(set(dfs._values(dfs._tile_choices(domain))),
                             {code//3 for code in dfs._values(domain)})

    def test_exact_arc_memo_keys_both_domains_and_sides_and_bounds_eviction(self):
        rng = random.Random(90318)
        problem = SearchProblem(simple_tiles(), double_face())
        tables = dfs._build_support_tables(problem, (63,63), 'gold', lambda:None)
        memo = dfs._ArcFilter(tables, limit=7)
        original = dfs._filter_arc
        with patch.object(dfs, '_filter_arc', wraps=original) as scanned:
            for _ in range(350):
                source, neighbor = rng.randrange(64), rng.randrange(64)
                sa, sb = rng.randrange(3), rng.randrange(3)
                expected = sum(1 << code for code in range(6)
                               if source & (1 << code) and any(
                                   neighbor & (1 << other) and code//3 != other//3
                                   and int(problem.masks[code,sa]) == dfs.reverse_mask(problem.masks[other,sb])
                                   for other in range(6)))
                self.assertEqual(memo(source,neighbor,sa,sb), expected)
                calls = scanned.call_count
                self.assertEqual(memo(source,neighbor,sa,sb), expected)
                self.assertEqual(scanned.call_count, calls)  # Includes cached zero.
                self.assertTrue(all(len(cache)<=7 for row in memo.results for cache in row))
        # Ownership/correspondence are tied to each memo's immutable table.
        different = mixed_loop_problem()
        other = dfs._ArcFilter(dfs._build_support_tables(different,(63,63),'gold',lambda:None))
        self.assertNotEqual(memo(1,16,0,0),other(1,16,0,0))

    def test_dirty_arc_schedule_preserves_exhaustive_leaf_order(self):
        problem = SearchProblem(multiple_loops(), double_face())
        for seed in (0,14,71):
            observed=[]
            for full in (True,False):
                seen=[]
                def notify(event):
                    if event['event']=='candidate':seen.append(tuple(event['codes']))
                if full:
                    with patch.object(dfs,'_starting_arcs',lambda count,incoming,changed:range(count)):
                        result=legacy_solve(problem,seconds=5,seed=seed,progress=notify)
                else:result=legacy_solve(problem,seconds=5,seed=seed,progress=notify)
                self.assertEqual(result['status'],'infeasible')
                self.assertEqual(len(seen),18)
                observed.append(seen)
            self.assertEqual(*observed)

    def test_dirty_schedule_is_ephemeral_and_interrupted_resume_rechecks_all_arcs(self):
        problem=SearchProblem(multiple_loops(),double_face())
        requested=[False];schedule=dfs._starting_arcs;observed=[]
        def interrupted(count,incoming,changed):
            observed.append(changed)
            if changed is not None:requested[0]=True
            return schedule(count,incoming,changed)
        with patch.object(dfs,'_starting_arcs',interrupted):
            result=legacy_solve(problem,seconds=5,stop=lambda:requested[0])
        self.assertEqual(result['status'],'stopped')
        self.assertTrue(any(cell is not None for cell in observed))
        checkpoint=json.loads(json.dumps(result['checkpoint']))
        self.assertEqual(checkpoint['version'],2)
        self.assertTrue(all(set(frame)=={'domains','cell','choices','next'} for frame in checkpoint['stack']))
        resumed=[];seen=[]
        def tracked(count,incoming,changed):
            resumed.append(changed)
            return schedule(count,incoming,changed)
        # Even a fabricated optimization hint in a resigned JSON frame is not
        # loaded as evidence of prior completed propagation.
        checkpoint['stack'][-1]['_changed_cell']=0
        checkpoint['stack'][-1]['_propagated']=True
        checkpoint['digest']=dfs._checkpoint_digest(checkpoint)
        with patch.object(dfs,'_starting_arcs',tracked):
            final=legacy_solve(problem,seconds=5,resume=checkpoint,
                            progress=lambda event:seen.append(tuple(event['codes'])) if event['event']=='candidate' else None)
        self.assertIsNone(resumed[0])
        self.assertEqual(final['status'],'infeasible')
        self.assertEqual(len(seen),18)
        self.assertEqual(len(set(seen)),18)

    def test_full_and_dirty_queues_keep_identical_partial_symmetry_tree(self):
        from geometry import build_board
        tiles=[{'id':i,'segments':[[[side,2],[side,10]] for side in range(3)]} for i in range(10)]
        outputs=[]
        for full in (True,False):
            problem=SearchProblem(tiles,build_board(1));checks=[0]
            original=problem.partial_loop_impossible
            def loop(codes):checks[0]+=1;return original(codes)
            problem.partial_loop_impossible=loop
            kwargs=dict(seconds=5,seed=29,stop=lambda:checks[0]>=3)
            if full:
                with patch.object(dfs,'_starting_arcs',lambda count,incoming,changed:range(count)):
                    result=legacy_solve(problem,**kwargs)
            else:result=legacy_solve(problem,**kwargs)
            self.assertEqual(result['status'],'stopped')
            self.assertTrue(result['root_symmetry']['reduction']['applied'])
            outputs.append(result['checkpoint']['stack'])
        self.assertEqual(*outputs)


class SymmetryIntegrationTests(unittest.TestCase):
    def fixture(self):
        from search_fixtures import planted_gold
        data, geometry, reference = planted_gold(17, 1)
        return SearchProblem(data, geometry), reference

    def ready_checkpoint(self, problem, cache=None, hints=None):
        requested = [False]
        class PauseBeforePruning:
            def __init__(self, problem, check, *, native='python'):
                requested[0] = True
                check()
        with patch.object(dfs, 'Pruner', PauseBeforePruning):
            result = legacy_solve(problem, seconds=5, hints=hints,
                               stop=lambda:requested[0], learning_state=cache)
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['root_symmetry']['state'], 'ready')
        return result

    def test_new_root_records_verified_restriction_without_mutating_problem(self):
        problem, reference = self.fixture()
        original = copy.deepcopy(problem.domains)
        cache = {}
        expected, metadata = dfs.canonical_root_domains(problem)
        first = self.ready_checkpoint(problem, cache, reference)
        self.assertEqual(first['root_symmetry']['reduction'], metadata)
        self.assertTrue(metadata['applied'])
        self.assertEqual(first['checkpoint']['stack'][0]['domains'], list(map(hex, expected)))
        self.assertEqual(cache['_nogoods'].initial_domains, tuple(expected))
        self.assertEqual(problem.domains, original)
        final = legacy_solve(problem, seconds=5, resume=json.loads(json.dumps(first['checkpoint'])),
                          learning_state=cache)
        self.assertEqual(final['status'], 'solved')
        self.assertTrue(problem.validate(final['codes'])['valid'])
        cell = next(cell for cell, code in enumerate(final['codes']) if code//3 == 0)
        self.assertEqual(problem.board.cells[cell].face, 0)
        self.assertEqual(final['root_symmetry'], first['root_symmetry'])

    def test_pending_stop_and_zero_budget_reuse_same_empty_learning_cache(self):
        problem, _ = self.fixture()
        for options, status in (({'seconds':0}, 'timeout'), ({'seconds':5,'stop':lambda:True}, 'stopped')):
            with self.subTest(status=status):
                cache = {}
                first = legacy_solve(problem, learning_state=cache, **options)
                self.assertEqual(first['status'], status)
                self.assertEqual(first['checkpoint']['root_symmetry']['state'], 'pending')
                self.assertEqual(cache, {})
                self.assertEqual(first['checkpoint']['nodes'], 0)
                final = legacy_solve(problem, seconds=5, resume=json.loads(json.dumps(first['checkpoint'])),
                                  learning_state=cache)
                self.assertEqual(final['status'], 'solved')
                self.assertTrue(final['root_symmetry']['reduction']['applied'])
                self.assertTrue(problem.validate(final['codes'])['valid'])

    def test_interrupted_setup_then_resume_matches_uninterrupted_pending_tree(self):
        from geometry import build_board
        tiles = [{'id':i,'segments':[[[side,2],[side,10]] for side in range(3)]} for i in range(10)]
        problem = SearchProblem(tiles, build_board(1))
        cache, requested = {}, [False]
        original = dfs.canonical_root_domains
        def interrupted(problem, **kwargs):
            def check():
                requested[0] = True
                kwargs['check']()
            return original(problem, check=check)
        with patch.object(dfs, 'canonical_root_domains', interrupted):
            pending = legacy_solve(problem, seconds=5, learning_state=cache, stop=lambda:requested[0])
        self.assertEqual(pending['root_symmetry']['state'], 'pending')
        self.assertEqual(cache, {})
        original_loop = problem.partial_loop_impossible
        outputs = []
        for resume in (None, pending['checkpoint']):
            checks = [0]
            def loop(codes):
                checks[0] += 1
                return original_loop(codes)
            problem.partial_loop_impossible = loop
            result = legacy_solve(problem, seconds=5, seed=0, resume=resume,
                               stop=lambda:checks[0]>=1, learning_state=cache if resume else {}, options={'lookahead':False})
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(result['root_symmetry']['state'], 'ready')
            outputs.append(result['checkpoint']['stack'])
        problem.partial_loop_impossible = original_loop
        self.assertEqual(*outputs)
        self.assertEqual(len(outputs[0]), 2)

    def test_legacy_v2_never_calls_symmetry_or_changes_its_search_space(self):
        from constraint_symmetry import proper_rotations
        problem, reference = self.fixture()
        reference = next(transform.apply(reference) for transform in proper_rotations(problem.board)
                         if problem.board.cells[next(i for i,c in enumerate(transform.apply(reference)) if c//3==0)].face != 0)
        legacy = dfs._solve_core(problem, seconds=0, hints=reference)['checkpoint']
        self.assertNotIn('root_symmetry', legacy)
        with patch.object(dfs, 'canonical_root_domains', side_effect=AssertionError('Legacy space must remain unchanged')):
            stopped = legacy_solve(problem, seconds=0, resume=legacy)
            final = legacy_solve(problem, seconds=5, resume=stopped['checkpoint'])
        self.assertNotIn('root_symmetry', stopped['checkpoint'])
        self.assertNotIn('root_symmetry', final)
        self.assertEqual(final['status'], 'solved')
        self.assertEqual(final['codes'], reference)

    def test_fixed_or_narrowed_initial_domains_are_documented_noops(self):
        problem, reference = self.fixture()
        fixed = SearchProblem(problem.tiles, problem.board, fixed={0:reference[0]})
        problem.domains[0].pop()
        for case in (fixed, problem):
            result = self.ready_checkpoint(case)
            self.assertFalse(result['root_symmetry']['reduction']['applied'])
            expected = [hex(sum(1<<code for code in row)) for row in case.domains]
            self.assertEqual(result['checkpoint']['stack'][0]['domains'], expected)

    def test_resigned_unknown_or_incorrect_metadata_and_root_domains_reject(self):
        problem, _ = self.fixture()
        saved = self.ready_checkpoint(problem)['checkpoint']
        mutations = [
            lambda cp: cp['root_symmetry'].update(state='unknown'),
            lambda cp: cp['root_symmetry'].update(version=2),
            lambda cp: cp['root_symmetry'].update(extra='unknown'),
            lambda cp: cp['root_symmetry']['reduction'].update(face=1),
            lambda cp: cp['root_symmetry']['reduction'].update(rotations=9),
            lambda cp: cp['root_symmetry']['reduction'].update(version=True),
            lambda cp: cp['root_symmetry']['reduction'].update(applied=False),
            lambda cp: cp['root_symmetry']['reduction'].update(extra='unknown'),
            lambda cp: cp['stack'][0]['domains'].__setitem__(1, hex((1 << (3*problem.n))-1)),
        ]
        for change in mutations:
            bad = copy.deepcopy(saved)
            change(bad)
            bad['digest'] = dfs._checkpoint_digest(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                legacy_solve(problem, seconds=5, resume=bad)
        narrowed = copy.deepcopy(problem.domains)
        problem.domains[0].pop()
        with self.assertRaisesRegex(ValueError, 'original cell domains'):
            legacy_solve(problem, seconds=5, resume=saved)
        problem.domains = narrowed

    def test_pending_metadata_cannot_be_attached_to_started_or_reduced_work(self):
        problem, _ = self.fixture()
        pending = legacy_solve(problem, seconds=0)['checkpoint']
        for change in (lambda cp:cp.update(nodes=1),
                       lambda cp:cp['stack'][0]['domains'].__setitem__(0, '0x1'),
                       lambda cp:cp['stats'].update(propagations=1)):
            bad = copy.deepcopy(pending)
            change(bad)
            bad['digest'] = dfs._checkpoint_digest(bad)
            with self.assertRaisesRegex(ValueError, 'untouched'):
                legacy_solve(problem, seconds=5, resume=bad)


class RoundFourDFSTests(unittest.TestCase):
    def probe_fixture(self):
        _,geometry=tetrahedron_tiles()
        segments=[[[0,2],[0,10]],[[1,2],[2,2]],[[1,10],[2,10]]]
        tiles=[{'id':i,'segments':copy.deepcopy(segments)} for i in range(4)]
        tiles[3]['segments']=[[[0,2],[1,2]],[[0,10],[2,2]],[[1,10],[2,10]]]
        problem=SearchProblem(tiles,geometry)
        problem.domains=[[0],[3,4],[6,7,8],[9,10,11]]
        return problem

    def probe_options(self,**extra):
        return {**LEGACY_OPTIONS,'lookahead':True,'failure_guided':True,'native':'python',**extra}

    def test_disabled_advanced_features_skip_helper_setup(self):
        with patch.object(dfs,'AdvancedPruner',side_effect=AssertionError('Disabled features must not initialize')):
            result=dfs.solve(self.probe_fixture(),seconds=5,options=self.probe_options())
        self.assertEqual(result['status'],'solved')
        self.assertTrue(all(result['stats'][key]==0 for key in dfs.ADVANCED_STAT_KEYS))

    def test_probe_conflict_includes_assumption_and_preserves_a_gold_completion(self):
        problem=self.probe_fixture();cache={}
        expected=brute(problem,'gold')
        self.assertTrue(expected)
        result=dfs.solve(problem,seconds=5,hints=[0,3,-1,-1],learning_state=cache,options=self.probe_options())
        self.assertEqual(result['status'],'solved')
        self.assertIn(tuple(result['codes']),expected)
        self.assertGreater(result['stats']['lookahead_prunes'],0)
        self.assertIn(frozenset(((1,3),)),cache['_nogoods'].clauses)
        self.assertNotIn(frozenset(),cache['_nogoods'].clauses)
        self.assertGreater(cache['_runtime'].failure_weights[1],1)

    def test_cancelled_probe_does_not_learn_or_leak_trial_domains(self):
        problem=self.probe_fixture();cache={};checks=[0];requested=[False]
        original=problem.partial_loop_impossible
        def loop(codes):
            checks[0]+=1
            if checks[0]==2:requested[0]=True
            return original(codes)
        problem.partial_loop_impossible=loop
        result=dfs.solve(problem,seconds=5,hints=[0,3,-1,-1],learning_state=cache,
                         options=self.probe_options(),stop=lambda:requested[0])
        self.assertEqual(result['status'],'stopped')
        self.assertGreater(result['stats']['lookahead_probes'],0)
        self.assertEqual(result['stats']['lookahead_prunes'],0)
        self.assertGreater(result['stats']['lookahead_aborted'],0)
        self.assertEqual(cache['_nogoods'].clauses,[])
        self.assertEqual(int(result['checkpoint']['stack'][-1]['domains'][1],16),(1<<3)|(1<<4))
        problem.partial_loop_impossible=original
        final=dfs.solve(problem,seconds=5,resume=result['checkpoint'],learning_state=cache)
        self.assertEqual(final['status'],'solved')

    def test_guidance_and_helper_caches_survive_slices_without_counter_duplication(self):
        problem=self.probe_fixture();cache={}
        def stop():return '_runtime' in cache and cache['_runtime'].failure_weights[1]>1
        first=dfs.solve(problem,seconds=5,hints=[0,3,-1,-1],options=self.probe_options(regional_cache=True),
                        stop=stop,learning_state=cache)
        self.assertEqual(first['status'],'stopped')
        runtime=cache['_runtime'];weights=runtime.failure_weights;advanced=runtime.advanced
        old_checks=dict(advanced.stats);before=dict(first['stats'])
        final=dfs.solve(problem,seconds=5,resume=first['checkpoint'],learning_state=cache)
        self.assertEqual(final['status'],'solved')
        self.assertIs(cache['_runtime'],runtime)
        self.assertIs(runtime.failure_weights,weights)
        self.assertIs(runtime.advanced,advanced)
        for key in dfs.ADVANCED_STAT_KEYS:
            expected=advanced.stats[key] if key=='regional_cache_entries' else before[key]+advanced.stats[key]-old_checks[key]
            self.assertEqual(final['stats'][key],expected,key)
        self.assertGreater(final['stats']['failure_weight_updates'],0)

    def test_options_overrides_preserve_recorded_siblings_and_legacy_checkpoints(self):
        problem=SearchProblem(multiple_loops(),double_face());seen=[]
        first=legacy_solve(problem,seconds=5,stop=lambda:bool(seen),
                           progress=lambda event:seen.append(event) if event['event']=='candidate' else None)
        saved=first['checkpoint'];self.assertTrue(any(frame['cell']>=0 for frame in saved['stack']))
        changed=dfs.solve(problem,seconds=0,resume=saved,options={'native':'python'})
        self.assertEqual(changed['checkpoint']['stack'],saved['stack'])
        self.assertEqual(changed['checkpoint']['search_options']['native'],'python')
        legacy=copy.deepcopy(saved);legacy.pop('search_options')
        for key in dfs.GUIDANCE_STAT_KEYS+dfs.NATIVE_STAT_KEYS+dfs.ADVANCED_STAT_KEYS:legacy['stats'].pop(key)
        legacy['digest']=dfs._checkpoint_digest(legacy)
        self.assertEqual(dfs.solve(problem,seconds=5,resume=legacy)['status'],'infeasible')
        cache={};dfs.solve(self.probe_fixture(),seconds=5,learning_state=cache,options=self.probe_options())
        cache['_runtime']={'failure_weights':[1]*4}
        with self.assertRaisesRegex(ValueError,'runtime cache'):dfs.solve(self.probe_fixture(),seconds=0,learning_state=cache)

    def test_all_new_modes_agree_with_independent_tiny_bruteforce(self):
        fixtures=[self.probe_fixture(),mixed_loop_problem(),SearchProblem(multiple_loops(),double_face())]
        data,geometry=tetrahedron_tiles();fixtures.append(SearchProblem(data,geometry))
        for problem in fixtures:
            for target in ('gold','edges'):
                expected=brute(problem,target)
                for native in ('python','auto'):
                    with self.subTest(n=problem.n,target=target,native=native):
                        result=dfs.solve(problem,seconds=5,target=target,seed=197,options={'native':native})
                        self.assertTrue(result['complete'])
                        if expected:self.assertIn(tuple(result['codes']),expected)
                        else:self.assertEqual(result['status'],'infeasible')


if __name__ == '__main__':
    unittest.main()