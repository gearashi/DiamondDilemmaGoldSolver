"""Gold GPU proof checks: explicit pairings, exact finite tails, and migration.

Hardware cases are opt-in (DIAMOND_TEST_CUDA / DIAMOND_TEST_WEBGPU).
Fixtures are synthetic and do not assert a solution of the actual puzzle.
"""
import json
import os
from itertools import permutations, product
from pathlib import Path
import unittest
import numpy as np
from dfs_gpu import DFS, fill_order
from geometry import build_board
from systematic_search import AdaptiveBatchBudget
from validator import orientation_masks, validate_arrangement


def gold_fixture(single_loop=True):
    board = build_board()
    tiles = [{'id': str(i), 'segments': []} for i in range(160)]
    if not single_loop:
        for tile in tiles:
            tile['segments'] = [[[s, 5], [s, 7]] for s in range(3)]
        return {'tiles': tiles}, board
    # An Euler tour of the doubled mesh supplies explicit endpoint pairings.
    # Every seam has positions 5 and 7. Pair consecutive tour edges at each
    # cell, giving one 480-segment loop which the independent checker verifies.
    edges, adjacency = [], [[] for _ in range(160)]
    for a, sa, b, sb in board.edges:
        for position in (5, 7):
            index = len(edges)
            edges.append((a, (sa, position), b, (sb, 12-position)))
            adjacency[a].append(index); adjacency[b].append(index)
    used, vertices, traversals, tour = set(), [0], [], []
    while vertices:
        cell = vertices[-1]
        while adjacency[cell] and adjacency[cell][-1] in used:
            adjacency[cell].pop()
        if adjacency[cell]:
            edge = adjacency[cell].pop(); used.add(edge)
            a, ap, b, bp = edges[edge]
            oriented = (a, ap, b, bp) if a == cell else (b, bp, a, ap)
            vertices.append(oriented[2]); traversals.append(oriented)
        else:
            vertices.pop()
            if traversals: tour.append(traversals.pop())
    tour.reverse()
    for edge, following in zip(tour, tour[1:] + tour[:1]):
        assert edge[2] == following[0]
        tiles[edge[2]]['segments'].append([list(edge[3]), list(following[1])])
    return {'tiles': tiles}, board


class GoldHostTests(unittest.TestCase):
    def test_euler_fixture_independently_has_one_loop(self):
        data, board = gold_fixture()
        report = validate_arrangement(data, list(range(0, 480, 3)), board)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['total_segments'], 480)

    def test_gold_tables_bind_internal_connections_not_only_masks(self):
        data, board = gold_fixture()
        other, _ = gold_fixture(False)
        masks = np.asarray(orientation_masks(data), np.uint16)
        fingerprints = []
        for fixture in (data, other):
            gpu = DFS.__new__(DFS)
            gpu._prepare_host(masks, board.neighbor_cells, board.neighbor_sides, 1, None, None, [])
            gpu._prepare_gold(fixture)
            self.assertEqual(gpu.gold_segments, 480)
            for code, row in enumerate(gpu.host_gold_partners):
                for port, partner in enumerate(row):
                    if partner != 255:
                        self.assertEqual(row[partner], port)
                        side, position = divmod(port, 11)
                        self.assertTrue(masks[code, side] & (1 << position))
            fingerprints.append(gpu.gold_fingerprint)
        self.assertNotEqual(*fingerprints)
        data['tiles'][0]['segments'][0][0][1] = 3
        with self.assertRaisesRegex(ValueError, 'masks'):
            gpu._prepare_gold(data)

    def test_v3_requires_target_marker_but_true_v2_remains_readable(self):
        from test_replica_resize import host_dfs
        gpu = host_dfs(1)
        saved = gpu.snapshot_checkpoint()
        self.assertEqual(int(saved['dfs_version']), 3)
        del saved['gold_fingerprint']
        with self.assertRaisesRegex(ValueError, 'target binding'):
            gpu.read_checkpoint(saved)
        saved['dfs_version'] = np.int32(2)
        migrated = gpu.read_checkpoint(saved)
        self.assertEqual(int(migrated['dfs_version']), 3)
        self.assertEqual(str(migrated['gold_fingerprint']), '')

    def test_adaptive_budget_has_bounded_growth_and_fast_slowdown(self):
        control = AdaptiveBatchBudget(32)
        self.assertEqual(control.observe(1), 64)
        self.assertEqual(control.observe(1), 128)
        self.assertEqual(control.observe(1000), 6)
        for _ in range(100): control.observe(.01)
        self.assertEqual(control.nodes, 512)
        self.assertEqual(control.observe(100000), 1)
        for value in (0, -1, float('nan'), float('inf')):
            before = control.nodes
            self.assertEqual(control.observe(value), before)
        fixed = AdaptiveBatchBudget(32, adaptive=False)
        self.assertEqual(fixed.observe(10000), 32)


class GoldHardwareMixin:
    def make(self, data, board, **kwargs):
        gpu = self.factory(np.asarray(orientation_masks(data), np.uint16),
                           board.neighbor_cells, board.neighbor_sides, **kwargs)
        if hasattr(gpu, 'close'): self.addCleanup(gpu.close)
        return gpu

    def test_full_single_loop_survives_and_multiloop_prefix_exhausts(self):
        for single in (True, False):
            with self.subTest(single_loop=single):
                data, board = gold_fixture(single)
                order = fill_order(board.neighbor_cells)
                gpu = self.make(data, board, n=1, order=order,
                                prefixes=[(3*order).tolist()], gold_data=data)
                self.assertEqual(gpu.states.get().tolist(), [1])
                gpu.step(1); gpu.verify()
                self.assertEqual(gpu.states.get().tolist(), [1 if single else 2])
                self.assertEqual(int(gpu.counters.get()[0, 0]), 0)
                if single:
                    self.assertTrue(validate_arrangement(data, gpu.boards[:, 0].get().astype(int).tolist(), board)['valid'])

    def test_early_cycle_pruning_exhausts_before_any_child_is_placed(self):
        data, board = gold_fixture(False)
        gpu = self.make(data, board, n=1, prefixes=[[0]], gold_data=data, seed=71)
        edge = self.make(data, board, n=1, prefixes=[[0]], seed=71)
        for _ in range(40):
            gpu.step(16)
            if gpu.states.get()[0] == 2: break
        self.assertEqual(gpu.states.get().tolist(), [2])
        self.assertEqual(int(gpu.counters.get()[1, 0]), 0)
        edge.step(16)
        self.assertGreater(int(edge.counters.get()[1, 0]), 0)
        gpu.verify(); edge.verify()

    def test_finite_two_cell_tail_matches_independent_gold_enumeration(self):
        data, board = gold_fixture()
        order = fill_order(board.neighbor_cells)
        reference = np.arange(160, dtype=np.int16)*3
        expected = set()
        for pieces in permutations(order[158:]):
            for rotations in product(range(3), repeat=2):
                codes = reference.copy()
                codes[order[158:]] = [3*int(tile)+rot for tile, rot in zip(pieces, rotations)]
                if validate_arrangement(data, codes.astype(int).tolist(), board)['valid']:
                    expected.add(tuple(map(int, codes)))
        self.assertTrue(expected)
        gpu = self.make(data, board, n=1, order=order,
                        prefixes=[reference[order[:158]].tolist()], gold_data=data)
        found = set()
        for _ in range(1000):
            gpu.step(16); gpu.verify()
            state = int(gpu.states.get()[0])
            if state == 1:
                codes = tuple(map(int, gpu.boards[:, 0].get()))
                self.assertNotIn(codes, found); found.add(codes)
                gpu.acknowledge([0])
            elif state == 2: break
        else: self.fail('Finite Gold tail did not exhaust')
        self.assertEqual(found, expected)

    def test_legacy_stack_upgrades_and_gold_checkpoint_cannot_be_downgraded(self):
        data, board = gold_fixture(False)
        edge = self.make(data, board, n=1, prefixes=[[0]], seed=71)
        edge.step(8)
        self.assertGreater(int(edge.depths.get()[0]), 1)
        saved = edge.snapshot_checkpoint()
        saved['dfs_version'] = np.int32(2)
        del saved['gold_fingerprint']
        gold = self.make(data, board, n=1, prefixes=[], gold_data=data)
        gold.restore_checkpoint(saved)
        for _ in range(40):
            gold.step(16)
            if gold.states.get()[0] == 2: break
        gold.verify()
        self.assertEqual(gold.states.get().tolist(), [2])
        with self.assertRaisesRegex(ValueError, 'Gold checkpoint target'):
            edge.restore_checkpoint(gold.snapshot_checkpoint())
        different, _ = gold_fixture(True)
        other = self.make(different, board, n=1, prefixes=[], gold_data=different)
        with self.assertRaisesRegex(ValueError, 'Gold checkpoint target'):
            other.restore_checkpoint(gold.snapshot_checkpoint())

    def test_mid_scan_stop_resume_and_late_prefix_cycle(self):
        data, board = gold_fixture(False)
        # The first assigned cell is isolated. Only the next two cells close a
        # proper loop, so a one-unit launch must defer that second check.
        first = 0
        forbidden = {first, *board.neighbor_cells[first]}
        a, b = next((x, int(y)) for x in range(160) for y in board.neighbor_cells[x]
                    if x not in forbidden and int(y) not in forbidden)
        order = np.array([first, a, b] + [x for x in range(160) if x not in (first, a, b)], np.int16)
        gpu = self.make(data, board, n=1, order=order,
                        prefixes=[[3*first, 3*a, 3*b]], gold_data=data)
        before = gpu.boards.get()
        gpu.step(1)
        self.assertEqual(gpu.states.get().tolist(), [0])
        self.assertEqual(gpu.gold_checked.get().tolist(), [1])
        self.assertEqual(int(gpu.counters.get()[0, 0]), 0)
        np.testing.assert_array_equal(gpu.boards.get(), before)
        restored = self.make(data, board, n=1, order=order, prefixes=[], gold_data=data)
        restored.restore_checkpoint(gpu.snapshot_checkpoint())
        self.assertEqual(restored.gold_checked.get().tolist(), [0])
        restored.step(1)
        self.assertEqual(restored.states.get().tolist(), [0])
        restored.step(1)
        self.assertEqual(restored.states.get().tolist(), [2])
        self.assertEqual(int(restored.counters.get()[0, 0]), 0)
        restored.verify()

    def test_legacy_paused_jobs_upgrade_resize_and_finish_without_losing_ownership(self):
        from dfs_gpu import normalize_prefixes
        from systematic_jobs import JobLedger
        data, board = gold_fixture(False)
        rows, lengths = normalize_prefixes([[0], [3], [6], [9]])
        edge = self.make(data, board, n=4, prefixes=[], seed=71)
        old_ledger = JobLedger.empty(4, 4)
        old_ledger.refill(edge, rows, lengths)
        edge.step(8)
        saved = edge.snapshot_checkpoint(); saved.update(old_ledger.fields())
        gold = self.make(data, board, n=1, prefixes=[], gold_data=data)
        ledger = JobLedger.resume(gold, saved, rows, lengths)
        self.assertEqual(ledger.paused_count, 3)
        for turn in range(150):
            gold.step(32)
            ledger.retire(gold.states.get())
            ledger.refill(gold, rows, lengths)
            if turn == 2:
                checkpoint = gold.snapshot_checkpoint(); checkpoint.update(ledger.fields())
                larger = self.make(data, board, n=3, prefixes=[], gold_data=data)
                ledger = JobLedger.resume(larger, checkpoint, rows, lengths)
                gold = larger
            if ledger.completed == 4: break
        else: self.fail('Gold upgrade failed to retire every legacy active/paused branch')
        self.assertEqual(ledger.paused_count, 0)
        self.assertTrue(np.all(ledger.ids == -1))
        self.assertEqual(ledger.cursor, 4)
        gold.verify()
        ledger.validate(gold.states.get())

    def test_gold_checkpoint_replay_with_different_budgets_keeps_same_frontier(self):
        data, board = gold_fixture()
        a = self.make(data, board, n=1, prefixes=[[0]], gold_data=data, seed=79)
        a.step(13)
        b = self.make(data, board, n=1, prefixes=[], gold_data=data)
        b.restore_checkpoint(a.snapshot_checkpoint())
        # Restored connectivity certificates are deliberately rebuilt within
        # bounded dispatches, with no search transition charged during scanning.
        while np.any(a.gold_checked.get() != 255): a.step(1)
        # Re-save the identical fully prepared frontier; the disposable flag is
        # intentionally absent, so restoration must scan it again.
        b.restore_checkpoint(a.snapshot_checkpoint())
        while np.any(b.gold_checked.get() != 255): b.step(1)
        a.step(40)
        for budget in (1, 3, 8, 12, 16): b.step(budget)
        a.verify(); b.verify()
        for name in DFS.DEVICE_FIELDS:
            np.testing.assert_array_equal(getattr(a, name).get(), getattr(b, name).get(), err_msg=name)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA') == '1', 'CUDA hardware test opt-in')
class GoldCudaTests(GoldHardwareMixin, unittest.TestCase):
    factory = staticmethod(DFS)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU') == '1', 'WebGPU hardware test opt-in')
class GoldWebGPUTests(GoldHardwareMixin, unittest.TestCase):
    @staticmethod
    def factory(*args, **kwargs):
        from dfs_webgpu import WebGPUDFS
        return WebGPUDFS(*args, **kwargs, allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU') == '1')


if __name__ == '__main__': unittest.main()
