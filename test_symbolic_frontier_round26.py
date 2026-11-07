"""Independent finite-set and ownership oracles for the symbolic work manager."""
import copy
import itertools
import random
import unittest

from constraint_symbolic_frontier import OrderedSets, SymbolicFrontier, Lease
from constraint_symbolic_work import DiagramLimit
from constraint_work import WorkScope, digest


def points(box):
    return set(itertools.product(*([i for i in range(mask.bit_length()) if mask >> i & 1] for mask in box)))


def represented(sets, node):
    return set().union(*(points(box) for box in sets.iter_boxes(node)))


def singleton(point):
    return tuple(1 << x for x in point)


class SymbolicSetsTests(unittest.TestCase):
    def test_all_boolean_three_cell_sets_and_orders(self):
        universe = sorted(points((3, 3, 3)))
        for bits in range(256):
            selected = {p for i, p in enumerate(universe) if bits >> i & 1}
            sets = OrderedSets((3, 3, 3))
            root = sets.union_boxes(map(singleton, selected))
            self.assertEqual(represented(sets, root), selected)
            for order in itertools.permutations(range(3)):
                changed, (new,) = sets.reordered((root,), order)
                self.assertEqual(represented(changed, new), selected)
                restored, (saved,) = OrderedSets.restore(changed.snapshot((new,)))
                self.assertEqual(represented(restored, saved), selected)
                self.assertEqual(restored.volume(saved), len(selected))

    def test_random_multivalue_boolean_algebra(self):
        rng = random.Random(26001)
        universe = points((7, 7, 7, 3))
        for _ in range(80):
            sets = OrderedSets((7, 7, 7, 3), order=rng.sample(range(4), 4))
            a = {p for p in universe if rng.randrange(3) == 0}
            b = {p for p in universe if rng.randrange(2) == 0}
            left, right = (sets.union_boxes(map(singleton, rows)) for rows in (a, b))
            self.assertEqual(represented(sets, sets.union(left, right)), a | b)
            self.assertEqual(represented(sets, sets.difference(left, right)), a - b)
            self.assertEqual(represented(sets, sets.intersection(left, right)), a & b)
            self.assertEqual(sets.subset(left, right), a <= b)
            changed, roots = sets.reordered((left, right), rng.sample(range(4), 4))
            self.assertEqual(represented(changed, roots[0]), a)
            self.assertEqual(represented(changed, roots[1]), b)

    def test_shared_snapshot_rejects_cycles_overlaps_and_bounds(self):
        sets = OrderedSets((7, 7))
        root = sets.union_boxes(((3, 1), (4, 6)))
        original = sets.snapshot((root,))
        for mutation in ('cycle', 'overlap', 'outside', 'root', 'bool'):
            payload = copy.deepcopy(original)
            if mutation == 'cycle': payload['nodes'][0][1][0][1] = 2
            if mutation == 'overlap': payload['nodes'][0][1].append(payload['nodes'][0][1][0])
            if mutation == 'outside': payload['nodes'][0][1][0][0] = '0x8000'
            if mutation == 'root': payload['roots'] = [-1]
            if mutation == 'bool': payload['nodes'][0][0] = True
            with self.assertRaises(ValueError, msg=mutation): OrderedSets.restore(payload)
        empty = OrderedSets((1,), max_nodes=2)
        restored, roots = OrderedSets.restore(empty.snapshot((0, 1)))
        self.assertEqual(roots, (0, 1))


class SymbolicOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.scope = WorkScope('finite-fixture', 'edges', (7, 7, 3), (7, 7, 3))

    def all_pending(self, frontier):
        result = represented(frontier.sets, frontier.pending)
        for _, node in frontier.leases.values():
            other = represented(frontier.sets, node)
            self.assertFalse(result & other)
            result |= other
        return result

    def test_random_commits_reorders_restarts_and_worker_loss(self):
        rng = random.Random(26002)
        for trial in range(25):
            frontier = SymbolicFrontier(self.scope)
            outstanding = points(self.scope.root)
            completed = set()
            for step in range(70):
                operation = rng.randrange(5)
                if operation == 0 and frontier.leases:
                    lease, _ = rng.choice(list(frontier.leases.values()))
                    frontier.reclaim(lease)
                elif operation == 1 and frontier.leases:
                    lease, node = rng.choice(list(frontier.leases.values()))
                    owned = represented(frontier.sets, node)
                    retired = {min(owned)}
                    remaining = owned - retired
                    frontier.commit(lease, map(singleton, remaining), release=bool(rng.randrange(2)))
                    outstanding -= retired
                    completed |= retired
                elif operation == 2:
                    frontier.reorder(rng.sample(range(3), 3))
                elif operation == 3:
                    frontier = SymbolicFrontier.restore(frontier.snapshot(), self.scope)
                    frontier.compact()
                else:
                    def select(box):
                        result = list(box)
                        axis = max(range(3), key=lambda i: box[i].bit_count())
                        result[axis] = box[axis] & -box[axis]
                        return result
                    frontier.claim('worker-' + str(step), choose=select)
                self.assertEqual(self.all_pending(frontier), outstanding)
                self.assertEqual(represented(frontier.sets, frontier.completed), completed)
                self.assertEqual(frontier.remaining_volume(), len(outstanding))

    def test_verified_exclusions_leave_active_ownership_untouched(self):
        frontier = SymbolicFrontier(self.scope)
        lease = frontier.claim('a', choose=lambda box: (1, box[1], box[2]))
        frontier.exclude_verified(((7, 1, 3),))
        self.assertEqual(self.all_pending(frontier), points((1, 7, 3)) | points((6, 6, 3)))
        frontier.reclaim(lease)
        with self.assertRaises(ValueError): frontier.reclaim(lease)

    def test_cancel_does_not_publish_partial_commit(self):
        for stop_at in range(1, 40):
            frontier = SymbolicFrontier(self.scope)
            lease = frontier.claim('a')
            before = frontier.snapshot()
            ticks = 0
            def check():
                nonlocal ticks
                ticks += 1
                if ticks == stop_at: raise InterruptedError
            try:
                frontier.commit(lease, ((1, 7, 3), (6, 3, 3)), check=check)
            except InterruptedError:
                self.assertEqual(frontier.snapshot(), before)

    def test_scope_lease_and_subset_validation(self):
        frontier = SymbolicFrontier(self.scope)
        lease = frontier.claim('a', choose=lambda box: (1, 7, 3))
        with self.assertRaises(ValueError): frontier.commit(lease, ((7, 7, 3),))
        with self.assertRaises(ValueError): frontier.commit(lease, ((1, 7, 3), (1, 1, 3)))
        with self.assertRaises(ValueError): frontier.reclaim(Lease(lease.ticket, 'b', lease.box))
        other = WorkScope('finite-fixture', 'gold', self.scope.original, self.scope.root)
        with self.assertRaises(ValueError): SymbolicFrontier.restore(frontier.snapshot(), other)
        payload = frontier.snapshot()
        payload['graph']['roots'][0] = payload['graph']['roots'][2]
        payload['digest'] = digest({k: v for k, v in payload.items() if k != 'digest'})
        with self.assertRaises(ValueError): SymbolicFrontier.restore(payload, self.scope)

    def test_many_exclusions_are_not_materialized_as_jobs(self):
        scope = WorkScope('wide', 'edges', (255,) * 8, (255,) * 8)
        frontier = SymbolicFrontier(scope)
        for axis in range(8):
            box = list(scope.root)
            box[axis] = 1
            frontier.exclude_verified((box,))
        self.assertEqual(frontier.remaining_volume(), 7 ** 8)
        snapshot = frontier.snapshot()
        self.assertLess(len(snapshot['graph']['nodes']), 40)
        self.assertEqual(snapshot['leases'], [])
        self.assertNotIn('boxes', snapshot)
        self.assertEqual(len(list(frontier.sets.iter_boxes(frontier.pending))), 1)


class OrderingLifecycleTests(unittest.TestCase):
    def test_automatic_trials_preserve_ownership_and_compact_on_trial_cap(self):
        from constraint_symbolic_frontier import SymbolicFrontier
        from constraint_work import WorkScope
        from test_constraint_dfs import simple_tiles,double_face
        from search_problem import SearchProblem
        problem=SearchProblem(simple_tiles(),double_face())
        scope=WorkScope.for_problem(problem,'gold')
        frontier=SymbolicFrontier(scope)
        lease=frontier.claim('owner',choose=lambda box:(box[0]&-box[0],box[1]))
        before=frontier.remaining_volume()
        owned=points(lease.box)
        for steps in (0,1,1000):
            report=frontier.optimize(steps=steps)
            self.assertEqual(frontier.remaining_volume(),before)
            row=frontier.leases[lease.ticket][1]
            self.assertEqual(set().union(*(points(box) for box in frontier.sets.iter_boxes(row))),owned)
            self.assertLessEqual(report['nodes_after'],report['nodes_before'])
            frontier=SymbolicFrontier.restore(frontier.snapshot(),scope)
        saved=frontier.snapshot()
        def cancelled():raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):frontier.optimize(check=cancelled)
        self.assertEqual(frontier.snapshot(),saved)

if __name__ == '__main__':
    unittest.main()