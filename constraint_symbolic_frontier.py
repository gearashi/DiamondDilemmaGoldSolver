"""Symbolic unfinished work with disjoint leases and compact durable ownership.

This is trusted local continuation, never an imported mathematical proof. A
worker may retire assignments only after its native checkpoint/result has been
validated by its engine adapter. Unfinished work stays as a decision DAG; only
the one cube given to an engine is materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from constraint_symbolic_work import DomainDiagram, DiagramLimit
from constraint_work import WorkScope, cube, contains, digest, disjoint

MAX_AXES = 160
MAX_BITS = 480
MAX_LEASES = 4096


def _order(order, n):
    order = tuple(range(n)) if order is None else tuple(order)
    if len(order) != n or any(type(i) is not int for i in order) or set(order) != set(range(n)):
        raise ValueError('Invalid symbolic variable order.')
    return order


def _mask(value):
    if not isinstance(value, str) or len(value) > 2 + (MAX_BITS + 3) // 4:
        raise ValueError('Invalid symbolic mask encoding.')
    try:
        result = int(value, 16)
    except ValueError:
        raise ValueError('Invalid symbolic mask encoding.') from None
    if result <= 0 or result.bit_length() > MAX_BITS:
        raise ValueError('Invalid symbolic mask width.')
    return result


class OrderedSets:
    """A common canonical node pool for sets of logical-cell domain boxes."""

    def __init__(self, root, *, order=None, max_nodes=200000):
        self.root = cube(root)
        if len(self.root) > MAX_AXES or any(x.bit_length() > MAX_BITS for x in self.root):
            raise ValueError('Symbolic dimensions exceed the puzzle bounds.')
        self.order = _order(order, len(self.root))
        self.graph = DomainDiagram(tuple(self.root[i] for i in self.order), max_nodes=max_nodes)

    def box(self, box):
        box = cube(box, len(self.root))
        if not contains(self.root, box):
            raise ValueError('Symbolic box escapes its root.')
        return self.graph.from_cube(tuple(box[i] for i in self.order))

    def union_boxes(self, boxes, check=lambda: None):
        result = 0
        for box in boxes:
            check()
            result = self.union(result, self.box(box), check)
        return result

    def union(self, a, b, check=lambda: None):
        return self.graph.apply('union', a, b, check)

    def difference(self, a, b, check=lambda: None):
        return self.graph.apply('difference', a, b, check)

    def intersection(self, a, b, check=lambda: None):
        return self.difference(a, self.difference(a, b, check), check)

    def subset(self, a, b, check=lambda: None):
        return self.difference(a, b, check) == 0

    def first_box(self, node, check=lambda: None):
        if not node:
            return None
        row = list(self.root)
        while node > 1:
            check()
            level, edges = self.graph.nodes[node]
            mask, node = min(edges, key=lambda edge: (edge[0] & -edge[0], edge[0]))
            row[self.order[level]] = mask
        return tuple(row)

    def iter_boxes(self, node, check=lambda: None):
        """Explicit legacy export only; the scheduler never calls this iterator."""
        row = list(self.root)

        def visit(index):
            check()
            if index == 0:
                return
            if index == 1:
                yield tuple(row)
                return
            level, edges = self.graph.nodes[index]
            axis = self.order[level]
            old = row[axis]
            for mask, child in edges:
                row[axis] = mask
                yield from visit(child)
            row[axis] = old

        yield from visit(node)

    def volume(self, node):
        """Raw Cartesian volume, not a count of valid boards or solve progress."""
        return self.graph.volume(node)

    def snapshot(self, roots, check=lambda: None):
        mapping = {0: 0, 1: 1}
        records = []

        def visit(node):
            check()
            if node in mapping:
                return mapping[node]
            level, edges = self.graph.nodes[node]
            encoded = [[hex(mask), visit(child)] for mask, child in edges]
            mapping[node] = len(records) + 2
            records.append([level, encoded])
            return mapping[node]

        translated = [visit(node) for node in roots]
        return {'root': [hex(x) for x in self.root], 'order': list(self.order),
                'nodes': records, 'roots': translated, 'max_nodes': self.graph.max_nodes}

    @classmethod
    def restore(cls, payload, check=lambda: None):
        fields = {'root', 'order', 'nodes', 'roots', 'max_nodes'}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError('Malformed symbolic graph.')
        if not isinstance(payload['root'], list) or not 1 <= len(payload['root']) <= MAX_AXES:
            raise ValueError('Invalid symbolic dimensions.')
        if type(payload['max_nodes']) is not int or not 2 <= payload['max_nodes'] <= 1000000:
            raise ValueError('Invalid symbolic node bound.')
        records = payload['nodes']
        if not isinstance(records, list) or len(records) + 2 > payload['max_nodes']:
            raise ValueError('Symbolic graph exceeds its node bound.')
        result = cls(tuple(_mask(x) for x in payload['root']), order=payload['order'],
                     max_nodes=payload['max_nodes'])
        mapping = [0, 1]
        levels = [len(result.root), len(result.root)]
        for record in records:
            check()
            if not isinstance(record, list) or len(record) != 2:
                raise ValueError('Malformed symbolic node.')
            level, encoded = record
            if type(level) is not int or not 0 <= level < len(result.root):
                raise ValueError('Invalid symbolic level.')
            if not isinstance(encoded, list) or not 1 <= len(encoded) <= MAX_BITS:
                raise ValueError('Invalid symbolic edges.')
            seen = 0
            edges = []
            for edge in encoded:
                if not isinstance(edge, list) or len(edge) != 2:
                    raise ValueError('Malformed symbolic edge.')
                mask, child = _mask(edge[0]), edge[1]
                if (mask & ~result.graph.root[level] or mask & seen or type(child) is not int
                        or not 1 <= child < len(mapping) or levels[child] <= level):
                    raise ValueError('Overlapping, out-of-root or cyclic symbolic edge.')
                seen |= mask
                edges.append((mask, mapping[child]))
            mapping.append(result.graph._make(level, edges))
            levels.append(level)
        roots = payload['roots']
        if not isinstance(roots, list) or len(roots) > MAX_LEASES + 2:
            raise ValueError('Invalid symbolic roots.')
        if any(type(x) is not int or not 0 <= x < len(mapping) for x in roots):
            raise ValueError('Unknown symbolic root.')
        return result, tuple(mapping[x] for x in roots)

    def reordered(self, roots, order, check=lambda: None):
        """Change axis order by exact cofactors, never by enumerating boards.

        A bounded failure leaves the caller's graph authoritative. Root objects
        are translated together, preserving sharing across all ownership sets.
        """
        order = _order(order, len(self.root))
        target = OrderedSets(self.root, order=order, max_nodes=self.graph.max_nodes)
        positions = {axis: level for level, axis in enumerate(self.order)}
        parts = {}
        for axis in order:
            level = positions[axis]
            partition = [self.root[axis]]
            for node_level, edges in tuple(self.graph.nodes.values()):
                check()
                if node_level != level:
                    continue
                for mask, _ in edges:
                    partition = [part for old in partition for part in (old & mask, old & ~mask) if part]
            parts[axis] = tuple(partition)
        cofactors = {}

        def cofactor(node, old_level, mask):
            check()
            if node < 2 or self.graph._level(node) > old_level:
                return node
            key = node, old_level, mask
            if key in cofactors:
                return cofactors[key]
            level, edges = self.graph.nodes[node]
            if level == old_level:
                children = {child for edge, child in edges if edge & mask}
                if len(children) > 1:
                    raise AssertionError('Cofactor partition is not uniform.')
                out = next(iter(children), 0)
            else:
                out = self.graph._make(level, [(edge, cofactor(child, old_level, mask)) for edge, child in edges])
            cofactors[key] = out
            if len(cofactors) > self.graph.max_nodes * 4:
                raise DiagramLimit('Symbolic reorder work cap reached.')
            return out

        memo = {}

        def convert(node, level):
            check()
            if node < 2:
                return node
            key = node, level
            if key in memo:
                return memo[key]
            if level >= len(order):
                raise AssertionError('Nonterminal after all symbolic axes.')
            axis = order[level]
            out = target.graph._make(level, [(mask, convert(cofactor(node, positions[axis], mask), level + 1))
                                             for mask in parts[axis]])
            memo[key] = out
            return out

        translated = tuple(convert(node, 0) for node in roots)
        return target, translated


@dataclass(frozen=True)
class Lease:
    ticket: int
    owner: str
    box: tuple


class SymbolicFrontier:
    VERSION = 1

    def __init__(self, scope, boxes=None, *, order=None, max_nodes=200000):
        if not isinstance(scope, WorkScope):
            raise ValueError('A symbolic frontier requires an exact work scope.')
        self.scope = scope
        self.sets = OrderedSets(scope.root, order=order, max_nodes=max_nodes)
        rows = (scope.root,) if boxes is None else tuple(cube(row, len(scope.root)) for row in boxes)
        if not disjoint(rows):
            raise ValueError('Initial symbolic frontier overlaps.')
        self.pending = self.sets.union_boxes(rows)
        self.completed = 0
        self.leases = {}
        self.next_ticket = 0
        self.stats = dict.fromkeys(('symbolic_leases', 'symbolic_commits', 'symbolic_requeues',
                                    'symbolic_compactions', 'symbolic_reorders'), 0)

    def claim(self, owner, *, choose=None, check=lambda: None):
        if not isinstance(owner, str) or not owner or len(owner) > 128:
            raise ValueError('Invalid symbolic lease owner.')
        if len(self.leases) >= MAX_LEASES:
            raise DiagramLimit('Too many live symbolic leases.')
        box = self.sets.first_box(self.pending, check)
        if box is None:
            return None
        if choose is not None:
            selected = cube(choose(box), len(box))
            if not contains(box, selected):
                raise ValueError('Lease selector escapes the available work.')
            box = selected
        root = self.sets.box(box)
        pending = self.sets.difference(self.pending, root, check)
        lease = Lease(self.next_ticket, owner, box)
        self.pending = pending
        self.leases[lease.ticket] = (lease, root)
        self.next_ticket += 1
        self.stats['symbolic_leases'] += 1
        return lease

    def _lease(self, lease):
        if not isinstance(lease, Lease) or self.leases.get(lease.ticket, (None,))[0] != lease:
            raise ValueError('Unknown, stale or wrong-owner symbolic lease.')
        return self.leases[lease.ticket][1]

    def commit(self, lease, remaining, *, release=True, check=lambda: None):
        """Commit a VALIDATED worker result; supplied rows are unfinished work."""
        owned = self._lease(lease)
        rows = tuple(cube(row, len(self.scope.root)) for row in remaining)
        if not disjoint(rows, check):
            raise ValueError('Worker unfinished work overlaps.')
        rest = self.sets.union_boxes(rows, check)
        if not self.sets.subset(rest, owned, check):
            raise ValueError('Worker result escapes its current lease.')
        closed = self.sets.difference(owned, rest, check)
        completed = self.sets.union(self.completed, closed, check)
        pending = self.sets.union(self.pending, rest, check) if release else self.pending
        # Commit all roots together, after every potentially interruptible step.
        self.pending, self.completed = pending, completed
        if release or not rest:
            del self.leases[lease.ticket]
        else:
            self.leases[lease.ticket] = (lease, rest)
        self.stats['symbolic_commits'] += 1

    def reclaim(self, lease, check=lambda: None):
        root = self._lease(lease)
        pending = self.sets.union(self.pending, root, check)
        self.pending = pending
        del self.leases[lease.ticket]
        self.stats['symbolic_requeues'] += 1

    def exclude_verified(self, boxes, check=lambda: None):
        """Subtract already verified exclusions from UNLEASED work only."""
        excluded = self.sets.union_boxes(boxes, check)
        removed = self.sets.intersection(self.pending, excluded, check)
        pending = self.sets.difference(self.pending, removed, check)
        completed = self.sets.union(self.completed, removed, check)
        self.pending, self.completed = pending, completed

    def remaining_volume(self):
        return self.sets.volume(self.pending) + sum(self.sets.volume(root) for _, root in self.leases.values())

    def snapshot(self, check=lambda: None):
        leases = sorted(self.leases.items())
        roots = [self.pending, self.completed] + [root for _, (_, root) in leases]
        body = {'symbolic_frontier_version': self.VERSION, 'scope': self.scope.payload(),
                'graph': self.sets.snapshot(roots, check), 'next_ticket': self.next_ticket,
                'leases': [{'ticket': lease.ticket, 'owner': lease.owner, 'box': [hex(x) for x in lease.box]}
                           for _, (lease, _) in leases], 'stats': dict(self.stats)}
        body['digest'] = digest(body)
        return body

    @classmethod
    def restore(cls, payload, expected, check=lambda: None):
        fields = {'symbolic_frontier_version', 'scope', 'graph', 'next_ticket', 'leases', 'stats', 'digest'}
        if not isinstance(payload, dict) or set(payload) != fields or type(payload['symbolic_frontier_version']) is not int or payload['symbolic_frontier_version'] != cls.VERSION:
            raise ValueError('Unknown symbolic frontier format.')
        if digest({k: v for k, v in payload.items() if k != 'digest'}) != payload['digest']:
            raise ValueError('Symbolic frontier digest differs.')
        scope = WorkScope.restore(payload['scope'])
        if not expected.accepts(scope):
            raise ValueError('Symbolic frontier belongs to another root/objective/puzzle.')
        sets, roots = OrderedSets.restore(payload['graph'], check)
        if sets.root != scope.root:
            raise ValueError('Symbolic universe and scope differ.')
        rows = payload['leases']
        if not isinstance(rows, list) or len(rows) > MAX_LEASES or len(roots) != len(rows) + 2:
            raise ValueError('Malformed symbolic leases.')
        out = cls(scope, (), order=sets.order, max_nodes=sets.graph.max_nodes)
        out.sets, out.pending, out.completed = sets, roots[0], roots[1]
        owned = out.sets.union(out.pending, out.completed, check)
        if out.sets.intersection(out.pending, out.completed, check):
            raise ValueError('Pending and completed symbolic work overlap.')
        for row, root in zip(rows, roots[2:]):
            check()
            if not isinstance(row, dict) or set(row) != {'ticket', 'owner', 'box'}:
                raise ValueError('Malformed lease record.')
            if (type(row['ticket']) is not int or row['ticket'] < 0 or row['ticket'] in out.leases
                    or not isinstance(row['owner'], str) or not 1 <= len(row['owner']) <= 128
                    or not isinstance(row['box'], list) or len(row['box']) != len(scope.root)):
                raise ValueError('Invalid lease metadata.')
            lease = Lease(row['ticket'], row['owner'], tuple(_mask(x) for x in row['box']))
            if not root or not sets.subset(root, sets.box(lease.box), check) or sets.intersection(owned, root, check):
                raise ValueError('Invalid or overlapping symbolic ownership.')
            owned = sets.union(owned, root, check)
            out.leases[lease.ticket] = (lease, root)
        if type(payload['next_ticket']) is not int or payload['next_ticket'] < 0 or payload['next_ticket'] <= max(out.leases, default=-1):
            raise ValueError('Invalid symbolic ticket counter.')
        out.next_ticket = payload['next_ticket']
        if not isinstance(payload['stats'], dict) or set(payload['stats']) != set(out.stats) or any(type(x) is not int or x < 0 for x in payload['stats'].values()):
            raise ValueError('Invalid symbolic work counters.')
        out.stats = dict(payload['stats'])
        return out

    def compact(self, check=lambda: None):
        restored = self.restore(self.snapshot(check), self.scope, check)
        self.sets = restored.sets
        self.pending, self.completed, self.leases = restored.pending, restored.completed, restored.leases
        self.stats['symbolic_compactions'] += 1

    def reorder(self, order, check=lambda: None):
        leases = sorted(self.leases.items())
        roots = [self.pending, self.completed] + [root for _, (_, root) in leases]
        sets, changed = self.sets.reordered(roots, order, check)
        self.sets, self.pending, self.completed = sets, changed[0], changed[1]
        self.leases = {ticket: (lease, node) for (ticket, (lease, _)), node in zip(leases, changed[2:])}
        self.stats['symbolic_reorders'] += 1
    def optimize(self, *, steps=20000, check=lambda:None):
        """Atomic bounded ordering trials plus reachable-node collection.

        Reordering is a representation change only. No completed or leased
        assignments move between ownership sets, even when a trial is capped.
        Trials inspect DAG nodes; they never enumerate full board assignments.
        """
        if type(steps) is not int or steps<0:raise ValueError('Invalid ordering work cap.')
        leases=sorted(self.leases.items())
        roots=[self.pending,self.completed]+[node for _,(_,node) in leases]
        baseline=self.sets.snapshot(roots,check)
        current=len(baseline['nodes']);used=0;capped=0
        def tick():
            nonlocal used
            check();used+=1
            if used>steps:raise DiagramLimit('Symbolic ordering trial work cap.')
        activity={axis:0 for axis in self.sets.order}
        for level,edges in baseline['nodes']:
            activity[self.sets.order[level]]+=len(edges)
        proposals=(tuple(sorted(self.sets.order,key=lambda axis:(-activity[axis],axis))),
                   tuple(reversed(self.sets.order)))
        best=self.sets;best_roots=roots;best_size=current;accepted=False
        for order in dict.fromkeys(proposals):
            if order==self.sets.order:continue
            try:
                trial,trial_roots=self.sets.reordered(roots,order,tick)
                size=len(trial.snapshot(trial_roots,tick)['nodes'])
            except DiagramLimit:
                capped+=1;break
            if size<best_size:best,best_roots,best_size=trial,trial_roots,size;accepted=True
        # Compact even after an optional trial reaches its own work cap.
        # The external stop check can abort before publication.
        compact,translated=OrderedSets.restore(best.snapshot(best_roots,check),check)
        check()
        self.sets,self.pending,self.completed=compact,translated[0],translated[1]
        self.leases={ticket:(lease,node) for (ticket,(lease,_)),node in zip(leases,translated[2:])}
        self.stats['symbolic_compactions']+=1
        self.stats['symbolic_reorders']+=int(accepted)
        return {'nodes_before':current,'nodes_after':best_size,'trial_steps':used,'capped_trials':capped,'reordered':accepted}
