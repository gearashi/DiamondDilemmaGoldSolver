"""Small scoped exact searches for decision-set and conflict experiments."""
from collections import Counter, defaultdict
from itertools import combinations
import time

from r27_reason_maxsat import Budget, checked_score, domain_rows
from validator import reverse_mask


class ScopedSearch:
    def __init__(self, problem, rows, target, max_checks, deadline):
        self.problem, self.rows, self.target = problem, rows, target
        self.max_checks, self.deadline = max_checks, deadline
        self.stats = Counter()
        self.last_gold = False
        self.sides = defaultdict(list)
        for a, sa, b, sb in problem.edges:
            self.sides[a,b].append((sa, sb))
            self.sides[b,a].append((sb, sa))

    def tick(self):
        self.stats['work'] += 1
        if self.stats['work'] > self.max_checks or time.monotonic() >= self.deadline:
            raise Budget

    def compatible(self, a, x, b, y):
        self.tick()
        self.stats['pair_checks'] += 1
        return x//3 != y//3 and all(int(self.problem.masks[x, s]) == reverse_mask(int(self.problem.masks[y, t]))
                                     for s, t in self.sides.get((a,b), ()))

    def blockers(self, cell, code, placed):
        for other, choice in placed.items():
            if not self.compatible(cell, code, other, choice):
                return frozenset(((other, choice),))
        return None

    def invalid(self, placed):
        codes = [placed.get(cell, -1) for cell in range(self.problem.n)]
        if len(placed) != len(codes):
            return None
        score, gold = checked_score(self.problem, self.rows, codes)
        self.last_gold = gold
        if score != len(self.problem.edges):
            raise RuntimeError('Search produced a mismatching seam.')
        return frozenset(self.problem.gold_nogood(codes)) if self.target == 'gold' and not gold else None

    def reason(self, reasons, context):
        for explanation in reasons:
            self.stats['replayed_reasons'] += 1
            for assignment in explanation:
                self.tick()
                self.stats['reason_literal_checks'] += 1
                if assignment not in context:
                    break
            else:
                return explanation
        return None

    def dynamic(self, retain):
        placed = {}
        reasons = defaultdict(set)
        n = self.problem.n
        while True:
            self.tick()
            if len(placed) == n:
                conflict = self.invalid(placed)
                if conflict is None:
                    return [placed[i] for i in range(n)]
            else:
                cell = next(c for c in range(n) if c not in placed)
                context = set(placed.items())
                conflict = set()
                for code in self.rows[cell]:
                    self.tick()
                    prior = self.reason(reasons[cell,code],context)
                    if prior is not None:
                        self.stats['reused_reasons'] += 1
                    else:
                        prior = self.blockers(cell, code, placed)
                    if prior is None:
                        placed[cell] = code
                        self.stats['assignments'] += 1
                        break
                    conflict.update(prior)
                else:
                    if not conflict:
                        return None
                    # All values are blocked by this concrete conjunction.
                if cell in placed:
                    continue
            if not conflict:
                return None
            order = list(placed)
            pivot = next(c for c in reversed(order) if (c, placed[c]) in conflict)
            value = placed[pivot]
            reasons[pivot,value].add(frozenset(conflict - {(pivot,value)}))
            later = order[order.index(pivot) + 1:]
            if retain:
                self.stats['retained_assignments'] += len(later)
            else:
                for c in later:
                    del placed[c]
                self.stats['discarded_assignments'] += len(later)
            del placed[pivot]
            self.stats['jumps'] += 1

    def reduced(self, rows):
        """Forward checking plus singleton unit propagation."""
        rows = list(rows)
        if not all(rows):
            return None
        queue = [i for i, row in enumerate(rows) if len(row) == 1]
        for i in queue:
            for j, row in enumerate(rows):
                if j == i:
                    continue
                keep = tuple(y for y in row if self.compatible(i, rows[i][0], j, y))
                if not keep:
                    return None
                if len(keep) == 1 < len(row):
                    queue.append(j)
                rows[j] = keep
        return rows

    def forest(self, rows):
        """After singleton propagation the constraint graph includes inventory overlaps."""
        active = [c for c,row in enumerate(rows) if len(row) > 1]
        graph = {c: set() for c in active}
        tiles = {c: {x//3 for x in rows[c]} for c in active}
        for c, d in combinations(active, 2):
            self.tick()
            if self.sides.get((c,d)) or tiles[c] & tiles[d]:
                graph[c].add(d)
                graph[d].add(c)
        return graph

    def tree_filter(self, graph, rows):
        """Bottom-up tree elimination preserves exactly the extendible root values."""
        parents = {}
        for root in graph:
            if root in parents:
                continue
            parents[root] = None
            visit = [root]
            for c in visit:
                for d in graph[c] - {parents[c]}:
                    if d in parents:
                        return False
                    parents[d] = c
                    visit.append(d)
            for d in reversed(visit[1:]):
                p = parents[d]
                rows[p] = tuple(x for x in rows[p] if any(self.compatible(p,x,d,y) for y in rows[d]))
                if not rows[p]:
                    return None
        return True

    def matching_remainder(self,rows):
        """When every seam is tautological the only hard local factor is inventory."""
        from constraint_advanced import _complete_matching
        for a,s,b,t in self.problem.edges:
            self.tick()
            left = {int(self.problem.masks[x,s]) for x in rows[a]}
            if len(left)!=1 or {reverse_mask(int(self.problem.masks[y,t])) for y in rows[b]} != left:
                return False,None
        tile_masks = [sum(1<<tile for tile in {x//3 for x in row}) for row in rows]
        self.stats['matching_remainders'] += 1
        match = _complete_matching(tile_masks,self.tick)
        if match is None:
            return True,None
        board = [next(code for code in row if code//3==tile) for row,tile in zip(rows,match)]
        placed = dict(enumerate(board))
        if self.invalid(placed) is None:
            return True,board
        # Another physical matching or orientation may form a single Gold loop.
        self.stats['matching_gold_fallbacks'] += 1
        return False,None

    def split_features(self,rows,cell):
        """Cell/tile-renaming invariant features for the held-out split pilot."""
        row = rows[cell]
        tiles = {x//3 for x in row}
        neighbors = [d for d,r in enumerate(rows) if len(r)>1 and (cell,d) in self.sides]
        overlaps = 0
        for d,r in enumerate(rows):
            if d!=cell and len(r)>1:
                self.tick()
                overlaps += sum(code//3 in tiles for code in r)
        return len(row),len(tiles),len(neighbors),min(3,overlaps//max(1,len(row)))

    def find_cutset(self, rows, max_size, trials):
        graph = self.forest(rows)
        tested = 0
        for size in range(min(max_size,len(graph))+1):
            for cut in combinations(sorted(graph,key=lambda c:-len(graph[c])),size):
                if tested >= trials:
                    return None
                self.tick()
                tested += 1
                self.stats['cutset_trials'] += 1
                sub = {c: ns - set(cut) for c,ns in graph.items() if c not in cut}
                # Leaf peeling determines whether the remainder is a forest.
                pending = [c for c,ns in sub.items() if len(ns) < 2]
                while pending:
                    c = pending.pop()
                    if c not in sub:
                        continue
                    for d in sub.pop(c):
                        self.tick()
                        sub[d].discard(c)
                        if len(sub[d]) == 1:
                            pending.append(d)
                if not sub:
                    return cut
        return None

    def split(self, rows, cell):
        first, rest = rows[cell][0], rows[cell][1:]
        a, b = list(rows), list(rows)
        a[cell], b[cell] = (first,), rest
        return a,b

    def explore(self, rows, mode, cut=None, propagated=False):
        self.tick()
        self.stats['nodes'] += 1
        if not propagated:
            rows = self.reduced(rows)
        if rows is None:
            return None
        if mode == 'backdoor':
            completed,witness = self.matching_remainder(rows)
            if completed:
                return witness
            if cut is None:
                cut = self.find_cutset(rows, 2, 48)
            if cut is not None and not any(len(rows[c])>1 for c in cut):
                self.stats['forest_calls'] += 1
                status = self.tree_filter(self.forest(rows), rows)
                if status is None:
                    self.stats['forest_failures'] += 1
                    return None
        active = sorted((c for c,row in enumerate(rows) if len(row)>1), key=lambda c:(len(rows[c]), c))
        if not active:
            placed = {c:row[0] for c,row in enumerate(rows)}
            return list(placed.values()) if self.invalid(placed) is None else None
        choice = next((c for c in (cut or ()) if c in active),active[0])
        if mode == 'learned' and len(active)>1:
            features = [(self.split_features(rows,c),c) for c in active[:3]]
            known = [(self.predictor[f],len(rows[c]),c) for f,c in features if f in self.predictor]
            if known:
                choice = min(known)[2]
                self.stats['learned_choices'] += 1
        if mode == 'orbit' and self.target=='gold':
            from r27_reason_orbits import orbit_choices
            orbit = orbit_choices(self.problem,domains=rows,cell=choice,
                                   work_limit=min(60000,max(0,self.max_checks-self.stats['work'])),
                                   check=self.tick)
            self.stats['orbit_mappings'] += len(orbit['suppressed'])
            if orbit['suppressed']:
                rows[choice] = tuple(orbit['choices'])
                if len(rows[choice])==1:
                    return self.explore(rows,mode,cut)
        probing = mode == 'strong' and len(active)>1
        if probing:
            # A two-sided trial computes exactly the same propagated child
            # domains used below. An empty child is a completed rejection.
            best = None
            for c in active[:3]:
                children = [self.reduced(branch) for branch in self.split(rows,c)]
                self.stats['probe_children'] += 2
                if not any(children):
                    return None
                sizes = [sum(len(s)-1 for s in r) if r else -1 for r in children]
                rank = (max(sizes),min(sizes),c)
                if best is None or rank < best[0]:
                    best = rank,children
            children = best[1]
        else:
            children = self.split(rows,choice)
        for child in children:
            if child is None:
                continue
            answer = self.explore(child,mode,cut,probing)
            if answer is not None:
                return answer
        return None


def scoped_search(problem, *, domains=None, target='gold', mode='mrv', seconds=5, max_checks=100000, predictor=None):
    if mode not in ('mrv','strong','backdoor','dynamic','backjump','learned','orbit') or target not in ('gold','edges'):
        raise ValueError('Unknown search method or target.')
    rows = domain_rows(problem, domains)
    machine = ScopedSearch(problem,rows,target,max_checks,time.monotonic()+seconds)
    machine.predictor = predictor or {}
    try:
        if mode in ('dynamic','backjump'):
            codes = machine.dynamic(mode == 'dynamic')
        else:
            codes = machine.explore(rows,mode)
        status = 'infeasible' if codes is None else ('solved' if machine.last_gold else 'edge_perfect')
    except Budget:
        codes, status = None, 'unknown'
    return {'status':status,'target':target,'codes':codes,'gold_valid':codes is not None and machine.last_gold,
            'stats':dict(machine.stats)}

