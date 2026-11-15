"""Solve bounded residuals of a swap conflict graph at articulation points and odd cycles."""
from functools import lru_cache


def articulation(active, conflicts, vertices):
    entered = {}
    low = {}
    terminals = []

    def visit(v, parent):
        entered[v] = low[v] = len(entered)
        children = 0
        separated = 0
        for u in vertices(conflicts[v] & active):
            if u not in entered:
                children += 1
                visit(u,v)
                low[v] = min(low[v],low[u])
                separated += int(low[u] >= entered[v])
            elif u != parent:
                low[v] = min(low[v],entered[u])
        if (parent is None and children>1) or (parent is not None and separated):
            terminals.append((separated,(conflicts[v]&active).bit_count(),v))

    visit((active & -active).bit_length()-1,None)
    return max(terminals)[-1] if terminals else None


def odd_cycle(active, conflicts, vertices):
    root = (active & -active).bit_length()-1
    colors = {root:0}
    parents = {root:None}
    queue = [root]
    for current in queue:
        for other in vertices(conflicts[current]&active):
            if other not in colors:
                colors[other] = colors[current]^1
                parents[other] = current
                queue.append(other)
            elif colors[other] == colors[current]:
                path = []
                v = current
                while v is not None:
                    path.append(v)
                    v = parents[v]
                tail = []
                v = other
                while v not in path:
                    tail.append(v)
                    v = parents[v]
                return tuple(path[:path.index(v)+1] + list(reversed(tail)))
    return ()


def separator_solver(conflicts, weights, vertices, component, small, bipartite, exact_limit, work):
    def weight(mask):
        return sum(weights[i] for i in vertices(mask))

    @lru_cache(maxsize=8192)
    def solve(active, depth):
        if not active:
            return 0
        if active.bit_count() <= exact_limit:
            return small(active)[1]
        if active.bit_count() > 64:
            return None
        parts = []
        rest = active
        while rest:
            members = component(rest & -rest,rest)
            parts.append(members)
            rest ^= members
        if len(parts)>1:
            result = 0
            for part in parts:
                chosen = solve(part,depth)
                if chosen is None:
                    return None
                result |= chosen
            return result
        found = bipartite(active)
        if found is not None:
            work['bipartite_residuals'] += 1
            return found
        if not depth:
            return None
        pivot = articulation(active,conflicts,vertices)
        if pivot is None:
            cycle = odd_cycle(active,conflicts,vertices)
            if not cycle:
                return None
            pivot = max(cycle,key=lambda v:(conflicts[v]&active).bit_count())
            work['odd_cycle_splits'] += 1
        else:
            work['articulation_splits'] += 1
        bit = 1 << pivot
        exclude = solve(active & ~bit,depth-1)
        if exclude is None:
            return None
        include = solve(active & ~bit & ~conflicts[pivot],depth-1)
        if include is None:
            return None
        include |= bit
        return include if weight(include) > weight(exclude) else exclude

    return solve
