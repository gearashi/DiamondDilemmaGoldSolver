"""Pack profitable swaps exactly on small or bipartite conflict components."""
from functools import lru_cache


def vertices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


def pack_swaps(model, endpoints, weights, *, exact_limit=22):
    # The index order is the established greedy priority (gain, then cell order).
    n = len(endpoints)
    if not n:
        return []
    incident = [0]*model.n
    for i,cells in enumerate(endpoints):
        for c in cells:
            incident[c] |= 1<<i
    blocked = incident.copy()
    for c in range(model.n):
        for nbr in model.problem.neighbors[c]:
            if nbr>=0:
                blocked[c] |= incident[nbr]
    conflicts = []
    for i,cells in enumerate(endpoints):
        neighbors = 0
        for c in cells:
            neighbors |= blocked[c]
        conflicts.append(neighbors ^ (1<<i))

    def component(root, active):
        seen = 0
        frontier = root
        while frontier:
            seen |= frontier
            follow = 0
            for v in vertices(frontier):
                follow |= conflicts[v]
            frontier = follow & active & ~seen
        return seen

    @lru_cache(maxsize=8192)
    def small(active):
        if not active:
            return 0,0
        pivot = max(vertices(active),key=lambda v:(conflicts[v]&active).bit_count())
        bit = 1<<pivot
        weight,mask = small(active & ~bit & ~conflicts[pivot])
        include = weight+weights[pivot],mask|bit
        exclude = small(active & ~bit)
        return include if include[0]>=exclude[0] else exclude

    def bipartite(active):
        left = active & -active
        frontier = left
        reached = left
        while frontier:
            following = 0
            for v in vertices(frontier):
                following |= conflicts[v]
            following &= active
            color = left if frontier & left else reached^left
            if following & color:
                return None
            following &= ~reached
            if not frontier & left:
                left |= following
            reached |= following
            frontier = following
        try:
            from ortools.graph.python.max_flow import SimpleMaxFlow
        except ImportError:
            return None
        # Complement of a minimum weighted vertex cover is a maximum-weight
        # independent set. Infinite-capacity arcs enforce every conflict.
        flow = SimpleMaxFlow()
        source,sink = n,n+1
        inf = 1+sum(weights[v] for v in vertices(active))
        for v in vertices(left):
            flow.add_arc_with_capacity(source,v,weights[v])
            for w in vertices(conflicts[v]&active):
                flow.add_arc_with_capacity(v,w,inf)
        for v in vertices(active^left):
            flow.add_arc_with_capacity(v,sink,weights[v])
        if flow.solve(source,sink)!=flow.OPTIMAL:
            return None
        reachable = sum(1<<int(v) for v in flow.get_source_side_min_cut() if v<n)
        return (left&reachable)|((active^left)&~reachable)

    chosen = 0
    remaining = (1<<n)-1
    while remaining:
        first = remaining & -remaining
        current = component(first,remaining)
        if current.bit_count()<=exact_limit:
            chosen |= small(current)[1]
            remaining &= ~current
            continue
        solved = bipartite(current)
        if solved is not None:
            chosen |= solved
            remaining &= ~current
            continue
        # Preserve the previous greedy move, then test smaller residual components.
        chosen |= first
        remaining &= ~first & ~conflicts[first.bit_length()-1]
    return list(vertices(chosen))
