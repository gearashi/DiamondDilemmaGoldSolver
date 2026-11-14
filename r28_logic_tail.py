"""Binary residual seam search and cut-certified soft rotation bounds."""
from collections import defaultdict
from itertools import combinations
import time

from validator import reverse_mask
from r27_reason_maxsat import Budget, domain_rows


def _two_sat(size, clauses, tick):
    """Iterative Kosaraju. A literal is the node 2*cell + selected index."""
    arcs = [[] for _ in range(2*size)]
    reverse = [[] for _ in arcs]
    for left, right in clauses:
        for a, b in ((left ^ 1, right), (right ^ 1, left)):
            arcs[a].append(b)
            reverse[b].append(a)
    visited = set()
    finish = []
    for root in range(len(arcs)):
        if root in visited:
            continue
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                finish.append(node)
            elif node not in visited:
                tick()
                visited.add(node)
                stack.append((node, True))
                stack.extend((n, False) for n in arcs[node] if n not in visited)
    components = [-1]*len(arcs)
    for index, root in enumerate(reversed(finish)):
        if components[root] >= 0:
            continue
        stack = [root]
        components[root] = index
        while stack:
            node = stack.pop()
            tick()
            for child in reverse[node]:
                if components[child] < 0:
                    components[child] = index
                    stack.append(child)
    if any(components[2*i] == components[2*i+1] for i in range(size)):
        return None
    return [int(components[2*i+1] > components[2*i]) for i in range(size)]


def binary_search(problem, rows, target, *, check=lambda: None, max_work=50000, seconds=2,
                  wide_limit=1):
    """Search two-choice cells jointly, splitting exceptional domains or Gold loop cuts."""
    if target not in ('gold', 'edges'):
        raise ValueError('Expected gold or edges.')
    rows = domain_rows(problem, rows)
    until = time.monotonic() + seconds
    work = 0
    def tick():
        nonlocal work
        work += 1
        if work > max_work or time.monotonic() >= until:
            raise Budget
        if work % 64 == 0:
            check()
    def visit(domains):
        tick()
        if any(not row for row in domains):
            return None
        wide = [i for i, row in enumerate(domains) if len(row)>2]
        if len(wide) > wide_limit:
            raise Budget
        if wide:
            c = min(wide, key=lambda i: len(domains[i]))
            for value in domains[c]:
                branch = list(domains)
                branch[c] = (value,)
                result = visit(branch)
                if result is not None:
                    return result
            return None
        clauses = {(2*i,2*i) for i,row in enumerate(domains) if len(row)==1}
        ownership = defaultdict(list)
        for c,row in enumerate(domains):
            for i,x in enumerate(row):
                ownership[x//3].append((c,i))
        for owners in ownership.values():
            for (c,i),(d,j) in combinations(owners,2):
                if c != d:
                    tick()
                    clauses.add((2*c+(1-i),2*d+(1-j)))
        for c,sc,d,sd in problem.edges:
            for i,x in enumerate(domains[c]):
                for j,y in enumerate(domains[d]):
                    tick()
                    if int(problem.masks[x,sc]) != reverse_mask(problem.masks[y,sd]):
                        clauses.add((2*c+(1-i),2*d+(1-j)))
        bits = _two_sat(problem.n, clauses, tick)
        if bits is None:
            return None
        board = [row[bit] for row,bit in zip(domains,bits)]
        report = problem.validate(board)
        if not report['unique_tiles'] or report['matched_edges'] != len(problem.edges):
            raise RuntimeError('Binary implication graph produced an invalid edge witness.')
        if target=='edges' or report['valid']:
            return board
        # The closed proper loop cannot survive changing all other cells.
        # Partition its disjunction by the first cell differing from this witness.
        branch = list(domains)
        for cell,value in problem.gold_nogood(board):
            if len(branch[cell])==1:
                continue
            other = tuple(x for x in branch[cell] if x!=value)
            branch[cell] = other
            found = visit(branch)
            if found is not None:
                return found
            branch[cell] = (value,)
        return None

    check()
    try:
        board = visit(rows)
        status = 'infeasible' if board is None else 'witness'
    except Budget:
        board,status = None,'unknown'
    check()
    return status,board,work


def binary_tail(problem, masks, target, check, *, max_work=50000, seconds=.08):
    """Dispatch only on sizable binary scopes outside the original eight-cell tail."""
    if not any(m<=0 for m in masks) and sum(m.bit_count()>1 for m in masks)>8 and all(
            m.bit_count()<=2 for m in masks):
        rows = []
        for mask in masks:
            row = []
            while mask:
                bit = mask & -mask
                row.append(bit.bit_length()-1)
                mask ^= bit
            rows.append(row)
        return binary_search(problem, rows, target, check=check, max_work=max_work, seconds=seconds)
    return None


def cut_bound(problem, rows):
    """Bound a binary rotation/placement scope by a rephased integer s-t cut.

    For inconsistent parity cycles relax complementary copies in a doubled
    submodular graph. Dropping all-different is a relaxation; independently check the
    minimum-cut witness before certifying attainability.
    """
    from ortools.graph.python import max_flow
    rows = domain_rows(problem,rows)
    if any(not 1<=len(row)<=2 for row in rows):
        raise ValueError('The cut requires one or two choices per cell.')
    m = problem.n
    graph = [[] for _ in rows]
    terms = []
    for a,sa,b,sb in problem.edges:
        values = [int(problem.masks[rows[a][i % len(rows[a])],sa]) !=
                  reverse_mask(problem.masks[rows[b][j % len(rows[b])],sb])
                  for i in range(2) for j in range(2)]
        z = (values[0]+values[3])-(values[1]+values[2])
        if z:
            graph[a].append((b,int(z>0)))
            graph[b].append((a,int(z>0)))
        terms.append((a,b,values))
    flip = {}
    for cell in range(m):
        if cell in flip:
            continue
        flip[cell] = 0
        stack = [cell]
        while stack:
            node = stack.pop()
            for other,p in graph[node]:
                if other not in flip:
                    flip[other] = flip[node]^p
                    stack.append(other)
    def minimize(size, interactions):
        flow = max_flow.SimpleMaxFlow()
        source,sink = size,size+1
        flow.add_arc_with_capacity(source,sink,0)
        const = 0
        unary = [0]*size
        for a,b,(v00,v01,v10,v11) in interactions:
            delta = v01+v10-v00-v11
            if delta<0:
                raise RuntimeError('Non-submodular table in the cut relaxation.')
            const += v00
            unary[a] += v10-v00
            unary[b] += v11-v10
            if delta:
                flow.add_arc_with_capacity(a,b,delta)
        for c,weight in enumerate(unary):
            const += min(0,weight)
            if weight<0:
                flow.add_arc_with_capacity(c,sink,-weight)
            elif weight:
                flow.add_arc_with_capacity(source,c,weight)
        if flow.solve(source,sink)!=flow.OPTIMAL:
            raise RuntimeError('Max flow failed.')
        source_side = set(flow.get_source_side_min_cut())
        return const+flow.optimal_flow(), [int(c not in source_side) for c in range(size)]

    interactions = []
    frustrated = 0
    for a,b,values in terms:
        v = [int(values[2*(i^flip[a])+(j^flip[b])]) for i in range(2) for j in range(2)]
        frustrated += int(v[0]+v[3]>v[1]+v[2])
        interactions.append((a,b,v))
    if not frustrated:
        minimum,bits = minimize(m,interactions)
        doubled_minimum = 2*minimum
    else:
        # Add a complementary copy of every variable. Consistent complementary
        # assignments have twice their original defect score. Allowing independent
        # copies gives a cut lower bound even when signed cycles are frustrated.
        doubled = []
        for a,b,(v00,v01,v10,v11) in interactions:
            if v00+v11 <= v01+v10:
                doubled.extend(((a,b,(v00,v01,v10,v11)),(a+m,b+m,(v11,v10,v01,v00))))
            else:
                doubled.extend(((a,b+m,(v01,v00,v11,v10)),(a+m,b,(v10,v11,v00,v01))))
        doubled_minimum,bits = minimize(2*m,doubled)
    upper = len(terms)-(doubled_minimum+1)//2
    codes = [row[(bits[c]^flip[c]) % len(row)] for c,row in enumerate(rows)]
    report = problem.validate(codes)
    legal = report['unique_tiles']
    if legal and report['matched_edges']>upper:
        raise RuntimeError('The cut bound contradicted its witness.')
    return dict(upper=upper, doubled_defect_lower=doubled_minimum, frustrated=frustrated, codes=codes if legal else None,
                score=report['matched_edges'] if legal else None, gold=bool(report['valid']))
