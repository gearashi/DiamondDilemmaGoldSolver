"""Exact regional Gold joins, including multiple boundary curves and cyclic partitions."""
from collections import defaultdict
from functools import lru_cache

from r27_structure_filters import Union, Work, WorkLimit, _vertex_links, domain_bits
from r27_structure_region import _states


def _one_tour(left, right):
    links = dict(left)
    links.update((b, a) for a, b in left)
    other = dict(right)
    other.update((b, a) for a, b in right)
    if not links or links.keys() != other.keys():
        return False
    seen = set()
    x = next(iter(links))
    while x not in seen:
        seen.add(x)
        seen.add(links[x])
        x = other[links[x]]
    return len(seen) == len(links)


@lru_cache(maxsize=8)
def _basis(n):
    if n == 2:
        return (((0, 1),),)
    output = []
    for row in _basis(n-2):
        a, b = next(p for p in row if n-3 in p)
        x = a if b == n-3 else b
        output.append(row + ((n-2, n-1),))
        output.append(tuple(p for p in row if p != (a, b)) + ((x, n-2), (n-3, n-1)))
    return tuple(output)


def representatives(rows, outside=None, *, max_ports=18):
    """Keep original matching witnesses for existence; group identical reactions for counts.

    Supplied exterior rows define the complete comparison scope in response mode.
    Multiplicities and all matching indices are returned separately in that mode.
    """
    rows = list(rows)
    if not rows:
        return [], {}
    ports = sorted(v for pair in rows[0] for v in pair)
    if len(ports) < 2 or len(ports) > max_ports or len(ports) != len(set(ports)):
        raise ValueError("Expected distinct paired boundary ports within the configured width.")
    if any(sorted(v for pair in row for v in pair) != ports or any(a == b for a,b in row)
           for row in rows):
        raise ValueError("All rows must match the same boundary.")
    i = {v:k for k,v in enumerate(ports)}
    if outside is None:
        basis = _basis(len(i))
    else:
        if any(sorted(v for pair in row for v in pair) != ports for row in outside):
            raise ValueError("Exterior boundary differs.")
        basis = [tuple((i[a],i[b]) for a,b in row) for row in outside]
    pivots, buckets, keep = {}, defaultdict(list), []
    for j,row in enumerate(rows):
        local = tuple((i[a],i[b]) for a,b in row)
        v = sum(1 << k for k,other in enumerate(basis) if _one_tour(local,other))
        if outside is not None:
            buckets[v].append(j)
            continue
        while v:
            pivot = v.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = v
                keep.append(j)
                break
            v ^= pivots[pivot]
    if outside is not None:
        keep = [items[0] for v,items in buckets.items() if v]
    return keep, dict(buckets)


def _action(problem, parts, ds):
    """C5 preserving the two geographic hemispheres and all per-cell domains."""
    board = problem.board
    index = {(c.face,c.i,c.j,c.inverted):c.index for c in board.cells}
    cell = {}
    for c in board.cells:
        f = c.face//5*5 + (c.face+1)%5
        cell[c.index] = index[f,c.i,c.j,c.inverted]
    if any(ds[c] != ds[cell[c]] for c in range(problem.n)) or any(
            {cell[c] for c in part} != set(part) for part in parts):
        return None
    edges = {}
    for i,(a,s,b,t) in enumerate(problem.edges):
        edges[a,s] = i,False
        edges[b,t] = i,True
    ports = {}
    for e,(a,s,b,t) in enumerate(problem.edges):
        dst,rev = edges[cell[a],s]
        for p in range(11):
            ports[11*e+p] = 11*dst+(10-p if rev else p)
    return cell,ports


def _orbits(table, action):
    _, ports = action
    orbits = {}
    seen = set()
    for key in table:
        if key in seen:
            continue
        members = []
        cur = key
        while cur not in members:
            seen.add(cur)
            members.append(cur)
            used,active,pairs = cur
            cur = (used, tuple(sorted(ports[x] for x in active)),
                   tuple(sorted(tuple(sorted((ports[a],ports[b]))) for a,b in pairs)))
        if cur != key or any(k not in table or table[k][0] != table[key][0] for k in members):
            raise ArithmeticError("Declared rotation failed to preserve the regional table.")
        orbits[key] = members
    return orbits


def _full_tour(rows):
    counts = defaultdict(int)
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for used, active, pairs in rows:
        for a,b in pairs:
            parent.setdefault(a,a)
            parent.setdefault(b,b)
            parent[find(a)] = find(b)
            counts[a] += 1
            counts[b] += 1
    return bool(counts) and all(n == 2 for n in counts.values()) and len({find(x) for x in counts}) == 1


def count_partition(problem, parts, domains=None, *, mode="trie", max_work=1000000):
    """Count labelled Gold boards over any disjoint nonempty proper regions on the sphere.

    The natural join indexes shared seam-mask variables. Physical inventory and
    complete strand pairings remain on the rows. All interrupted results are lower bounds.
    """
    base = domain_bits(problem)
    ds = base if domains is None else tuple(domains)
    if len(ds) != problem.n or any(type(d) is not int or d < 0 or d & ~b for d,b in zip(ds,base)):
        raise ValueError("Invalid domain restriction.")
    parts = tuple(tuple(p) for p in parts)
    if (len(parts) < 2 or any(not p for p in parts) or
            sorted(c for part in parts for c in part) != list(range(problem.n)) or
            _vertex_links(problem) is None):
        raise ValueError("Expected a partition of the closed spherical board.")
    if mode not in ("trie","plain","orbit","response","basis") or mode in ("orbit","response","basis") and len(parts) != 2:
        raise ValueError("Invalid join mode for the number of regions.")
    # Rank aligns both sides of every seam by its actual geographic endpoints.
    rank = {(e,k):11*e+k-1 for e in range(len(problem.edges)) for k in range(1,12)}
    region = {c:r for r,part in enumerate(parts) for c in part}
    shared = [(e,region[a],region[b]) for e,(a,s,b,t) in enumerate(problem.edges) if region[a] != region[b]]
    w = Work(max_work)
    solutions, witness = 0,None
    def score(keys,values,mult=1):
        nonlocal solutions,witness
        w.tick("complete_joins")
        if not _full_tour(keys):
            return
        if witness is None:
            assignment = dict(item for ways,example in values for item in example)
            witness = [assignment[c] for c in range(problem.n)]
            if not problem.validate(witness)["valid"]:
                raise ArithmeticError("Invalid regional witness.")
        for ways,example in values:
            mult *= ways
        solutions += mult
    try:
        tables = [_states(problem,ds,part,rank,False,w) for part in parts]
        w.counters["states"] = sum(map(len,tables))
        action = _action(problem,parts,ds) if mode == "orbit" else None
        if action:
            first,second = (_orbits(t,action) for t in tables)
            w.counters["orbit_representatives"] = len(first)+len(second)
            # Fix a left representative. Its stabilizer acts on the right orbit.
            # In C5 all subgroup sizes are either 1 or 5.
            for a,la in first.items():
                for b,rb in second.items():
                    if a[0] & b[0]:
                        continue
                    for j in range(len(rb) if len(la) == 5 else 1):
                        right = rb[j]
                        w.tick("relative_alignments")
                        if a[1] == right[1]:
                            score((a,right),(tables[0][a],tables[1][right]),len(la) if len(la)==5 else len(rb))
        elif mode in ("response","basis"):
            universe = (1 << problem.n)-1
            groups = [defaultdict(list),defaultdict(list)]
            for g,table in zip(groups,tables):
                for key in table:
                    g[key[:2]].append(key)
            for (used,active),aa in groups[0].items():
                bb = groups[1].get((universe ^ used,active),[])
                if not bb or not active:
                    continue
                rr = [b[2] for b in bb]
                if len(active) > 16:
                    for a in aa:
                        for b in bb:
                            score((a,b),(tables[0][a],tables[1][b]))
                            if mode == "basis" and witness:
                                break
                        if mode == "basis" and witness:
                            break
                elif mode == "basis":
                    keep,_ = representatives([a[2] for a in aa])
                    w.tick("basis_columns",len(aa)*(1<<(len(active)//2-1)))
                    for i in keep:
                        for b in bb:
                            score((aa[i],b),(tables[0][aa[i]],tables[1][b]))
                            if witness:break
                        if witness:break
                else:
                    _,reactions = representatives([a[2] for a in aa],rr)
                    w.tick("response_tests",len(aa)*len(bb))
                    for vector,indices in reactions.items():
                        if not vector:continue
                        a = aa[indices[0]]
                        total = sum(tables[0][aa[i]][0] for i in indices)
                        for j,b in enumerate(bb):
                            if vector >> j & 1:
                                score((a,b),((total,tables[0][a][1]),tables[1][b]))
                if mode == "basis" and witness:break
        else:
            sets = [list(t) for t in tables]
            selected = [None]*len(parts)
            def finish(rem,used):
                if not rem:
                    score(selected,[t[k] for t,k in zip(tables,selected)])
                    return
                r,*tail = rem
                for k in sets[r]:
                    w.tick("row_extensions")
                    if k[0] & used:continue
                    selected[r] = k
                    finish(tail,used|k[0])
            # Per-variable prefix filtering intersects supported next keys before
            # descending; an edge key is the mask of occupied endpoint positions.
            indexes = []
            for e,a,b in shared:
                indexes.append((a,b,{key:sum(1<<(p%11) for p in key[1] if p//11==e) for key in tables[a]},
                                      {key:sum(1<<(p%11) for p in key[1] if p//11==e) for key in tables[b]}))
            indexes.sort(key=lambda line:len(set(line[2].values()) & set(line[3].values())))
            def join(depth):
                if depth == len(indexes):
                    finish(sorted(range(len(parts)),key=lambda r:len(sets[r])),0)
                    return
                a,b,ia,ib = indexes[depth]
                la,lb = sets[a],sets[b]
                left,right = defaultdict(list),defaultdict(list)
                for k in la:left[ia[k]].append(k)
                for k in lb:right[ib[k]].append(k)
                for mask in left.keys() & right.keys():
                    w.tick("trie_prefixes")
                    sets[a],sets[b] = left[mask],right[mask]
                    join(depth+1)
                sets[a],sets[b] = la,lb
            if mode == "plain":
                # _full_tour also rejects unequal masks (degree one).
                finish(range(len(parts)),0)
            else:join(0)
        complete = True
    except WorkLimit:
        complete = False
    out = dict(complete=complete,witness=witness,work=dict(w.counters))
    if mode == "basis":
        out["exists"] = True if witness is not None else False if complete else None
    else:out["solutions"] = solutions
    return out


def paged_trie(problem, parts, domains=None, *, max_steps=10000, table_work=1000000, cursor=None, cache_path=None):
    """Resume a regional join. A cache path also persists partial tables and completed indexes."""
    if cache_path is None:
        return _paged_trie(problem, parts, domains, max_steps=max_steps, table_work=table_work, cursor=cursor)
    from r29_regional_cache import cached_page
    return cached_page(cache_path, _paged_trie, problem, parts, domains,
                       max_steps=max_steps, table_work=table_work, cursor=cursor)


def _paged_trie(problem, parts, domains=None, *, max_steps, table_work, cursor, store=None):
    """Resume the exact multiway join after fixed seam-row tables are compiled.

    The server can save the explicit branch path, next child indices and counts
    atomically. Resuming regenerates only prefix indexes and restricted tables,
    never reopens a completed row combination. One consumer owns a cursor.
    """
    import hashlib
    import json
    ds = domain_bits(problem) if domains is None else tuple(domains)
    base = domain_bits(problem)
    if len(ds) != len(base) or any(type(m) is not int or m<0 or m & ~b for m,b in zip(ds,base)):
        raise ValueError("Invalid domain restriction.")
    parts = tuple(tuple(sorted(p)) for p in parts)
    if (len(parts)<2 or any(not p for p in parts) or
            sorted(x for p in parts for x in p) != list(range(problem.n)) or _vertex_links(problem) is None):
        raise ValueError("Expected a partition of the spherical mesh.")
    if type(max_steps) is not int or max_steps<1:raise ValueError("Invalid join page size.")
    binding = hashlib.sha256(json.dumps(["r28-trie-1",problem.fingerprint,parts,ds],separators=(",",":")).encode()).hexdigest()
    def checksum(data):
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if cursor:
        if type(cursor) is not dict or checksum(cursor.get("data")) != cursor.get("sha256"):
            raise ValueError("Invalid trie cursor checksum.")
        state = cursor["data"]
        if state["binding"] != binding:raise ValueError("Trie source, domains or partition changed.")
        path,next_index = list(state["path"]),list(state["next"])
        solutions,checks,witness = state["solutions"],state["checks"],state["witness"]
        if state.get("done"):
            return dict(complete=True,solutions=solutions,witness=witness,checks=checks,steps=0,cursor=cursor)
    else:
        path,next_index,solutions,checks,witness = [],[0],0,0,None
    rank = {(e,k):11*e+k-1 for e in range(len(problem.edges)) for k in range(1,12)}
    w = Work(table_work)
    bindings = []
    if store is not None:
        from r29_regional_cache import signature
        # With durable tables the budget is measured in DFS transitions. Each transition
        # finishes atomically, including the strand analysis of one regional placement.
        if table_work < 1:raise ValueError("Table page must be positive.")
        tables = []
        w.limit = float("inf")
        for part in parts:
            binding_id, table = store.table(problem,part,ds,rank,table_work,w)
            bindings.append(binding_id)
            if table is None:
                return dict(complete=False,reason="table-budget",cursor=cursor,solutions=solutions,
                            checks=checks,witness=witness,work=dict(w.counters))
            tables.append(table)
    try:
        if store is None:tables = [_states(problem,ds,p,rank,False,w) for p in parts]
    except WorkLimit:
        return dict(complete=False,reason="table-budget",cursor=cursor,solutions=solutions,checks=checks,witness=witness)
    region = {c:i for i,p in enumerate(parts) for c in p}
    keys = [sorted(t) for t in tables]
    shared = []
    for e,(a,s,b,t) in enumerate(problem.edges):
        i,j = region[a],region[b]
        if i==j:continue
        def labels(r):
            return tuple(sum(1<<(k%11) for k in key[1] if k//11 == e) for key in keys[r])
        if store is None:
            left,right = labels(i),labels(j)
        else:
            left = store.labels(bindings[i],e,lambda: labels(i),w)
            right = store.labels(bindings[j],e,lambda: labels(j),w)
        shared.append((len(set(left)&set(right)),e,i,j,left,right))
    shared.sort()
    extent = len(shared)+len(parts)
    initial = tuple(tuple(range(len(row))) for row in keys),0,()
    def children(depth,ctx):
        sets,used,ids = ctx
        if depth < len(shared):
            _,_,i,j,aa,bb = shared[depth]
            la,lb = defaultdict(list),defaultdict(list)
            for x in sets[i]:la[aa[x]].append(x)
            for x in sets[j]:lb[bb[x]].append(x)
            out = []
            for label in sorted(la.keys() & lb.keys()):
                tmp = list(sets)
                tmp[i],tmp[j] = la[label],lb[label]
                out.append((tuple(tmp),used,ids))
            return out
        i = depth-len(shared)
        return [(sets,used|keys[i][x][0],ids+(x,)) for x in sets[i] if not used & keys[i][x][0]]
    stack = [initial]
    if len(next_index) != len(path)+1 or len(path)>extent:raise ValueError("Invalid trie cursor stack.")
    for i,j in enumerate(path):stack.append(children(i,stack[-1])[j])
    child_cache = {}
    # An exact empty suffix depends only on selected row patterns and the remaining
    # admissible row patterns. Row multiplicities and example IDs cannot alter emptiness.
    def gap_key(d, ctx):
        if store is None or d < len(shared) or d == extent:return None
        allowed, _, chosen = ctx
        picked = [keys[i][j] for i,j in enumerate(chosen)]
        future = [[keys[i][j] for j in allowed[i]] for i in range(len(chosen),len(parts))]
        return signature(["empty-gold-join-v1",problem.fingerprint,parts,picked,future])
    baseline = list(state.get("entry_solutions",[None]*len(stack))) if cursor else [0]
    if len(baseline)!=len(stack):raise ValueError("Invalid trie cursor counts.")
    gap_keys = {}
    steps = 0
    while stack and steps < max_steps:
        steps += 1
        d = len(stack)-1
        if d == extent:
            checks += 1
            ids = stack[-1][2]
            rows = [keys[i][j] for i,j in enumerate(ids)]
            if _full_tour(rows):
                ways = 1
                assign = {}
                for table,row in zip(tables,rows):
                    mult,codes = table[row]
                    ways *= mult
                    assign.update(codes)
                solutions += ways
                if witness is None:
                    witness = [assign[i] for i in range(problem.n)]
                    if not problem.validate(witness)["valid"]:raise ArithmeticError("Invalid paged witness.")
        else:
            if d not in child_cache:
                key = gap_key(d,stack[-1])
                gap_keys[d] = key
                if key is not None and store.empty(key):
                    w.counters["empty_hits"] += 1
                    child_cache[d] = []
                else:child_cache[d] = children(d,stack[-1])
            choices = child_cache[d]
            j = next_index[-1]
            if j < len(choices):
                next_index[-1] += 1
                stack.append(choices[j])
                path.append(j)
                next_index.append(0)
                baseline.append(solutions)
                continue
            key = gap_keys.pop(d,None)
            if key is not None and baseline[-1] == solutions:
                store.record_empty(key)
            child_cache.pop(d,None)
        baseline.pop()
        stack.pop()
        next_index.pop()
        if path:path.pop()
    done = not stack
    state = dict(binding=binding,path=path,next=next_index,solutions=solutions,checks=checks,witness=witness,done=done,entry_solutions=baseline)
    return dict(complete=done,solutions=solutions,witness=witness,checks=checks,steps=steps,
                work=dict(w.counters),cursor=dict(data=state,sha256=checksum(state)))
