"""Physical tile-neighbor multiplicity and connectivity implied by a specific geographic mesh."""
from collections import Counter, defaultdict
from itertools import combinations

from r27_structure_filters import Work, WorkLimit, separator_ok


def mesh_multiplicity(problem):
    if any(v<0 for row in problem.neighbors for v in row):
        return None
    counts=Counter(tuple(sorted((a,b))) for a,_,b,_ in problem.edges)
    if any(a==b for a,b in counts):
        return None
    return max(counts.values(),default=0)


def three_connected(graph,work):
    """Every zero-, one-, or two-vertex deletion leaves a connected graph."""
    if len(graph)<4:
        return False
    for skip in range(len(graph)):
        compressed=[{v-(v>skip) for v in row if v!=skip} for i,row in enumerate(graph) if i!=skip]
        if not separator_ok(compressed,work,width=1):
            return False
    return True


def _local_supported(owners, dart, partner, limit):
    tile,side=divmod(dart,3)
    u=partner//3
    left=owners[3*tile+(side+1)%3]
    right=owners[3*tile+(side+2)%3]
    if not left or not right:
        return False
    if limit==1:
        a=len(left)-(u in left)
        b=len(right)-(u in right)
        if not a or not b:
            return False
        return a>1 or b>1 or left-{u} != right-{u}
    if limit==2:
        return left != {u} or right != {u}
    return True


def filter_neighbors(problem,options,required,*,connect=False,work_limit=1000000):
    limit=mesh_multiplicity(problem)
    live=set(options)
    stats=Counter()
    if limit is None:
        return live,dict(applicable=False)
    fixed=set(required)
    n=3*problem.n
    while True:
        rows=[set() for _ in range(n)]
        for a,b in live:
            rows[a].add(b)
            rows[b].add(a)
        if any(not row for row in rows) or not fixed<=live:
            stats['empty_side']+=1
            return set(),dict(stats)
        fixed.update(tuple(sorted((a,next(iter(row))))) for a,row in enumerate(rows) if len(row)==1)
        by_dart={};by_tile=Counter()
        for a,b in fixed:
            if a in by_dart or b in by_dart:
                stats['fixed_conflict']+=1
                return set(),dict(stats)
            by_dart[a],by_dart[b]=b,a
            by_tile[a//3,b//3]+=1
        if any(k>limit for k in by_tile.values()):
            stats['fixed_conflict']+=1
            return set(),dict(stats)
        owners=[{b//3 for b in row} for row in rows]
        removed=set()
        for a,b in live:
            if ((a in by_dart and by_dart[a]!=b) or (b in by_dart and by_dart[b]!=a) or
                ((a,b) not in fixed and by_tile[a//3,b//3]>=limit)):
                removed.add((a,b))
                stats['fixed_capacity']+=1
            elif not _local_supported(owners,a,b,limit) or not _local_supported(owners,b,a,limit):
                removed.add((a,b))
                stats['neighbor_hall']+=1
        if not removed:
            break
        live-=removed
    if connect:
        work=Work(work_limit)
        try:
            topology=[set(map(int,row)) for row in problem.neighbors]
            if three_connected(topology,work):
                optimistic=[set() for _ in range(problem.n)]
                for a,b in live:
                    optimistic[a//3].add(b//3)
                    optimistic[b//3].add(a//3)
                if not three_connected(optimistic,work):
                    stats['tile_separator']+=1
                    return set(),dict(stats)
        except WorkLimit:
            stats['separator_unknown']+=1
    stats['multiplicity']=limit
    return live,dict(stats)


def post_tile_limits(check):
    maximum=mesh_multiplicity(check.problem)
    if maximum is None:
        return 0
    groups=defaultdict(list)
    for a,b in check.mate:
        groups[a//3,b//3].append((a,b))
    count=0
    for pairs in groups.values():
        if any(len({d for pair in group for d in pair})==2*len(group)
               for group in combinations(pairs,maximum+1)):
            check.m.add(sum(check.mate[p] for p in pairs)<=maximum)
            count+=1
    return count


def geographic_inventory(problem):
    """Gold mate restrictions for actual mesh placement, leaving abstract sewings available."""
    from r32_gold_bonds import BondInventory, inventory
    cached=getattr(problem,'_geographic_mate_inventory',None)
    if cached is not None:
        return cached
    base=inventory(problem)
    if not base.closed_surface:
        return base
    possible,_=filter_neighbors(problem,base.possible,base.required)
    remaining=Counter(d for pair in possible for d in pair)
    required=frozenset(pair for pair in possible if any(remaining[d]==1 for d in pair))
    cached=BondInventory(frozenset(possible),base.forbidden|(base.possible-possible),
                         required,base.loops,base.closed_surface)
    problem._geographic_mate_inventory=cached
    return cached
