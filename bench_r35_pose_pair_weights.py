"""Rejected after full geographic/ordinary owner closure. Numerical benchmark only."""
from functools import lru_cache
from itertools import combinations


def away_matching_size(rows, stop):
    """Find up to stop disjoint off-image cell/tile choices. A full run proves the maximum."""
    order = sorted(range(len(rows)), key=lambda c: rows[c].bit_count())
    owners = {}
    matched = set()
    used = 0
    for c in order:
        options = rows[c] & ~used
        if options:
            bit = options & -options
            owners[bit] = c
            used |= bit
            matched.add(c)
            if len(owners) >= stop:
                return len(owners)
    def augment(c, seen):
        options = rows[c]
        while options:
            bit = options & -options
            options ^= bit
            if bit in seen:
                continue
            seen.add(bit)
            if bit not in owners or augment(owners[bit],seen):
                owners[bit] = c
                return True
        return False
    for c in order:
        if c not in matched and augment(c,set()) and len(owners) >= stop:
            break
    return len(owners)


def exact_pair_price(domains, union, cap):
    """Near saturation, enforce the quota on complete bijections, pricing each orientation separately."""
    from constraint_connectivity_round26 import assignment_upper_bound
    from bench_r35_pose_matching import exact_margins
    n=len(domains)
    weights=[]
    for mask,paid in zip(domains,union):
        free=mask & ~paid
        weights.append([1 if free & (7<<(3*t)) else 0 if mask & (7<<(3*t)) else None for t in range(n)])
    proof=assignment_upper_bound(weights)
    if any(weights[c][t] is None for c,t in enumerate(proof['assignment'])):
        return domains,True,n+1
    lower=n-proof['bound']
    if lower>cap:
        return domains,True,lower
    slack=cap-lower
    margins,_=exact_margins(range(n),range(n),weights,proof,slack)
    output=list(domains)
    for (c,t),extra in margins.items():
        tile_mask=7<<(3*t)
        if extra>slack:
            output[c]&=~tile_mask
        elif extra==slack and weights[c][t]:
            # This (cell,tile) can saturate the remaining budget only in a free orientation.
            output[c]&=~(tile_mask & union[c])
    return output,not all(output),lower


@lru_cache(maxsize=64)
def _owner_pair(domains, union, cap, starts):
    need=len(domains)-cap
    away=[mask & ~paid for mask,paid in zip(domains,union)]
    rows=[(m | m>>1 | m>>2) & starts for m in away]
    # With at least two spare matches, conditioning one cell and tile cannot
    # violate the off-image matching relaxation. Retain the cheaper baseline there.
    size=away_matching_size(rows,need+2)
    if size<need:
        return (),True,len(rows)-size,'away_matching'
    if size>=need+2:
        return (),False,0,'spare'
    filtered,over,lower=exact_pair_price(domains,union,cap)
    return tuple((c,mask^out) for c,(mask,out) in enumerate(zip(domains,filtered)) if mask!=out),over,lower,'assignment'


def pair_owner_bounds(domains, images, cap):
    n=len(domains);limit=2*cap
    stats=dict(over=False,removed=0,matching_queries=0,assignments=0,cache_hits=0,lower=0)
    starts=((1<<(3*n))-1)//7
    active=sum(bool((d | d>>1 | d>>2) & ~(d & (d>>1) | d & (d>>2) | (d>>1) & (d>>2)) & starts) for d in domains)
    if not (0<limit<=n) or active<limit:
        return domains,stats
    # A row can charge this quota only if one named orientation is its sole
    # available turn of that tile. The two original images place distinct tiles at every site.
    tables=[]
    candidates=[]
    for image in images:
        by_cell=dict(image)
        table=tuple(by_cell[c] for c in range(n))
        tables.append(table)
        candidates.append(sum(1<<c for c,(mask,pose) in enumerate(zip(domains,table))
            if mask & (7 << (3*((pose.bit_length()-1)//3))) == pose))
    current=list(domains)
    starts=((1<<(3*n))-1)//7
    for a,b in combinations(range(len(images)),2):
        if (candidates[a]|candidates[b]).bit_count()<limit:
            continue
        union=tuple(x|y for x,y in zip(tables[a],tables[b]))
        hits=_owner_pair.cache_info().hits
        removals,over,lower,kind=_owner_pair(tuple(current),union,limit,starts)
        if _owner_pair.cache_info().hits>hits:
            stats['cache_hits']+=1
        else:
            stats['matching_queries']+=1
            stats['assignments']+=int(kind=='assignment')
        stats['lower']=max(stats['lower'],lower)
        if over:
            stats.update(over=True,reason='pose_image_pair_owners',images=[a,b],method=kind)
            return current,stats
        for c,remove in removals:
            stats['removed']+=(current[c]&remove).bit_count()
            current[c]&=~remove
    return current,stats
