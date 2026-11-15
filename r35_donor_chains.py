"""Regional star upper bounds retaining the endpoints and lengths of physical donor chains."""
from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations


@lru_cache(8192)
def stitch(arcs):
    """Collapse a partial injective permutation into open chains and closed cycles."""
    following = {a:(b,length) for a,b,length in arcs}
    inbound = {b for _,b,_ in arcs}
    paths = []
    cycles = []
    for start in sorted(set(following)-inbound):
        c, length = start, 0
        while c in following:
            c, edges = following.pop(c)
            length += edges
        paths.append((start,c,length))
    while following:
        c = next(iter(following))
        length = 0
        while c in following:
            c, edges = following.pop(c)
            length += edges
        if length>1:
            cycles.append(length)
    return tuple(paths),tuple(sorted(cycles))


def groups_for(variables,stars):
    near={c:set() for c in variables}
    for _,row in stars:
        touched=variables & row
        for c in touched:
            near[c].update(touched)
    result=[]
    left=set(variables)
    while left:
        group={left.pop()}
        fringe=set(group)
        while fringe:
            nxt=set.union(*(near[c] for c in fringe)) & left
            left-=nxt
            fringe=nxt
            group.update(nxt)
        result.append(tuple(sorted(group)))
    return sorted(result,key=lambda row:(len(row),row))


def family_budget(component,family,max_rows=100000,separator=2,target=None):
    family=tuple(sorted(family))
    pool=component.pool
    if not family or any(type(k) is not int or k<2 for k in family) or sum(family)>len(pool):
        raise ValueError('Expected a nontrivial cycle family fitting the pool.')
    if max_rows<1 or not 0<=separator<=2:
        raise ValueError('Invalid chain join or geographic separator budget.')
    b=component.bound
    work=Counter()
    broad=component.upper(sum(family)-len(family),min_moved=sum(family),max_rows=max_rows)
    fallback=dict(upper=broad['upper'],complete=False,work=work,coarse=broad)
    if target is not None and broad['upper']<target:
        return fallback
    if broad['upper']<0:
        return dict(**(fallback | {'complete':True}))
    allowed=set(pool)
    stars=[]
    for c in range(b.n):
        variables=allowed & {c,*map(int,b.neighbors[c])}
        if variables:
            stars.append((c,variables))
    pins=()
    groups=groups_for(allowed,stars)
    if len(groups)<2:
        candidates=[]
        for width in range(1,separator+1):
            for cut in combinations(pool,width):
                parts=groups_for(allowed-set(cut),stars)
                if len(parts)>1:
                    candidates.append((max(map(len,parts)),cut,parts))
            if candidates:
                break
        if not candidates:
            return fallback
        _,pins,groups=min(candidates)
    assigned={c:idx for idx,group in enumerate(groups) for c in group}
    terms=[[] for _ in groups]
    pin_terms=[]
    for c,row in stars:
        group_ids={assigned[v] for v in row if v in assigned}
        if group_ids:
            if len(group_ids)!=1:
                raise AssertionError('Conditioning failed to separate the star.')
            terms[group_ids.pop()].append(c)
        else:
            pin_terms.append(c)
    def price(cells,board):
        if not cells:
            return 0
        return int(b.stars(board[cells],board[b.neighbors[cells]]).sum()-b.base[cells].sum())
    def legal(cell,donor):
        tile=b.tiles[donor]
        return b.model.legal[cell,3*tile:3*tile+3].any()
    required=Counter(family)
    def fits(paths,cycles):
        if len(cycles)>len(family) or any(cycles.count(k)>required[k] for k in set(cycles)):
            return False
        if not paths:
            return True
        max_size=max((k for k in required if cycles.count(k)<required[k]),default=0)
        return all(length<max_size for _,_,length in paths)
    best=None
    for assigned_donors in permutations(pool,len(pins)):
        if any(not legal(c,d) for c,d in zip(pins,assigned_donors)):
            continue
        work['conditionings']+=1
        board=b.pieces.copy()
        for c,d in zip(pins,assigned_donors):
            board[c]=b.tiles[d]
        paths,closed=stitch(tuple(sorted((c,d,1) for c,d in zip(pins,assigned_donors))))
        if not fits(paths,closed):
            continue
        p_mask=sum(1<<component.index[d] for d in assigned_donors)
        state={(p_mask,paths,closed):price(pin_terms,board)}
        donors=tuple(d for d in pool if d not in assigned_donors)
        for group,touched in zip(groups,terms):
            options={}
            for incoming in permutations(donors,len(group)):
                if work['rows']>=max_rows:
                    return fallback
                work['rows']+=1
                if any(not legal(c,d) for c,d in zip(group,incoming)):
                    continue
                edges=tuple((c,d,1) for c,d in zip(group,incoming))
                paths,closed=stitch(edges)
                if not fits(paths,closed):
                    continue
                board[list(group)]=b.tiles[list(incoming)]
                gain=price(touched,board)
                mask=sum(1<<component.index[d] for d in incoming)
                key=(mask,paths,closed)
                options[key]=max(options.get(key,gain),gain)
            # Rows of previous components never appear in these star terms.
            # Reset them anyway to keep the base explicit on the next component.
            board[list(group)]=b.pieces[list(group)]
            by_mask={}
            for (mask,paths,closed),gain in options.items():
                by_mask.setdefault(mask,[]).append((paths,closed,gain))
            following={}
            for (taken,old_paths,old_closed),score in state.items():
                available=[i for i in range(len(pool)) if not taken>>i&1]
                for ids in combinations(available,len(group)):
                    mask=sum(1<<i for i in ids)
                    for paths,closed,gain in by_mask.get(mask,()):
                        if work['joins']>=max_rows:
                            return fallback
                        work['joins']+=1
                        rest,new=stitch(tuple(sorted((*old_paths,*paths))))
                        finished=tuple(sorted((*old_closed,*closed,*new)))
                        if not fits(rest,finished):
                            continue
                        key=(mask|taken,rest,finished)
                        following[key]=max(following.get(key,score+gain),score+gain)
            state=following
            if not state:
                break
        for (_,paths,closed),gain in state.items():
            if not paths and closed==family:
                best=gain if best is None else max(best,gain)
    upper=-1 if best is None else min(broad['upper'],(b.total+best)//2)
    return dict(upper=upper,complete=True,coarse=broad,work=dict(work),separator=list(pins),parts=groups)
