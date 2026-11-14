"""Resume colored exact-cover traversal at a deterministic choice trail."""
from hashlib import sha256
import json

from r27_structure_filters import Work, WorkLimit, bits, domain_bits
from validator import reverse_mask


def cover_page(problem,domains=None,*,cursor=None,node_limit=5000,max_work=500000,check=None,retain=5):
    """Each returned page counts the new labelled Gold answers on that page.

    The cursor stores the next unprocessed path in the deterministic row order.
    Replaying its ancestors reconstructs purified columns without revisiting their
    completed siblings. Caller saves the page and its cursor as one transaction.
    """
    base=domain_bits(problem)
    ds=base if domains is None else tuple(domains)
    if len(ds)!=len(base) or any(type(d)is not int or d<0 or d&~b for d,b in zip(ds,base)):
        raise ValueError('Invalid placement domains.')
    if type(node_limit)is not int or node_limit<1 or type(retain)is not int or retain<0:raise ValueError('Invalid page limits.')
    bound=sha256(json.dumps([1,problem.fingerprint,ds],separators=(',',':')).encode()).hexdigest()
    trail=() if cursor is None else tuple(cursor['next'])
    if cursor is not None and (cursor.get('binding')!=bound or any(type(x)is not int or x<0 for x in trail)):
        raise ValueError('Invalid cover cursor for this board.')
    near=[[] for _ in ds]
    for a,s,b,t in problem.edges:
        near[a].append((s,b,t));near[b].append((t,a,s))
    order=[]
    for c,m in enumerate(ds):order.extend((c,v) for v in bits(m))
    by_cell=[set() for _ in ds]
    by_tile=[set() for _ in ds]
    for i,(c,v) in enumerate(order):
        by_cell[c].add(i);by_tile[v//3].add(i)
    w=Work(max_work,check)
    count=nodes=0
    witness=[]
    next_path=None
    last_cursor=trail
    def walk(path,live,pending,assignment,seek):
        nonlocal count,nodes,next_path,last_cursor
        if seek and len(path)==len(trail):seek=False
        if not seek:
            last_cursor=path
            if nodes>=node_limit:
                next_path=path
                return True
            w.tick('cover_nodes');nodes+=1
        else:w.tick('replayed_ancestors')
        if not pending:
            if seek:raise ValueError('Cursor extends a terminal.')
            if problem.validate(assignment)['valid']:
                count+=1
                if len(witness)<retain:witness.append(tuple(assignment))
            return False
        def choices(col):
            return (by_cell[col] if col<problem.n else by_tile[col-problem.n])&live
        col=min(pending,key=lambda c:(len(choices(c)),c))
        rows=sorted(choices(col))
        if seek:
            step=trail[len(path)]
            if step not in rows:raise ValueError('Cursor does not identify a live choice.')
            rows=rows[rows.index(step):]
        for i in rows:
            c,v=order[i]
            assignment[c]=v
            if not problem.partial_loop_impossible(assignment):
                allowed=live-by_cell[c]-by_tile[v//3]
                for s,b,t in near[c]:
                    want=reverse_mask(int(problem.masks[v,s]))
                    allowed.difference_update(j for j in by_cell[b] if int(problem.masks[order[j][1],t])!=want)
                if walk(path+(i,),allowed,pending-{c,problem.n+v//3},assignment,seek):return True
            elif seek:raise ValueError('Cursor enters a closed proper loop.')
            assignment[c]=-1
            seek=False
        return False
    try:
        walk((),set(range(len(order))),set(range(2*problem.n)),[-1]*problem.n,bool(trail))
    except WorkLimit:
        # The only budget charges are before node publication, including replay.
        next_path=last_cursor
    return {'complete':next_path is None,'solutions':count,'witnesses':witness,'work':dict(w.counters),
            'cursor':None if next_path is None else {'binding':bound,'next':list(next_path)}}

