"""Small exact Hall witnesses; failure to find one only loses precision."""
from collections import deque

def removal_witness(rows,matching,cell,tile,check=lambda:None,limit=8192):
    """A tight Hall set excluding cell but requiring tile forbids cell=tile.

    Starting with tile's matched owner, follow domain-to-matched-owner arcs.
    Closure that does not reach cell has equal numbers of cells and available
    tiles. Validate that cardinality directly before returning the witness.
    """
    n=len(rows)
    if matching is None or len(matching)!=n or sorted(matching)!=list(range(n)) or any(not row&(1<<t) for row,t in zip(rows,matching)):return None
    if matching[cell]==tile:return None
    owner=[-1]*n
    for c,t in enumerate(matching):owner[t]=c
    seen={owner[tile]};pending=deque(seen);steps=0;union=0
    while pending:
        check();c=pending.popleft();bits=rows[c];union|=bits
        while bits:
            steps+=1
            if steps>limit:return None
            bit=bits&-bits;bits^=bit;t=bit.bit_length()-1
            if t==tile:continue
            other=owner[t]
            if other==cell:return None
            if other not in seen:seen.add(other);pending.append(other)
    if cell in seen or union.bit_count()!=len(seen) or not union&(1<<tile):return None
    return tuple(sorted(seen))

def failure_witness(rows,check=lambda:None,limit=8192):
    """Find a deficient Hall set with bounded independent augmenting paths."""
    n=len(rows);matched=[-1]*n;owners=[-1]*n;steps=0
    for start in sorted(range(n),key=lambda c:rows[c].bit_count()):
        check();pending=deque([start]);seen={start};parents={};free=-1
        while pending and free<0:
            c=pending.popleft();bits=rows[c]
            while bits:
                steps+=1
                if steps>limit:return None
                if steps%128==0:check()
                bit=bits&-bits;bits^=bit;t=bit.bit_length()-1
                if t in parents:continue
                parents[t]=c;owner=owners[t]
                if owner<0:free=t;break
                if owner not in seen:seen.add(owner);pending.append(owner)
        if free<0:
            union=0
            for c in seen:union|=rows[c]
            return tuple(sorted(seen)) if union.bit_count()<len(seen) else None
        while free>=0:
            c=parents[free];old=matched[c];matched[c]=free;owners[free]=c;free=old
    return None
