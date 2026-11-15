"""Maximum-weight square assignment after restricting a previously solved matrix."""
from collections import Counter

import numpy as np


def matching(weights, allowed, previous=None):
    """Return optimal donors and a feasible integer dual, or None for infeasible.

    Restricting an edge or lowering its gain preserves a parent's feasible
    dual. A surviving tight arc also retains its old matched position.
    """
    n=len(weights)
    work=Counter()
    if previous is None:
        if np.any(~allowed.any(axis=1)):
            return None,work
        rows=np.where(allowed,weights,np.iinfo(np.int64).min).max(axis=1).astype(int)
        cols=np.zeros(n,dtype=int)
        donor=np.full(n,-1,dtype=int)
    else:
        old,original_rows,original_cols=previous
        rows=original_rows.copy()
        cols=original_cols.copy()
        ids=np.arange(n)
        if np.any(allowed & (rows[:,None]+cols < weights)):
            raise ArithmeticError('Restricted assignment increased a parent arc beyond its dual.')
        retained=allowed[ids,old] & (rows+cols[old]==weights[ids,old])
        donor=np.where(retained,old,-1)
        work['parent_arcs_retained']=int(retained.sum())
    owner=np.full(n,-1,dtype=int)
    matched=np.flatnonzero(donor>=0)
    owner[donor[matched]]=matched
    inf=np.iinfo(np.int64).max//4
    for root in np.flatnonzero(donor<0):
        work['augmentations']+=1
        active_rows=np.zeros(n,dtype=bool)
        active_cols=np.zeros(n,dtype=bool)
        slack=np.full(n,inf,dtype=np.int64)
        parent=np.full(n,-1,dtype=int)
        row=root
        while True:
            active_rows[row]=True
            eligible=allowed[row] & ~active_cols
            cost=rows[row]+cols-weights[row]
            update=eligible & (cost<slack)
            parent[update]=row
            slack[update]=cost[update]
            candidates=np.where(active_cols,inf,slack)
            column=int(candidates.argmin())
            step=int(candidates[column])
            if step==inf:
                return None,work
            if step<0:
                raise ArithmeticError('Negative dual slack on an allowed arc.')
            if step:
                rows[active_rows]-=step
                cols[active_cols]+=step
                finite=(slack!=inf) & ~active_cols
                slack[finite]-=step
            active_cols[column]=True
            work['alternating_columns']+=1
            row=owner[column]
            if row<0:
                while column>=0:
                    r=parent[column]
                    previous_col=donor[r]
                    donor[r]=column
                    owner[column]=r
                    column=previous_col
                break
    ids=np.arange(n)
    if (not np.all(allowed[ids,donor]) or len(np.unique(donor))!=n
            or np.any(allowed & (rows[:,None]+cols<weights))
            or int(weights[ids,donor].sum())!=int(rows.sum()+cols.sum())):
        raise ArithmeticError('Invalid restricted assignment witness or dual.')
    return (donor,rows,cols),work


def identity_dual(weights, allowed, limit):
    """Test a zero-objective assignment dual with the current donor orientation.

    For an unchanged feasible identity, rows=p and columns=-p have objective
    zero. Every allowed arc must satisfy p[i] >= weights[i,j]+p[j].
    """
    if np.any(np.diag(weights)>0):
        return False,0
    potential=np.zeros(len(weights),dtype=np.int64)
    for steps in range(limit+1):
        required=np.max(weights+potential,axis=1,where=allowed,initial=np.iinfo(np.int64).min)
        if np.all(required<=potential):
            return True,steps
        if steps<limit:
            potential=np.maximum(potential,required)
    return False,limit
