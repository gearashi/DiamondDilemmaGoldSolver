"""Exact rotated seam gain against the same boundary, with bounded-width elimination."""
import numpy as np


def eliminate(factors, eliminate_cells, limit, work):
    pending = set(eliminate_cells)
    while pending:
        touched = {c:[f for f in factors if c in f[0]] for c in pending}
        scopes = {c:tuple(sorted(set().union(*(s for s,_ in rows))))
                  for c,rows in touched.items()}
        cell = min(pending, key=lambda c:(len(scopes[c]), c))
        rows = touched[cell]
        pending.remove(cell)
        if not rows:
            continue
        scope = scopes[cell]
        entries = 3 ** len(scope)
        if entries > limit:
            return None
        work['largest'] = max(work.get('largest',0),entries)
        work['table_entries'] = work.get('table_entries',0) + entries
        combined = 0
        for axes,table in rows:
            shape = tuple(3 if c in axes else 1 for c in scope)
            combined = combined + table.reshape(shape)
        leave = tuple(c for c in scope if c != cell)
        result = np.max(combined,axis=scope.index(cell))
        factors = [f for f in factors if cell not in f[0]]
        factors.append((leave,result))
    return factors


def differential(bound, before, after, common, *, limit=59049, work=None):
    """Return max_boundary(opt_after-opt_before) or None on a width/legality limit."""
    if set(before) != set(after):
        raise ValueError('Both arrangements must price the same destinations.')
    if set(common) & set(before):
        raise ValueError('Common boundary donors must be exterior.')
    if limit < 1:
        raise ValueError('Expected a positive factor budget.')
    work = {} if work is None else work
    cells = set(before)
    seams = [(a,s,b,t) for a,s,b,t in bound.model.problem.edges if a in cells or b in cells]
    exterior = {c for a,_,b,_ in seams for c in (a,b)} - cells
    external_rows = []
    for c in exterior:
        codes = 3*bound.tiles[common.get(c,c)] + np.arange(3)
        legal = bound.model.legal[c,codes]
        if not legal.any():
            return None
        if not legal.all():
            external_rows.append(((c,),np.where(legal,0,-10000)))
    combined = []
    for multiplier,local in ((-1,before),(1,after)):
        sources = common | local
        options = {c:3*bound.tiles[sources.get(c,c)]+np.arange(3) for c in cells | exterior}
        factors = []
        for c in cells:
            legal = bound.model.legal[c,options[c]]
            if not legal.any():
                return None
            if not legal.all():
                factors.append(((c,),np.where(legal,0,-10000)))
        for a,s,b,t in seams:
            pair = (bound.model.masks[options[a],s,None] ==
                    bound.model.reverse[bound.model.masks[options[b],t]]).astype(int)
            factors.append(((a,b),pair) if a<b else ((b,a),pair.T))
        terms = eliminate(factors,cells,limit,work)
        if terms is None:
            return None
        combined.extend((scope,multiplier*value) for scope,value in terms)
    terms = eliminate([*combined,*external_rows],exterior,limit,work)
    if terms is None:
        return None
    return int(sum(value for _,value in terms))
