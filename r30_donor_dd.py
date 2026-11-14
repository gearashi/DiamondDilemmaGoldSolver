"""Exact and width-relaxed decision diagrams for cycle-constrained donor assignments.

The objective is a sum of optimistic rotation-consistent local stars, counted twice.
A diagram may certify a whole support. The exact rotation oracle handles survivors.
"""
import numpy as np


def donor_factors(bound, cells):
    """Each changed destination is a variable; a star depends on at most four of them."""
    cells = tuple(cells)
    if not cells or len(cells)!=len(set(cells)) or any(c<0 or c>=bound.n for c in cells):
        raise ValueError("Expected distinct valid exchange cells.")
    local = {c:i for i,c in enumerate(cells)}
    touched = sorted(set(cells).union(*(set(bound.neighbors[c]) for c in cells))-{bound.n})
    constant = bound.total - sum(int(bound.base[v]) for v in touched)
    tiles = bound.tiles[list(cells)]
    choices = np.arange(len(cells))
    factors = []
    for v in touched:
        around = (v,*map(int,bound.neighbors[v]))
        axes = tuple(sorted({local[c] for c in around if c in local}))
        grids = np.ix_(*(choices for _ in axes))
        inputs = np.broadcast_arrays(*[tiles[grids[axes.index(local[c])]] if c in local else bound.pieces[c]
                                       for c in around])
        table = bound.stars(inputs[0], np.stack(inputs[1:], axis=-1))
        # A constant axis carries no boundary information.
        axes = list(axes)
        for i in reversed(range(len(axes))):
            if np.all(table == np.take(table,[0],axis=i)):
                table = np.take(table,0,axis=i)
                axes.pop(i)
        if axes:
            factors.append((tuple(axes),table))
        else:
            constant += int(table)
    return constant, factors


def _advance_paths(paths, dest, donor):
    """A partial injection consists of open directed paths and completed cycles."""
    left = next(((a,b,n) for a,b,n in paths if b==dest),None)
    right = next(((a,b,n) for a,b,n in paths if a==donor),None)
    new = [p for p in paths if p!=left and p!=right]
    if left is not None and left==right:
        return tuple(new),left[2]+1
    start, prior = (left[0],left[2]) if left else (dest,0)
    stop, after = (right[1],right[2]) if right else (donor,0)
    new.append((start,stop,prior+1+after))
    return tuple(sorted(new)),0


def diagram(bound,cells,cycle_type,*,width=None,merge=True):
    """Return an integral upper bound; a finite width can merge distinct cycle histories.

    Overflow represents the same used donors with unconstrained path/cycle history.
    It preserves every represented assignment and may allow other cycle types.
    Factor boundary assignments are always retained, including after an overflow.
    """
    cells = tuple(map(int,cells))
    sizes = tuple(sorted(cycle_type))
    if not sizes or min(sizes)<2 or sum(sizes)!=len(cells) or (width is not None and width<1):
        raise ValueError("Invalid cycle partition or width.")
    constant,factors = donor_factors(bound,cells)
    n = len(cells)
    allowed = bound.model.legal.reshape(bound.n,bound.n,3).any(2)
    eligible = [tuple(j for j,c in enumerate(cells) if j!=i and allowed[cells[i],bound.tiles[c]])
                for i in range(n)]
    boundaries = [tuple(sorted({a for axes,_ in factors if max(axes)>=i for a in axes if a<i}))
                  for i in range(n+1)]
    ends = [[(axes,t) for axes,t in factors if max(axes)==i] for i in range(n)]
    # state = used donor mask, exposed donor choices, open paths, unclosed cycle lengths
    states = {(0,(),(),sizes,()):constant}
    visits = merged = relaxed = 0
    history_dropped = False
    for i in range(n):
        next_states = {}
        index = {a:k for k,a in enumerate((*boundaries[i],i))}
        keep_axes = [index[a] for a in boundaries[i+1]]
        local = [(tuple(index[a] for a in axes),table) for axes,table in ends[i]]
        for (used,bdy,paths,remaining,trail),base in states.items():
            for j in eligible[i]:
                bit = 1<<j
                if used & bit:
                    continue
                visits += 1
                active, wanted = paths,remaining
                if paths is not None:
                    active,length = _advance_paths(paths,i,j)
                    if length:
                        if length not in remaining:
                            continue
                        values = list(remaining)
                        values.remove(length)
                        wanted = tuple(values)
                    if any(length >= max(wanted,default=0) for _,_,length in active):
                        continue
                values = (*bdy,j)
                score = base + sum(int(table[tuple(values[k] for k in axes)]) for axes,table in local)
                key = (used|bit,tuple(values[k] for k in keep_axes),active,wanted,() if merge else trail+(j,))
                if key in next_states:
                    merged += 1
                if score > next_states.get(key,-1):
                    next_states[key] = score
        if width and len(next_states)>width:
            # Keep one optimistic supremum per immutable inventory/boundary. This is
            # a soft limit when more distinct mandatory boundaries than width remain.
            combined = {}
            for (used,bdy,_,_,_),score in next_states.items():
                key = (used,bdy,None,None,())
                combined[key] = max(score,combined.get(key,-1))
            history_dropped = True
            relaxed += len(next_states)-len(combined)
            next_states = combined
        states = next_states
    return dict(upper_twice=max(states.values(),default=-1),transitions=visits,merged=merged,
                relaxed_merges=relaxed,history_dropped=history_dropped,width=width)


def screened_diagram(bound,cells,cycle_type,target,width=64):
    """Refine the relaxation only when a threshold survives a lossy merge."""
    fast = diagram(bound,cells,cycle_type,width=width)
    if fast['upper_twice']<2*target or not fast['history_dropped']:
        return fast
    exact = diagram(bound,cells,cycle_type)
    exact['relaxed_transitions'] = fast['transitions']
    return exact
