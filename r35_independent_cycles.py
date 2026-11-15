"""Exact best rotations of single physical donor cycles against an unchanged exterior."""
from collections import Counter, defaultdict
from itertools import combinations, permutations

import numpy as np

from r35_cycle_contacts import choose


def independent_cycles(model, board, cells, max_length):
    cells = tuple(sorted(set(map(int, cells))))
    donors = np.array([board[c]//3 for c in cells], dtype=int)
    codes = 3*donors[:,None]+np.arange(3)
    legal = model.legal[np.array(cells,dtype=int)[:,None,None],codes]
    before = model.matches(board).astype(np.int16)
    slots = {c:i for i,c in enumerate(cells)}
    unary = np.zeros((len(cells),len(cells),3),dtype=np.int16)
    internal = defaultdict(list)
    work = Counter()
    for e,(a,side,b,opposite) in enumerate(model.problem.edges):
        if a not in slots and b not in slots:
            continue
        old = before[e]
        left = model.masks[codes,side] if a in slots else None
        right = model.reverse[model.masks[codes,opposite]] if b in slots else None
        if a == b:
            unary[slots[a]] += (left==right)-old
            continue
        if a in slots:
            old_right = model.reverse[model.masks[board[b],opposite]]
            unary[slots[a]] += (left==old_right)-old
        if b in slots:
            old_left = model.masks[board[a],side]
            unary[slots[b]] += (old_left==right)-old
        if a in slots and b in slots:
            i,j = slots[a],slots[b]
            internal[min(i,j),max(i,j)].append((i,j,left,right,old_left,old_right,int(old)))
    unary[~legal] = -1000
    valid = legal.any(axis=2)
    single = unary.max(axis=2)
    rotation = unary.argmax(axis=2)
    tables = {}
    choices = {}
    work['single_donor_rotations'] = int(legal.sum())

    def pair_table(x, y, donor_x, donor_y):
        k = (x,y,donor_x,donor_y)
        if k not in tables:
            terms = np.zeros((3,3),dtype=np.int16)
            for i,j,left,right,old_left,old_right,old in internal[min(x,y),max(x,y)]:
                if x == i:
                    l,r = left[donor_x][:,None],right[donor_y][None,:]
                else:
                    l,r = left[donor_y][None,:],right[donor_x][:,None]
                terms += ((l==r).astype(np.int16)+old-(l==old_right)-(old_left==r))
            tables[k] = terms
            work['single_pair_tables'] += 1
        return tables[k]

    for length in range(2,max_length+1):
        for support in combinations(range(len(cells)),length):
            for tail in permutations(support[1:]):
                cycle = (support[0],*tail)
                sources = (*tail,support[0])
                if any(not valid[c,d] for c,d in zip(cycle,sources)):
                    continue
                work['cycles_priced'] += 1
                orient = [int(rotation[c,d]) for c,d in zip(cycle,sources)]
                value = sum(int(single[c,d]) for c,d in zip(cycle,sources))
                crossed = [(a,b) for a,b in combinations(range(length),2)
                           if (min(cycle[a],cycle[b]),max(cycle[a],cycle[b])) in internal]
                footprint = frozenset(cells[c] for c in cycle)
                threshold = choices.get(footprint,(0,None))[0]
                if not crossed:
                    work['separable_cycles'] += 1
                else:
                    upper = value + sum(1+edge[-1] for a,b in crossed
                                        for edge in internal[min(cycle[a],cycle[b]),max(cycle[a],cycle[b])])
                    if upper <= threshold:
                        work['single_bound_skips'] += 1
                        continue
                    active = sorted({axis for edge in crossed for axis in edge})
                    grid = np.zeros((3,)*len(active),dtype=np.int16)
                    fixed = value
                    for pos,axis in enumerate(active):
                        c,d = cycle[axis],sources[axis]
                        dims = [1]*len(active)
                        dims[pos] = 3
                        fixed -= int(single[c,d])
                        grid += unary[c,d].reshape(dims)
                    for a,b in crossed:
                        dims = [1]*len(active)
                        dims[active.index(a)] = dims[active.index(b)] = 3
                        grid += pair_table(cycle[a],cycle[b],sources[a],sources[b]).reshape(dims)
                    best = np.unravel_index(grid.argmax(),grid.shape)
                    value = fixed+int(grid[best])
                    for axis,r in zip(active,best):
                        orient[axis] = int(r)
                    work['single_joint_rotation_entries'] += grid.size
                if value > threshold:
                    choices[footprint] = value, {cells[c]:int(codes[d,r]) for c,d,r in zip(cycle,sources,orient)}
    gain,changes = choose(model, choices)
    result = board.copy()
    for item in changes:
        for cell,code in item.items():
            result[cell] = code
    if int(model.matches(result).sum()-before.sum()) != gain:
        raise ArithmeticError('Separable cycle prices disagree with the reconstructed physical board.')
    return result,dict(work)
