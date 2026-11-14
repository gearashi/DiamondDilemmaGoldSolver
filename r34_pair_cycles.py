"""Price contacting donor cycles over legal rotations with an exact shared-seam correction."""
from collections import Counter
from itertools import combinations, permutations, product

import numpy as np

from r34_swap_packing import pack_swaps


def pair_repair(model, board, cells, max_length=2, *, interact=True):
    if max_length not in (2,3):
        raise ValueError("Expected a maximum cycle length of 2 or 3.")
    cells = tuple(sorted(set(map(int,cells))))
    before = model.matches(board).astype(np.int8)
    incident = {c:[] for c in cells}
    for edge,(a,s,b,t) in enumerate(model.problem.edges):
        for c in (a,b):
            if c in incident:
                incident[c].append((edge,a,s,b,t))
    choices = {}
    rows = []
    work = Counter()
    def save(footprint,codes,gain):
        key = frozenset(footprint)
        if gain>0 and (key not in choices or choices[key][0]<gain):
            choices[key] = gain, dict(zip(footprint,map(int,codes)))

    for length in range(2,max_length+1):
        rotations = np.array(list(product(range(3),repeat=length)))
        for support in combinations(cells,length):
            edges = sorted({e for c in support for e in incident[c]})
            for tail in permutations(support[1:]):
                cycle = (support[0],*tail)
                sources = (*tail,support[0])
                codes = 3*(board[list(sources)]//3) + rotations
                codes = codes[model.legal[list(cycle),codes].all(axis=1)]
                if not len(codes):
                    continue
                local = {c:codes[:,i] for i,c in enumerate(cycle)}
                delta = np.zeros(len(codes),dtype=np.int16)
                boundary = {}
                for e,a,s,b,t in edges:
                    left = model.masks[local.get(a,board[a]),s]
                    right = model.reverse[model.masks[local.get(b,board[b]),t]]
                    match = (left==right).astype(np.int8)
                    delta += match-before[e]
                    if a not in local or b not in local:
                        boundary[e] = (left if a in local else right, match)
                best = int(delta.argmax())
                gain = int(delta[best])
                save(cycle,codes[best],gain)
                rows.append((frozenset(cycle),cycle,codes,delta,boundary,gain))
                work['cycles_priced'] += 1
    singles = choices.copy() if interact else choices
    if interact:
        for i,(sa,ca,ra,da,ba,ga) in enumerate(rows):
            for j in range(i):
                sb,cb,rb,db,bb,gb = rows[j]
                if sa & sb or min(ga,gb)>0:
                    continue
                cross = ba.keys() & bb.keys()
                if not cross or ga+gb+2*len(cross)<=max(0,ga,gb):
                    continue
                joint = da[:,None]+db
                for edge in cross:
                    ma,pa = ba[edge]
                    mb,pb = bb[edge]
                    joint += (ma[:,None]==mb).astype(np.int8)-pa[:,None]-pb+before[edge]
                work['joint_prices'] += 1
                a,b = np.unravel_index(joint.argmax(),joint.shape)
                gain = int(joint[a,b])
                if gain>max(0,ga,gb):
                    work['joint_improvements'] += 1
                    save((*ca,*cb),(*ra[a],*rb[b]),gain)
    def pack(options):
        items = sorted(options.values(),key=lambda x:-x[0])
        selected = pack_swaps(model,[tuple(item) for _,item in items],[gain for gain,_ in items])
        return sum(items[i][0] for i in selected),[items[i][1] for i in selected]
    predicted,changes = pack(choices)
    if len(choices)>len(singles):
        previous,baseline = pack(singles)
        if previous>predicted:
            predicted,changes = previous,baseline
    result = board.copy()
    for change in changes:
        for c,code in change.items():
            result[c] = code
    if int(model.matches(result).sum()-before.sum()) != predicted:
        raise ArithmeticError("Packed shared-seam corrections disagree with full board.")
    return result, dict(work)
