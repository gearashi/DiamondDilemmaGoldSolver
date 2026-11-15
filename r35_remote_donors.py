"""Select geographically distant repair donors with a legal, priced return exchange."""
import numpy as np

from r31_fresh import forest_pool


def response_pool(model,board,size,rng,*,imports=2,anchors=None,available=None,three=True,adaptive=False):
    board=model.legal_board(board)
    pool=sorted(set(map(int,model.free if available is None else available)))
    if not 0<=imports<size<=len(pool):
        raise ValueError('Invalid physical donor pool size or import budget.')
    forest=list(map(int,forest_pool(model,board,pool,size,rng)))
    if anchors is None:
        anchors=forest[:size-imports]
    anchors=tuple(dict.fromkeys(map(int,anchors)))
    if not set(anchors)<=set(pool) or not 1<=len(anchors)<=size:
        raise ValueError('Invalid physical anchor set.')
    score={}
    neighbors=[set(map(int,row)) - {-1} for row in model.problem.neighbors]
    near=set(anchors)
    for c in anchors:
        near.update(neighbors[c])
        for adj in neighbors[c]:
            near.update(neighbors[adj])
    remotes=[c for c in pool if c not in near]
    for c in (*anchors,*remotes):
        candidates=anchors if c in remotes else (*anchors,*remotes)
        for d in (*candidates,c):
            if (c,d) in score:
                continue
            codes=3*(board[d]//3)+np.arange(3)
            legal=codes[model.legal[c,codes]]
            if not len(legal):
                score[c,d]=None
                continue
            count=np.zeros(len(legal),dtype=int)
            for side,other in enumerate(model.problem.neighbors[c]):
                if other>=0:
                    opposite=model.problem.sides[c,side]
                    count+=model.masks[legal,side]==model.reverse[model.masks[board[other],opposite]]
            score[c,d]=int(max(count))
    selected=list(anchors)
    proposals=[]
    for d in remotes:
        best=None
        for a in anchors:
            cycles=[(a,d)]
            if three:
                cycles.extend((a,c,d) for c in anchors if c!=a and c not in neighbors[a])
            for cycle in cycles:
                fits=[score[x,y] for x,y in zip(cycle,(*cycle[1:],cycle[0]))]
                if any(v is None for v in fits):
                    continue
                gain=sum(fits)-sum(score[c,c] for c in cycle)
                options=(gain,-len(cycle),cycle)
                if best is None or options>best:
                    best=options
        if best is not None:
            proposals.append((best,d))
    # Keep distinct, complete return cycles; the exact repair may choose any
    # permutation of the resulting pool with every physical tile used once.
    imports_used=[]
    busy=set()
    for (gain,_,cycle),d in sorted(proposals,reverse=True):
        if len(selected)>=size or len(imports_used)>=imports:
            break
        if adaptive and (gain<=0 or busy.intersection(cycle)):
            continue
        selected.append(d)
        busy.update(cycle)
        imports_used.append(dict(donor=d,cycle=cycle,conditional_gain=gain))
    for c in (*forest,*pool):
        if len(selected)>=size:
            break
        if c not in selected:
            selected.append(c)
    return selected,dict(imports=imports_used,anchor_count=len(anchors),examined=len(remotes),
                         usable=len(proposals),response_rows=len(score))
