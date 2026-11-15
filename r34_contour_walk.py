"""Count-preserving vertex flips with a physical donor assignment on changed cells."""
from itertools import combinations

import numpy as np

from r27_incumbent import assignment
from r29_gold_geography import checked_domains
from r32_fresh_geometry import _apex_path
from r32_fresh_skeleton import parity_seed


def contour_walk(problem, seed, domains=None, *, limit=8, trials=2000):
    if limit<1 or trials<1:
        raise ValueError('Positive contour and proposal limits required.')
    ds=checked_domains(problem,domains)
    initial=parity_seed(problem,seed,domains=ds,trials=trials,conditioned=True)
    if 'codes' not in initial:
        return dict(patterns=[],complete=False,sample=initial,proposals=0,assignment_cells=0)
    vertices=np.asarray([c.vertices for c in problem.board.cells],dtype=int)
    width=int(vertices.max())+1
    pattern=np.array(initial['patterns'],dtype=int)
    board=np.asarray(initial['codes'],dtype=int)
    code_pattern=np.array([sum((int(v).bit_count()%2)<<s for s,v in enumerate(row)) for row in problem.masks])
    rng=np.random.default_rng(seed)
    neighbor=[{} for _ in range(width)]
    for c,triangle in enumerate(vertices):
        for s,v in enumerate(triangle):
            other=triangle[(s+1)%3]
            neighbor[v][other]=neighbor[other][v]=int((pattern[c]>>s)&1)
    colors=np.zeros(width,dtype=np.int8)
    seen={0};stack=[0]
    while stack:
        u=stack.pop()
        for v,d in neighbor[u].items():
            if v not in seen:
                seen.add(v);stack.append(v);colors[v]=colors[u]^d
    apices=set(np.flatnonzero(np.bincount(vertices.ravel())%2)) if _apex_path(problem) else set()
    movable=[v for v in range(width) if v not in apices]
    incidence=[set(np.flatnonzero(np.any(vertices==v,axis=1)).tolist()) for v in range(width)]
    parity_weights=np.asarray([1,2,4])
    def changed(flip,cells):
        row=colors[vertices[cells]].copy()
        for v in flip:
            row ^= vertices[cells]==v
        return (row ^ row[:,[1,2,0]]) @ parity_weights
    samples=[pattern.tolist()]
    visited={pattern.tobytes()}
    best=dict(codes=board.tolist(),matched_edges=initial['actual_matches'])
    work=problem.n
    proposals=0
    while len(samples)<limit and proposals<trials:
        single={}
        for v in movable:
            cells=sorted(incidence[v])
            single[v]=int(np.count_nonzero(changed((v,),cells))-np.count_nonzero(pattern[cells]))
        moves=[(v,) for v in movable if single[v]==0]
        for v,w in combinations(movable,2):
            # Only adjacent vertices can affect the same face.
            if w in neighbor[v] or single[v]+single[w]==0:
                moves.append((v,w))
        for i in rng.permutation(len(moves)):
            if proposals>=trials:
                break
            proposals+=1
            move=moves[i]
            cells=sorted(set.union(*(incidence[v] for v in move)))
            new=changed(move,cells)
            if np.count_nonzero(new)!=np.count_nonzero(pattern[cells]):
                continue
            candidate=pattern.copy()
            candidate[cells]=new
            key=candidate.tobytes()
            if key in visited:
                continue
            changed_cells=np.asarray([c for c in cells if candidate[c]!=pattern[c]],dtype=int)
            if not len(changed_cells):
                continue
            donors=3*(board[changed_cells]//3)[:,None]+np.arange(3)
            legal=np.array([[(int(ds[c])>>int(v))&1 for v in donors.ravel()] for c in changed_cells],
                           dtype=bool).reshape(len(changed_cells),len(changed_cells),3)
            legal &= code_pattern[donors][None,:,:]==candidate[changed_cells,None,None]
            weights=np.where(legal,rng.integers(0,64,legal.shape),-1)
            work+=len(changed_cells)
            try:
                selected=assignment(weights.max(axis=2),legal.any(axis=2))
            except ValueError:
                # Another trajectory may reach this pattern with different donors.
                continue
            rot=weights[np.arange(len(changed_cells)),selected].argmax(axis=1)
            board[changed_cells]=donors[selected,rot]
            if not np.array_equal(code_pattern[board],candidate):
                raise ArithmeticError('Incremental contour assignment changed the requested pattern.')
            score=problem.validate(board.tolist())['matched_edges']
            if score>best['matched_edges']:
                best=dict(codes=board.tolist(),matched_edges=int(score))
            colors[list(move)] ^= 1
            visited.add(key)
            samples.append(candidate.tolist())
            pattern=candidate
            break
        else:
            break
    return dict(patterns=samples,complete=False,sample=initial,proposals=proposals,assignment_cells=work,best=best)
