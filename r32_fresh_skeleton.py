"""Sample and lift geographic odd-seam contours without walking the full frontier."""
import numpy as np

from r27_incumbent import assignment
from r29_gold_geography import checked_domains
from r27_structure_filters import bits
from r32_fresh_geometry import _apex_path, skeleton_domains


def parity_seed(problem, seed, *, domains=None, trials=2000, max_masks=64):
    """Geographic vertex colors give zero/two odd edges per face, including multiple contours."""
    if trials<1 or max_masks<1 or seed<0:
        raise ValueError('Positive contour trial limits and nonnegative seed required.')
    ds=checked_domains(problem,domains)
    rng=np.random.default_rng(seed)
    vertices=np.array([row.vertices for row in problem.board.cells],dtype=int)
    width=int(vertices.max())+1
    parity=np.array([[int(m).bit_count()%2 for m in row] for row in problem.masks])
    target=int((parity[::3].sum(axis=1)>0).sum())
    apices=np.flatnonzero(np.bincount(vertices.ravel())%2) if _apex_path(problem) else []
    proposed=set()
    queries=0
    for step in range(1,trials+1):
        # Vary color density to cover both large and small occupied contours.
        density=rng.uniform(.18,.82)
        colors=(rng.random(width)<density).astype(np.int8)
        if len(apices):
            colors[apices[0]],colors[apices[1]]=0,1
        side=colors[vertices]^colors[vertices[:,[1,2,0]]]
        if int((side.sum(axis=1)>0).sum())!=target:
            continue
        pattern=side @ np.array([1,2,4])
        key=pattern.tobytes()
        if key in proposed:
            continue
        proposed.add(key)
        sub=skeleton_domains(problem,ds,pattern)
        if not all(sub):
            continue
        allowed=np.zeros((problem.n,3*problem.n),dtype=bool)
        for c,d in enumerate(sub):
            allowed[c,list(bits(d))]=True
        options=allowed.reshape(problem.n,problem.n,3)
        weights=np.where(options,rng.integers(0,64,size=options.shape),-1)
        queries+=1
        try:
            donors=assignment(weights.max(axis=2),options.any(axis=2))
        except ValueError:
            if queries>=max_masks:
                break
            continue
        rotations=weights[np.arange(problem.n),donors].argmax(axis=1)
        codes=(3*donors+rotations).tolist()
        return dict(status='PARITY_PLACEMENT',trials=step,assignments=queries,active=target,
                    patterns=pattern.tolist(),codes=codes,actual_matches=int(problem.validate(codes)['matched_edges']))
    return dict(status='UNKNOWN',trials=step,assignments=queries,active=target)
