"""Sample and lift geographic odd-seam contours without walking the full frontier."""
import numpy as np

from r27_incumbent import assignment
from r29_gold_geography import checked_domains
from r27_structure_filters import bits
from r32_fresh_geometry import _apex_path, skeleton_domains


def _condition_count(vertices, incident, neighbors, movable, colors, target, rng):
    """Fill independent vertex colors to attain exactly the required active-face count."""
    chosen = []
    blocked = set()
    for v in rng.permutation(movable):
        v = int(v)
        if v not in blocked:
            chosen.append(v)
            blocked.update(neighbors[v])
    faces = colors[vertices]
    active = np.any(faces != faces[:, :1],axis=1)
    remaining = target-int(active.sum())
    counts = []
    for v in chosen:
        cells, left, right = incident[v]
        remaining += int(active[cells].sum())
        counts.append((int(np.count_nonzero(colors[left] | colors[right])),
                       int(np.count_nonzero(1-(colors[left] & colors[right])))))
    # Suffix counts allow direct sampling among reachable assignments without
    # enumerating 2**len(chosen) possibilities or changing the fixed apices.
    ways = [None]*(len(chosen)+1)
    ways[-1] = {0:1}
    for i in range(len(chosen)-1,-1,-1):
        row = {}
        for cost in counts[i]:
            for prior, weight in ways[i+1].items():
                total = prior+cost
                if total<=remaining:
                    row[total] = row.get(total,0)+weight
        ways[i] = row
    if remaining not in ways[0]:
        return False
    for i,v in enumerate(chosen):
        low,high = counts[i]
        n0 = ways[i+1].get(remaining-low,0)
        n1 = ways[i+1].get(remaining-high,0)
        value = int(rng.random() >= n0/(n0+n1))
        colors[v] = value
        remaining -= counts[i][value]
    return True


def parity_seed(problem, seed, *, domains=None, trials=2000, max_masks=64, conditioned=False):
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
    if conditioned:
        incident = {}
        neighbors = {}
        movable = [v for v in range(width) if v not in apices]
        for v in movable:
            cells, corner = np.where(vertices==v)
            left = vertices[cells,(corner+1)%3]
            right = vertices[cells,(corner+2)%3]
            incident[v] = cells,left,right
            neighbors[v] = set(left)|set(right)
    proposed=set()
    queries=0
    for step in range(1,trials+1):
        # Vary color density to cover both large and small occupied contours.
        density=rng.uniform(.18,.82)
        colors=(rng.random(width)<density).astype(np.int8)
        if len(apices):
            colors[apices[0]],colors[apices[1]]=0,1
        if conditioned and not _condition_count(vertices,incident,neighbors,movable,colors,target,rng):
            continue
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
