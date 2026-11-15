"""Propagate the joint retained-pose quota of two distinct reference rotations."""
from itertools import combinations
from functools import lru_cache


def pair_candidate(domains, cap):
    """A saturated pair needs 2*cap cells restricted entirely to its two poses."""
    return cap > 0 and 2*cap <= len(domains) and sum(0 < mask.bit_count() <= 2 for mask in domains) >= 2*cap


@lru_cache(maxsize=2)
def image_columns(images):
    sites = [dict() for _ in images[0]]
    for i,image in enumerate(images):
        for cell,pose in image:
            if pose in sites[cell]:
                raise ValueError('Pair quotas require distinct oriented images in each cell.')
            sites[cell][pose] = i
    return sites


def pair_bounds(domains, images, cap):
    """Keep all cell poses compatible with the saturated pairs. One-image rules run separately."""
    result = dict(over=False,removed=0,checked_pairs=0,tight_pairs=0)
    if not pair_candidate(domains,cap):
        return domains,result
    index=image_columns(images)
    fixed=[0]*len(images)
    pair_cells={}
    for c,mask in enumerate(domains):
        n=mask.bit_count()
        if n==1 and mask in index[c]:
            fixed[index[c][mask]]+=1
        elif n==2:
            bit=mask & -mask
            a=index[c].get(bit)
            b=index[c].get(mask^bit)
            if a is not None and b is not None:
                key=(min(a,b),max(a,b))
                pair_cells[key]=pair_cells.get(key,0)+1
    limit=2*cap
    tight=[]
    for a,b in combinations(range(len(images)),2):
        forced=fixed[a]+fixed[b]+pair_cells.get((a,b),0)
        result['checked_pairs']+=1
        if forced>limit:
            result.update(over=True,reason='pose_image_pair_quota',images=[a,b],forced=forced)
            return domains,result
        if forced==limit:tight.append((a,b))
    if not tight:return domains,result
    narrowed=list(domains)
    result['tight_pairs']=len(tight)
    # Clear the pair in every site which was not forced into it on entry.
    masks=[{c:pose for c,pose in images[i]} for i in range(len(images))]
    for a,b in tight:
        for c,old in enumerate(domains):
            owned=masks[a][c] | masks[b][c]
            if old & ~owned:
                narrowed[c]&=~owned
                if not narrowed[c]:
                    result.update(over=True,reason='pose_image_pair_domain',images=[a,b],cell=c)
                    return narrowed,result
    result['removed']=sum((old^new).bit_count() for old,new in zip(domains,narrowed))
    return narrowed,result
