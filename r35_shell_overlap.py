"""Count exact cycle-family shell intersections by relabeling untouched cells."""
from functools import lru_cache
from itertools import combinations,groupby
from math import comb,factorial
import numpy as np
from r34_cycle_orders import CycleOrders,family_count


def cycle_type(mapping):
    seen=set();result=[]
    for start in range(len(mapping)):
        if start in seen:continue
        here=start;size=0
        while here not in seen:
            seen.add(here);size+=1;here=mapping[here]
        if size>1:result.append(size)
    return tuple(sorted(result))


def distance(family):
    return sum(family)-len(family)


def can_meet(delta,a,b):
    d=distance(delta);da=distance(a);db=distance(b)
    return sum(delta)<=sum(a)+sum(b) and abs(da-db)<=d<=da+db and (da+db-d)%2==0


@lru_cache(maxsize=32)
def cycle_masks(length):
    """Binary subsets of one directed cycle, up to cyclic shifts, with multiplicity."""
    masks={}
    for mask in range(1<<length):
        if mask in masks:continue
        rotated=mask;orbit=set()
        for _ in range(length):
            orbit.add(rotated)
            rotated=((rotated<<1)&((1<<length)-1)) | (rotated>>(length-1))
        for value in orbit:masks[value]=min(orbit)
    from collections import Counter
    multiplicity=Counter(masks.values())
    return tuple((x,x.bit_count(),multiplicity[x]) for x in sorted(multiplicity))


def patterns(delta,k,max_right):
    h=sum(delta)
    def visit(i,offset,chosen,weight,old):
        if i==len(delta):
            count=len(chosen)
            if count<=k and h-count+k-count<=max_right:yield chosen,weight
            return
        length=delta[i];same=i>0 and delta[i-1]==length
        for mask,size,ways in cycle_masks(length):
            if len(chosen)+size>k or (same and mask<old):continue
            cells=tuple(offset+j for j in range(length) if mask&(1<<j))
            yield from visit(i+1,offset+length,chosen+cells,weight*ways,mask)
    for chosen,weight in visit(0,0,(),1,0):
        # Equal directed cycles may be interchanged. Their subset masks were
        # emitted in nondecreasing order above. Restore the multinomial factor.
        used=set(chosen);offset=0;masks=[]
        for length in delta:
            masks.append((length,sum(1<<j for j in range(length) if offset+j in used)))
            offset+=length
        for _,block in groupby(masks,key=lambda t:t[0]):
            masks_block=list(block);factor=factorial(len(masks_block))
            for _,same in groupby(masks_block):factor//=factorial(sum(1 for _ in same))
            weight*=factor
        yield chosen,weight


@lru_cache(maxsize=40)
def order_matrix(family):
    return np.asarray(tuple(CycleOrders(family)),dtype=np.int16)


def class_mask(rows,family):
    m,n=rows.shape
    cells=np.arange(n,dtype=np.int16)
    active=np.arange(m);power=np.broadcast_to(cells,(m,n))
    for exponent in range(1,max(family,default=1)+1):
        power=np.take_along_axis(rows[active],power,axis=1)
        expected=n-sum(family)+sum(size for size in family if exponent%size==0)
        ok=np.count_nonzero(power==cells,axis=1)==expected
        active=active[ok];power=power[ok]
        if not len(active):break
    return active


@lru_cache(maxsize=4096)
def intersection(delta,left,right,free):
    """Exact labeled endpoint count before geography, orientation and prior-shell exclusions."""
    delta=tuple(sorted(delta));left=tuple(sorted(left));right=tuple(sorted(right))
    if not can_meet(delta,left,right) or max(map(sum,(left,right,delta)))>free:return 0
    if not delta:return comb(free,sum(left))*family_count(left) if left==right else 0
    # Enumerate the less populous order catalogue. Membership in the other
    # class is tested using counts of fixed points of successive powers.
    if family_count(right)<family_count(left):left,right=right,left
    h=sum(delta);k=sum(left);orders=order_matrix(left);out=0
    moved=[];offset=0
    for length in delta:
        moved.extend(range(offset+1,offset+length));moved.append(offset);offset+=length
    for inside,weight in patterns(delta,k,sum(right)):
        z=k-len(inside)
        if z>free-h:continue
        cells=np.array((*inside,*range(h,h+z)),dtype=np.int16)
        inverse=np.argsort([*moved,*range(h,h+z)])
        base=np.broadcast_to(np.arange(h+z,dtype=np.int16),(len(orders),h+z)).copy()
        base[:,cells]=cells[orders]
        count=len(class_mask(inverse[base],right))
        out+=weight*comb(free-h,z)*count
    return out


def any_family(delta,families,free):
    return any(intersection(delta,a,b,free)>0 for a in families for b in families)

def union_count(family,relations,free,limit=500000):
    """Count overlapping earlier copies only if their small common label set is affordable."""
    signatures=sorted({tuple(sorted(relation.items())) for relation in relations})
    if not signatures:return 0
    special=sorted({cell for pairs in signatures for cell,_ in pairs})
    h=len(special);k=sum(family)
    eligible=range(max(0,k-free+h),min(k,h)+1)
    catalog_size=family_count(family)
    if sum(comb(h,q) for q in eligible)*catalog_size*len(signatures)>limit:return None
    index={cell:i for i,cell in enumerate(special)}
    physical=[dict(pairs) for pairs in signatures]
    local=[np.argsort([index[perm.get(c,c)] for c in special]) for perm in physical]
    orders=order_matrix(family);total=0
    for q in eligible:
        z=k-q;weight=comb(free-h,z)
        inv=[np.array([*a,*range(h,h+z)],dtype=np.int16) for a in local]
        for inside in combinations(range(h),q):
            support=np.array((*inside,*range(h,h+z)),dtype=np.int16)
            values=np.broadcast_to(np.arange(h+z,dtype=np.int16),(len(orders),h+z)).copy()
            values[:,support]=support[orders]
            used=np.zeros(len(orders),dtype=bool)
            for transform in inv:used[class_mask(transform[values],family)]=True
            total+=weight*int(used.sum())
    return total
