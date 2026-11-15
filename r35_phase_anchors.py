"""Test geographic Gold poses against the mates allowed by pinned physical tile phases."""
from collections import Counter, deque
from time import monotonic

from r29_gold_geography import checked_domains
from r29_gold_phase import checked_pins
from r35_mixed_sewings import mate_supports
from r35_signed_mate_support import raw_edges, reciprocal_classes


def allowed_mates(problem, pins, required=(), forbidden=()):
    sides,classes=reciprocal_classes(problem.tiles)
    forbidden={tuple(sorted(e)) for e in forbidden}
    required={tuple(sorted(e)) for e in required}
    fixed={}
    for a,b in required:
        if fixed.get(a,b)!=b or fixed.get(b,a)!=a:
            return set()
        fixed[a],fixed[b]=b,a
    allowed=set()
    for group in classes:
        for a,b,sign in raw_edges(sides,group):
            x,y=pins.get(a//3),pins.get(b//3)
            if ((x is None or y is None or x^y==sign) and (a,b) not in forbidden
                    and fixed.get(a,b)==b and fixed.get(b,a)==a):
                allowed.add((a,b))
    return allowed if required<=allowed else set()


def _rotate(mask,turn,first,full):
    for _ in range(turn):
        mask=((mask<<1)&(full^first))|((mask>>2)&first)
    return mask


def _symmetries(problem,domains,first,full):
    from constraint_symmetry import proper_rotations
    try:
        rotations=proper_rotations(problem.board)
    except ValueError:
        return ()
    if (tuple(tuple(map(int,row)) for row in problem.neighbors)!=problem.board.neighbor_cells
            or tuple(tuple(map(int,row)) for row in problem.sides)!=problem.board.neighbor_sides):
        return ()
    return tuple(r for r in rotations if all(
        domains[r.cells[c]]==_rotate(d,r.side_offsets[c],first,full) for c,d in enumerate(domains)))


def _branch(mesh,domains,supports,home,code,unique,stats):
    current=list(domains)
    current[home]=1<<code
    queue=deque([home])
    reserved=set()
    while queue:
        cell=queue.popleft()
        choices=current[cell]
        if not choices:return None
        if choices.bit_count()>32:continue
        if unique:
            tile=((choices&-choices).bit_length()-1)//3
            family=7<<(3*tile)
            if not choices&~family and cell not in reserved:
                reserved.add(cell)
                for other,mask in enumerate(current):
                    if other!=cell and mask&family:
                        remaining=mask&~family
                        if not remaining:return None
                        current[other]=remaining
                        if remaining.bit_count()<=32:queue.append(other)
        stats['cell_revisions']+=1
        for side,(other,opposite) in enumerate(mesh[cell]):
            if other<0:continue
            allowed=0
            choices=current[cell]
            row=supports[side,opposite]
            while choices:
                bit=choices&-choices
                allowed|=row[bit.bit_length()-1]
                choices^=bit
            updated=current[other]&allowed
            if not updated:return None
            if updated!=current[other]:
                current[other]=updated
                queue.append(other)
    return current


def narrow(problem,domains,pins,*,required=(),forbidden=(),anchors=None,symmetry=True,
           unique=True,project=True,repeat=True,deadline=float('inf')):
    """A completed split may exclude poses. A deadline discards its untested branches."""
    initial=checked_domains(problem,domains)
    pins=checked_pins(problem.n,pins)
    allowed=allowed_mates(problem,pins,required,forbidden)
    supports=mate_supports(problem.n,allowed)
    mesh=tuple(tuple(zip(map(int,neighbors),map(int,sides))) for neighbors,sides in zip(problem.neighbors,problem.sides))
    stats=Counter()
    full=(1<<(3*problem.n))-1
    first=full//7
    rotations=_symmetries(problem,initial,first,full) if symmetry else ()
    degrees=Counter(side for edge in allowed for side in edge)
    def rarity(tile):
        a,b,c=sorted(degrees[3*tile+s] for s in range(3))
        return a*b*c,b,c,tile
    order=sorted(range(problem.n),key=rarity) if anchors is None else list(dict.fromkeys(anchors))
    if any(type(t) is not int or not 0<=t<problem.n for t in order):
        raise ValueError('Invalid physical anchor tile.')
    current=list(initial)
    def answer(status):
        return dict(domains=tuple(current),status=status,removed=sum((a^b).bit_count() for a,b in zip(initial,current)),
                    stats=dict(stats),symmetries=len(rotations) or 1)
    if not all(current):return answer('INFEASIBLE')
    while True:
        old=sum(d.bit_count() for d in current)
        for tile in order:
            if monotonic()>=deadline:return answer('LIMIT')
            offset=3*tile
            roots={(c,r) for c,d in enumerate(current) for r in range(3) if d&(1<<(offset+r))}
            remaining=set(roots)
            forbidden_roots=set()
            common=None
            any_live=False
            for c,r in sorted(roots):
                if (c,r) not in remaining:continue
                if monotonic()>=deadline:return answer('LIMIT')
                orbit=[(c,r,None)] if not rotations else [(g.cells[c],(r+g.side_offsets[c])%3,g) for g in rotations]
                remaining.difference_update((d,t) for d,t,_ in orbit)
                stats['root_probes']+=1
                result=_branch(mesh,current,supports,c,offset+r,unique,stats)
                if result is None:
                    forbidden_roots.update((d,t) for d,t,_ in orbit)
                    continue
                any_live=True
                if project and (common is None or common):
                    # Intersect the exclusions across every surviving placement
                    # of this tile, transporting each rotated branch back onto the board.
                    changed={i:before^after for i,(before,after) in enumerate(zip(current,result)) if before!=after}
                    for _,_,g in orbit:
                        image=changed if g is None else {g.cells[i]:_rotate(m,g.side_offsets[i],first,full) for i,m in changed.items()}
                        if common is None:common=image.copy()
                        else:common={i:m&image[i] for i,m in common.items() if i in image and m&image[i]}
                        if not common:break
            if not any_live:return answer('INFEASIBLE')
            for c,r in forbidden_roots:current[c]&=~(1<<(offset+r))
            for c,mask in (common or {}).items():current[c]&=~mask
            stats['anchor_splits']+=1
            if not all(current):return answer('INFEASIBLE')
        if not repeat or sum(d.bit_count() for d in current)==old:break
        stats['changed_passes']+=1
    return answer('SURVIVING')
