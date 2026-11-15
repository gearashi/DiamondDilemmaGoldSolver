"""Joint support for labelled rigid-component poses and residual unrestricted tile owners."""
from collections import Counter,defaultdict

from constraint_advanced import _complete_matching
from constraint_pruning import _tile_choices
from r27_structure_filters import Work,WorkLimit,bits


def narrowed_poses(domains,clusters,*,residual=True,work_limit=200000,free_support=True):
    n=len(domains)
    groups=[]
    masks=[]
    for members,member_codes,embeddings in clusters:
        mandatory=sum(1<<c for c,row in enumerate(domains) if not row&~member_codes)
        choices=defaultdict(list)
        for pose in embeddings:
            if all(domains[c]>>v&1 for c,v in pose):
                footprint=sum(1<<c for c,_ in pose)
                if mandatory&~footprint==0:
                    choices[footprint].append(pose)
        groups.append(choices);masks.append(member_codes)
    stats=Counter(initial_labelled=sum(len(v) for g in groups for v in g.values()),
                  initial_footprints=sum(map(len,groups)))
    free_mask=(1<<(3*n))-1
    for mask in masks:free_mask&=~mask
    free_rows=[_tile_choices(row&free_mask) for row in domains]
    visible=[set(g) for g in groups]
    proven=[set() for g in groups]
    free_proven=[0]*n
    free_failed=[set() for _ in domains]
    distinct_pools=set(free_rows)-{0}
    covered_tiles=0
    disjoint_pools=True
    for pool in distinct_pools:
        if covered_tiles&pool:disjoint_pools=False;break
        covered_tiles|=pool
    work=Work(work_limit)
    complete=True

    def unfillable():
        return dict(domains=tuple(0 for _ in domains),poses=(),complete=True,stats=dict(stats,impossible=1))

    if any(not row for row in domains) or any(not g for g in groups):return unfillable()
    occupied_by=[0]*len(groups)
    for g,choices in enumerate(groups):
        for foot in choices:occupied_by[g]|=foot
    connected=[]
    for root in range(len(groups)):
        seen={root};stack=[root]
        while stack:
            i=stack.pop()
            for j,areas in enumerate(occupied_by):
                if j not in seen and areas&occupied_by[i]:
                    seen.add(j);stack.append(j)
        connected.append(seen)

    def has_support(group=None,footprint=0,free_pair=None):
        chosen={}
        if group is not None:chosen[group]=footprint
        active=set(range(len(groups))) if residual or group is None else connected[group]
        todo=sorted(active-chosen.keys(),key=lambda g:len(visible[g]))
        forbidden=0 if free_pair is None else 1<<free_pair[0]
        def search(i,used):
            work.tick('pose_pack_nodes')
            if i<len(todo):
                g=todo[i]
                for foot in sorted(visible[g]):
                    work.tick('pose_overlap_tests')
                    if not foot&(used|forbidden):
                        chosen[g]=foot
                        if search(i+1,used|foot):return True
                chosen.pop(g,None)
                return False
            if residual:
                cells=[c for c in range(n) if not used>>c&1]
                rows=[free_rows[c] for c in cells]
                if disjoint_pools:
                    histogram=Counter(rows)
                    if any(count>pool.bit_count() for pool,count in histogram.items()):return False
                    for c in cells:free_proven[c]|=free_rows[c]
                    stats['disjoint_pool_witnesses']+=1
                else:
                    if free_pair is not None:
                        cell,tile=free_pair
                        rows[cells.index(cell)]&=1<<tile
                    match=_complete_matching(rows,lambda:work.tick('free_matching_steps'))
                    if match is None:return False
                    for c,t in zip(cells,match):free_proven[c]|=1<<t
            stats['support_witnesses']+=1
            for g,foot in chosen.items():proven[g].add(foot)
            return True
        return search(0,footprint)

    try:
        if not has_support():return unfillable()
        for g,choices in enumerate(groups):
            for foot in sorted(choices):
                if foot not in proven[g] and not has_support(g,foot):
                    visible[g].remove(foot)
                    stats['unsupported_footprints']+=1
        if residual and free_support:
            blocked=0
            for family in visible:
                always=(1<<n)-1
                for foot in family:always&=foot
                blocked|=always
            for cell,row in enumerate(free_rows):
                if blocked>>cell&1:continue
                for tile in bits(row):
                    if not free_proven[cell]>>tile&1 and not has_support(free_pair=(cell,tile)):
                        free_failed[cell].add(tile);stats['unsupported_free_owners']+=1
    except WorkLimit:
        stats['unknown_budget']+=1
        complete=False
    kept=list(domains)
    posed=[]
    for g,choices in enumerate(groups):
        poses=tuple(p for foot in sorted(visible[g]) for p in choices[foot])
        posed.append(poses)
        always=(1<<n)-1;projected=[0]*n
        for foot in visible[g]:always&=foot
        for pose in poses:
            for c,v in pose:projected[c]|=1<<v
        for c in range(n):
            kept[c]&=projected[c] if always>>c&1 else projected[c]|~masks[g]
    for c,tiles in enumerate(free_failed):
        for tile in tiles:kept[c]&=~(7<<(3*tile))
    stats.update(work.counters)
    stats['removed_codes']=sum((x^y).bit_count() for x,y in zip(domains,kept))
    stats['remaining_labelled']=sum(map(len,posed))
    return dict(domains=tuple(kept),poses=tuple(posed),complete=complete,stats=dict(stats))


def branch_domains(state):
    """Distinct labelled poses of a component partition its geographic assignments."""
    families=[poses for poses in state['poses'] if len(poses)>1]
    if not families:return ()
    choices=min(families,key=len)
    result=[]
    for pose in choices:
        child=list(state['domains'])
        owned=sum(7<<(3*(v//3)) for _,v in pose)
        occupied={c:1<<v for c,v in pose}
        for c in range(len(child)):
            child[c]&=occupied.get(c,~owned)
        result.append(tuple(child))
    return tuple(result)
