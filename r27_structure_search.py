"""Bounded exact search experiments over placements, seam labels and physical sewings.

Every completed witness is checked with the original independent validator. A
resource-limited run reports partial enumeration explicitly. The standalone
comparison always enumerates the labelled answers, including all rotations.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from validator import reverse_mask
from r27_structure_filters import (Union, Work, WorkLimit, _vertex_links, bits,
                                   domain_bits, physical_graph)


def _edge_filter(problem, masks, work):
    """Exact-one singleton elimination and ordinary seam arc consistency."""
    ds=list(masks)
    while True:
        before=ds.copy()
        forced={}
        for c,mask in enumerate(ds):
            if not mask:return None
            tiles={x//3 for x in bits(mask)}
            if len(tiles)==1:
                t=tiles.pop()
                if t in forced:return None
                forced[t]=c
        used=sum(7<<(3*t) for t in forced)
        for c,mask in enumerate(ds):
            kept=mask&~used
            if not kept and mask and all((x//3 in forced and forced[x//3]==c) for x in bits(mask)):
                kept=mask
            ds[c]=kept
        for a,s,b,t in problem.edges:
            left=_side_set(problem,ds[a],s)
            right=_side_set(problem,ds[b],t)
            common={m for m in left if reverse_mask(m) in right}
            work.tick('arc_alternatives',len(left)+len(right))
            ds[a]=sum(left[m] for m in common)
            ds[b]=sum(right[reverse_mask(m)] for m in common)
        if not all(ds):return None
        if before==ds:return tuple(ds)


def _side_set(problem,mask,side):
    choices=defaultdict(int)
    for v in bits(mask):
        choices[int(problem.masks[v,side])]|=1<<v
    return choices


def _coarse_filter(problem,ds):
    keys={}
    inventory=Counter()
    for t in range(problem.n):
        p=tuple(int(x).bit_count() for x in problem.masks[3*t])
        orbit=min(p[r:]+p[:r] for r in range(3))
        inventory[orbit]+=1
        for r in range(3):
            keys[3*t+r]=orbit
    choices=[{keys[v] for v in bits(mask)} for mask in ds]
    for typ,needed in inventory.items():
        supplied=sum(typ in row for row in choices)
        forced=sum(row=={typ} for row in choices)
        if supplied<needed or forced>needed:return None
        if forced==needed or supplied==needed:
            ds=list(ds)
            for c,row in enumerate(choices):
                if typ not in row or row=={typ}:continue
                ds[c]=sum(1<<v for v in bits(ds[c]) if (keys[v]==typ)==(supplied==needed))
            ds=tuple(ds)
    return ds


def _branch(problem,ds,mode,seam_table=None):
    mrv=min((c for c,m in enumerate(ds) if m.bit_count()>1),key=lambda c:ds[c].bit_count())
    if mode in ('seam','seam-table'):
        choice=None
        for a,s,b,t in problem.edges:
            if seam_table is None:
                labels=_side_set(problem,ds[a],s)
            else:
                labels={m:mask&ds[a] for m,mask in seam_table[s].items() if mask&ds[a]}
            if len(labels)>1 and (choice is None or len(labels)<len(choice[1])):
                choice=(a,labels)
        if choice:
            c,labels=choice
            return c,tuple(labels.values()),'seam_decisions'
    if mode=='coarse':
        choice=None
        for c,mask in enumerate(ds):
            groups=defaultdict(int)
            for v in bits(mask):
                groups[tuple(int(x).bit_count() for x in problem.masks[v])]|=1<<v
            if len(groups)>1 and (choice is None or len(groups)<len(choice[1])):
                choice=(c,groups)
        if choice:
            c,groups=choice
            return c,tuple(groups.values()),'coarse_decisions'
    return mrv,tuple(1<<v for v in bits(ds[mrv])),'placement_decisions'


def solve_placements(problem,domains,mode,work,emit):
    # An exhausted coarse context contains every identity/port lift that was
    # still possible there. A later restriction of all its masks is also dead.
    dead=[]
    table=None
    if mode=='seam-table':
        full=(1<<(3*problem.n))-1
        table=[_side_set(problem,full,s) for s in range(3)]
        work.tick('compiled_table_entries',sum(map(len,table)))
    def search(ds):
        work.tick('search_nodes')
        if mode=='coarse' and any(all(not (x&~y) for x,y in zip(ds,old)) for old in dead):
            work.counters['lift_nogood_hits']+=1
            return False
        ds=_edge_filter(problem,ds,work)
        if ds is None:return
        if mode=='coarse':
            ds=_coarse_filter(problem,ds)
            if not ds or not all(ds):return
        if all(v.bit_count()==1 for v in ds):
            return emit(tuple(v.bit_length()-1 for v in ds))
        c,parts,kind=_branch(problem,ds,mode,table)
        work.tick(kind,len(parts))
        lifted=False
        for part in parts:
            success=search(ds[:c]+(part,)+ds[c+1:])
            lifted=bool(success) or lifted
        if not lifted and kind=='coarse_decisions' and len(dead)<128:
            dead.append(ds)
            work.counters['checked_lift_nogoods']+=1
        return lifted
    search(domains)


def colored_cover(problem,domains,work,emit):
    """Algorithm X with two primary items and three color-controlled seam items.

    A sparse row index picks the shortest remaining *cell or physical tile*
    column. Canonically directed seam colors purify selected-row neighbors.
    """
    rows=[]
    prim=[set() for _ in range(2*problem.n)]
    colors=defaultdict(lambda:defaultdict(set))
    contacts=[[] for _ in domains]
    for e,(a,s,b,t) in enumerate(problem.edges):
        contacts[a].append((e,s,False))
        contacts[b].append((e,t,True))
    for c,mask in enumerate(domains):
        for v in bits(mask):
            n=len(rows)
            seam=[]
            for e,s,inv in contacts[c]:
                m=int(problem.masks[v,s])
                m=reverse_mask(m) if inv else m
                seam.append((e,m))
                colors[e][m].add(n)
            rows.append((c,v,seam))
            prim[c].add(n)
            prim[problem.n+v//3].add(n)
    assignment=[-1]*problem.n
    def search(pending,live):
        work.tick('search_nodes')
        if not pending:
            emit(tuple(assignment))
            return
        column=min(pending,key=lambda p:len(prim[p]&live))
        for i in sorted(prim[column]&live):
            work.tick('cover_rows')
            c,v,seams=rows[i]
            t=problem.n+v//3
            changed=live-prim[c]-prim[t]
            for e,m in seams:
                for other,ids in colors[e].items():
                    if other!=m:
                        changed.difference_update(ids)
                        work.tick('purified_color_classes')
            assignment[c]=v
            if not problem.partial_loop_impossible(assignment):
                search(pending-{c,t},changed)
            assignment[c]=-1
    search(set(range(2*problem.n)),set(range(len(rows))))


def _embeddings(problem,domains,pair,work):
    """A connected, completely sewn oriented map is determined by one tile's image."""
    n=problem.n
    for root in range(n):
        for rotation in range(3):
            work.tick('embedding_seeds')
            if not domains[root]&(1<<rotation):continue
            cell_image={0:root}
            code_image={root:rotation}
            queue=[0]
            consistent=True
            while queue and consistent:
                tile=queue.pop()
                c=cell_image[tile]
                r=code_image[c]%3
                for s in range(3):
                    work.tick('embedding_checks')
                    d=3*tile+s
                    other,os=divmod(pair[d],3)
                    bs=(s+r)%3
                    adj=int(problem.neighbors[c,bs])
                    adj_side=int(problem.sides[c,bs])
                    value=3*other+(adj_side-os)%3
                    if not domains[adj]&(1<<value) or cell_image.get(other,adj)!=adj or code_image.get(adj,value)!=value:
                        consistent=False
                        break
                    if other not in cell_image:
                        cell_image[other]=adj
                        code_image[adj]=value
                        queue.append(other)
            if consistent and len(cell_image)==n:
                yield tuple(code_image[c] for c in range(n))


def _component_embeds(problem,domains,pair,start,work):
    """Every partially assembled connected component must fit somewhere."""
    for root,mask in enumerate(domains):
        for rot in range(3):
            if not mask&(1<<(3*start+rot)):continue
            work.tick('partial_embedding_seeds')
            placed={start:(root,rot)}
            occupied={root:start}
            todo=[start]
            valid=True
            while todo and valid:
                tile=todo.pop()
                cell,r=placed[tile]
                for side in range(3):
                    mate=pair[3*tile+side]
                    if mate<0:continue
                    work.tick('partial_embedding_links')
                    other,other_side=divmod(mate,3)
                    bs=(side+r)%3
                    adj=int(problem.neighbors[cell,bs])
                    orientation=(int(problem.sides[cell,bs])-other_side)%3
                    point=(adj,orientation)
                    if (not domains[adj]&(1<<(3*other+orientation))
                            or occupied.get(adj,other)!=other or placed.get(other,point)!=point):
                        valid=False
                        break
                    if other not in placed:
                        occupied[adj]=other
                        placed[other]=point
                        todo.append(other)
            if valid:return True
    return False


def _vertex_cycles(partners,capacity,work):
    """Partial face sewing fixes a partial permutation of oriented corners."""
    done=set()
    used=Counter()
    for d in range(len(partners)):
        if d in done:continue
        here=d
        path=set()
        while here not in path and here not in done and partners[here]>=0:
            work.tick('vertex_link_steps')
            path.add(here)
            mate=partners[here]
            here=3*(mate//3)+(mate%3+1)%3
        if here in path:
            used[len(path)]+=1
            if used[len(path)]>capacity[len(path)]:return False
        done.update(path)
    return True


def sewing(problem,domains,mode,work,emit):
    vlinks=_vertex_links(problem)
    if vlinks is None:
        raise ValueError('Physical assembly requires a connected oriented triangle sphere.')
    capacity=Counter(Counter(vlinks).values())
    graph=physical_graph(problem,domains,work)
    n=3*problem.n
    pair=[-1]*n
    ends={}
    i=0
    for t,tile in enumerate(problem.tiles):
        for path in tile['segments']:
            for s,k in path:ends[3*t+s,k]=i
            i+=1
    endpoint_by_strand=defaultdict(list)
    for pt,idx in ends.items():endpoint_by_strand[idx].append(pt)
    def strands():
        groups=Union(i)
        open_ends=Counter()
        dangles=[]
        for (d,k),strand in ends.items():
            work.tick('trail_steps')
            if pair[d]>=0:groups.join(strand,ends[pair[d],12-k])
            else:dangles.append((d,strand))
        for _,strand in dangles:open_ends[groups.find(strand)]+=1
        root=groups.find(0)
        comps={groups.find(q) for q in range(i)}
        closed=any(open_ends[g]==0 for g in comps) and len(comps)>1
        leaders={d for d,v in dangles if groups.find(v)==root}
        # Pin a physical direction through strand 0, follow its second endpoint
        # then the other endpoint of every joined physical strand.
        paths=list(endpoint_by_strand[0])
        point=paths[1]
        seen=set()
        while pair[point[0]]>=0 and point not in seen:
            seen.add(point)
            joined=(pair[point[0]],12-point[1])
            a,b=endpoint_by_strand[ends[joined]]
            point=b if a==joined else a
        return closed,({point[0]} if mode=='directed-tour' and pair[point[0]]<0 else leaders)
    def search():
        work.tick('search_nodes')
        if not _vertex_cycles(pair,capacity,work):return
        leaders=set()
        if mode in ('tour','directed-tour'):
            bad,leaders=strands()
            if bad:return
        free=[d for d in range(n) if pair[d]<0]
        if not free:
            for result in _embeddings(problem,domains,pair,work):emit(result)
            return
        allowed=lambda d:graph[d].intersection(free)
        branches=leaders or free
        a=min(branches,key=lambda d:(len(allowed(d)),d))
        for b in sorted(allowed(a)):
            work.tick('sewing_decisions')
            pair[a]=b
            pair[b]=a
            if mode!='embedded' or _component_embeds(problem,domains,pair,a//3,work):
                search()
            pair[a]=pair[b]=-1
    search()


def enumerate_boards(problem,mode='cell',*,domains=None,max_work=200000,check=None,retain=64):
    if mode not in ('cell','seam','seam-table','coarse','cover','glue','embedded','tour','directed-tour'):
        raise ValueError('Unknown structural search.')
    if type(retain)is not int or retain<0:raise ValueError('Invalid retention bound.')
    ds=domain_bits(problem) if domains is None else tuple(domains)
    base=domain_bits(problem)
    if len(ds)!=len(base) or any(type(v)is not int or v<0 or v&~b for v,b in zip(ds,base)):
        raise ValueError('Domains must restrict original choices.')
    work=Work(max_work,check)
    found=[]
    count=0
    def emit(values):
        nonlocal count
        work.tick('independent_checks')
        if not problem.validate(list(values))['valid']:return False
        count+=1
        if len(found)<retain:found.append(values)
        return True
    complete=True
    try:
        if mode in ('cell','seam','seam-table','coarse'):solve_placements(problem,ds,mode,work,emit)
        elif mode=='cover':colored_cover(problem,ds,work,emit)
        else:sewing(problem,ds,mode,work,emit)
    except WorkLimit:complete=False
    return dict(complete=complete,solutions=count,witnesses=found,work=dict(work.counters),units=work.used)

