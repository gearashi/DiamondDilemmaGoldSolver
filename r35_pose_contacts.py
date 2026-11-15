"""Project boundary seam compatibility between different labelled rigid component poses."""
from collections import Counter,defaultdict
from functools import lru_cache

from r27_structure_filters import WorkLimit,bits
from r32_gold_bonds import inventory
from validator import reverse_mask



class PhaseRelations:
    """Necessary relative tile phases from specified physical mates and explicitly pinned signs."""
    def __init__(self,problem,bonds=(),pins=None):
        from r29_logic_binary import Parity
        from r29_gold_phase import checked_pins
        from r35_signed_mate_support import reciprocal_classes
        self.sides,_=reciprocal_classes(problem.tiles)
        union=Parity(problem.n+1)
        self.valid=True
        for t,phase in checked_pins(problem.n,pins or {}).items():
            self.valid &= union.bind(t,problem.n,phase)
        for a,b in bonds:
            self.valid &= union.bind(a//3,b//3,self.required(a,b))
        self.parts=[union.find(i) for i in range(problem.n)]

    def required(self,a,b):
        slots,phase=self.sides[a]
        return phase^self.sides[b][1]^(len(slots)%2)

    def accepts(self,a,b):
        (root,left),(other,right)=self.parts[a//3],self.parts[b//3]
        return root!=other or left^right==self.required(a,b)


class Boundaries:
    """For a specified pose, price all its internal physical contacts from actual geographic neighbors."""
    def __init__(self,problem,physical=True,phases=None):
        self.neighbors=[tuple((int(b),int(j)) for b,j in zip(row,sides))
                        for row,sides in zip(problem.neighbors,problem.sides)]
        self.phases=phases
        self.mates=None
        if physical:
            mates=[set() for _ in range(3*problem.n)]
            for a,b in inventory(problem).possible:
                mates[a].add(b);mates[b].add(a)
            self.mates=mates
        self.sides=[int(x) for row in problem.masks[::3] for x in row]

    def accepts(self,a,b):
        if self.phases is not None and not self.phases.accepts(a,b):return False
        if self.mates is not None:return b in self.mates[a]
        return self.sides[b]==reverse_mask(self.sides[a])

    def describe(self,pose):
        cells=dict(pose);outside=[];faces=[];failed=0
        for cell,code in pose:
            tile,turn=divmod(code,3)
            for side,(neighbor,facing) in enumerate(self.neighbors[cell]):
                dart=3*tile+(side-turn)%3
                faces.append((cell,side,dart))
                if neighbor<0:continue
                if neighbor not in cells:
                    outside.append((neighbor,facing,dart))
                elif cell<neighbor:
                    other,rot=divmod(cells[neighbor],3)
                    failed+=not self.accepts(dart,3*other+(facing-rot)%3)
        return tuple(cells),outside,faces,failed


class PoseIndex:
    def __init__(self,signatures,geometry):
        self.occupied=defaultdict(int)
        self.labels=defaultdict(dict)
        for i,(cells,outside,faces,failed) in enumerate(signatures):
            bit=1<<i
            for cell in cells:self.occupied[cell]|=bit
            for cell,side,dart in faces:
                labels=self.labels[cell,side]
                labels[dart]=labels.get(dart,0)|bit
        @lru_cache(maxsize=16384)
        def bad(cell,side,dart):
            result=0
            for other,mask in self.labels.get((cell,side),{}).items():
                if not geometry.accepts(dart,other):result|=mask
            return result
        self.bad=bad

    def allowed(self,signature,visible):
        cells,boundary,_,_=signature
        for cell in cells:visible&=~self.occupied.get(cell,0)
        for dest,side,dart in boundary:visible&=~self.bad(dest,side,dart)
        return visible


def compatible_poses(problem,groups,work,*,physical=True,joint=True,restricted=True,cover_test=True,phases=None):
    """Exclude only independently impossible labelled poses, preserving undecided alternatives at a work limit."""
    poses=[tuple(p for rows in family.values() for p in rows) for family in groups]
    stats=Counter()
    if any(not family for family in poses):return groups,stats,True
    if not poses or restricted and min(map(len,poses))>64:
        stats['broad_contact_pass_skipped']=1
        return groups,stats,True
    geometry=Boundaries(problem,physical,phases)
    descriptions=[[geometry.describe(p) for p in family] for family in poses]
    indices=[PoseIndex(sig,geometry) for sig in descriptions]
    # If the target family has more surviving labels than can touch the entire
    # neighboring area of a pose, at least one placement is disjoint and separate.
    areas=[max((len(set(cells)|{b for b,_,_ in boundary}) for cells,boundary,_,_ in family),default=0)
           for family in descriptions]
    occupancy=[max((row.bit_count() for row in index.occupied.values()),default=0) for index in indices]
    visible=[]
    for family in descriptions:
        mask=sum(1<<i for i,sig in enumerate(family) if sig[3]==0)
        stats['internally_incompatible']+=len(family)-mask.bit_count()
        visible.append(mask)
    @lru_cache(maxsize=32768)
    def links(group,index,other):
        return indices[other].allowed(descriptions[group][index],(1<<len(poses[other]))-1)
    complete=True
    try:
        while True:
            changed=False
            for g,mask in enumerate(visible):
                possible=[other for other,support in enumerate(visible) if other!=g and
                          (not cover_test or support.bit_count()<=areas[g]*occupancy[other])]
                stats['separate_area_supports']+=(len(visible)-1-len(possible))*mask.bit_count()
                for i in bits(mask):
                    for other in possible:
                        support=visible[other]
                        work.tick('contact_family_queries')
                        if not links(g,i,other)&support:
                            visible[g]&=~(1<<i);stats['pairwise_unsupported']+=1
                            changed=True;break
            if not changed or not all(visible):break
        # Pairwise consistency can leave cyclic obligations. Check the simultaneous
        # projection of every small family while retaining all other groups as a relaxation.
        small=tuple(g for g,row in enumerate(visible) if 0<row.bit_count()<=64)
        if joint and 3<=len(small)<=8 and all(visible):
            seen=[0]*len(groups)
            def pack(candidates,picked):
                work.tick('contact_joint_nodes')
                if not candidates:
                    for g,i in picked:seen[g]|=1<<i
                    return True
                g=min(candidates,key=lambda k:candidates[k].bit_count())
                rest={k:v for k,v in candidates.items() if k!=g}
                for i in bits(candidates[g]):
                    child={k:v&links(g,i,k) for k,v in rest.items()}
                    if all(child.values()) and pack(child,[*picked,(g,i)]):return True
                return False
            for g in small:
                for i in bits(visible[g]&~seen[g]):
                    if seen[g]>>i&1:continue
                    child={k:visible[k]&links(g,i,k) for k in small if k!=g}
                    if not all(child.values()) or not pack(child,[(g,i)]):
                        visible[g]&=~(1<<i);stats['joint_contact_unsupported']+=1
    except WorkLimit:
        stats['contact_budget_exhausted']+=1;complete=False
    if not any(stats[k] for k in ('internally_incompatible','pairwise_unsupported','joint_contact_unsupported')):
        return groups,stats,complete
    result=[]
    for g,rows in enumerate(poses):
        choices=defaultdict(list)
        for i in bits(visible[g]):
            p=rows[i]
            choices[sum(1<<c for c,v in p)].append(p)
        result.append(choices)
    return result,stats,complete

def conditioned_components(problem,domains,clusters,bonds,phases=None):
    """Join explicit geographic bond assumptions, keeping the other required component poses."""
    from types import SimpleNamespace
    from r35_speculative_poses import PoseJoins
    class Legal:
        def __getitem__(self,key):
            c,v=key
            return bool(domains[c]>>v&1)
    model=SimpleNamespace(problem=problem,n=problem.n,legal=Legal())
    initial=[(members,[p for p in poses if all(domains[c]>>v&1 for c,v in p)]) for members,_,poses in clusters]
    joins=PoseJoins(SimpleNamespace(model=model,groups=initial),gold=True,phases=phases)
    groups=list(joins.base)
    for a,b in sorted({tuple(sorted(e)) for e in bonds}):
        if a//3==b//3:continue
        i,j=(next((i for i,g in enumerate(groups) if t in g[0]),None) for t in (a//3,b//3))
        if i is not None and i==j:
            group=groups[i]
            rows=[]
            for pose in group[1]:
                pos={code//3:(c,code%3) for c,code in pose}
                c,r=pos[a//3];d,q=pos[b//3]
                if tuple(joins.neighbors[c][(a%3+r)%3])==(d,(b%3+q)%3):
                    rows.append(pose)
            groups[i]=(group[0],rows,tuple(sorted((*group[2],(a,b)))))
            joins.work['cycle_pose_tests']+=len(group[1])
            if not rows:break
            continue
        left=(frozenset((a//3,)),None,()) if i is None else groups[i]
        right=(frozenset((b//3,)),None,()) if j is None else groups[j]
        if len(left[0])+len(right[0])>9:
            joins.work['large_named_join_deferred']+=1
            continue
        merged=joins.joined(left,right,(a,b))
        groups=[g for k,g in enumerate(groups) if k not in (i,j)]+[merged]
        if not merged[1]:break
    result=[(tuple(sorted(tiles)),sum(7<<(3*t) for t in tiles),rows) for tiles,rows,_ in groups]
    return result,dict(joins.work)
