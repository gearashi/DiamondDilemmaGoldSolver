"""Necessary signed physical partner support without allocating a whole sewing model."""
from collections import defaultdict
from functools import lru_cache
from itertools import product
from time import monotonic

from r35_loss_witness import raw_sides,reciprocal


def reciprocal_classes(tiles):
    sides=raw_sides(tiles);groups=defaultdict(list)
    for i,(slots,color) in enumerate(sides):
        groups[min(slots,reciprocal((slots,color))[0])].append(i)
    return sides,[tuple(v) for _,v in sorted(groups.items())]


def raw_edges(sides,group,allowed=None):
    edges=[]
    for i,a in enumerate(group):
        for b in group[i+1:]:
            if a//3==b//3 or sides[b][0]!=reciprocal(sides[a])[0]:continue
            if allowed is not None and (a,b) not in allowed:continue
            edges.append((a,b,sides[a][1]^sides[b][1]^(len(sides[a][0])%2)))
    return edges


def matching(group,edges,pins):
    loc={a:i for i,a in enumerate(group)};rows=[0]*len(group)
    for a,b,rhs in edges:
        x,y=pins.get(a//3),pins.get(b//3)
        if x is not None and y is not None and x^y!=rhs:continue
        i,j=loc[a],loc[b];rows[i]|=1<<j;rows[j]|=1<<i
    @lru_cache(None)
    def solve(mask):
        if not mask:return True
        least=None;left=mask
        while left:
            first=left&-left;left^=first;partners=rows[first.bit_length()-1]&mask
            if least is None or partners.bit_count()<least.bit_count():selected=first;least=partners
            if not least:break
        while least:
            b=least&-least;least^=b
            if solve(mask^selected^b):return True
        return False
    return solve((1<<len(group))-1)


def zero_supports(edges,group):
    by_side=defaultdict(list)
    for a,b,r in edges:by_side[a].append((b,r));by_side[b].append((a,r))
    found=set()
    for a in group:
        for value in (0,1):
            blocked={a//3:value}
            for b,rhs in by_side[a]:
                need=(value^rhs)^1;t=b//3
                if t in blocked and blocked[t]!=need:break
                blocked[t]=need
            else:found.add(tuple(sorted(blocked.items())))
    return found


def hall_clauses(tiles,*,allowed=None,max_atoms=12,max_sides=16):
    sides,classes=reciprocal_classes(tiles);single=set();strong=set();work=defaultdict(int)
    def add_minimal(found,rule):
        if any(set(e)<=set(rule) for e in found):return
        found.difference_update([e for e in found if set(rule)<set(e)])
        found.add(rule)
    for group in classes:
        edges=raw_edges(sides,group,allowed);atoms=sorted({a//3 for a in group})
        roots=zero_supports(edges,group);single.update(roots)
        if len(atoms)>max_atoms or len(group)>max_sides:continue
        work['examined_classes']+=1
        local=set()
        for i in range(1<<(len(atoms)-1)):
            pins={t:((i<<1)>>k)&1 for k,t in enumerate(atoms)}
            if any(all(pins[t]==v for t,v in rule) for rule in roots|local):continue
            # The full signed inventory already rejects unequal class counts.
            tally=defaultdict(int)
            for a in group:
                slots,c=sides[a];tally[slots,c^pins[a//3]]+=1
            if any((n%2 if reciprocal(k)==k else n!=tally.get(reciprocal(k),0)) for k,n in tally.items()):continue
            work['balanced_assignments']+=1
            if matching(group,edges,pins):continue
            work['matching_failures']+=1
            for t in atoms:
                value=pins.pop(t)
                if matching(group,edges,pins):pins[t]=value
            pair=tuple(sorted(pins.items()));rev=tuple((t,1-v) for t,v in pair)
            add_minimal(local,pair);add_minimal(local,rev)
        strong.update(local)
    def clauses(rows):
        return sorted(set(tuple(-t-1 if v else t+1 for t,v in r) for r in rows),key=lambda r:(len(r),r))
    return clauses(single),clauses(strong),dict(work)


def rejects(clauses,phase):
    return [row for row in clauses if all(((phase>>(abs(v)-1))&1)!=(v>0) for v in row)]
