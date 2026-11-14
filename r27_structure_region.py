"""Exact two-region enumeration with physical inventory, strand incidence and multiplicity."""
from collections import defaultdict

from r27_structure_filters import Union, Work, WorkLimit, _vertex_links, bits, domain_bits, oriented_segments
from validator import reverse_mask


def _boundary(problem,inside):
    corners=_vertex_links(problem)
    if corners is None:return None
    steps={}
    for e,(a,s,b,t) in enumerate(problem.edges):
        if (a in inside)==(b in inside):continue
        c,j,rev=(a,s,False) if a in inside else (b,t,True)
        start=corners[3*c+j]
        if start in steps:return None
        steps[start]=(corners[3*c+(j+1)%3],e,rev)
    if not steps:return None
    order=[]
    x=min(steps)
    visited=set()
    while x not in visited:
        if x not in steps:return None
        visited.add(x)
        x,e,rev=steps[x]
        slots=range(11,0,-1) if rev else range(1,12)
        order.extend((e,k) for k in slots)
    if len(visited)!=len(steps) or x!=min(steps):return None
    return {slot:i for i,slot in enumerate(order)}


def _pack(pairs):
    """A fixed boundary endpoint list plus a parenthesis word uniquely decodes a noncrossing match."""
    mate=dict(pairs)
    mate.update({b:a for a,b in pairs})
    keys=sorted(mate)
    stack=[]
    word=0
    for i,a in enumerate(keys):
        b=mate[a]
        if a<b:
            stack.append(a)
            word|=1<<i
        elif not stack or stack.pop()!=b:return None
    return word


def _unpack(keys,word):
    stack=[]
    out=[]
    for i,x in enumerate(keys):
        if word&(1<<i):stack.append(x)
        else:out.append((stack.pop(),x))
    if stack:raise ValueError('Unbalanced boundary word.')
    return tuple(out)


def _states(problem,ds,cells,rank,compact,work):
    inside=set(cells)
    near=defaultdict(list)
    for e,(a,s,b,t) in enumerate(problem.edges):
        if a in inside:near[a].append((s,b,t,e,False))
        if b in inside:near[b].append((t,a,s,e,True))
    order=sorted(cells,key=lambda c:(ds[c].bit_count(),c))
    assignment={}
    groups={}
    def finish(used):
        lookup={}
        count=0
        for c,v in assignment.items():
            for seg in oriented_segments(problem,v):
                for s,k in seg:lookup[c,s,k]=count
                count+=1
        uf=Union(count)
        external={}
        work.tick('regional_strands',count)
        for c in assignment:
            for s,b,t,e,rev in near[c]:
                for k in bits(int(problem.masks[assignment[c],s])):
                    u=lookup[c,s,k+1]
                    if b in inside:
                        if c<b:uf.join(u,lookup[b,t,11-k])
                    else:external[rank[e,11-k if rev else k+1]]=u
        parts=defaultdict(list)
        for k,u in external.items():parts[uf.find(u)].append(k)
        if any(uf.find(i) not in parts for i in range(count)):return
        if any(len(x)!=2 for x in parts.values()):raise ValueError('Invalid regional endpoint degree.')
        pairs=tuple(sorted(tuple(sorted(p)) for p in parts.values()))
        active=tuple(sorted(external))
        word=_pack(pairs) if compact else None
        key=(used,active,pairs if word is None else word)
        if key in groups:groups[key][0]+=1
        else:
            from sys import getsizeof
            payload=key[2]
            work.counters['boundary_pattern_object_bytes']+=(getsizeof(payload) if type(payload)is int else
                                                        getsizeof(payload)+sum(getsizeof(p) for p in payload))
            groups[key]=[1,tuple((c,assignment[c]) for c in order)]
            work.counters['boundary_key_bytes']+=((len(active)+7)//8 if word is not None else
                 2*len(pairs)*max(1,(max(active,default=0).bit_length()+7)//8))
        work.tick('region_histories')
    def visit(i,used):
        work.tick('region_nodes')
        if i==len(order):
            finish(used)
            return
        c=order[i]
        for v in bits(ds[c]):
            bit=1<<(v//3)
            if used&bit:continue
            if any(b in assignment and reverse_mask(int(problem.masks[v,s]))!=int(problem.masks[assignment[b],t])
                   for s,b,t,e,inverse in near[c]):continue
            assignment[c]=v
            visit(i+1,used|bit)
            del assignment[c]
    visit(0,0)
    return groups


def _determinant(matrix,work):
    """Fraction-free Bareiss elimination with singular and pivot-swap handling."""
    n=len(matrix)
    if not n:return 1
    a=[r.copy() for r in matrix]
    previous=1
    sign=1
    for k in range(n-1):
        row=next((i for i in range(k,n) if a[i][k]),None)
        if row is None:return 0
        if row!=k:a[k],a[row]=a[row],a[k];sign=-sign
        pivot=a[k][k]
        work.tick('determinant_updates',(n-k-1)**2)
        for i in range(k+1,n):
            for j in range(k+1,n):
                a[i][j]=(a[i][j]*pivot-a[i][k]*a[k][j])//previous
        previous=pivot
    return sign*a[-1][-1]


def regional_count(problem,inside,domains=None,*,compact=True,determinant=False,max_work=1000000,check=None):
    ds=domain_bits(problem) if domains is None else tuple(domains)
    base=domain_bits(problem)
    if len(ds)!=len(base) or any(type(v)is not int or v<0 or v&~b for v,b in zip(ds,base)):
        raise ValueError('Domains must restrict original choices.')
    inside=frozenset(inside)
    outside=set(range(problem.n))-inside
    if inside-set(range(problem.n)) or not inside or not outside:raise ValueError('Choose a nonempty proper region.')
    rank=_boundary(problem,inside)
    if rank is None:raise ValueError('Cut must have one simple oriented boundary on the sphere.')
    w=Work(max_work,check)
    total=0
    witness=None
    complete=False
    try:
        left=_states(problem,ds,inside,rank,compact,w)
        right=_states(problem,ds,outside,rank,compact,w)
        by_inventory=defaultdict(list)
        for (used,keys,x),data in right.items():by_inventory[used,keys].append((x,data))
        all_tiles=(1<<problem.n)-1
        w.counters['regional_states']=len(left)+len(right)
        for (used,keys,pattern),(ways,example) in left.items():
            p1=_unpack(keys,pattern) if type(pattern)is int else pattern
            for pat,(ways2,example2) in by_inventory.get((all_tiles^used,keys),()):
                w.tick('region_joins')
                p2=_unpack(keys,pat) if type(pat)is int else pat
                ids={v:i for i,v in enumerate(keys)}
                n=len(keys)
                if not n:continue
                if determinant:
                    # Each internal path was contracted exactly once. A single cycle on
                    # n cut endpoints has exactly n trees, including two parallel edges.
                    w.tick('matrix_entries',n*n)
                    lap=[[0]*n for _ in range(n)]
                    for a,b in (*p1,*p2):
                        i,j=ids[a],ids[b]
                        lap[i][i]+=1;lap[j][j]+=1;lap[i][j]-=1;lap[j][i]-=1
                    trees=_determinant([r[:-1] for r in lap[:-1]],w)
                    if trees not in (0,n):raise ArithmeticError('Invalid cycle-cover cofactor.')
                    passed=bool(trees)
                else:
                    uf=Union(n)
                    for a,b in (*p1,*p2):
                        uf.join(ids[a],ids[b])
                        w.tick('boundary_unions')
                    passed=len({uf.find(i) for i in range(n)})==1
                if passed:
                    total+=ways*ways2
                    if witness is None:
                        vals=dict((*example,*example2))
                        witness=tuple(vals[c] for c in range(problem.n))
                        if not problem.validate(list(witness))['valid']:raise ArithmeticError('Invalid regional witness.')
        complete=True
    except WorkLimit:pass
    return dict(complete=complete,solutions=total,witness=witness,work=dict(w.counters),units=w.used)

