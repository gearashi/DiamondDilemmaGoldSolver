"""Evaluate bundled transition switches starting at a disconnected physical sewing."""
from collections import defaultdict

from r27_structure_filters import Union, bits
from r28_geo_color import _sphere
from r28_geo_sewing import _cycles, _nullity, _bonds
from r30_gold import embedding
from validator import reverse_mask


def cover_switches(successor, bundles, max_nodes=100000):
    """Use disjoint virtual joining switches to construct an auxiliary Euler tour."""
    n=len(successor)
    _cycles(successor)
    switches=[tuple(p) for bundle in bundles for p in bundle]
    occupied={x for pair in switches for x in pair}
    if (any(len(p)!=2 or any(type(x)is not int or not 0<=x<n for x in p) for p in switches)
            or len(occupied)!=2*len(switches) or any(not b for b in bundles)):
        raise ValueError("Expected disjoint switch positions in nonempty whole-bond bundles.")
    uf=Union(n)
    for i,j in enumerate(successor):uf.join(i,j)
    # First join components using switches already in the family. For the
    # auxiliary Euler tour those toggles are mathematical reference choices.
    aux=list(successor)
    ref=0
    for j,(a,b) in enumerate(switches):
        if uf.find(a)!=uf.find(b):
            ref|=1<<j
            aux[a],aux[b]=aux[b],aux[a]
            uf.join(a,b)
    # Components still disconnected cannot be connected by any allowed switch.
    if len({uf.find(i) for i in range(n)})!=1:
        return dict(complete=True,patterns=[],nodes=1,reference_components=_cycles(successor))
    if _cycles(aux)!=1:raise ArithmeticError("Auxiliary joining switches failed.")
    order={}
    v=0
    for i in range(n):
        order[v]=i
        v=aux[v]
    chords=[sorted((order[a],order[b])) for a,b in switches]
    adj=[sum(1<<j for j,(c,d) in enumerate(chords) if a<c<b<d or c<a<d<b) for a,b in chords]
    masks=[]
    index=0
    for bundle in bundles:
        masks.append(((1<<len(bundle))-1)<<index)
        index+=len(bundle)
    patterns=[]
    nodes=0
    def walk(k,active,selection):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise OverflowError
        free=sum(masks[k:])
        fixed=active & ~free
        defect=_nullity([adj[i]&fixed for i in bits(fixed)])
        if defect>free.bit_count():return
        if k==len(masks):
            patterns.append(selection)
            return
        walk(k+1,active,selection)
        walk(k+1,active^masks[k],selection|1<<k)
    try:
        walk(0,ref,0)
        complete=True
    except OverflowError:
        complete=False
    return dict(complete=complete,patterns=patterns,nodes=nodes,reference_components=_cycles(successor))


def sewn_switches(problem, domains, bonds, phases, trades, max_nodes=100000):
    """Two physical bonds in a trade exchange partners and switch every occupied slot."""
    _,regs=_sphere(problem)
    starts={}
    offset=0
    for t,tile in enumerate(problem.tiles):
        for i,seg in enumerate(tile['segments']):
            for s,k in seg:
                starts[3*t+s,k]=(offset+i,(regs[t][12*s+k].bit_count()&1)^phases[t])
        offset+=len(tile['segments'])
    def successor(mate):
        out=[-1]*offset
        for (a,k),(i,d) in starts.items():
            if not d:continue
            j,f=starts[mate[a],12-k]
            if f:raise ValueError("Inconsistent directed bond.")
            out[i]=j
        _cycles(out)
        return out
    base=_bonds(problem,bonds)
    if len(base)!=3*problem.n:raise ValueError("Expected a complete physical matching.")
    def validate_mates(m):
        for a,b in m.items():
            ta,sa=divmod(a,3)
            tb,sb=divmod(b,3)
            ma=int(problem.masks[3*ta,sa])
            if ta==tb or reverse_mask(ma)!=int(problem.masks[3*tb,sb]):
                raise ValueError("Trade introduced an invalid physical bond.")
            ca=(regs[ta][12*sa].bit_count()&1)^phases[ta]
            cb=(regs[tb][12*sb].bit_count()&1)^phases[tb]
            if ca^cb!=ma.bit_count()%2:raise ValueError("Invalid signed side pairing.")
    validate_mates(base)
    s=successor(base)
    inv={v:i for i,v in enumerate(s)}
    bundles=[]
    seen=set()
    changes=[]
    for a,b in trades:
        c,d=base[a],base[b]
        if len({a,b,c,d})!=4 or {a,b,c,d}&seen:raise ValueError("Overlapping side trades.")
        seen.update((a,b,c,d))
        change={a:d,d:a,b:c,c:b}
        other=base|change
        validate_mates(other)
        new=successor(other)
        permutation=[inv[j] for j in new]
        if any(permutation[j]!=i for i,j in enumerate(permutation)):
            raise ValueError("Not a disjoint successor switch.")
        bundle=[(i,j) for i,j in enumerate(permutation) if i<j]
        if not bundle:raise ValueError("Empty seam trade.")
        bundles.append(bundle)
        changes.append(change)
    result=cover_switches(s,bundles,max_nodes)
    witnesses=[]
    for pattern in result['patterns']:
        mates=base.copy()
        for j in bits(pattern):mates.update(changes[j])
        codes=embedding(problem,domains,[mates[i] for i in range(3*problem.n)])
        if codes is not None:
            if not problem.validate(codes)['valid']:raise ArithmeticError("Invalid recovered Gold board.")
            witnesses.append(codes)
    return dict(result,witnesses=witnesses)
