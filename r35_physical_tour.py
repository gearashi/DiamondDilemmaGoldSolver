"""A bundled directed circuit on mandatory physical strands or their fixed path components."""
from collections import Counter,defaultdict

from r27_structure_filters import Union


def tour_structure(check,required,contract):
    """Each directed physical slot is controlled by its one seam and its tile's phase."""
    n=check.strands
    uf=Union(n)
    degrees=[0]*n
    compulsory=set()
    if contract:
        counts=Counter(d for pair in check.mate for d in pair)
        compulsory=set(required)&check.mate.keys()
        compulsory.update(pair for pair in check.mate if any(counts[d]==1 for d in pair))
        darts=[d for pair in compulsory for d in pair]
        if len(darts)!=len(set(darts)):
            return 0,{},set(),True
        for pair in compulsory:
            for left,right in check.transitions[pair]:
                uf.join(left,right)
                degrees[left]+=1;degrees[right]+=1
    names={}
    ids=[names.setdefault(uf.find(i),len(names)) for i in range(n)]
    opened=[0]*len(names)
    for i,degree in enumerate(degrees):
        opened[ids[i]]+=2-degree
    if len(opened)>1 and 0 in opened:
        return len(opened),{},set(),True
    if len(names)<=1:
        return len(names),{},set(),False
    fixed_darts={d for pair in compulsory for d in pair}
    blocked=set()
    arcs=defaultdict(set)
    for (a,b),links in check.transitions.items():
        if (a,b) in compulsory:
            continue
        if a in fixed_darts or b in fixed_darts or any(ids[u]==ids[v] for u,v in links):
            blocked.add((a,b))
            continue
        _,color=check.inputs.sides[a]
        # The signed model equates the colors of boundary intervals across each seam.
        # Crossing each occupied slot toggles color; the two matched endpoints oppose.
        for i,(u,v) in enumerate(links):
            outgoing=color^((i+1)%2)
            u,v=ids[u],ids[v]
            arcs[u,v].add(((a,b),1^outgoing))
            arcs[v,u].add(((a,b),outgoing))
    return len(names),arcs,blocked,False


def post_circuit(check,required,mode):
    n,arcs,blocked,impossible=tour_structure(check,required,mode!='raw')
    counts=dict(vertices=n,arcs=len(arcs),support_terms=sum(map(len,arcs.values())),
                rejected_internal_or_owned=len(blocked),active=True)
    if mode=='sparse' and len(arcs)>256:
        counts['active']=False
        return counts
    model=check.m
    if impossible:
        model.add_bool_or([])
        return counts
    for pair in blocked:
        model.add(check.mate[pair]==0)
    if n<=1:
        return counts
    if {a for a,_ in arcs}!=set(range(n)) or {b for _,b in arcs}!=set(range(n)):
        model.add_bool_or([])
        return counts
    selected={}
    entries=[]
    for (u,v),supports in sorted(arcs.items()):
        literals=[]
        for pair,value in sorted(supports):
            if (pair,value) not in selected:
                flag=model.new_bool_var('directed_mate')
                on=check.mate[pair]
                phase=check.phase[pair[0]//3]
                direction=phase if value else ~phase
                model.add_implication(flag,on)
                model.add_implication(flag,direction)
                model.add_bool_or([~on,~direction,flag])
                selected[pair,value]=flag
            literals.append(selected[pair,value])
        if len(literals)==1:
            arc=literals[0]
        else:
            arc=model.new_bool_var('strand_successor')
            model.add_max_equality(arc,literals)
        entries.append((u,v,arc))
    model.add_circuit(entries)
    counts['or_groups']=sum(len(s)>1 for s in arcs.values())
    return counts
