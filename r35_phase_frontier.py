"""Exact labelled phase projections with grouped internal increments and live balance classes."""
from collections import Counter
from math import comb,gcd

from r35_phase_lattice import PhaseColumns


def compile_projection(interface,pins,boundary=(),*,relations=(),euler=True,max_states=10000,
                       max_transitions=10000000,multiplicities=False,split=False):
    if max_states<1 or max_transitions<1:raise ValueError('Positive compilation bounds required.')
    boundary=tuple(boundary)
    cols=PhaseColumns(interface,relations,pins,euler)
    if len(set(boundary))!=len(boundary) or any(t not in cols.owner for t in boundary):
        raise ValueError('Invalid labelled boundary.')
    if not cols.consistent:return dict(status='COMPLETE',patterns=[],counts={},transitions=0)
    base=cols.base.copy();mod=cols.mod
    tags=Counter();origin=0
    for i,tile in enumerate(boundary):
        k,z=cols.owner[tile]
        origin|=z<<i
        if k is not None:tags[k]|=1<<i
    merged=Counter();groups=[]
    for k,raw in enumerate(cols.columns):
        delta=tuple(v%2 if m else v for v,m in zip(raw,mod))
        if k in tags:groups.append((delta,1,tags[k]));continue
        reverse=tuple((-v)%2 if m else -v for v,m in zip(raw,mod))
        if reverse<delta:
            for j,v in enumerate(delta):base[j]+=v
            delta=reverse
        merged[delta]+=1
    factor=1
    for delta,count in merged.items():
        if any(delta):groups.append((delta,count,0))
        else:factor*=1<<count
    support={i:{j for j,v in enumerate(delta) if v} for i,(delta,_,_) in enumerate(groups)}
    left=Counter(j for indexes in support.values() for j in indexes)
    active=set();order=[]
    while support:
        def cost(i):
            new=(active|support[i])-{j for j in support[i] if left[j]==1}
            return len(new),-sum(left[j]==1 for j in support[i]),groups[i][1],i
        i=min(support,key=cost)
        touched=support.pop(i)
        active.update(touched)
        for j in touched:
            left[j]-=1
            if left[j]==0:active.remove(j)
        order.append(groups[i])
    dimension=len(base)
    goals=[(0,) for _ in base]
    if euler:goals[-1]=tuple(cols.euler)
    def spans(sequence, other=()):
        span=[(0,0,0)]*dimension
        suffix=[None]*(len(sequence)+1)
        def add(group):
            delta,number,_=group
            for j,v in enumerate(delta):
                if v:
                    low,high,divisor=span[j]
                    span[j]=low+min(0,v*number),high+max(0,v*number),gcd(divisor,abs(v))
        for group in other:add(group)
        suffix[-1]=tuple(span)
        for i in reversed(range(len(sequence))):
            add(sequence[i]);suffix[i]=tuple(span)
        return suffix

    def possible(j,v,room):
        low,high,divisor=room
        if mod[j]:return bool(divisor%2) or v%2==0
        return any(low<=target-v<=high and (not divisor or (target-v)%divisor==0) for target in goals[j])

    summary=dict(groups=len(groups),free_components=len(cols.columns),max_live_coordinates=0,peak_states=1,
                 transitions=0,interchangeable_internal=sum(n-1 for _,n,mask in groups if not mask))
    whole=spans(order)[0]
    if any(not possible(j,v,whole[j]) for j,v in enumerate(base)):
        return dict(status='COMPLETE',patterns=[],counts={},**summary)

    def run_side(sequence,other,shared,initial_tag,initial_count,stored=0):
        suffix=spans(sequence,other)
        first=[len(sequence)]*dimension;last=[-1]*dimension
        for i,(delta,_,_) in enumerate(sequence):
            for j,v in enumerate(delta):
                if v:
                    first[j]=min(first[j],i)
                    last[j]=len(sequence) if j in shared else i
        states={((),initial_tag):initial_count}; before=()
        for i,(delta,number,btag) in enumerate(sequence):
            moving=tuple(j for j in range(dimension) if first[j]<=i<=last[j])
            keep=tuple(j for j in moving if last[j]>i)
            summary['max_live_coordinates']=max(summary['max_live_coordinates'],len(keep))
            updated={}
            for (previous,pattern),weight in states.items():
                values=dict(zip(before,previous))
                for k in range(number+1):
                    summary['transitions']+=1
                    if summary['transitions']>max_transitions:return None
                    mark=pattern^(btag if k else 0)
                    row=[]
                    for j in moving:
                        v=values.get(j,base[j])+k*delta[j]
                        if mod[j]:v%=2
                        if not possible(j,v,suffix[i+1][j]):break
                        if last[j]>i:row.append(v)
                    else:
                        key=tuple(row),mark
                        if multiplicities:updated[key]=updated.get(key,0)+weight*comb(number,k)
                        else:updated[key]=1
                        if len(updated)+stored>max_states:return None
            states=updated;before=keep
            summary['peak_states']=max(summary['peak_states'],len(states)+stored)
            if not states:break
        return states

    if split is None:split=len(order)>=20
    if not split or len(order)<2:
        states=run_side(order,(),(),origin,factor if multiplicities else 1)
        if states is None:return dict(status='UNKNOWN',**summary)
        counts={mark:v for (_,mark),v in states.items()}
    else:
        # Across balanced splits of the planned elimination order, retain only
        # coordinates touched by both halves. Private constraints retire locally.
        supports=[{j for j,v in enumerate(g[0]) if v} for g in order]
        def interface_at(i):
            return set.union(set(),*supports[:i]) & set.union(set(),*supports[i:])
        middle=min(range(max(1,len(order)//3),min(len(order),2*len(order)//3+1)),
                   key=lambda i:(len(interface_at(i)),abs(len(order)-2*i)))
        shared=tuple(sorted(interface_at(middle)))
        left,right=order[:middle],list(reversed(order[middle:]))
        summary.update(split_groups=[len(left),len(right)],shared_coordinates=len(shared))
        a=run_side(left,right,shared,origin,factor if multiplicities else 1)
        if a is None:return dict(status='UNKNOWN',**summary)
        counts={}
        if a:
            b=run_side(right,left,shared,0,1,stored=len(a))
            if b is None:return dict(status='UNKNOWN',**summary)
            from collections import defaultdict
            index=defaultdict(list)
            for (row,tag),weight in a.items():index[row].append((tag,weight))
            target_values=tuple(cols.euler) if euler and dimension-1 in shared else (0,)
            for (row,tag),weight in b.items():
                for final in target_values:
                    key=[]
                    for v,j in zip(row,shared):
                        wanted=(final if euler and j==dimension-1 else 0)+base[j]-v
                        key.append(wanted%2 if mod[j] else wanted)
                    for other_tag,other_weight in index.get(tuple(key),()):
                        summary['transitions']+=1
                        if summary['transitions']>max_transitions:return dict(status='UNKNOWN',**summary)
                        mark=tag^other_tag
                        if multiplicities:counts[mark]=counts.get(mark,0)+weight*other_weight
                        else:counts[mark]=1
                        if len(counts)>max_states:return dict(status='UNKNOWN',**summary)
    return dict(status='COMPLETE',patterns=sorted(counts),counts=counts if multiplicities else None,**summary)
