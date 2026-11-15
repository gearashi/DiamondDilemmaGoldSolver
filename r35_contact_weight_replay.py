"""Encode a checked weighted-cover margin with an independent binary tree of attainable sums."""

def budget_clauses(ids, loss, terms, maximum, cap):
    available=cap*maximum-sum(w for _,w in terms)
    if available<0:
        return [[]]
    contributions=[]
    clauses=[]
    for e,v in sorted(loss.items()):
        deficit=maximum-sum(w for g,w in terms if e in g)
        if deficit<0:
            raise ValueError("incident weight exceeds certified maximum")
        if deficit:
            contributions.append((deficit,v))
    for group,w in terms:
        named=[loss[e] for e in sorted(group)]
        previous=named[0]
        for j,v in enumerate(named[1:],start=1):
            # Every hit after the first costs the term weight once. The two
            # forward implications suffice: witnesses can set the auxiliary
            # bits to precisely their intended OR and AND values.
            extra=ids.id()
            clauses.append([-previous,-v,extra])
            contributions.append((w,extra))
            if j<len(named)-1:
                any_seen=ids.id()
                clauses.extend(([-previous,any_seen],[-v,any_seen]))
                previous=any_seen
    limit=available+1
    # A threshold means that at least its indexed subtotal was incurred.
    # Merge disjoint leaves; clamping to limit preserves every overflow.
    nodes=[{min(w,limit):v} for w,v in contributions]
    while len(nodes)>1:
        combined=[]
        for i in range(0,len(nodes)-1,2):
            left,right=nodes[i:i+2]
            sums=set(left)|set(right)|{min(a+b,limit) for a in left for b in right}
            parent={s:ids.id() for s in sorted(sums)}
            for source in (left,right):
                clauses.extend([[-v,parent[s]] for s,v in source.items()])
            for a,x in left.items():
                for b,y in right.items():
                    clauses.append([-x,-y,parent[min(a+b,limit)]])
            ordered=list(parent.values())
            clauses.extend([[-v,u] for u,v in zip(ordered,ordered[1:])])
            combined.append(parent)
        if len(nodes)%2:combined.append(nodes[-1])
        nodes=combined
    if nodes and limit in nodes[0]:
        clauses.append([-nodes[0][limit]])
    return clauses
