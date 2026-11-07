"""Scoped relational deductions and exact joint signature capacities.

A retained table is a necessary relation derived under an explicit domain
scope. Widening outside that scope disables it. Capacity filtering is an exact
bipartite matching relaxation; physical tile correlations remain elsewhere.
"""
from collections import OrderedDict, deque


def bits(mask):
    while mask:
        low=mask&-mask;mask^=low;yield low.bit_length()-1


class ScopedRelations:
    def __init__(self, limit=128):
        self.limit=limit;self.records=OrderedDict()
        self.stats={'relation_records':0,'relation_hits':0,'relation_prunes':0}

    def remember(self, scope, left, right, allowed):
        scope=tuple(sorted(scope));allowed=tuple(sorted((int(x),int(y)) for x,y in allowed))
        key=(scope,int(left),int(right))
        self.records[key]=allowed;self.records.move_to_end(key)
        while len(self.records)>self.limit:self.records.popitem(last=False)
        self.stats['relation_records']=len(self.records)

    def prune(self, domains, check=lambda:None):
        current=list(domains)
        for (scope,left,right),allowed in tuple(self.records.items()):
            check()
            if any(current[cell]&~mask for cell,mask in scope):continue
            a=b=0
            for x,y in allowed:
                if current[left]&(1<<x) and current[right]&(1<<y):a|=1<<x;b|=1<<y
            if a!=current[left] or b!=current[right]:self.stats['relation_hits']+=1
            current[left]&=a;current[right]&=b
            if not all(current):break
        check();self.stats['relation_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed


def capacity_supports(choices, capacities, check=lambda:None, stats=None):
    """All supported seam-to-class choices with exact integer capacities.

    Expanding each capacity into interchangeable slots reduces GCC to perfect
    matching. An unmatched edge occurs in some perfect matching exactly when
    it lies on an alternating cycle. The SCC computation below is independent
    of the solver's Hall implementation and also handles zero capacities.
    """
    choices=tuple(frozenset(row) for row in choices);capacities=dict(capacities)
    if any(type(v)is not int or v<0 for v in capacities.values()):raise ValueError('Invalid capacity.')
    if sum(capacities.values())!=len(choices):return None
    slots=[];by_class={}
    for label,count in sorted(capacities.items()):
        by_class[label]=tuple(range(len(slots),len(slots)+count));slots.extend([label]*count)
    rows=[tuple(slot for label in sorted(row) for slot in by_class.get(label,())) for row in choices]
    n=len(rows);matched=[-1]*n;owner=[-1]*n
    for start in sorted(range(n),key=lambda c:(len(rows[c]),c)):
        check();pending=deque([start]);seen={start};parents={};free=None
        while pending and free is None:
            cell=pending.popleft()
            for slot in rows[cell]:
                if stats is not None:stats['capacity_edges']=stats.get('capacity_edges',0)+1
                if slot in parents:continue
                parents[slot]=cell
                if owner[slot]<0:free=slot;break
                other=owner[slot]
                if other not in seen:seen.add(other);pending.append(other)
            check()
        if free is None:return None
        while free>=0:
            cell=parents[free];previous=matched[cell];matched[cell]=free;owner[free]=cell;free=previous
    graph=[{owner[slot] for slot in row} for row in rows]
    reverse=[set() for _ in rows]
    for cell,targets in enumerate(graph):
        for target in targets:reverse[target].add(cell)
    seen=set();order=[]
    for root in range(n):
        if root in seen:continue
        seen.add(root);stack=[(root,iter(sorted(graph[root])))]
        while stack:
            check();node,children=stack[-1]
            try:other=next(children)
            except StopIteration:order.append(node);stack.pop();continue
            if other not in seen:seen.add(other);stack.append((other,iter(sorted(graph[other]))))
    labels=[-1]*n
    for root in reversed(order):
        if labels[root]>=0:continue
        labels[root]=root;stack=[root]
        while stack:
            check();node=stack.pop()
            for other in reverse[node]:
                if labels[other]<0:labels[other]=root;stack.append(other)
    return tuple(frozenset(slots[slot] for slot in row if labels[cell]==labels[owner[slot]])
                 for cell,row in enumerate(rows))
