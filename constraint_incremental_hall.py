"""Incremental exact AllDifferent SCC filtering with rollback snapshot reuse.

Matching repairs may change the residual graph globally and force a full pass.
With an unchanged matching, edge deletion can only split existing SCCs: only
components losing an internal edge need another traversal. Widening without an
exact saved snapshot also uses a full pass. Partial passes never publish state.
"""
from collections import OrderedDict,deque

STAT_KEYS=('hall_scc_vertices','hall_scc_reused_vertices','hall_scc_full_passes','hall_scc_partial_passes','hall_scc_snapshot_hits','hall_matching_edge_tests')


def bits(value):
    while value:
        bit=value&-value;yield bit.bit_length()-1;value^=bit


def components(graph,selected,check,stats):
    selected=frozenset(selected);mask=sum(1<<x for x in selected)
    reverse={x:0 for x in selected}
    for x in selected:
        check()
        for y in bits(graph[x]&mask):reverse[y]|=1<<x
    seen=0;order=[]
    for root in sorted(selected):
        if seen>>root&1:continue
        stack=[(root,False)]
        while stack:
            v,finish=stack.pop()
            if finish:order.append(v);continue
            if seen>>v&1:continue
            seen|=1<<v;stack.append((v,True))
            for w in bits(graph[v]&mask&~seen):stack.append((w,False))
    labels={};groups=[]
    for root in reversed(order):
        if root in labels:continue
        group=[];todo=[root];labels[root]=root
        while todo:
            check();v=todo.pop();group.append(v)
            for w in bits(reverse[v]):
                if w not in labels:labels[w]=root;todo.append(w)
        groups.append(tuple(sorted(group)))
    stats['hall_scc_vertices']+=len(selected)
    return tuple(groups)


class IncrementalHall:
    def __init__(self,limit=32):
        self.limit=limit;self.snapshots=OrderedDict();self.previous=None;self.stats=dict.fromkeys(STAT_KEYS,0)

    def filter(self,rows,previous_matching=None,check=lambda:None):
        rows=tuple(rows);n=len(rows)
        if not n or any(type(x)is not int or x<0 or x>>n for x in rows):raise ValueError('Invalid square cell/piece graph.')
        check()
        if rows in self.snapshots:
            saved=self.snapshots.pop(rows);self.snapshots[rows]=saved;self.previous=saved
            self.stats['hall_scc_snapshot_hits']+=1;self.stats['hall_scc_reused_vertices']+=n
            return saved['allowed'],saved['matching'],-1
        matched=[-1]*n;owner=[-1]*n
        seed=previous_matching if previous_matching is not None else self.previous['matching'] if self.previous else None
        if seed is not None:
            if len(seed)!=n:raise ValueError('Matching shape differs.')
            for cell,tile in enumerate(seed):
                if type(tile)is not int or not -1<=tile<n:raise ValueError('Matching code invalid.')
                if tile>=0 and rows[cell]>>tile&1 and owner[tile]<0:matched[cell]=tile;owner[tile]=cell
        for start in sorted(range(n),key=lambda x:rows[x].bit_count()):
            if matched[start]>=0:continue
            queue=deque([start]);seen={start};parents={};free=-1
            while queue and free<0:
                check();cell=queue.popleft()
                for tile in bits(rows[cell]):
                    self.stats['hall_matching_edge_tests']+=1
                    if tile in parents:continue
                    parents[tile]=cell;other=owner[tile]
                    if other<0:free=tile;break
                    if other not in seen:seen.add(other);queue.append(other)
            if free<0:return None,None,start
            while free>=0:
                cell=parents[free];old=matched[cell];matched[cell]=free;owner[free]=cell;free=old
        matched=tuple(matched)
        graph=tuple(sum(1<<owner[tile] for tile in bits(row)) for row in rows)
        old=self.previous
        if old is not None and old['matching']==matched and len(old['rows'])==n and all(not new&~prior for new,prior in zip(rows,old['rows'])):
            groups=[]
            for group in old['groups']:
                mask=sum(1<<x for x in group)
                if any((old['graph'][v]^graph[v])&mask for v in group):
                    groups.extend(components(graph,group,check,self.stats))
                else:
                    groups.append(group);self.stats['hall_scc_reused_vertices']+=len(group)
            self.stats['hall_scc_partial_passes']+=1
        else:
            groups=list(components(graph,range(n),check,self.stats));self.stats['hall_scc_full_passes']+=1
        allowed=list(rows)
        for group in groups:
            tiles=sum(1<<matched[x] for x in group)
            for cell in group:allowed[cell]&=tiles
        saved={'rows':rows,'matching':matched,'graph':graph,'groups':tuple(groups),'allowed':tuple(allowed)}
        check() # Commit complete immutable state only.
        self.previous=saved;self.snapshots[rows]=saved
        while len(self.snapshots)>self.limit:self.snapshots.popitem(last=False)
        return saved['allowed'],matched,-1
