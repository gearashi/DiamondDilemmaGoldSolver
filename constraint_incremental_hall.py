"""Incremental exact AllDifferent SCC filtering with rollback snapshot reuse.

Optional rematch reuse verifies that repaired owners stay inside old SCCs.
Under narrowing, a perfect matching cannot use an old unsupported cross-SCC
edge, so repaired components may split but cannot merge.
With an unchanged matching, edge deletion can only split existing SCCs: only
components losing an internal edge need another traversal. Widening without an
exact saved snapshot also uses a full pass. Partial passes never publish state.
"""
from collections import OrderedDict,deque

STAT_KEYS=('hall_scc_vertices','hall_scc_reused_vertices','hall_scc_full_passes','hall_scc_partial_passes','hall_scc_snapshot_hits','hall_matching_edge_tests','hall_scc_rematched_partial_passes')


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
    def __init__(self,limit=32,*,rematch_reuse=False):
        if type(rematch_reuse)is not bool:raise ValueError('Rematch reuse must be boolean.')
        self.rematch_reuse=rematch_reuse
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
        narrowing=old is not None and len(old['rows'])==n and all(not new&~prior for new,prior in zip(rows,old['rows']))
        same_matching=narrowing and old['matching']==matched
        safe_rematch=False
        if narrowing and self.rematch_reuse and not same_matching:
            labels=[-1]*n;old_owner=[-1]*n
            for index,group in enumerate(old['groups']):
                for cell in group:labels[cell]=index
            for cell,tile in enumerate(old['matching']):old_owner[tile]=cell
            # A new matching belongs to the previous feasible graph. Explicitly
            # check that its alternating cycles stay inside old components.
            safe_rematch=all(labels[cell]==labels[old_owner[tile]] for cell,tile in enumerate(matched))
        if same_matching or safe_rematch:
            groups=[]
            for group in old['groups']:
                mask=sum(1<<x for x in group)
                if any((old['graph'][v]^graph[v])&mask for v in group):
                    groups.extend(components(graph,group,check,self.stats))
                else:
                    groups.append(group);self.stats['hall_scc_reused_vertices']+=len(group)
            self.stats['hall_scc_partial_passes']+=1
            self.stats['hall_scc_rematched_partial_passes']+=int(safe_rematch)
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
