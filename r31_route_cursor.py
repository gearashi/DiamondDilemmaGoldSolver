"""A directly seekable DFS cursor for the canonical connected-cycle generator."""
from itertools import permutations
from math import comb,factorial

from r28_exchange_cuda import unrank
from r31_route_geometry import cycle_mapping,parent


class AttachmentPage:
    """Direct cursors index raw attachments, including those rejected by the canonical owner.

    Reconstructing a frame derives its frontier from the selected cycles. It
    never iterates over earlier siblings to recover a saved offset.
    """
    def __init__(self,join,group):
        self.join=join
        self.group=group
        self.sources=cycle_mapping(group)
        pending=list(join.family)
        for cycle in group:
            pending.remove(len(cycle))
        self.final=len(pending)==1
        self.available=join.pool-self.sources.keys()
        self.boundary=set().union(*(join.near[c] for c in self.sources)) & self.available
        self.anchors=sorted(self.boundary)
        self.lengths=sorted(set(pending))
        self.score=join.base+sum(int(join.fit[self.sources.get(a,a),self.sources.get(b,b)])-int(join.fit[a,b])
                                 for a,b in join.edges if a in self.sources or b in self.sources)
        self.blocked=bool(join.actors and join.family==(2,2,3) and tuple(map(len,group))==(2,2)
                          and join.actors.actor_upper(group,sum(pending),pool=join.pool)<join.target)
        if not self.blocked and tuple(map(len,group))==(2,2) and join.family==(2,2,3):
            self.blocked = join.covered_swap_prefix(group)
        self.bonus={} if not self.final or self.blocked else join.attachment_bonuses(self.available,self.sources)
        self.cache={}

    def candidate(self,cursor):
        """Consume at most one row or bucket boundary, and return a surviving child."""
        if self.blocked:
            return None,None
        li,ai,pos=cursor
        if li>=len(self.lengths):
            return None,None
        if ai>=len(self.anchors):
            return (li+1,0,0),None
        length,anchor=self.lengths[li],self.anchors[ai]
        key=li,ai
        if key not in self.cache:
            available=self.available
            can_use=sorted(x for x in available if x!=anchor and (x not in self.boundary or x>anchor))
            if self.final:
                from bisect import bisect_right
                prices,rows=self.join.buckets(length,anchor)
                bonus=self.bonus[anchor]+sum(sorted((self.bonus[x] for x in can_use),reverse=True)[:length-1])
                end=bisect_right(prices,self.score+bonus-self.join.target) if len(can_use)>=length-1 else 0
            else:
                rows=None
                end=comb(len(can_use),length-1)*factorial(length-1)
            self.cache.clear()
            self.cache[key]=(can_use,set(can_use)|{anchor},rows,end)
        others,allowed,rows,end=self.cache[key]
        if pos>=end:
            return (li,ai+1,0),None
        after=(li,ai,pos+1)
        if rows is None:
            index,ori=divmod(pos,factorial(length-1))
            rest=[others[j] for j in unrank(index,len(others),length-1)]
            first,*tail=sorted([anchor,*rest])
            cycle=(first,*list(permutations(tail))[ori])
        else:
            cycle=tuple(map(int,rows[pos]))
            if not allowed.issuperset(cycle):
                return after,None
        child=tuple(sorted((*self.group,cycle)))
        if parent(child,self.join.close)!=self.group:
            return after,None
        if not self.final:
            return after,child
        moves=cycle_mapping((cycle,))
        change=self.sources|moves
        changed=sum(int(self.join.fit[change.get(a,a),change.get(b,b)])-
                    int(self.join.fit[self.sources.get(a,a),self.sources.get(b,b)])
                    for e in {i for c in cycle for i in self.join.incident[c]}
                    for a,b in (self.join.edges[e],))
        if self.score+changed<self.join.target:
            return after,None
        cells=tuple(sorted(change))
        index={c:i for i,c in enumerate(cells)}
        covered=getattr(self.join,'covered_support',None)
        if covered is not None and covered(cells):
            self.join.work['ranked_stars_avoided']+=1
            return after,None
        order=tuple(index[change[c]] for c in cells)
        self.join.work['exact_star_queries']+=1
        return after,(cells,order) if self.join.stars.price(cells,order)>=2*self.join.target else None


def restore_frames(state):
    return [(tuple(map(tuple,g)),tuple(pos)) for g,pos in state]


def advance(join,stack,frames,work,consume,expired,quantum):
    """One bounded page. An unresolved exact witness leaves its row as the next work item."""
    steps=0
    while stack and steps<quantum and not expired():
        group,pos=stack[-1]
        frame=frames.get(group)
        if frame is None:
            frame=frames[group]=AttachmentPage(join,group)
        after,item=frame.candidate(pos)
        if after is None:
            stack.pop()
            frames.pop(group,None)
        elif item is not None and frame.final:
            if not consume(*item):
                break
            stack[-1]=(group,after)
        else:
            stack[-1]=(group,after)
            if item is not None:
                stack.append((item,(0,0,0)))
        steps+=1
    work['cursor_steps']+=steps
    return steps
