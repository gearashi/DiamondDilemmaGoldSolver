"""Necessary vertex-ring constraints on a partial physical sewing of a triangular sphere."""
from collections import Counter
from functools import lru_cache

from r27_structure_filters import _vertex_links


def degree_inventory(problem):
    corners = _vertex_links(problem)
    if corners is None or any(len(set(corners[i:i+3])) != 3 for i in range(0,len(corners),3)):
        return None
    return Counter(Counter(corners).values())


@lru_cache(maxsize=4096)
def pack_lengths(open_counts, available, limit=4096):
    """Fit indivisible open paths in the remaining vertex degrees, ignoring identities."""
    sizes = [s for s,k in enumerate(open_counts) if s >= 3 for _ in range(k)][::-1]
    pairs = open_counts[2] if len(open_counts)>2 else 0
    if sum(i*k for i,k in enumerate(open_counts)) != sum(i*k for i,k in enumerate(available)):
        return False
    calls = 0
    @lru_cache(None)
    def visit(i,slots):
        nonlocal calls
        if calls >= limit:
            return None
        calls += 1
        if i==len(sizes):
            return pairs <= sum((d//2)*k for d,k in enumerate(slots))
        w = sizes[i]
        unknown = False
        for d in range(w,len(slots)):
            if not slots[d]:
                continue
            out=list(slots)
            out[d]-=1
            if d>w:
                out[d-w]+=1
            result=visit(i+1,tuple(out))
            if result is True:
                return True
            unknown |= result is None
        return None if unknown else False
    return visit(0,available)


class CornerChains:
    """Rollback components of the directed corner paths induced by fixed side pairs."""
    def __init__(self,n,inventory,packing=True):
        self.n = n
        self.maximum = max(inventory)
        self.slots = [inventory.get(d,0) for d in range(self.maximum+1)]
        self.open = [0]*(self.maximum+1)
        self.open[1] = 3*n
        self.parent = list(range(3*n))
        self.tiles = [1<<(i//3) for i in self.parent]
        self.length = [1]*(3*n)
        self.mates = [-1]*(3*n)
        self.trail = []
        self.packing = packing
        self.reason = None

    def root(self,i):
        while i != self.parent[i]:
            i = self.parent[i]
        return i

    def mark(self):
        return len(self.trail)

    def rollback(self,mark):
        while len(self.trail)>mark:
            kind,a,b = self.trail.pop()
            if kind=='pair':
                self.mates[a]=self.mates[b]=-1
            elif kind=='close':
                self.slots[a]+=1
                self.open[a]+=1
            else:
                self.open[self.length[a]]-=1
                self.length[a]-=self.length[b]
                self.open[self.length[a]]+=1
                self.open[self.length[b]]+=1
                self.tiles[a]^=self.tiles[b]
                self.parent[b]=b

    def _join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a==b:
            d=self.length[a]
            if not self.slots[d]:
                return 'unavailable_degree'
            self.trail.append(('close',d,0))
            self.slots[d]-=1
            self.open[d]-=1
            return None
        if self.tiles[a]&self.tiles[b]:
            return 'repeated_tile'
        d=self.length[a]+self.length[b]
        if d>self.maximum or not any(self.slots[d:]):
            return 'overlong_chain'
        if self.length[a]<self.length[b]:
            a,b=b,a
        self.trail.append(('join',a,b))
        self.parent[b]=a
        self.tiles[a]|=self.tiles[b]
        self.open[self.length[a]]-=1
        self.open[self.length[b]]-=1
        self.open[d]+=1
        self.length[a]=d
        return None

    def push(self,pair):
        a,b=pair
        if self.mates[a] == b:
            return True
        if a==b or self.mates[a]>=0 or self.mates[b]>=0:
            self.reason='side_ownership'
            return False
        mark=self.mark()
        self.trail.append(('pair',a,b))
        self.mates[a],self.mates[b]=b,a
        reason=self._join(a,b-b%3+(b+1)%3) or self._join(b,a-a%3+(a+1)%3)
        if not reason:
            largest = max((d for d,k in enumerate(self.slots) if k),default=0)
            if any(self.open[largest+1:]):
                reason='overlong_chain'
            elif self.packing and pack_lengths(tuple(self.open),tuple(self.slots)) is False:
                reason='length_packing'
        if reason:
            self.rollback(mark)
        self.reason=reason
        return reason is None


    def conflict(self,active):
        mark=self.mark()
        free=[]
        try:
            for pair in active:
                if self.mates[pair[0]]==pair[1]:
                    continue
                free.append(pair)
                if not self.push(pair):
                    return free
            return None
        finally:
            self.rollback(mark)


def filter_options(problem,options,required,*,packing=True,check_exclusion=None):
    """Propagate mandatory joins and reject unsupported single additions to that scope."""
    inventory = degree_inventory(problem)
    if inventory is None:
        return set(options),None,dict(applicable=False)
    state = CornerChains(problem.n,inventory,packing)
    live=set(options)
    fixed=set(required)
    stats=Counter()
    while True:
        by_side=[[] for _ in state.mates]
        for a,b in live:
            by_side[a].append((a,b))
            by_side[b].append((a,b))
        if any(not row for row in by_side) or not fixed <= live:
            stats['empty_side']+=1
            return set(),state,dict(stats)
        fixed.update(row[0] for row in by_side if len(row)==1)
        for pair in sorted(fixed):
            if not state.push(pair):
                if check_exclusion is not None:
                    check_exclusion(fixed)
                stats[state.reason]+=1
                return set(),state,dict(stats)
        mark=state.mark()
        for pair in sorted(live-fixed):
            stats['trial_joins']+=1
            if not state.push(pair):
                if check_exclusion is not None:
                    check_exclusion(fixed|{pair})
                live.remove(pair)
                stats[state.reason]+=1
            state.rollback(mark)
        # Repeat only when the filtering newly dictates a single mate.
        count=Counter(d for pair in live for d in pair)
        if len(count) != len(state.mates):
            stats['empty_side']+=1
            return set(),state,dict(stats)
        new_fixed = {pair for pair in live if any(count[d] == 1 for d in pair)} - fixed
        if not new_fixed:
            stats['fixed_pairs']=len(fixed)
            return live,state,dict(stats)


def post_positions(check):
    """Give each corner its vertex, ring length and position, with exact degree multiplicities."""
    from ortools.sat.python import cp_model
    p,m = check.problem,check.m
    inventory = degree_inventory(p)
    if inventory is None:
        return dict(applicable=False)
    count=3*p.n
    degrees=sorted(inventory)
    reps=[]; ranks=[]; lengths=[]; next_ranks=[]
    labels={d:[] for d in degrees}
    for i in range(count):
        v=m.new_int_var(0,i,'corner_representative')
        pos=m.new_int_var(0,max(degrees)-1,'corner_position')
        length=m.new_int_var_from_domain(cp_model.Domain.from_values(degrees),'corner_degree')
        nxt=m.new_int_var(0,max(degrees)-1,'next_position')
        zero=m.new_bool_var('root_corner')
        last=m.new_bool_var('last_corner')
        m.add(pos==0).only_enforce_if(zero)
        m.add(pos!=0).only_enforce_if(~zero)
        m.add(v==i).only_enforce_if(zero)
        m.add(pos<length)
        m.add(pos==length-1).only_enforce_if(last)
        m.add(pos!=length-1).only_enforce_if(~last)
        m.add(nxt==0).only_enforce_if(last)
        m.add(nxt==pos+1).only_enforce_if(~last)
        for d in degrees:
            yes=m.new_bool_var('degree_selected')
            m.add(length==d).only_enforce_if(yes)
            m.add(length!=d).only_enforce_if(~yes)
            labels[d].append(yes)
        reps.append(v);ranks.append(pos);lengths.append(length);next_ranks.append(nxt)
    for d,row in labels.items():
        m.add(sum(row)==inventory[d]*d)
    for tile in range(p.n):
        m.add_all_different(reps[3*tile:3*tile+3])
    for (a,b),chosen in check.mate.items():
        for x,y in ((a,b),(b,a)):
            z=y-y%3+(y+1)%3
            m.add(reps[x]==reps[z]).only_enforce_if(chosen)
            m.add(lengths[x]==lengths[z]).only_enforce_if(chosen)
            m.add(next_ranks[x]==ranks[z]).only_enforce_if(chosen)
    return dict(applicable=True,corners=count,degrees=dict(inventory),guarded_steps=2*len(check.mate))
