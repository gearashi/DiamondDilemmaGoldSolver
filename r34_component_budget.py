"""A regional swap upper bound with disjoint star components and shared donor inventory."""
from itertools import combinations, permutations


from r30_bounds import budget_upper,defect_packings


class ComponentBudget:
    def __init__(self,bound,allowed):
        self.bound=bound
        self.pool=tuple(sorted(set(map(int,allowed))))
        if not set(self.pool)<=set(map(int,bound.model.free)):
            raise ValueError("Expected only movable cells.")
        footprints={c:{c,*[int(d) for d in bound.neighbors[c] if d!=bound.n]} for c in self.pool}
        self.groups=[]
        todo=set(self.pool)
        while todo:
            group={min(todo)}
            overlap=set(footprints[min(todo)])
            todo-=group
            while True:
                more={c for c in todo if footprints[c] & overlap}
                if not more:
                    break
                group.update(more)
                todo-=more
                for c in more:
                    overlap.update(footprints[c])
            self.groups.append((tuple(sorted(group)),tuple(sorted(overlap))))
        self.index={c:i for i,c in enumerate(self.pool)}
        self.tables={}
        self.finished={}
        self.packings=defect_packings(bound)

    def _responses(self,group,footprint,cap,work):
        b=self.bound
        entries={}
        indices=list(group)
        touched=list(footprint)
        delta=int(b.base[touched].sum())
        for donors in permutations(self.pool,len(group)):
            if work['priced']>=cap:
                return None
            work['priced']+=1
            change={c:d for c,d in zip(group,donors) if c!=d}
            if any(not b.model.legal[c,3*b.tiles[d]:3*b.tiles[d]+3].any() for c,d in change.items()):
                continue
            # Removing complete local cycles leaves open chains. A global
            # cycle crossing a component boundary uses at least two crossings.
            pending=set(change)
            cycles=0
            while pending:
                c=next(iter(pending))
                path=set()
                while c in pending:
                    pending.remove(c)
                    path.add(c)
                    c=change[c]
                cycles+=int(c in path)
            crossing=sum(d not in group for d in donors)
            cost=2*(len(change)-cycles)-crossing
            tiles=b.pieces.copy()
            tiles[indices]=b.tiles[list(donors)]
            gain=int(b.stars(tiles[touched],tiles[b.neighbors[touched]]).sum())-delta
            mask=sum(1<<self.index[d] for d in donors)
            key=(mask,cost,len(change))
            entries[key]=max(gain,entries.get(key,gain))
        by_mask={}
        for (mask,cost,moved),gain in entries.items():
            by_mask.setdefault(mask,[]).append((cost,moved,gain))
        return by_mask

    def upper(self,swaps,max_rows=100000,min_moved=0):
        if type(swaps) is not int or swaps<0 or max_rows<1 or not 0<=min_moved<=len(self.pool):
            raise ValueError("Invalid swap or row budget.")
        scope=(swaps,min_moved)
        if scope in self.finished:
            return dict(**self.finished[scope],work=dict(priced=0,joins=0,reused=0,result_reuses=1))
        b=self.bound
        cap=budget_upper(b,self.packings,swaps,self.pool)
        work=dict(priced=0,joins=0,reused=0)
        if len(self.groups)<2:
            return dict(upper=cap,complete=False,groups=len(self.groups),reason='one component',work=work)
        state={(0,0,0):0}
        for group,footprint in self.groups:
            table=self.tables.get(group)
            if table is None:
                table=self._responses(group,footprint,max_rows,work)
                if table is None:
                    return dict(upper=cap,complete=False,groups=len(self.groups),reason='row budget',work=work)
                self.tables[group]=table
            else:
                work['reused']+=sum(map(len,table.values()))
            following={}
            for (mask,old,moved),value in state.items():
                available=[i for i in range(len(self.pool)) if not mask>>i&1]
                for ids in combinations(available,len(group)):
                    donors=sum(1<<i for i in ids)
                    if work['joins']>=max_rows:
                        return dict(upper=cap,complete=False,groups=len(self.groups),reason='join budget',work=work)
                    work['joins']+=1
                    for extra,count,gain in table.get(donors,()):
                        if old+extra<=2*swaps:
                            key=(mask|donors,old+extra,min(min_moved,moved+count))
                            following[key]=max(following.get(key,value+gain),value+gain)
            state=following
        best=max((value for (_,_,moved),value in state.items() if moved==min_moved),default=None)
        upper=-1 if best is None else min(cap,(b.total+best)//2)
        record=dict(upper=upper,complete=True,groups=len(self.groups),coarse=cap,min_moved=min_moved)
        self.finished[scope]=record
        return dict(**record,work=work)
