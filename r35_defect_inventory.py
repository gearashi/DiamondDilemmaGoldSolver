"""Compile exact local donor-prefix defect clauses from only changed star signatures."""
from collections import Counter
from functools import lru_cache
from itertools import combinations

from r31_route_geometry import cycle_mapping


class DefectInventory:
    def __init__(self,bound):
        self.bound=bound
        self.tiles=tuple(map(int,bound.tiles))
        self.work=Counter()
        self.adj=[[] for _ in self.tiles]
        for e,(a,s,b,t) in enumerate(bound.model.problem.edges):
            self.adj[a].append((e,b,s,t,True))
            self.adj[b].append((e,a,t,s,False))
        self.pair=lru_cache(maxsize=16384)(self._pair)
        self.star=lru_cache(maxsize=8192)(self._star)
        self.base=tuple(self.star(c,self.tiles[c],tuple(self.tiles[b] for _,b,*_ in self.adj[c])) for c in range(len(self.tiles)))

    def _pair(self,left,side,right,opposite):
        self.work['directional_pair_tests']+=1
        target=set(self.bound.model.reverse[self.bound.masks[right,:,opposite]])
        return sum(1<<r for r,m in enumerate(self.bound.masks[left,:,side]) if m in target)

    def _star(self,cell,tile,neighbors):
        self.work['stars_built']+=1
        good=[];singles=[];multi=[]
        for (e,b,s,t,owner),other in zip(self.adj[cell],neighbors):
            support=self.pair(tile,s,other,t)
            if support:good.append((e,b,support))
            elif owner:singles.append(((e,),(1<<cell)|(1<<b)))
        smaller=[]
        for count in range(2,len(good)+1):
            for subset in combinations(good,count):
                edges=tuple(sorted(e for e,_,_ in subset))
                if any(c<=set(edges) for c in smaller):continue
                common=7;cells=1<<cell
                for _,b,mask in subset:
                    common&=mask;cells|=1<<b
                if not common:
                    smaller.append(set(edges))
                    multi.append((edges,cells))
        return tuple(singles),tuple(multi)

    def rows(self,group,free):
        sources=cycle_mapping(group)
        available=sum(1<<int(c) for c in set(free)-sources.keys())
        affected=set(sources)
        for c in sources:affected.update(b for _,b,*_ in self.adj[c])
        current=list(self.base)
        for c in affected:
            current[c]=self.star(c,self.tiles[sources.get(c,c)],tuple(self.tiles[sources.get(b,b)] for _,b,*_ in self.adj[c]))
        self.work['queries']+=1
        self.work['unchanged_stars']+=len(current)-len(affected)
        self.work['changed_signatures_requested']+=len(affected)
        # Preserve the previous order of model clauses for deterministic CP replay.
        singles=sorted(row for items,_ in current for row in items)
        groups=(row for _,items in current for row in items)
        return [(edges,mask&available) for edges,mask in [*singles,*groups]]


def for_bound(bound):
    previous=getattr(bound,'_defect_inventory',None)
    if previous is None or previous.tiles!=tuple(bound.tiles):
        previous=bound._defect_inventory=DefectInventory(bound)
    return previous
