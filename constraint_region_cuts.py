"""Conservative Gold connectivity bounds for several ports and endpoint supply.

An optimistic graph contains every local strand allowed by a domain and every
potential matching seam.  Removing k vertices from a single cycle leaves at
most k nonempty path components.  We count only components that must contain
an active vertex, never optional-only components.

For a cell region, every component of the optimistic outside cell graph needs
at least two crossing endpoints.  Bounds take the smaller of cell capacity
and distinct-physical-piece capacity.  A piece that can serve several cuts is
counted only once in the joint bound.
"""
from __future__ import annotations
from itertools import combinations, zip_longest
from constraint_pruning import values
from validator import reverse_mask

STAT_KEYS=('multi_port_checks','multi_port_cuts','multi_port_prunes',
    'multi_port_failures','multi_port_budget_skips','endpoint_supply_checks',
    'endpoint_supply_prunes','endpoint_supply_failures')


class MultiPortCutPruner:
    def __init__(self,problem):
        self.problem=problem
        self.max_cuts=48
        self.max_candidates=12
        self.max_vertices=6000
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.pairs=tuple(tuple(((a[0]+r)%3*11+a[1]-1,
                                (b[0]+r)%3*11+b[1]-1)
                              for a,b in tile['segments'])
                         for tile in problem.tiles for r in range(3))
        self.masks=tuple(tuple(map(int,row)) for row in problem.masks)

    @staticmethod
    def violates_cut(graph,mandatory,removed,check=lambda:None,cell_choices=()):
        removed=set(removed);labels={};component=0
        for start in graph:
            if start in removed or start in labels:continue
            check();labels[start]=component;queue=[start]
            for current in queue:
                for other in graph[current]:
                    if other not in removed and other not in labels:
                        labels[other]=component;queue.append(other)
            component+=1
        required={labels[p] for p in mandatory if p in labels}
        for choices in cell_choices:
            possible=set();always=True
            for ports in choices:
                remaining=ports-removed
                if not remaining:always=False;break
                possible.update(labels[p] for p in remaining)
            if always and len(possible)==1:required.update(possible)
        limit=max(1,len(removed))
        return len(required)>limit

    def _graph(self,domains,check):
        graph={};mandatory=set();cell_choices=[]
        def edge(a,b):
            graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        possible=[]
        for cell,domain in enumerate(domains):
            check();all_ports=set();intersection=None;choices=[]
            for code in values(domain):
                ports=set()
                for p,q in self.pairs[code]:
                    p+=33*cell;q+=33*cell;edge(p,q);ports.update((p,q))
                all_ports.update(ports);choices.append(ports)
                intersection=ports.copy() if intersection is None else intersection&ports
            mandatory.update(intersection or ())
            possible.append(all_ports);cell_choices.append(choices)
        for a,sa,b,sb in self.problem.edges:
            check()
            for position in range(11):
                p=33*a+11*sa+position;q=33*b+11*sb+10-position
                if p in possible[a] and q in possible[b]:edge(p,q)
        return graph,mandatory,cell_choices

    def impossible(self,domains,check=lambda:None):
        graph,mandatory,choices=self._graph(domains,check)
        if len(graph)>self.max_vertices:
            self.stats['multi_port_budget_skips']+=1;return False
        # Pairs and triples extend ordinary bridge/articulation tests.  Small
        # degree and forced vertices preferentially expose useful separators.
        ranked=sorted(graph,key=lambda p:(p not in mandatory,len(graph[p]),p))[:16]
        cuts=[()]
        for pair,triple in zip_longest(combinations(ranked,2),combinations(ranked[:8],3)):
            if pair is not None:cuts.append(pair)
            if triple is not None:cuts.append(triple)
            if len(cuts)>=self.max_cuts:break
        for cut in cuts[:self.max_cuts]:
            check();self.stats['multi_port_cuts']+=1
            if self.violates_cut(graph,mandatory,cut,check,choices):return True
        return False

    def prune(self,domains,check=lambda:None):
        if not all(domains):return False
        self.stats['multi_port_checks']+=1
        if self.impossible(domains,check):
            self.stats['multi_port_failures']+=1
            self.stats['multi_port_prunes']+=domains[0].bit_count();domains[0]=0
            return True
        tested=0;changed=False
        for cell in sorted(range(len(domains)),key=lambda c:domains[c].bit_count()):
            if domains[cell].bit_count()<2:continue
            for code in tuple(values(domains[cell])):
                if tested>=self.max_candidates:return changed
                check();tested+=1;trial=domains.copy();trial[cell]=1<<code
                if self.impossible(trial,check):
                    domains[cell]&=~(1<<code);changed=True
                    self.stats['multi_port_prunes']+=1
                    if not domains[cell]:return True
        return changed


class EndpointSupplyPruner:
    def __init__(self,problem):
        self.problem=problem
        self.masks=tuple(tuple(map(int,row)) for row in problem.masks)
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.max_candidates=24

    def _components(self,domains,region,check):
        remaining=set(range(self.problem.n))-set(region)
        graph={c:set() for c in remaining}
        for a,sa,b,sb in self.problem.edges:
            if a not in remaining or b not in remaining:continue
            left={self.masks[c][sa] for c in values(domains[a])}
            right={reverse_mask(self.masks[c][sb]) for c in values(domains[b])}
            if (left&right)-{0}:graph[a].add(b);graph[b].add(a)
        groups=[]
        while remaining:
            check();seen={min(remaining)};queue=list(seen)
            for cell in queue:
                for other in graph[cell]-seen:seen.add(other);queue.append(other)
            remaining-=seen;groups.append(seen)
        return groups

    def impossible(self,domains,region,check=lambda:None):
        region=tuple(sorted(set(region)))
        if not region or len(region)==self.problem.n:return False
        components=self._components(domains,region,check)
        def capacity(outside):
            # Optimistic capacity respects each physical tile's single use.
            boundary={c:[] for c in region}
            for a,sa,b,sb in self.problem.edges:
                if a in boundary and b in outside:boundary[a].append((sa,b,sb))
                if b in boundary and a in outside:boundary[b].append((sb,a,sa))
            by_tile={};cell_sum=0
            for cell,sides in boundary.items():
                check();best=0
                for code in values(domains[cell]):
                    count=0
                    for side,other,other_side in sides:
                        # If a mask has no equal neighbor mask it cannot supply
                        # these endpoints in any completion.
                        mask=self.masks[code][side]
                        if any(mask==reverse_mask(self.masks[c][other_side]) for c in values(domains[other])):
                            count+=mask.bit_count()
                    best=max(best,count);by_tile[code//3]=max(by_tile.get(code//3,0),count)
                cell_sum+=best
            return min(cell_sum,sum(sorted(by_tile.values(),reverse=True)[:len(region)]))
        if capacity(set().union(*components))<2*len(components):return True
        return any(capacity(component)<2 for component in components)

    def prune_region(self,domains,region,check=lambda:None):
        if not all(domains):return False
        self.stats['endpoint_supply_checks']+=1
        region=tuple(sorted(set(region)))
        if self.impossible(domains,region,check):
            cell=region[0];self.stats['endpoint_supply_failures']+=1
            self.stats['endpoint_supply_prunes']+=domains[cell].bit_count();domains[cell]=0
            return True
        changed=False;tested=0
        for cell in region:
            if domains[cell].bit_count()<2:continue
            for code in tuple(values(domains[cell])):
                if tested>=self.max_candidates:return changed
                check();tested+=1;trial=domains.copy();trial[cell]=1<<code
                if self.impossible(trial,region,check):
                    domains[cell]&=~(1<<code);changed=True
                    self.stats['endpoint_supply_prunes']+=1
                    if not domains[cell]:return True
        return changed
