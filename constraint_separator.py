"""Exact bounded frontier summaries for small connected puzzle regions.

A summary carries used physical tiles, every frontier edge mask (including
zero), the pairing of exposed gold endpoints, and completed-cycle count.
Histories merge only when all these agree. Their supported placements are
unioned: every history then has exactly the same possible local continuations.
Outside cells are optimistic, but each must have one unused tile/orientation
supporting all its incident region edges. No outside loop pairing is assumed.
Only a completed DP may narrow domains or publish a cache entry.
"""
from __future__ import annotations
from collections import deque
from constraint_pruning import values, _orientation_choices
from validator import reverse_mask

STAT_KEYS=('separator_checks','separator_prunes','separator_failures',
           'separator_cache_hits','separator_cache_entries','separator_states',
           'separator_merges','separator_budget_skips')


class _Budget(Exception):
    pass


class SeparatorPruner:
    def __init__(self,problem,target='gold'):
        if target not in ('gold','edges'):
            raise ValueError('Target must be gold or edges.')
        self.problem,self.target=problem,target
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.max_cells=8
        self.max_states=2048
        self.max_transitions=20000
        self.max_queries=2
        self.domain_cap=9
        self.cache_limit=64
        self.max_cached_summaries=8192
        self.cache={}
        self.last_region=None
        self.masks=tuple(tuple(map(int,row)) for row in problem.masks)
        self.pairs=tuple(tuple(((a[0]+rotation)%3*11+a[1]-1,
                                (b[0]+rotation)%3*11+b[1]-1)
                              for a,b in tile['segments'])
                         for tile in problem.tiles for rotation in range(3))
        self.adj=[[] for _ in range(problem.n)]
        self.neighbors=[set() for _ in range(problem.n)]
        for eid,(a,sa,b,sb) in enumerate(problem.edges):
            self.adj[a].append((b,sa,sb,eid));self.adj[b].append((a,sb,sa,eid))
            self.neighbors[a].add(b);self.neighbors[b].add(a)
        buckets=[{} for _ in range(3)]
        for code,row in enumerate(self.masks):
            for side,mask in enumerate(row):
                buckets[side][mask]=buckets[side].get(mask,0)|(1<<code)
        self.buckets=tuple(buckets)

    def _key(self,domains,cells):
        outside=tuple(sorted(set().union(*(self.neighbors[c] for c in cells))-set(cells)))
        # Full adjacent domains include physical inventory and all possible
        # pairings, even though the relaxation only uses their mask support.
        return (cells,tuple(domains[c] for c in cells),
                tuple((c,domains[c]) for c in outside))

    def _order(self,cells,domains):
        pending=set(cells);done=set();order=[]
        while pending:
            frontier=[c for c in pending if self.neighbors[c]&done] or list(pending)
            cell=min(frontier,key=lambda c:(sum((a in done|{c})!=(b in done|{c})
                                             for a,_,b,_ in self.problem.edges),
                                           domains[c].bit_count(),c))
            order.append(cell);done.add(cell);pending.remove(cell)
        return tuple(order)

    @staticmethod
    def _join(mates,u,v):
        """Join two exposed path ends; return one exactly when a cycle closes."""
        x,y=mates.pop(u),mates.pop(v)
        if x==v:
            return 1
        mates[x]=y;mates[y]=x
        return 0

    def _outside_possible(self,used,frontier,boundary,processed):
        excluded=_orientation_choices(used)
        for cell,domain in boundary:
            possible=domain&~excluded
            for other,side,_,eid in self.adj[cell]:
                if other not in processed:
                    continue
                canonical=frontier[eid]
                expected=canonical if self.problem.edges[eid][0]==cell else reverse_mask(canonical)
                possible &= self.buckets[side].get(expected,0)
            if not possible:
                return False
        return True

    def _solve(self,domains,key,check):
        cells,_,boundary=key
        order=self._order(cells,domains)
        offsets={c:i for i,c in enumerate(cells)}
        # key: (used physical tile bitset, frontier masks, endpoint pairs, cycles)
        # value: (candidate-support tuple, number of concrete local histories)
        layer={(0,(),(),0):((0,)*len(cells),1)}
        processed=set();old_edges=();transitions=0;created=1;merges=0
        for cell in order:
            check()
            new_processed=processed|{cell}
            new_edges=tuple(eid for eid,(a,_,b,_) in enumerate(self.problem.edges)
                            if (a in new_processed)!=(b in new_processed))
            following={}
            for (used,mask_tuple,pairs,cycles),(support,count) in layer.items():
                check()
                frontier=dict(zip(old_edges,mask_tuple))
                for code in values(domains[cell]):
                    transitions+=1
                    if transitions>self.max_transitions:
                        raise _Budget
                    if transitions%32==0:
                        check()
                    tile=1<<(code//3)
                    if used&tile:
                        continue
                    row=self.masks[code]
                    if any(self.problem.neighbors[cell,s]<0 and row[s] for s in range(3)):
                        continue
                    new_frontier=frontier.copy();valid=True
                    for other,side,other_side,eid in self.adj[cell]:
                        canonical=row[side] if self.problem.edges[eid][0]==cell else reverse_mask(row[side])
                        if other in processed:
                            if new_frontier.pop(eid)!=canonical:
                                valid=False;break
                        else:
                            new_frontier[eid]=canonical
                    if not valid:
                        continue
                    new_used=used|tile
                    if not self._outside_possible(new_used,new_frontier,boundary,new_processed):
                        continue
                    mates={p:q for a,b in pairs for p,q in ((a,b),(b,a))}
                    for a,b in self.pairs[code]:
                        a+=33*cell;b+=33*cell
                        mates[a]=b;mates[b]=a
                    new_cycles=cycles
                    for other,side,other_side,_ in self.adj[cell]:
                        if other not in processed:
                            continue
                        for position in values(row[side]):
                            u=33*cell+11*side+position
                            v=33*other+11*other_side+10-position
                            new_cycles+=self._join(mates,u,v)
                    # A closed component cannot be joined later. Every physical
                    # tile has at least one segment, so a Gold cycle can close
                    # only after ALL board cells have been placed, with no paths.
                    if self.target=='gold' and new_cycles and not (
                            len(new_processed)==self.problem.n and new_cycles==1 and not mates):
                        continue
                    new_pairs=tuple(sorted((a,b) for a,b in mates.items() if a<b))
                    signature=(new_used,tuple(new_frontier[e] for e in new_edges),new_pairs,new_cycles)
                    new_support=list(support);new_support[offsets[cell]]|=1<<code
                    previous=following.get(signature)
                    if previous is None:
                        if len(following)>=self.max_states:
                            raise _Budget
                        following[signature]=(tuple(new_support),count);created+=1
                    else:
                        following[signature]=(tuple(a|b for a,b in zip(previous[0],new_support)),previous[1]+count)
                        merges+=1
            layer=following;processed=new_processed;old_edges=new_edges
            if not layer:
                break
        supports=[0]*len(cells)
        completions=0
        for supported,count in layer.values():
            completions+=count
            supports=[a|b for a,b in zip(supports,supported)]
        summaries=tuple(sorted(layer))
        return {'supports':tuple(supports),'completions':completions,
                'summary_count':len(layer),'summaries':summaries,'frontier_edges':old_edges,
                'order':order,'transitions':transitions,'states':created,'merges':merges,
                'complete':True,'outside_pairing_assumptions':False}

    def prune_region(self,domains,cells,check=lambda:None):
        cells=tuple(sorted(set(cells)))
        if not 2<=len(cells)<=self.max_cells or not all(domains[c] for c in cells):
            return False
        check();key=self._key(domains,cells)
        self.stats['separator_checks']+=1
        result=self.cache.get(key)
        if result is None:
            try:
                result=self._solve(domains,key,check)
            except _Budget:
                self.stats['separator_budget_skips']+=1
                return False
            # Bound stored summaries as well as entry count: open inventories
            # can produce many distinct completed frontiers in one query.
            while self.cache and (len(self.cache)>=self.cache_limit or
                    sum(value['summary_count'] for value in self.cache.values())+result['summary_count']>self.max_cached_summaries):
                del self.cache[next(iter(self.cache))]
            self.cache[key]=result
            self.stats['separator_cache_entries']=len(self.cache)
            self.stats['separator_states']+=result['states']
            self.stats['separator_merges']+=result['merges']
        else:
            self.stats['separator_cache_hits']+=1
        self.last_region={'cells':cells,**{k:v for k,v in result.items() if k not in ('supports','summaries')}}
        changed=False
        for cell,support in zip(cells,result['supports']):
            check();reduced=domains[cell]&support
            if reduced!=domains[cell]:
                self.stats['separator_prunes']+=domains[cell].bit_count()-reduced.bit_count()
                domains[cell]=reduced;changed=True
                if not reduced:
                    self.stats['separator_failures']+=1
                    break
        return changed

    def _regions(self,domains,check):
        eligible={c for c,d in enumerate(domains) if 0<d.bit_count()<=self.domain_cap}
        regions=set()
        # Connected deterministic BFS prefixes avoid enumerating exponentially
        # many overlapping subsets. These include open-inventory 5..8-cell
        # regions rather than repeating the older closed-inventory 2..4 filter.
        for first in sorted(c for c in eligible if domains[c].bit_count()>1):
            check();queue=deque([first]);seen={first};region=[]
            while queue and len(region)<self.max_cells:
                cell=queue.popleft();region.append(cell)
                if len(region)>=5:
                    regions.add(tuple(sorted(region)))
                for other in sorted(self.neighbors[cell]&eligible):
                    if other not in seen:
                        seen.add(other);queue.append(other)
        if 2<=self.problem.n<=self.max_cells and len(eligible)==self.problem.n:
            regions.add(tuple(range(self.problem.n)))
        return sorted(regions,key=lambda r:(sum(domains[c].bit_count() for c in r),-len(r),r))

    def prune(self,domains,check=lambda:None):
        if not all(domains):
            return False
        changed=False
        for region in self._regions(domains,check)[:self.max_queries]:
            changed |= self.prune_region(domains,region,check)
            if not all(domains):
                break
        return changed
