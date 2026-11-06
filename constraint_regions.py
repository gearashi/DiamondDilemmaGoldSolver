"""Bounded hierarchical joins of exact regional boundary summaries.

Each child table represents every feasible local history under optimistic
outside constraints. A separator join requires disjoint physical inventories
and equal masks on every shared seam, then joins the exact exposed strand
pairing and counts any closed cycles. Only identical full summaries merge.
Histories in a merged summary have identical future feasibility, so their
placement supports can be unioned and their counts multiplied at later joins.
No incomplete table is cached or used to narrow a domain.
"""
from __future__ import annotations
from collections import deque, OrderedDict
from itertools import combinations
from math import log2, comb
from constraint_separator import SeparatorPruner, _Budget
from constraint_pruning import values, _tile_choices
from validator import reverse_mask

STAT_KEYS=('hierarchical_checks','hierarchical_prunes','hierarchical_failures',
           'hierarchical_cache_hits','hierarchical_cache_entries',
           'hierarchical_states','hierarchical_merges','hierarchical_budget_skips',
           'hierarchical_joins','hierarchical_subtable_hits','hierarchical_subtable_misses',
           'hierarchical_subtable_entries','hierarchical_subtable_states',
           'hierarchical_subtable_support_slots','hierarchical_subtable_evictions',
           'hierarchical_cached_summaries','hierarchical_cached_support_slots',
           'hierarchical_pair_checks','hierarchical_pair_comparisons','hierarchical_pair_joins','hierarchical_pair_prunes',
           'hierarchical_pair_failures','hierarchical_pair_budget_skips','hierarchical_pair_summaries_removed')


class HierarchicalRegionPruner(SeparatorPruner):
    def __init__(self,problem,target='gold'):
        super().__init__(problem,target)
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.max_cells=16
        self.max_states=4096
        self.max_transitions=200000
        self.max_created=20000
        self.max_queries=1
        self.max_pair_queries=1
        self.max_pair_transitions=20000
        self.max_pair_candidates=256
        self.domain_cap=6
        self.cache_limit=32
        self.max_cached_summaries=8192
        self.subtable_cache_limit=128
        self.max_cached_support_slots=65536
        self.subtables=OrderedDict()
        self._subtable_index={}
        self._subtable_projections={}
        self._query_cached_summaries=0
        self._query_support_slots=0
        self.last_pair=None
        self._subtable_states=0
        self._subtable_support_slots=0
        # Called immediately after publishing a proven reduction. The callback
        # receives current-query dependency cells. Cancellation preserves the
        # sound, already narrowed domain and lets its owner reconcile reasons.
        self.on_narrow=None
        self.extended_regions=False
        self.max_frontier_edges=None
        self._network=None
        self._disk_store=None
        self._plan_cache=OrderedDict()
        self._region_activity={}
        self.extended_stats={}

    def enable_advanced(self, cache_dir=None):
        from constraint_region_network import RegionNetwork
        self.extended_regions=True
        self.max_cells=self.problem.n
        self.max_frontier_edges=16
        if self._network is None:self._network=RegionNetwork(self)
        if cache_dir is not None:
            from constraint_region_store import RegionalTableStore
            self._disk_store=RegionalTableStore(cache_dir,self)
        return self

    def note_conflict(self,cells):
        for cell in cells:
            if 0<=cell<self.problem.n:
                self._region_activity[cell]=min(100000,self._region_activity.get(cell,0)+1)

    def prune_network(self,domains,regions,check=lambda:None):
        if self._network is None:
            from constraint_region_network import RegionNetwork
            self._network=RegionNetwork(self)
        changed=self._network.prune(domains,regions,check)
        self.extended_stats.update(self._network.stats)
        return changed

    def prune_overlap(self,domains,left,right,check=lambda:None):
        if self._network is None:
            from constraint_region_network import RegionNetwork
            self._network=RegionNetwork(self)
        changed=self._network.prune_overlap(domains,left,right,check)
        self.extended_stats.update(self._network.stats)
        return changed

    def prune_components(self,domains,components,check=lambda:None):
        if self._network is None:
            from constraint_region_network import RegionNetwork
            self._network=RegionNetwork(self)
        changed=self._network.prune_components(domains,components,check)
        self.extended_stats.update(self._network.stats)
        return changed

    def _choose_partition(self,cells,domains,check):
        if not self.extended_regions:return self._partition(cells,domains,check)
        key=(cells,tuple(domains[c].bit_count() for c in cells))
        if key in self._plan_cache:
            self.extended_stats['region_plan_hits']=self.extended_stats.get('region_plan_hits',0)+1
            self._plan_cache.move_to_end(key)
            return self._plan_cache[key]
        baseline=self._partition(cells,domains,check)
        order=self._order(cells,domains)
        candidates={baseline}
        for cut in {max(1,len(cells)//3),len(cells)//2,max(1,2*len(cells)//3)}:
            if cut<len(cells):candidates.add((tuple(sorted(order[:cut])),tuple(sorted(order[cut:]))))
        def score(pair):
            widths=[len(self._frontier(set(part))) for part in pair]
            return (max(widths),sum(widths),max(sum(log2(domains[c].bit_count()) for c in part) for part in pair),pair)
        result=min(candidates,key=score)
        self._plan_cache[key]=result
        while len(self._plan_cache)>2048:self._plan_cache.popitem(last=False)
        self.extended_stats['region_plan_alternatives']=self.extended_stats.get('region_plan_alternatives',0)+len(candidates)
        return result

    def _cache_stats(self):
        self.stats['hierarchical_cache_entries']=len(self.cache)
        self.stats['hierarchical_subtable_entries']=len(self.subtables)
        self.stats['hierarchical_subtable_states']=self._subtable_states
        self.stats['hierarchical_subtable_support_slots']=self._subtable_support_slots
        self.stats['hierarchical_cached_summaries']=self._query_cached_summaries+self._subtable_states
        self.stats['hierarchical_cached_support_slots']=self._query_support_slots+self._subtable_support_slots

    def _drop_subtable(self):
        key,table=self.subtables.popitem(last=False)
        bucket=self._subtable_index[key[:3]];bucket.remove(key)
        if not bucket:del self._subtable_index[key[:3]]
        self._subtable_projections.pop(key,None)
        self._subtable_states-=len(table)
        self._subtable_support_slots-=len(table)*len(key[1])
        self.stats['hierarchical_subtable_evictions']+=1

    def _drop_query(self):
        key=next(iter(self.cache));result=self.cache[key]
        del self.cache[key]
        self._query_cached_summaries-=result['summary_count']
        self._query_support_slots-=result['summary_count']*len(key[0])

    def _remember_subtable(self,key,table,check):
        # A completed child is reusable even when a later parent join stops.
        # Resource limits do not change table semantics or justify truncation.
        check();states=len(table);slots=states*len(key[1])
        if (self.subtable_cache_limit<=0 or states>self.max_cached_summaries
                or slots>self.max_cached_support_slots):return
        while self.subtables and (len(self.subtables)>=self.subtable_cache_limit or
                self._query_support_slots+self._subtable_support_slots+slots>self.max_cached_support_slots or
                self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries):
            self._drop_subtable()
        while self.cache and (self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries or
                self._query_support_slots+self._subtable_support_slots+slots>self.max_cached_support_slots):
            self._drop_query()
        self.subtables[key]=table
        self._subtable_index.setdefault(key[:3],set()).add(key)
        self._subtable_states+=states;self._subtable_support_slots+=slots
        self._cache_stats()

    def _lookup_subtable(self,node,domains,boundary,check):
        # Most cold nodes never recur. Index cheap exact node/domain keys first,
        # and derive projected parent dependencies only for a plausible match.
        # Full raw boundary records remain authoritative for exact hits.
        key=(self.target,node,tuple(domains[c] for c in node),boundary)
        table=self.subtables.get(key)
        if table is not None:return key,table
        candidates=self._subtable_index.get(key[:3])
        if not candidates:return key,None
        projected=self._subtable_key(node,domains,boundary,check)
        for other in candidates:
            check();old=self._subtable_projections.get(other)
            if old is None:
                old=self._subtable_key(node,domains,other[3],check)
                self._subtable_projections[other]=old
            if old==projected:return other,self.subtables[other]
        return key,None

    def _subtable_key(self,node,domains,boundary,check):
        """Canonical exact dependencies of this completed child computation.

        Every parent-boundary cell excludes the child's used physical tiles,
        even when it does not touch this child. A nonadjacent constraint with
        a tile unavailable anywhere in the child is identically satisfied;
        otherwise only its physical inventory matters. Adjacent cells retain
        all orientation codes, preserving joint seam and inventory choices.
        The immutable problem owns topology, masks, and strand pairings.
        """
        members=set(node);possible=0
        for cell in node:possible|=_tile_choices(domains[cell])
        adjacent=[];inventory=[]
        for cell,domain in boundary:
            check()
            if self.neighbors[cell]&members:adjacent.append((cell,domain))
            else:
                tiles=_tile_choices(domain)
                if not tiles&~possible:inventory.append(tiles)
        return (self.target,node,tuple(domains[c] for c in node),tuple(adjacent),tuple(sorted(set(inventory))))

    def _frontier(self,cells):
        return tuple(sorted(eid for cell in cells for other,_,_,eid in self.adj[cell] if other not in cells))

    def _connected(self,cells):
        if not cells:return True
        seen={next(iter(cells))};queue=list(seen)
        for cell in queue:
            for other in self.neighbors[cell]&cells-seen:
                seen.add(other);queue.append(other)
        return len(seen)==len(cells)

    def _partition(self,cells,domains,check):
        """Choose the same bounded balanced cut using exact incidence masks.

        XOR marks each edge crossing a child boundary. The two child masks
        intersect exactly on their shared cut, including distinct parallel
        seams. This changes only the bookkeeping of the existing width/domain
        heuristic, not the candidate cuts or their deterministic ranking. The maintained
        growth frontier is exactly the adjacent unselected member set.
        """
        members=set(cells);cuts={};minimum=max(1,len(cells)//3)
        incident=self._incident_edges;all_cut=0
        for cell in cells:all_cut^=incident[cell]
        volumes={cell:log2(domains[cell].bit_count()) for cell in cells}
        for seed in cells:
            check();left={seed};left_cut=incident[seed]
            frontier=(self.neighbors[seed]&members)-left
            while len(left)<=len(cells)-minimum:
                right=members-left
                if len(left)>=minimum:
                    a,b=tuple(sorted(left)),tuple(sorted(right))
                    pair=min((a,b),(b,a))
                    cuts[pair]=left_cut if pair[0]==a else all_cut^left_cut
                choices=frontier
                if not choices:choices=right
                if not choices:break
                cell=min(choices,key=lambda c:((left_cut^incident[c]).bit_count(),domains[c].bit_count(),c))
                left.add(cell);left_cut^=incident[cell]
                frontier.discard(cell);frontier.update((self.neighbors[cell]&members)-left)
        def score(pair):
            left,right=map(set,pair);left_cut=cuts[pair];right_cut=all_cut^left_cut
            widths=(left_cut.bit_count(),right_cut.bit_count())
            cut=(left_cut&right_cut).bit_count()
            volume=max(sum(volumes[c] for c in part) for part in pair)
            return (not(self._connected(left) and self._connected(right)),max(widths),cut,
                    abs(len(left)-len(right)),volume,pair)
        return min(cuts,key=score)

    def _solve(self,domains,key,check):
        cells,_,boundary=key
        work={'transitions':0,'states':0,'merges':0,'joins':0,'peak_states':0,'peak_frontier_edges':0}
        decomposition=[]
        def tick(join=False):
            work['transitions']+=1
            if join:work['joins']+=1
            if work['transitions']>self.max_transitions:raise _Budget
            if work['transitions']%32==0:check()
        def insert(table,signature,support,count):
            previous=table.get(signature)
            if previous is None:
                if len(table)>=self.max_states or work['states']>=self.max_created:raise _Budget
                table[signature]=(support,count);work['states']+=1
                work['peak_states']=max(work['peak_states'],len(table))
            else:
                table[signature]=(tuple(a|b for a,b in zip(previous[0],support)),previous[1]+count)
                work['merges']+=1
        def solve_node(node):
            check();members=set(node);edges=self._frontier(members)
            work['peak_frontier_edges']=max(work['peak_frontier_edges'],len(edges))
            if self.max_frontier_edges is not None and len(edges)>self.max_frontier_edges:raise _Budget
            child_key,cached=self._lookup_subtable(node,domains,boundary,check) if node!=cells and self.subtable_cache_limit>0 else (None,None)
            if child_key is not None:
                if cached is not None:
                    check();self.subtables.move_to_end(child_key)
                    self.stats['hierarchical_subtable_hits']+=1
                    return cached,edges
                self.stats['hierarchical_subtable_misses']+=1
            def completed(table):
                check()
                if child_key is not None:self._remember_subtable(child_key,table,check)
                return table,edges
            if len(node)==1:
                cell=node[0];table={}
                for code in values(domains[cell]):
                    tick();row=self.masks[code];used=1<<(code//3)
                    if any(self.problem.neighbors[cell,s]<0 and row[s] for s in range(3)):continue
                    frontier={eid:row[side] if self.problem.edges[eid][0]==cell else reverse_mask(row[side])
                              for _,side,_,eid in self.adj[cell]}
                    if not self._outside_possible(used,frontier,boundary,members):continue
                    pairing=tuple(sorted(tuple(sorted((33*cell+a,33*cell+b))) for a,b in self.pairs[code]))
                    insert(table,(used,tuple(frontier[e] for e in edges),pairing,0),(1<<code,),1)
                return completed(table)
            left,right=self._choose_partition(node,domains,check)
            decomposition.append((node,left,right))
            left_table,left_edges=solve_node(left)
            if not left_table:return completed({})
            right_table,right_edges=solve_node(right)
            if not right_table:return completed({})
            offsets={cell:i for i,cell in enumerate(node)}
            left_positions=tuple(offsets[cell] for cell in left)
            right_positions=tuple(offsets[cell] for cell in right)
            crossing=tuple(sorted(set(left_edges)&set(right_edges)))
            li={edge:i for i,edge in enumerate(left_edges)};ri={edge:i for i,edge in enumerate(right_edges)}
            # Mask and physical-inventory indexing avoids a blind Cartesian
            # product. Generate remaining inventories (or their complements)
            # directly only when that takes fewer probes than scanning groups.
            index={};right_union=0
            for signature,payload in right_table.items():
                right_union|=signature[0]
                seam=tuple(signature[1][ri[e]] for e in crossing)
                index.setdefault(seam,{}).setdefault(signature[0],[]).append((signature,payload))
            table={}
            for (used_l,masks_l,pairs_l,cycles_l),(support_l,count_l) in left_table.items():
                check();seam=tuple(masks_l[li[e]] for e in crossing);groups=index.get(seam)
                if groups is None:continue
                available=right_union&~used_l;extra=available.bit_count()-len(right)
                if extra<0:continue
                if comb(available.bit_count(),len(right))<=len(groups):
                    if extra<=len(right):
                        inventories=(available^sum(1<<tile for tile in omitted)
                                     for omitted in combinations(tuple(values(available)),extra))
                    else:
                        inventories=(sum(1<<tile for tile in selected)
                                     for selected in combinations(tuple(values(available)),len(right)))
                else:inventories=iter(groups)
                for used_r in inventories:
                    tick()
                    if used_l&used_r:continue
                    for (used_r,masks_r,pairs_r,cycles_r),(support_r,count_r) in groups.get(used_r,()):
                        tick(join=True);used=used_l|used_r
                        frontier=dict(zip(left_edges,masks_l));frontier.update(zip(right_edges,masks_r))
                        for edge in crossing:del frontier[edge]
                        if not self._outside_possible(used,frontier,boundary,members):continue
                        mates={p:q for pair in (*pairs_l,*pairs_r) for p,q in (pair,pair[::-1])}
                        cycles=cycles_l+cycles_r
                        for edge in crossing:
                            a,sa,b,sb=self.problem.edges[edge]
                            for position in values(masks_l[li[edge]]):
                                cycles+=self._join(mates,33*a+11*sa+position,33*b+11*sb+10-position)
                        if self.target=='gold' and cycles and not(len(node)==self.problem.n and cycles==1 and not mates):continue
                        pairing=tuple(sorted((a,b) for a,b in mates.items() if a<b))
                        signature=(used,tuple(frontier[e] for e in edges),pairing,cycles)
                        support=[0]*len(node)
                        for position,value in zip(left_positions,support_l):support[position]=value
                        for position,value in zip(right_positions,support_r):support[position]=value
                        insert(table,signature,tuple(support),count_l*count_r)
            return completed(table)
        table,edges=solve_node(cells);check()
        supports=[0]*len(cells);completions=0
        for support,count in table.values():
            supports=[a|b for a,b in zip(supports,support)];completions+=count
        return {'supports':tuple(supports),'completions':completions,'summary_count':len(table),
                'summaries':tuple(sorted(table)),'table':table,'frontier_edges':edges,'decomposition':tuple(decomposition),
                **work,'complete':True,'outside_pairing_assumptions':False}

    def _completed_table(self,domains,cells,check):
        """Fetch only a complete table bound to its precise current query."""
        check();key=self._key(domains,cells);self.stats['hierarchical_checks']+=1
        result=self.cache.get(key)
        if result is None:
            if self._disk_store is not None:result=self._disk_store.load(key,check)
            try:
                if result is None:result=self._solve(domains,key,check)
            except _Budget:
                self.stats['hierarchical_budget_skips']+=1;return key,None
            if self._disk_store is not None and not result.get('persistent_cache'):
                self._disk_store.save(key,result,check)
            states=result['summary_count'];slots=states*len(cells)
            if self.cache_limit>0 and states<=self.max_cached_summaries and slots<=self.max_cached_support_slots:
                while self.cache and (len(self.cache)>=self.cache_limit or
                        self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries or
                        self._query_support_slots+self._subtable_support_slots+slots>self.max_cached_support_slots):
                    self._drop_query()
                while self.subtables and (self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries or
                        self._query_support_slots+self._subtable_support_slots+slots>self.max_cached_support_slots):
                    self._drop_subtable()
                self.cache[key]=result;self._query_cached_summaries+=states;self._query_support_slots+=slots
            self._cache_stats()
            for stat in ('states','merges','joins'):self.stats['hierarchical_'+stat]+=result[stat]
        else:
            self.stats['hierarchical_cache_hits']+=1
            self.cache[key]=self.cache.pop(key)
        return key,result

    def _publish(self,domains,cells,supports,dependencies,check,*,pair=False):
        changed=False;prefix='hierarchical_pair_' if pair else 'hierarchical_'
        for cell,support in zip(cells,supports):
            check();reduced=domains[cell]&support
            if reduced!=domains[cell]:
                self.stats[prefix+'prunes']+=domains[cell].bit_count()-reduced.bit_count()
                domains[cell]=reduced;changed=True
                if self.on_narrow is not None:self.on_narrow(cell,reduced,dependencies)
                if not reduced:
                    self.stats[prefix+'failures']+=1
                    if self.extended_regions:self.note_conflict(cells)
                    break
        return changed

    def prune_region(self,domains,cells,check=lambda:None):
        cells=tuple(sorted(set(cells)))
        if not 2<=len(cells)<=self.max_cells or not all(domains[c] for c in cells):return False
        key,result=self._completed_table(domains,cells,check)
        if result is None:return False
        self.last_region={'cells':cells,**{k:v for k,v in result.items() if k not in ('supports','summaries','table')}}
        dependencies=tuple(sorted((*cells,*(cell for cell,_ in key[2])))) if self.on_narrow is not None else None
        return self._publish(domains,cells,result['supports'],dependencies,check)

    def prune_pair(self,domains,left,right,check=lambda:None):
        """Cross-filter two disjoint complete tables, never a partial join.

        A summary has a partner exactly when physical inventories are disjoint,
        shared masks agree, and their exposed path pairing can coexist. Local
        histories with the same signature have the same future compatibility.
        Only support unions are constructed; no Cartesian parent table is kept.
        """
        self.last_pair=None
        left,right=tuple(sorted(set(left))),tuple(sorted(set(right)))
        if (not 2<=len(left)<=self.max_cells or not 2<=len(right)<=self.max_cells or
                set(left)&set(right) or not all(domains)):
            return False
        check();self.stats['hierarchical_pair_checks']+=1
        lk,lr=self._completed_table(domains,left,check)
        if lr is None:return False
        rk,rr=self._completed_table(domains,right,check)
        if rr is None:return False
        members=set(left)|set(right)
        boundary=tuple(sorted({(c,d) for c,d in (*lk[2],*rk[2]) if c not in members}))
        le,re=lr['frontier_edges'],rr['frontier_edges']
        shared=tuple(sorted(set(le)&set(re)))
        li={edge:i for i,edge in enumerate(le)};ri={edge:i for i,edge in enumerate(re)}
        supports_l=[0]*len(left);supports_r=[0]*len(right)
        supported_l=set();supported_r=set();steps=0;joins=0
        index={}
        for signature,payload in rr['table'].items():
            check();seam=tuple(signature[1][ri[e]] for e in shared)
            index.setdefault(seam,[]).append((signature,payload))
        for ls,(support_l,_) in lr['table'].items():
            check();used_l,masks_l,pairs_l,cycles_l=ls
            seam=tuple(masks_l[li[e]] for e in shared)
            for rs,(support_r,_) in index.get(seam,()):
                steps+=1
                if steps>self.max_pair_transitions:
                    self.stats['hierarchical_pair_budget_skips']+=1
                    return False
                self.stats['hierarchical_pair_comparisons']+=1
                if steps%16==0:check()
                used_r,masks_r,pairs_r,cycles_r=rs
                if used_l&used_r:continue
                joins+=1;self.stats['hierarchical_pair_joins']+=1
                frontier=dict(zip(le,masks_l));frontier.update(zip(re,masks_r))
                for edge in shared:del frontier[edge]
                if not self._outside_possible(used_l|used_r,frontier,boundary,members):continue
                if self.target=='gold':
                    mates={p:q for pair in (*pairs_l,*pairs_r) for p,q in (pair,pair[::-1])}
                    cycles=cycles_l+cycles_r
                    for edge in shared:
                        a,sa,b,sb=self.problem.edges[edge]
                        for position in values(masks_l[li[edge]]):
                            cycles+=self._join(mates,33*a+11*sa+position,33*b+11*sb+10-position)
                    if cycles and not(len(members)==self.problem.n and cycles==1 and not mates):continue
                supported_l.add(ls);supported_r.add(rs)
                supports_l=[a|b for a,b in zip(supports_l,support_l)]
                supports_r=[a|b for a,b in zip(supports_r,support_r)]
        check()
        removed=lr['summary_count']+rr['summary_count']-len(supported_l)-len(supported_r)
        self.stats['hierarchical_pair_summaries_removed']+=removed
        self.last_pair={'left':left,'right':right,'complete':True,'comparisons':steps,'joins':joins,
                        'summaries_removed':removed,'supported_summaries':(len(supported_l),len(supported_r))}
        # Dependencies include both tables' original domains and every outside
        # cell used by either local relaxation or by the combined outside test.
        dependencies=tuple(sorted(members|{c for c,_ in (*lk[2],*rk[2])})) if self.on_narrow is not None else None
        changed=self._publish(domains,left,supports_l,dependencies,check,pair=True)
        if all(domains):changed|=self._publish(domains,right,supports_r,dependencies,check,pair=True)
        return changed

    def prune(self,domains,check=lambda:None):
        if not all(domains):return False
        regions=self._regions(domains,check);changed=False
        if self.extended_regions:
            selected=[];claimed=set()
            for region in regions:
                if not set(region)&claimed:
                    selected.append(region);claimed.update(region)
                    if len(selected)>=self._network.max_tables:break
            if selected:
                changed|=self.prune_network(domains,selected,check)
                if not all(domains):return changed
                if len(selected)>=4:
                    changed|=self.prune_components(domains,selected,check)
                    if not all(domains):return changed
            # Exact assignments, not per-cell support unions, justify overlaps.
            for left,right in combinations(regions[:12],2):
                overlap=set(left)&set(right)
                if overlap and set(left)-overlap and set(right)-overlap:
                    changed|=self.prune_overlap(domains,left,right,check)
                    break
            if not all(domains):return changed

        for region in regions[:self.max_queries]:
            changed|=self.prune_region(domains,region,check)
            if not all(domains):return changed
        # A disjoint partner is selected independently of max_queries: the
        # default single-region budget must not silently disable cross-filtering.
        pairs=0
        minimum=min(map(len,regions),default=self.problem.n)
        candidates=[r for r in regions if len(r)+minimum<=self.problem.n][:self.max_pair_candidates]
        for i,left in enumerate(candidates):
            for right in candidates[i+1:]:
                check()
                if pairs>=self.max_pair_queries:return changed
                if set(left)&set(right):continue
                pairs+=1;changed|=self.prune_pair(domains,left,right,check)
                if not all(domains):return changed
        return changed

    def _regions(self,domains,check):
        eligible={c for c,d in enumerate(domains) if 0<d.bit_count()<=self.domain_cap}
        regions=set();faces={}
        for cell in self.problem.board.cells:
            if hasattr(cell,'face'):faces.setdefault(cell.face,[]).append(cell.index)
        for face in faces.values():
            if 9<=len(face)<=self.max_cells and set(face)<=eligible and any(domains[c].bit_count()>1 for c in face):
                regions.add(tuple(sorted(face)))
        for first in sorted(c for c in eligible if domains[c].bit_count()>1):
            check();queue=deque([first]);seen={first};region=[]
            while queue and len(region)<self.max_cells:
                cell=queue.popleft();region.append(cell)
                if len(region) in ((4,6,9,12,16,24,32,48,64,96,128,160) if self.extended_regions else (9,12,16)):
                    if self.max_frontier_edges is None or len(self._frontier(set(region)))<=self.max_frontier_edges:
                        regions.add(tuple(sorted(region)))
                for other in sorted(self.neighbors[cell]&eligible):
                    if other not in seen:seen.add(other);queue.append(other)
        if 9<=self.problem.n<=self.max_cells and len(eligible)==self.problem.n:
            regions.add(tuple(range(self.problem.n)))
        return sorted(regions,key=lambda r:(-sum(self._region_activity.get(c,0) for c in r) if self.extended_regions else 0,
                                            sum(log2(domains[c].bit_count()) for c in r)/len(r),
                                            len(self._frontier(set(r))),-len(r),r))
