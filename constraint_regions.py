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
           'hierarchical_cached_summaries')


class HierarchicalRegionPruner(SeparatorPruner):
    def __init__(self,problem,target='gold'):
        super().__init__(problem,target)
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.max_cells=16
        self.max_states=4096
        self.max_transitions=200000
        self.max_created=20000
        self.max_queries=1
        self.domain_cap=6
        self.cache_limit=32
        self.max_cached_summaries=8192
        self.subtable_cache_limit=128
        self.max_cached_support_slots=65536
        self.subtables=OrderedDict()
        self._subtable_index={}
        self._subtable_projections={}
        self._query_cached_summaries=0
        self._subtable_states=0
        self._subtable_support_slots=0
        # Called immediately after publishing a proven reduction. The callback
        # receives current-query dependency cells. Cancellation preserves the
        # sound, already narrowed domain and lets its owner reconcile reasons.
        self.on_narrow=None

    def _cache_stats(self):
        self.stats['hierarchical_cache_entries']=len(self.cache)
        self.stats['hierarchical_subtable_entries']=len(self.subtables)
        self.stats['hierarchical_subtable_states']=self._subtable_states
        self.stats['hierarchical_subtable_support_slots']=self._subtable_support_slots
        self.stats['hierarchical_cached_summaries']=self._query_cached_summaries+self._subtable_states

    def _drop_subtable(self):
        key,table=self.subtables.popitem(last=False)
        bucket=self._subtable_index[key[:3]];bucket.remove(key)
        if not bucket:del self._subtable_index[key[:3]]
        self._subtable_projections.pop(key,None)
        self._subtable_states-=len(table)
        self._subtable_support_slots-=len(table)*len(key[1])
        self.stats['hierarchical_subtable_evictions']+=1

    def _drop_query(self):
        self._query_cached_summaries-=self.cache.pop(next(iter(self.cache)))['summary_count']

    def _remember_subtable(self,key,table,check):
        # A completed child is reusable even when a later parent join stops.
        # Resource limits do not change table semantics or justify truncation.
        check();states=len(table);slots=states*len(key[1])
        if (self.subtable_cache_limit<=0 or states>self.max_cached_summaries
                or slots>self.max_cached_support_slots):return
        while self.subtables and (len(self.subtables)>=self.subtable_cache_limit or
                self._subtable_support_slots+slots>self.max_cached_support_slots or
                self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries):
            self._drop_subtable()
        while self.cache and self._query_cached_summaries+self._subtable_states+states>self.max_cached_summaries:
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
        """Choose a bounded balanced cut minimizing child frontier width.

        Greedy connected growth from every seed proposes cuts. This is not an
        optimal separator claim. Disconnected complements remain legal but are
        ranked after connected splits; table semantics do not assume connected
        children. Inventory/domain volume breaks equal-width ties.
        """
        members=set(cells);cuts=set();minimum=max(1,len(cells)//3)
        for seed in cells:
            check();left={seed}
            while len(left)<=len(cells)-minimum:
                right=members-left
                if len(left)>=minimum:
                    a,b=tuple(sorted(left)),tuple(sorted(right))
                    cuts.add(min((a,b),(b,a)))
                choices=(set().union(*(self.neighbors[c] for c in left))&members)-left
                if not choices:choices=right
                if not choices:break
                cell=min(choices,key=lambda c:(len(self._frontier(left|{c})),domains[c].bit_count(),c))
                left.add(cell)
        def score(pair):
            left,right=map(set,pair)
            widths=(len(self._frontier(left)),len(self._frontier(right)))
            cut=sum(a in left and b in right or b in left and a in right for a,_,b,_ in self.problem.edges)
            volume=max(sum(log2(domains[c].bit_count()) for c in part) for part in pair)
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
            left,right=self._partition(node,domains,check)
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
                'summaries':tuple(sorted(table)),'frontier_edges':edges,'decomposition':tuple(decomposition),
                **work,'complete':True,'outside_pairing_assumptions':False}

    def prune_region(self,domains,cells,check=lambda:None):
        cells=tuple(sorted(set(cells)))
        if not 2<=len(cells)<=self.max_cells or not all(domains[c] for c in cells):return False
        check();key=self._key(domains,cells);self.stats['hierarchical_checks']+=1
        result=self.cache.get(key)
        if result is None:
            try:result=self._solve(domains,key,check)
            except _Budget:
                self.stats['hierarchical_budget_skips']+=1;return False
            if self.cache_limit>0 and result['summary_count']<=self.max_cached_summaries:
                while self.cache and (len(self.cache)>=self.cache_limit or
                        self._query_cached_summaries+self._subtable_states+result['summary_count']>self.max_cached_summaries):
                    self._drop_query()
                while self.subtables and self._query_cached_summaries+self._subtable_states+result['summary_count']>self.max_cached_summaries:
                    self._drop_subtable()
                self.cache[key]=result;self._query_cached_summaries+=result['summary_count']
            self._cache_stats()
            for stat in ('states','merges','joins'):self.stats['hierarchical_'+stat]+=result[stat]
        else:
            self.stats['hierarchical_cache_hits']+=1
            self.cache[key]=self.cache.pop(key)  # Refresh bounded query LRU.
        self.last_region={'cells':cells,**{k:v for k,v in result.items() if k not in ('supports','summaries')}}
        changed=False
        dependencies=tuple(sorted((*cells,*(cell for cell,_ in key[2])))) if self.on_narrow is not None else None
        for cell,support in zip(cells,result['supports']):
            check();reduced=domains[cell]&support
            if reduced!=domains[cell]:
                self.stats['hierarchical_prunes']+=domains[cell].bit_count()-reduced.bit_count()
                domains[cell]=reduced;changed=True
                if self.on_narrow is not None:self.on_narrow(cell,reduced,dependencies)
                if not reduced:self.stats['hierarchical_failures']+=1;break
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
                if len(region) in (9,12,16):regions.add(tuple(sorted(region)))
                for other in sorted(self.neighbors[cell]&eligible):
                    if other not in seen:seen.add(other);queue.append(other)
        if 9<=self.problem.n<=self.max_cells and len(eligible)==self.problem.n:
            regions.add(tuple(range(self.problem.n)))
        return sorted(regions,key=lambda r:(sum(log2(domains[c].bit_count()) for c in r)/len(r),
                                            len(self._frontier(set(r))),-len(r),r))
