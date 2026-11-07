"""Reversible restriction of complete exact regional DAGs (round25 #14).

A compiled envelope is reusable only inside its original cell domains and with
identical outside assumptions. Counts are recomputed from changed levels toward
the root; widening within the envelope restores edges without recompilation.
Incomplete compilers and interrupted restrictions never publish supports.
"""
from collections import OrderedDict,defaultdict
from collections.abc import Mapping
import heapq
from constraint_region_symbolic import DirectRegionDiagram,DirectRegionSession
from constraint_separator import _Budget

CURSOR_STAT_KEYS=('region_cursor_loads','region_cursor_writes','region_cursor_rejections',
 'region_cursor_write_failures','region_cursor_saved_bytes','region_cursor_loaded_ticks',
 'region_cursor_loaded_transitions','region_cursor_no_directory','region_cursor_limited_hits','region_cursor_store_caps')

STAT_KEYS=('region_restrict_compilations','region_restrict_compile_states',
 'region_restrict_compile_transitions','region_restrict_reuses','region_restrict_index_edges',
 'region_restrict_node_updates','region_restrict_update_edges','region_restrict_query_edges',
 'region_restrict_copied_nodes','region_restrict_completed','region_restrict_caps',
 'region_restrict_prunes','region_restrict_binding_checks') + CURSOR_STAT_KEYS


def source_binding(owner):
    problem=owner.problem
    return (problem.fingerprint,owner.target,tuple(tuple(row) for row in problem.domains),
            tuple(problem.edges),tuple(tuple(map(int,row)) for row in problem.masks),
            tuple(tuple(tuple(tuple(endpoint) for endpoint in pair) for pair in tile['segments'])
                  for tile in problem.tiles))


class _RestrictedNodes(Mapping):
    def __init__(self,base,masks,counts,stats):
        self.base,self.masks,self.counts,self.stats=base,masks,counts,stats
    def __iter__(self):return iter(self.base)
    def __len__(self):return len(self.base)
    def __getitem__(self,node):
        level,edges=self.base[node];result=[]
        for code,child in edges:
            self.stats['region_restrict_query_edges']+=1
            if self.masks[level]>>code&1 and self.counts[child]:result.append((code,child))
        return level,tuple(result)


class IncrementalDiagram:
    def __init__(self,diagram,domains,stats,check=lambda:None):
        self.base=diagram;self.cells=diagram.cells;self.envelope=tuple(domains)
        self.domains=self.envelope;self.stats=stats;self.counts=dict(diagram._counts)
        self.by_level=defaultdict(list);self.parents=defaultdict(set)
        for node,(level,edges) in diagram.nodes.items():
            check();self.by_level[level].append(node)
            for code,child in edges:
                self.parents[child].add(node);stats['region_restrict_index_edges']+=1
        check()
    def covers(self,cells,domains):
        if tuple(cells)!=self.cells:return False
        members=set(self.cells)
        return all(not new&~old if cell in members else new==old
                   for cell,(new,old) in enumerate(zip(domains,self.envelope)))
    def restrict(self,domains,check=lambda:None):
        domains=tuple(domains)
        if len(domains)!=len(self.envelope) or not self.covers(self.cells,domains):
            raise ValueError('Restriction escapes the compiled envelope or changes outside assumptions.')
        check();changed=[level for level,cell in enumerate(self.cells) if domains[cell]!=self.domains[cell]]
        if changed:
            counts=dict(self.counts);self.stats['region_restrict_copied_nodes']+=len(counts)
            pending=[];scheduled=set()
            def add(node):
                if node not in scheduled:
                    scheduled.add(node);heapq.heappush(pending,(-self.base.nodes[node][0],node))
            for level in changed:
                for node in self.by_level[level]:add(node)
            while pending:
                check();_,node=heapq.heappop(pending);level,edges=self.base.nodes[node];total=0
                for code,child in edges:
                    self.stats['region_restrict_update_edges']+=1
                    if domains[self.cells[level]]>>code&1:total+=counts[child]
                self.stats['region_restrict_node_updates']+=1
                if total!=counts[node]:
                    counts[node]=total
                    for parent in self.parents[node]:add(parent)
            check();self.domains=domains;self.counts=counts
        # A view holds immutable references. A later sibling restriction cannot
        # alter previously returned histories, supports or overlap-join inputs.
        result=object.__new__(DirectRegionDiagram)
        result.__dict__.update(self.base.__dict__)
        result._counts=self.counts
        result.nodes=_RestrictedNodes(self.base.nodes,tuple(domains[c] for c in self.cells),self.counts,self.stats)
        result.root=self.base.root if self.base.root is not None and self.counts[self.base.root] else None
        result.rows=0 if result.root is None else self.counts[result.root]
        return result


class IncrementalRegionCache:
    def __init__(self,owner,*,limit=4,durable=False):
        if type(limit)is not int or not 1<=limit<=16:raise ValueError('Invalid incremental region cache bound.')
        self.owner,self.limit=owner,limit;self.binding=source_binding(owner)
        self.stats=dict.fromkeys(STAT_KEYS,0);self.diagrams=OrderedDict();self.pending=OrderedDict()
        if type(durable)is not bool:raise ValueError('Durable regional policy must be boolean.')
        self.durable=durable;self.store=None;self._store_entry=dict.fromkeys(CURSOR_STAT_KEYS,0);self._last_saved={}
    def _sync_cursor_stats(self):
        if self.store is not None:
            for key,value in self.store.stats.items():
                self.stats[key]+=value-self._store_entry[key];self._store_entry[key]=value
    def configure_cache_dir(self,directory):
        if not self.durable:return
        from pathlib import Path
        from constraint_region_cursor import LocalRegionCursorStore
        if self.store is not None and directory is not None and self.store.directory==Path(directory)/'regional-cursors-v1':return
        self._sync_cursor_stats();self.store=None if directory is None else LocalRegionCursorStore(directory,self.owner)
        self._store_entry=dict.fromkeys(CURSOR_STAT_KEYS,0);self._last_saved={}
    def _save_cursor(self,key,session):
        if self.store is None or getattr(session,'broken',False) or self._last_saved.get(key)==session.ticks:return
        if self.store.save(key,session):self._last_saved[key]=session.ticks
        self._sync_cursor_stats()
    def get(self,cells,domains,check=lambda:None,*,max_rows=4096,max_steps=50000):
        self.stats['region_restrict_binding_checks']+=1
        if source_binding(self.owner)!=self.binding:raise ValueError('Incremental regional source binding changed.')
        cells=tuple(sorted(set(cells)));domains=tuple(domains)
        problem=self.owner.problem
        if (len(domains)!=problem.n or not cells or any(type(c)is not int or not 0<=c<problem.n for c in cells)
                or any(type(mask)is not int or mask<0 or mask&~sum(1<<c for c in set(problem.domains[i]))
                       for i,mask in enumerate(domains))):raise ValueError('Invalid regional query domains.')
        if any(type(value)is not int or value<0 for value in (max_rows,max_steps)):
            raise ValueError('Invalid regional compilation bound.')
        check()
        for key,diagram in tuple(reversed(self.diagrams.items())):
            if diagram.covers(cells,domains):
                result=diagram.restrict(domains,check);self.diagrams.move_to_end(key)
                self.stats['region_restrict_reuses']+=1;self.stats['region_restrict_completed']+=1
                return result
        key=(cells,domains,max_rows,max_steps);session=self.pending.get(key)
        if session is None:
            if self.durable:
                from constraint_region_cursor import PortableRegionSession
                if self.store is not None:
                    try:session=self.store.load(key,check)
                    finally:self._sync_cursor_stats()
                else:self.stats['region_cursor_no_directory']+=1
                if session is not None:self._last_saved[key]=session.ticks
                else:session=PortableRegionSession(self.owner,cells,domains,max_rows,max_steps);self.stats['region_restrict_compilations']+=1
            else:session=DirectRegionSession(self.owner,cells,domains,max_rows,max_steps);self.stats['region_restrict_compilations']+=1
            self.pending[key]=session
            while len(self.pending)>self.limit:
                dropped,_=self.pending.popitem(last=False);self._last_saved.pop(dropped,None)
        self.pending.move_to_end(key);before=(session.states,session.steps)
        try:
            while not session.advance(256,check):
                if self.durable and session.ticks-self._last_saved.get(key,0)>=4096:self._save_cursor(key,session)
            if self.durable:self._save_cursor(key,session)
            base=object.__new__(DirectRegionDiagram);base.__dict__.update(session.result())
            indexed=IncrementalDiagram(base,domains,self.stats,check)
            result=indexed.restrict(domains,check)
        except _Budget:
            if self.durable:self._save_cursor(key,session)
            self.pending.pop(key,None);self._last_saved.pop(key,None);self.stats['region_restrict_caps']+=1;raise
        except BaseException:
            if self.durable:
                if getattr(session,'broken',False):self.pending.pop(key,None);self._last_saved.pop(key,None)
                else:self._save_cursor(key,session)
            raise
        finally:
            self.stats['region_restrict_compile_states']+=session.states-before[0]
            self.stats['region_restrict_compile_transitions']+=session.steps-before[1]
        self.pending.pop(key,None);self._last_saved.pop(key,None);self.diagrams[key]=indexed
        while len(self.diagrams)>self.limit:self.diagrams.popitem(last=False)
        self.stats['region_restrict_completed']+=1
        return result
    def prune(self,cells,domains,check=lambda:None,*,max_rows=4096,max_steps=50000):
        try:diagram=self.get(cells,domains,check,max_rows=max_rows,max_steps=max_steps)
        except _Budget:return False
        supports=diagram.supports(check);current=list(domains)
        for cell,mask in zip(diagram.cells,supports):current[cell]&=mask
        check();self.stats['region_restrict_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
