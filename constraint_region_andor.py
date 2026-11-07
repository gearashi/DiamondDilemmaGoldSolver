"""Automatic bounded separator AND/OR search with exact Gold interfaces (#18).

Component relations are compiled under the original outside-domain envelope,
then joined with all conditioned/fixed cells. They retain inventories, seam
masks and strand pairings. Boolean independence of Gold components is never
assumed. Only full valid witnesses or completed negative searches are published.
"""
from collections import OrderedDict
from itertools import combinations,product
from constraint_region_incremental import source_binding
from constraint_region_symbolic import DirectRegionSession,DirectRegionDiagram
from constraint_region_network import assignment_signature,joined_signature
from constraint_pruning import values
from constraint_separator import _Budget

STAT_KEYS=('andor_queries','andor_completed','andor_unknown','andor_admission_skips',
 'andor_cache_hits','andor_plan_tests','andor_condition_cases','andor_component_builds',
 'andor_component_reuses','andor_component_states','andor_component_transitions',
 'andor_relation_entries','andor_interface_joins','andor_work_ticks','andor_failures','andor_validations')


class AndOrLimit(Exception):pass


class _Work:
    def __init__(self,limit,check,stats):self.limit,self.check,self.stats=limit,check,stats;self.steps=0
    def tick(self):
        self.check();self.steps+=1;self.stats['andor_work_ticks']+=1
        if self.steps>self.limit:raise AndOrLimit


class SeparatorSearch:
    def __init__(self,owner,*,reuse_components=True,limit=8):
        if type(reuse_components)is not bool or type(limit)is not int or not 1<=limit<=32:raise ValueError('Invalid separator search policy.')
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.reuse_components=reuse_components;self.limit=limit
        self.original=tuple(sum(1<<code for code in set(row)) for row in self.problem.domains)
        self.cache=OrderedDict();self.relations=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0);self.solution=None
    @staticmethod
    def _components(graph,removed,work):
        pending=set(graph)-set(removed);result=[]
        while pending:
            work.tick();start=min(pending);pending.remove(start);group=[start];queue=[start]
            while queue:
                work.tick();cell=queue.pop()
                for other in sorted(graph[cell]&pending):pending.remove(other);queue.append(other);group.append(other)
            result.append(tuple(sorted(group)))
        return tuple(sorted(result))
    def _plan(self,domains,work,max_active,max_separator,max_cases):
        active=tuple(c for c,mask in enumerate(domains) if mask.bit_count()>1)
        if len(active)>max_active or any(domains[c].bit_count()>8 for c in active):return None
        resources={c:{code//3 for code in values(domains[c])} for c in active};graph={c:set() for c in active}
        for a,b in combinations(active,2):
            work.tick()
            if b in self.owner.neighbors[a] or resources[a]&resources[b]:graph[a].add(b);graph[b].add(a)
        best=None
        for size in range(max_separator+1):
            for separator in combinations(active,size):
                work.tick();self.stats['andor_plan_tests']+=1
                cases=1
                for cell in separator:cases*=domains[cell].bit_count()
                if cases>max_cases:continue
                components=self._components(graph,separator,work)
                if len(components)<2:continue
                score=(max(map(len,components)),sum(len(c)**2 for c in components),cases,separator)
                if best is None or score<best[0]:best=(score,separator,components)
            if best is not None and size==0:break
        return None if best is None else best[1:]
    def _relation(self,cells,domains,work,max_states,max_entries):
        key=(cells,tuple(domains[c] for c in cells),max_states,max_entries)
        if self.reuse_components and key in self.relations:
            self.stats['andor_component_reuses']+=1;self.relations.move_to_end(key);return self.relations[key]
        # This fixed broad outside envelope is deliberately independent of the
        # separator assignment. Actual interfaces are checked in the full join.
        envelope=list(self.original)
        for cell in cells:envelope[cell]=domains[cell]
        session=DirectRegionSession(self.owner,cells,envelope,max_states,work.limit)
        self.stats['andor_component_builds']+=1
        try:
            while not session.advance(128,work.tick):pass
        finally:
            self.stats['andor_component_states']+=session.states;self.stats['andor_component_transitions']+=session.steps
        diagram=object.__new__(DirectRegionDiagram);diagram.__dict__.update(session.result());memo={};entries=0
        def visit(node):
            nonlocal entries
            work.tick()
            if node in memo:return memo[node]
            if node in diagram.terminals:return {diagram.terminals[node]:((0,)*len(cells),1,())}
            level,edges=diagram.nodes[node];result={}
            for code,child in edges:
                for signature,(support,count,witness) in visit(child).items():
                    work.tick();candidate=list(support);candidate[level]|=1<<code
                    if signature in result:
                        old,number,saved=result[signature];result[signature]=(tuple(a|b for a,b in zip(old,candidate)),number+count,saved)
                    else:
                        entries+=1;self.stats['andor_relation_entries']+=1
                        if entries>max_entries:raise AndOrLimit
                        result[signature]=(tuple(candidate),count,(code,*witness))
            memo[node]=result;return result
        relation={} if diagram.root is None else visit(diagram.root)
        work.tick()
        if self.reuse_components:
            self.relations[key]=relation
            while len(self.relations)>self.limit:self.relations.popitem(last=False)
        return relation
    def _run(self,domains,work,max_active,max_separator,max_cases,max_states,max_entries):
        active=[c for c,mask in enumerate(domains) if mask.bit_count()>1]
        if not active:
            codes=tuple(mask.bit_length()-1 for mask in domains);report=self.problem.validate(codes)
            accepted=report['valid'] if self.owner.target=='gold' else report['unique_tiles'] and report['matched_edges']==len(self.problem.edges)
            return {'possible':bool(accepted),'witness':codes if accepted else None}
        plan=self._plan(domains,work,max_active,max_separator,max_cases)
        if plan is None:self.stats['andor_admission_skips']+=1;return None
        separator,components=plan
        for chosen in product(*(tuple(values(domains[cell])) for cell in separator)):
            work.tick();self.stats['andor_condition_cases']+=1
            current=list(domains)
            for cell,code in zip(separator,chosen):current[cell]=1<<code
            fixed=tuple(c for c,mask in enumerate(current) if mask.bit_count()==1)
            fixed_codes=tuple(current[c].bit_length()-1 for c in fixed)
            if len({code//3 for code in fixed_codes})!=len(fixed):continue
            used=sum(1<<(code//3) for code in fixed_codes);forbidden=sum(7<<(3*tile) for tile in values(used))
            for cell in range(self.problem.n):
                if cell not in fixed:current[cell]&=~forbidden
            if not all(current):continue
            # Original graph components remain a valid partition after further
            # narrowing; extra splits are optional optimization, not necessity.
            fixed_signature=assignment_signature(self.owner,fixed,fixed_codes,current,work.tick) if fixed else None
            if fixed and fixed_signature is None:continue
            tables=[self._relation(cells,current,work,max_states,max_entries) for cells in components]
            if any(not table for table in tables):continue
            def join(index,members,signature,assigned):
                work.tick()
                if index==len(components):
                    codes=tuple(assigned[c] for c in range(self.problem.n));report=self.problem.validate(codes)
                    self.stats['andor_validations']+=1
                    accepted=report['valid'] if self.owner.target=='gold' else report['unique_tiles'] and report['matched_edges']==len(self.problem.edges)
                    if not accepted:raise RuntimeError('Exact separator interfaces produced an invalid full witness.')
                    return codes
                cells=components[index]
                for selected,(_,count,witness) in tables[index].items():
                    work.tick();self.stats['andor_interface_joins']+=1
                    merged=selected if not members else joined_signature(self.owner,members,signature,cells,selected,current,work.tick)
                    if merged is None:continue
                    found=join(index+1,tuple(sorted((*members,*cells))),merged,{**assigned,**dict(zip(cells,witness))})
                    if found is not None:return found
                return None
            found=join(0,fixed,fixed_signature,dict(zip(fixed,fixed_codes)))
            if found is not None:return {'possible':True,'witness':found}
        return {'possible':False,'witness':None}
    def prune(self,domains,check=lambda:None,*,max_active=24,max_separator=2,max_cases=64,max_states=4096,max_entries=8192,max_steps=50000):
        self.solution=None
        if source_binding(self.owner)!=self.binding:raise ValueError('Separator search source binding changed.')
        limits=(max_active,max_separator,max_cases,max_states,max_entries,max_steps)
        if any(type(bound)is not int or bound<0 for bound in limits) or max_separator>2:raise ValueError('Invalid separator search bounds.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Separator search domains escape the original root.')
        if not all(domains):return False
        check();key=(tuple(domains),limits)
        if key in self.cache:
            self.stats['andor_cache_hits']+=1;result=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['andor_queries']+=1;work=_Work(max_steps,check,self.stats)
            try:result=self._run(domains,work,max_active,max_separator,max_cases,max_states,max_entries)
            except (AndOrLimit,_Budget):result=None
            check();self.cache[key]=result
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
            self.stats['andor_unknown' if result is None else 'andor_completed']+=1
        if result is None:return False
        check()
        if result['possible']:
            if self.owner.target=='gold':self.solution=result['witness']
            return False
        domains[0]=0;self.stats['andor_failures']+=1;return True
