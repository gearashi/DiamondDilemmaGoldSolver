"""Counterexample-guided selective factor DD refinement (round25 #19).

Union merges are optimistic. Concrete spurious completions identify variables
whose values must stop merging. Layers before the first newly protected variable
remain identical and are reused. Only empty overapproximations prove negatives;
positive answers require an actual finite-factor witness.
"""
from collections import OrderedDict,Counter
from constraint_region_elimination import Factor,problem_factors
from constraint_region_incremental import source_binding
from constraint_pruning import values
from constraint_separator import _Budget

STAT_KEYS=('selective_dd_queries','selective_dd_passes','selective_dd_refinements',
 'selective_dd_transitions','selective_dd_merges','selective_dd_reused_layers',
 'selective_dd_factor_rows','selective_dd_factor_cache_hits','selective_dd_witness_checks',
 'selective_dd_unknown','selective_dd_unsat','selective_dd_witnesses','selective_dd_prunes',
 'selective_dd_model_builds','selective_dd_model_rows','selective_dd_model_caps',
 'selective_dd_query_cache_hits','selective_dd_full_queries','selective_dd_regional_queries')

class RefinementLimit(Exception):pass

class _Work:
    def __init__(self,limit,check):self.limit,self.check=limit,check;self.steps=0
    def tick(self):
        self.check();self.steps+=1
        if self.steps>self.limit:raise RefinementLimit


class SelectiveDiagram:
    def __init__(self,factors,choices,*,stats=None):
        if not isinstance(choices,dict) or any(type(v)is not int or v<0 for v in choices):raise ValueError('Invalid exact diagram variables.')
        self.variables=tuple(sorted(choices));self.index={v:i for i,v in enumerate(self.variables)}
        self.choices=self._choices(choices);self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        self.factors=[]
        for factor in factors:
            scope=tuple(factor.scope)
            if len(scope)!=len(set(scope)) or any(type(v)is not int or v not in self.index for v in scope):raise ValueError('Invalid exact factor scope.')
            rows=[]
            for row in factor.rows:
                if len(row)!=len(scope) or any(type(code)is not int or code<0 for code in row):raise ValueError('Invalid exact factor row.')
                rows.append(tuple(row));self.stats['selective_dd_model_rows']+=1
            self.factors.append((tuple(self.index[v] for v in scope),frozenset(rows),tuple(sorted(set(rows)))))
        self.factors=tuple(self.factors);self.support_cache=OrderedDict()
    def _choices(self,choices):
        if set(choices)!=set(self.variables):raise ValueError('Diagram variable set differs.')
        result=[]
        for v in self.variables:
            row=choices[v]
            if not isinstance(row,(tuple,list,set,frozenset)) or any(type(c)is not int or c<0 for c in row):raise ValueError('Invalid exact diagram choices.')
            result.append(frozenset(row))
        return tuple(result)
    def _supported(self,state,work):
        if any(not row for row in state):return False
        for number,(scope,rows,ordered) in enumerate(self.factors):
            key=(number,tuple(state[i] for i in scope))
            if key in self.support_cache:
                self.stats['selective_dd_factor_cache_hits']+=1;possible=self.support_cache[key]
            else:
                possible=False
                for row in ordered:
                    work.tick();self.stats['selective_dd_factor_rows']+=1
                    if all(code in state[i] for i,code in zip(scope,row)):possible=True;break
                self.support_cache[key]=possible
                while len(self.support_cache)>8192:self.support_cache.popitem(last=False)
            if not possible:return False
        return True
    def possible(self,choices=None,check=lambda:None,*,width=1,max_states=64,max_refinements=8,max_steps=50000,reuse_prefix=True,work=None):
        for bound,minimum in ((width,1),(max_states,1),(max_refinements,0),(max_steps,0)):
            if type(bound)is not int or bound<minimum:raise ValueError('Invalid selective diagram bound.')
        if type(reuse_prefix)is not bool:raise ValueError('Prefix reuse must be boolean.')
        root=self.choices if choices is None else self._choices(choices)
        if any(not row<=old for row,old in zip(root,self.choices)):raise ValueError('Diagram query escaped its compiled factor choices.')
        work=work or _Work(max_steps,check);self.stats['selective_dd_queries']+=1
        protected=set();layers=[(root,)];start=0;refinements=0
        order_key=lambda state:tuple(tuple(sorted(row)) for row in state)
        try:
            work.tick()
            if not self._supported(root,work):self.stats['selective_dd_unsat']+=1;return False,None
            while True:
                self.stats['selective_dd_passes']+=1
                for index in range(start,len(self.variables)):
                    following=set()
                    for state in layers[-1]:
                        for code in sorted(state[index]):
                            work.tick();self.stats['selective_dd_transitions']+=1
                            child=list(state);child[index]=frozenset((code,));child=tuple(child)
                            if self._supported(child,work):following.add(child)
                            if len(following)>4096:raise RefinementLimit
                    if not following:self.stats['selective_dd_unsat']+=1;return False,None
                    groups={}
                    for state in sorted(following,key=order_key):
                        signature=tuple(state[p] for p in sorted(protected));groups.setdefault(signature,[]).append(state)
                    merged=[]
                    for group in groups.values():
                        if len(group)<=width:merged.extend(group);continue
                        for part in (group[k::width] for k in range(width)):
                            work.tick();merged.append(tuple(frozenset().union(*(state[j] for state in part)) for j in range(len(root))))
                        self.stats['selective_dd_merges']+=len(group)-width
                    if len(merged)>max_states:raise RefinementLimit
                    layers.append(tuple(sorted(set(merged),key=order_key)))
                bad=Counter()
                for state in layers[-1]:
                    witness=tuple(min(row) for row in state);violated=[]
                    for scope,rows,ordered in self.factors:
                        work.tick();self.stats['selective_dd_witness_checks']+=1
                        if tuple(witness[i] for i in scope) not in rows:violated.append(scope)
                    if not violated:
                        self.stats['selective_dd_witnesses']+=1
                        return True,dict(zip(self.variables,witness))
                    for scope in violated:
                        for index in scope:
                            if index not in protected and len(state[index])>1:bad[index]+=1
                if refinements>=max_refinements:raise RefinementLimit
                if not bad:raise RuntimeError('A spurious diagram witness has no merged coordinate.')
                selected=min(bad,key=lambda index:(-bad[index],len(root[index]),index));protected.add(selected)
                refinements+=1;self.stats['selective_dd_refinements']+=1
                start=selected if reuse_prefix else 0
                self.stats['selective_dd_reused_layers']+=start
                layers=layers[:start+1]
        except RefinementLimit:
            self.stats['selective_dd_unknown']+=1;return None,None


def project_fixed(factors,choices,check=lambda:None):
    fixed={v:row[0] for v,row in choices.items() if len(row)==1};output=[]
    for factor in factors:
        check();keep=tuple(i for i,v in enumerate(factor.scope) if v not in fixed);rows=set()
        for row in factor.rows:
            check()
            if all(v not in fixed or row[i]==fixed[v] for i,v in enumerate(factor.scope)):
                rows.add(tuple(row[i] for i in keep))
        scope=tuple(factor.scope[i] for i in keep)
        if scope or not rows:output.append(Factor(scope,frozenset(rows)))
    return tuple(output),{v:row for v,row in choices.items() if v not in fixed}


class SelectiveRefinementPruner:
    def __init__(self,owner,*,limit=8):
        if type(limit)is not int or limit<1:raise ValueError('Selective refinement cache limit must be positive.')
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner)
        self.cache=OrderedDict();self.limit=limit;self.stats=dict.fromkeys(STAT_KEYS,0)
    def prune(self,domains,check=lambda:None,*,max_candidates=16,max_steps=50000,max_states=64,max_refinements=8):
        if source_binding(self.owner)!=self.binding:raise ValueError('Selective refinement source binding changed.')
        if any(type(value)is not int or value<minimum for value,minimum in ((max_candidates,0),(max_steps,0),(max_states,1),(max_refinements,0))):raise ValueError('Invalid selective refinement budget.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~sum(1<<code for code in set(self.problem.domains[c])) for c,mask in enumerate(domains)):
            raise ValueError('Selective refinement domains escape the original root.')
        if not all(domains):return False
        full=self.problem.n<=20 or sum(mask.bit_count()>1 for mask in domains)<=8
        cells=tuple(range(self.problem.n)) if full else tuple(sorted((c for c,mask in enumerate(domains) if 1<mask.bit_count()<=8),key=lambda c:(domains[c].bit_count(),c))[:8])
        if not cells:return False
        limits=(max_candidates,max_steps,max_states,max_refinements);key=(cells,tuple(domains[c] for c in cells),limits)
        check()
        if key in self.cache:self.stats['selective_dd_query_cache_hits']+=1;supports=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['selective_dd_model_builds']+=1;self.stats['selective_dd_full_queries' if full else 'selective_dd_regional_queries']+=1
            current=list(domains);work=_Work(max_steps,check)
            try:
                factors,choices=problem_factors(self.problem,domains,cells,check,max_entries=max_steps)
                factors,choices=project_fixed(factors,choices,work.tick)
                model=SelectiveDiagram(factors,choices,stats=self.stats)
                possible,_=model.possible(check=check,max_states=max_states,max_refinements=max_refinements,max_steps=max_steps,work=work)
                if possible is False:current[cells[0]]=0
                elif work.steps<=max_steps:
                    candidates=[(c,code) for c in choices for code in choices[c]]
                    for cell,code in candidates[:max_candidates]:
                        check()
                        if not all(current) or work.steps>max_steps:break
                        if not current[cell]>>code&1:continue
                        trial={c:tuple(values(current[c])) for c in choices};trial[cell]=(code,)
                        answer,_=model.possible(trial,check,max_states=max_states,max_refinements=max_refinements,max_steps=max_steps,work=work)
                        if answer is False:current[cell]&=~(1<<code)
            except (_Budget,RefinementLimit):self.stats['selective_dd_model_caps']+=1
            supports=tuple(current[c] for c in cells);check();self.cache[key]=supports
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        current=list(domains)
        for cell,mask in zip(cells,supports):current[cell]&=mask
        check();self.stats['selective_dd_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
