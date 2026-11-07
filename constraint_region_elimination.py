"""Bounded factor elimination and conservative regional decision diagrams.

Factors express exact finite relations. Mini-bucket splitting and DD merging
only relax them. A complete relaxed contradiction is a valid contradiction;
resource exhaustion is always unknown. Gold connectivity remains a separate
constraint unless an exact regional relation explicitly encodes it.
"""
from dataclasses import dataclass
from itertools import product,combinations
from collections import OrderedDict
from constraint_separator import _Budget
from constraint_pruning import values
from validator import reverse_mask


@dataclass(frozen=True)
class Factor:
    scope:tuple
    rows:frozenset
    def __post_init__(self):
        if len(set(self.scope))!=len(self.scope) or any(len(row)!=len(self.scope) for row in self.rows):
            raise ValueError('Invalid finite factor.')
    def restrict(self,assignment):
        indices=[(i,assignment[v]) for i,v in enumerate(self.scope) if v in assignment]
        return Factor(self.scope,frozenset(row for row in self.rows if all(row[i]==x for i,x in indices)))


class WorkBudget:
    def __init__(self,limit=200000,max_rows=8192,check=lambda:None):
        self.limit=limit;self.max_rows=max_rows;self.check=check;self.steps=0;self.peak_rows=0
    def tick(self,rows=None):
        self.steps+=1
        if self.steps>self.limit:raise _Budget
        if rows is not None:
            self.peak_rows=max(self.peak_rows,rows)
            if rows>self.max_rows:raise _Budget
        self.check()


def join_factors(left,right,budget):
    scope=tuple(sorted(set(left.scope)|set(right.scope)))
    shared=tuple(sorted(set(left.scope)&set(right.scope)))
    li=tuple(left.scope.index(v) for v in shared);ri=tuple(right.scope.index(v) for v in shared)
    index={}
    for row in right.rows:
        budget.tick();index.setdefault(tuple(row[i] for i in ri),[]).append(row)
    positions=[(0,left.scope.index(v)) if v in left.scope else (1,right.scope.index(v)) for v in scope]
    output=set()
    for a in left.rows:
        for b in index.get(tuple(a[i] for i in li),()):
            budget.tick();output.add(tuple((a,b)[which][i] for which,i in positions));budget.tick(len(output))
    return Factor(scope,frozenset(output))


def project(factor,variable,budget):
    if variable not in factor.scope:return factor
    i=factor.scope.index(variable);rows=set()
    for row in factor.rows:budget.tick();rows.add(row[:i]+row[i+1:]);budget.tick(len(rows))
    return Factor(tuple(v for v in factor.scope if v!=variable),frozenset(rows))


def eliminate(factors,order,*,width=None,budget=None):
    """Existence under exact elimination, or a sound mini-bucket relaxation.

    width bounds each mini-bucket's variable union, not an individual row cap.
    No discarded factor can create a negative result. Return (possible, stats).
    """
    budget=budget or WorkBudget();pending=list(factors);groups_count=0;forgotten=0
    if width is not None and width<1:raise ValueError('Mini-bucket width must be positive.')
    for var in order:
        budget.tick();bucket=[f for f in pending if var in f.scope]
        pending=[f for f in pending if var not in f.scope]
        if not bucket:continue
        groups=[]
        for factor in sorted(bucket,key=lambda f:(-len(f.scope),f.scope)):
            for group in groups:
                union=set(factor.scope).union(*(set(f.scope) for f in group))
                if width is None or len(union)<=width:group.append(factor);break
            else:groups.append([factor])
        for group in groups:
            merged=group[0]
            for factor in group[1:]:
                merged=join_factors(merged,factor,budget)
                if not merged.rows:return False,{'steps':budget.steps,'groups':groups_count,'forgotten':forgotten,'peak_rows':budget.peak_rows}
            reduced=project(merged,var,budget);groups_count+=1
            if not reduced.rows:return False,{'steps':budget.steps,'groups':groups_count,'forgotten':forgotten,'peak_rows':budget.peak_rows}
            pending.append(reduced)
        forgotten+=1
    if any(not f.rows for f in pending):possible=False
    else:
        merged=Factor((),frozenset({()}))
        for factor in pending:
            merged=join_factors(merged,factor,budget)
            if not merged.rows:break
        possible=bool(merged.rows)
    return possible,{'steps':budget.steps,'groups':groups_count,'forgotten':forgotten,'peak_rows':budget.peak_rows}


def problem_factors(problem,domains,cells=None,check=lambda:None,max_entries=200000):
    """Exact seams and physical-piece uniqueness on selected cells.

    Omitting other cells is an optimistic relaxation. A global piece is encoded
    by constraints between all its possible selected occurrences; elimination
    forgets its information only once every such occurrence has been handled.
    """
    cells=tuple(range(problem.n)) if cells is None else tuple(sorted(set(cells)))
    choices={c:tuple(values(domains[c])) for c in cells};factors=[];entries=0
    for c in cells:factors.append(Factor((c,),frozenset((x,) for x in choices[c])))
    links={}
    for a,sa,b,sb in problem.edges:
        if a in choices and b in choices:
            if a>b:a,sa,b,sb=b,sb,a,sa
            links.setdefault((a,b),[]).append((sa,sb))
    for a,b in combinations(cells,2):
        check();seams=links.get((a,b),())
        if not seams and not ({x//3 for x in choices[a]}&{y//3 for y in choices[b]}):continue
        rows=set()
        for x in choices[a]:
            for y in choices[b]:
                entries+=1
                if entries>max_entries:raise _Budget
                if x//3!=y//3 and all(int(problem.masks[x,sa])==reverse_mask(int(problem.masks[y,sb])) for sa,sb in seams):rows.add((x,y))
        # A relation equal to the entire product imposes no dependency. Keeping
        # it would artificially extend resource lifetimes and elimination width.
        if len(rows)!=len(choices[a])*len(choices[b]):
            factors.append(Factor((a,b),frozenset(rows)))
    return factors,choices


def min_fill_order(factors,variables):
    graph={v:set() for v in variables}
    for f in factors:
        for a,b in combinations(f.scope,2):graph[a].add(b);graph[b].add(a)
    result=[]
    while graph:
        v=min(graph,key=lambda v:(sum(b not in graph[a] for a,b in combinations(graph[v],2)),len(graph[v]),v))
        neighbors=graph.pop(v);result.append(v)
        for a,b in combinations(neighbors,2):graph[a].add(b);graph[b].add(a)
        for a in neighbors:graph[a].discard(v)
    return tuple(result)


def relaxed_diagram_possible(factors,domains,*,width=16,budget=None):
    """Layered optimistic DD. State merging unions per-variable possibilities.

    The union contains every represented assignment but may add combinations.
    Each factor is tested existentially; therefore a complete empty layer proves
    impossibility even after merging. No approximate zero becomes a proof.
    """
    if width<1:raise ValueError('Width must be positive.')
    budget=budget or WorkBudget();variables=tuple(sorted(domains));index={v:i for i,v in enumerate(variables)}
    root=tuple(frozenset(domains[v]) for v in variables);states={root};merges=0
    def supported(state):
        for factor in factors:
            budget.tick()
            if not any(all(value in state[index[v]] for v,value in zip(factor.scope,row)) for row in factor.rows):return False
        return True
    if not supported(root):return False,{'steps':budget.steps,'merges':0}
    for v in variables:
        following=set();i=index[v]
        for state in states:
            for value in state[i]:
                budget.tick();child=list(state);child[i]=frozenset({value});child=tuple(child)
                if supported(child):following.add(child)
        if not following:return False,{'steps':budget.steps,'merges':merges}
        if len(following)>width:
            # Deterministic groups. Union is a proved overapproximation; every
            # input state belongs to exactly one output group.
            ordered=sorted(following,key=lambda s:tuple(tuple(sorted(x)) for x in s));groups=[ordered[k::width] for k in range(width)]
            merged={tuple(frozenset().union(*(state[j] for state in group)) for j in range(len(variables))) for group in groups}
            merges+=len(following)-len(merged);following=merged
        states=following
    return True,{'steps':budget.steps,'merges':merges}


class EliminationPruner:
    """Optional exact/relaxed local consequence oracle for real puzzle domains."""
    def __init__(self,problem,max_cells=8,max_domain=8,max_steps=50000):
        self.problem=problem;self.max_cells=max_cells;self.max_domain=max_domain;self.max_steps=max_steps
        self.cache=OrderedDict();self.stats={'factor_queries':0,'factor_steps':0,'factor_prunes':0,
            'factor_cache_hits':0,'factor_unknown':0,'factor_dd_merges':0}
    def prune(self,domains,check=lambda:None,mode='exact',width=4,full=False):
        if mode not in ('exact','mini','diagram'):raise ValueError('Unknown factor reasoning mode.')
        selected=tuple(range(self.problem.n)) if full else tuple(sorted((c for c,d in enumerate(domains) if 1<d.bit_count()<=self.max_domain),key=lambda c:(domains[c].bit_count(),c))[:self.max_cells])
        if len(selected)<3:return False
        key=(mode,width,selected,tuple(domains[c] for c in selected))
        if key in self.cache:
            self.stats['factor_cache_hits']+=1;supports=self.cache[key]
        else:
            budget=WorkBudget(self.max_steps,8192,check);supports={c:domains[c] for c in selected};complete=True
            try:
                factors,choices=problem_factors(self.problem,domains,selected,check,self.max_steps)
                order=min_fill_order(factors,selected)
                lifetimes=piece_lifetimes(choices,order)
                self.stats['factor_live_piece_width']=max(self.stats.get('factor_live_piece_width',0),max((len(live) for live in lifetimes['live']),default=0))
                self.stats['factor_forgotten_resources']=self.stats.get('factor_forgotten_resources',0)+len(lifetimes['last'])
                if mode=='exact':
                    self.stats['factor_queries']+=1
                    exact_supports,work=support_elimination(factors,order,budget)
                    supports={c:sum(1<<code for code in exact_supports[c]) for c in selected}
                # A whole-constraint query first proves or refutes feasibility;
                # fixed choices need no redundant individual support queries.
                if full and mode!='exact':
                    self.stats['factor_queries']+=1
                    if mode=='diagram':
                        possible,work=relaxed_diagram_possible(factors,choices,width=width,budget=budget)
                        self.stats['factor_dd_merges']+=work['merges']
                    else:possible,work=eliminate(factors,order,width=width if mode=='mini' else None,budget=budget)
                    if not possible:supports[selected[0]]=0
                for cell in selected:
                    if mode=='exact' or not supports[selected[0]]:break
                    if len(choices[cell])==1:continue
                    for code in choices[cell]:
                        check();restricted=[f.restrict({cell:code}) for f in factors]
                        self.stats['factor_queries']+=1
                        if mode=='diagram':
                            source=dict(choices);source[cell]=(code,)
                            possible,work=refined_diagram_possible(restricted,source,widths=tuple(dict.fromkeys((1,2,width))),budget=budget)
                            self.stats['factor_dd_merges']+=work['merges']
                        else:possible,work=eliminate(restricted,order,width=width if mode=='mini' else None,budget=budget)
                        if not possible:supports[cell]&=~(1<<code)
            except _Budget:complete=False;self.stats['factor_unknown']+=1
            self.stats['factor_steps']+=budget.steps
            # Completed candidate queries remain sound after a later query caps.
            if complete:
                self.cache[key]=supports.copy()
                while len(self.cache)>32:self.cache.popitem(last=False)
        changed=False
        for cell,keep in supports.items():
            check();new=domains[cell]&keep
            if new!=domains[cell]:self.stats['factor_prunes']+=(domains[cell]^new).bit_count();domains[cell]=new;changed=True
        return changed


def piece_lifetimes(choices,order):
    """Explicit physical-resource lifetime in the full factor decomposition.

    Every shared occurrence constraint is compiled before elimination. A piece
    disappears after its last occurrence variable is eliminated, never merely
    because a geometric region was completed.
    """
    position={cell:i for i,cell in enumerate(order)};first={};last={}
    for cell,row in choices.items():
        for code in row:
            tile=code//3;first[tile]=min(first.get(tile,len(order)),position[cell]);last[tile]=max(last.get(tile,-1),position[cell])
    live=tuple(frozenset(tile for tile in first if first[tile]<=i<last[tile]) for i in range(len(order)))
    return {'first':first,'last':last,'live':live}


def refined_diagram_possible(factors,domains,widths=(1,2,4,8),budget=None):
    """Refine only a still-possible relaxed query; exact/negative results stop."""
    budget=budget or WorkBudget();passes=0;merges=0
    for width in widths:
        possible,stats=relaxed_diagram_possible(factors,domains,width=width,budget=budget)
        passes+=1;merges+=stats['merges']
        if not possible or stats['merges']==0:
            return possible,{'passes':passes,'merges':merges,'exact':True,'steps':budget.steps}
    return True,{'passes':passes,'merges':merges,'exact':False,'steps':budget.steps}


def conditioned_elimination(factors,domains,cutset,*,budget=None):
    """Complete disjoint cutset conditioning before full factor elimination."""
    budget=budget or WorkBudget();order=min_fill_order(factors,domains);queries=0
    for row in product(*(domains[var] for var in cutset)):
        assignment=dict(zip(cutset,row));restricted=[f.restrict(assignment) for f in factors]
        possible,_=eliminate(restricted,order,budget=budget);queries+=1
        if possible:return True,{'conditions':queries,'steps':budget.steps}
    return False,{'conditions':queries,'steps':budget.steps}


def support_elimination(factors,order,budget=None):
    """Exact bucket-tree inward/outward messages recover ALL unary supports.

    Separator tuples, not independent unary projections, connect child buckets.
    Thus every accepted child row extends a complete assignment of all factors.
    Disconnected roots are joined as independent components only after their
    full shared-resource factors have been included in the decomposition.
    """
    budget=budget or WorkBudget();pending=list(factors);rank={v:i for i,v in enumerate(order)};records={}
    for var in order:
        budget.tick();bucket=[f for f in pending if var in f.scope];pending=[f for f in pending if var not in f.scope]
        if not bucket:continue
        bucket.sort(key=lambda f:len(f.rows));joint=bucket[0]
        for factor in bucket[1:]:joint=join_factors(joint,factor,budget)
        if not joint.rows:return {v:set() for v in order},{'steps':budget.steps,'peak_rows':budget.peak_rows,'buckets':len(records)}
        message=project(joint,var,budget);parent=min(message.scope,key=rank.get) if message.scope else None
        records[var]=(joint,message.scope,parent);pending.append(message)
    if any(not f.rows for f in pending):return {v:set() for v in order},{'steps':budget.steps,'peak_rows':budget.peak_rows,'buckets':len(records)}
    reachable={};supports={v:set() for v in order}
    for var in reversed(order):
        if var not in records:continue
        joint,separator,parent=records[var];positions=tuple(joint.scope.index(v) for v in separator)
        if parent is None:contexts={()}
        else:
            pjoint=records[parent][0];pi=tuple(pjoint.scope.index(v) for v in separator)
            contexts={tuple(row[i] for i in pi) for row in reachable[parent]}
        rows=set()
        for row in joint.rows:
            budget.tick()
            if tuple(row[i] for i in positions) in contexts:rows.add(row)
        reachable[var]=rows;vi=joint.scope.index(var);supports[var].update(row[vi] for row in rows)
    return supports,{'steps':budget.steps,'peak_rows':budget.peak_rows,'buckets':len(records)}
