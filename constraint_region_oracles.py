"""Exact scoped residual reuse and bounded label-consistent routing.

These oracles may answer unknown; an exhausted budget is never a failed route
or a completed residual search. Residual cache entries are internal completed
search results, not portable certificates from untrusted sources.
"""
from collections import OrderedDict
from constraint_pruning import values
from constraint_separator import SeparatorPruner,_Budget
from validator import reverse_mask


RESIDUAL_STAT_KEYS=('residual_queries','residual_hits','residual_closed')


def residual_key(problem,domains,target='gold',root_scope=(),_owner=None):
    """Sufficient contracted state, retaining all live inventory and port wiring.

    Complete fixed interiors may differ while the remaining problem is the
    same. Actual cell/physical-piece labels remain; this implementation does
    not assert arbitrary graph isomorphisms or contextual piece interchange.
    """
    if target not in ('gold','edges') or len(domains)!=problem.n:raise ValueError('Invalid residual query.')
    fixed={c:d.bit_length()-1 for c,d in enumerate(domains) if d.bit_count()==1}
    remaining=tuple(c for c in range(problem.n) if c not in fixed)
    used=0;invalid=False
    for code in fixed.values():
        bit=1<<(code//3)
        if used&bit:invalid=True
        used|=bit
    owner=_owner if _owner is not None else SeparatorPruner(problem,target);mates={};cycles=0;boundary=[]
    for cell,code in fixed.items():
        for a,b in owner.pairs[code]:
            a+=33*cell;b+=33*cell;mates[a]=b;mates[b]=a
    for a,sa,b,sb in problem.edges:
        if a in fixed and b in fixed:
            mask=int(problem.masks[fixed[a],sa])
            if mask!=reverse_mask(int(problem.masks[fixed[b],sb])):invalid=True;continue
            for pos in values(mask):cycles+=owner._join(mates,33*a+11*sa+pos,33*b+11*sb+10-pos)
        elif (a in fixed)!=(b in fixed):
            f,fs,r,rs=(a,sa,b,sb) if a in fixed else (b,sb,a,sa)
            boundary.append((r,rs,reverse_mask(int(problem.masks[fixed[f],fs]))))
    exposed={}
    for a,sa,b,sb in problem.edges:
        if (a in fixed)!=(b in fixed):
            f,fs,r,rs=(a,sa,b,sb) if a in fixed else (b,sb,a,sa)
            for pos in values(int(problem.masks[fixed[f],fs])):
                exposed[33*f+11*fs+pos]=33*r+11*rs+10-pos
    pairs=[]
    for a,b in mates.items():
        if a<b:
            if a not in exposed or b not in exposed:invalid=True
            else:pairs.append(tuple(sorted((exposed[a],exposed[b]))))
    unavailable=sum(7<<(3*tile) for tile in values(used))
    live=tuple((cell,domains[cell]&~unavailable) for cell in remaining)
    return (problem.fingerprint,target,tuple(root_scope),remaining,used,live,
            tuple(sorted(boundary)),tuple(sorted(pairs)),cycles,invalid)


class ResidualProofCache:
    def __init__(self,problem,target='gold',root_scope=(),limit=4096):
        self.problem=problem;self.target=target;self.root_scope=tuple(root_scope);self.limit=limit
        self.closed=OrderedDict();self.stats=dict.fromkeys(RESIDUAL_STAT_KEYS,0)
        self._owner=SeparatorPruner(problem,target)
    def key(self,domains):return residual_key(self.problem,domains,self.target,self.root_scope,self._owner)
    def contains(self,domains):
        self.stats['residual_queries']+=1;key=self.key(domains)
        if key in self.closed:self.closed.move_to_end(key);self.stats['residual_hits']+=1;return True
        return False
    def remember_completed(self,domains,*,status):
        if status!='unsatisfiable':return False
        key=self.key(domains);self.closed[key]=True
        while len(self.closed)>self.limit:self.closed.popitem(last=False)
        self.stats['residual_closed']=len(self.closed);return True
    def group(self,queries):
        groups={}
        for ident,domains in queries:groups.setdefault(self.key(domains),[]).append(ident)
        return tuple(tuple(group) for group in groups.values())


class LabelledRouteOracle:
    """Exhaustive simple-path linkage with shared orientation/inventory labels."""
    def __init__(self,graph,compatible=None,max_steps=50000):
        self.graph={v:tuple(edges) for v,edges in graph.items()};self.compatible=compatible
        self.max_steps=max_steps;self.steps=0;self.paths=0
    def _extend(self,assignment,labels):
        result=dict(assignment);owners={code//3:cell for cell,code in result.items()}
        for cell,code in labels:
            if cell in result and result[cell]!=code:return None
            if code//3 in owners and owners[code//3]!=cell:return None
            result[cell]=code;owners[code//3]=cell
        if self.compatible is not None and not self.compatible(result):return None
        return result
    def possible(self,start,end,*,count=1,check=lambda:None):
        if count not in (1,2) or start==end:raise ValueError('Expected one or two routes between distinct ports.')
        self.steps=0;self.paths=0
        def tick():
            self.steps+=1
            if self.steps>self.max_steps:raise _Budget
            check()
        def enumerate_paths(vertex,seen,assignment,forbidden,blocked_edges,trail):
            tick()
            if vertex==end:
                self.paths+=1;yield frozenset(seen)-{start,end},assignment,tuple(trail);return
            for other,labels in self.graph.get(vertex,()):
                if other in seen or other in forbidden or tuple(sorted((vertex,other))) in blocked_edges:continue
                extended=self._extend(assignment,labels)
                if extended is not None:
                    yield from enumerate_paths(other,seen|{other},extended,forbidden,blocked_edges,(*trail,tuple(sorted((vertex,other)))))
        try:
            for used,assignment,edges in enumerate_paths(start,{start},{},set(),set(),()):
                if count==1:return True
                for _ in enumerate_paths(start,{start},assignment,used,set(edges),()):return True
            return False
        except _Budget:return None

    @classmethod
    def from_problem(cls,problem,domains,max_steps=50000):
        owner=SeparatorPruner(problem);graph={}
        def edge(a,b,labels=()):
            graph.setdefault(a,[]).append((b,labels));graph.setdefault(b,[]).append((a,labels))
        occupied={}
        for cell,domain in enumerate(domains):
            for code in values(domain):
                for a,b in owner.pairs[code]:edge(33*cell+a,33*cell+b,((cell,code),))
            for side in range(3):
                mask=0
                for code in values(domain):mask|=int(problem.masks[code,side])
                occupied[cell,side]=mask
        for a,sa,b,sb in problem.edges:
            for pos in values(occupied[a,sa]&reverse_mask(occupied[b,sb])):edge(33*a+11*sa+pos,33*b+11*sb+10-pos)
        def compatible(assignment):
            return all(a not in assignment or b not in assignment or
                int(problem.masks[assignment[a],sa])==reverse_mask(int(problem.masks[assignment[b],sb]))
                for a,sa,b,sb in problem.edges)
        return cls(graph,compatible,max_steps)


class LabelledRoutingPruner:
    def __init__(self,problem,max_steps=50000,max_unfixed=16):
        self.problem=problem;self.max_steps=max_steps;self.max_unfixed=max_unfixed
        self.stats={'labelled_queries':0,'labelled_steps':0,'labelled_unknown':0,
                    'labelled_failures':0,'labelled_admission_skips':0}
    def prune(self,domains,check=lambda:None):
        fixed=[(cell,d.bit_length()-1) for cell,d in enumerate(domains) if d.bit_count()==1]
        if not fixed or sum(d.bit_count()!=1 for d in domains)>self.max_unfixed:
            self.stats['labelled_admission_skips']+=1;return False
        owner=SeparatorPruner(self.problem)
        for cell,code in fixed:
            if owner.pairs[code]:
                a,b=owner.pairs[code][0];start,end=33*cell+a,33*cell+b;break
        else:return False
        oracle=LabelledRouteOracle.from_problem(self.problem,domains,self.max_steps)
        result=oracle.possible(start,end,count=2,check=check)
        self.stats['labelled_queries']+=1;self.stats['labelled_steps']+=oracle.steps
        if result is None:self.stats['labelled_unknown']+=1
        if result is False:
            self.stats['labelled_failures']+=1;domains[cell]=0;return True
        return False
