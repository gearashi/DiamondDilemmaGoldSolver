"""Verified residual incidence-graph reuse of complete Gold supports (#22).

Fixed pieces are contracted into mandatory boundary paths. Active cell choices,
physical resources, occupied ports, strand pairings and seam relationships are
all encoded. A complete color/edge-preserving bijection is verified before any
support transport. Refinement hashes or partial mappings never prove equality.
"""
from collections import OrderedDict
from constraint_region_incremental import source_binding
from constraint_region_compact import GoldCompactFactors,compile_compact,CompactLimit
from constraint_pruning import values
from validator import reverse_mask

STAT_KEYS=('isomorphic_queries','isomorphic_completed','isomorphic_unknown','isomorphic_admission_skips',
 'isomorphic_hits','isomorphic_compilations','isomorphic_transitions','isomorphic_states',
 'isomorphic_graph_vertices','isomorphic_graph_edges','isomorphic_refinements','isomorphic_mapping_trials',
 'isomorphic_verified_mappings','isomorphic_work_ticks','isomorphic_prunes')

class IsomorphismLimit(Exception):pass

class _Work:
    def __init__(self,limit,check,stats):self.limit,self.check,self.stats=limit,check,stats;self.steps=0
    def tick(self):
        self.check();self.steps+=1;self.stats['isomorphic_work_ticks']+=1
        if self.steps>self.limit:raise IsomorphismLimit

class ResidualGraph:
    def __init__(self,max_vertices,work):self.colors=[];self.adj=[];self.max_vertices=max_vertices;self.work=work;self.choices={}
    def node(self,color):
        self.work.tick()
        if len(self.colors)>=self.max_vertices:raise IsomorphismLimit
        index=len(self.colors);self.colors.append(color);self.adj.append({});return index
    def edge(self,a,b,color):
        self.work.tick()
        if a==b or b in self.adj[a]:raise ValueError('Invalid residual incidence edge.')
        self.adj[a][b]=color;self.adj[b][a]=color


def residual_graph(owner,domains,work,max_vertices=1024):
    problem=owner.problem
    if any(a==b for a,_,b,_ in problem.edges):raise IsomorphismLimit
    fixed={c:mask.bit_length()-1 for c,mask in enumerate(domains) if mask.bit_count()==1}
    if len({code//3 for code in fixed.values()})!=len(fixed):return None
    if any(problem.neighbors[c,s]<0 and owner.masks[code][s] for c,code in fixed.items() for s in range(3)):return None
    active=tuple(c for c in range(problem.n) if c not in fixed);used={code//3 for code in fixed.values()}
    choices={}
    for cell in active:
        row=[]
        for code in values(domains[cell]):
            work.tick()
            if code//3 in used:continue
            if any(problem.neighbors[cell,side]<0 and owner.masks[code][side] for side in range(3)):continue
            if any(other in fixed and owner.masks[code][side]!=reverse_mask(owner.masks[fixed[other]][oside]) for other,side,oside,_ in owner.adj[cell]):continue
            row.append(code)
        if not row:return None
        choices[cell]=tuple(row)
    # Remove seam-unsupported choices before omitting permanently empty ports.
    # This is ordinary arc consistency and preserves every complete board.
    changed=True
    while changed:
        changed=False
        for cell in active:
            allowed={(other,side,oside):{reverse_mask(owner.masks[code][oside]) for code in choices[other]}
                     for other,side,oside,_ in owner.adj[cell] if other in choices}
            row=[]
            for code in choices[cell]:
                work.tick()
                if all(owner.masks[code][side] in allowed[other,side,oside] for other,side,oside,_ in owner.adj[cell] if other in choices):row.append(code)
            if not row:return None
            if len(row)!=len(choices[cell]):changed=True;choices[cell]=tuple(row)
    parent={}
    def find(p):
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def unite(a,b):
        parent.setdefault(a,a);parent.setdefault(b,b);parent[find(a)]=find(b)
    for cell,code in fixed.items():
        for a,b in owner.pairs[code]:work.tick();unite(33*cell+a,33*cell+b)
    boundary=[]
    for a,sa,b,sb in problem.edges:
        work.tick()
        if a in fixed and b in fixed:
            mask=owner.masks[fixed[a]][sa]
            if mask!=reverse_mask(owner.masks[fixed[b]][sb]):return None
            for p in values(mask):unite(33*a+11*sa+p,33*b+11*sb+10-p)
        elif a in fixed or b in fixed:
            f,fs,c,cs=(a,sa,b,sb) if a in fixed else (b,sb,a,sa)
            for p in values(owner.masks[fixed[f]][fs]):boundary.append((33*f+11*fs+p,33*c+11*cs+10-p))
    ends={find(p):[] for p in parent}
    for p,q in boundary:ends[find(p)].append(q)
    if active and any(not row for row in ends.values()):return None
    if any(len(row)!=2 for row in ends.values()) and active:raise ValueError('Malformed fixed Gold path boundary.')
    graph=ResidualGraph(max_vertices,work);cellnodes={c:graph.node(0) for c in active}
    resources={t:graph.node(1) for t in range(problem.n) if t not in used};mandatory={q for _,q in boundary};ports={}
    for cell in active:
        occupied={p for code in choices[cell] for pair in owner.pairs[code] for p in pair}
        for p in sorted(occupied):
            ident=33*cell+p;ports[ident]=graph.node(3 if ident in mandatory else 2);graph.edge(cellnodes[cell],ports[ident],0)
    for cell,row in choices.items():
        for code in row:
            node=graph.node(4);graph.choices[node]=(cell,code);graph.edge(node,cellnodes[cell],1);graph.edge(node,resources[code//3],2)
            for a,b in owner.pairs[code]:
                segment=graph.node(5);graph.edge(node,segment,3);graph.edge(segment,ports[33*cell+a],4);graph.edge(segment,ports[33*cell+b],4)
    for a,sa,b,sb in problem.edges:
        if a in cellnodes and b in cellnodes:
            for p in range(11):
                left=33*a+11*sa+p;right=33*b+11*sb+10-p
                if (left in ports)!=(right in ports):raise RuntimeError('Residual seam arc consistency omitted an occupied opposite port.')
                if left in ports:graph.edge(ports[left],ports[right],5)
    for row in ends.values():
        if row:graph.edge(ports[row[0]],ports[row[1]],6)
    return graph


def verified_isomorphism(left,right,work,stats,max_trials=10000):
    if len(left.colors)!=len(right.colors):return None
    a=list(left.colors);b=list(right.colors)
    while True:
        work.tick();stats['isomorphic_refinements']+=1
        signatures=[]
        for graph,colors in ((left,a),(right,b)):
            rows=[]
            for v in range(len(colors)):
                work.tick();rows.append((colors[v],tuple(sorted((edge,colors[w]) for w,edge in graph.adj[v].items()))))
            signatures.append(rows)
        palette={row:i for i,row in enumerate(sorted(set(signatures[0]+signatures[1])))}
        aa=[palette[row] for row in signatures[0]];bb=[palette[row] for row in signatures[1]]
        if sorted(aa)!=sorted(bb):return None
        stable=len(set(aa))==len(set(a));a,b=aa,bb
        if stable:break
    classes={}
    for v,color in enumerate(b):classes.setdefault(color,[]).append(v)
    mapping={};reverse={};used=set();trials=0
    def walk():
        nonlocal trials
        work.tick()
        if len(mapping)==len(a):return True
        def compatible(v,w):
            work.tick()
            return (all(right.adj[w].get(mapping[x])==edge for x,edge in left.adj[v].items() if x in mapping)
                    and all(left.adj[v].get(reverse[y])==edge for y,edge in right.adj[w].items() if y in reverse))
        remaining={color:sum(w not in used for w in vertices) for color,vertices in classes.items()}
        v=min((v for v in range(len(a)) if v not in mapping),
              key=lambda v:(remaining[a[v]],-sum(x in mapping for x in left.adj[v]),v))
        candidates=[w for w in classes[a[v]] if w not in used and compatible(v,w)]
        for w in candidates:
            work.tick();trials+=1;stats['isomorphic_mapping_trials']+=1
            if trials>max_trials:raise IsomorphismLimit
            mapping[v]=w;reverse[w]=v;used.add(w)
            if walk():return True
            used.remove(w);del mapping[v];del reverse[w]
        return False
    if not walk():return None
    # Independent full structural replay, not trust in refinement or search.
    if len(set(mapping.values()))!=len(a):raise RuntimeError('Residual mapping is not bijective.')
    for v,w in mapping.items():
        work.tick()
        if left.colors[v]!=right.colors[w] or {mapping[x]:edge for x,edge in left.adj[v].items()}!=right.adj[w]:
            raise RuntimeError('Residual graph mapping failed exact verification.')
    stats['isomorphic_verified_mappings']+=1;return mapping


class IsomorphicResidualPruner:
    def __init__(self,owner):
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.cache=[];self.stats=dict.fromkeys(STAT_KEYS,0)
        self.original=tuple(sum(1<<code for code in set(row)) for row in self.problem.domains)
    def prune(self,domains,check=lambda:None,*,max_vertices=768,max_active=8,max_states=4096,max_steps=50000,max_trials=10000):
        if source_binding(self.owner)!=self.binding:raise ValueError('Residual isomorphism source binding changed.')
        if any(type(bound)is not int or bound<0 for bound in (max_vertices,max_active,max_states,max_steps,max_trials)) or max_vertices>768:raise ValueError('Invalid residual isomorphism bounds.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Residual isomorphism domains escape the original root.')
        if not all(domains):return False
        active=sum(mask.bit_count()>1 for mask in domains)
        if self.owner.target!='gold' or not 1<=active<=max_active:self.stats['isomorphic_admission_skips']+=1;return False
        self.stats['isomorphic_queries']+=1;work=_Work(max_steps,check,self.stats);model=None;supports=None
        try:
            graph=residual_graph(self.owner,domains,work,max_vertices)
            if graph is None:supports=(0,)*self.problem.n
            else:
                self.stats['isomorphic_graph_vertices']+=len(graph.colors);self.stats['isomorphic_graph_edges']+=sum(map(len,graph.adj))//2
                for saved,accepted,possible in self.cache:
                    mapping=verified_isomorphism(saved,graph,work,self.stats,max_trials)
                    if mapping is None:continue
                    self.stats['isomorphic_hits']+=1
                    supports=[mask if mask.bit_count()==1 and possible else 0 for mask in domains]
                    for node in accepted:
                        cell,code=graph.choices[mapping[node]];supports[cell]|=1<<code
                    supports=tuple(supports);break
                if supports is None:
                    self.stats['isomorphic_compilations']+=1;model=GoldCompactFactors(self.problem,domains,check=work.tick)
                    result=compile_compact(model,check=work.tick,max_states=max_states,max_steps=max_steps)
                    supports=result['supports'];accepted=frozenset(node for node,(cell,code) in graph.choices.items() if supports[cell]>>code&1)
                    work.tick();self.cache.append((graph,accepted,bool(result['rows'])))
                    if len(self.cache)>8:self.cache.pop(0)
            check();self.stats['isomorphic_completed']+=1
        except (IsomorphismLimit,CompactLimit):self.stats['isomorphic_unknown']+=1;return False
        finally:
            if model is not None:
                self.stats['isomorphic_transitions']+=model.stats['compact_factor_transitions'];self.stats['isomorphic_states']+=model.stats['compact_factor_states']
        current=[mask&keep for mask,keep in zip(domains,supports)];check()
        self.stats['isomorphic_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
