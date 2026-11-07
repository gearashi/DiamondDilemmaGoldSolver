"""Joint label-consistent linkage through every mandatory fixed chain (#24).

All routes share placement and physical-owner labels. Distinct strands of one
chosen tile may be used by different routes; capacity is on ports, not tiles.
A positive linkage may omit other strands and is only a necessary relaxation.
"""
from collections import OrderedDict
from constraint_region_incremental import source_binding
from constraint_region_isomorphism import residual_graph,IsomorphismLimit
from constraint_region_oracles import LabelledRouteOracle
from constraint_separator import _Budget
from validator import reverse_mask

STAT_KEYS=('multichain_queries','multichain_completed','multichain_unknown','multichain_admission_skips',
 'multichain_cache_hits','multichain_chains','multichain_steps','multichain_label_tests',
 'multichain_failed_state_hits','multichain_failures','multichain_graph_work')


class JointChainOracle:
    def __init__(self,graph,chains,compatible=None,max_steps=50000):
        if type(max_steps)is not int or max_steps<0:raise ValueError('Invalid joint linkage budget.')
        self.graph={v:tuple(edges) for v,edges in graph.items()};self.chains=tuple(tuple(pair) for pair in chains)
        ends=tuple(v for pair in self.chains for v in pair)
        if not self.chains or any(len(pair)!=2 for pair in self.chains) or any(type(v)is not int or v<0 for v in ends) or len(ends)!=len(set(ends)):
            raise ValueError('Mandatory chains require distinct endpoint pairs.')
        for vertex,edges in self.graph.items():
            if type(vertex)is not int or vertex<0:raise ValueError('Invalid linkage vertex.')
            for other,labels in edges:
                if type(other)is not int or other<0 or other==vertex:raise ValueError('Invalid linkage edge.')
                if any(len(label)!=2 or any(type(x)is not int or x<0 for x in label) for label in labels):raise ValueError('Invalid placement labels.')
                if (vertex,labels) not in self.graph.get(other,()):raise ValueError('Linkage graph must be undirected.')
        self.labels=LabelledRouteOracle(self.graph,compatible,max_steps);self.max_steps=max_steps
        self.steps=0;self.label_tests=0;self.hits=0;self.failed=OrderedDict()
    def possible(self,check=lambda:None):
        self.steps=0;self.label_tests=0;self.hits=0;self.failed.clear()
        def tick():
            check();self.steps+=1
            if self.steps>self.max_steps:raise _Budget
        mate={a:b for pair in self.chains for a,b in (pair,pair[::-1])}
        number={v:i for i,pair in enumerate(self.chains) for v in pair};start,current=self.chains[0]
        all_chains=(1<<len(self.chains))-1
        def walk(vertex,visited,seen,assignment):
            tick();key=(vertex,visited,seen,tuple(sorted(assignment.items())))
            if key in self.failed:self.hits+=1;return False
            for other,labels in self.graph.get(vertex,()):
                tick()
                if other in seen and other!=start:continue
                self.label_tests+=1;extended=self.labels._extend(assignment,labels)
                if extended is None:continue
                if other==start:
                    if visited==all_chains:return True
                    continue
                if other in mate:
                    index=number[other]
                    if visited>>index&1 or mate[other] in seen:continue
                    if walk(mate[other],visited|(1<<index),seen|{other,mate[other]},extended):return True
                elif walk(other,visited,seen|{other},extended):return True
            self.failed[key]=True
            while len(self.failed)>4096:self.failed.popitem(last=False)
            return False
        try:return walk(current,1,frozenset((start,current)),{})
        except _Budget:return None


class MultichainRoutingPruner:
    def __init__(self,owner):
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
        self.original=tuple(sum(1<<code for code in set(row)) for row in self.problem.domains)
    def prune(self,domains,check=lambda:None,*,max_active=16,max_chains=8,max_vertices=768,max_ports=128,max_steps=50000):
        if source_binding(self.owner)!=self.binding:raise ValueError('Multichain source binding changed.')
        limits=(max_active,max_chains,max_vertices,max_ports,max_steps)
        if any(type(bound)is not int or bound<0 for bound in limits) or max_vertices>768 or max_chains>8 or max_ports>128:raise ValueError('Invalid multichain limits.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Multichain domains escape the original root.')
        if not all(domains):return False
        if self.owner.target!='gold' or not 1<=sum(mask.bit_count()>1 for mask in domains)<=max_active:
            self.stats['multichain_admission_skips']+=1;return False
        check();key=(tuple(domains),limits)
        if key in self.cache:self.stats['multichain_cache_hits']+=1;result=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['multichain_queries']+=1;steps=0;oracle=None
            def tick():
                nonlocal steps
                check();steps+=1;self.stats['multichain_graph_work']+=1
                if steps>max_steps:raise IsomorphismLimit
            class Work:
                pass
            work=Work();work.tick=tick
            try:
                model=residual_graph(self.owner,domains,work,max_vertices)
                if model is None:result=False
                else:
                    graph={};chains=[]
                    def edge(a,b,labels=()):
                        tick();graph.setdefault(a,[]).append((b,labels));graph.setdefault(b,[]).append((a,labels))
                    for vertex,neighbors in enumerate(model.adj):
                        for other,color in neighbors.items():
                            if vertex<other and color==5:edge(vertex,other)
                            if vertex<other and color==6:chains.append((vertex,other))
                    for choice,label in model.choices.items():
                        for segment,color in model.adj[choice].items():
                            if color!=3:continue
                            ports=[v for v,c in model.adj[segment].items() if c==4]
                            edge(*ports,(label,))
                    self.stats['multichain_chains']+=len(chains)
                    if not 2<=len(chains)<=max_chains or len(graph)>max_ports:
                        self.stats['multichain_admission_skips']+=1;result=None
                    else:
                        def compatible(assignment):
                            return all(a not in assignment or b not in assignment or
                                self.owner.masks[assignment[a]][sa]==reverse_mask(self.owner.masks[assignment[b]][sb]) for a,sa,b,sb in self.problem.edges)
                        oracle=JointChainOracle(graph,chains,compatible,max(0,max_steps-steps));result=oracle.possible(check)
            except IsomorphismLimit:result=None
            finally:
                if oracle is not None:
                    self.stats['multichain_steps']+=oracle.steps;self.stats['multichain_label_tests']+=oracle.label_tests;self.stats['multichain_failed_state_hits']+=oracle.hits
            check();self.cache[key]=result
            while len(self.cache)>16:self.cache.popitem(last=False)
            self.stats['multichain_unknown' if result is None else 'multichain_completed']+=1
        if result is not False:return False
        check();domains[0]=0;self.stats['multichain_failures']+=1;return True
