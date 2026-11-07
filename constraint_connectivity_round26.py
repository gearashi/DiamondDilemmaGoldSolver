"""Exact necessary Gold connectivity obligations with scoped certificates (F12).

Assignment duals, joint component demands and possible-graph blocks are
relaxations: a positive answer is never a Gold solution. All-strand labelled
routing refines a positive partial route by adding omitted orientation strands.
"""
from collections import OrderedDict,defaultdict
from constraint_region_compiler_round26 import _domains,_digest
from constraint_region_incremental import source_binding
from constraint_region_cuts import EndpointSupplyPruner
from constraint_region_multichain import JointChainOracle
from constraint_region_isomorphism import residual_graph,IsomorphismLimit
from constraint_gold_graph import PossibleGoldGraph
from constraint_pruning import values
from constraint_separator import _Budget
from validator import reverse_mask

STAT_KEYS=('connectivity_assignment_queries','connectivity_assignment_relaxations','connectivity_assignment_dual_checks',
 'connectivity_joint_states','connectivity_joint_transitions','connectivity_joint_failures',
 'connectivity_refinement_rounds','connectivity_refinement_obligations','connectivity_route_steps',
 'connectivity_route_label_checks','connectivity_route_graph_steps','connectivity_route_unknown','connectivity_route_failures',
 'connectivity_graph_input_edges','connectivity_graph_recomputed_vertices','connectivity_graph_reused_components',
 'connectivity_graph_block_edges','connectivity_graph_restores','connectivity_prunes','connectivity_scope_checks',
 'connectivity_reason_checks','connectivity_reason_steps')


def assignment_upper_bound(weights,check=lambda:None):
    """Exact rectangular maximum-weight injection with a checked integer dual.

    None means a disallowed cell-piece edge. Artificial negative weights allow
    Hungarian construction; the published dual is verified on every legal edge.
    """
    rows=tuple(tuple(row) for row in weights);n=len(rows);m=len(rows[0]) if rows else 0
    if n>m or any(len(row)!=m for row in rows) or any(v is not None and (type(v)is not int or v<0) for row in rows for v in row):raise ValueError('Invalid capacity assignment matrix.')
    if not n:return {'bound':0,'rows':(),'columns':(),'assignment':(),'relaxations':0}
    high=max((v for row in rows for v in row if v is not None),default=0);bad=(high+1)*(n+1)
    cost=[[bad if v is None else -v for v in row] for row in rows]
    u=[0]*(n+1);v=[0]*(m+1);p=[0]*(m+1);way=[0]*(m+1);work=0;infinity=bad*(n+m+1)+1
    for i in range(1,n+1):
        p[0]=i;j0=0;minimum=[infinity]*(m+1);used=[False]*(m+1)
        while True:
            check();used[j0]=True;i0=p[j0];delta=infinity;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                check();work+=1;current=cost[i0-1][j-1]-u[i0]-v[j]
                if current<minimum[j]:minimum[j]=current;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if not p[j0]:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if not j0:break
    assignment=[-1]*n
    for j in range(1,m+1):
        if p[j]:assignment[p[j]-1]=j-1
    certificate={'bound':-sum(u[1:])-sum(v[1:]),'rows':tuple(-x for x in u[1:]),
      'columns':tuple(-x for x in v[1:]),'assignment':tuple(assignment),'relaxations':work}
    certificate['dual_checks']=verify_assignment_bound(rows,certificate,check)
    return certificate


def verify_assignment_bound(weights,certificate,check=lambda:None):
    row=certificate['rows'];col=certificate['columns'];bound=certificate['bound']
    if len(row)!=len(weights) or any(len(values)!=len(col) for values in weights) or type(bound)is not int or any(type(x)is not int for x in (*row,*col)) or any(x<0 for x in col):raise ValueError('Invalid exact assignment dual.')
    if sum(row)+sum(col)!=bound:raise ValueError('Assignment dual objective differs.')
    checks=0
    for i,values in enumerate(weights):
        for j,value in enumerate(values):
            check();checks+=1
            if value is not None and (type(value)is not int or value<0 or row[i]+col[j]<value):raise ValueError('Assignment dual misses a legal edge.')
    return checks


def joint_demands(rows,demands,check=lambda:None,*,max_states=50000,stats=None):
    """Exact injection DP of capped component-supply vectors; no seam optimism lost.

    Each row contains (physical_piece, capacity_vector). Orientation alternatives
    of one piece remain separate; taking their componentwise maxima would be
    unsound for simultaneous demands.
    """
    demands=tuple(demands)
    if any(type(x)is not int or x<0 for x in demands) or type(max_states)is not int or max_states<1:raise ValueError('Invalid component demand budget.')
    layer={(0,(0,)*len(demands))}
    for choices in rows:
        following=set()
        for used,supply in layer:
            for tile,vector in choices:
                check()
                if stats is not None:stats['connectivity_joint_transitions']+=1
                if type(tile)is not int or tile<0 or len(vector)!=len(demands) or any(type(x)is not int or x<0 for x in vector):raise ValueError('Invalid component supply row.')
                if used>>tile&1:continue
                following.add((used|(1<<tile),tuple(min(goal,a+b) for goal,a,b in zip(demands,supply,vector))))
                if len(following)>max_states:raise _Budget
        if stats is not None:stats['connectivity_joint_states']+=len(following)
        layer=following
        if not layer:return False
    return any(supply==demands for used,supply in layer)


class ReversibleBlockCertificates:
    """Recompute only graph components touched by an edge delta; exact rollback."""
    def __init__(self,limit=8):
        self.limit=limit;self.snapshots=OrderedDict();self.previous=None;self.stats=dict.fromkeys(STAT_KEYS,0)
    def update(self,edges,check=lambda:None):
        from constraint_advanced import cyclic_edge_blocks
        key=tuple(sorted(tuple(sorted(edge)) for edge in edges))
        if any(len(e)!=2 or any(type(v)is not int or v<0 for v in e) or e[0]==e[1] for e in key) or len(set(key))!=len(key):raise ValueError('Expected a simple undirected possible graph.')
        self.stats['connectivity_graph_input_edges']+=len(key)
        if key in self.snapshots:
            self.stats['connectivity_graph_restores']+=1;self.snapshots.move_to_end(key);self.previous=self.snapshots[key];return self.previous
        graph=defaultdict(set)
        for a,b in key:graph[a].add(b);graph[b].add(a)
        old=self.previous;dirty=set(graph) if old is None else {v for edge in set(key)^old['edges'] for v in edge}
        if old is not None:
            dirty.update(v for component in old['components'] if component&dirty for v in component)
        # New edges can connect an affected old component to unchanged ones.
        todo=list(dirty)
        for vertex in todo:
            for other in graph.get(vertex,()):
                if other not in dirty:dirty.add(other);todo.append(other)
        components=[];blocks={};pending=dirty&set(graph)
        if old is not None:
            for component in old['components']:
                if not component&dirty:
                    components.append(component);self.stats['connectivity_graph_reused_components']+=1
            blocks.update({edge:block for edge,block in old['blocks'].items() if not set(edge)&dirty})
        while pending:
            check();todo=[min(pending)];seen=set(todo);pending.remove(todo[0])
            for vertex in todo:
                for other in graph[vertex]-seen:seen.add(other);pending.discard(other);todo.append(other)
            component=frozenset(seen);components.append(component);self.stats['connectivity_graph_recomputed_vertices']+=len(component)
            local_edges=tuple(edge for edge in key if edge[0] in component);vertices=sorted(component);indices={v:i for i,v in enumerate(vertices)}
            local=cyclic_edge_blocks(len(vertices),[(indices[a],indices[b]) for a,b in local_edges],check)
            self.stats['connectivity_graph_block_edges']+=len(local_edges)
            groups=defaultdict(list)
            for edge,block in zip(local_edges,local):
                if block>=0:groups[block].append(edge)
            canonical={block:tuple(group) for block,group in groups.items()}
            for edge,block in zip(local_edges,local):blocks[edge]=None if block<0 else canonical[block]
        result={'edges':frozenset(key),'components':tuple(components),'blocks':blocks};check()
        self.snapshots[key]=result;self.previous=result
        while len(self.snapshots)>self.limit:self.snapshots.popitem(last=False)
        return result


class ObligationChainOracle(JointChainOracle):
    def __init__(self,graph,chains,obligations,compatible=None,max_steps=50000):
        super().__init__(graph,chains,compatible,max_steps)
        self.obligations={tuple(label):frozenset(tuple(sorted(edge)) for edge in edges) for label,edges in obligations.items()}
        self.refinements=0;self.added=0
    def possible(self,check=lambda:None):
        self.steps=0;self.label_tests=0;protected=set()
        mate={a:b for pair in self.chains for a,b in (pair,pair[::-1])};number={v:i for i,pair in enumerate(self.chains) for v in pair}
        start,current=self.chains[0];all_chains=(1<<len(self.chains))-1
        def tick():
            check();self.steps+=1
            if self.steps>self.max_steps:raise _Budget
        def query():
            failed=set()
            def walk(vertex,visited,seen,assignment,taken):
                tick();key=(vertex,visited,seen,tuple(sorted(assignment.items())),taken)
                if key in failed:return None
                for other,labels in self.graph.get(vertex,()):
                    tick()
                    if other in seen and other!=start:continue
                    self.label_tests+=1;extended=self.labels._extend(assignment,labels)
                    if extended is None:continue
                    next_taken=taken|{tuple(sorted((vertex,other)))}
                    if other==start:
                        if visited!=all_chains:continue
                        if any(label in protected and not self.obligations.get(label,frozenset())<=next_taken for label in extended.items()):continue
                        return extended,next_taken
                    if other in mate:
                        index=number[other]
                        if visited>>index&1 or mate[other] in seen:continue
                        answer=walk(mate[other],visited|(1<<index),seen|{other,mate[other]},extended,next_taken)
                    else:answer=walk(other,visited,seen|{other},extended,next_taken)
                    if answer is not None:return answer
                failed.add(key);return None
            return walk(current,1,frozenset((start,current)),{},frozenset())
        try:
            while True:
                answer=query()
                if answer is None:return False
                assignment,taken=answer
                omitted={label for label in assignment.items() if not self.obligations.get(label,frozenset())<=taken}
                if not omitted:return True
                new=omitted-protected
                if not new:raise RuntimeError('Connectivity refinement made no progress.')
                protected.update(new);self.refinements+=1;self.added+=len(new)
        except _Budget:return None


class ConnectivityObligations:
    def __init__(self,owner):
        self.owner=owner;self.problem=owner.problem;self.semantic=source_binding(owner)
        self.graph=PossibleGoldGraph(self.problem);self.blocks=ReversibleBlockCertificates();self.supply=EndpointSupplyPruner(self.problem)
        self.stats=dict.fromkeys(STAT_KEYS,0);self.solution=None;self.reasons=[];self.cache=OrderedDict()
    def _scope(self,domains):
        self.stats['connectivity_scope_checks']+=1
        if source_binding(self.owner)!=self.semantic:raise ValueError('Connectivity source changed.')
        return _domains(self.problem,domains)
    def _capacities(self,domains,region,components,check):
        rows=[]
        for cell in region:
            choices=[]
            for code in values(domains[cell]):
                vector=[]
                for outside in components:
                    count=0
                    for other,side,other_side,_ in self.owner.adj[cell]:
                        if other in outside:
                            mask=self.owner.masks[code][side]
                            if any(mask==reverse_mask(self.owner.masks[v][other_side]) for v in values(domains[other])):count+=mask.bit_count()
                    vector.append(count)
                choices.append((code//3,tuple(vector)))
            rows.append(choices)
        return rows
    def _supply_impossible(self,domains,region,check):
        components=self.supply._components(domains,region,check)
        rows=self._capacities(domains,region,components,check)
        for weights,need in [(tuple(1 for _ in components),2*len(components)),*((tuple(int(i==j) for i in range(len(components))),2) for j in range(len(components)))]:
            matrix=[]
            for choices in rows:
                row=[None]*self.problem.n
                for tile,vector in choices:
                    value=sum(w*v for w,v in zip(weights,vector));row[tile]=value if row[tile] is None else max(row[tile],value)
                matrix.append(row)
            self.stats['connectivity_assignment_queries']+=1;certificate=assignment_upper_bound(matrix,check)
            self.stats['connectivity_assignment_relaxations']+=certificate['relaxations']
            self.stats['connectivity_assignment_dual_checks']+=certificate.get('dual_checks',0)+verify_assignment_bound(matrix,certificate,check)
            if certificate['bound']<need:
                self.reasons.append({'kind':'endpoint_assignment','domains':tuple(domains),'region':tuple(region),'weights':weights,'demand':need,'dual':certificate});return True
        try:possible=joint_demands(rows,(2,)*len(components),check,stats=self.stats)
        except _Budget:return False
        if not possible:self.stats['connectivity_joint_failures']+=1;self.reasons.append({'kind':'joint_endpoint_demands','domains':tuple(domains),'region':tuple(region)});return True
        return False
    def _route_impossible(self,domains,check,max_steps=50000):
        active=sum(d.bit_count()>1 for d in domains)
        if not 1<=active<=24:return False
        workcount=0
        def tick():
            nonlocal workcount
            check();workcount+=1
            if workcount>max_steps:raise IsomorphismLimit
        class Work:pass
        work=Work();work.tick=tick;oracle=None
        try:
            model=residual_graph(self.owner,domains,work,2048)
            if model is None:return True
            graph=defaultdict(list);chains=[];obligations=defaultdict(list)
            def edge(a,b,labels=()):graph[a].append((b,labels));graph[b].append((a,labels))
            for vertex,neighbors in enumerate(model.adj):
                for other,color in neighbors.items():
                    if vertex<other and color==5:edge(vertex,other)
                    if vertex<other and color==6:chains.append((vertex,other))
            for choice,label in model.choices.items():
                for segment,color in model.adj[choice].items():
                    if color==3:
                        ports=[v for v,c in model.adj[segment].items() if c==4]
                        edge(*ports,(label,));obligations[label].append(tuple(ports))
            if not chains or len(chains)>16 or len(graph)>256:return False
            def compatible(assignment):
                return all(a not in assignment or b not in assignment or self.owner.masks[assignment[a]][sa]==reverse_mask(self.owner.masks[assignment[b]][sb]) for a,sa,b,sb in self.problem.edges)
            oracle=ObligationChainOracle(graph,chains,obligations,compatible,max(0,max_steps-workcount));possible=oracle.possible(check)
            if possible is None:self.stats['connectivity_route_unknown']+=1
            if possible is False:self.stats['connectivity_route_failures']+=1;return True
            return False
        except IsomorphismLimit:self.stats['connectivity_route_unknown']+=1;return False
        finally:
            self.stats['connectivity_route_graph_steps']+=workcount
            if oracle is not None:
                self.stats['connectivity_refinement_rounds']+=oracle.refinements;self.stats['connectivity_refinement_obligations']+=oracle.added
                self.stats['connectivity_route_steps']+=oracle.steps;self.stats['connectivity_route_label_checks']+=oracle.label_tests
    def prune(self,domains,check=lambda:None):
        self._scope(domains)
        if self.owner.target!='gold' or not all(domains):return False
        state=self.graph.update(domains,check);before=dict(self.blocks.stats);cert=self.blocks.update(state.edges,check)
        for key,value in self.blocks.stats.items():self.stats[key]+=value-before[key]
        mappings=[];common=None
        for cell,domain in enumerate(domains):
            mapping={}
            for code in values(domain):
                hosting=set()
                for port in self.graph.code_ports[code]:
                    edge_id=state.seam_ids[33*cell+port]
                    hosting.add(None if edge_id<0 else cert['blocks'].get(tuple(sorted(state.edges[edge_id]))))
                if len(hosting)==1 and None not in hosting:
                    block=next(iter(hosting));mapping[block]=mapping.get(block,0)|(1<<code)
            mappings.append(mapping);common=set(mapping) if common is None else common&set(mapping)
        updated=[sum(mapping[b] for b in common) for mapping in mappings];check()
        changed=updated!=list(domains)
        if not all(updated):self.reasons=[{'kind':'possible_graph_blocks','domains':tuple(domains)}]
        else:
            region=tuple(sorted(range(self.problem.n),key=lambda c:(updated[c].bit_count(),c))[:min(6,self.problem.n-1)])
            if region and self._supply_impossible(updated,region,check):updated[0]=0;changed=True
            elif self._route_impossible(updated,check):
                self.reasons.append({'kind':'all_strand_routing','domains':tuple(updated)});updated[0]=0;changed=True
        check();self.stats['connectivity_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,updated));domains[:]=updated
        self.reasons=[{'schema':1,'binding':_digest(self.semantic),**reason} for reason in self.reasons[-16:]]
        return changed



def verify_joint_demands(rows,demands,check=lambda:None,*,max_states=100000):
    """Independent row-recursive check of a completed joint-supply failure."""
    demands=tuple(demands);rows=tuple(tuple((tile,tuple(vector)) for tile,vector in row) for row in rows)
    if any(type(v)is not int or v<0 for v in demands) or any(type(tile)is not int or tile<0 or len(vector)!=len(demands) or any(type(v)is not int or v<0 for v in vector) for row in rows for tile,vector in row):raise ValueError('Invalid joint demand proof inputs.')
    order=tuple(sorted(range(len(rows)),key=lambda i:len(rows[i])));failed=set();steps=0
    def visit(depth,used,needs):
        nonlocal steps
        check();steps+=1
        if steps>max_states:raise _Budget
        if depth==len(order):return not any(needs)
        key=depth,used,needs
        if key in failed:return False
        for tile,vector in rows[order[depth]]:
            if tile in used:continue
            next_needs=tuple(max(0,n-v) for n,v in zip(needs,vector))
            if visit(depth+1,used|{tile},next_needs):return True
        failed.add(key);return False
    possible=visit(0,frozenset(),demands)
    return not possible,steps


def verify_connectivity_reason(owner,record,check=lambda:None,*,stats=None):
    """No unchecked diagnostic dictionary is a proof; replay its actual scope."""
    callback=check
    if stats is not None:stats['connectivity_reason_checks']=stats.get('connectivity_reason_checks',0)+1
    def check():
        callback()
        if stats is not None:stats['connectivity_reason_steps']=stats.get('connectivity_reason_steps',0)+1
    check()
    if type(record)is not dict or type(record.get('schema'))is not int or record.get('schema')!=1 or record.get('binding')!=_digest(source_binding(owner)):
        raise ValueError('Connectivity reason source differs.')
    domains=_domains(owner.problem,record.get('domains',()));kind=record.get('kind')
    helper=ConnectivityObligations(owner)
    if kind=='endpoint_assignment':
        if set(record)!={'schema','binding','kind','domains','region','weights','demand','dual'}:raise ValueError('Invalid assignment reason schema.')
        region=tuple(record['region'])
        if not region or len(region)!=len(set(region)) or any(type(c)is not int or not 0<=c<owner.problem.n for c in region):raise ValueError('Invalid capacity region.')
        components=helper.supply._components(domains,region,check);rows=helper._capacities(domains,region,components,check)
        weights=record['weights']
        if len(weights)!=len(components) or any(type(v)is not int or v<0 for v in weights) or type(record['demand'])is not int or record['demand']!=2*sum(weights):raise ValueError('Invalid weighted endpoint demand.')
        matrix=[]
        for choices in rows:
            row=[None]*owner.problem.n
            for tile,vector in choices:
                value=sum(w*v for w,v in zip(weights,vector));row[tile]=value if row[tile] is None else max(row[tile],value)
            matrix.append(row)
        verify_assignment_bound(matrix,record['dual'],check)
        return record['dual']['bound']<record['demand']
    if kind=='joint_endpoint_demands':
        if set(record)!={'schema','binding','kind','domains','region'}:raise ValueError('Invalid joint demand reason schema.')
        region=tuple(record['region'])
        if not region or len(region)!=len(set(region)) or any(type(c)is not int or not 0<=c<owner.problem.n for c in region):raise ValueError('Invalid joint capacity region.')
        components=helper.supply._components(domains,region,check)
        rows=helper._capacities(domains,region,components,check)
        return verify_joint_demands(rows,(2,)*len(components),check)[0]
    if kind=='possible_graph_blocks':
        if set(record)!={'schema','binding','kind','domains'}:raise ValueError('Invalid graph reason schema.')
        from constraint_advanced import cyclic_edge_blocks
        graph=PossibleGoldGraph(owner.problem);state=graph.update(domains,check)
        # Independent original full traversal, not the new delta-cache result.
        return any(not support for cell,support in graph.blocks(state,cyclic_edge_blocks,check))
    if kind=='all_strand_routing':
        if set(record)!={'schema','binding','kind','domains'}:raise ValueError('Invalid route reason schema.')
        # Reconstruct the entire labelled graph and exhaust the bounded exact
        # obligation search again; a cap returns false, never proof admission.
        return helper._route_impossible(domains,check)
    raise ValueError('Unknown connectivity proof mechanism.')
