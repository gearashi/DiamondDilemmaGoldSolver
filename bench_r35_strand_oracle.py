"""Certify Hamiltonian necessary matching obstructions on an optimistic strand graph."""
from collections import defaultdict

from r27_structure_filters import Union, Work, WorkLimit


class Blossom:
    """Maximum cardinality matching and even-reachable vertices in a blossom forest."""
    def __init__(self, graph, work):
        self.graph=graph
        self.work=work
        self.match=[-1]*len(graph)
        for v,row in enumerate(graph):
            if self.match[v]<0:
                for u in row:
                    if u!=v and self.match[u]<0:
                        self.match[v]=u
                        self.match[u]=v
                        break

    def forest(self,root):
        n=len(self.graph)
        parent=[-1]*n
        base=list(range(n))
        even={root}
        queue=[root]

        def ancestor(a,b):
            reached=set()
            while True:
                a=base[a]
                reached.add(a)
                if self.match[a]<0:
                    break
                a=parent[self.match[a]]
            while base[b] not in reached:
                b=parent[self.match[base[b]]]
            return base[b]

        def trace(v,child,join,petals):
            while base[v]!=join:
                paired=self.match[v]
                petals.update((base[v],base[paired]))
                parent[v]=child
                child=paired
                v=parent[paired]

        for v in queue:
            self.work.tick('strand_matching_edges',len(self.graph[v])+1)
            for u in self.graph[v]:
                if base[u]==base[v] or self.match[v]==u:
                    continue
                if u==root or (self.match[u]>=0 and parent[self.match[u]]>=0):
                    join=ancestor(v,u)
                    petals=set()
                    trace(v,u,join,petals)
                    trace(u,v,join,petals)
                    for i in range(n):
                        if base[i] in petals:
                            base[i]=join
                            if i not in even:
                                even.add(i)
                                queue.append(i)
                elif parent[u]<0:
                    parent[u]=v
                    if self.match[u]<0:
                        return u,parent,even
                    mate=self.match[u]
                    if mate not in even:
                        even.add(mate)
                        queue.append(mate)
        return -1,parent,even

    def maximize(self):
        for root in range(len(self.graph)):
            if self.match[root]>=0:
                continue
            v,parent,_=self.forest(root)
            while v>=0:
                p=parent[v]
                u=self.match[p]
                self.match[v],self.match[p]=p,v
                v=u
        return [i for i,m in enumerate(self.match) if m<0]

    def barrier(self,exposed):
        reachable=set()
        for v in exposed:
            _,_,even=self.forest(v)
            reachable.update(even)
        return {u for v in reachable for u in self.graph[v]}-reachable


def verify_obstruction(graph,certificate):
    """Exact independent odd-component count after the optional named vertex deletion."""
    removed=certificate.get('vertex')
    barrier=set(certificate['barrier'])
    if ((removed is not None and (len(graph)%2!=1 or removed not in range(len(graph))))
            or removed in barrier or any(i not in range(len(graph)) for i in barrier)):
        return False
    unseen=set(range(len(graph)))-barrier
    if removed is not None:
        unseen.remove(removed)
    if len(unseen)+len(barrier)==0 or (len(unseen)+len(barrier))%2:
        return False
    odd=0
    while unseen:
        start=unseen.pop()
        size=0
        stack=[start]
        while stack:
            v=stack.pop();size+=1
            for u in graph[v]:
                if u in unseen:
                    unseen.remove(u);stack.append(u)
        odd+=size%2
    return odd>len(barrier)


def matching_condition(graph,work):
    """Return a checked obstruction, or no cut if the condition passes or work expires."""
    clean=[tuple(sorted(v for v in row if v!=i)) for i,row in enumerate(graph)]
    n=len(clean)
    if n<=2:
        return None,'trivial'
    try:
        oracle=Blossom(clean,work)
        exposed=oracle.maximize()
        if n%2:
            if len(exposed)==1:
                _,_,even=oracle.forest(exposed[0])
                if len(even)==n:
                    return None,'passes'
                v=next(i for i in range(n) if i not in even)
            else:
                v=exposed[0]
            names=[i for i in range(n) if i!=v]
            idx={name:i for i,name in enumerate(names)}
            smaller=[tuple(idx[u] for u in clean[name] if u!=v) for name in names]
            residual=Blossom(smaller,work)
            free=residual.maximize()
            cert=dict(vertex=v,barrier=sorted(names[i] for i in residual.barrier(free)))
        else:
            if not exposed:
                return None,'passes'
            cert=dict(vertex=None,barrier=sorted(oracle.barrier(exposed)))
    except WorkLimit:
        return None,'unknown'
    if not verify_obstruction(clean,cert):
        raise ArithmeticError("Unverified strand matching obstruction.")
    return cert,'obstruction'


def physical_graph(problem,options,required):
    """Contract fixed open strand chains; adjacency uses only the still-exposed endpoints."""
    endpoint={}
    next_id=0
    for tile,record in enumerate(problem.tiles):
        for segment in record['segments']:
            for side,k in segment:
                endpoint[3*tile+side,k]=next_id
            next_id+=1
    uf=Union(next_id)
    slots=defaultdict(list)
    for (d,k) in endpoint:
        slots[d].append(k)
    fixed_darts=set()
    degrees=[0]*next_id
    for a,b in required:
        fixed_darts.update((a,b))
        for k in slots[a]:
            left=endpoint[a,k];right=endpoint[b,12-k]
            uf.join(left,right)
            degrees[left]+=1;degrees[right]+=1
    ids={}
    owners=[]
    for i in range(next_id):
        root=uf.find(i)
        owners.append(ids.setdefault(root,len(ids)))
    opened=[0]*len(ids)
    for i,degree in enumerate(degrees):
        opened[owners[i]]+=2-degree
    proper=bool(len(ids)>1 and 0 in opened)
    graph=[set() for _ in ids]
    for a,b in options:
        if a in fixed_darts or b in fixed_darts:
            continue
        for k in slots[a]:
            left=owners[endpoint[a,k]];right=owners[endpoint[b,12-k]]
            if left!=right:
                graph[left].add(right);graph[right].add(left)
    return graph,proper


def check_sewing(problem,options,required,work):
    graph,proper=physical_graph(problem,options,required)
    if proper:
        return dict(result='proper_circuit',vertices=len(graph))
    certificate,result=matching_condition(graph,work)
    record=dict(result=result,vertices=len(graph))
    if certificate is not None:
        record['certificate']=certificate
    return record
