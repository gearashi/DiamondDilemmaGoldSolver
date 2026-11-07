"""Exact reversible support counts for the CPU possible Gold graph.

Snapshots contain only derived in-process state, never imported proof claims.
Strand/component traversals remain complete; shared support construction updates
only changed placements and incident seams, with exact saved-state rollback.
"""
from collections import OrderedDict
from dataclasses import dataclass
from constraint_relations import bits

STAT_KEYS=('gold_graph_builds','gold_graph_updates','gold_graph_snapshot_hits','gold_graph_local_hits',
           'gold_graph_changed_cells','gold_graph_reused_cells','gold_graph_support_updates',
           'gold_graph_seam_tests','gold_graph_edge_materializations')

@dataclass(frozen=True)
class _Cell:
    domain:int
    counts:tuple
    port_counts:tuple
    pairs:tuple
    ports:int

@dataclass(frozen=True)
class _Graph:
    domains:tuple
    cells:tuple
    seam_masks:tuple
    edges:tuple
    seam_ids:tuple


def _segments(problem):
    return tuple(tuple(tuple(sorted(((a[0]+r)%3*11+a[1]-1,(b[0]+r)%3*11+b[1]-1)))
                       for a,b in tile['segments']) for tile in problem.tiles for r in range(3))


class PossibleGoldGraph:
    def __init__(self,problem,*,limit=16):
        if type(limit)is not int or not 1<=limit<=32:raise ValueError('Invalid Gold graph snapshot bound.')
        self.problem=problem;self.limit=limit;self.n=problem.n;self.mesh=tuple(problem.edges)
        self.segments=_segments(problem);self.heads=tuple(tuple(a for a,b in row) for row in self.segments)
        self.original=tuple(sum(1<<code for code in set(row)) for row in problem.domains)
        self.binding=self._binding();self.snapshots=OrderedDict();self.locals=OrderedDict();self.previous=None
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.catalog=tuple(sorted({pair for row in self.segments for pair in row}));indices={pair:i for i,pair in enumerate(self.catalog)}
        self.code_edges=tuple(tuple(sorted({indices[pair] for pair in row})) for row in self.segments)
        self.code_ports=tuple(tuple(sorted({p for pair in row for p in pair})) for row in self.segments)
        self.incident=[set() for _ in range(self.n)]
        for edge,(a,sa,b,sb) in enumerate(self.mesh):self.incident[a].add(edge);self.incident[b].add(edge)

    def _binding(self):
        return (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),tuple(self.problem.edges),tuple(tuple(map(int,row)) for row in self.problem.masks),_segments(self.problem))

    def _local(self,domain,old,check):
        if domain in self.locals:
            self.stats['gold_graph_local_hits']+=1;self.locals.move_to_end(domain);return self.locals[domain]
        counts=list(old.counts) if old is not None else [0]*len(self.catalog)
        ports=list(old.port_counts) if old is not None else [0]*33
        previous=old.domain if old is not None else 0
        for code in bits(previous^domain):
            check();delta=1 if domain>>code&1 else -1
            for edge in self.code_edges[code]:counts[edge]+=delta;self.stats['gold_graph_support_updates']+=1
            for port in self.code_ports[code]:ports[port]+=delta;self.stats['gold_graph_support_updates']+=1
        if any(value<0 for value in counts) or any(value<0 for value in ports):raise RuntimeError('Gold graph support count underflow.')
        result=_Cell(domain,tuple(counts),tuple(ports),tuple(pair for pair,count in zip(self.catalog,counts) if count),sum(1<<p for p,count in enumerate(ports) if count))
        check();self.locals[domain]=result
        while len(self.locals)>512:self.locals.popitem(last=False)
        return result

    def update(self,domains,check=lambda:None):
        if self._binding()!=self.binding:raise ValueError('Possible Gold graph source binding changed.')
        if len(domains)!=self.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Possible Gold graph domains escape original root.')
        check();key=tuple(domains)
        if key in self.snapshots:
            self.stats['gold_graph_snapshot_hits']+=1;self.snapshots.move_to_end(key);self.previous=self.snapshots[key];return self.previous
        # Use a nearby exact snapshot for both removals and additions. No
        # monotone-domain assumption is needed for sibling/backtrack updates.
        old=min(self.snapshots.values(),key=lambda state:sum((a^b).bit_count() for a,b in zip(key,state.domains)),default=None)
        changed=[c for c in range(self.n) if old is None or key[c]!=old.domains[c]]
        cells=list(old.cells) if old is not None else [None]*self.n
        for cell in changed:check();cells[cell]=self._local(key[cell],cells[cell],check)
        self.stats['gold_graph_changed_cells']+=len(changed);self.stats['gold_graph_reused_cells']+=self.n-len(changed)
        seams=list(old.seam_masks) if old is not None else [0]*len(self.mesh)
        affected=sorted(set().union(*(self.incident[c] for c in changed))) if changed else []
        for edge in affected:
            check();a,sa,b,sb=self.mesh[edge];mask=0
            for port in range(11):
                self.stats['gold_graph_seam_tests']+=1
                if cells[a].ports>>(11*sa+port)&1 and cells[b].ports>>(11*sb+10-port)&1:mask|=1<<port
            seams[edge]=mask
        edges=[]
        for cell,state in enumerate(cells):
            check();base=33*cell;edges.extend((base+a,base+b) for a,b in state.pairs)
        seam_ids=[-1]*(33*self.n)
        for (a,sa,b,sb),mask in zip(self.mesh,seams):
            check()
            for port in bits(mask):
                left=33*a+11*sa+port;right=33*b+11*sb+10-port
                seam_ids[left]=seam_ids[right]=len(edges);edges.append((left,right))
        self.stats['gold_graph_edge_materializations']+=len(edges)
        result=_Graph(key,tuple(cells),tuple(seams),tuple(edges),tuple(seam_ids));check()
        self.snapshots[key]=result;self.previous=result
        while len(self.snapshots)>self.limit:self.snapshots.popitem(last=False)
        self.stats['gold_graph_builds' if old is None else 'gold_graph_updates']+=1
        return result

    def strands(self,state,check=lambda:None):
        parent=list(range(33*self.n))
        def find(node):
            while parent[node]!=node:parent[node]=parent[parent[node]];node=parent[node]
            return node
        for a,b in state.edges:
            check();left=find(a);right=find(b)
            if left!=right:parent[right]=left
        choices=[];common=None
        for cell,domain in enumerate(state.domains):
            check();mapping={};roots={port:find(33*cell+port) for port in bits(state.cells[cell].ports)}
            for code in bits(domain):
                hosting={roots[port] for port in self.heads[code]}
                if len(hosting)==1:
                    root=next(iter(hosting));mapping[root]=mapping.get(root,0)|(1<<code)
            choices.append(mapping);common=set(mapping) if common is None else common.intersection(mapping)
            if not common:return ((cell,0),)
        return tuple((cell,sum(mapping[root] for root in common)) for cell,mapping in enumerate(choices))

    def blocks(self,state,block_function,check=lambda:None):
        blocks=block_function(33*self.n,state.edges,check);choices=[];common=None
        for cell,domain in enumerate(state.domains):
            check();mapping={}
            for code in bits(domain):
                hosting=set()
                for port in self.code_ports[code]:
                    edge=state.seam_ids[33*cell+port];hosting.add(blocks[edge] if edge>=0 else -1)
                if len(hosting)==1 and -1 not in hosting:
                    block=next(iter(hosting));mapping[block]=mapping.get(block,0)|(1<<code)
            choices.append(mapping);common=set(mapping) if common is None else common.intersection(mapping)
            if not common:return ((cell,0),)
        return tuple((cell,sum(mapping[block] for block in common)) for cell,mapping in enumerate(choices))
