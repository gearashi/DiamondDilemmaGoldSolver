"""Exact symbolic regional primitives; resource limits never mean impossible.

R24 experiments: direct diagram compilation, ZDD inventory products, exact
subset convolution and representative endpoint matchings. The routines expose
operation counts so reduced enumeration can be measured independently of time.
"""
from collections import OrderedDict
from functools import lru_cache
from itertools import combinations
from constraint_separator import _Budget
from constraint_pruning import values


class DirectRegionDiagram:
    """Compile a relation from sufficient frontier states, before listing rows.

    A node is keyed by level, used physical inventory, every exposed seam mask,
    exposed strand matching, and closed-cycle count. Equal future subproblems
    therefore share computation. The returned DAG retains labelled edges, so
    every concrete cell history and its exact final signature remains available.
    max_rows is retained as a compatibility spelling for the live state budget;
    max_histories optionally limits expansion by consumers needing explicit rows.
    """
    def __init__(self, owner, cells, domains, check=lambda:None,
                 max_rows=4096, max_steps=200000, max_histories=None):
        session=DirectRegionSession(owner,cells,domains,max_rows=max_rows,max_steps=max_steps)
        while not session.advance(256,check):pass
        self.__dict__.update(session.result())
        if max_histories is not None and self.rows>max_histories:raise _Budget

    def histories(self,check=lambda:None):
        def walk(node,codes):
            check()
            if node in self.terminals:
                yield tuple(codes),self.terminals[node];return
            for code,child in self.nodes[node][1]:yield from walk(child,(*codes,code))
        if self.root is not None:yield from walk(self.root,())

    def supports(self,check=lambda:None):
        result=[0]*len(self.cells);seen=set()
        def walk(node):
            check()
            if node in seen or node in self.terminals:return
            seen.add(node);level,edges=self.nodes[node]
            for code,child in edges:result[level]|=1<<code;walk(child)
        if self.root is not None:walk(self.root)
        return tuple(result)


class InventoryZDD:
    """Canonical zero-suppressed families of finite physical-piece subsets."""
    def __init__(self,max_nodes=50000,check=lambda:None):
        self.nodes={};self.unique={};self.max_nodes=max_nodes;self.check=check
        self.operations=0;self._union={};self._product={}
    def _tick(self):self.operations+=1;self.check()
    def node(self,var,low,high):
        self._tick()
        if not high:return low
        key=(int(var),low,high)
        if key not in self.unique:
            if len(self.nodes)>=self.max_nodes:raise _Budget
            ident=len(self.nodes)+2;self.unique[key]=ident;self.nodes[ident]=key
        return self.unique[key]
    def top(self,root):return self.nodes[root][0] if root>1 else 10**30
    def split(self,root,var):
        return self.nodes[root][1:] if self.top(root)==var else (root,0)
    def singleton(self,mask):
        if type(mask)is not int or mask<0:raise ValueError('Inventory masks must be nonnegative integers.')
        root=1
        for var in reversed(tuple(values(mask))):root=self.node(var,0,root)
        return root
    def from_sets(self,masks):
        result=0
        for mask in masks:result=self.union(result,self.singleton(mask))
        return result
    def union(self,left,right):
        self._tick()
        if left==right or not right:return left
        if not left:return right
        key=tuple(sorted((left,right)))
        if key in self._union:return self._union[key]
        var=min(self.top(left),self.top(right))
        a,b=self.split(left,var);c,d=self.split(right,var)
        result=self.node(var,self.union(a,c),self.union(b,d));self._union[key]=result
        return result
    def disjoint_product(self,left,right):
        self._tick()
        if not left or not right:return 0
        if left==1:return right
        if right==1:return left
        key=tuple(sorted((left,right)))
        if key in self._product:return self._product[key]
        var=min(self.top(left),self.top(right));a,b=self.split(left,var);c,d=self.split(right,var)
        low=self.disjoint_product(a,c)
        high=self.union(self.disjoint_product(b,c),self.disjoint_product(a,d))
        result=self.node(var,low,high);self._product[key]=result;return result
    def exists_disjoint(self,root,forbidden):
        memo={}
        def visit(node):
            self._tick()
            if node<2:return bool(node)
            if node in memo:return memo[node]
            var,low,high=self.nodes[node]
            answer=visit(low) or (not(forbidden>>var&1) and visit(high))
            memo[node]=answer;return answer
        return visit(root)
    def sets(self,root):
        def visit(node,mask):
            self.check()
            if node==1:yield mask
            elif node:
                var,low,high=self.nodes[node]
                yield from visit(low,mask);yield from visit(high,mask|(1<<var))
        return tuple(visit(root,0))
    def count(self,root):
        memo={0:0,1:1}
        def visit(node):
            self.check()
            if node not in memo:
                _,low,high=self.nodes[node];memo[node]=visit(low)+visit(high)
            return memo[node]
        return visit(root)


def inventory_supports(groups,check=lambda:None,max_nodes=50000):
    """Exact global disjoint-inventory supports, without a Cartesian product."""
    zdd=InventoryZDD(max_nodes,check);roots=[zdd.from_sets(g) for g in groups]
    before=[1]
    for root in roots:before.append(zdd.disjoint_product(before[-1],root))
    after=[1]*(len(roots)+1)
    for i in range(len(roots)-1,-1,-1):after[i]=zdd.disjoint_product(roots[i],after[i+1])
    result=[]
    for i,group in enumerate(groups):
        others=zdd.disjoint_product(before[i],after[i+1])
        result.append({mask for mask in group if zdd.exists_disjoint(others,mask)})
    return result,{'nodes':len(zdd.nodes),'operations':zdd.operations,
                   'union_inventories':zdd.count(before[-1])}


def subset_convolution(left,right,n,check=lambda:None):
    """Exact ranked zeta/Mobius disjoint convolution over an n-piece universe.

    Integer counts are essential: Mobius subtraction is invalid for naive
    Boolean unions. This routine does not silently use modular zero as absence.
    """
    if not 0<=n<=16:raise ValueError('Subset convolution is restricted to at most 16 pieces.')
    size=1<<n
    if len(left)!=size or len(right)!=size:raise ValueError('Expected one integer per subset.')
    if any(type(x)is not int or x<0 for x in (*left,*right)):
        raise ValueError('Subset counts must be nonnegative integers.')
    ranks_a={m.bit_count() for m,x in enumerate(left) if x};ranks_b={m.bit_count() for m,x in enumerate(right) if x}
    a={rank:[0]*size for rank in ranks_a};b={rank:[0]*size for rank in ranks_b};operations=0
    for mask in range(size):
        if left[mask]:a[mask.bit_count()][mask]=left[mask]
        if right[mask]:b[mask.bit_count()][mask]=right[mask]
    for table in (a,b):
        for row in table.values():
            for bit in range(n):
                check();flag=1<<bit
                for mask in range(size):
                    if mask&flag:row[mask]+=row[mask^flag];operations+=1
    out={rank:[0]*size for rank in {r+s for r in a for s in b if r+s<=n}}
    for r,left_row in a.items():
        for s,right_row in b.items():
            check()
            if r+s>n:continue
            row=out[r+s]
            for mask in range(size):row[mask]+=left_row[mask]*right_row[mask];operations+=1
    for row in out.values():
        for bit in range(n):
            check();flag=1<<bit
            for mask in range(size):
                if mask&flag:row[mask]-=row[mask^flag];operations+=1
    return tuple(out[mask.bit_count()][mask] if mask.bit_count() in out else 0 for mask in range(size)),operations


def endpoint_matchings(endpoints):
    endpoints=tuple(endpoints)
    if len(endpoints)%2:raise ValueError('An endpoint matching requires an even number of endpoints.')
    if not endpoints:yield ();return
    first=endpoints[0]
    for index in range(1,len(endpoints)):
        other=endpoints[index]
        for tail in endpoint_matchings(endpoints[1:index]+endpoints[index+1:]):
            yield ((first,other),*tail)


def single_cycle(left,right):
    """Whether two perfect matchings form exactly one connected cycle."""
    vertices={x for pair in left for x in pair}
    if not vertices or vertices!={x for pair in right for x in pair}:return False
    if len(left)*2!=len(vertices) or len(right)*2!=len(vertices):raise ValueError('Not a perfect matching.')
    parent={x:x for x in vertices}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    for a,b in (*left,*right):parent[find(a)]=find(b)
    return len({find(x) for x in vertices})==1


def representative_matchings(matchings,check=lambda:None,max_endpoints=10):
    """GF(2) row basis preserving existence for every outside matching.

    Use only inside one identical inventory/boundary class. This preserves
    existence, NOT all histories, counts, or candidate supports unconditionally.
    """
    rows=tuple(dict.fromkeys(tuple(sorted(tuple(sorted(p)) for p in row)) for row in matchings))
    if not rows:return (),{'rows':0,'representatives':0,'columns':0,'operations':0}
    vertices=tuple(sorted(x for pair in rows[0] for x in pair))
    if len(vertices)>max_endpoints:raise _Budget
    if len(set(vertices))!=len(vertices):raise ValueError('Repeated endpoint.')
    if any(tuple(sorted(x for pair in row for x in pair))!=vertices for row in rows):
        raise ValueError('Representatives require the same complete endpoint set.')
    columns=tuple(endpoint_matchings(vertices));basis={};selected=[];operations=0
    for row in rows:
        check();vector=0
        for index,outside in enumerate(columns):
            operations+=1
            if single_cycle(row,outside):vector|=1<<index
        while vector:
            pivot=vector.bit_length()-1
            if pivot in basis:vector^=basis[pivot];operations+=1
            else:basis[pivot]=vector;selected.append(row);break
    return tuple(selected),{'rows':len(rows),'representatives':len(selected),
                            'columns':len(columns),'operations':operations}


def join_diagrams(owner,left,right,domains,check=lambda:None,max_steps=50000):
    """Exact overlap join on DAG states; never expand all concrete histories."""
    from constraint_region_network import assignment_signature,joined_signature
    cells=tuple(sorted(set(left.cells)|set(right.cells)));ls=set(left.cells);rs=set(right.cells)
    memo={};singletons={};steps=0;hits=0
    def choices(diagram,node,cell,present):
        if cell not in present:return {None:node}
        if node is None or node in diagram.terminals:return {}
        level,edges=diagram.nodes[node]
        if diagram.cells[level]!=cell:raise ValueError('Diagram levels do not match sorted cells.')
        return dict(edges)
    def walk(level,ln,rn,signature):
        nonlocal steps,hits
        check();key=(level,ln,rn,signature)
        if key in memo:hits+=1;return memo[key]
        if ln is None or rn is None:return (0,(0,)*len(cells))
        if level==len(cells):return (1,(0,)*len(cells))
        cell=cells[level];le=choices(left,ln,cell,ls);re=choices(right,rn,cell,rs)
        codes=(set(le)&set(re)) if cell in ls and cell in rs else set(le if cell in ls else re)
        count=0;support=[0]*len(cells)
        for code in sorted(codes):
            steps+=1
            if steps>max_steps:raise _Budget
            skey=(cell,code)
            if skey not in singletons:singletons[skey]=assignment_signature(owner,(cell,),(code,),domains,check)
            single=singletons[skey]
            if single is None:continue
            following=single if level==0 else joined_signature(owner,cells[:level],signature,(cell,),single,domains,check)
            if following is None:continue
            number,child=walk(level+1,le[code] if cell in ls else ln,re[code] if cell in rs else rn,following)
            if number:
                count+=number;support[level]|=1<<code
                support=[a|b for a,b in zip(support,child)]
        result=count,tuple(support);memo[key]=result;return result
    count,support=walk(0,left.root,right.root,None)
    return dict(zip(cells,support)),{'histories':count,'steps':steps,'memo_hits':hits,'states':len(memo)}


def subset_inventory_supports(groups,check=lambda:None,max_pieces=10):
    universe=tuple(sorted(set().union(*(set(values(mask)) for group in groups for mask in group))))
    if len(universe)>max_pieces:raise _Budget
    offsets={piece:i for i,piece in enumerate(universe)};size=1<<len(universe)
    encode=lambda mask:sum(1<<offsets[piece] for piece in values(mask))
    arrays=[]
    for group in groups:
        row=[0]*size
        for mask in group:row[encode(mask)]=1
        arrays.append(row)
    identity=[0]*size;identity[0]=1;before=[tuple(identity)];operations=0
    for row in arrays:
        combined,work=subset_convolution(before[-1],row,len(universe),check);operations+=work;before.append(combined)
    after=[None]*(len(arrays)+1);after[-1]=tuple(identity)
    for i in range(len(arrays)-1,-1,-1):
        after[i],work=subset_convolution(arrays[i],after[i+1],len(universe),check);operations+=work
    supported=[]
    for i,group in enumerate(groups):
        others,work=subset_convolution(before[i],after[i+1],len(universe),check);operations+=work
        sums=list(others)
        for bit in range(len(universe)):
            check()
            for mask in range(size):
                if mask>>bit&1:sums[mask]+=sums[mask^(1<<bit)];operations+=1
        supported.append({mask for mask in group if sums[(size-1)^encode(mask)]>0})
    return supported,{'operations':operations,'pieces':len(universe),'allocations':sum(before[-1])}


def representative_component_feasible(owner,domains,components,tables,check=lambda:None,max_steps=50000):
    """Existence-only regional join after verified matching-rank reduction."""
    from constraint_region_network import joined_signature
    if owner.target!='gold':raise ValueError('Connectivity representatives require the Gold target.')
    reduced=[];stats={'representative_rows':0,'representative_kept':0,'representative_steps':0,'representative_join_steps':0}
    for table in tables:
        groups={}
        for signature in table:
            used,masks,pairs,cycles=signature;groups.setdefault((used,masks,cycles),[]).append(signature)
        selected=[]
        for group in groups.values():
            check();stats['representative_rows']+=len(group)
            if len(group)==1 or not group[0][2] or len(group[0][2])>5:selected.extend(group);continue
            representatives,work=representative_matchings([signature[2] for signature in group],check)
            stats['representative_steps']+=work['operations'];keep=set(representatives)
            selected.extend(signature for signature in group if signature[2] in keep)
        stats['representative_kept']+=len(selected);reduced.append(selected)
    failed=set()
    def walk(index,members,signature):
        check();stats['representative_join_steps']+=1
        if stats['representative_join_steps']>max_steps:raise _Budget
        if index==len(components):return True
        key=(index,signature)
        if key in failed:return False
        for other in reduced[index]:
            merged=other if index==0 else joined_signature(owner,members,signature,components[index],other,domains,check)
            if merged is not None and walk(index+1,tuple(sorted((*members,*components[index]))),merged):return True
        failed.add(key);return False
    try:return walk(0,(),None),stats
    except _Budget:return None,stats


class DirectRegionSession:
    """Resumable in-process compiler, bound to immutable exact query inputs.

    Soft work slices suspend the generator at transition boundaries; a stop
    callback is polled outside the generator so it cannot destroy continuation.
    Complete results alone become authoritative. This is not a portable disk
    checkpoint format: a process restart must reconstruct this query.
    """
    def __init__(self,owner,cells,domains,max_rows=4096,max_steps=200000):
        self.owner=owner;self.cells=tuple(sorted(set(cells)));self.domains=tuple(domains)
        if not self.cells or any(not 0<=c<owner.problem.n for c in self.cells):raise ValueError('Invalid diagram cells.')
        self.binding=(owner.problem.fingerprint,owner.target,self.cells,self.domains)
        self.max_rows=max_rows;self.max_steps=max_steps;self.complete=False;self.failed=False
        self.nodes={};self.terminals={};self.rows=0;self.steps=0;self.states=0;self.memo_hits=0
        self.root=None;self._counts={};self._memo={};self._intern={};self._singletons={};self._next_id=0
        self._generator=self._compile(0,None)
    def _compile(self,level,signature):
        from constraint_region_network import assignment_signature,joined_signature
        key=(level,signature)
        if key in self._memo:
            self.memo_hits+=1;yield;return self._memo[key]
        self.states+=1
        if self.states>self.max_rows:raise _Budget
        yield
        if level==len(self.cells):
            structural=('terminal',signature);ident=self._intern.get(structural)
            if ident is None:
                ident=self._next_id;self._next_id+=1;self._intern[structural]=ident
                self.terminals[ident]=signature;self._counts[ident]=1
            self._memo[key]=ident;return ident
        cell=self.cells[level];edges=[]
        for code in values(self.domains[cell]):
            self.steps+=1
            if self.steps>self.max_steps:raise _Budget
            yield
            skey=(cell,code)
            if skey not in self._singletons:
                self._singletons[skey]=assignment_signature(self.owner,(cell,),(code,),self.domains,lambda:None)
            single=self._singletons[skey]
            if single is None:continue
            following=single if level==0 else joined_signature(self.owner,self.cells[:level],signature,(cell,),single,self.domains,lambda:None)
            if following is None:continue
            child=yield from self._compile(level+1,following)
            if child is not None:edges.append((code,child))
        if not edges:self._memo[key]=None;return None
        structural=(level,tuple(edges));ident=self._intern.get(structural)
        if ident is None:
            ident=self._next_id;self._next_id+=1;self._intern[structural]=ident
            self.nodes[ident]=(level,tuple(edges));self._counts[ident]=sum(self._counts[c] for _,c in edges)
        self._memo[key]=ident;return ident
    def advance(self,quantum=256,check=lambda:None):
        if type(quantum)is not int or quantum<1:raise ValueError('A work slice must be a positive integer.')
        if self.failed:raise _Budget
        if self.complete:return True
        for _ in range(quantum):
            check()
            try:next(self._generator)
            except StopIteration as done:
                self.root=done.value;self.rows=0 if self.root is None else self._counts[self.root]
                self.complete=True;return True
            except _Budget:self.failed=True;raise
        return False
    def result(self):
        if not self.complete:raise ValueError('An unfinished diagram is not an authoritative relation.')
        return {name:getattr(self,name) for name in ('cells','nodes','terminals','rows','steps','states','memo_hits','root','_counts')}
