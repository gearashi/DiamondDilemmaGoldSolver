"""Complete numeric responses for independent pieces of a rotation factor graph."""
from collections import OrderedDict, namedtuple

import numpy as np


DeferredTuples=namedtuple('DeferredTuples','source order perms')


class ComponentCache:
    def __init__(self,limit,*,max_cells=160,normalize=True):
        self.limit=limit
        self.max_cells=max_cells
        self.normalize=normalize
        self.frames=3
        self.vertices=3
        self.defer=True
        self.rows=OrderedDict()
        self.bytes=0

    def get(self,key):
        row=self.rows.get(key)
        if row is not None:self.rows.move_to_end(key)
        return row

    def store(self,key,score,trace):
        # Include the serialized key, every ternary response mask and a per-row
        # allowance. A response is kept only if it fits the chosen optional budget.
        size=len(repr(key).encode('ascii'))+64+sum(32+int(np.size(alts)) for _,_,alts in trace)
        if size>self.limit:return
        rows=tuple(np.asarray(alts,dtype=np.uint8) for _,_,alts in trace)
        self._put(key,score,rows,size)

    def remember_source(self,key,score,source,order,perms):
        ref=DeferredTuples(source,order,perms)
        size=len(repr((key,ref)).encode('ascii'))+64
        self._put(key,score,ref,size)

    def _put(self,key,score,value,size):
        if size>self.limit:return
        previous=self.rows.pop(key,None)
        if previous is not None:self.bytes-=previous[2]
        while self.bytes+size>self.limit:
            _,(_,_,old)=self.rows.popitem(last=False)
            self.bytes-=old
        self.rows[key]=(score,value,size)
        self.bytes+=size


def rotation_frame(key):
    unary,raw,scopes=key
    tables=[(a,b,np.frombuffer(data,dtype='<i4').reshape(len(unary[a]),len(unary[b]))) for a,b,data in raw]
    perms=[]
    for c,u in enumerate(unary):
        incident=[v if a==c else v.T for a,b,v in tables if c in (a,b)]
        labels=[(x,tuple(tuple(sorted(map(int,tab[r]))) for tab in incident)) for r,x in enumerate(u)]
        perms.append(tuple(sorted(range(len(u)),key=labels.__getitem__)))
    canonical=(tuple(tuple(u[i] for i in perm) for u,perm in zip(unary,perms)),
               tuple((a,b,v[np.ix_(perms[a],perms[b])].tobytes()) for a,b,v in tables),scopes)
    return canonical,perms


def translate_responses(tables,scopes,perms):
    # Each permutation maps output rotation ranks to source ranks. Axis order
    # and every bit in each conditional tie set move with its physical variable.
    result=[]
    for i,(table,scope) in enumerate(zip(tables,scopes)):
        row=np.asarray(table,np.uint8)
        for axis,q in enumerate(scope):
            row=np.take(row,perms[q],axis=axis)
        mapped=np.zeros(row.shape,np.uint8)
        for r,previous in enumerate(perms[i]):
            mapped|=((row>>np.uint8(previous))&np.uint8(1))<<np.uint8(r)
        result.append(mapped)
    return result


def vertex_frame(key):
    unaries,raw,scopes=key
    edges=[(a,b,np.frombuffer(data,dtype='<i4').reshape(len(unaries[a]),len(unaries[b]))) for a,b,data in raw]
    def signature(c):
        incident=[tuple(sorted(map(int,v.ravel()))) for a,b,v in edges if c in (a,b)]
        return tuple(sorted(unaries[c])),len(incident),tuple(sorted(incident))
    order=tuple(sorted(range(len(unaries)),key=signature))
    index={c:i for i,c in enumerate(order)}
    moved=[]
    for a,b,v in edges:
        a,b=index[a],index[b]
        if a>b:a,b,v=b,a,v.T
        moved.append((a,b,v.tobytes()))
    identity,perms=rotation_frame((tuple(unaries[c] for c in order),tuple(sorted(moved)),()))
    return ('joint',identity),order,perms


def optimal_tuples(cells,widths,order,perms,rows):
    current={}
    answers=[]
    def visit(i):
        if i<0:
            answers.append([perms[j].index(current[cells[q]]) for j,q in enumerate(order)])
            return
        c,rest,masks=rows[i]
        allowed=int(masks[tuple(current[z] for z in rest)])
        for r in range(widths[c]):
            if allowed&(1<<r):
                current[c]=r
                visit(i-1)
    visit(len(rows)-1)
    return np.asarray(answers,np.uint8)


def tuple_responses(cells,widths,steps,order,perms,winners):
    # Project each exact tuple onto the elimination separator. Within a
    # disconnected component VE makes earlier choices conditional only on this
    # separator, so independent recursive reconstruction preserves every tie.
    rows=[(c,tuple(z for z in scope if z!=c),
           np.zeros(tuple(widths[z] for z in scope if z!=c),np.uint8)) for _,c,scope,_ in steps]
    for win in winners:
        current={cells[q]:perms[j][r] for j,(q,r) in enumerate(zip(order,win))}
        for c,rest,table in rows:
            table[tuple(current[z] for z in rest)]|=np.uint8(1<<current[c])
    return [a for _,_,a in rows]


def solve_components(oracle,widths,unary,edges,plan,cap):
    # The order of disconnected eliminations has no numerical effect. Put each
    # tied response back at its original global trace index for identical DFS witnesses.
    n=len(widths)
    roots=list(range(n))
    def find(c):
        while roots[c]!=c:
            roots[c]=roots[roots[c]]
            c=roots[c]
        return c
    for (a,b),_ in edges:
        a,b=find(a),find(b)
        if a!=b:roots[b]=a
    groups={}
    for index,(c,scope,amount) in enumerate(plan):
        groups.setdefault(find(c),[]).append((index,c,scope,amount))
    local_edges={g:[] for g in groups}
    for scope,values in edges:local_edges[find(scope[0])].append((scope,values))
    cache=oracle.component_cache
    total=0
    trace=[None]*n
    for root,steps in groups.items():
        if oracle.poll is not None:oracle.poll()
        if len(steps)==1:
            idx,c,_,_=steps[0]
            vals=unary[c]
            high=int(vals[0]) if len(vals)==1 else int(vals.max())
            bits=1 if len(vals)==1 else sum(1<<i for i,x in enumerate(vals) if x==high)
            trace[idx]=(c,(),np.uint8(bits))
            total+=high
            oracle.work['singleton_responses']+=1
            # No general factor grid is materialized for an isolated unary.
            oracle.work['avoided']+=len(vals)
            continue
        cells=[s[1] for s in steps]
        pairs=local_edges[root]
        key=None
        saved=None
        canonical=None
        perms=None
        vertex_order=None
        offset=0
        if len(cells)<n and len(cells)<=cache.max_cells and cache.limit:
            index={c:i for i,c in enumerate(cells)}
            # Map the actual domain ranks and numeric table axes together. The
            # current board maps ranks back to its own physical rotations at decode.
            numeric=[]
            for (a,b),v in pairs:
                a,b=index[a],index[b]
                if a>b:a,b,v=b,a,v.T
                numeric.append((a,b,v.astype('<i4').tobytes()))
            u=[]
            for c in cells:
                value=unary[c]
                base=int(value[0]) if cache.normalize else 0
                offset+=base
                u.append(tuple(int(x)-base for x in value))
            key=(tuple(u),tuple(sorted(numeric)),tuple(tuple(index[z] for z in scope if z!=c) for _,c,scope,_ in steps))
            saved=cache.get(key)
            oracle.work['component_keys']+=1
            if saved is None and len(cells)<=cache.vertices:
                canonical,vertex_order,perms=vertex_frame(key)
                oracle.work['component_reindex']+=1
                alternative=cache.get(canonical)
                if alternative is not None:
                    value,payload,_=alternative
                    if isinstance(payload,DeferredTuples):
                        donor=cache.get(payload.source)
                        payload_value=None
                        if donor is not None:
                            shape,_,old_scopes=payload.source
                            old_rows=[(q,rest,a) for q,(rest,a) in enumerate(zip(old_scopes,donor[1]))]
                            tuples=optimal_tuples(range(len(shape)),list(map(len,shape)),payload.order,payload.perms,old_rows)
                            cache.store(canonical,value,[(None,None,tuples)])
                            oracle.work['component_tuples_encoded']+=len(tuples)
                            payload_value=tuples
                    else:
                        payload_value=payload[0]
                    if payload_value is not None:
                        arrays=tuple_responses(cells,widths,steps,vertex_order,perms,payload_value)
                        cache.store(key,value,[(None,None,a) for a in arrays])
                        saved=value,arrays,0
                        oracle.work['component_vertex_hits']+=1
                        oracle.work['component_tuples_decoded']+=len(payload_value)
            elif saved is None and len(cells)<=cache.frames:
                canonical,perms=rotation_frame(key)
                oracle.work['component_reindex']+=1
                if canonical!=key:
                    alternate=cache.get(canonical)
                    if alternate is not None:
                        value,masks,_=alternate
                        inverse=[tuple(np.argsort(p)) for p in perms]
                        arrays=translate_responses(masks,key[2],inverse)
                        cache.store(key,value,[(None,None,a) for a in arrays])
                        saved=value,arrays,0
                        oracle.work['component_frame_hits']+=1
        if saved is not None:
            score,responses,_=saved
            total+=score+offset
            for (idx,c,scope,_),alts in zip(steps,responses):
                trace[idx]=(c,tuple(z for z in scope if z!=c),alts)
            oracle.work['component_hits']+=1
            oracle.work['component_factors_reused']+=len(steps)
            oracle.work['avoided']+=sum(product for _,_,_,product in steps)
            continue
        local=pairs+[((c,),unary[c]) for c in sorted(cells)]
        order=[(c,scope,amount) for _,c,scope,amount in steps]
        score,rows=oracle._factor_trace(widths,local,order,cap)
        total+=score
        for (idx,_,_,_),row in zip(steps,rows):trace[idx]=row
        if key is not None:
            cache.store(key,score-offset,rows)
            if vertex_order is not None and cache.defer:
                cache.remember_source(canonical,score-offset,key,vertex_order,perms)
            elif vertex_order is not None:
                tuples=optimal_tuples(cells,widths,vertex_order,perms,rows)
                cache.store(canonical,score-offset,[(None,None,tuples)])
                oracle.work['component_tuples_encoded']+=len(tuples)
            elif canonical is not None and canonical!=key:
                arrays=translate_responses([alts for _,_,alts in rows],key[2],perms)
                cache.store(canonical,score-offset,[(None,None,a) for a in arrays])
    return total,trace
