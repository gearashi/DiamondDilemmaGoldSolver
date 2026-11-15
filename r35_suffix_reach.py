"""Exact fixed-cardinality suffix reachability for sparse reference-loss phase errors."""
from math import comb
from time import monotonic


class SuffixReachability:
    def __init__(self, rhs, columns, projected_bits=10, targets=(0,)):
        width=max([rhs,*columns],default=0).bit_length()
        # Locally rare independent checks leave the least freedom at a fixed
        # cardinality. Original check order resolves equal support counts.
        rows=sorted(((sum(((v>>bit)&1)<<i for i,v in enumerate(columns)),bit)
                    for bit in range(width)), key=lambda kv:(kv[0].bit_count(),kv[1]))
        chosen=[]
        basis={}
        for row,bit in rows:
            while row:
                pivot=row.bit_length()-1
                if pivot not in basis:
                    basis[pivot]=row
                    chosen.append(bit)
                    break
                row^=basis[pivot]
            if len(chosen)>=projected_bits:
                break
        def project(v):
            return sum(((v>>b)&1)<<i for i,b in enumerate(chosen))
        self.rhs=project(rhs)
        self.cols=tuple(map(project,columns))
        self.width=len(chosen)
        size=1<<self.width
        self.low_masks=[((1<<size)-1)//((1<<(1<<bit))+1) for bit in range(self.width)]
        acceptable = sum(1<<v for v in set(map(project,targets)))
        self.reach=[[acceptable]*(len(columns)+1)]
        self.states_built=0

    def prepare(self, k):
        n=len(self.cols)
        while len(self.reach)<=k:
            prev=self.reach[-1]
            row=[0]*(n+1)
            for i in range(n-len(self.reach),-1,-1):
                value=prev[i+1]
                col=self.cols[i]
                while col:
                    unit=col&-col
                    lower=self.low_masks[unit.bit_length()-1]
                    value=((value&lower)<<unit)|((value>>unit)&lower)
                    col^=unit
                row[i]=row[i+1]|value
                self.states_built+=1
            self.reach.append(row)

    def ranks(self, errors, start, stop, deadline, stats, companion=None):
        tail=min(2,errors)
        k=errors-tail
        n=len(self.cols)-tail
        self.prepare(errors)
        if companion is not None:
            companion.prepare(errors)
        prefix=[]
        def visit(first, left, rank, residual, other):
            if monotonic()>=deadline:
                yield max(start,rank), None
                return
            if (not ((self.reach[left+tail][first] >> residual) & 1) or
                    (companion is not None and not ((companion.reach[left+tail][first] >> other) & 1))):
                end=rank+comb(n-first,left)
                stats['unreachable_prefixes']+=min(end,stop)-max(start,rank)
                return
            if left==0:
                yield rank,tuple(prefix)
                return
            for i in range(first,n-left+1):
                size=comb(n-i-1,left-1)
                if rank>=stop:
                    break
                if rank+size>start:
                    prefix.append(i)
                    yield from visit(i+1,left-1,rank,residual^self.cols[i],
                                    other^companion.cols[i] if companion is not None else 0)
                    prefix.pop()
                rank+=size
        yield from visit(0,k,0,self.rhs,companion.rhs if companion is not None else 0)
