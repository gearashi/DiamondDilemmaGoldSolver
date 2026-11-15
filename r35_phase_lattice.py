"""Sparse integer column checks for phase components tied by signed physical bonds."""
from functools import reduce
from math import gcd

from constraint_lattice import IntegerLattice,LatticeLimit
from constraint_lattice_round26 import separating_functional


class PhaseColumns:
    def __init__(self,interface,relations=(),pins=None,euler=True):
        n=len(interface.rows)
        adj=[[] for _ in range(n)]
        for u,v,x in relations:
            if not 0<=u<n or not 0<=v<n or x not in (0,1):raise ValueError('Invalid phase relation.')
            adj[u].append((v,x));adj[v].append((u,x))
        pins=dict(pins or {})
        if any(t not in range(n) or v not in (0,1) for t,v in pins.items()):raise ValueError('Invalid phase pin.')
        dim=len(interface.mod) if euler else len(interface.mod)-1
        self.mod=interface.mod[:dim]
        self.base=[0]*dim
        self.columns=[]
        self.owner={}
        self.euler=sorted(interface.euler) if euler else [None]
        self.consistent=True
        unseen=set(range(n))
        while unseen:
            root=min(unseen);orient={root:0};stack=[root]
            while stack:
                u=stack.pop()
                for v,x in adj[u]:
                    val=orient[u]^x
                    if v in orient:
                        if orient[v]!=val:self.consistent=False
                    else:orient[v]=val;stack.append(v)
            unseen-=orient.keys()
            specified={pins[t]^z for t,z in orient.items() if t in pins}
            if len(specified)>1:self.consistent=False
            anchor=min(specified,default=0)
            vals=[0]*dim;delta=[0]*dim
            for t,z in orient.items():
                a=interface.rows[t][z^anchor];b=interface.rows[t][1^z^anchor]
                self.owner[t]=(None if specified else len(self.columns),z^anchor)
                for j in range(dim):
                    vals[j]+=a[j];delta[j]+=b[j]-a[j]
            for j in range(dim):self.base[j]+=vals[j]
            if not specified:self.columns.append(tuple(delta))

    def targets(self):
        for final in self.euler:
            target=[-v for v in self.base]
            if final is not None:target[-1]+=final
            yield tuple(target)

    def vectors(self):
        cols=[{j:v for j,v in enumerate(col) if v} for col in self.columns]
        return cols+[{j:2} for j,mod in enumerate(self.mod) if mod]


def verify(vectors,target,certificate):
    weights=dict(certificate['weights']);modulus=certificate['modulus']
    if not weights or modulus<0 or any(i not in range(len(target)) for i in weights):return False
    def zero(value):return value%modulus==0 if modulus else value==0
    return all(zero(sum(value*weights.get(j,0) for j,value in vector.items())) for vector in vectors) and not zero(
        sum(target[j]*w for j,w in weights.items()))


def price(columns,max_operations=250000,max_bits=256):
    if not columns.consistent:return dict(status='XOR_CONFLICT')
    vec=columns.vectors();lattice=None
    divisors=[reduce(gcd,(col.get(j,0) for col in vec),0) for j in range(len(columns.base))]
    proofs=[]
    for target in columns.targets():
        cert=None
        for j,(divisor,value) in enumerate(zip(divisors,target)):
            if (value%divisor if divisor else value):
                cert=dict(weights=[[j,1]],modulus=abs(divisor))
                break
        if cert is None:
            try:
                if lattice is None:lattice=IntegerLattice(vec,max_operations=max_operations,max_bits=max_bits)
                cert=separating_functional(lattice,{j:v for j,v in enumerate(target) if v})
            except LatticeLimit:
                return dict(status='UNKNOWN',proofs=proofs)
        if cert is None:return dict(status='PASSES',proofs=proofs)
        if not verify(vec,target,cert):raise ArithmeticError('Phase lattice separator failed independent column replay.')
        proofs.append(dict(target=target,separator=cert))
    return dict(status='INFEASIBLE',proofs=proofs)
