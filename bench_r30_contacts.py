"""Verify four-swap cross-cycle contact and count certified predecessor exclusions.

CUDA exhausts the directed four-cycle baseline. A paired CPU calculation checks every
survivor. The other constituent cycles and the predecessor counts are computed on CPU.
"""
import argparse
from collections import Counter
import itertools
import math
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, combination_batch, read_json, domain_binding
from r28_exchange_cuda import CudaShell
from r29_exchange_clusters import StarClusters
from r29_exchange_route import physical_hash
from r30_cycles import Predecessors
from search_problem import SearchProblem


def physical_gains(bound,supports,order):
    incoming = supports[:,order]
    after = bound.tiles[incoming]
    touched = np.sort(np.concatenate((supports,bound.neighbors[supports].reshape(len(supports),-1)),axis=1),axis=1)
    unique = np.ones_like(touched,dtype=bool)
    unique[:,1:] = touched[:,1:]!=touched[:,:-1]
    unique &= touched!=bound.n
    neighbors = bound.neighbors[touched]
    ct = bound.pieces[touched].copy()
    nt = bound.pieces[neighbors].copy()
    for pos,cells in enumerate(supports.T):
        ct = np.where(touched==cells[:,None],after[:,pos,None],ct)
        nt = np.where(neighbors==cells[:,None,None],after[:,pos,None,None],nt)
    return ((bound.stars(ct,nt)-bound.base[touched])*unique).sum(axis=1)


def local_certificate(bound,batch=524288):
    n = bound.n
    clusters = StarClusters(bound)
    best = {}
    records = {}
    for size,orders in ((2,[(1,0)]),(3,[(1,2,0),(2,0,1)])):
        hist = Counter()
        examples = []
        for pos in range(0,math.comb(n,size),4096):
            rows = combination_batch(range(n),size,pos,min(4096,math.comb(n,size)-pos))
            # Disjoint local stars can be read as additive substitution deltas.
            mixed = np.any([clusters.close[rows[:,i],rows[:,j]]
                            for i,j in itertools.combinations(range(size),2)],axis=0)
            for order in orders:
                values = clusters.delta[rows,rows[:,order]].sum(axis=1)
                if np.any(mixed):
                    values[mixed] = physical_gains(bound,rows[mixed],order)
                for value,num in zip(*np.unique(values,return_counts=True)):
                    hist[int(value)] += int(num)
                for i in np.flatnonzero(values>=0):
                    examples.append(dict(cells=rows[i].tolist(),order=order,gain=int(values[i])))
        top = sorted(hist.elements(),reverse=True)[:4]
        best[size] = top
        records[str(size)] = dict(top=top,nonnegative=examples,count=sum(hist.values()))
    # For a noncontact 4+2 with 208 potential, the four-cycle cannot fall
    # below target minus the best available standalone swap gain.
    limit = 416-best[2][0]
    cuda = CudaShell(bound,4)
    ncomb = math.comb(n,4)
    quads = []
    quad_orders = [p for p in itertools.permutations(range(4)) if p[0]!=0 and
                   len({0,p[0],p[p[0]],p[p[p[0]]]})==4]
    assert len(quad_orders)==6
    for order in quad_orders:
        for pos in range(0,ncomb,batch):
            # Passing a half-integer requests the exact doubled integer threshold.
            survivors = cuda.survivors(pos,min(batch,ncomb-pos),order,limit/2)
            if survivors:
                scores = physical_gains(bound,np.asarray(survivors,dtype=int),order)
                for cells,gain in zip(survivors,scores):
                    if bound.total+int(gain)<limit:
                        raise RuntimeError("CUDA survivor failed the independent check.")
                    quads.append(dict(cells=cells,order=order,gain=int(gain)))
    records['4'] = dict(tested=ncomb*len(quad_orders),doubled_threshold=limit,survivors=quads)
    return best,records,clusters


def certify(problem):
    root = Path(__file__).parent
    archive = read_json(root/'docs/R29_EXCHANGE_CENTERS.json')
    model = BoardModel(problem)
    if archive['domain']!=domain_binding(model):
        raise ValueError("Needs the published three-center puzzle.")
    centers = [r['pieces'] for r in archive['centers']]
    out = []
    for pieces in centers:
        b = ExchangeBound(model,[3*t for t in pieces])
        top,raw,clusters = local_certificate(b)
        # At outer centers any positive swap relocates to the middle center. Undo
        # this factor, and the endpoint is within its published radius-five ball.
        positive = [row for row in raw['2']['nonnegative'] if row['gain']>0]
        for row in positive:
            a,c = row['cells']
            other = pieces.copy()
            other[a],other[c] = other[c],other[a]
            if other not in centers:
                raise RuntimeError("Positive swap is not a verified center change.")
        for family in ((3,3),(2,2,3),(2,2,2,2)):
            count = Counter(family)
            if b.total+sum(sum(top[k][:n]) for k,n in count.items())>=416:
                # Dropping a positive swap is safe only when the remaining cycles
                # move at most five physical pieces relative to its proven center.
                if sum(family)-2>5 or not positive:
                    raise RuntimeError("Cannot certify this cycle partition.")
                nonpos = sorted([r['gain'] for r in raw['2']['nonnegative'] if r['gain']<=0]+[-1]*4,reverse=True)
                if b.total+sum(nonpos[:count[2]])+sum(top[3][:count[3]])>=416:
                    raise RuntimeError("Nonpositive swaps still allow the threshold.")
        for quad in raw['4']['survivors']:
            for swap in raw['2']['nonnegative']:
                if swap['gain']>0:
                    continue
                if b.total+quad['gain']+swap['gain']>=416 and all(not clusters.close[i,j] for i in quad['cells'] for j in swap['cells']):
                    raise RuntimeError("Separated cycles remain unresolved.")
            # Negative swaps can gain at most -1.
            if b.total+quad['gain']-1>=416:
                raise RuntimeError("Need to enumerate negative-swap ties as well.")
        flags,counts = predecessor_counts(b)
        out.append(dict(center=physical_hash(pieces),star=b.total,constituents=raw,
                        isolated_swap_count=flags,predecessor_coverage=counts,cross_cycle_required=True))
    return out


def predecessor_counts(b):
    pred = Predecessors(b,208)
    g = pred.gain if pred.gain is not None else pred.local_gains()
    n = b.n
    nb = b.neighbors
    around = [sum(1<<x for x in {i,*map(int,nb[i])} if x<n) for i in range(n)]
    ev = [((1<<i)|(1<<j),around[i]|around[j]) for i in range(n) for j in range(i)
          if not (around[i]>>j)&1 and g[i,j]+g[j,i]<=0]
    def words(field):
        return np.asarray([[(e[field]>>(64*k)) & (2**64-1) for k in range((n+63)//64)]
                           for e in ev],dtype=np.uint64)
    supports,footprints = words(0),words(1)
    s1 = s2 = 0
    ways = np.array([3*math.comb(n-k,4) for k in range(17)],dtype=np.int64)
    for i,(_,foot) in enumerate(ev):
        s1 += 15*math.comb(n-foot.bit_count(),6)
        eligible = ((supports[i+1:]&footprints[i])==0).all(1)
        sizes = np.bitwise_count(footprints[i+1:][eligible]|footprints[i]).sum(1)
        s2 += int(ways[sizes].sum())
    eight = (2*s1-s2+2)//3
    negative_three = 0
    for i in range(n-2):
        j,k = np.triu_indices(n-i-1,1)
        j+=i+1
        k+=i+1
        ok = (~np.isin(j,nb[i])) & (~np.isin(k,nb[i])) & ~(nb[j]==k[:,None]).any(1)
        a,d = j[ok],k[ok]
        negative_three += int((g[i,a]+g[a,d]+g[d,i]<=0).sum()+(g[i,d]+g[d,a]+g[a,i]<=0).sum())
    counts = {'4+2':6*len(ev)*math.comb(n-8,4),'3+3':negative_three*math.comb(n-12,3),
              '3+2+2':10*len(ev)*math.comb(n-8,5),'2+2+2+2':eight}
    return len(ev),dict(counts=counts,total=sum(counts.values()),singles=s1,pairs=s2,
                        flagged_directed_triples=negative_three)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    atomic_json(args.output,certify(SearchProblem(read_json(args.tiles))))
