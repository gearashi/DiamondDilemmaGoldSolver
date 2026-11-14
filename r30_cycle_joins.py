"""Unique cycle-type joins for larger supports with a rigorous partial-score envelope."""
import argparse
import itertools
import json
import math
from collections import Counter,defaultdict
from bisect import bisect_right
from r28_exchange import read_json

from r29_exchange_clusters import StarClusters


def cycle_rows(pool,size):
    for group in itertools.combinations(sorted(pool),size):
        for tail in itertools.permutations(group[1:]):
            yield (group[0],*tail)


class TypedJoin:
    """Rank and combine cycles from an explicit support or physical region.

    A pair of disjoint cycles has an exactly quadratic physical-seam score.
    For a range of partners, sum pair upper bounds separately. Negative
    interactions are retained on a completed join.
    """
    def __init__(self,bound,pool,family,target,*,contact=False):
        self.bound=bound
        self.family=tuple(sorted(family,reverse=True))
        allowed=set(map(int,bound.model.free))
        self.pool=tuple(sorted(map(int,pool)))
        if len(set(self.pool))!=len(self.pool) or not set(self.pool)<=allowed:
            raise ValueError('Expected distinct movable cells.')
        if sum(self.family)>len(self.pool) or min(self.family)<2:
            raise ValueError('Not enough rows for this cycle family.')
        self.target=target
        self.contact=contact
        # This explicit catalogue is for a small donor pool. The full-board
        # multi-center runner streams supports without allocating global cycle lists.
        if sum(math.comb(len(self.pool),k)*math.factorial(k-1) for k in set(self.family))>5000:
            raise ValueError("Use a smaller explicit pool, or the ranked full-board runner.")
        self.cost=Counter()
        n=bound.n
        sides=bound.masks[bound.tiles,0,:]
        self.fit=(sides[:,None,:,None]==bound.model.reverse[sides[None,:,None,:]]).any(axis=(2,3))
        self.edges=[(int(a),int(b)) for a,_,b,_ in bound.model.problem.edges]
        self.base=sum(int(self.fit[a,b]) for a,b in self.edges)
        self.near={c:set(map(int,bound.neighbors[c]))-{n} for c in self.pool}
        self.close={c:{z for d in ({c}|nbrs) for z in ({d}|set(map(int,bound.neighbors[d])))}-{n}
                    for c,nbrs in self.near.items()}
        self.cycles={}
        self.price={}
        self.groups={}
        self.bits={}
        self.cross={}
        self.stars=StarClusters(bound)
        for length in set(self.family):
            group=[]
            for row in cycle_rows(self.pool,length):
                moves=dict(zip(row,row[1:]+row[:1]))
                self.cycles[row]=moves
                self.bits[row]=sum(1<<c for c in row)
                self.price[row]=sum(int(self.fit[moves.get(a,a),moves.get(b,b)])-int(self.fit[a,b])
                                      for a,b in self.edges if a in moves or b in moves)
                group.append(row)
            self.groups[length]=sorted(group,key=lambda c:(-self.price[c],c))
        self.cost['catalogued']=len(self.price)

    def correction(self,a,b):
        k=tuple(sorted((a,b)))
        if k not in self.cross:
            left,right=self.cycles[a],self.cycles[b]
            self.cross[k]=sum(int(self.fit[ap,right[y]])-int(self.fit[ap,y])
                              -int(self.fit[x,right[y]])+int(self.fit[x,y])
                              for x,ap in left.items() for y in self.near[x] if y in right)
            self.cost['pair_prices']+=1
        else:
            self.cost['pair_reuse']+=1
        return self.cross[k]

    def survivors(self, *, order_by_gain=True):
        """Yield each surviving endpoint once, in best-bound-first sibling order."""
        def visit(chosen,occupied,total):
            if len(chosen)==len(self.family):
                if self.contact and not any(self.close[a].intersection(B) for i,A in enumerate(chosen)
                                          for B in chosen[:i] for a in A):
                    self.cost['contact']+=1
                    return
                sources={c:d for cycle in chosen for c,d in self.cycles[cycle].items()}
                cells=tuple(sorted(sources))
                local={c:i for i,c in enumerate(cells)}
                order=tuple(local[sources[c]] for c in cells)
                self.cost['star_queries']+=1
                if self.stars.price(cells,order)>=2*self.target:
                    self.cost['survivors']+=1
                    yield tuple(chosen),cells,order
                return
            remaining=self.family[len(chosen):]
            layers=[]
            for length in remaining:
                after=next((c for c in reversed(chosen) if len(c)==length),None)
                opts=[(self.price[c]+sum(self.correction(c,d) for d in chosen),c)
                      for c in self.groups[length] if not (self.bits[c]&occupied)
                      and (after is None or after<c)]
                if not opts:return
                layers.append(sorted(opts,key=lambda r:(-r[0],r[1])))
            # Cover inter-layer coupling. This dense relaxation belongs to
            # restricted pools. Each positive correction is at most 2 per
            # edge crossing the two cycles, and the graph is subcubic.
            extra=0
            for i in range(len(remaining)):
                for j in range(i):
                    if len(layers[i])*len(layers[j])>3000:
                        extra+=6*min(remaining[i],remaining[j])
                        continue
                    max_pair=max((self.correction(a,b) for _,a in layers[i] for _,b in layers[j]
                                  if self.bits[a]&self.bits[b]==0),default=0)
                    extra+=max_pair
            suffix=sum(row[0][0] for row in layers[1:])+extra
            options=layers[0] if order_by_gain else sorted(layers[0],key=lambda entry:entry[1])
            for ix,(g,c) in enumerate(options):
                if total+g+suffix<self.target:
                    if order_by_gain:
                        self.cost['priced_siblings_skipped']+=len(options)-ix
                        break
                    self.cost['priced_siblings_skipped']+=1
                    continue
                self.cost['prefixes']+=1
                yield from visit((*chosen,c),occupied|self.bits[c],total+g)
        yield from visit((),0,self.base)


    def half_survivors(self,representatives=False):
        """Group a 3+2+2 or 2+2+2+2 join into two bounded partner lists."""
        if self.family not in ((3,2,2),(2,2,2,2)):
            yield from self.survivors()
            return
        def halves(lengths):
            rows=[]
            for first in self.groups[lengths[0]]:
                if len(lengths)==1:
                    rows.append((self.price[first],self.bits[first],(first,)))
                    continue
                for second in self.groups[lengths[1]]:
                    if first>=second or self.bits[first]&self.bits[second]:continue
                    gain=self.price[first]+self.price[second]+self.correction(first,second)
                    rows.append((gain,self.bits[first]|self.bits[second],(first,second)))
            rows.sort(key=lambda row:(-row[0],row[2]))
            return rows
        cut=1 if self.family[0]==3 else 2
        left=halves(self.family[:cut])
        right=(left if cut==2 else halves(self.family[cut:]))
        self.cost['half_entries']=len(left)+(0 if left is right else len(right))
        full=(1<<len(right))-1
        occupied_index=defaultdict(int)
        for j,(_,_,cycles) in enumerate(right):
            for cycle in cycles:
                for c in cycle:occupied_index[c]|=1<<j
        negative_gains=[-row[0] for row in right]
        for gain,left_bits,lcycles in left:
            outside=0
            boundary={x for cycle in lcycles for c in cycle for x in self.close[c]}
            for cell in boundary:outside|=occupied_index[cell]
            eligible=full
            for cycle in lcycles:
                for c in cycle:eligible &= ~occupied_index[c]
            # A half with no cell on the affected boundary has no coupling.
            # All other halves share a certified bound on positive coupling.
            max_cross=sum(max((self.correction(a,b) for b in self.groups[length]
                               if not (self.bits[b]&left_bits)),default=0)
                          for a in lcycles for length in self.family[cut:])
            ranges=((eligible&outside,max_cross),(eligible&~outside,0))
            lanes=0
            for members,bonus in ranges:
                end=bisect_right(negative_gains,self.base+gain+bonus-self.target)
                selected=members&((1<<end)-1)
                self.cost['half_suffix_skips']+=members.bit_count()-selected.bit_count()
                lanes|=selected
            def eligible_rows(mask):
                while mask:
                    bit=mask&-mask
                    mask^=bit
                    row=right[bit.bit_length()-1]
                    if cut!=2 or lcycles[-1]<row[2][0]:
                        yield row
            rows=eligible_rows(lanes)
            if representatives:
                from r32_half_response import group_rows
                groups=group_rows(self,lcycles,rows)
            else:
                groups=((row,) for row in rows)
            for items in groups:
                other,_,representative=items[0]
                self.cost['half_queries']+=1
                if representatives:
                    self.cost['response_rows_shared']+=len(items)-1
                interaction=sum(self.correction(a,b) for a in lcycles for b in representative)
                if self.base+gain+other+interaction<self.target:
                    continue
                for _,_,rcycles in items:
                    cs=lcycles+rcycles
                    if self.contact and not any(self.close[a].intersection(B) for i,A in enumerate(cs)
                                               for B in cs[:i] for a in A):continue
                    sources={a:b for cycle in cs for a,b in self.cycles[cycle].items()}
                    cells=tuple(sorted(sources));pos={c:i for i,c in enumerate(cells)}
                    order=tuple(pos[sources[c]] for c in cells)
                    self.cost['star_queries']+=1
                    if self.stars.price(cells,order)>=2*self.target:
                        self.cost['survivors']+=1
                        yield cs,cells,order



def matching_bound(weights, family):
    """Maximum sum of directed donor prices under an exact partition of â‰¤8 rows.

    Enumerate directed cycles once, then solve a disjoint set partition by DP.
    In the all-swap case each edge has price w[i,j]+w[j,i], a perfect matching.
    """
    from functools import lru_cache
    k=len(weights)
    if sum(family)!=k or any(len(row)!=k for row in weights):
        raise ValueError('Unexpected dimension.')
    @lru_cache(None)
    def visit(remaining,lengths):
        if not lengths:return 0 if remaining==0 else None
        anchor=(remaining&-remaining).bit_length()-1
        cells=[i for i in range(anchor+1,k) if remaining>>i&1]
        best=None
        for length in set(lengths):
            rest=list(lengths);rest.remove(length)
            for tail in itertools.permutations(cells,length-1):
                cycle=(anchor,*tail);used=sum(1<<c for c in cycle)
                suffix=visit(remaining^used,tuple(rest))
                if suffix is None:continue
                arcs=[weights[a][b] for a,b in zip(cycle,cycle[1:]+cycle[:1])]
                if any(x is None for x in arcs):continue
                value=suffix+sum(arcs)
                if best is None or value>best:best=value
        return best
    return visit((1<<k)-1,tuple(sorted(family)))


def solve_pool(problem,board,pool,family,target,*,stop=False,order_by_gain=True,halves=False,representatives=False):
    from r27_incumbent import BoardModel
    from r28_exchange import ExchangeBound
    from r28_exchange_rotations import RotationOracle
    from r29_exchange_route import rotation_optimum
    from r30_cycles import Predecessors
    model=BoardModel(problem)
    bound=ExchangeBound(model,board)
    pred=Predecessors(bound,target)
    join=TypedJoin(bound,pool,family,target,contact=pred.certified)
    best=None
    oracle=RotationOracle(model)
    rows=join.half_survivors(representatives) if halves or representatives else join.survivors(order_by_gain=order_by_gain)
    for cycles,cells,order in rows:
        if pred.reject(cycles):
            join.cost['predecessor']+=1
            continue
        seed=bound.witness_seed(cells,order)
        if seed is None:continue
        row=rotation_optimum(bound,oracle,seed)
        join.cost['exact']+=1
        if row['score']>=target:
            join.cost['hits']+=1
            if best is None or row['score']>best['score']:best=row
            if stop:break
    return dict(work=dict(join.cost),best=best,complete=not(stop and best is not None),
                pool=sorted(pool),family=family,target=target)


def main():
    from search_problem import SearchProblem
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--incumbent',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--cells',nargs='+',type=int,required=True)
    parser.add_argument('--family',choices=('42','33','322','2222'),required=True)
    parser.add_argument('--target',type=int,default=208)
    parser.add_argument('--stop-on-hit',action='store_true')
    parser.add_argument('--halves',action='store_true',help='Group large cycle families into two sorted joins.')
    parser.add_argument('--representatives',action='store_true',help='Share the seam price across compatible half-join responses.')
    args=parser.parse_args()
    result=solve_pool(SearchProblem(read_json(args.tiles)),read_json(args.incumbent)['codes'],
                      args.cells,tuple(map(int,args.family)),args.target,stop=args.stop_on_hit,halves=args.halves,representatives=args.representatives)
    print(json.dumps(result))


if __name__=='__main__':main()
