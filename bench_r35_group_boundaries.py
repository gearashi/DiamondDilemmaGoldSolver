"""Bounded min-plus interfaces between disjoint groups of parity equations."""
from math import inf
from time import monotonic


def _extend(basis, rows):
    out=basis.copy()
    for row in rows:
        while row:
            pivot=row.bit_length()-1
            if pivot not in out:
                out[pivot]=row
                break
            row ^= out[pivot]
    return out


def _intersection(left, right):
    # Tag every independent left form with itself. Reducing to zero modulo
    # the right span gives a common linear form rather than a vector label.
    basis={pivot:(row,0) for pivot,row in right.items()}
    common=[]
    for original in left.values():
        row=tag=original
        while row:
            bit=row.bit_length()-1
            if bit not in basis:
                basis[bit]=row,tag
                break
            value,combo=basis[bit]
            row^=value
            tag^=combo
        else:
            common.append(tag)
    return common


def _attainable(forms):
    # Generate all evaluations of the listed linear forms with free variables.
    width=max((v.bit_length() for v in forms),default=0)
    columns=[sum(((v>>i)&1)<<j for j,v in enumerate(forms)) for i in range(width)]
    images=[0]
    for col in _extend({},columns).values():
        images.extend([x^col for x in images])
    return images


class GroupBoundaries:
    @classmethod
    def compile(cls, q, entry_limit=150000):
        if not 2<=len(q.groups)<=12:
            return None
        bonds=[sorted(g) for g in q.groups]
        coeff=[[q._joint_bonds[b][0]>>1 for b in g] for g in bonds]
        active=(1<<len(bonds))-1
        bases=[{}]
        for subset in range(1,active+1):
            bit=subset&-subset
            bases.append(_extend(bases[subset^bit],coeff[bit.bit_length()-1]))
        ranks=[len(b) for b in bases]
        total=ranks[active]
        local_entries=sum(1<<ranks[1<<i] for i in range(len(bonds)))
        if local_entries>entry_limit:
            return None
        # Minimum entry-count order. For processed groups P, next group G
        # and remainder F, the joined boundary spans (P intersection (G+F))
        # plus (G intersection (P+F)).
        costs=[inf]*(active+1)
        costs[0]=0
        predecessor={}
        for done in range(active):
            left=active^done
            while left:
                bit=left&-left
                left^=bit
                nxt=done|bit
                width=ranks[active^done]+ranks[active^bit]+ranks[nxt]-2*total
                cost=costs[done]+(1<<width)
                if cost<costs[nxt]:
                    costs[nxt]=cost
                    predecessor[nxt]=bit
        if local_entries+costs[active]>entry_limit:
            return None
        order=[]
        node=active
        while node:
            bit=predecessor[node]
            order.append(bit)
            node^=bit
        steps=[]
        done=0
        previous=[]
        for bit in reversed(order):
            index=bit.bit_length()-1
            boundary=_intersection(bases[bit],bases[active^bit])
            nxt=done|bit
            following=_intersection(bases[nxt],bases[active^nxt])
            rows=coeff[index]
            mask=(1<<len(rows))-1
            required=sum(1<<j for j,b in enumerate(bonds[index]) if b in q.required)
            symbols=[q._joint_bonds[b][1] for b in bonds[index]]
            leaf=[(v&mask,v>>len(rows)) for v in _attainable(rows+boundary)]
            left_size=len(previous)
            right_size=len(boundary)
            joins=[(v&((1<<left_size)-1),(v>>left_size)&((1<<right_size)-1),v>>(left_size+right_size))
                   for v in _attainable(previous+boundary+following)]
            steps.append((mask,required,symbols,leaf,joins))
            done=nxt
            previous=following
        return cls(steps,local_entries,costs[active])

    def __init__(self,steps,leaves,joins):
        self.steps=steps
        self.leaf_entries=leaves
        self.join_entries=joins

    def minimum(self, flipped, deadline=inf):
        value=1|sum(2<<i for i in flipped)
        pending={0:0}
        for mask,required,symbols,leaf,joins in self.steps:
            if monotonic()>=deadline:
                raise TimeoutError
            if required==mask:
                return inf
            rhs=sum(((value&s).bit_count()&1)<<i for i,s in enumerate(symbols))
            prices={}
            for j,(v,boundary) in enumerate(leaf):
                if j%256==0 and monotonic()>=deadline:
                    raise TimeoutError
                bad=rhs^v
                if not bad&required:
                    prices[boundary]=min(prices.get(boundary,inf),max(1,bad.bit_count()))
            result={}
            for j,(a,b,c) in enumerate(joins):
                if j%256==0 and monotonic()>=deadline:
                    raise TimeoutError
                if a in pending and b in prices:
                    result[c]=min(result.get(c,inf),pending[a]+prices[b])
            pending=result
        return pending.get(0,inf)


def benchmark_boundaries(tiles, output):
    from pathlib import Path
    from random import Random
    from r32_gold_bonds import inventory
    from r32_gold_loss import read_json,reference_groups
    from r32_gold_syndrome import OutsideSyndromes
    from r35_group_images import GroupImages
    from search_problem import SearchProblem
    import json
    problem=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,groups=reference_groups(problem,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(problem).required
    full=OutsideSyndromes(problem,ref,groups,req)
    order=list(full.outside)
    Random(35).shuffle(order)
    patterns=[[],*read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']]
    results=[]
    for omitted in (0,8,16,32,59,65,80,120):
        q=OutsideSyndromes(problem,ref-set(order[:omitted]),full.groups,req)
        model=GroupBoundaries.compile(q)
        row=dict(omitted=omitted,estimated_entry_cap=150000,compiled=model is not None)
        if model is not None:
            row.update(leaf=model.leaf_entries,joins=model.join_entries,verdicts=[])
            for pattern in patterns:
                local=[q.outside.index(full.outside[i]) for i in pattern if full.outside[i] in q.outside]
                rhs=1|sum(2<<i for i in local)
                if any((row&rhs).bit_count()&1 for row in q._joint_dependencies):
                    continue
                cost=model.minimum(local)+len(local)
                for cap in (cost-1,cost):
                    started=monotonic()
                    try:
                        expected=q.joint(local,cap,started+.25)
                    except TimeoutError:
                        expected=None
                    if expected is not None and expected!=(cost<=cap):
                        raise AssertionError((omitted,local,cost,cap,expected))
                    row['verdicts'].append(dict(local=local,cap=cap,cost=cost,baseline=expected))
        print(omitted,row['compiled'],row.get('joins'),sum(x['baseline'] is None for x in row.get('verdicts',[])), flush=True)
        results.append(row)
    Path(output).write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    benchmark_boundaries(args.tiles,args.output)
