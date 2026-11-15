"""Check every added cap-20 clause against physical colors and independent parity elimination."""
import argparse
from collections import Counter
from itertools import combinations,product
import json
from pathlib import Path
import numpy as np
from bench_r35_loss_proof import original
from bench_r35_loss_tight import tight_clauses


def audit(tiles):
    p,ref,groups,required,outside=original(tiles)
    required=set(required)
    labels=[]
    for tile in p.tiles:
        sides=[[] for _ in range(3)]
        for segment in tile['segments']:
            for side,port in segment:
                sides[side].append(port)
        bit=0
        for side in sides:
            assert len(set(side))==len(side)
            labels.append((tuple(sorted(side)),bit))
            bit^=len(side)%2
        assert bit==0

    def reciprocal(label):
        slots,bit=label
        return tuple(sorted(12-s for s in slots)),bit^(len(slots)%2)
    def label(side,flip):
        slots,base=labels[side]
        return slots,base^flip
    def compatible(pair,x,y):
        a,b=pair
        return label(a,x)==reciprocal(label(b,y))

    # Form the complete parity-inventory matrix by directly counting class
    # representatives at zero and at each unit vector. Flipping one tile
    # affects only that tile, so finite differences give exact coefficients.
    def odd_counts(flipped):
        keys=Counter()
        for d in range(len(labels)):
            v=label(d,int(d//3==flipped))
            keys[min(v,reciprocal(v))]+=1
        return keys
    base=odd_counts(-1)
    samples=[odd_counts(i) for i in range(p.n)]
    keys=sorted(set(base).union(*samples))
    matrix=[[((q[key]-base[key])%2) for q in samples]+[base[key]%2] for key in keys]
    def consistent(pairs):
        rows=list(matrix)
        for a,b in pairs:
            valid=[int(compatible((a,b),0,y)) for y in (0,1)]
            assert sum(valid)==1
            row=[0]*(p.n+1)
            row[a//3]^=1
            row[b//3]^=1
            row[-1]=valid[1]
            rows.append(row)
        work=np.array(rows,dtype=np.uint8)
        pivot=0
        for i in range(p.n):
            matches=np.flatnonzero(work[pivot:,i])
            if not len(matches):continue
            at=pivot+matches[0]
            work[[pivot,at]]=work[[at,pivot]]
            targets=np.flatnonzero(work[pivot+1:,i])+pivot+1
            work[targets]^=work[pivot]
            pivot+=1
            if pivot==len(work):break
        return not np.any(work[pivot:,-1])

    links={e:1000+i for i,e in enumerate(sorted(ref-required))}
    high=max(links.values())
    clauses,meta=tight_clauses(p,groups,required,outside,links,high)
    # Each conjunction is derived from independently evaluated truth assignments
    # that must be forbidden; generating clauses this way also catches sign errors.
    expected=[]
    def disallow(values):
        expected.append(tuple(sorted(-v if bit else v for v,bit in values.items())))
    rigid=set(outside)
    flex=[]
    for g in groups:
        rest=set(g)-required
        if consistent(set(g)|required):flex.append(sorted(rest))
        else:rigid.update(rest)
    clean_vars=set()
    for clause in clauses:
        clean_vars.update(abs(v) for v in clause if abs(v)>high)
    owner={}
    for clean in clean_vars:
        ids={-lit for c in clauses if clean in c for lit in c if -lit in links.values()}
        assert ids
        matches=[i for i,g in enumerate(flex) if ids=={links[e] for e in g}]
        assert len(matches)==1 and matches[0] not in owner
        owner[matches[0]]=clean
    assert len(owner)==len(flex)
    def bond_rules(pair,clean=None):
        a,b=pair;loss=links[pair]
        for x,y in product((0,1),repeat=2):
            assignment={a//3+1:x,b//3+1:y}
            if compatible(pair,x,y):
                values={**assignment,loss:1}
                if clean is not None:values[clean]=0
                disallow(values)
            elif clean is not None:
                disallow({**assignment,clean:1})
    for pair in rigid:bond_rules(pair)
    for i,group in enumerate(flex):
        c=owner[i]
        for pair in group:bond_rules(pair,c)
        for a,b in combinations(group,2):disallow({c:1,links[a]:1,links[b]:1})
    got=Counter(tuple(sorted(c)) for c in clauses)
    assert got==Counter(expected)
    assert len(ref)==207 and len(required)==8 and len(groups)==10
    assert len(rigid)==meta['rigid']
    return dict(reference_bonds=len(ref),required=len(required),inventory_parity_classes=len(keys),
                disjoint_groups=len(groups),rigid_bonds=len(rigid),flexible_groups=[len(g) for g in flex],
                compared_clauses=len(clauses),unique_clauses=len(got),puzzle_fingerprint=p.fingerprint,
                assertion='Every added clause equals an independently derived local equality-at-optimum obligation.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    result=audit(args.tiles)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result)
