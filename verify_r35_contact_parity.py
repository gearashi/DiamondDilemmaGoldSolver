"""Rebuild the independently checked 25-loss CNF using only the 40 weighted raw-piece rules."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

from verify_r35_contact_proof import RawPuzzle, audit


def formula(data,codes,certificate,*,inventory_parity=True):
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool

    cap=audit(data,codes,certificate)['minimum_lost']
    if cap!=25:
        raise ValueError('Expected the independently verified original 25-loss weighted certificate.')
    puzzle=RawPuzzle(data)
    reference=puzzle.reference(codes)
    required=set(map(tuple,certificate['required']))
    optional=sorted(reference-required)
    terms=[(set(map(tuple,r['bonds']))-required,r['weight']) for r in certificate['weighted_cores']]
    maximum=certificate['maximum']

    ids=IDPool(start_from=puzzle.count+1)
    loss={e:ids.id() for e in optional}
    clauses=[[-1]]
    for a,b in sorted(reference|required):
        x,y=a//3+1,b//3+1
        sign=puzzle.phase[a]^puzzle.phase[b]^(len(puzzle.slots[a])%2)
        free=[loss[a,b]] if (a,b) in loss else []
        clauses.extend([[x,y if sign else -y]+free,[-x,-y if sign else y]+free])
    clauses.extend([[loss[e] for e in sorted(group)] for group,_ in terms])

    # With k specified lost bonds, at most cap-k other losses can cover the
    # requirements still untouched. Each can contribute at most maximum.
    def over(assumed):
        residual=sum(weight for group,weight in terms if group.isdisjoint(assumed))
        return residual>(cap-len(assumed))*maximum
    keep={e for e in optional if over({e})}
    candidates=sorted(set(optional)-keep)
    clauses.extend([[-loss[e]] for e in sorted(keep)])
    clauses.extend([[-loss[a],-loss[b]] for a,b in combinations(candidates,2) if over({a,b})])
    clauses.extend(CardEnc.atmost([loss[e] for e in candidates],bound=cap,vpool=ids,encoding=EncType.totalizer).clauses)

    if inventory_parity:
        # For each unoriented signed reciprocal class the number of sides on
        # the whole board is even. A tile flip contributes an affine XOR.
        parity={}
        for side,slots in enumerate(puzzle.slots):
            rev=tuple(12-x for x in reversed(slots))
            def label(sign):
                return min((slots,sign),(rev,sign^(len(slots)%2)))
            first,other=label(puzzle.phase[side]),label(puzzle.phase[side]^1)
            parity[first]=parity.get(first,0)^(1<<puzzle.count)
            if first!=other:
                parity[first]^=1<<(side//3)
                parity[other]=parity.get(other,0)^(1<<(side//3))
        for mask in sorted(set(parity.values())):
            variables=[i+1 for i in range(puzzle.count) if mask>>i&1]
            odd=mask>>puzzle.count
            if not variables:
                if odd:clauses.append([])
                continue
            x=variables[0]
            for y in variables[1:]:
                z=ids.id()
                clauses.extend([[x,y,-z],[-x,-y,-z],[x,-y,z],[-x,y,z]])
                x=z
            clauses.append([x if odd else -x])
    return ids.top,clauses


def write_formula(path,data,codes,certificate):
    n,clauses=formula(data,codes,certificate)
    with path.open('w',encoding='ascii',newline='\n') as out:
        out.write(f'p cnf {n} {len(clauses)}\n')
        for row in clauses:
            out.write(' '.join(map(str,row))+' 0\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--reference',type=Path,default=root/'docs/R29_INCUMBENT.json')
    cli.add_argument('--certificate',type=Path,default=root/'docs/R35_CONTACT_LOSS_CERTIFICATE.json')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    read=lambda path:json.loads(path.read_bytes())
    print(write_formula(args.output,read(args.tiles),read(args.reference)['codes'],read(args.certificate)))
