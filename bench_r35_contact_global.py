"""Compare shared-CNF search work and export the actual all-phase induced-contact relaxation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_integer_sat import InventorySAT
from r35_loss_witness import physical_price


def run(tiles,geom,contacts,out,caps,budget,cnf):
    from pysat.solvers import Solver
    p,ref,groups,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],req)
    read=lambda path,key:[frozenset(map(tuple,g))-req for g in json.loads(path.read_bytes())[key]]
    zero=read(geom,'geographic_cores')
    raw=[frozenset(map(tuple,g)) for g in json.loads(contacts.read_bytes())['edge_perfect_cores']]
    layers={k:{g-req for g in raw if len(g)<=k} for k in (0,5,9,13,19)}
    all_contact=sorted(layers[19],key=lambda g:(len(g),sorted(g)))
    result=dict(fingerprint=p.fingerprint,physical_groups=13,geometry=len(zero),comparisons=[])
    for cap in caps:
        model=InventorySAT(p,packed,req,outside,cap,native=None,additional_groups=[g for g in [*groups[:13],*zero,*all_contact] if g not in packed])
        def clause(g):return tuple(model.physical_losses[pair] for pair in sorted(g))
        for length,wanted in layers.items():
            removed=Counter(clause(g) for g in all_contact if g not in wanted)
            clauses=[]
            for c in model.clauses:
                key=tuple(c)
                if removed[key]:removed[key]-=1
                else:clauses.append(c)
            assert not any(removed.values())
            if cnf and cap==caps[0] and length==13:
                with cnf.open('w',encoding='ascii',newline='\n') as f:
                    f.write(f'p cnf {max(abs(v) for c in clauses for v in c)} {len(clauses)}\n')
                    for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
                result['exported_sha256']=hashlib.sha256(cnf.read_bytes()).hexdigest()
            with Solver(name='cadical153',bootstrap_with=clauses) as s:
                s.conf_budget(budget)
                verdict=s.solve_limited();work=s.accum_stats()
                row=dict(cap=cap,limit=length,conflict_budget=budget,verdict=verdict,**work)
                if verdict:
                    phase=sum(1<<(v-1) for v in s.get_model() if 0<v<=p.n)
                    price=physical_price(p.tiles,ref,[*groups[:13],*zero,*wanted],req,phase)
                    if price is None or price[0]>cap:raise ArithmeticError('Invalid global signed witness.')
                    row.update(phase=hex(phase),price=price[0])
                result['comparisons'].append(row)
            out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('tiles','geom','contacts','out'):cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--caps',nargs='+',type=int,default=[25,26,32]);cli.add_argument('--budget',type=int,default=15000)
    cli.add_argument('--cnf',type=Path)
    a=cli.parse_args();run(**vars(a))
