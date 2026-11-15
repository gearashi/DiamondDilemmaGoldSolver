"""Export and independently validate an exact weighted-slack signed reference-loss formula."""
import argparse
import hashlib
import json
from pathlib import Path

from bench_r35_contact_arity import extra_cnf
from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_contact_certificate import CERTIFICATE,verify
from r35_integer_sat import InventorySAT


def export(tiles,geo,path,cap,dest,parity=False):
    p,ref,groups,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],req)
    read=lambda fp,k:[frozenset(map(tuple,g)) for g in json.loads(fp.read_bytes())[k]]
    condition=[g-req for g in [*read(geo,'geographic_cores'),*[x for x in read(path,'edge_perfect_cores') if len(x)<=13]]]
    terms=json.loads(CERTIFICATE.read_bytes())
    if verify(p,ref,req,groups[:13],terms,charged=[g-req for g in [*groups[:13],*condition]])!=25:
        raise ValueError('The weighted certificate is not valid under the requested groups.')
    model=InventorySAT(p,packed,req,outside,cap,native=None,parity_only=parity,
                       additional_groups=[g for g in [*groups[:13],*condition] if g not in packed])
    maximum=max(abs(v) for c in model.clauses for v in c)
    extra=extra_cnf(terms,req,ref-req,model.physical_losses,cap,'exact',maximum)
    clauses=model.clauses+extra
    with dest.open('w',encoding='ascii',newline='\n') as f:
        f.write(f'p cnf {max(abs(x) for c in clauses for x in c)} {len(clauses)}\n')
        for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
    print(len(clauses),len(extra),hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for key in ('tiles','geo','path','dest'):a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--cap',type=int,required=True);a.add_argument('--parity',action='store_true')
    export(**vars(a.parse_args()))
