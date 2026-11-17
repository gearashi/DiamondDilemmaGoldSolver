"""Compare cap-conditioned individual and pair deductions on the exact same native CNF."""
import argparse
import hashlib
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_contact_slack import excluded_losses
from r35_contact_certificate import CERTIFICATE,verify
from r35_integer_sat import InventorySAT


def run(tiles,geo,paths,cap,budget,out,cnf,parity=False):
    from pysat.solvers import Solver
    p,ref,groups,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],req)
    read=lambda path,key:[frozenset(map(tuple,g)) for g in json.loads(path.read_bytes())[key]]
    geom=read(geo,'geographic_cores');contact=[g for g in read(paths,'edge_perfect_cores') if len(g)<=13]
    family=[*(g-req for g in geom),*(g-req for g in contact)]
    cert=json.loads(CERTIFICATE.read_bytes())
    assert verify(p,ref,req,groups[:13],cert,charged=[g-req for g in [*groups[:13],*family]])==25
    sat=InventorySAT(p,packed,req,outside,cap,native=None,parity_only=parity,
                     additional_groups=[g for g in [*groups[:13],*family] if g not in packed])
    unit,pair=excluded_losses(ref,req,cap,cert)
    u=[[-sat.physical_losses[b]] for b in sorted(unit)]
    two=[[-sat.physical_losses[a],-sat.physical_losses[b]] for a,b in sorted(pair)]
    result=dict(puzzle=p.fingerprint,cap=cap,units=len(u),pairs=len(two),queries=[])
    for label,extra in [('base',[]),('units',u),('units_pairs',u+two)]:
        clauses=sat.clauses+extra
        if cnf and label=='units_pairs':
            with cnf.open('w',encoding='ascii',newline='\n') as f:
                f.write(f'p cnf {max(abs(v) for c in clauses for v in c)} {len(clauses)}\n')
                for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
            result['sha256']=hashlib.sha256(cnf.read_bytes()).hexdigest()
        with Solver(name='cadical153',bootstrap_with=clauses) as s:
            s.conf_budget(budget);answer=s.solve_limited()
            result['queries'].append(dict(variant=label,answer=answer,**s.accum_stats()))
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('tiles','geo','paths','out'):cli.add_argument('--'+k,required=True,type=Path)
    cli.add_argument('--cap',type=int,default=25);cli.add_argument('--budget',type=int,default=15000)
    cli.add_argument('--cnf',type=Path);cli.add_argument('--parity',action='store_true')
    a=cli.parse_args();run(**vars(a))
