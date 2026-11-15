"""Compare equivalent cardinality, higher-arity and exact integer slack implications."""
import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

from bench_r35_contact_conditioned import price
from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_contact_certificate import CERTIFICATE
from r35_contact_slack import excluded_losses
from r35_integer_sat import InventorySAT


def extra_cnf(terms, required, optional, ids, cap, mode, top):
    from pysat.card import CardEnc, ITotalizer
    from pysat.formula import IDPool
    pool=IDPool(start_from=top+1)
    M=terms['maximum'];slack=cap*M-terms['weight_sum'];degree=Counter()
    rows=[]
    for term in terms['weighted_cores']:
        g=frozenset(map(tuple,term['bonds']))-required;w=term['weight']
        for x in g:degree[x]+=w
        rows.append((g,w))
    clauses=[]
    def atmost(literals,bound):
        if bound<len(literals):
            enc=CardEnc.atmost(lits=literals,bound=bound,vpool=pool);clauses.extend(enc.clauses)
    losses={ids[b]:M-degree[b] for b in optional if M>degree[b]}
    if mode=='bands':
        for w in sorted(set(losses.values())):
            atmost([v for v in sorted(losses) if losses[v]>=w],slack//w)
        for g,w in rows:atmost([ids[b] for b in sorted(g)],slack//w+1)
        return clauses
    from r35_contact_slack import exact_budget_clauses
    return exact_budget_clauses(terms,required,optional,ids,cap,top)

def run(tiles,geo,paths,conditional,caps,budget,out):
    from pysat.solvers import Solver
    p,ref,gs,req=problem_scope(tiles);packed,_,outside=OutsideSyndromes.layout(ref,gs[:13],req)
    read=lambda file,key:[frozenset(map(tuple,g)) for g in json.loads(file.read_bytes())[key]]
    contact=[g-req for g in read(paths,'edge_perfect_cores') if len(g)<=13]
    groups=[*gs[:13],*(x-req for x in read(geo,'geographic_cores')),*contact]
    cert=json.loads(CERTIFICATE.read_bytes());terms=[frozenset(map(tuple,r['bonds'])) for r in cert['weighted_cores']]
    result=[]
    for cap in caps:
        model=InventorySAT(p,packed,req,outside,cap,native=None,additional_groups=[g for g in groups if g not in packed])
        top=max(abs(v) for c in model.clauses for v in c);ids=model.physical_losses
        single,pairs=excluded_losses(ref,req,cap,cert)
        basic=[[-ids[b]] for b in sorted(single)]+[[-ids[x],-ids[y]] for x,y in sorted(pairs)]
        for mode in ('base','pairs','bands','pairs_bands','exact','pairs_exact','conditional8','conditional32'):
            if mode.startswith('conditional'):
                n=int(mode[len('conditional'):]);file=Path(str(conditional).replace('{cap}',str(cap)))
                rows=json.loads(file.read_bytes())['conditionals']
                x,y,z=price(rows,terms,req,ref-req,cap,n,triples=True)
                extra=basic+[[-ids[b]] for b in sorted(x)]+[[-ids[b] for b in q] for q in sorted(y|z)]
            elif mode in ('base','pairs'):extra=basic if mode=='pairs' else []
            else:
                extra=extra_cnf(cert,req,ref-req,ids,cap,mode.split('_')[-1],top)
                if mode.startswith('pairs_'):extra=basic+extra
            with Solver(name='cadical153',bootstrap_with=model.clauses+extra) as engine:
                engine.conf_budget(budget);result.append(dict(cap=cap,variant=mode,added=len(extra),
                    answer=engine.solve_limited(),**engine.accum_stats()))
            out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('tiles','geo','paths','conditional','out'):cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--caps',nargs='+',type=int,default=[26,27]);cli.add_argument('--budget',type=int,default=15000)
    a=cli.parse_args();run(**vars(a))
