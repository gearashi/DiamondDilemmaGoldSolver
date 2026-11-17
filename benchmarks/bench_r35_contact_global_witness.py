"""Find and independently reprice actual signed witnesses at the strong geographic threshold."""
import argparse
from itertools import chain
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_integer_sat import InventorySAT
from r35_loss_witness import cover_missing,physical_price


def run(tiles,geom,paths,out,cap,radii,budget):
    from pysat.card import CardEnc
    from pysat.solvers import Solver
    p,ref,gs,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,gs[:13],req)
    seeds=[0xce55aaa6e89109fca073547e8c580ba32e0e5fa2,0xd237aba7e99101e88041403e8c589902025e5326]
    read=lambda path,k:[frozenset(map(tuple,g)) for g in json.loads(path.read_bytes())[k]]
    contact=[g for g in read(paths,'edge_perfect_cores') if len(g)<=13]
    groups=[*gs[:13],*(g-req for g in read(geom,'geographic_cores')+contact)]
    m=InventorySAT(p,packed,req,outside,cap,native=None,additional_groups=[g for g in groups if g not in packed])
    top=max(abs(v) for c in m.clauses for v in c)
    result=dict(cap=cap,queries=[])
    for seed in seeds:
        _,failed=physical_price(p.tiles,ref,groups,req,seed)
        initial=(failed&ref)|cover_missing(groups,req,failed)
        for width in radii:
            signs=[i+1 if (seed>>i)&1 else -i-1 for i in range(1,p.n)]
            extra=CardEnc.atmost(lits=[-v for v in signs],bound=width,top_id=top).clauses if width<p.n else []
            with Solver(name='cadical153',bootstrap_with=m.clauses+extra) as solver:
                solver.set_phases(signs+[v if pair in initial else -v for pair,v in m.physical_losses.items()])
                solver.conf_budget(budget);answer=solver.solve_limited()
                row=dict(seed=hex(seed),radius=width,answer=answer,**solver.accum_stats())
                if answer:
                    phase=sum(1<<(v-1) for v in solver.get_model() if 0<v<=p.n)
                    cost,failed=physical_price(p.tiles,ref,groups,req,phase)
                    losses=(failed&ref)|cover_missing(groups,req,failed)
                    assert len(losses)==cost<=cap and all(g&losses for g in groups) and not (req&losses)
                    row.update(phase=hex(phase),cost=cost,lost=sorted(losses))
            result['queries'].append(row);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            if answer:return


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('tiles','geom','paths','out'):cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--cap',required=True,type=int);cli.add_argument('--radii',nargs='+',type=int,default=[4,8,16,32,160])
    cli.add_argument('--budget',type=int,default=30000)
    args=cli.parse_args();run(**vars(args))
