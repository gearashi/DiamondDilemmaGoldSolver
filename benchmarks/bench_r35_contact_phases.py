"""Price all incumbent-neighbor phases under typed induced-contact portfolios and independent CP covers."""
import argparse
import json
from pathlib import Path
from time import monotonic

from bench_r35_loss_boundary344 import problem_scope
from r35_cover_search import reduced_cover
from r35_loss_witness import physical_price


def native(rows,upper):
    from ortools.sat.python import cp_model
    if not rows:return 0,True
    m=cp_model.CpModel();vs={p:m.new_bool_var('') for p in sorted(set().union(*rows))}
    for g in rows:m.add_bool_or([vs[p] for p in sorted(g)])
    m.add(sum(vs.values())<len(upper))
    e=cp_model.CpSolver();e.parameters.num_search_workers=1
    e.parameters.max_time_in_seconds=30
    status=e.solve(m)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise ArithmeticError('The independent solver found a cheaper assignment.')
    return e.num_branches,status==cp_model.INFEASIBLE


def run(tiles, paths, prior, geom, out, level, checked):
    p,r,gs,req=problem_scope(tiles)
    evidence=json.loads(prior.read_bytes())
    geo=[frozenset(map(tuple,g))-req for g in json.loads(geom.read_bytes())['geographic_cores']]
    source=json.loads(paths.read_bytes())
    contact=[frozenset(map(tuple,g))-req for g in source['edge_perfect_cores'] if len(g)<=level]
    result=dict(level=level,contacts=len(contact),prices=[])
    start=monotonic()
    for j,old in enumerate(evidence['fixed_prices']['prices']):
        phase=int(old['phase'],16)
        _,failed=physical_price(p.tiles,r,gs,req,phase)
        rows=[frozenset(g)-req for g in [*gs,*geo,*contact] if not failed&g]
        chosen=reduced_cover(rows)
        if not all(g&chosen for g in rows):raise AssertionError('Unmet physical departure.')
        cost=len(r&failed)+len(chosen)
        if cost<old['expanded']:raise AssertionError('Adding a physical obligation reduced the optimum.')
        rec=dict(phase=hex(phase),base=old['expanded'],cost=cost,remaining=len(rows))
        if j<checked or phase==0xd237aba7e91101e88041413e8c589902021e5126:
            work,complete=native(rows,chosen)
            rec['native_branches']=work
            if not complete and level>13:
                short=[frozenset(map(tuple,g))-req for g in source['edge_perfect_cores'] if len(g)<=13]
                weaker=[frozenset(g)-req for g in [*gs,*geo,*short] if not failed&g]
                rec['weaker_native_branches'],complete=native(weaker,chosen)
            rec['native_lower_verified']=complete
        result['prices'].append(rec);result['seconds']=monotonic()-start
        out.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf8')


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    for field in ('tiles','paths','prior','geom','out'):arg.add_argument('--'+field,required=True,type=Path)
    arg.add_argument('--level',required=True,type=int);arg.add_argument('--checked',default=4,type=int)
    args=arg.parse_args();run(**vars(args))
