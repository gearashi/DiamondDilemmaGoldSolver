"""Lift a physically supported near-reference phase to one complete strand circuit."""
import argparse
from collections import defaultdict
from pathlib import Path
from time import monotonic
from bench_r35_loss_tight import problem_scope,read_json,raw_sides
from r32_gold_bonds import inventory,endpoints,loop_groups
from r35_loss_witness import reciprocal
from r30_gold import embedding
from r31_geographic import ribbon_genus
from r27_structure_filters import domain_bits
from io_utils import atomic_json


from r35_phase_sewing import strand_components as raw_cycles


def run(tiles,phase,geo,seconds,destination,*,claims=None,contact_claims=None,price_bound=False,hint=None):
    from ortools.sat.python import cp_model
    p,ref,g,req=problem_scope(tiles);g=g[:13]
    from r35_geographic_replay import geographic_departures
    extras=()
    if geo or claims:
        path=claims or 'docs/R35_GEOGRAPHIC_LOSS_CORES.json'
        raw=[frozenset(map(tuple,row)) for row in read_json(path)['geographic_cores']]
        extras=geographic_departures(p,ref,req,raw)
        g.extend(extras)
    contacts=()
    if contact_claims:
        from r35_induced_contacts import contact_departures
        raw=[frozenset(map(tuple,row)) for row in read_json(contact_claims)['edge_perfect_cores']]
        contacts=contact_departures(p,ref,req,raw)
        g.extend(contacts)
    inv=inventory(p);sides=raw_sides(p.tiles)
    labels=[(slots,c^(phase>>(i//3)&1)) for i,(slots,c) in enumerate(sides)]
    m=cp_model.CpModel();links={e:m.new_bool_var('m%d_%d'%e) for e in sorted(inv.possible) if labels[e[1]]==reciprocal(labels[e[0]])}
    incident=defaultdict(list)
    for (a,b),v in links.items():incident[a].append(v);incident[b].append(v)
    for a in range(3*p.n):m.add_exactly_one(incident[a])
    for e in req:m.add(links.get(e,0)==1)
    for group in g:m.add(sum(links.get(e,0) for e in group)<=len(group)-1)
    retained=sum(v for e,v in links.items() if e in ref)
    m.maximize(retained)
    mate_bound=None
    if price_bound:
        from r35_mate_price import ClassPrices
        from r35_loss_witness import physical_price
        ordinary=physical_price(p.tiles,ref,g,req,phase)
        if ordinary is None:raise ValueError('Phase fails signed inventory.')
        mate_bound=max(ordinary[0],ClassPrices(p,ref,g,req,limit=12).memo(phase))
        m.add(retained<=len(ref)-mate_bound)
    if hint:
        old=read_json(hint)
        if old.get('phase')!=hex(phase) or old.get('fingerprint')!=p.fingerprint:
            raise ValueError('Sewing hint must belong to the same phase and puzzle.')
        saved=set(map(tuple,old.get('pairs',old.get('first_pairing',()))))
        for edge,var in links.items():m.add_hint(var,int(edge in saved))
    cli=cp_model.CpSolver();cli.parameters.num_search_workers=1
    report=dict(phase=hex(phase),geographic_cores=len(extras),edge_perfect_cores=len(contacts),status='UNKNOWN',rounds=[],fingerprint=p.fingerprint)
    if price_bound:report['certified_mate_bound']=mate_bound
    end=monotonic()+seconds;arcs=endpoints(p);slots=[u for u,_ in sides];seen=set()
    while monotonic()<end:
        cli.parameters.max_time_in_seconds=max(.001,end-monotonic())
        status=cli.solve(m)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            report['status']=cli.status_name(status);break
        mesh={e for e,v in links.items() if cli.value(v)}
        cycles=raw_cycles(p,mesh);cuts=loop_groups(arcs,slots,mesh,p.total_segments);assert (not cuts)==(cycles==[p.total_segments])
        row=dict(status=cli.status_name(status),loss=len(ref-mesh),lower_loss=len(ref)-cli.best_objective_bound,cycles=cycles,added=len(cuts-seen),branches=cli.num_branches,seconds=cli.wall_time)
        if claims:
            row['new_loop_cores']=[sorted(c) for c in sorted(cuts-seen,key=lambda c:(len(c),sorted(c)))]
            if not report['rounds']:
                report['first_pairing']=sorted(mesh)
        report['rounds'].append(row)
        if not cuts:
            table=[-1]*(3*p.n)
            for a,b in mesh:table[a]=b;table[b]=a
            codes=embedding(p,domain_bits(p),table);assert codes is None or p.validate(codes)['valid']
            report.update(status='GOLD' if codes else 'CONNECTED_SEWING',pairs=sorted(mesh),codes=codes,genus=ribbon_genus(p,sorted(mesh)),retained=len(ref&mesh),loss=len(ref-mesh))
            break
        if not cuts-seen:raise AssertionError('Previously excluded proper subloops returned.')
        for cut in cuts-seen:m.add(sum(links[e] for e in cut)<=len(cut)-1)
        if price_bound:
            m.clear_hints()
            for e,var in links.items():m.add_hint(var,int(e in mesh))
        seen|=cuts;atomic_json(destination,report)
    report['cuts']=len(seen);atomic_json(destination,report);return report

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--tiles',required=True);a.add_argument('--phase',type=lambda x:int(x,0),required=True)
    a.add_argument('--geographic-cores',action='store_true');a.add_argument('--seconds',type=float,default=120);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--claims',type=Path,help='Replay the specified geographic departures before lifting the phase.')
    a.add_argument('--contact-claims',type=Path,help='Replay edge-perfect incidental contact conflicts on the original mesh.')
    a.add_argument('--price-bound',action='store_true',help='Enforce the independently checked phasewise matching lower bound.')
    a.add_argument('--hint',type=Path,help='Use a physical pairing from a previous solve on the same phase.')
    v=a.parse_args();result=run(v.tiles,v.phase,v.geographic_cores,v.seconds,v.output,claims=v.claims,contact_claims=v.contact_claims,price_bound=v.price_bound,hint=v.hint);print(result['status'],len(result['rounds']),result.get('loss'),result.get('genus'))
