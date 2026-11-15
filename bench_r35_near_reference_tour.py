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


def raw_cycles(p,mesh):
    # Build a graph on the 365 labelled physical strands. Each paired side
    # joins its endpoints bijectively at reversed slot positions.
    index={};n=0
    for t,tile in enumerate(p.tiles):
        for strand in tile['segments']:
            for side,k in strand:index[3*t+side,k]=n
            n+=1
    graph=[[] for _ in range(n)]
    for a,b in mesh:
        for k in range(1,12):
            if (a,k) in index:
                u,v=index[a,k],index[b,12-k];graph[u].append(v);graph[v].append(u)
    assert all(len(g)==2 for g in graph)
    pending=set(range(n));sizes=[]
    while pending:
        work=[pending.pop()];size=0
        while work:
            u=work.pop();size+=1
            for v in graph[u]:
                if v in pending:pending.remove(v);work.append(v)
        sizes.append(size)
    return sorted(sizes)


def run(tiles,phase,geo,seconds,destination):
    from ortools.sat.python import cp_model
    p,ref,g,req=problem_scope(tiles);g=g[:13]
    if geo:g+=[set(map(tuple,row))-req for row in read_json('docs/R35_GEOGRAPHIC_LOSS_CORES.json')['geographic_cores']]
    inv=inventory(p);sides=raw_sides(p.tiles)
    labels=[(slots,c^(phase>>(i//3)&1)) for i,(slots,c) in enumerate(sides)]
    m=cp_model.CpModel();links={e:m.new_bool_var('m%d_%d'%e) for e in sorted(inv.possible) if labels[e[1]]==reciprocal(labels[e[0]])}
    incident=defaultdict(list)
    for (a,b),v in links.items():incident[a].append(v);incident[b].append(v)
    for a in range(3*p.n):m.add_exactly_one(incident[a])
    for e in req:m.add(links.get(e,0)==1)
    for group in g:m.add(sum(links.get(e,0) for e in group)<=len(group)-1)
    m.maximize(sum(v for e,v in links.items() if e in ref))
    cli=cp_model.CpSolver();cli.parameters.num_search_workers=1
    report=dict(phase=hex(phase),geographic_cores=int(geo)*2,status='UNKNOWN',rounds=[],fingerprint=p.fingerprint)
    end=monotonic()+seconds;arcs=endpoints(p);slots=[u for u,_ in sides];seen=set()
    while monotonic()<end:
        cli.parameters.max_time_in_seconds=max(.001,end-monotonic())
        status=cli.solve(m)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            report['status']=cli.status_name(status);break
        mesh={e for e,v in links.items() if cli.value(v)}
        cycles=raw_cycles(p,mesh);cuts=loop_groups(arcs,slots,mesh,p.total_segments);assert (not cuts)==(cycles==[p.total_segments])
        row=dict(status=cli.status_name(status),loss=len(ref-mesh),lower_loss=len(ref)-cli.best_objective_bound,cycles=cycles,added=len(cuts-seen),branches=cli.num_branches,seconds=cli.wall_time)
        report['rounds'].append(row)
        if not cuts:
            table=[-1]*(3*p.n)
            for a,b in mesh:table[a]=b;table[b]=a
            codes=embedding(p,domain_bits(p),table);assert codes is None or p.validate(codes)['valid']
            report.update(status='GOLD' if codes else 'CONNECTED_SEWING',pairs=sorted(mesh),codes=codes,genus=ribbon_genus(p,sorted(mesh)),retained=len(ref&mesh),loss=len(ref-mesh))
            break
        if not cuts-seen:raise AssertionError('Previously excluded proper subloops returned.')
        for cut in cuts-seen:m.add(sum(links[e] for e in cut)<=len(cut)-1)
        seen|=cuts;atomic_json(destination,report)
    report['cuts']=len(seen);atomic_json(destination,report);return report

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--tiles',required=True);a.add_argument('--phase',type=lambda x:int(x,0),required=True)
    a.add_argument('--geographic-cores',action='store_true');a.add_argument('--seconds',type=float,default=120);a.add_argument('--output',type=Path,required=True)
    v=a.parse_args();result=run(v.tiles,v.phase,v.geographic_cores,v.seconds,v.output);print(result['status'],len(result['rounds']),result.get('loss'),result.get('genus'))
