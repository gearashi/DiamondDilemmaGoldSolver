"""Separate signed inconsistencies and authentic physical contacts at fractional reference losses."""
from collections import defaultdict
import argparse
from heapq import heappop,heappush
import json
from pathlib import Path
from random import Random
from time import monotonic

from bench_r35_induced_contacts import assess,incidence
from bench_r35_loss_boundary344 import problem_scope
from r35_contact_certificate import xor_conflict
from r35_induced_contacts import InducedContacts
from r35_loss_witness import raw_sides


def shortest(graph,start,end,cost,exclude=None):
    # Scalar distances are nonnegative. The tiny tie breakers only order paths;
    # exact certification and the true unperturbed LP charge are checked later.
    heap=[(0,start)];dist={start:0};prev={}
    while heap:
        w,x=heappop(heap)
        if w!=dist[x]:continue
        if x==end:break
        for v,e in graph[x]:
            if e==exclude:continue
            d=w+cost[e]
            if d>=dist.get(v,float('inf')):continue
            dist[v]=d;prev[v]=(x,e);heappush(heap,(d,v))
    if end not in prev:return None
    out=set();x=end
    while x!=start:
        x,e=prev[x];out.add(e)
    return frozenset(out)


def solve(tiles,geo,paths,output,seconds,iterations,alternatives):
    from ortools.linear_solver import pywraplp
    p,ref,gs,req=problem_scope(tiles);ref=frozenset(ref);req=frozenset(req)
    load=lambda f,k:[frozenset(map(tuple,g)) for g in json.loads(f.read_bytes())[k]]
    old=[*gs,*load(geo,'geographic_cores'),*load(paths,'edge_perfect_cores')]
    groups={frozenset(g)-req for g in old}
    lp=pywraplp.Solver.CreateSolver('GLOP')
    var={b:lp.NumVar(0,1,'') for b in sorted(ref-req)}
    for g in groups:lp.Add(sum(var[b] for b in sorted(g))>=1)
    lp.Minimize(sum(var.values()))
    codes=json.loads((Path(__file__).parent/'docs/R29_INCUMBENT.json').read_bytes())['codes']
    bad=[(codes[r['cell_a']]//3,codes[r['cell_b']]//3) for r in p.validate(codes)['unmatched_edges']]
    phy=defaultdict(list);signed=defaultdict(list);labels=raw_sides(p.tiles)
    for b in sorted(ref|req):
        u,v=(d//3 for d in b)
        sign=labels[b[0]][1]^labels[b[1]][1]^(len(labels[b[0]][0])%2)
        if b in ref:phy[u].append((v,b));phy[v].append((u,b))
        for j in range(2):
            a,z=2*u+j,2*v+(j^sign)
            signed[a].append((z,b));signed[z].append((a,b))
    ch=InducedContacts(p);table=incidence(p);rng=Random(81358)
    extra=defaultdict(list);audit=[];start=monotonic()
    for serial in range(iterations):
        assert lp.Solve()==lp.OPTIMAL
        values={b:v.solution_value() for b,v in var.items()};values.update({b:0 for b in req})
        cost={b:max(0,v)+1e-8*(1+rng.random()) for b,v in values.items()}
        new={}
        def trial(g,kind):
            if not g or g-req in groups or g in new or sum(values[b] for b in g)>=1-1e-7:return
            if kind=='required_xor':
                if not xor_conflict(p,g|req):return
            else:
                if not ch.proves(g):return
                independent=assess(p,table,g)
                if not independent['poses'] or independent['compatible']:raise ArithmeticError('Independent path check failed.')
            new[g]=kind
        for x in range(p.n):trial(shortest(signed,2*x,2*x+1,cost),'required_xor')
        for a,b in bad:
            case=shortest(phy,a,b,cost)
            trial(case,'edge_perfect_cores')
            if case and sum(values[e] for e in case)<1-1e-7:
                for omit in sorted(case)[:alternatives]:trial(shortest(phy,a,b,cost,omit),'edge_perfect_cores')
        for g,k in new.items():
            projected=g-req;groups.add(projected);extra[k].append(sorted(g))
            lp.Add(sum(var[b] for b in sorted(projected))>=1)
        lower=sum(values[b] for b in ref-req)
        audit.append(dict(iteration=serial,lp=lower,signed=sum(k=='required_xor' for k in new.values()),
                          physical=sum(k=='edge_perfect_cores' for k in new.values()),
                          max_length=max(map(len,new),default=0),time=monotonic()-start))
        output.write_text(json.dumps(dict(fingerprint=p.fingerprint,rounds=audit,**extra),separators=(',',':'))+'\n',encoding='utf8')
        if monotonic()-start>=seconds or (serial>=3 and all(not (r['signed']+r['physical']) for r in audit[-3:])):return


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__)
    for k in ('tiles','geo','paths','output'):c.add_argument('--'+k,type=Path,required=True)
    c.add_argument('--seconds',type=int,default=100);c.add_argument('--iterations',type=int,default=96)
    c.add_argument('--alternatives',type=int,default=32)
    solve(**vars(c.parse_args()))
