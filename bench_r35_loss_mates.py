"""Check how many original physical mates can coexist once every side has a partner."""
import argparse
from pathlib import Path
from time import monotonic
from collections import defaultdict
from bench_r35_loss_tight import problem_scope,read_json,verify_21,physical_price,raw_sides
from r32_gold_bonds import inventory,loop_groups,endpoints
from r35_loss_witness import reciprocal
from io_utils import atomic_json

def fixed(p,ref,g,phase,seconds):
    from ortools.sat.python import cp_model
    inv=inventory(p);m=cp_model.CpModel()
    labels=[(s,c^(phase>>(i//3)&1)) for i,(s,c) in enumerate(raw_sides(p.tiles))]
    edges={e:m.new_bool_var('m%d_%d'%e) for e in sorted(inv.possible) if labels[e[1]]==reciprocal(labels[e[0]])}
    incidence=defaultdict(list)
    for (a,b),v in edges.items():incidence[a].append(v);incidence[b].append(v)
    for i in range(3*p.n):m.add_exactly_one(incidence[i])
    for e in inv.required:m.add(edges.get(e,0)==1)
    for group in g:m.add(sum(edges.get(e,0) for e in group)<=len(group)-1)
    m.maximize(sum(v for e,v in edges.items() if e in ref))
    s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=seconds
    ans=s.solve(m);row=dict(phase=hex(phase),signed_price=physical_price(p.tiles,ref,g,set(inv.required),phase)[0],
                status=s.status_name(ans),branches=s.num_branches,seconds=s.wall_time,possible_signed_pairs=len(edges))
    if ans in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        mesh={e for e,v in edges.items() if s.value(v)}
        assert len(mesh)==3*p.n//2 and len(set(i for pair in mesh for i in pair))==3*p.n and all(not group<=mesh for group in g)
        row.update(physical_pairing_price=len(ref-mesh),lower_cost=len(ref)-s.best_objective_bound,
                   full_pairs=sorted(mesh),closed_subloops=len(loop_groups(endpoints(p),[sl for sl,_ in raw_sides(p.tiles)],mesh,p.total_segments)))
    return row
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--tiles',required=True);a.add_argument('--seconds',type=float,default=25);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();p,ref,g,req=problem_scope(args.tiles);a21,_,_=verify_21(p,ref,g,req);a22=int(read_json('docs/R35_OVERLAP_BOUNDARY.json')['disjoint20']['phase'],16)
    r=[fixed(p,ref,g[:13],w,args.seconds) for w in (a21,a22)];atomic_json(args.output,r)
    for row in r: print({k:v for k,v in row.items() if k!='full_pairs'})
