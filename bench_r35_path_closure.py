"""Price all substitution depths together at fixed signs, then compare with single proper-loop substitutions."""
import argparse
from collections import defaultdict
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from io_utils import atomic_json
from r28_exchange import read_json
from r32_gold_bonds import endpoints, loop_groups
from r35_loss_witness import raw_sides
from verify_r35_path_circuits import labelled_sides, opposite


def reachable_rules(screen, initial):
    rules=set()
    for block in screen["sources"].values():
        for row in block["cases"]:
            path=frozenset(map(tuple,row["bonds"]))
            for alternative in row["alternatives"]:
                if len(alternative)==1:
                    rules.add((path,tuple(alternative[0])))
    available=set(initial)
    used=set()
    waves=[]
    while True:
        new={(path,e) for path,e in rules-used if path<=available}
        if not new:break
        used.update(new)
        extra={mate for _,mate in new}-available
        waves.append(dict(paths=len(new),new_mates=len(extra)))
        available.update(extra)
    return used,available,waves


def native_cost(p, reference, required, groups, phase, rules, available, seconds):
    from ortools.linear_solver import pywraplp
    solver=pywraplp.Solver.CreateSolver("SCIP")
    solver.SetNumThreads(1)
    solver.SetTimeLimit(round(1000*seconds))
    labels=labelled_sides(p.tiles,phase)
    keep={e:solver.BoolVar(f"kept_{e[0]}_{e[1]}") for e in sorted(available)}
    owners=defaultdict(list)
    for (a,b),v in keep.items():
        if labels[a]!=opposite(labels[b]):solver.Add(v==0)
        if (a,b) in required:solver.Add(v==1)
        owners[a].append(v);owners[b].append(v)
    for row in owners.values():
        if len(row)>1:solver.Add(sum(row)<=1)
    for g in groups:solver.Add(sum(keep[e] for e in g)<=len(g)-1)
    for path,e in sorted(rules,key=lambda r:(len(r[0]),sorted(r[0]),r[1])):
        solver.Add(sum(keep[b] for b in path)-keep[e]<=len(path)-1)
    solver.Minimize(len(reference)-sum(keep[e] for e in reference))
    status=solver.Solve()
    return dict(status={solver.OPTIMAL:"OPTIMAL",solver.FEASIBLE:"FEASIBLE",solver.INFEASIBLE:"INFEASIBLE"}.get(status,"UNKNOWN"),
                lower=solver.Objective().BestBound(),
                loss=round(solver.Objective().Value()) if status in (solver.OPTIMAL,solver.FEASIBLE) else None,
                nodes=solver.nodes())


def main(args):
    docs=Path(__file__).parent/"docs"
    p,ref,physical,req=problem_scope(args.tiles)
    screen=read_json(args.screen)
    if screen["fingerprint"]!=p.fingerprint:raise ValueError("Different puzzle")
    learned,available,waves=reachable_rules(screen,ref|req)
    # Independent replay of each path and its alternatives is a prerequisite for use.
    from verify_r35_path_mates import VectorReplay
    check=VectorReplay(p)
    by_path=defaultdict(list)
    for path,e in learned:by_path[path].append({e})
    for path,qs in by_path.items():
        geo,matched,misses=check.count(path,qs)
        if not matched or any(misses):raise ArithmeticError("Missed geographic placement")
    rows=[*physical[:13]]
    for name,key in (("R35_GEOGRAPHIC_RESIDUALS.json","geographic_cores"),("R35_EDGE_CONTACTS.json","edge_perfect_cores")):
        rows.extend(frozenset(map(tuple,g)) for g in read_json(docs/name)[key])
    composed=read_json(docs/"R35_PATH_CIRCUITS.json")
    circuit_rows=[frozenset(map(tuple,h["path"])) | (frozenset(map(tuple,h["loop"]))-{tuple(h["mate"])})
                  for h in composed["gold_path_circuits"]]
    # Requiring a mate outside the original reference can close an additional proper circuit.
    extras=available-(ref|req)
    arcs=endpoints(p); slots=[s for s,_ in raw_sides(p.tiles)]
    loops=set()
    for mate in extras:
        others={b for b in ref if not set(b)&set(mate)}
        loops.update(g for g in loop_groups(arcs,slots,others|{mate},p.total_segments) if mate in g)
    phases=dict(raw31=0xce77aac6e89509e8867b457e8c580a233a5e5f22,
                euler33=0xcc57eae4e99109f8b073557e8c590b9202465766,
                euler33b=0xcc778d82e99c517987f6b57ec15837c21207cf06,
                donor37=0xd237aba7e91101e88041413e8c589902021e5126)
    out=dict(fingerprint=p.fingerprint,reference=len(ref),reachable_rules=len(learned),
             reachable_mates=len(available),additional_loops=len(loops),waves=waves,phases={})
    for key,phase in phases.items():
        results={}
        for mode in ("published","closure"):
            extra=(mode=="closure")
            results[mode]=native_cost(p,ref,req,[*rows,*circuit_rows,*(loops if extra else ())],phase,
                                      learned if extra else (),available if extra else ref|req,args.seconds)
        before=results["published"]
        if before["status"]=="OPTIMAL" and before["loss"]!=composed["fixed_phase_prices"]["13"][key]["published"]:
            raise ValueError("Fixed-price control differs.")
        out["phases"][key]=results
        print(key,results,flush=True)
        atomic_json(args.output,out)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",required=True,type=Path)
    cli.add_argument("--screen",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--seconds",type=float,default=120)
    main(cli.parse_args())
