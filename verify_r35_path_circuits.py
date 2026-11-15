"""Replay path-to-loop deductions using separate pose, strand and fixed-phase optimizers."""
import argparse
from collections import Counter
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from io_utils import atomic_json
from r28_exchange import read_json
from r35_phase_sewing import strand_components
from verify_r35_path_mates import VectorReplay


def labelled_sides(tiles,phase):
    sides=[]
    for tile,record in enumerate(tiles):
        color=(phase>>tile)&1
        for side in range(3):
            slots=sorted(k for pair in record["segments"] for s,k in pair if s==side)
            sides.append((tuple(slots),color))
            color ^= len(slots)%2
    return sides


def opposite(label):
    slots,sign=label
    return tuple(12-k for k in reversed(slots)),sign^(len(slots)%2)


def fixed_cost(problem, ref, required, groups, phase):
    from ortools.linear_solver import pywraplp
    labels=labelled_sides(problem.tiles,phase)
    counts=Counter(labels)
    if any(counts[opposite(label)]!=n or (label==opposite(label) and n%2)
           for label,n in counts.items()):
        raise ArithmeticError("Fixed signs fail integer inventory.")
    m=pywraplp.Solver.CreateSolver("SCIP")
    m.SetNumThreads(1)
    loss={e:m.BoolVar(f"lost_{e[0]}_{e[1]}") for e in sorted(ref|required)}
    for (a,b),v in loss.items():
        if (a,b) in required:m.Add(v==0)
        if labels[a]!=opposite(labels[b]):m.Add(v==1)
    for group in groups:m.Add(sum(loss[e] for e in group)>=1)
    m.Minimize(sum(loss[e] for e in ref))
    if m.Solve()!=pywraplp.Solver.OPTIMAL:
        raise ArithmeticError("No independent optimum for the fixed signs.")
    return round(m.Objective().Value())


def verify(problem,ref,physical,req,data):
    if problem.fingerprint!=data["fingerprint"]:
        raise ValueError("Mismatched puzzle.")
    vector=VectorReplay(problem)
    projected=[]
    for hint in data["gold_path_circuits"]:
        path=frozenset(map(tuple,hint["path"]))
        mate=tuple(hint["mate"])
        loop=frozenset(map(tuple,hint["loop"]))
        _,matched,misses=vector.count(path,[{mate}])
        closed=strand_components(problem,loop,complete=False)
        if mate not in loop or not matched or any(misses) or not any(0<n<problem.total_segments for n in closed):
            raise ValueError("Not a verified geographic implication into a proper loop.")
        projected.append(path|(loop-{mate}))
    docs=Path(__file__).parent/"docs"
    geo=[frozenset(map(tuple,g)) for g in read_json(docs/"R35_GEOGRAPHIC_RESIDUALS.json")["geographic_cores"]]
    negative=[frozenset(map(tuple,g)) for g in read_json(docs/"R35_EDGE_CONTACTS.json")["edge_perfect_cores"]]
    phases=dict(raw31=0xce77aac6e89509e8867b457e8c580a233a5e5f22,
                euler33=0xcc57eae4e99109f8b073557e8c590b9202465766,
                euler33b=0xcc778d82e99c517987f6b57ec15837c21207cf06,
                donor37=0xd237aba7e91101e88041413e8c589902021e5126)
    prices={}
    for count in (13,344):
        groups=[*physical[:count],*geo,*negative]
        prices[count]={}
        for name,phase in phases.items():
            got=[fixed_cost(problem,ref,req,scope,phase) for scope in (groups,[*groups,*projected])]
            saved=data["fixed_phase_prices"][str(count)][name]
            if got!=[saved["before"],saved["published"]]:
                raise ValueError("Independent fixed-phase prices differ.")
            prices[count][name]=got
    return dict(fingerprint=problem.fingerprint,verified=len(projected),labelled_roots=3*problem.n*len(projected),
                optimizer="SCIP, one worker",independent_fixed_prices=prices)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--claims",type=Path,default=Path(__file__).parent/"docs/R35_PATH_CIRCUITS.json")
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    result=verify(*problem_scope(args.tiles),read_json(args.claims))
    atomic_json(args.output,result)
    print(result)
