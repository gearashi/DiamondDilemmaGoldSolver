"""Geographic, phase, and connected-witness residual searches for one-loop Gold."""
import argparse
import json
from pathlib import Path

from geometry import build_board
from r29_gold_cut_count import cut_witness
from r29_gold_geography import checked_domains, layer_bound, partition_capacity
from r29_gold_phase import EndpointCuts, SignedParity, checked_pins, propagate_rules
from search_problem import SearchProblem


def phase_search(problem, *, pins=None, bonds=(), width=2, max_subsets=10000,
                 boundary=(), seconds=3, cp=True):
    from r28_geo_color import signed_phase_model
    if seconds <= 0:
        raise ValueError("Expected positive query time.")
    cuts = EndpointCuts(problem)
    parity = SignedParity(problem,cuts.sides)
    base = parity.base
    metrics = dict(rank=len(base.pivots),equations=parity.equations,cp_calls=0)
    pins = checked_pins(problem.n,pins or {})
    bonds = tuple(tuple(pair) for pair in bonds)
    for a,b in bonds:
        parity.bond(a,b)
    for t,v in pins.items():
        parity.pin(t,v)
    def inconsistent():
        return dict(status="INFEASIBLE",reason="parity",core=sorted(base.conflict),metrics=metrics)
    if base.conflict is not None:
        return inconsistent()
    rules,summary = cuts.clauses(width,max_subsets)
    metrics.update(summary)
    values = dict(pins)
    while True:
        linear = base.forced()
        values.update(linear)
        propagated = propagate_rules(rules,values)
        if propagated is None:
            return dict(status="INFEASIBLE",reason="endpoint-cut",metrics=metrics)
        added = [(t,v) for t,v in propagated.items() if t not in values]
        values = propagated
        if not added:
            break
        for t,value in added:
            if not parity.pin(t,value):
                return inconsistent()
    if not cuts.possible_tour(values,bonds):
        return dict(status="INFEASIBLE",reason="physical-bundles",metrics=metrics)
    if any(type(t) is not int or not 0<=t<problem.n for t in boundary):
        raise ValueError("Invalid boundary tile ID.")
    scope = sum(1<<t for t in set(boundary))
    projection = [dict(mask=hex(m),rhs=v,sources=sorted(s)) for m,v,s in base.project(scope)]
    if not cp:
        return dict(status="UNKNOWN",metrics=metrics,projection=projection,phases=values)
    from ortools.sat.python import cp_model
    # Explicit pins (including {}) preserve requested phase and global complement choices.
    model,x = signed_phase_model(problem,pins=values,bonds=bonds)
    for rule in rules:
        model.add_bool_or([x[t] if val==0 else ~x[t] for t,val in rule])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    metrics["cp_calls"] = 1
    result = solver.solve(model)
    return dict(status=solver.status_name(result),metrics=metrics,projection=projection,phases=values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["phase","rings","soft-rings","capacity","cut-witness"])
    p.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    p.add_argument("--input",type=Path)
    p.add_argument("--subdivision",type=int,default=4)
    p.add_argument("--seconds",type=float,default=3)
    p.add_argument("--max-work",type=int,default=50000)
    args = p.parse_args()
    problem = SearchProblem(json.loads(args.tiles.read_text(encoding="utf-8-sig")),build_board(args.subdivision))
    data = json.loads(args.input.read_text(encoding="utf-8-sig")) if args.input else {}
    def domains():
        if "domains" not in data:
            return checked_domains(problem,None)
        out = []
        for row in data["domains"]:
            if not isinstance(row,list) or any(type(v) is not int or not 0<=v<3*problem.n for v in row):
                raise ValueError("Invalid placement list.")
            out.append(sum(1<<v for v in set(row)))
        return checked_domains(problem,out)
    pins = {int(k):v for k,v in data.get("pins",{}).items()}
    if args.mode == "phase":
        result = phase_search(problem,pins=pins,bonds=data.get("bonds",()),seconds=args.seconds,
                              width=data.get("width",2),boundary=data.get("boundary",()),
                              cp=data.get("cp",True),max_subsets=args.max_work)
    elif args.mode in ("rings","soft-rings"):
        result = layer_bound(problem,domains(),gold=args.mode=="rings",detail=data.get("detail","rings"),
                             seconds=args.seconds,pins=pins,max_atoms=args.max_work)
    elif args.mode == "capacity":
        result = partition_capacity(problem,domains(),pins=pins,bags=data.get("bags"),
                                    width=data.get("width",2),max_bags=args.max_work)
    else:
        cells = data.get("inside",[c.index for c in problem.board.cells if c.face<5])
        result = cut_witness(problem,domains(),cells,seed=data.get("seed",29),trials=data.get("trials",4),
                             max_work=args.max_work)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
