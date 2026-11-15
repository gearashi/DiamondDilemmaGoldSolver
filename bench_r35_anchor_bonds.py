"""Measure reuse of geographic phase exclusions from physical-bond restrictions alone."""
import argparse
from pathlib import Path
from random import Random

from bench_r35_mixed_sewings import original_sources
from io_utils import atomic_json
from r28_exchange import read_json
from r29_gold_phase import SignedParity
from r30_gold import search
from r35_anchor_rules import PROVED_PHASE, check_rules, has_proved_mesh, project_rule
from search_problem import SearchProblem


def different_artwork_control(problem):
    """Plant the archived impossible phase on other artwork on the very same mesh."""
    from ortools.sat.python import cp_model
    from bench_r35_mixed_sewings import labels
    from r29_gold_phase import EndpointCuts
    from r31_geographic import physical_bonds

    model = cp_model.CpModel()
    choices = [(a,b,model.new_bool_var(f"e{a}_{b}")) for left,s,right,t in problem.edges
               for a,b in ((left,right),(right,left))]
    model.add_circuit(choices)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 60
    if solver.solve(model) not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError("Failed to construct the planted mesh cycle.")
    cycle = {tuple(sorted((a,b))) for a,b,active in choices if solver.value(active)}
    rng = Random(35083)
    points = [[] for _ in range(problem.n)]
    for a,s,b,t in problem.edges:
        if tuple(sorted((a,b))) in cycle:
            slot = rng.randint(1,11)
            points[a].append([s,slot])
            points[b].append([t,12-slot])
    if any(len(row)!=2 for row in points):
        raise AssertionError("The planted tour must cross two sides of every tile.")
    def tiles():
        return [{"id":f"planted-{i}","segments":[pair]} for i,pair in enumerate(points)]
    initial = SearchProblem(tiles(),problem.board)
    home = [3*t for t in range(problem.n)]
    sides = EndpointCuts(initial).sides
    before = labels(initial,physical_bonds(initial,home),sides)
    witness = []
    for tile,pair in enumerate(points):
        target = (PROVED_PHASE>>tile)&1
        rotation = next(s for s in range(3) if before[tile]^sides[3*tile+s][1]==target)
        points[tile] = [[(s-rotation)%3,k] for s,k in pair]
        witness.append(3*tile+rotation)
    other = SearchProblem(tiles(),problem.board)
    bonds = sorted(physical_bonds(other,witness))
    phases = labels(other,bonds,EndpointCuts(other).sides)
    if sum(value<<t for t,value in enumerate(phases))!=PROVED_PHASE or not other.validate(witness)["valid"]:
        raise ArithmeticError("Independently rederived loop or signs differ from the control.")
    full = {t:value for t,value in enumerate(phases)}
    ds = [1<<code for code in witness]
    modes = {}
    for mode in ("off",None,"rules"):
        result = search(other,geography=True,domains=ds,pins=full,bonds=bonds,seconds=60,phase_anchors=mode)
        if result["status"]!="GOLD" or not other.validate(result["codes"])["valid"]:
            raise ArithmeticError(f"Valid changed-artwork Gold failed under {mode}: {result['status']}")
        modes[mode or "default"]={k:result[k] for k in ("status","branches","cp_calls","phase_rules") if k in result}
    return dict(planted=True,phase=hex(PROVED_PHASE),different_signed_mesh=not has_proved_mesh(other),
                checked_edges=len(bonds),public=modes)


def measure(problem, public=False):
    docs = Path(__file__).resolve().parent / "docs"
    rules = read_json(docs/"R35_PHASE_ANCHOR_CORES.json")["rules"]
    conditions = [tuple((t, (PROVED_PHASE>>t)&1) for t in range(problem.n)),
                  *(tuple(map(tuple,q["pins"])) for q in rules)]
    donors = original_sources()
    donors["contact"] = read_json(docs/"R35_MIXED_CONTACT_DONOR.json")
    rows = []
    for name, donor in donors.items():
        earliest = []
        examples = []
        for seed in (29, 71, 223, 2026):
            bonds = list(map(tuple, donor["pairs"]))
            Random(seed).shuffle(bonds)
            parity = SignedParity(problem)
            first = None
            for n in range(0,len(bonds)+1):
                if n and not parity.bond(*bonds[n-1]):
                    raise ArithmeticError("Stored physical mating contradicts its own signed inventory.")
                if n%8 and n!=len(bonds):continue
                masks = [project_rule(parity.base,c) for c in conditions]
                if not any(row is not None and len(row)<=1 for row in masks):continue
                first = n
                if seed==29:examples = bonds[:n]
                break
            earliest.append(first)
        entry = dict(donor=name,physical_bonds=len(donor["pairs"]),first_proved_prefix=earliest)
        if public and examples:
            entry["public"]={}
            for mode in ("off", "roots", "rules", None):
                ans=search(problem,geography=True,bonds=examples,seconds=5,phase_anchors=mode)
                entry["public"][mode or "default"]={k:ans[k] for k in (
                    "status","branches","cp_calls","phase_anchors","phase_rules") if k in ans}
        rows.append(entry)
    checks = []
    for value in (PROVED_PHASE,PROVED_PHASE^((1<<problem.n)-1)):
        pins = {i:value>>i&1 for i in range(problem.n)}
        flags = check_rules(problem,pins,float("inf"))[1]
        if not flags["terminal"] or flags["replayed"]:
            raise ArithmeticError("The certified phase did not close at zero probes.")
        checks.append(flags)
    original_side = int(problem.sides[0,0])
    problem.sides[0,0]=(original_side+1)%3
    try:
        if has_proved_mesh(problem):
            raise ArithmeticError("Changed geometry reused the original proof.")
    finally:
        problem.sides[0,0]=original_side
    adverse = different_artwork_control(problem) if public else None
    return dict(fingerprint=problem.fingerprint,phase=hex(PROVED_PHASE),full_phase_and_complement=checks,
                other_artwork=adverse,
                changed_mesh_rejected=True,step=8,seeds=[29,71,223,2026],donors=rows)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--public",action="store_true")
    args=cli.parse_args()
    atomic_json(args.output,measure(SearchProblem(read_json(args.tiles)),args.public))
