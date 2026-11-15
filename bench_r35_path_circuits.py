"""Compose verified geographic path implications with a proper loop on named tile sides."""
import argparse
from collections import Counter
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from io_utils import atomic_json
from r28_exchange import read_json
from r32_gold_bonds import endpoints, loop_groups
from r35_loss_witness import physical_price, raw_sides
from r35_path_mates import verified_circuits


def compose(problem, reference, required, screen):
    if problem.fingerprint != screen["fingerprint"]:
        raise ValueError("The path screen is for different artwork or board.")
    slots = [s for s, _ in raw_sides(problem.tiles)]
    arcs = endpoints(problem)
    allowed = reference | required
    by_side = {d: bond for bond in reference for d in bond}
    circuits = {}
    claims = {}
    for block in screen["sources"].values():
        for row in block["cases"]:
            path = frozenset(map(tuple, row["bonds"]))
            if not path <= allowed:
                continue
            for alternatives in row["alternatives"]:
                if len(alternatives) != 1:
                    continue
                mate = tuple(alternatives[0])
                if mate not in circuits:
                    # Other uses of this mate's two physical sides cannot occur with it.
                    keep = (reference - {by_side.get(d) for d in mate}) | {mate}
                    circuits[mate] = [g for g in loop_groups(arcs, slots, keep, problem.total_segments)
                                      if mate in g]
                for loop in circuits[mate]:
                    group = path | (loop - {mate})
                    claims.setdefault(group, dict(path=sorted(path), mate=mate, loop=sorted(loop)))
    if set(verified_circuits(problem, list(claims.values()), allowed)) != set(claims):
        raise ArithmeticError("Some composed departure did not replay from the source puzzle.")
    return claims


def native_poses(problem, hints):
    """Compare translating a compound cut with checking the actual pinned cells."""
    from r32_gold_watch import placed_bonds
    from r35_geographic_packing_check import component_poses
    from r35_phase_sewing import strand_components
    from validator import reverse_mask

    neighbors={}
    for c,s,d,t in problem.edges:
        neighbors[c,s]=d,t
        neighbors[d,t]=c,s
    result=Counter()
    for hint in hints:
        mate=tuple(hint["mate"])
        loop=frozenset(map(tuple,hint["loop"]))
        group=frozenset(map(tuple,hint["path"])) | (loop-{mate})
        if not any(0<n<problem.total_segments for n in strand_components(problem,loop,complete=False)):
            raise ArithmeticError("The forced circuit was not proper.")
        geometric=0
        for _,poses in component_poses(problem.n,neighbors,group):
            geometric+=1
            placed={cell:3*tile+rotation for tile,(cell,rotation) in poses.items()}
            if any(int(problem.masks[placed[c],s])!=reverse_mask(int(problem.masks[placed[d],t]))
                   for c,s,d,t in problem.edges if c in placed and d in placed):
                result["direct_edge_mismatch"]+=1
                continue
            if not loop<=placed_bonds(problem,placed.items()).keys():
                raise ArithmeticError("An internally matched root failed to contain the physical circuit.")
            result["direct_proper_loop"]+=1
        result["zero_geometric_roots"]+=3*problem.n-geometric
    result["roots"]=len(hints)*3*problem.n
    return dict(result)


def replay_reuse(problem,reference,required,hints):
    """Shuffle real claims and vary the reference; count calls to the full pose enumerator."""
    from random import Random
    from unittest.mock import patch
    from r35_path_mates import PathMates
    sample=list(hints)
    cases=[]
    for seed in (29,71,223,2026):
        Random(seed).shuffle(sample)
        cases.extend([(sample[:],reference|required),(sample[:9],reference|required),
                      (sample[:],(reference-set(map(tuple,sample[0]["path"])))|required)])
    def run(cold):
        checks=0
        previous=PathMates.patterns
        def count(checker,*args,**kwargs):
            nonlocal checks
            checks+=1
            return previous(checker,*args,**kwargs)
        setattr(problem,"_path_circuit_replay",None)
        rows=[]
        with patch.object(PathMates,"patterns",count):
            for candidates,allowed in cases:
                if cold:problem._path_circuit_replay=None
                rows.append(verified_circuits(problem,candidates,allowed))
        return rows,checks
    first,cold=run(True)
    later,warm=run(False)
    if first!=later:raise ArithmeticError("Warm replay changed the proved groups.")
    return dict(queries=len(cases),cold_pose_enumerations=cold,warm_pose_enumerations=warm,
                identical_projected_groups=True)


def report(tiles, screen_file):
    problem, ref, physical, req = problem_scope(tiles)
    claims = compose(problem, ref, req, read_json(screen_file))
    docs = Path(__file__).parent / "docs"
    def groups(name, key):
        return [frozenset(map(tuple, row)) for row in read_json(docs/name)[key]]
    geo = groups("R35_GEOGRAPHIC_RESIDUALS.json", "geographic_cores")
    contacts = groups("R35_EDGE_CONTACTS.json", "edge_perfect_cores")
    base13 = [*physical[:13], *geo, *contacts]
    base344 = [*physical, *geo, *contacts]
    useful = {g:h for g,h in claims.items()
              if not any(old-req <= g-req for old in base344) and
              not any(other-req < g-req for other in claims)}
    phases = dict(raw31=0xce77aac6e89509e8867b457e8c580a233a5e5f22,
                  euler33=0xcc57eae4e99109f8b073557e8c590b9202465766,
                  euler33b=0xcc778d82e99c517987f6b57ec15837c21207cf06,
                  donor37=0xd237aba7e91101e88041413e8c589902021e5126)
    prices = {}
    for name, rows in (("13",base13),("344",base344)):
        prices[name] = {}
        for key, phase in phases.items():
            def price(extra):
                return physical_price(problem.tiles,ref,[*rows,*extra],req,phase)[0]
            prices[name][key] = dict(before=price(()),after=price(claims),published=price(useful))
            if prices[name][key]["after"] != prices[name][key]["published"]:
                raise ArithmeticError("The published subset lost a phase price.")
    return dict(fingerprint=problem.fingerprint,compositions=len(claims),nondominated=len(useful),
                widths=dict(Counter(map(len,useful))),fixed_phase_prices=prices,
                placement_specialization=native_poses(problem,useful.values()),
                repeated_claims=replay_reuse(problem,ref,req,useful.values()),
                gold_path_circuits=[v for _,v in sorted(useful.items(), key=lambda x:(len(x[0]),sorted(x[0])))])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles", required=True, type=Path)
    cli.add_argument("--screen", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    result = report(args.tiles,args.screen)
    atomic_json(args.output,result)
    print({k:v for k,v in result.items() if k!="gold_path_circuits"})
