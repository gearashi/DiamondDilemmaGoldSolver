"""Compare the two public 207 boards and the geographic physical-loss cutoff."""
from argparse import ArgumentParser
from contextlib import nullcontext
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from constraint_symmetry import proper_rotations
from io_utils import atomic_json
from r31_geographic import physical_bonds
from r32_gold import maximize_disjoint
from r32_gold_loss import geographic, reference_groups
from r35_geographic_loss import certified_reference, overlap_lower_bound
from search_problem import SearchProblem
from validator import validate_arrangement


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--tiles", type=Path, required=True)
    cli.add_argument("--seconds", type=float, default=10)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    docs = Path(__file__).resolve().parent / "docs"
    raw = json.loads(args.tiles.read_bytes())
    problem = SearchProblem(raw, physical_gold_support=True)
    source = json.loads((docs / "R29_INCUMBENT.json").read_bytes())["codes"]
    alternate = json.loads((docs / "R29_GOLD_RESULTS.json").read_bytes())["residual_reference"]
    assert alternate == json.loads((docs / "R35_JOINT_REPAIR.json").read_bytes())["installed_center"]["codes"]
    bonds,limit = certified_reference(problem)
    assert bonds == frozenset(physical_bonds(problem,source))
    cases=[]
    report=dict(source="R29_INCUMBENT.json",alternate="R29_GOLD_RESULTS.json/residual_reference",cases=cases)
    for name,codes in (("proof reference",source),("published alternate",alternate)):
        target,cert,groups=reference_groups(problem,codes)
        partition=maximize_disjoint(groups)
        score=validate_arrangement(raw,codes)
        ds=[sum(a!=b for a,b in zip(codes,rotation.apply(source))) for rotation in proper_rotations(problem.board)]
        row=dict(name=name,edges=score["matched_edges"],components=score["line_components"],
                 differences_by_rotation=ds,shared_bonds=len(target&bonds),
                 loss_lower=overlap_lower_bound(bonds,limit,target),queries=[])
        cases.append(row)
        for cap,branch in ((row["loss_lower"]-1,[]),(row["loss_lower"],[]),(row["loss_lower"]-1,[0])):
            for policy in ("prior","overlap"):
                treatment=patch("r35_geographic_loss.geographic_loss_cutoff",return_value=None) if policy=="prior" else nullcontext()
                # Each comparison starts with an empty work database and the same domains.
                with treatment:
                    start=perf_counter()
                    result=geographic(problem,target,partition,cert.rows,branch,0,cap,None,args.seconds,None)
                    measured={k:result[k] for k in ("status","cp_calls","branches","certified_reference_overlap","channel") if k in result}
                row["queries"].append(dict(cap=cap,path=branch,policy=policy,seconds=round(perf_counter()-start,4),**measured))
                atomic_json(args.output,report)
                print(name,cap,branch,policy,measured.get("status"),measured.get("cp_calls"),measured.get("branches"),flush=True)


if __name__=="__main__":
    main()
