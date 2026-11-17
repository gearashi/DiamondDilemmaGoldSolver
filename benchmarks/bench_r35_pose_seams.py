"""Compare seam propagation widths and physical tile matching on original geographic domains."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from bench_r35_pose_departures import cases, crossovers
from bench_r35_pose_frontiers import saved_stacks
from bench_r35_pose_image_pairs import fixtures
from bench_r35_pose_matching import cases as owner_cases
from io_utils import atomic_json
from r32_gold_bonds import inventory
from r35_pose_departures import reference_images
from r35_pose_matching import tight_support
from r35_repair_propagation import RepairPropagation
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


def evaluate(engine, domains, codes, width, matching):
    current = domains
    rounds = 0
    started = perf_counter()
    while True:
        rounds += 1
        reduced = engine.narrow((), (), width, True, domains=current)
        if reduced is None:
            reason = "seams"
            break
        if not matching:
            reason = "live"
            break
        filtered = tight_support(reduced, codes)
        if filtered is None:
            reduced = None
            reason = "tile_matching"
            break
        if all(a == b for a,b in zip(filtered, reduced)):
            reason = "live"
            break
        current = filtered
    identity = sha256(repr(tuple(reduced)).encode()).hexdigest() if reduced else None
    return dict(reason=reason, poses=sum(d.bit_count() for d in reduced) if reduced else 0,
                digest=identity, rounds=rounds, seconds=round(perf_counter()-started, 6))


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--tiles", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--checkpoints", type=Path)
    args = cli.parse_args()
    problem = SearchProblem(json.loads(args.tiles.read_bytes()), physical_gold_support=True)
    codes = tuple(active_catalog(problem)["codes"])
    images = reference_images(codes)
    corpus = {"reference": cases(problem), "pool": dict(crossovers(problem)),
              "owners": owner_cases(problem, codes), "pair": dict(fixtures(problem, images))}
    if args.checkpoints:
        corpus["historical"] = dict(saved_stacks(args.checkpoints))
    partners = [[] for _ in range(3*problem.n)]
    for a,b in inventory(problem).possible:
        partners[a].append(b)
        partners[b].append(a)
    engine = RepairPropagation(problem, partners)
    report = {}
    for family, inputs in corpus.items():
        records = {}
        for label, ds in inputs.items():
            records[label] = {f"{width}-{int(matching)}":evaluate(engine, ds, codes, width, matching)
                              for width in (2, 10, 32, 128, 480) for matching in (False,True)}
        report[family] = records
        atomic_json(args.output,report)
        print(family, len(records), flush=True)


if __name__ == "__main__":
    main()
