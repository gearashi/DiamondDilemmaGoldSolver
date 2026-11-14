"""Geometry and regional Gold checks for a real or restricted board.

Run python r28_geometry.py --help for input formats and supported queries.
"""
import argparse
import json
from pathlib import Path

from geometry import build_board
from r27_structure_filters import domain_bits
from r28_geo_color import dual_cuts, phase_probe
from r28_geo_regions import count_partition
from r28_geo_sewing import PosePacking, boundary_lower, ring_census, side_options, sign_filter, supply_supports, switch_family
from search_problem import SearchProblem


def run(problem, dataset, mode, max_work, seconds, method):
    ds = domain_bits(problem)
    if "domains" in dataset:
        rows = dataset["domains"]
        if (len(rows) != problem.n or any(not isinstance(row,list) or
                any(type(x) is not int or x < 0 or x >= 3*problem.n for x in row) for row in rows)):
            raise ValueError("Expected one array of placement codes per cell.")
        ds = tuple(sum(1<<x for x in set(row)) for row in rows)
    bonds = dataset.get("bonds",())
    if mode == "core" and "domains" in dataset:
        from r28_geo_color import prune_signed_core
        from r27_structure_filters import bits
        kept,record = prune_signed_core(problem,ds,bonds,seconds=seconds)
        return dict(record,domains=[list(bits(row)) for row in kept])
    if mode == "core":
        from r28_geo_color import phase_bond_core
        return phase_bond_core(problem,bonds,euler=dataset.get("euler",True),seconds=seconds)
    if mode == "phase":
        pins = dataset.get("pins")
        if pins is not None:pins = {int(k):v for k,v in pins.items()}
        return phase_probe(problem,euler=dataset.get("euler",True),pins=pins,bonds=bonds,
                           probe_tiles=dataset.get("probe_tiles",()),potential_bonds=dataset.get("probe_bonds",()),seconds=seconds)
    if mode == "dual":
        return dual_cuts(problem,ds,max_boards=max_work,deterministic_seconds=seconds)
    if mode == "stars":
        options = side_options(problem,ds)
        return ring_census(problem,options,max_work=max_work,seconds=seconds)
    if mode == "poses":
        return PosePacking(problem,ds).check(bonds,max_steps=max_work)
    if mode == "supply":
        subset = dataset["subset"]
        k = min(len(set(subset)),problem.n-len(set(subset)))
        lb,sizes = boundary_lower(problem,k,max_shapes=max_work)
        if k >= len(lb):return dict(feasible=None,completed_sizes=sizes)
        return dict(supply_supports(problem,subset,lb[k],bonds=bonds,options=side_options(problem,ds)),lower=lb[k])
    if mode == "sign":return sign_filter(problem,dataset["phases"],bonds)
    if mode == "switches" and "codes" in dataset:
        from r28_geo_sewing import switch_boards
        return switch_boards(problem,dataset["codes"],dataset["patches"],max_nodes=max_work)
    if mode == "switches":return switch_family(dataset["tour"],dataset["switches"],dataset.get("mandatory",()),max_nodes=max_work)
    parts = dataset.get("regions")
    if parts is None:parts = [[c.index for c in problem.board.cells if (c.face < 5) == top] for top in (True,False)]
    return count_partition(problem,parts,ds,mode=method,max_work=max_work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["phase","core","dual","stars","poses","supply","sign","switches","regions"])
    p.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    p.add_argument("--input",type=Path,help="JSON with domains (allowed orientation codes by cell) and mode-specific fields.")
    p.add_argument("--subdivision",type=int,default=4)
    p.add_argument("--max-work",type=int,default=200000)
    p.add_argument("--seconds",type=float,default=3,help="CP-SAT per-query time; deterministic units for dual.")
    p.add_argument("--method",choices=["trie","plain","basis","response","orbit","paged"],default="trie")
    p.add_argument("--checkpoint",type=Path,help="Durable cursor for paged regional joins.")
    p.add_argument("--table-work",type=int,default=1000000,help="Regional table compilation work per page.")
    args = p.parse_args()
    problem = SearchProblem(json.loads(args.tiles.read_text(encoding="utf-8-sig")),build_board(args.subdivision))
    dataset = json.loads(args.input.read_text(encoding="utf-8-sig")) if args.input else {}
    if args.mode == "regions" and args.method == "paged":
        from io_utils import atomic_json,read_json_shared
        from r28_geo_regions import paged_trie
        if args.checkpoint is None:p.error("Paged regional joins require --checkpoint PATH.")
        parts = dataset.get("regions")
        if parts is None:parts = [[c.index for c in problem.board.cells if (c.face<5)==top] for top in (True,False)]
        ds = (None if "domains" not in dataset else
              tuple(sum(1<<v for v in set(row)) for row in dataset["domains"]))
        cursor = read_json_shared(args.checkpoint) if args.checkpoint.exists() else None
        record = paged_trie(problem,parts,ds,cursor=cursor,max_steps=args.max_work,table_work=args.table_work)
        if record.get("cursor"):atomic_json(args.checkpoint,record["cursor"])
        print(json.dumps({k:v for k,v in record.items() if k!="cursor"},indent=2))
    else:print(json.dumps(run(problem,dataset,args.mode,args.max_work,args.seconds,args.method),indent=2))


if __name__ == "__main__":
    main()
