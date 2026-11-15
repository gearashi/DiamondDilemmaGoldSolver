"""Compare reference pose exclusions on the published geographic Gold search."""
from argparse import ArgumentParser
from contextlib import nullcontext
import json
from pathlib import Path
from random import Random
from time import perf_counter
from unittest.mock import patch

from constraint_symmetry import proper_rotations
from io_utils import atomic_json
from r27_structure_filters import domain_bits
from r30_gold import search
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


def cases(problem):
    record = active_catalog(problem)
    seed = record["codes"]
    rows = record["rows"]
    region = set(range(problem.n)) - set(record["hit"])
    turns = proper_rotations(problem.board)
    random = Random(8935)
    result = {}
    for label,anchors,rotation in (
        ("direct-pair",next(x for x in rows if len(x)==2),0),
        ("rotated-triple",next(x for x in reversed(rows) if len(x)==3),4),
        ("rotated-regional",rows[-1],7),
        ("near-distance",sorted(region) + [min(record["hit"])],0),
        ("distance-holdout",random.sample(range(problem.n),66),6),
        ("single-anchor",[rows[0][0]],0),
    ):
        turn = turns[rotation]
        image = turn.apply(seed)
        ds = list(domain_bits(problem))
        for a in anchors:
            c = turn.cells[a]
            ds[c] = 1 << image[c]
        result[label] = ds
    for index,first in enumerate((4,32,49,71,103,139)):
        rotation=turns[index+1]
        cell=rotation.cells[first]
        image=rotation.apply(seed)
        ds=list(domain_bits(problem))
        ds[cell]=1 << image[cell]
        result["anchor-"+str(first)]=ds
    fixture=json.loads((Path(__file__).parent/'docs/R35_REPAIR_PROPOSALS.json').read_bytes())
    hits=[set(record['hit'])]
    for section in ('single_exchanges','two_exchanges'):
        for sample in fixture[section]['sample']:
            mask=int(sample['hit'],16)
            hits.append({c for c in range(problem.n) if mask >> c & 1})
    for k,hit in enumerate(hits):
        ds=list(domain_bits(problem))
        turn=turns[(3*k)%len(turns)]
        rotated=turn.apply(seed)
        for a in set(range(problem.n))-hit:
            c=turn.cells[a]
            ds[c]=1 << rotated[c]
        result["hit-"+str(k)]=ds
    return result


def crossovers(problem):
    from r35_pose_departures import reference_images
    record = active_catalog(problem)
    images = tuple(dict(image) for image in reference_images(tuple(record["codes"])))
    for size in (2, 3, 4, 5, 7, 10):
        for count in (40, 80, 120, 160):
            random = Random(89100 + 10 * size + count)
            selected = random.sample(range(len(images)), size)
            domains = list(domain_bits(problem))
            for cell in random.sample(range(problem.n), count):
                domains[cell] &= sum(images[i][cell] for i in selected)
            yield f"pool-{size}-{count}", domains


def audit_reuse(problem, queries):
    """The cache can skip repeated proof-file decoding. Each query still authenticates the mesh."""
    from r35_claim_catalogue import published_scope
    from r35_pose_departures import original_pose_departures
    from r35_repair_seeds import active_catalog
    records = []
    for cached in (False, True):
        problem.__dict__.pop("_geographic_pose_profile", None)
        outcomes = {}
        with patch("r35_repair_seeds.active_catalog", wraps=active_catalog) as source, \
                patch("r35_claim_catalogue.published_scope", wraps=published_scope) as scope:
            for name, domains in queries.items():
                if not cached:
                    problem.__dict__.pop("_geographic_pose_profile", None)
                reduced, result = original_pose_departures(problem, domains)
                outcomes[name] = [result["over"], result["removed"], tuple(reduced)]
            records.append(dict(cached=cached, catalog_reads=source.call_count,
                                mesh_checks=scope.call_count, outcomes=outcomes))
    if records[0]["outcomes"] != records[1]["outcomes"]:
        raise AssertionError("Proof reuse changed a geographic result")
    for row in records:
        row["distinct_queries"] = len(row.pop("outcomes"))
    return records


def public_cli(tiles, domains, seconds, rounds):
    import io
    import sys
    from contextlib import redirect_stdout
    from tempfile import TemporaryDirectory

    from r30_gold import main
    from r35_pose_departures import cell_indices
    with TemporaryDirectory() as temporary:
        source, result = (Path(temporary) / name for name in ("input.json", "result.json"))
        source.write_text(json.dumps({"domains":[list(cell_indices(d)) for d in domains]}), encoding="utf-8")
        args = ["r30_gold.py", "embed", "--reference-bound", "--tiles", str(tiles),
                "--input", str(source), "--output", str(result), "--seconds", str(seconds),
                "--rounds", str(rounds)]
        with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
            main()
        return json.loads(result.read_bytes())


def run(tiles,output,seconds,labels,rounds,compare_guards=False,compare_seams=False,reuse=False,cli=False):
    problem=SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
    queries=cases(problem)
    if reuse:
        atomic_json(output,audit_reuse(problem,queries))
        return
    if compare_seams:
        queries = dict(crossovers(problem))
    data=[]
    for label in labels or list(queries)[:6]:
        policies = ("without-crossover-seams","poses") if compare_seams else (
            ("master-only","poses") if compare_guards else ("prior","poses"))
        for policy in policies:
            if policy=="prior":
                scope=patch("r35_pose_departures.original_pose_departures",side_effect=lambda p,ds:(ds,None))
            elif policy=="without-crossover-seams":
                scope=patch("r35_pose_departures.crossover_seams",
                            side_effect=lambda problem, domains, images:(domains,{"over":False,"removed":0}))
            elif policy=="master-only":
                scope=patch("r35_repair_seeds.known_hint_conflicts",return_value=())
            else:
                scope=nullcontext()
            with scope:
                start=perf_counter()
                if cli:
                    out=public_cli(tiles,queries[label],seconds,rounds)
                else:
                    out=search(problem,geography=True,domains=queries[label],reference_bound=True,seconds=seconds,max_rounds=rounds)
                record={k:out[k] for k in ("status","cp_calls","branches","partner_variables","channel","certified_pose_departures","certified_reference") if k in out}
                data.append(dict(case=label,policy=policy,seconds=perf_counter()-start,result=record))
                atomic_json(output,data)
                print(label,policy,record["status"],record["cp_calls"],record["branches"],flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--tiles",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=8)
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--rounds",type=int,default=32)
    parser.add_argument("--compare-guards",action="store_true")
    parser.add_argument("--compare-seams",action="store_true")
    parser.add_argument("--reuse",action="store_true")
    parser.add_argument("--cli",action="store_true",help="Use normal embed CLI defaults, including the geographic channel.")
    opts=parser.parse_args()
    run(opts.tiles,opts.output,opts.seconds,opts.cases,opts.rounds,opts.compare_guards,opts.compare_seams,opts.reuse,opts.cli)
