"""Compare released and candidate loss work on identical original-puzzle prefix intervals."""
import argparse
from collections import Counter
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc
from unittest.mock import patch

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
import r32_gold_syndrome as current
from search_problem import SearchProblem


def module_from_file(path):
    spec = importlib.util.spec_from_file_location("r34_loss_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def work_breakdown(results, arguments, baseline):
    q = current.OutsideSyndromes(*arguments)
    old = baseline.OutsideSyndromes(*arguments)
    rows, checks = current.eliminate_phases(q.base,q.out_eq)
    examples = {tuple(a) for r in results for a in r["arrivals"]}
    examples.update([(),(0,), (1,2,3,4), (4,12,24,54)])
    counts = Counter()
    statuses = []
    def counted(basis, row):
        counts["inserts"] += 1
        while row>1:
            pivot=row.bit_length()-1
            old_row=basis.get(pivot)
            if old_row is None:
                basis[pivot]=row
                return True
            counts["pivot_xors"] += 1
            row ^= old_row
        return row==0
    with patch.object(baseline, "insert", counted), patch.object(old,"out_eq", []):
        for indices in sorted(examples):
            bits = 1 | sum(1<<(i+1) for i in indices)
            if any((check & bits).bit_count() & 1 for check in checks):
                statuses.append(False)
                continue
            base = {pivot:(row,(symbol & bits).bit_count() & 1) for pivot,(row,symbol) in rows.items()}
            with patch.object(old, "base", base):
                statuses.append(old.joint(set(),17-len(indices),float("inf")))
    projections = []
    for result in results:
        cap,errors = result["cap"],result["errors"]
        start,stop = result["interval"]
        query = q._build_suffix_table(cap,errors)
        k = max(0,errors-2)
        old_xors = (2 + len(q.tables) - int(query.anchor is not None))*k*(stop-start)
        packed_xors = 0
        previous = []
        for rank, prefix in current.ranked_combinations(len(q.outside)-min(errors,2),k,start,stop):
            same = 0
            while same<len(previous) and previous[same]==prefix[same]:
                same += 1
            packed_xors += k-same
            previous = list(prefix)
        projections.append({"cap":cap,"errors":errors,"interval":[start,stop],
                            "released_prefix_xors":old_xors,"packed_prefix_xors":packed_xors})
    return {"compiled_outside_only":{"counts":dict(counts),"decisions":statuses},
            "prefix_projection_work":projections}


def compare(tiles, released):
    problem = SearchProblem(read_json(tiles))
    codes = read_json(Path(__file__).parent/"docs/R29_INCUMBENT.json")["codes"]
    reference, _, groups = reference_groups(problem, codes)
    arguments = (problem, reference, groups, inventory(problem).required)
    baseline = module_from_file(released)
    modes = {"released": (baseline, {}), "packed": (current, {"use_second_group": False}),
             "split": (current, {})}
    cases = ((14,4,0,6903), (15,5,60000,68192), (16,6,7044800,7052992),
             (17,4,0,6903), (17,5,33361,34385), (18,5,33361,33617))
    results = []
    for case in cases:
        cap, errors, start, stop = case
        row = {}
        arrivals = None
        for name, (module, options) in modes.items():
            q = module.OutsideSyndromes(*arguments, **options)
            def run(candidate, deadline):
                observed.append(sorted(candidate))
                return False
            observed = []
            t = time.perf_counter()
            result = q.indexed(cap, errors, float("inf"), start, stop, accept_errors=run)
            seconds = time.perf_counter()-t
            if result["next_rank"] != stop or result["status"] != "INFEASIBLE":
                raise AssertionError("experiment requires complete traversal")
            if arrivals is None:
                arrivals = observed
            elif arrivals != observed:
                raise AssertionError("candidate arrival order changed")
            row[name] = {"seconds":round(seconds,5), "work":result["work"]}
        results.append({"cap":cap,"errors":errors,"interval":[start, stop],"arrivals":arrivals,"arms":row})
        print(case, [(n,v["work"].get("indexed_tails",0),v["work"].get("joint_queries",0),v["seconds"])
                     for n,v in row.items()],len(arrivals),flush=True)
    memory = {}
    for name, (module, options) in modes.items():
        gc.collect()
        tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()
        q = module.OutsideSyndromes(*arguments, **options)
        q.indexed(15, 5, float("inf"), 266913, 266914)
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[name] = {"bytes_retained":retained-before,"bytes_peak":peak-before}
        del q
    # Count actual insertion calls and phase-pivot XORs in the same joint oracle.
    # Include a true joint hit and a group rejection after outside consistency.
    joint = {}
    examples = {tuple(a) for r in results for a in r["arrivals"]}
    examples.update([(),(0,), (1,2,3,4), (4,12,24,54)])
    for name, module in (("released",baseline), ("compiled",current)):
        q = module.OutsideSyndromes(*arguments)
        counts = Counter()
        def counted(basis, row):
            counts["inserts"] += 1
            while row>1:
                pivot=row.bit_length()-1
                old=basis.get(pivot)
                if old is None:
                    basis[pivot]=row
                    return True
                counts["pivot_xors"] += 1
                row ^= old
            return row==0
        with patch.object(module, "insert", counted):
            statuses = [q.joint(set(a), 17, float("inf")) for a in sorted(examples)]
        joint[name] = {"counts":dict(counts),"decisions":statuses}
    if joint["released"]["decisions"] != joint["compiled"]["decisions"]:
        raise AssertionError("joint decisions disagree")
    breakdown = work_breakdown(results,arguments,baseline)
    joint["outside_only"] = breakdown.pop("compiled_outside_only")
    if joint["outside_only"]["decisions"] != joint["compiled"]["decisions"]:
        raise AssertionError("isolated compilation changed a decision")
    return {"fingerprint":problem.fingerprint, "cases":results, "index_memory":memory,"joint":joint, **breakdown}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles", type=Path, default=Path("data/tiles.json"))
    cli.add_argument("--released", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    evidence = compare(args.tiles,args.released)
    args.output.write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf8")
