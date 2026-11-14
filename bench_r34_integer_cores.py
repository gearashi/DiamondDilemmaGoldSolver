"""Measure actual integer core hits, partial-proof replay and deterministic shrinking."""
import argparse
from collections import Counter
from math import inf
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerCoreStore, IntegerPatternFilter
from search_problem import SearchProblem


def stored_core_ablation(query, arguments, arrivals):
    """Try a disjoint five-error prefix whose joint inputs match a stored four-error core."""
    from itertools import combinations
    from math import comb
    with TemporaryDirectory() as folder:
        path = Path(folder)/"partial.sqlite"
        db = sqlite3.connect(path)
        first = IntegerPatternFilter(*arguments,store=IntegerCoreStore(db,"disjoint"))
        for trial in arrivals:
            if len(trial)==4:
                assert not first(trial,inf)
        db.commit()
        db.close()
        # Price all compatible completions of each actual stored conjunction,
        # independently of the suffix index. Stop at the first possible input.
        ranked = {}
        for mask,value in sorted(first.verified):
            if (mask,value)==first.published:
                continue
            pins = [i for i in range(len(query.outside)) if (value>>i)&1]
            free = [i for i in range(len(query.outside)) if not ((mask>>i)&1)]
            if not 0<=5-len(pins)<=len(free):
                continue
            for extra in combinations(free,5-len(pins)):
                choice = sorted(pins+list(extra))
                rhs,cols = query.outside_check
                for i in choice:
                    rhs ^= cols[i]
                if rhs:
                    continue
                charge = 5
                for goal,cols,prices in query.tables:
                    for i in choice:
                        goal ^= cols[i]
                    charge += prices.get(goal,18)
                if charge>17:
                    continue
                n=len(query.outside)-2
                rank=0
                lo=0
                for j,hi in enumerate(choice[:3]):
                    rank += sum(comb(n-i-1,2-j) for i in range(lo,hi))
                    lo=hi+1
                ranked.setdefault(rank,choice)
                if len(ranked)>=8:
                    break
            if len(ranked)>=8:
                break
        if not ranked:
            return {"found_prefix":False}
        db=sqlite3.connect(path)
        arms={}
        for name,store in (("cold",None),("reopened",IntegerCoreStore(db,"disjoint"))):
            check = IntegerPatternFilter(*arguments,store=store)
            totals=Counter()
            outcomes=[]
            for rank in sorted(ranked):
                result=query.indexed(17,5,inf,rank,rank+1,accept_errors=check,reject_before_joint=check.excludes)
                totals.update(result["work"])
                outcomes.append((result["status"],result.get("outside_errors")))
            arms[name]={"outcomes":outcomes,"work":dict(totals),"integer_work":dict(check.work)}
        db.close()
        if arms["cold"]["outcomes"]!=arms["reopened"]["outcomes"]:
            raise AssertionError("reopened core changed a prefix answer")
        return {"found_prefix":True,"selected_prefixes":sorted(ranked),
                "selection":"up to eight projected matches of four-error native cores",
                "five_error_candidates":[ranked[r] for r in sorted(ranked)],"arms":arms}


def compare(tiles, measured, prefix_end, previous=None):
    p = SearchProblem(read_json(tiles))
    ref,_,groups = reference_groups(p,read_json(Path(__file__).parent/"docs/R29_INCUMBENT.json")["codes"])
    q = OutsideSyndromes(p,ref,groups,inventory(p).required)
    arguments = (p,ref,q.groups,q.required,q.outside,17)
    original = next(c for c in measured["cases"] if c["cap"]==17 and c["errors"]==4)
    arrivals = list(original["arrivals"])
    control = IntegerPatternFilter(*arguments)
    published = next(iter(control.pending))
    # The baseline callback accepts no pattern here to collect a full trace.
    def record(indices, deadline):
        arrivals.append(sorted(indices))
        return False
    if previous:
        if (previous["fingerprint"]!=p.fingerprint or previous["loss_budget"]!=17 or
                previous["five_error_prefix_range"]!=[0,prefix_end] or
                previous["arrivals"][:len(arrivals)]!=arrivals):
            raise ValueError("previous native trace scope differs")
        arrivals = list(previous["arrivals"])
    else:
        out = q.indexed(17,5,inf,0,prefix_end,accept_errors=record)
        assert out["status"]=="INFEASIBLE" and out["next_rank"]==prefix_end
    print("Arrivals", len(arrivals), flush=True)
    bits = [sum(1<<i for i in row) for row in arrivals]
    full = (1<<len(q.outside))-1
    report = {}
    def solve_arm(name, q, use_core):
        before = perf_counter()
        if use_core:
            decisions = [q(row,inf) for row in arrivals]
        else:
            decisions = []
            for v in bits:
                s = q._decide(full,v,inf)
                assert s in (q.cp.OPTIMAL, q.cp.FEASIBLE,q.cp.INFEASIBLE)
                decisions.append(s != q.cp.INFEASIBLE)
        report[name] = {"decisions":decisions,"work":dict(q.work),
                        "seconds":round(perf_counter()-before,5),
                        "checked_cores":[[hex(m),hex(v)] for m,v in sorted(q.verified)]}

    solve_arm("fresh",control,False)
    local = IntegerPatternFilter(*arguments)
    local.pending.clear()
    solve_arm("learned",local,True)
    store_scope = "matched-budget-17"
    with TemporaryDirectory() as folder:
        dbpath = Path(folder)/"cores.sqlite"
        db = sqlite3.connect(dbpath)
        scoped = IntegerPatternFilter(*arguments,store=IntegerCoreStore(db,store_scope))
        solve_arm("seeded_and_persisted",scoped,True)
        db.commit()
        db.close()
        db = sqlite3.connect(dbpath)
        reopened = IntegerPatternFilter(*arguments,store=IntegerCoreStore(db,store_scope))
        solve_arm("reopened",reopened,True)
        rows = db.execute("SELECT COUNT(*), SUM(length(mask)+length(value)) FROM r34_integer_cores").fetchone()
        db.close()
        storage = {"bytes_in_separate_sqlite_file":dbpath.stat().st_size,"rows":rows[0],"hex_characters":rows[1]}
    # Shrink sufficient native assumptions deterministically. Only complete
    # infeasibility decisions permit dropping a chunk.
    smaller = IntegerPatternFilter(*arguments)
    smaller.pending.clear()
    base_learn = smaller._learn
    def shrink(mask,value):
        base_learn(mask,value)
        mask,value = min((m,v) for m,v in smaller.verified if (v==(v & m)) and
                         (mask & m)==m and (value & m)==v)
        original = mask,value
        for width in (16,8,4,2,1):
            order = [i for i in range(len(q.outside)) if mask & (1<<i)]
            for j in range(0,len(order),width):
                trial = mask & ~sum(1<<i for i in order[j:j+width])
                trial_status = smaller._decide(trial,value & trial,monotonic()+0.25)
                if trial_status==smaller.cp.UNKNOWN:
                    smaller._count("unknown_shrink_trials")
                if trial_status==smaller.cp.INFEASIBLE:
                    mask,value = trial,value & trial
        smaller.verified.discard(original)
        smaller.verified.add((mask,value))
    smaller._learn = shrink
    if previous:
        report["deterministic_shrink"] = previous["arms"]["deterministic_shrink"]
    else:
        solve_arm("deterministic_shrink",smaller,True)
    trace_db = sqlite3.connect(":memory:")
    checked = IntegerPatternFilter(*arguments,store=IntegerCoreStore(trace_db,"disjoint"))
    begun = False
    positives = []
    def accept(indices,deadline):
        nonlocal begun
        begun = True
        if checked(indices,deadline):
            positives.append(sorted(indices))
        return False
    early = []
    for errors,end in ((4,original["interval"][1]),(5,prefix_end)):
        if errors==5:
            checked = IntegerPatternFilter(*arguments,store=IntegerCoreStore(trace_db,"disjoint"))
        def early_cut(indices,deadline):
            return begun and checked.excludes(indices,deadline)
        result = q.indexed(17,errors,inf,0,end,accept_errors=accept,reject_before_joint=early_cut)
        assert result["status"]=="INFEASIBLE" and result["next_rank"]==end
        early.append({"errors":errors,"work":result["work"],"integer_work":dict(checked.work)})
    trace_db.close()
    if positives != [a for a,d in zip(arrivals,report["fresh"]["decisions"]) if d]:
        raise AssertionError("applying cores before the joint oracle removed a feasible integer pattern")
    if any(r["decisions"]!=report["fresh"]["decisions"] for r in report.values()):
        raise AssertionError("integer filter changed a definitive answer")
    return {"fingerprint":p.fingerprint,"loss_budget":17,"complete_four_error_layer":True,
            "five_error_prefix_range":[0,prefix_end],"unique_arrivals":len(set(bits)), "arrivals":arrivals,
            "published_subset_matches":sum((v & published[0])==published[1] for v in bits),
            "storage":storage,"arms":report,"stored_core_ablation":stored_core_ablation(q,arguments,arrivals),"early_joint_filter":early,"early_integer_work":dict(checked.work)}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    cli.add_argument("--measured",type=Path,default=Path("docs/R34_LOSS_MEASUREMENTS.json"))
    cli.add_argument("--prefix-end",type=int,default=65536)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--previous",type=Path,help="Reuse a checked native arrival trace and deterministic shrink arm at identical scope.")
    args=cli.parse_args()
    report=compare(args.tiles,read_json(args.measured),args.prefix_end,read_json(args.previous) if args.previous else None)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print({k:v["work"] for k,v in report["arms"].items()})
