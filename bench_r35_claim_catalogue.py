"""Count claim replays in a fresh interpreter for each public solver query."""
import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from functools import wraps
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch


def measure(label, policy, tiles):
    import r32_gold
    from r28_geo_sewing import PosePacking
    from r35_geographic_replay import GeographicReplay
    from r35_induced_contacts import InducedContacts
    from r35_path_mates import PathMates

    options = {
        "integer33": ["integer-bound","--upper","33","--all-groups","--sphere-euler","--mate-support"],
        "weighted30": ["integer-bound","--upper","30","--all-groups"],
        "ranked31": ["loss","--upper","31","--all-groups","--refine-integer","--sphere-euler"],
        "embed": ["embed"],
        "embed-narrow": ["embed"],
        "sewing": ["phase-sewing","--upper","38","--phase","0xce77aac6e89509e8867b457e8c580a233a5e5f22"],
    }
    args = [*options[label],"--tiles",str(tiles),"--seconds","60"]
    docs = Path(__file__).resolve().parent / "docs"
    counts = Counter()
    with ExitStack() as stack:
        if label.startswith("embed"):
            import r30_gold
            run = r30_gold.main
            directory = Path(stack.enter_context(TemporaryDirectory()))
            prior = {}
            for file,key in (("R35_GEOGRAPHIC_RESIDUALS","geographic_cores"),
                             ("R35_EDGE_CONTACTS","edge_perfect_cores"),("R35_PATH_CIRCUITS","gold_path_circuits")):
                prior[key] = json.loads((docs/(file+".json")).read_bytes())[key]
            givens = {"bonds":prior["geographic_cores"][0]}
            if label=="embed-narrow":
                # Fix tile 0, turn 0 in the first cell and exclude it elsewhere.
                givens["domains"] = [[0],*[list(range(3,480)) for _ in range(159)]]
            (directory/"prior.json").write_text(json.dumps(prior))
            (directory/"input.json").write_text(json.dumps(givens))
            args.extend(["--reuse",str(directory/"prior.json"),"--input",str(directory/"input.json")])
            if policy=="independent":
                args.extend(["--phase-anchors","off"])
        else:
            run = r32_gold.main
            for name in ("R35_GEOGRAPHIC_RESIDUALS","R35_EDGE_CONTACTS"):
                args += ["--claims",str(docs/(name+".json"))]
            if label == "sewing":
                args += ["--claims",str(docs/"R35_PATH_CIRCUITS.json")]
            if policy == "independent":
                args.append("--independent-native")
        def wrap(owner, name, counter):
            raw = getattr(owner,name)
            @wraps(raw)
            def wrapped(*args,**kwargs):
                counts[counter] += 1
                return raw(*args,**kwargs)
            stack.enter_context(patch.object(owner,name,wrapped))
        wrap(PosePacking,"check","packing_calls")
        wrap(GeographicReplay,"check","geographic_batches")
        wrap(InducedContacts,"count","contact_pose_enumerations")
        wrap(PathMates,"patterns","path_pose_enumerations")
        if policy == "previous":
            stack.enter_context(patch("r35_claim_catalogue.published_scope",return_value=False))
        stack.enter_context(patch.object(sys,"argv",["r32_gold.py",*args]))
        result = io.StringIO()
        start = perf_counter()
        with redirect_stdout(result):
            run()
        elapsed = perf_counter()-start
    # Ranked progress may emit multiple JSON objects before the final pretty-printed one.
    text = result.getvalue()
    offset = text.rfind("\n{\n")
    row = json.loads(text[offset+1:])
    reused = row.pop("published_claims_reused",{})
    return dict(label=label,policy=policy,separate_process=True,seconds=elapsed,reused=reused,
                work=dict(counts),result=row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--child",nargs=2)
    parser.add_argument("--cases",nargs="+",default=["integer33","ranked31","weighted30","sewing","embed","embed-narrow"])
    args = parser.parse_args()
    if args.child:
        print(json.dumps(measure(*args.child,args.tiles)))
    else:
        rows = []
        for label in args.cases:
            policies = ("previous","published","independent") if label in ("sewing","embed-narrow") else ("previous","published")
            for policy in policies:
                cmd=[sys.executable,"-B",__file__,"--tiles",str(args.tiles),"--child",label,policy]
                out=subprocess.run(cmd,capture_output=True,text=True,check=True)
                row=json.loads(out.stdout)
                rows.append(row)
                if args.output:
                    args.output.write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
                print(label,policy,row["result"]["status"],row["work"],round(row["seconds"],3),flush=True)
            controls=[r["result"] for r in rows if r["label"]==label]
            if any(row != controls[0] for row in controls[1:]):
                raise ValueError("Fresh policies changed the solver decision or charged scope.")
