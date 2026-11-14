"""Replay a released cap-16 checkpoint read-only at a tighter bound."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from r32_gold import run_syndrome
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles", type=Path, default=Path("data/tiles.json"))
    cli.add_argument("--database",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    ref,_,groups=reference_groups(p,read_json(Path(__file__).parent/"docs/R29_INCUMBENT.json")["codes"])
    results={}
    uri=args.database.resolve().as_uri()+"?mode=ro"
    for key,path,integer in (("cold",":memory:",False),("inherited",uri,False),("inherited_integer",uri,True)):
        t=perf_counter()
        if key=="cold":
            result=run_syndrome(p,ref,groups,15,None,path,0,page=8192)
        else:
            with patch.object(OutsideSyndromes,"__init__",side_effect=AssertionError("unexpected compilation")):
                result=run_syndrome(p,ref,groups,15,None,path,0,page=8192,refine_integer=integer)
        results[key]={"seconds":round(perf_counter()-t,6),"result":result}
        print(key,result["status"],result["completed_prefixes"],result["work"],flush=True)
    assert len({(v["result"]["status"],v["result"]["completed_prefixes"]) for v in results.values()})==1
    args.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
