"""Measure exact in-prefix recovery and simultaneous disjoint original-board support pages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from r30_multicenter import run
from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerPatternFilter
from r34_rank_leases import RankedLeases
from search_problem import SearchProblem


def interrupted(problem):
    codes=read_json(Path(__file__).parent/"docs/R29_INCUMBENT.json")["codes"]
    reference,_,groups=reference_groups(problem,codes)
    args=(problem,reference,groups,inventory(problem).required)
    results=[]
    for rank,errors,stage in ((874,4,"integer"),(1301,4,"integer"),(6166,4,"integer"),(7797,5,"joint")):
        query=OutsideSyndromes(*args)
        pattern_args=(problem,reference,query.groups,query.required,query.outside,17)
        calls=[]
        if stage=="integer":
            def interrupt(indices,deadline):
                calls.append(sorted(indices))
                raise TimeoutError
            partial=query.indexed(17,errors,float("inf"),rank,rank+1,accept_errors=interrupt)
        else:
            native=query.joint
            def interrupt(indices,cap,deadline):
                calls.append(sorted(indices))
                if len(calls)==30:
                    raise TimeoutError
                return native(indices,cap,deadline)
            with patch.object(query,"joint",interrupt):
                partial=query.indexed(17,errors,float("inf"),rank,rank+1,accept_errors=IntegerPatternFilter(*pattern_args))
        assert partial["status"]=="UNKNOWN" and partial["cursor"][0][-1]==stage
        arms={}
        for arm,cursor in (("cold",None),("resume",partial["cursor"])):
            q=OutsideSyndromes(*args)
            out=q.indexed(17,errors,float("inf"),rank,rank+1,
                          cursor=cursor,accept_errors=IntegerPatternFilter(*pattern_args))
            arms[arm]={"status":out["status"],"outside_errors":out.get("outside_errors"),
                       "next_rank":out["next_rank"],"work":out["work"]}
        assert (arms["cold"]["status"],arms["cold"]["outside_errors"],arms["cold"]["next_rank"])==(
                arms["resume"]["status"],arms["resume"]["outside_errors"],arms["resume"]["next_rank"])
        results.append({"cap":17,"errors":errors,"rank":rank,"stage":stage,
                        "interrupted_work":partial["work"],"cursor_bytes":len(json.dumps(partial["cursor"])),
                        "arms":arms})
    return results


def disjoint(problem):
    boards=[[3*t for t in c["pieces"]] for c in read_json(Path(__file__).parent/"docs/R29_EXCHANGE_CENTERS.json")["centers"]]
    args={"families":((2,2,3),),"page":256,"start":0,"stop":1024,"seconds":0,
          "target":208,"backend":"cpu"}
    with TemporaryDirectory() as folder:
        root=Path(folder)
        cold=run(problem,boards,root/"cold.db",**args)
        path=root/"shared.db"
        ready,release=Event(),Event()
        actual=RankedLeases.computing
        taken=[]
        @contextmanager
        def gated(holder,claim):
            with actual(holder,claim):
                taken.append((claim.lo,claim.hi))
                if claim.lo==0:
                    ready.set()
                    if not release.wait(120):
                        raise TimeoutError("comparison worker did not finish")
                yield
        with patch.object(RankedLeases,"computing",gated),ThreadPoolExecutor(max_workers=1) as pool:
            first=pool.submit(run,problem,boards,path,**args)
            try:
                if not ready.wait(120):
                    raise TimeoutError("first worker did not enter its page")
                db=sqlite3.connect(path,timeout=1)
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
                db.close()
                next_worker=run(problem,boards,path,max_pages=3,**args)
            finally:
                release.set()
            initial=first.result()
        rebuilt=run(problem,boards,path,**args)
    expected=sorted(taken)
    assert [(0,256),(256,512),(512,768),(768,1024)]==expected
    assert next_worker["range_finished"] is False and rebuilt["range_finished"] is True
    assert rebuilt["best"]==cold["best"] and rebuilt["covered_supports"]==cold["covered_supports"]
    assert sum(r["work"].get("raw_center_endpoints",0) for r in (initial,next_worker))==cold["work"]["raw_center_endpoints"]
    return {"completed_ranges":expected,"independent_write_transaction_succeeded":True,
            "cold":cold,"first":initial,"second":next_worker,"reopen":rebuilt}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",default="data/tiles.json",type=Path)
    cli.add_argument("--output",required=True,type=Path)
    args=cli.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    data={"fingerprint":problem.fingerprint,"loss":interrupted(problem),"multicenter":disjoint(problem)}
    args.output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
