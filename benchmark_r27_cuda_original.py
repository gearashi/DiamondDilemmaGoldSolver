"""Dispatch comparisons on the original board with bounded, identical work quanta."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import split_cubes
from search_problem import SearchProblem


def run(problem, lanes=128, ticks=12):
    roots=split_cubes([WorkScope.for_problem(problem,"gold").root],lanes)
    rows=[]
    engine=DomainCUDA(problem,roots,target="gold",joint_support=True,event_propagation=True)
    try:
        source=engine.checkpoint()
        for method in (engine.step,engine.step_grouped):
            method(1)
            engine.restore(source)
        for repeat in range(3):
            digests=[]
            for grouped in ((False,True) if repeat%2==0 else (True,False)):
                engine.restore(source)
                method=engine.step_grouped if grouped else engine.step
                intervals=[]
                for _ in range(ticks):
                    start=time.perf_counter()
                    status=method(256)
                    intervals.append(time.perf_counter()-start)
                    if np.any(status==1):break
                state=engine.checkpoint()
                digests.append(state["state_sha256"])
                work=engine.stats()
                rows.append(dict(repeat=repeat,grouped=grouped,lanes=lanes,dispatches=len(intervals),
                                 seconds=round(sum(intervals),6),max_dispatch_ms=round(1000*max(intervals),3),
                                 devices_bytes=engine.state_bytes,decisions=work["decisions"],
                                 support_checks=work["support_checks"],removals=work["domain_removals"],
                                 final_state_sha256=state["state_sha256"]))
            if len(set(digests))!=1:raise AssertionError("Original-board dispatch state diverged")
        return rows
    finally:engine.close()


if __name__=="__main__":
    data=json.loads(Path("data/tiles.json").read_text(encoding="utf-8-sig"))
    print(json.dumps(run(SearchProblem(data)),indent=2))

