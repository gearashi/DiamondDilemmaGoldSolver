"""Check CUDA dispatch equivalence and shared-prefix donation on finite search trees."""
import json
import time
from collections import Counter
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import split_cubes
from gpu_lane_pool_round25 import LanePool
from constraint_work import WorkScope
from test_gpu_domain_round25 import precise_reason_fixture, fork_gold_fixture, independent_solutions
import numpy as np


def batching(lanes, cooperative, quantum):
    prob=precise_reason_fixture()
    roots=split_cubes([WorkScope.for_problem(prob,"gold").root],lanes)
    policy=dict(joint_support=True,event_propagation=True,precise_reasons=True,
                watched_clauses=True,global_matching=True,cooperative=cooperative)
    left,right=(DomainCUDA(prob,roots,target="gold",**policy) for _ in range(2))
    timings=[[],[]]
    methods=(left.step,right.step_grouped)
    try:
        # Compile and warm up both paths outside the paired measurements.
        for engine,method in zip((left,right),methods):
            saved=engine.checkpoint()
            method(1)
            engine.restore(saved)
        for tick in range(20000):
            for idx in ((0,1) if tick%2==0 else (1,0)):
                begin=time.perf_counter()
                status=methods[idx](quantum)
                timings[idx].append(time.perf_counter()-begin)
            np.testing.assert_array_equal(left.state.get(),right.state.get())
            pending=left.candidates()
            if pending:
                ids=[item["lane"] for item in pending]
                left.acknowledge(ids)
                right.acknowledge(ids)
            if tick==3:
                right.restore(right.checkpoint())
            if all(status==2):
                break
        else:
            raise AssertionError("Finite fixture exceeded dispatch bound")
        return dict(lanes=lanes,cooperative=cooperative,quantum=quantum,steps=tick+1,
                    decisions=left.stats()["decisions"],solutions=len(left.delivered),
                    seconds=[round(sum(x),6) for x in timings],
                    p99_ms=[round(float(np.percentile(x,99))*1000,3) for x in timings])
    finally:
        left.close()
        right.close()


def donation(share):
    prob=fork_gold_fixture()
    # One immediately rejected root and a broad root holding both valid tours.
    prob.domains[0]=[0,1]
    scope=WorkScope.for_problem(prob,"gold")
    roots=[]
    for mask in (1,2):
        row=list(scope.root)
        row[0]=mask
        roots.append(tuple(row))
    pool=LanePool(scope,roots)
    eng=DomainCUDA(prob,pool.take_batch(2),event_propagation=True,support_reuse=True)
    try:
        for _ in range(10000):
            statuses=eng.step(1)
            pool.retire(eng,statuses)
            depth=eng.state[1].get()
            if pool.lanes[1] is None and depth[0]:
                break
        else:
            raise AssertionError("No fork opportunity on planted ring")
        old=pool.donate(eng,share_prefix=share)
        if old is None:
            raise AssertionError("No donation")
        eng.restore(eng.checkpoint())
        LanePool.restore(scope,pool.queue,eng,pool.snapshot())
        assigned=eng.state[9].get().tolist()
        unions=eng.state[10].get().tolist()
        if pool.queue:
            raise AssertionError("Unexpected third root")
        for i in range(5000):
            status=eng.step(64)
            found=eng.candidates()
            if found:eng.acknowledge([v["lane"] for v in found])
            if all(status==2):break
        else:raise AssertionError("Incomplete donation coverage")
        answers=[tuple(item["codes"]) for item in eng.delivered]
        if Counter(answers)!=Counter({a:1 for a in independent_solutions(prob,"gold")}):
            raise AssertionError("Donation lost or duplicated solutions")
        work=eng.stats()
        return dict(shared=share,assigned=assigned,unions=unions,solutions=len(answers),bytes=eng.state_bytes,
                    strand_joins=work["strand_joins"],checks=work["support_checks"],decisions=work["decisions"])
    finally:eng.close()


if __name__=="__main__":
    print(json.dumps({"donation":[donation(i) for i in (False,True)],
                      "dispatch":[batching(n,coop,q) for n in (1,8,128) for coop in (False,True) for q in (64,256)]},indent=2))

