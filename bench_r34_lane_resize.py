"""Measure native lane resize on real CUDA, with exact finite enumeration and matched fixed points."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from constraint_work import WorkScope,digest,frontier_snapshot
from gpu_checkpoint_codec_round25 import encode_native
from gpu_checkpoint_round25 import CheckpointStore,resolve_checkpoint
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import COUNTERS,pack_native,solve,split_cubes,unpack_native
from gpu_lane_pool_round25 import LanePool
from r32_gold_loss import read_json
from search_problem import SearchProblem
from test_gpu_domain_round25 import fork_gold_fixture,joint_hall_fixture,precise_reason_fixture,independent_solutions


def checkpoint(e,*,balanced=None,totals=None,completed=0,owner=None):
    scope=WorkScope.for_problem(e.problem,e.target)
    queued=balanced.queue if balanced else []
    payload={"version":1,"scope":scope.payload(),"native":pack_native(e),"completed_lanes":completed,
             "totals":dict.fromkeys(COUNTERS,0) if totals is None else totals,"active_stats":e.stats(),
             "queue":[[hex(v) for v in row] for row in queued],
             "frontier":frontier_snapshot(scope,[*queued,*e.remaining_domain_cubes()])}
    if balanced:
        payload.update(lane_balance=True,lane_pool=balanced.snapshot(),balance_stats=balanced.public_stats())
    if owner is not None:
        payload.update(propagated_forks=True,preparing=False,prepared_owner=[hex(v) for v in owner])
    payload["digest"]=digest(payload)
    return payload


def resize(problem,target,payload,count,options=None,folder=None):
    calls=iter((False,True))
    result=solve(problem,target=target,replicas=count,resume=payload,options=options or {},
                 learning_state={"_gpu_checkpoint_dir":folder} if folder else None,
                 stop=lambda:next(calls,True))
    saved=result["checkpoint"]
    if "gpu_checkpoint_reference" in saved:
        saved=resolve_checkpoint(saved,folder)
    return saved,result["stats"]


def open_native(p,target,native):
    roots=[tuple(int(v,16) for v in row) for row in native["cubes"]]
    e=DomainCUDA(p,roots,target=target)
    e.restore(unpack_native(native,e))
    return e


def fixed_point(e):
    for _ in range(100):
        states,ready=e.propagate_root(65536)
        if ready or all(states==2):
            return e.remaining_domain_cubes()
    raise AssertionError("root fixed point did not finish")


def original(p):
    scope=WorkScope.for_problem(p,"gold")
    roots=split_cubes([scope.root],4)
    cases=[]
    with closing(DomainCUDA(p,roots)) as e:
        for quantum in (20480,65536):
            e.propagate_root(quantum)
            initial=e.stats()
            saved=checkpoint(e)
            narrow,telemetry=resize(p,"gold",saved,1)
            assert telemetry["gpu_domain_lanes"]==1 and telemetry["gpu_domain_parked_lanes"]==3
            assert narrow["frontier"]==saved["frontier"]
            fixed=[]
            for native in [narrow["native"],*narrow["native_queue"]]:
                with closing(open_native(p,"gold",native)) as live:
                    before=live.stats()
                    start=live.remaining_domain_cubes()
                    assert len(start)==1
                    native_fixed=fixed_point(live)
                    after=live.stats()
                with closing(DomainCUDA(p,start)) as cold:
                    cold_fixed=fixed_point(cold)
                    cold_stats=cold.stats()
                assert cold_fixed==native_fixed
                fixed.append({"native_new_checks":after["support_checks"]-before["support_checks"],
                              "portable_new_checks":cold_stats["support_checks"],"frontier_digest":digest(cold_fixed)})
            cases.append({"saved_checks":initial["support_checks"],"saved_removals":initial["domain_removals"],
                          "fixed_points":fixed,"remaining_gpu_lanes":1,"host_native_lanes":3})
        return {"device":e.device,"fingerprint":p.fingerprint,"cases":cases}


def all_solutions(p,target,result):
    e_archives=[result["native"],*result.get("native_queue",())]
    found=set()
    for archive in e_archives:
        if archive is None:continue
        with closing(open_native(p,target,archive)) as e:
            for _ in range(1000):
                for item in e.candidates():
                    value=tuple(item["codes"])
                    assert value not in found
                    found.add(value)
                    e.acknowledge([item["lane"]])
                if all(e.step(64)==2):break
            else:raise AssertionError("finite slice did not finish")
    for row in result["queue"]:
        with closing(DomainCUDA(p,[tuple(int(v,16) for v in row)],target=target)) as e:
            for _ in range(1000):
                for item in e.candidates():
                    value=tuple(item["codes"])
                    assert value not in found
                    found.add(value)
                    e.acknowledge([item["lane"]])
                if all(e.step(64)==2):break
            else:raise AssertionError("portable tail did not finish")
    return found


def finite():
    outcomes=[]
    for kind in ("branch","donation","fork","factored"):
        p=precise_reason_fixture() if kind=="donation" else fork_gold_fixture() if kind=="fork" else joint_hall_fixture()
        target="gold" if kind=="fork" else "edges"
        scope=WorkScope.for_problem(p,target)
        opts={}
        if kind=="fork":
            opts["gpu_propagated_forks"]=True
            with closing(DomainCUDA(p,target=target)) as root:
                fixed=fixed_point(root)
                e=root.fork_root(split_cubes(fixed,4))
        else:
            roots=split_cubes([scope.root],4)
            e=DomainCUDA(p,roots,target=target)
        with closing(e):
            e.step(80 if kind=="donation" else 16)
            pool=None
            totals=dict.fromkeys(COUNTERS,0)
            retired=0
            if kind=="donation":
                pool=LanePool(scope,e.cubes)
                pool.take_batch(4)
                counts,retired,_=pool.retire(e,e.state[0].get())
                for key in COUNTERS:totals[key]+=counts.get(key,0)
                donor=pool.donate(e)
                assert donor is not None
                for key in COUNTERS:totals[key]+=donor.get(key,0)
                opts["gpu_lane_balance"]=True
            saved=checkpoint(e,balanced=pool,totals=totals,completed=retired,owner=scope.root if kind=="fork" else None)
            native_before=e.state.get()
            with TemporaryDirectory() as tmp:
                if kind=="factored":
                    opts["gpu_incremental_checkpoint"]=True
                    store=CheckpointStore(tmp,scope.payload())
                    saved["native"],blobs,_=encode_native(e,e.checkpoint())
                    saved["digest"]=digest({k:v for k,v in saved.items() if k!="digest"})
                    source=store.write(saved,blobs)
                else:
                    source=saved
                narrow,stats=resize(p,target,source,1,opts,tmp)
                mid,_=resize(p,target,narrow,2,opts,tmp)
                wide,_=resize(p,target,mid,8,opts,tmp)
                with closing(open_native(p,target,wide["native"])) as restored:
                    if kind!="factored":assert (restored.state.get()==native_before).all()
                assert saved["frontier"]==narrow["frontier"]==mid["frontier"]==wide["frontier"]
                assert wide.get("lane_pool")==saved.get("lane_pool")
                expected=independent_solutions(p,target)
                assert all_solutions(p,target,narrow)==expected
                assert all_solutions(p,target,mid)==expected
                fallback,fstats=resize(p,target,narrow,2,{"gpu_joint_support":True})
                assert fstats["gpu_domain_portable_restart"]==1 and not fallback.get("native_queue")
                assert fallback["frontier"]==saved["frontier"]
                outcomes.append({"case":kind,"solutions":len(expected),"source_width":e.lanes,
                                 "source_frame_depths":list(map(int,native_before[1])),
                                 "parked_at_one":stats["gpu_domain_parked_lanes"],
                                 "saved_decisions":stats["decisions"],"saved_support_checks":stats["support_checks"],
                                 "donations":stats.get("gpu_lane_donations",0),
                                 "portable_fallback_decisions":fstats["decisions"]})
    return outcomes


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,default=Path("data/tiles.json"))
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    result={"finite":finite(),"original":original(SearchProblem(read_json(args.tiles)))}
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result,flush=True)
