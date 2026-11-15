"""Host-only native lane resize, archiving, ownership and accounting controls."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from constraint_work import WorkScope,digest,frontier_snapshot
from gpu_domain_runner import COUNTERS,pack_native,solve,split_cubes
from gpu_lane_pool_round25 import LanePool
import test_gpu_domain_checkpoint_validation as host_fixtures


class Array(np.ndarray):
    def get(self):
        return self.view(np.ndarray).copy()


def new_engine(problem,cubes,**options):
    engine=host_fixtures.HostCheckpointTests().engine()
    engine.problem=problem
    engine.cp.cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None))
    engine.joint_support=False
    engine.target=options.get("target","edges")
    engine.binding["target"]=engine.target
    engine.cubes=tuple(cubes)
    engine.lanes=len(cubes)
    engine.state=engine._initial_state().view(Array)
    # These synthetic native columns have been searched for different amounts.
    # One bounded dispatch exhausts each, used only to test host accounting.
    def step(work):
        engine.state[11]+=engine.state[0]!=2
        engine.state[0]=2
        return engine.state[0].get()
    engine.step=step
    return engine


def setup_saved(balance=False,fork=False):
    problem=host_fixtures.HostCheckpointTests().engine().problem
    scope=WorkScope.for_problem(problem,"edges")
    cubes=split_cubes([scope.root],4)
    e=new_engine(problem,cubes)
    e.state[3]=[0,1,0,1]
    e.state[11]=[3,2,7,6]
    e.state[13]=[10,20,30,40]
    checkpoint={"scope":scope.payload(),"queue":[],"frontier":frontier_snapshot(scope,cubes),
                "totals":dict.fromkeys(COUNTERS,0),"completed_lanes":0,"native":pack_native(e),"version":1,
                "active_stats":e.stats()}
    if balance:
        # A propagated or donated owner has four disjoint native descendants.
        pool=LanePool(scope,[scope.root])
        pool.take_batch(1)
        pool.fork(4)
        pool.counters["gpu_lane_donations"]=1
        checkpoint.update(lane_balance=True,lane_pool=pool.snapshot(),balance_stats=pool.public_stats())
    if fork:
        checkpoint.update(propagated_forks=True,preparing=False,
                          prepared_owner=None if balance else [hex(x) for x in scope.root])
    checkpoint["digest"]=digest(checkpoint)
    return problem,checkpoint


def pause_restore(problem,saved,count,opts):
    checks=iter((False,True))
    with patch("gpu_domain_round24.DomainCUDA",side_effect=new_engine):
        return solve(problem,replicas=count,target="edges",resume=saved,options=opts,stop=lambda:next(checks,True))


class NativeResizeTests(TestCase):
    def test_shrink_reopen_and_grow_keep_native_columns_and_portable_frontier(self):
        for balanced in (False,True):
            for forks in (False,True):
                p,old=setup_saved(balanced,forks)
                config={"gpu_lane_balance":balanced,"gpu_propagated_forks":forks}
                old_digest=digest(old)
                small=pause_restore(p,old,1,config)
                self.assertEqual(small["stats"]["gpu_domain_lanes"],1)
                self.assertEqual(small["stats"]["gpu_domain_parked_lanes"],3)
                self.assertEqual(small["stats"]["gpu_domain_portable_restart"],0)
                self.assertEqual(small["stats"]["support_checks"],100)
                self.assertEqual(small["nodes"],18)
                partial=json.loads(json.dumps(small["checkpoint"]))
                self.assertEqual(partial["frontier"],old["frontier"])
                self.assertEqual(len(partial["native_queue"]),3)
                larger=pause_restore(p,partial,3,config)
                self.assertEqual(larger["stats"]["gpu_domain_lanes"],3)
                self.assertEqual(larger["stats"]["gpu_domain_parked_lanes"],1)
                full=pause_restore(p,larger["checkpoint"],8,config)
                self.assertEqual(full["checkpoint"]["native"],old["native"])
                self.assertEqual(full["checkpoint"]["frontier"],old["frontier"])
                self.assertEqual(full["checkpoint"].get("lane_pool"),old.get("lane_pool"))
                self.assertEqual(digest(old),old_digest)
                self.assertEqual(full["checkpoint"].get("prepared_owner"),old.get("prepared_owner"))

    def test_exhausting_one_column_cannot_retire_its_parked_siblings(self):
        for balance in (False,True):
            p,old=setup_saved(balance)
            cfg={"gpu_lane_balance":balance}
            with patch("gpu_domain_round24.DomainCUDA",side_effect=new_engine):
                result=solve(p,target="edges",replicas=1,resume=old,options=cfg)
            self.assertEqual(result["status"],"infeasible")
            self.assertEqual(result["stats"]["gpu_domain_completed"],4)
            self.assertEqual(result["stats"]["support_checks"],100)
            self.assertEqual(result["nodes"],22)
            self.assertFalse(result["checkpoint"]["frontier"]["boxes"])
            if balance:
                self.assertEqual(result["stats"]["gpu_lane_owners_completed"],1)
                self.assertEqual(result["stats"]["gpu_lane_donations"],1)

    def test_crash_boundary_between_slices_retains_parked_tasks_and_owner(self):
        for balance in (False,True):
            p,old=setup_saved(balance,True)
            config={"gpu_lane_balance":balance,"gpu_propagated_forks":True}
            dispatched=[]
            def pause_after_one(*args,**kwargs):
                e=new_engine(*args,**kwargs)
                raw=e.step
                def step(work):
                    dispatched.append(1)
                    return raw(work)
                e.step=step
                return e
            with patch("gpu_domain_round24.DomainCUDA",side_effect=pause_after_one):
                cut=solve(p,target="edges",replicas=1,resume=old,options=config,stop=lambda:bool(dispatched))
            saved=json.loads(json.dumps(cut["checkpoint"]))
            self.assertIsNone(saved["native"])
            self.assertEqual(len(saved["native_queue"]),3)
            self.assertEqual(saved["completed_lanes"],1)
            self.assertEqual(saved["frontier"]["boxes"],old["frontier"]["boxes"][1:])
            if balance:
                self.assertEqual(saved["lane_pool"]["lanes"],[])
                self.assertEqual(len(saved["lane_pool"]["parked"]),3)
            else:
                self.assertEqual(saved["deferred_fork_owner"],old["prepared_owner"])
            with patch("gpu_domain_round24.DomainCUDA",side_effect=new_engine):
                end=solve(p,target="edges",replicas=2,resume=saved,options=config)
            self.assertEqual((end["status"],end["nodes"]),("infeasible",22))
            self.assertEqual(end["stats"]["gpu_domain_completed"],4)
            if balance:
                self.assertEqual(end["stats"]["gpu_lane_owners_completed"],1)

    def test_corrupted_parked_column_or_counters_cannot_be_used(self):
        p,old=setup_saved()
        saved=pause_restore(p,old,2,{})["checkpoint"]
        for kind in ("checksum","sum","binding"):
            bad=deepcopy(saved)
            if kind=="checksum":bad["native_queue"][0]["state_sha256"]="f"*64
            if kind=="sum":bad["active_stats"]["decisions"]+=1
            if kind=="binding":bad["native_queue"][0]["binding"]["fingerprint"]="wrong"
            bad["digest"]=digest({k:v for k,v in bad.items() if k!="digest"})
            with self.assertRaises(ValueError):
                pause_restore(p,bad,1,{})
