"""Deterministic regressions for interrupted full-root preparation leases."""
import copy,threading,unittest
from unittest.mock import patch
import constraint_parallel as parallel
from search_options import normalize_options
from test_constraint_parallel import OPTIONS
from test_constraint_pair_lookahead import turn_problem

class PreparationLeaseTests(unittest.TestCase):
    def test_job_specific_attempts_eventually_receive_enough_time_to_commit(self):
        original={'stack':[{'domains':['0xf'],'cell':0,'choices':[0,1,2,3],'next':0}],
                  'root_symmetry':{'state':'ready'},'nodes':0,'stats':{},'elapsed_seconds':0}
        current=copy.deepcopy(original);stalls={};budgets=[];other_job=202
        stalls[other_job]=2
        for attempt in range(8):
            budget=parallel._slice_budget(2,slots=4,active=4,queued=4,remaining=100,stalled=stalls.get(101,0))
            budgets.append(budget);after=copy.deepcopy(current)
            # An attempt count is explicitly NOT an expanded search node.
            after['nodes']+=100;after['stats']['attempts']=attempt+1;after['elapsed_seconds']+=budget
            if budget>=10:after['stack'][0]['next']=1
            moved=parallel._record_frontier_progress(stalls,101,current,after);current=after
            if moved:break
        self.assertTrue(moved);self.assertEqual(current['stack'][0]['next'],1)
        self.assertGreater(max(budgets),parallel.MAX_SLICE_SECONDS)
        self.assertEqual(stalls[101],0);self.assertEqual(stalls[other_job],2)
        self.assertEqual(original['stack'][0]['next'],0)
        self.assertEqual(parallel._slice_budget(2,slots=4,active=4,queued=4,remaining=100,stalled=stalls[101]),parallel.MAX_SLICE_SECONDS)

    def test_changed_domain_or_cursor_resets_only_its_job_and_caps_remain_finite(self):
        before={'stack':[{'domains':['0xf'],'cell':0,'choices':[0,1,2,3],'next':0}]};stalls={}
        for _ in range(50):self.assertFalse(parallel._record_frontier_progress(stalls,7,before,copy.deepcopy(before)))
        self.assertEqual(stalls[7],parallel.MAX_PREPARATION_STALLS)
        self.assertEqual(parallel._slice_budget(0,slots=8,active=8,queued=8,remaining=100,stalled=stalls[7]),parallel.MAX_PREPARATION_SECONDS)
        self.assertEqual(parallel._slice_budget(0,slots=8,active=8,queued=8,remaining=.03,stalled=stalls[7]),.03)
        self.assertEqual(parallel._slice_budget(0,slots=8,active=8,queued=8,remaining=-1,stalled=stalls[7]),0)
        for change in ('domain','cursor'):
            after=copy.deepcopy(before)
            if change=='domain':after['stack'][0]['domains']=['0x7']
            else:after['stack'][0]['next']=1
            stalls[7]=5;self.assertTrue(parallel._record_frontier_progress(stalls,7,before,after));self.assertEqual(stalls[7],0)

    def test_stall_hints_roundtrip_without_altering_authoritative_job_frontier(self):
        problem=turn_problem(False)
        result=parallel.solve(problem,seconds=0,workers=2,options=OPTIONS);saved=result['checkpoint']
        job_id=saved['jobs'][0]['id'];saved['preparation_stalls']=[[job_id,7]];saved['digest']=parallel._digest(saved)
        again=parallel.solve(problem,seconds=0,workers=2,options=OPTIONS,resume=saved)['checkpoint']
        self.assertEqual(again['jobs'],saved['jobs']);self.assertEqual(again['preparation_stalls'],[[job_id,7]])
        for invalid in (None,[[job_id,8]],[[job_id,-1]],[[job_id,True]],[[job_id+999,2]],[[job_id,1],[job_id,2]]):
            forged=copy.deepcopy(saved);forged['preparation_stalls']=invalid;forged['digest']=parallel._digest(forged)
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                parallel.solve(problem,seconds=0,workers=2,options=OPTIONS,resume=forged)

    def test_idle_peer_cannot_preempt_stalled_preparation_but_stop_can(self):
        ordinary=parallel._idle_yield_delay(0,60)
        protected=parallel._idle_yield_delay(7,60)
        self.assertLess(ordinary,1)
        self.assertEqual(protected,60)
        self.assertLess(.5,protected)
        self.assertLessEqual(parallel._idle_yield_delay(7,1000),parallel.MAX_PREPARATION_SECONDS)
        cancel=threading.Event();yield_request=threading.Event()
        signal=parallel._WorkerSignal(cancel,yield_request)
        self.assertFalse(signal.is_set());cancel.set();self.assertTrue(signal.is_set())
        cancel.clear();yield_request.set();self.assertTrue(signal.is_set())
    def test_long_preparation_lease_keeps_global_cooperative_stop(self):
        problem=turn_problem(False);cancel=threading.Event();options=normalize_options(OPTIONS)
        request={'id':1,'target':'gold','options':options,'checkpoint':{},'seconds':60,'region_key':'scope'}
        class Connection:
            def __init__(self):self.items=iter([request,None]);self.sent=[]
            def recv(self):return next(self.items)
            def send(self,item):self.sent.append(item)
            def close(self):pass
        connection=Connection()
        def fake_solve(p,**kwargs):
            self.assertEqual(kwargs['seconds'],60);self.assertFalse(kwargs['stop']())
            cancel.set();self.assertTrue(kwargs['stop']())
            return {'status':'stopped','nodes':0,'stats':{}}
        with patch.object(parallel,'dfs_solve',side_effect=fake_solve):parallel._worker(problem,connection,cancel)
        self.assertEqual(connection.sent[0]['result']['status'],'stopped')

if __name__=='__main__':unittest.main()
