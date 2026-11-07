import copy
import unittest
from unittest.mock import patch
import time
import os
from constraint_coordinator_round26 import Coordinator, solve, AllocationPolicy, split_box
from constraint_work import WorkScope, frontier_snapshot, digest
from constraint_symbolic_transfer import portable_snapshot, restore_portable
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, multiple_loops, double_face
from test_constraint_sat_assist import HAS_SAT
from test_symbolic_frontier_round26 import points

def checkpoint_worker(problem,connection,cancel):
    from constraint_dfs import _solve_core
    while True:
        request=connection.recv()
        if request is None:break
        result=_solve_core(problem,seconds=0,target=request['target'],options=request['options'],
                           resume=request['checkpoint'],_initial_domains=request['box'])
        connection.send({'ticket':request['ticket'],'backend':'cpu','result':result,'seconds':0.0})
    connection.close()

def failed_worker(problem,connection,cancel):
    request=connection.recv()
    connection.send({'ticket':request['ticket'],'backend':request['backend'],'error':'injected native failure'})
    time.sleep(5)

class CoordinatorTests(unittest.TestCase):
    def test_zero_duration_and_restart_do_not_start_workers(self):
        problem = SearchProblem(simple_tiles(), double_face())
        first = solve(problem, seconds=0, workers=2)
        self.assertEqual(first['status'], 'timeout')
        self.assertEqual(first['stats']['coordinator_worker_starts'], 0)
        second = solve(problem, seconds=0, workers=1, resume=first['checkpoint'])
        self.assertEqual(second['checkpoint']['frontier'], first['checkpoint']['frontier'])

    def test_cpu_parallel_gold_and_infeasible(self):
        for tiles, expected in ((simple_tiles(), 'solved'), (multiple_loops(), 'infeasible')):
            problem = SearchProblem(tiles, double_face())
            result = solve(problem, seconds=12, workers=2, options={'native':'python','lookahead':False})
            self.assertEqual(result['status'], expected, result.get('error'))
            self.assertTrue(result['complete'])
            if expected == 'solved': self.assertTrue(problem.validate(result['codes'])['valid'])
            else:
                restored = Coordinator(problem, resume=result['checkpoint'])
                self.assertEqual(restored.frontier.remaining_volume(), 0)
                again = restored.advance(1)
                self.assertEqual(again['stats']['coordinator_worker_starts'], result['stats']['coordinator_worker_starts'])

    @unittest.skipUnless(HAS_SAT, 'optional SAT dependency')
    def test_cpu_and_sat_share_disjoint_work(self):
        problem = SearchProblem(multiple_loops(), double_face())
        result = solve(problem, seconds=12, workers=2,
                       options={'native':'python','coordinated_sat':True,'adaptive_allocation':True,'sat_guard_reuse':True})
        self.assertEqual(result['status'], 'infeasible', result.get('error'))
        self.assertGreater(result['stats']['coordinator_jobs_closed'], 1)
        self.assertEqual(result['stats']['coordinator_raw_remaining'], '0')
        self.assertIn('coordinator_sat_decisions', result['stats'])

    @unittest.skipUnless(HAS_SAT, 'optional SAT dependency')
    def test_sat_adapter_gold_and_legacy_cpu_frontier(self):
        from constraint_dfs import _solve_core
        problem = SearchProblem(simple_tiles(), double_face())
        native = _solve_core(problem, seconds=0)['checkpoint']
        result = solve(problem, seconds=12, resume=native, engine_selection='sat')
        self.assertEqual(result['status'], 'solved', result.get('error'))
        self.assertTrue(problem.validate(result['codes'])['valid'])

    def test_stopped_checkpoint_keeps_exact_pending_work(self):
        problem = SearchProblem(simple_tiles(), double_face())
        result = solve(problem, seconds=4, stop=lambda:True, workers=2)
        self.assertEqual(result['status'], 'stopped')
        state = Coordinator(problem, resume=result['checkpoint'], workers=1)
        self.assertEqual(state.frontier.remaining_volume(), 36)
        state.close()

    def test_forged_lease_cursor_mismatch_is_rejected(self):
        problem = SearchProblem(simple_tiles(), double_face())
        coordinator = Coordinator(problem)
        lease = coordinator.frontier.claim('coordinator', choose=lambda box:(1, box[1]))
        from constraint_coordinator_round26 import Job
        coordinator.jobs[lease.ticket] = Job(lease, 'cpu')
        saved = coordinator.snapshot()
        saved['jobs'][0]['ticket'] += 1
        saved['digest'] = digest({k:v for k,v in saved.items() if k != 'digest'})
        with self.assertRaises(ValueError): Coordinator(problem, resume=saved)

    def test_terminal_solution_is_revalidated_without_starting_workers(self):
        problem=SearchProblem(simple_tiles(),double_face())
        result=solve(problem,seconds=12,options={'native':'python'})
        again=solve(problem,seconds=12,resume=result['checkpoint'])
        self.assertEqual(again['status'],'solved')
        self.assertEqual(again['codes'],result['codes'])
        self.assertEqual(again['stats']['coordinator_worker_starts'],result['stats']['coordinator_worker_starts'])
        corrupt=copy.deepcopy(result['checkpoint']);corrupt['terminal']=[0,0]
        corrupt['digest']=digest({k:v for k,v in corrupt.items() if k!='digest'})
        with self.assertRaises(ValueError):Coordinator(problem,resume=corrupt)

    def test_warm_owners_resume_and_resource_change_closes_old_process(self):
        problem=SearchProblem(simple_tiles(),double_face());state={}
        with patch('constraint_coordinator_round26._worker',checkpoint_worker):
            first=solve(problem,seconds=.1,learning_state=state,options={'native':'python','symbolic_frontier':True})
            self.assertEqual(first['status'],'timeout',first['error'])
            owner=state['_symbolic_coordinator'];process=owner.slots[0]['process']
            self.assertTrue(process.is_alive())
            second=solve(problem,seconds=.05,learning_state=state,resume=first['checkpoint'])
            self.assertIs(owner,state['_symbolic_coordinator'])
            self.assertEqual(second['stats']['coordinator_worker_starts'],1)
            third=solve(problem,seconds=0,learning_state=state,resume=second['checkpoint'],workers=2)
            self.assertFalse(process.is_alive())
            self.assertIsNot(owner,state['_symbolic_coordinator'])
            self.assertEqual(Coordinator(problem,resume=third['checkpoint']).frontier.remaining_volume(),36)
        state['_symbolic_coordinator'].close()

    @unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','optional CUDA adapter')
    def test_cuda_adapter_solution_and_native_checkpoint_transfer(self):
        from test_constraint_dfs import tetrahedron_tiles
        problem=SearchProblem(*tetrahedron_tiles())
        options={'gpu_support_reuse':True,'gpu_watched_clauses':True,'gpu_global_matching':True}
        result=solve(problem,seconds=30,engine_selection='gpu-domains',replicas=4,options=options)
        self.assertEqual(result['status'],'solved',result['error'])
        self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertIn('coordinator_gpu_decisions',result['stats'])
        from gpu_domain_runner import solve as gpu_solve
        from constraint_symbolic_transfer import engine_remaining
        native=gpu_solve(problem,seconds=.0001,replicas=4,options=options)
        if native.get('checkpoint') is not None and native['status'] in ('timeout','stopped'):
            scope,rows=engine_remaining(problem,'gpu',native['checkpoint'])
            self.assertTrue(WorkScope.for_problem(problem,'gold').accepts(scope))
            self.assertTrue(rows)

    def test_dispatcher_uses_symbolic_engine(self):
        from constraint_search import run_solver
        problem=SearchProblem(simple_tiles(),double_face())
        result=run_solver(problem,'parallel-dfs',seconds=0,workers=2,replicas=4096,
                          options={'symbolic_frontier':True},seed_board=None)
        self.assertEqual(result['checkpoint']['schema'],'diamond-symbolic-coordinator-v1')
        self.assertEqual(result['stats']['coordinator_cpu_workers'],2)

    def test_warm_resource_change_without_explicit_resume_preserves_completed_set(self):
        problem=SearchProblem(simple_tiles(),double_face());owner=Coordinator(problem,options={'native':'python','symbolic_frontier':True})
        lease=owner.frontier.claim('coordinator',choose=lambda box:(1,box[1]));owner.frontier.commit(lease,())
        expected=owner.frontier.remaining_volume();state={'_symbolic_coordinator':owner}
        try:
            result=solve(problem,seconds=0,workers=2,learning_state=state)
            self.assertEqual(int(result['stats']['coordinator_raw_remaining']),expected)
            self.assertEqual(state['_symbolic_coordinator'].options['native'],'python')
        finally:state['_symbolic_coordinator'].dispose()

    def test_warm_options_keep_requested_gpu_replicas(self):
        problem=SearchProblem(simple_tiles(),double_face())
        owner=Coordinator(problem,options={'coordinated_gpu':True},replicas=256)
        state={'_symbolic_coordinator':owner}
        try:
            result=solve(problem,seconds=0,replicas=256,learning_state=state)
            self.assertIs(state['_symbolic_coordinator'],owner)
            self.assertEqual(owner.replicas,256)
        finally:state['_symbolic_coordinator'].dispose()

    def test_failed_worker_keeps_committed_root_and_charges_one_replay(self):
        problem=SearchProblem(simple_tiles(),double_face())
        with patch('constraint_coordinator_round26._worker',failed_worker):
            started=time.monotonic();result=solve(problem,seconds=5,options={'native':'python'})
        self.assertEqual(result['status'],'error')
        self.assertIn('injected native failure',result['error'])
        self.assertLess(time.monotonic()-started,4)
        self.assertEqual(result['stats']['coordinator_replayed_slices'],1)
        self.assertEqual(int(result['stats']['coordinator_raw_remaining']),36)
        final=solve(problem,seconds=10,resume=result['checkpoint'],options={'native':'python'})
        self.assertEqual(final['status'],'solved',final['error'])

    def test_split_is_proper_and_complete(self):
        box = (7, 3, 15)
        half = split_box(box)
        axis = next(i for i in range(3) if half[i] != box[i])
        other = list(box); other[axis] &= ~half[axis]
        self.assertEqual(points(half) | points(other), points(box))
        self.assertFalse(points(half) & points(other))

    def test_adaptation_does_not_compare_unlike_decision_units(self):
        policy = AllocationPolicy()
        for _ in range(4):
            policy.observe((7,7), 'cpu', 1, True)
            policy.observe((7,7), 'sat', 1, False)
        observed = [policy.choose((7,7), ['cpu','sat'], 'cpu') for _ in range(8)]
        self.assertIn('cpu', observed)
        # A distinct engine can still receive exploration despite poor history.
        policy.observe((7,7), 'cpu', 2, True)
        restored = AllocationPolicy(policy.snapshot())
        self.assertEqual(restored.snapshot(), policy.snapshot())


class EngineTelemetryTests(unittest.TestCase):
    def owner_job(self):
        from constraint_coordinator_round26 import Job
        from constraint_dfs import _solve_core
        problem=SearchProblem(simple_tiles(),double_face());owner=Coordinator(problem,options={'native':'python'})
        self.addCleanup(owner.dispose);lease=owner.frontier.claim('coordinator');job=Job(lease,'cpu');owner.jobs[lease.ticket]=job
        checkpoint=_solve_core(problem,seconds=0,_initial_domains=lease.box,options={'native':'python'})['checkpoint']
        return problem,owner,job,checkpoint
    @staticmethod
    def deliver(owner,job,checkpoint,stats):
        from types import SimpleNamespace
        result={'status':'timeout','complete':False,'nodes':job.metric,'checkpoint':checkpoint,'stats':stats}
        message={'ticket':job.lease.ticket,'backend':job.backend,'result':result,'seconds':.1}
        slot={'ticket':job.lease.ticket,'connection':SimpleNamespace(recv=lambda:message)}
        owner._receive(slot)
    def test_counter_deltas_and_unknown_maxima_survive_restart_and_resource_change(self):
        problem,owner,job,checkpoint=self.owner_job()
        self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':10,'native_search_peak_depth':4,'future_metric':20})
        self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':15,'native_search_peak_depth':3,'future_metric':2})
        self.assertEqual(owner.public_stats()['coordinator_cpu_monotonic_delta_regional_compiler_transitions'],15)
        self.assertEqual(owner.public_stats()['coordinator_cpu_maximum_observed_native_search_peak_depth'],4)
        self.assertEqual(owner.public_stats()['coordinator_cpu_maximum_observed_future_metric'],20)
        resumed=Coordinator(problem,resume=owner.snapshot(),workers=2);self.addCleanup(resumed.dispose);job=next(iter(resumed.jobs.values()))
        self.deliver(resumed,job,job.checkpoint,{'regional_compiler_transitions':18,'future_metric':25})
        self.assertEqual(resumed.public_stats()['coordinator_cpu_monotonic_delta_regional_compiler_transitions'],18)
        self.assertEqual(resumed.public_stats()['coordinator_cpu_maximum_observed_future_metric'],25)
        state={'_symbolic_coordinator':resumed};result=solve(problem,seconds=0,learning_state=state,workers=1)
        self.addCleanup(state['_symbolic_coordinator'].dispose)
        self.assertEqual(result['stats']['coordinator_cpu_monotonic_delta_regional_compiler_transitions'],18)
    def test_backend_units_and_new_job_counters_do_not_mix_or_replay_prefix(self):
        from constraint_coordinator_round26 import Job
        problem,owner,job,checkpoint=self.owner_job()
        self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':11})
        saved=owner.engine_stats['cpu'].copy();new=Job(job.lease,'cpu')
        staged,prior=owner._stage_engine_stats(new,{'regional_compiler_transitions':4});owner.engine_stats['cpu']=staged;new.prior_stats=prior
        self.assertEqual(staged['regional_compiler_transitions']['value'],15)
        sat=Job(job.lease,'sat');staged,_=owner._stage_engine_stats(sat,{'session_decisions':7,'regional_dag_arcs':5})
        self.assertEqual(staged['session_decisions']['value'],7)
        self.assertNotIn('session_decisions',owner.engine_stats['cpu'])
        repeated,_=owner._stage_engine_stats(new,{'regional_compiler_transitions':4})
        self.assertEqual(repeated['regional_compiler_transitions']['value'],15)
    def test_reset_is_explicit_maximum_not_a_fabricated_work_total(self):
        _,owner,job,checkpoint=self.owner_job()
        self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':10})
        self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':3})
        row=owner.engine_stats['cpu']['regional_compiler_transitions']
        self.assertEqual(row,{'mode':'maximum_observed','value':10,'maximum':10,'resets':1})
        self.assertEqual(owner.public_stats()['coordinator_engine_metric_resets'],1)
    def test_counter_schema_and_legacy_partial_history_are_explicit(self):
        problem,owner,job,checkpoint=self.owner_job();self.deliver(owner,job,checkpoint,{'regional_compiler_transitions':1<<1400})
        snapshot=owner.snapshot();restored=Coordinator(problem,resume=snapshot);self.addCleanup(restored.dispose)
        self.assertEqual(restored.engine_stats,owner.engine_stats)
        bad=copy.deepcopy(snapshot);bad['engine_stats']['cpu']['regional_compiler_transitions']['value']=False;bad['digest']=digest({k:v for k,v in bad.items() if k!='digest'})
        with self.assertRaises(ValueError):Coordinator(problem,resume=bad)
        legacy=copy.deepcopy(snapshot);legacy.pop('engine_stats');legacy.pop('engine_stats_complete')
        for row in legacy['jobs']:row.pop('prior_stats')
        legacy['digest']=digest({k:v for k,v in legacy.items() if k!='digest'})
        previous=Coordinator(problem,resume=legacy);self.addCleanup(previous.dispose)
        self.assertFalse(previous.public_stats()['coordinator_engine_stats_complete'])


class StalledReallocationTests(unittest.TestCase):
    def owner(self, options=None):
        from constraint_coordinator_round26 import Job
        problem=SearchProblem(simple_tiles(),double_face())
        owner=Coordinator(problem, options={'native':'python','adaptive_allocation':True,**(options or {})})
        self.addCleanup(owner.dispose)
        lease=owner.frontier.claim('coordinator')
        job=Job(lease,'cpu');owner.jobs[lease.ticket]=job
        return problem,owner,job

    @staticmethod
    def remaining(owner):
        nodes=[owner.frontier.pending]+[node for _,node in owner.frontier.leases.values()]
        return set().union(*(points(box) for node in nodes for box in owner.frontier.sets.iter_boxes(node)))

    @staticmethod
    def deliver(owner,job,rows):
        from types import SimpleNamespace
        message={'ticket':job.lease.ticket,'backend':job.backend,'seconds':.1,
                 'result':{'status':'timeout','complete':False,'nodes':job.metric+1,
                           'checkpoint':{'trusted_test_cursor':True},'stats':{}}}
        slot={'ticket':job.lease.ticket,'connection':SimpleNamespace(recv=lambda:message)}
        with patch('constraint_coordinator_round26.engine_remaining',return_value=(owner.frontier.scope,rows)):
            owner._receive(slot)
        return slot

    def test_stalled_alternative_preserves_exact_set_and_resumes_without_old_cursor(self):
        from constraint_coordinator_round26 import STALLED_REALLOCATION_SLICES
        problem,owner,job=self.owner({'coordinated_sat':True})
        rows=((1,owner.frontier.scope.root[1]),(2,3))
        owner.frontier.commit(job.lease,rows,release=False)
        expected=self.remaining(owner)
        closed=owner.frontier.sets.volume(owner.frontier.completed)
        for _ in range(STALLED_REALLOCATION_SLICES-1):
            self.deliver(owner,job,rows)
            self.assertIn(job.lease.ticket,owner.jobs)
        self.assertEqual(owner.stats['coordinator_stalled_reallocations'],0)
        slot=self.deliver(owner,job,rows)
        self.assertIsNone(slot['ticket'])
        replacement=next(iter(owner.jobs.values()))
        self.assertEqual(replacement.backend,'sat')
        self.assertIsNone(replacement.checkpoint)
        self.assertNotEqual(replacement.lease.ticket,job.lease.ticket)
        self.assertEqual(self.remaining(owner),expected)
        self.assertEqual(owner.frontier.sets.volume(owner.frontier.completed),closed)
        self.assertEqual(owner.stats['coordinator_stalled_reallocations'],1)
        self.assertEqual(owner.stats['coordinator_adaptive_cursor_restarts'],1)
        self.assertEqual(owner.stats['coordinator_cpu_nodes'],STALLED_REALLOCATION_SLICES)
        resumed=Coordinator(problem,resume=owner.snapshot(),workers=2);self.addCleanup(resumed.dispose)
        self.assertEqual(self.remaining(resumed),expected)
        self.assertEqual(next(iter(resumed.jobs.values())).backend,'sat')
        self.assertEqual(resumed.stats['coordinator_adaptive_cursor_restarts'],1)

    def test_cpu_only_stall_changes_real_profile_and_disabled_policy_never_reallocates(self):
        from constraint_coordinator_round26 import STALLED_REALLOCATION_SLICES
        from constraint_context_policy_round26 import profile_options
        _,owner,job=self.owner()
        job.unchanged=STALLED_REALLOCATION_SLICES-1
        before=self.remaining(owner)
        self.deliver(owner,job,(job.lease.box,))
        replacement=next(iter(owner.jobs.values()))
        self.assertEqual(replacement.backend,'cpu')
        self.assertNotEqual(profile_options(owner.options,replacement.profile),profile_options(owner.options,job.profile))
        self.assertEqual(self.remaining(owner),before)
        _,disabled,other=self.owner({'adaptive_allocation':False})
        other.unchanged=STALLED_REALLOCATION_SLICES
        self.deliver(disabled,other,(other.lease.box,))
        self.assertIs(disabled.jobs[other.lease.ticket],other)
        self.assertEqual(disabled.stats['coordinator_adaptive_cursor_restarts'],0)

    def test_no_meaningful_alternative_and_failed_set_construction_keep_owned_cursor(self):
        from constraint_coordinator_round26 import STALLED_REALLOCATION_SLICES
        from constraint_context_policy_round26 import profile_options,PROFILES
        all_profiles={}
        for profile in PROFILES:all_profiles=profile_options(all_profiles,profile)
        _,owner,job=self.owner(all_profiles)
        job.unchanged=STALLED_REALLOCATION_SLICES
        self.assertFalse(owner._reallocate_stalled(job))
        self.assertIs(owner.jobs[job.lease.ticket],job)
        _,other,other_job=self.owner({'coordinated_sat':True})
        other_job.unchanged=STALLED_REALLOCATION_SLICES
        before=other.snapshot()
        with patch.object(other.frontier.sets,'union',side_effect=RuntimeError('injected set cap')):
            with self.assertRaisesRegex(RuntimeError,'injected set cap'):other._reallocate_stalled(other_job)
        # Optional policy observations can change, but every ownership/cursor
        # root and its counters must remain exactly as before the failure.
        after=other.snapshot()
        for key in ('frontier','jobs','states','stats'):self.assertEqual(after[key],before[key])

    def test_received_slice_accounting_survives_optional_reallocation_failure(self):
        from constraint_coordinator_round26 import STALLED_REALLOCATION_SLICES
        from constraint_symbolic_work import DiagramLimit
        for error in (DiagramLimit('allocation cap'),RuntimeError('unexpected allocator failure')):
            _,owner,job=self.owner({'coordinated_sat':True})
            job.unchanged=STALLED_REALLOCATION_SLICES-1
            before=self.remaining(owner)
            with patch.object(owner,'_reallocate_stalled',side_effect=error):
                if isinstance(error,DiagramLimit):self.deliver(owner,job,(job.lease.box,))
                else:
                    with self.assertRaisesRegex(RuntimeError,'unexpected'):self.deliver(owner,job,(job.lease.box,))
            self.assertEqual(owner.stats['coordinator_cpu_nodes'],1)
            self.assertEqual(owner.stats['coordinator_slices'],1)
            self.assertEqual(job.metric,1)
            self.assertIs(owner.jobs[job.lease.ticket],job)
            self.assertEqual(self.remaining(owner),before)
            self.assertIsNotNone(job.checkpoint)

    def test_legacy_statistics_restore_with_new_counters_zero(self):
        problem,owner,_=self.owner()
        saved=owner.snapshot()
        saved['stats'].pop('coordinator_stalled_reallocations')
        saved['stats'].pop('coordinator_adaptive_cursor_restarts')
        saved['digest']=digest({key:value for key,value in saved.items() if key!='digest'})
        resumed=Coordinator(problem,resume=saved);self.addCleanup(resumed.dispose)
        self.assertEqual(resumed.stats['coordinator_stalled_reallocations'],0)
        self.assertEqual(resumed.stats['coordinator_adaptive_cursor_restarts'],0)

if __name__ == '__main__': unittest.main()
