"""Independent finite bounds and lifecycle tests for retained repair models."""
import itertools,json,unittest
from unittest.mock import patch
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from constraint_lns import _Control,_Cancelled,_load_cp_model,score
from constraint_repair_service_round26 import RetainedRepairService


def brute(problem,current,cells,minimum=None,gold=False):
    answers=[];pieces=[current[c]//3 for c in cells]
    for perm in itertools.permutations(pieces):
        for rotations in itertools.product(range(3),repeat=len(cells)):
            value=current.copy()
            for cell,tile,rotation in zip(cells,perm,rotations):value[cell]=3*tile+rotation
            if value==current:continue
            checked=score(problem,value)
            if checked['domain_violations'] or minimum is not None and checked['matched_edges']<minimum:continue
            if gold and not problem.validate(value)['valid']:continue
            answers.append((checked['matched_edges'],value))
    return answers

class RetainedRepairTests(unittest.TestCase):
    def setUp(self):
        self.problem=SearchProblem(*tetrahedron_tiles())
        try:self.cp=_load_cp_model()
        except ImportError:self.skipTest('OR-Tools unavailable')
    def query(self,service,current,cells,**kwargs):
        return service.patch_query(current,cells,cp_model=self.cp,control=_Control(5,None),seconds=5,seed=26,target=kwargs.pop('target','edges'),**kwargs)
    def test_changed_patches_outside_and_bounds_match_independent_optima(self):
        service=RetainedRepairService(self.problem)
        try:
            for current in ([0,3,6,9],[4,2,6,9]):
                for cells in ([0,1],[1,2],[0,2,3]):
                    for minimum in (None,4):
                        expected=brute(self.problem,list(current),cells,minimum)
                        candidate,status=self.query(service,list(current),cells,minimum_edges=minimum)
                        self.assertEqual(status,'OPTIMAL' if expected else 'INFEASIBLE')
                        self.assertEqual(None if candidate is None else score(self.problem,candidate)['matched_edges'],max((v for v,_ in expected),default=None))
            self.assertEqual(service.stats['retained_model_builds'],1);self.assertGreater(service.stats['retained_model_reuses'],0)
        finally:service.close()
    def test_completed_result_reuse_durable_roundtrip_and_rebuild_guard(self):
        service=RetainedRepairService(self.problem,guard_limit=2);tail=None;current=[0,3,6,9]
        try:
            result=self.query(service,current,[0,1]);calls=service.stats['retained_native_calls']
            self.assertEqual(self.query(service,current,[0,1]),result);self.assertEqual(service.stats['retained_native_calls'],calls)
            self.query(service,current,[1,2]);self.query(service,current,[2,3]);self.assertEqual(service.stats['retained_model_builds'],2)
            saved=json.loads(json.dumps(service.snapshot()));tail=RetainedRepairService(self.problem,guard_limit=2,resume=saved)
            self.assertEqual(self.query(tail,current,[0,1]),result);self.assertIsNone(tail.model)
        finally:service.close();tail and tail.close()
    def test_gold_solution_and_policy_guard_switch(self):
        service=RetainedRepairService(self.problem);current=[0,3,6,9]
        try:
            expected=brute(self.problem,current,list(range(4)),gold=True);self.assertTrue(expected)
            candidate,status=self.query(service,current,list(range(4)),target='gold')
            self.assertTrue(self.problem.validate(candidate)['valid'])
            expected_edges=brute(self.problem,current,[0,1]);candidate,status=self.query(service,current,[0,1],target='edges')
            self.assertEqual(score(self.problem,candidate)['matched_edges'],max(v for v,_ in expected_edges))
        finally:service.close()
    def test_stop_before_model_and_scoped_input_change_reject(self):
        service=RetainedRepairService(self.problem)
        try:
            with self.assertRaises(_Cancelled):service.patch_query([0,3,6,9],[0,1],cp_model=self.cp,control=_Control(5,lambda:True),seconds=5,seed=1,target='edges')
            self.assertIsNone(service.model)
            original=self.problem.domains[0];self.problem.domains[0]=original[:1]
            with self.assertRaisesRegex(ValueError,'input or source'):service.snapshot()
            self.problem.domains[0]=original
        finally:service.close()
    def test_completed_policy_updates_fairness_and_bounds(self):
        service=RetainedRepairService(self.problem)
        try:
            service.observe('8',10,complete=False);self.assertEqual(service.stats['retained_policy_observations'],0)
            service.observe('8',10,complete=True);service.observe('16',0,complete=True)
            chosen=[service.select(['8','16','24']) for _ in range(18)]
            self.assertEqual(set(chosen),{'8','16','24'});self.assertGreater(chosen.count('8'),chosen.count('16'))
            for i in range(70):service.observe('policy'+str(i),0,complete=True)
            self.assertLessEqual(len(service.outcomes),64);self.assertGreater(service.stats['retained_policy_evictions'],0)
            saved=json.loads(json.dumps(service.snapshot()));tail=RetainedRepairService(self.problem,resume=saved);tail.close()
        finally:service.close()
    def test_interrupted_native_query_and_capacity_rebuild_keep_context(self):
        service=RetainedRepairService(self.problem,guard_limit=2);tail=None;current=[0,3,6,9]
        try:
            service._build(self.cp,_Control(5,None))
            with patch.object(service.solver,'solve',side_effect=_Cancelled()):
                with self.assertRaises(_Cancelled):self.query(service,current,[0,1])
            self.assertEqual(len(service.pending),1);self.assertTrue(service.pending[0]['retry'])
            self.assertEqual(service.stats['retained_native_exceptions'],1);self.assertIsNone(service.model)
            saved=json.loads(json.dumps(service.snapshot()));tail=RetainedRepairService(self.problem,guard_limit=2,resume=saved)
            candidate,status=self.query(tail,current,[0,1]);self.assertEqual(status,'OPTIMAL')
            self.assertEqual(tail.stats['retained_native_retries'],1);self.assertEqual(tail.pending,[])
            expected=brute(self.problem,current,[0,1]);self.assertEqual(score(self.problem,candidate)['matched_edges'],max(v for v,_ in expected))
        finally:service.close();tail and tail.close()

    def test_ordinary_session_keeps_model_and_external_owner_lifetime(self):
        from constraint_lns_resume_round25 import RepairSession
        from test_constraint_dfs import multiple_loops,double_face
        p=SearchProblem(multiple_loops(),double_face());owner=RetainedRepairService(p);session=None;tail=None
        try:
            session=RepairSession(p,[0,3],phase_seconds=5,target='gold',max_attempts=4,patch_sizes=(2,),retained=True,retained_service=owner)
            session.advance(1,max_attempts_per_slice=1);self.assertEqual(session.index,1)
            saved=json.loads(json.dumps(session.snapshot()));session.close();self.assertFalse(owner.closed)
            tail=RepairSession(p,[0,3],phase_seconds=5,target='gold',max_attempts=4,patch_sizes=(2,),retained=True,retained_service=owner,resume=saved)
            tail.advance(1,max_attempts_per_slice=1);self.assertEqual(tail.index,2);self.assertEqual(owner.stats['retained_model_builds'],1)
        finally:
            if session:session.close()
            if tail:tail.close()
            owner.close()

    def test_allocation_and_nondefault_mode_order_survive_pending_restore(self):
        from constraint_lns_resume_round25 import RepairSession
        owner=RetainedRepairService(self.problem);session=tail=None
        try:
            owner.observe('mode:fragment_crossover',20,complete=True)
            owner.observe('mode:core_repair',0,complete=True);owner.turn=1
            kwargs=dict(phase_seconds=12,target='gold',max_attempts=4,patch_sizes=(2,3),retained=True,
                        experiments={'core_repair':True,'fragment_crossover':True})
            session=RepairSession(self.problem,[0,3,6,9],retained_service=owner,**kwargs)
            self.assertGreater(session.experiment_budgets['mode:fragment_crossover'],session.experiment_budgets['mode:core_repair'])
            self.assertAlmostEqual(sum(session.experiment_budgets.values()),12)
            count=owner.stats['retained_policy_observations']
            with patch('constraint_repair_round24.fragment_crossover',return_value={'status':'timeout','continuation_complete':False,'checkpoint':{'pending':True}}):
                session._experiment(_Control(1,None),None)
            self.assertEqual(session.experiment_order[0],'fragment_crossover')
            self.assertEqual(owner.stats['retained_policy_observations'],count)
            saved=json.loads(json.dumps(session.snapshot()))
            tail=RepairSession(self.problem,[0,3,6,9],retained_service=owner,resume=saved,**kwargs)
            self.assertEqual(tail.experiment_order,session.experiment_order)
            self.assertEqual(tail.experiment_budgets,session.experiment_budgets)
            self.assertEqual(tail.experiment_pending,session.experiment_pending)
            before=owner.stats['retained_policy_selections']
            with patch('constraint_repair_round24.fragment_crossover',return_value={'status':'inconclusive','continuation_complete':True}):
                tail._experiment(_Control(1,None),None)
            self.assertEqual(owner.stats['retained_policy_selections'],before)
            self.assertEqual(owner.stats['retained_policy_observations'],count+1)
        finally:
            if session:session.close()
            if tail:tail.close()
            owner.close()
    def test_partial_model_construction_cancel_is_charged_and_rebuilt(self):
        service=RetainedRepairService(self.problem)
        class StopDuringBuild:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls==5:raise _Cancelled()
        try:
            with self.assertRaises(_Cancelled):service._build(self.cp,StopDuringBuild())
            self.assertIsNone(service.model)
            self.assertEqual(service.stats['retained_model_builds'],0)
            self.assertEqual(service.stats['retained_model_build_interruptions'],1)
            self.assertGreater(service.stats['retained_abandoned_variables'],0)
            self.assertGreater(service.stats['retained_abandoned_constraints'],0)
            self.query(service,[0,3,6,9],[0,1])
            self.assertEqual(service.stats['retained_model_build_attempts'],2)
            self.assertEqual(service.stats['retained_model_builds'],1)
        finally:service.close()
    def test_eviction_rebuild_and_changed_root_reject_without_false_reuse(self):
        from constraint_work import digest
        service=RetainedRepairService(self.problem,cache_limit=1);tail=None
        try:
            self.query(service,[0,3,6,9],[0,1]);self.query(service,[0,3,6,9],[1,2])
            self.assertEqual(len(service.results),1);self.assertEqual(service.stats['retained_result_evictions'],1)
            saved=json.loads(json.dumps(service.snapshot()))
            changed=json.loads(json.dumps(saved));changed['results'][0]['context']['patch']=[[0]]
            changed['digest']=digest({k:v for k,v in changed.items() if k!='digest'})
            with self.assertRaises(ValueError):RetainedRepairService(self.problem,cache_limit=1,resume=changed)
            domains=self.problem.domains[0];self.problem.domains[0]=domains[:1]
            with self.assertRaisesRegex(ValueError,'binding'):RetainedRepairService(self.problem,cache_limit=1,resume=saved)
            self.problem.domains[0]=domains
            tail=RetainedRepairService(self.problem,cache_limit=1,resume=saved)
            calls=tail.stats['retained_native_calls'];self.query(tail,[0,3,6,9],[0,1])
            self.assertGreater(tail.stats['retained_native_calls'],calls)
        finally:service.close();tail and tail.close()
    def test_model_size_limit_explicitly_rejects_and_leaves_no_native_owner(self):
        service=RetainedRepairService(self.problem,model_text_limit=1024)
        try:
            self.assertEqual(self.query(service,[0,3,6,9],[0,1]),(None,'MODEL_BUDGET'))
            self.assertEqual(service.stats['retained_model_caps'],1)
            self.assertIsNone(service.model);self.assertIsNone(service.solver)
            self.assertEqual(service.stats['retained_model_builds'],0)
            self.assertEqual(service.stats['retained_model_build_interruptions'],1)
            self.assertEqual(service.results,[]);self.assertEqual(service.pending,[])
        finally:service.close()

    def test_completed_edge_bound_reuses_changed_threshold_without_native_query(self):
        service=RetainedRepairService(self.problem)
        try:
            board=[0,3,6,9];cells=[0,1]
            candidate,status=self.query(service,board,cells);maximum=score(self.problem,candidate)['matched_edges']
            calls=service.stats['retained_native_calls']
            for minimum in range(len(self.problem.edges)+1):
                answer,state=self.query(service,board,cells,minimum_edges=minimum)
                expected=brute(self.problem,board,cells,minimum)
                self.assertEqual(state,'OPTIMAL' if expected else 'INFEASIBLE')
                self.assertEqual(None if answer is None else score(self.problem,answer)['matched_edges'],max((value for value,_ in expected),default=None))
            self.assertEqual(service.stats['retained_native_calls'],calls)
            self.assertEqual(service.stats['retained_bound_hits'],len(self.problem.edges)+1)
            # The incompatible excluded seed changes the local feasible set.
            self.query(service,[4,2,6,9],cells,minimum_edges=0)
            self.assertGreater(service.stats['retained_native_calls'],calls)
        finally:service.close()

    def test_nested_gold_owner_resume_stats_and_shared_lifetime(self):
        from constraint_lns_resume_round25 import RepairSession
        owner=RetainedRepairService(self.problem);tail=None
        try:
            pool=owner.gold_pool();answer=pool.oracle(_Control(5,None)).query(())
            self.assertEqual(answer['status'],'solved');calls=pool.stats['gold_pool_native_calls']
            saved=json.loads(json.dumps(owner.snapshot()));tail=RetainedRepairService(self.problem,resume=saved)
            self.assertEqual(tail.gold_pool().oracle(_Control(5,None)).query(())['status'],'solved')
            self.assertEqual(tail.gold_pool().stats['gold_pool_native_calls'],calls)
            self.assertGreater(tail.telemetry()['gold_pool_completed_hits'],0)
            session=RepairSession(self.problem,[0,4,6,9],phase_seconds=5,retained=True,retained_service=owner)
            checkpoint=json.loads(json.dumps(session.snapshot()))
            # Gold scope-check counters change on every snapshot. They do not
            # change the retained semantic owner identity.
            owner.snapshot();again=RepairSession(self.problem,[0,4,6,9],phase_seconds=5,retained=True,retained_service=owner,resume=checkpoint)
            again.close();session.close();self.assertFalse(pool.closed)
        finally:
            owner.close();tail and tail.close()
        self.assertTrue(pool.closed)
    def test_real_hybrid_ordinary_repair_reaches_exact_and_closes_only_owned_service(self):
        import gpu_sampling
        from test_constraint_dfs import multiple_loops,double_face
        p=SearchProblem(multiple_loops(),double_face());closed=[];oldclose=RetainedRepairService.close
        def close(service):closed.append(service);return oldclose(service)
        kwargs=dict(seconds=5,sampling_fraction=0,repair_fraction=.5,seed_board=[0,3],options={'retained_repair':True,'native':'python'})
        with patch('gpu_sampling._make_sampler',side_effect=AssertionError('GPU allocated')),patch.object(RetainedRepairService,'close',close):
            result=gpu_sampling.solve(p,**kwargs)
        self.assertEqual(result['status'],'infeasible');self.assertTrue(result['complete'])
        self.assertEqual(len(closed),1);self.assertTrue(closed[0].closed)
        self.assertGreater(closed[0].stats['retained_model_builds'],0)
        shared={'_hybrid_run_seconds':5}
        try:
            with patch('gpu_sampling._make_sampler',side_effect=AssertionError('GPU allocated')):result=gpu_sampling.solve(p,learning_state=shared,**kwargs)
            self.assertTrue(result['complete']);self.assertFalse(shared['_repair_service'].closed)
            self.assertIn('_runtime',shared)
        finally:
            if '_repair_service' in shared:shared['_repair_service'].close()

    def test_default_nonperfect_archive_json_roundtrip_remains_independently_checked(self):
        from constraint_lns_resume_round25 import RepairSession
        from constraint_work import digest
        initial=[0,4,6,9];session=RepairSession(self.problem,initial,phase_seconds=5)
        self.assertTrue(score(self.problem,initial)['unmatched_edges'])
        saved=json.loads(json.dumps(session.snapshot()));tail=RepairSession(self.problem,initial,phase_seconds=5,resume=saved)
        self.assertEqual(tail.archive,session.archive)
        saved['archive'][0]['matched_edges']+=1;saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
        with self.assertRaisesRegex(ValueError,'independent scoring'):RepairSession(self.problem,initial,phase_seconds=5,resume=saved)

    def test_gold_cut_invalid_incumbent_keeps_heuristic_but_roundtrips_current_model(self):
        from test_constraint_dfs import multiple_loops,double_face
        p=SearchProblem(multiple_loops(),double_face());service=RetainedRepairService(p);tail=None
        try:
            candidate,status=service.patch_query([0,3],[0,1],cp_model=self.cp,control=_Control(10,None),seconds=5,seed=71,target='gold')
            self.assertIn(status,('OPTIMAL','INFEASIBLE'));self.assertGreater(service.stats['retained_gold_cuts'],1)
            self.assertGreater(service.stats['retained_incumbent_invalidations'],0)
            self.assertIsNotNone(candidate);self.assertFalse(p.validate(candidate)['valid'])
            record=service.results[-1];self.assertEqual(record['heuristic'],candidate)
            if record['candidate'] is not None:
                self.assertTrue(service._allowed_by_cuts(record['context'],record['candidate']))
                self.assertGreaterEqual(record['bound'],score(p,record['candidate'])['matched_edges'])
            saved=json.loads(json.dumps(service.snapshot()));tail=RetainedRepairService(p,resume=saved)
            repeated=tail.patch_query([0,3],[0,1],cp_model=self.cp,control=_Control(5,None),seconds=2,seed=1,target='gold')
            self.assertEqual(repeated,(candidate,status));self.assertIsNone(tail.model)
        finally:service.close();tail and tail.close()
    def test_live_validator_board_and_segment_tables_are_bound(self):
        service=RetainedRepairService(self.problem);edges=self.problem.board.edges
        try:
            self.problem.board.edges=tuple(reversed(edges))
            with self.assertRaisesRegex(ValueError,'input or source'):service.snapshot()
            self.problem.board.edges=edges;ports=self.problem._segment_ports
            self.problem._segment_ports=tuple(reversed(ports))
            with self.assertRaisesRegex(ValueError,'input or source'):service.snapshot()
            self.problem._segment_ports=ports
        finally:service.close()

    def test_saved_native_upper_bound_is_installed_after_rebuild(self):
        service=RetainedRepairService(self.problem);tail=None
        try:
            service._build(self.cp,_Control(5,None));solve=service.solver.solve
            def feasible(model):
                status=solve(model);self.assertEqual(status,self.cp.OPTIMAL);return self.cp.FEASIBLE
            with patch.object(service.solver,'solve',side_effect=feasible):candidate,status=self.query(service,[0,3,6,9],[0,1])
            self.assertEqual(status,'FEASIBLE');self.assertTrue(service.pending)
            tail=RetainedRepairService(self.problem,resume=json.loads(json.dumps(service.snapshot())))
            result,state=self.query(tail,[0,3,6,9],[0,1]);self.assertEqual(state,'OPTIMAL')
            self.assertEqual(score(self.problem,result)['matched_edges'],score(self.problem,candidate)['matched_edges'])
            self.assertEqual(tail.stats['retained_upper_bound_reuses'],1)
        finally:service.close();tail and tail.close()
