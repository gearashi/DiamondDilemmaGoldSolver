import tempfile
import unittest
from pathlib import Path
from search_problem import SearchProblem
from constraint_dfs import solve as dfs, _solve_core
from constraint_models import solve as native
from constraint_sat_session import solve as session_solve
from constraint_regions_round26 import KEYS
from test_constraint_dfs import simple_tiles, multiple_loops, double_face, brute
from test_constraint_separator import permutation_ring
from test_dependencies import HAS_SAT, HAS_ORTOOLS
from constraint_work import digest

class RegionalIntegrationTests(unittest.TestCase):
    def test_dfs_each_and_combined_preserve_independent_gold_oracle(self):
        problems=[SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                  permutation_ring([(1,0,2),(0,2,1)]*2)]
        policies=[{key:True} for key in KEYS]+[dict.fromkeys(KEYS,True)]
        for problem in problems:
            for target in ('gold','edges'):
                answers=brute(problem,target)
                for policy in policies:
                    with self.subTest(n=problem.n,target=target,policy=policy):
                        result=dfs(problem,seconds=5,target=target,options={'native':'python',**policy})
                        self.assertTrue(result['complete'],result['status'])
                        if answers:self.assertIn(tuple(result['codes']),answers)
                        else:self.assertEqual(result['status'],'infeasible')

    def test_native_dag_models(self):
        engines=(['sat'] if HAS_SAT else [])+(['cp-sat','cp-sat-regions','cp-sat-circuit'] if HAS_ORTOOLS else [])
        for engine in engines:
            for tiles in (simple_tiles(),multiple_loops()):
                problem=SearchProblem(tiles,double_face());answers=brute(problem,'gold')
                result=native(problem,engine=engine,seconds=8,options={'regional_compiler':True})
                self.assertTrue(result['complete'],result['status'])
                if answers:self.assertIn(tuple(result['codes']),answers)
                else:self.assertEqual(result['status'],'infeasible')
                self.assertEqual(result['stats']['regional_dag_expanded_histories'],0)
                self.assertGreater(result['stats']['regional_dag_regions'],0)

    @unittest.skipUnless(HAS_SAT,'optional SAT')
    def test_sat_session_region_guard_rebuild_and_restart(self):
        from constraint_sat_session import PersistentSatSession
        from constraint_work import WorkScope
        problem=SearchProblem(simple_tiles(),double_face())
        session=PersistentSatSession(problem,WorkScope.for_problem(problem,'gold'),guard_reuse=True,regional_compiler=True)
        try:
            self.assertTrue(session.prepare(lambda:None,5))
            first_max=session.next_id
            session._activate();old_pool=session._guards
            session.configure_regions(False)
            self.assertIsNone(session._guards)
            self.assertTrue(session.prepare(lambda:None,5))
            session.configure_regions(True)
            self.assertTrue(session.prepare(lambda:None,5))
            self.assertEqual(session.next_id,first_max)
            saved=session.snapshot()
        finally:session.close()
        result=session_solve(problem,seconds=8,resume=saved,options={'regional_compiler':True,'sat_guard_reuse':True,'sat_core_lifting':True})
        self.assertEqual(result['status'],'solved')
        self.assertTrue(problem.validate(result['codes'])['valid'])

class ReasoningIntegrationTests(unittest.TestCase):
    def test_source_bound_archive_load_and_scope_isolation(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        options={'native':'python','lcg':True,'durable_reasoning':True,'lookahead':True,'pair_lookahead':True}
        with tempfile.TemporaryDirectory() as path:
            state={'_reasoning_cache_dir':path}
            first=dfs(problem,seconds=5,options=options,learning_state=state)
            self.assertTrue(first['complete'])
            files=list(Path(path).rglob('*.json'));self.assertTrue(files)
            second=dfs(problem,seconds=5,options=options,learning_state={'_reasoning_cache_dir':path})
            self.assertEqual(second['status'],first['status'])
            self.assertGreater(second['stats']['reasoning_cache_hits'],0)
            self.assertTrue(problem.validate(second['codes'])['valid'])
            original=[sum(1<<c for c in row) for row in problem.domains]
            restricted=list(original);restricted[0]&=-restricted[0]
            other={"_reasoning_cache_dir":path}
            _solve_core(problem,seconds=.05,options=options,_initial_domains=restricted,learning_state=other)
            self.assertNotEqual(state['_runtime'].reasoning_archive.local.context_hash,other['_runtime'].reasoning_archive.local.context_hash)

    def test_interrupted_probe_plan_resumes_from_durable_cpu_archive(self):
        from constraint_lookahead import CompletedProbeCache
        from search_options import BOOLEAN_OPTIONS
        from unittest.mock import patch
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        options={**dict.fromkeys(BOOLEAN_OPTIONS,False),'native':'python','lookahead':True,'pair_lookahead':True,'durable_reasoning':True}
        original=CompletedProbeCache.run;calls=[]
        def probe(owner,*args,**kwargs):
            result=original(owner,*args,**kwargs);calls.append(result);return result
        with tempfile.TemporaryDirectory() as directory:
            state={'_reasoning_cache_dir':directory}
            with patch.object(CompletedProbeCache,'run',probe):
                first=dfs(problem,seconds=8,options=options,learning_state=state,stop=lambda:bool(calls))
            self.assertTrue(calls);self.assertEqual(first['status'],'stopped')
            self.assertTrue(state['_runtime'].probes.plans.records)
            state['_runtime'].close()
            cold={'_reasoning_cache_dir':directory}
            second=dfs(problem,seconds=8,options=options,learning_state=cold,resume=first['checkpoint'])
            self.assertTrue(second['complete']);self.assertIn(tuple(second['codes']),brute(problem,'gold'))
            self.assertGreater(second['stats']['reasoning_cache_hits'],0)
            self.assertGreater(second['stats']['probe_plan_resumes'],0)
            cold['_runtime'].close()

    def test_partial_propagation_queue_survives_cold_resume(self):
        from constraint_dfs import _ArcFilter
        from constraint_native import NativePropagation
        from search_options import BOOLEAN_OPTIONS
        from unittest.mock import patch
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        for mode in ('python','auto'):
            options={**dict.fromkeys(BOOLEAN_OPTIONS,False),'native':mode,'durable_reasoning':True}
            baseline=dfs(problem,seconds=8,options=options)
            method=NativePropagation if baseline['native_backend']=='compiled' else _ArcFilter
            name='propagate' if method is NativePropagation else '__call__';original=getattr(method,name);calls=[]
            def completed(owner,*args,**kwargs):
                answer=original(owner,*args,**kwargs);calls.append(1);return answer
            with tempfile.TemporaryDirectory() as directory:
                state={'_reasoning_cache_dir':directory}
                with patch.object(method,name,completed):
                    first=dfs(problem,seconds=8,options=options,learning_state=state,stop=lambda:bool(calls))
                self.assertEqual(first['status'],'stopped');self.assertTrue(state['_runtime'].propagation_cursors)
                state['_runtime'].close();cold={'_reasoning_cache_dir':directory}
                second=dfs(problem,seconds=8,options=options,learning_state=cold,resume=first['checkpoint'])
                self.assertTrue(second['complete']);self.assertIn(tuple(second['codes']),brute(problem,'gold'))
                self.assertGreater(second['stats']['propagation_cursor_resumes'],0)
                self.assertGreater(second['stats']['reasoning_cache_hits'],0)
                self.assertEqual(second['stats']['python_arcs_checked'],baseline['stats']['python_arcs_checked'])
                cold['_runtime'].close()

    def test_disabling_local_reasoning_does_not_write_cache(self):
        problem=SearchProblem(simple_tiles(),double_face())
        with tempfile.TemporaryDirectory() as path:
            dfs(problem,seconds=5,learning_state={'_reasoning_cache_dir':path},options={'native':'python'})
            self.assertEqual(list(Path(path).iterdir()),[])

@unittest.skipUnless(HAS_SAT,'optional SAT service')
class ServiceIntegrationTests(unittest.TestCase):
    def test_public_dfs_creates_parent_owner_and_local_projection_helper(self):
        from constraint_sat_service_round26 import SATServiceClient
        from constraint_sat_assist import SatAssist
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        state={}
        result=dfs(problem,seconds=8,options={'native':'python','shared_sat_service':True,
            'higher_order_relations':True,'relational_consistency':True},learning_state=state)
        self.assertEqual(result['status'],'solved')
        self.assertTrue(problem.validate(result['codes'])['valid'])
        runtime=state['_runtime']
        self.assertIsInstance(runtime.sat,SATServiceClient)
        self.assertIsInstance(runtime.higher_sat,SatAssist)
        self.assertNotIn('_sat_service',state)
        runtime.close()

    def test_parallel_and_symbolic_workers_use_parent_service(self):
        from constraint_parallel import solve as parallel
        from constraint_coordinator_round26 import solve as coordinated
        problem=permutation_ring([(1,0,2),(0,2,1)]*2)
        for function in (parallel,coordinated):
            result=function(problem,seconds=12,workers=2,options={'native':'python','shared_sat_service':True,'sat_service_instances':2})
            self.assertEqual(result['status'],'solved',result.get('error'))
            self.assertTrue(problem.validate(result['codes'])['valid'])

    def test_warm_service_selection_and_scope(self):
        from constraint_sat_service_round26 import SATServiceOwner,SATServiceClient
        problem=SearchProblem(simple_tiles(),double_face())
        with SATServiceOwner(problem) as owner:
            state={'_sat_service':owner.client()}
            options={'native':'python','shared_sat_service':True}
            first=dfs(problem,seconds=4,options=options,learning_state=state)
            self.assertTrue(first['complete'])
            self.assertIsInstance(state['_runtime'].sat,SATServiceClient)
            second=dfs(problem,seconds=4,options={**options,'shared_sat_service':False},learning_state=state)
            self.assertTrue(second['complete'])
            self.assertNotIsInstance(state['_runtime'].sat,SATServiceClient)
            state['_runtime'].close();state['_sat_service'].close()

if __name__=='__main__':unittest.main()
