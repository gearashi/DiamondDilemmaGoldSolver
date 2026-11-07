"""Independent consumer verification, durable claim files and disjoint policies."""
import copy,json,threading,unittest
from unittest.mock import patch
import constraint_parallel as parallel
import constraint_search as runner
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_deep import CertificateStore
from search_options import normalize_options
from test_constraint_sat_assist import GoldFixtureOracle,HAS_SAT
from test_constraint_sat_domains import fixture
from test_constraint_parallel import OPTIONS,legacy_root
from test_constraint_pair_lookahead import turn_problem
import test_constraint_search as runner_fixtures


class TransportTests(unittest.TestCase):
    def test_worker_reuses_only_independently_verified_global_masks(self):
        problem,_=fixture();source=VerifiedMaskConflictStore(problem)
        self.addCleanup(source.close);source.offer([(0,3),(1,3)])
        initial=list(source.initial);payload=source.snapshot()
        options=normalize_options({**OPTIONS,'smart_learning':True,'proof_certificates':True})
        cert=CertificateStore(problem);scope=initial.copy();scope[0]=scope[1]=3
        self.assertTrue(cert.capture(scope))
        requests=[{'id':job,'region_key':str(job),'target':'gold','options':options,
                   'checkpoint':{},'seconds':.1,'mask_conflicts':payload if job==1 else None,
                   'certificates':cert.snapshot() if job==1 else None,'regional_cache_dir':None} for job in (1,2)]
        class Connection:
            def __init__(self):self.items=iter([*requests,None]);self.sent=[]
            def recv(self):return next(self.items)
            def send(self,item):self.sent.append(item)
            def close(self):pass
        connection=Connection();states=[];stores=[]
        def factory(p,target,**kwargs):return VerifiedMaskConflictStore(p,target,factory=lambda:GoldFixtureOracle(p,initial),**kwargs)
        def fake_solve(p,**kwargs):
            state=kwargs['learning_state'];states.append(state);store=state['_mask_conflicts'];stores.append(store)
            if len(states)==1:
                self.assertFalse(store.records());self.assertEqual(store.pending_count,1)
            store.advance(seconds=1)
            self.assertEqual(len(store),1);self.assertEqual(len(state['_certificates'].records),1)
            return {'status':'timeout','nodes':0,'stats':{}}
        with patch.object(parallel,'VerifiedMaskConflictStore',side_effect=factory),patch.object(parallel,'dfs_solve',side_effect=fake_solve):
            parallel._worker(problem,connection,threading.Event())
        self.assertIsNot(states[0],states[1]);self.assertIs(stores[0],stores[1])
        self.assertTrue(all(item['result']['mask_conflicts']['records'] for item in connection.sent))
        self.assertTrue(all(item['result']['certificates']['records'] for item in connection.sent))

    def test_auxiliary_snapshot_restores_claims_and_rejects_cross_scope(self):
        problem,_=fixture();store=VerifiedMaskConflictStore(problem)
        self.addCleanup(store.close);store.offer([(0,3),(1,3)])
        certificates=CertificateStore(problem);domains=list(store.initial);domains[0]=domains[1]=3;certificates.capture(domains)
        options={**OPTIONS,'smart_learning':True,'proof_certificates':True}
        result=parallel.solve(problem,seconds=0,options=options,learning_state={'_mask_conflicts':store,'_certificates':certificates})
        cp=result['checkpoint'];self.assertTrue(cp['mask_conflicts']['records']);self.assertTrue(cp['certificates']['records'])
        restored={};again=parallel.solve(problem,seconds=0,options=options,resume=json.loads(json.dumps(cp)),learning_state=restored)
        self.assertFalse(restored['_mask_conflicts'].records())
        self.assertEqual(restored['_mask_conflicts'].pending_count,1)
        self.assertEqual(len(restored['_certificates'].records),1)
        restored['_mask_conflicts'].close()
        forged=copy.deepcopy(cp);forged['mask_conflicts']['target']='edges';forged['digest']=parallel._digest(forged)
        with self.assertRaises(ValueError):parallel.solve(problem,seconds=0,options=options,resume=forged)

    def test_priority_selection_overrides_warm_affinity_without_dropping_jobs(self):
        queue=[3,2,1];owners={1:0};last={0:(1,1)}
        selected,hit=parallel._choose_job(queue,0,owners,last,priority=True)
        self.assertEqual(selected,3);self.assertFalse(hit);self.assertEqual(queue,[2,1])
        selected,_=parallel._choose_job(queue,0,owners,last)
        self.assertIn(selected,(1,2));self.assertEqual(len(queue),1)

    def test_real_spawn_portfolio_and_discrepancy_preserve_exhaustion(self):
        problem=turn_problem(False)
        options={**OPTIONS,'frontier_restarts':True,'worker_portfolio':True,'limited_discrepancy':True}
        answer=parallel.solve(problem,seconds=15,workers=2,resume=legacy_root(problem),options=options)
        self.assertEqual(answer['status'],'infeasible',answer)
        self.assertTrue(answer['complete']);self.assertGreater(answer['stats']['portfolio_jobs'],0)
        self.assertGreater(answer['stats']['discrepancy_jobs'],0)
        self.assertGreater(answer['stats']['parallel_workers_started'],0)


class DurableRunnerTests(unittest.TestCase):
    setUp=runner_fixtures.RunnerTests.setUp
    args=runner_fixtures.RunnerTests.args
    read=runner_fixtures.RunnerTests.read
    terminal=runner_fixtures.RunnerTests.terminal
    assert_released=runner_fixtures.RunnerTests.assert_released

    def test_final_claims_and_certificates_persist_and_reload_as_verified_or_pending(self):
        masks=VerifiedMaskConflictStore(self.problem);self.addCleanup(masks.close)
        masks.offer([(0,3),(1,3)])
        certificates=CertificateStore(self.problem);scope=list(masks.initial);scope[0]=scope[1]=3
        self.assertTrue(certificates.capture(scope))
        options={'smart_learning':True,'proof_certificates':True,'extended_regions':True}
        first=self.terminal('stopped',mask_conflicts=masks.snapshot(),certificates=certificates.snapshot())
        with patch.object(runner,'run_solver',return_value=first):
            runner.main(self.args('dfs','--search-options',json.dumps(options)))
        saved=self.read('mask-conflicts-gold.json');self.assertTrue(saved['records'])
        self.assertTrue(self.read('certificates-gold.json')['records'])
        def resumed(problem,algorithm,**kwargs):
            state=kwargs['learning_state'];self.assertEqual(state['_mask_conflicts'].pending_count,1)
            self.assertFalse(state['_mask_conflicts'].records())
            self.assertEqual(len(state['_certificates'].records),1)
            self.assertEqual(state['_regional_cache_dir'],str(self.run/'regional-tables'))
            return self.terminal('stopped')
        with patch.object(runner,'run_solver',side_effect=resumed):
            runner.main(self.args('dfs','--resume','--search-options',json.dumps(options)))
        self.assert_released()

    def test_malformed_optional_files_do_not_discard_authoritative_checkpoint(self):
        self.run.mkdir();(self.run/'mask-conflicts-gold.json').write_text('{}')
        (self.run/'certificates-gold.json').write_text('{}')
        with patch.object(runner,'run_solver',return_value=self.terminal('stopped')):
            runner.main(self.args('dfs','--search-options',json.dumps({'smart_learning':True,'proof_certificates':True})))
        state=self.read('status.json');self.assertEqual(state['state'],'stopped')
        self.assertIn('mask_conflicts_warning',state);self.assertIn('certificates_warning',state)
        self.assertEqual(self.read('searches/dfs-gold/checkpoint.json')['payload'],{'saved':'final'})

class SnapshotSizeTests(unittest.TestCase):
    def test_maximum_width_claims_fit_own_durable_reader(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from constraint_mask_conflicts import MAX_BYTES,MAX_RECORDS,MAX_LITERALS
        problem=SimpleNamespace(n=160,fingerprint='maximum-width-fixture',domains=[list(range(480)) for _ in range(160)])
        source=VerifiedMaskConflictStore(problem);consumer=VerifiedMaskConflictStore(problem)
        self.addCleanup(source.close);self.addCleanup(consumer.close)
        root=(1<<480)-1
        for index in range(MAX_RECORDS):
            self.assertTrue(source.offer([(cell,root^(1<<((index+cell)%480))) for cell in range(MAX_LITERALS)]))
        payload=source.snapshot()
        self.assertGreater(len(payload['records']),0)
        self.assertLess(len(payload['records']),MAX_RECORDS)
        self.assertLessEqual(len(json.dumps(payload,indent=2).replace('\n','\r\n').encode()),MAX_BYTES)
        self.assertEqual(consumer.import_snapshot(payload),len(payload['records']))
        self.assertFalse(consumer.records())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.json';source.save(path)
            self.assertLessEqual(path.stat().st_size,MAX_BYTES)
            restored=VerifiedMaskConflictStore(problem);self.addCleanup(restored.close)
            self.assertEqual(restored.load(path),len(payload['records']))
            self.assertFalse(restored.records())

class CanonicalMaskResumeTests(unittest.TestCase):
    def test_pending_to_ready_and_zero_budget_validation_keep_exact_scope(self):
        import constraint_dfs as dfs
        from constraint_symmetry import canonical_root_domains
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        data,board,reference=planted_gold(23941,1)
        problem=SearchProblem(data,board);options={**OPTIONS,'smart_learning':True}
        initial=[sum(1<<c for c in row) for row in problem.domains]
        canonical,reduction=canonical_root_domains(problem)
        self.assertNotEqual(canonical,initial)
        pending=dfs.solve(problem,seconds=0,options=options)['checkpoint']
        self.assertEqual(pending['root_symmetry']['state'],'pending')
        self.assertEqual(pending['local_mask_conflicts']['initial'],[hex(v) for v in initial])
        # Deterministically finish canonical setup and stop before searching:
        # no wall-time threshold is part of this regression witness.
        original_core=dfs._solve_core
        def no_search(p,**kwargs):
            kwargs['seconds']=0
            return original_core(p,**kwargs)
        with patch.object(dfs,'_solve_core',side_effect=no_search):
            ready=dfs.solve(problem,seconds=10,options=options,resume=pending)['checkpoint']
        self.assertEqual(ready['root_symmetry']['state'],'ready')
        self.assertEqual(ready['local_mask_conflicts']['initial'],[hex(v) for v in canonical])
        self.assertEqual(ready['nodes'],0)
        self.assertEqual(ready['stack'][0]['domains'],[hex(v) for v in canonical])
        source=copy.deepcopy(ready)
        validated=dfs.solve(problem,seconds=0,options=options,resume=ready)['checkpoint']
        self.assertEqual(ready,source)
        self.assertEqual(validated['stack'],ready['stack'])
        self.assertEqual(validated['local_mask_conflicts'],ready['local_mask_conflicts'])
        # Descendants retain optional claims, but each must bind to its own
        # root and independently verify them. Parent certificates are not relabelled.
        children=parallel.partition_checkpoint(problem,validated,parts=2,options=options)
        self.assertEqual(len(children),2)
        for child in children:
            self.assertEqual(child['local_mask_conflicts']['initial'],child['owned_root_domains'])
            self.assertEqual(child['local_mask_conflicts']['records'],ready['local_mask_conflicts']['records'])
            checked=dfs.solve(problem,seconds=0,options=options,resume=child)['checkpoint']
            consumer=VerifiedMaskConflictStore(problem,initial=[int(v,16) for v in child['owned_root_domains']])
            self.addCleanup(consumer.close)
            consumer.import_snapshot(checked['local_mask_conflicts'])
            self.assertFalse(consumer.records())
