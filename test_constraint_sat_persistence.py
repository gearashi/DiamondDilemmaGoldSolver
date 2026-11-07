"""Verified clause persistence: pure checks everywhere, native logging opt-in CI only."""
import copy,itertools,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import constraint_sat_persistence as persistence
from constraint_sat_persistence import VerifiedNativeClauses
from constraint_sat_session import PersistentSatSession,SessionInterrupted,solve
from constraint_work import WorkScope,digest
from test_constraint_sat_domains import fixture
from test_constraint_dfs import brute
from test_constraint_sat_assist import HAS_SAT


def example(max_visits=persistence.MAX_VISITS):
    # a is entailed, but is not a root unit; adding ~a yields b and ~b.
    return VerifiedNativeClauses(((1,2),(1,-2)),({0:1,1:2},),
        fingerprint='finite-cnf',target='edges',initial=(3,),max_visits=max_visits)


def finish(store):
    for _ in range(100000):
        if store.advance(ticks=4096):return
    raise AssertionError('Bounded checker did not finish')


def bundle_for(store,proof):
    return {'version':1,'binding':copy.deepcopy(store.binding),'proof':proof}


def problem_store(problem,scope):
    from constraint_sat_assist import SatAssist
    from constraint_proof import _CaptureSink
    encoder=SatAssist(problem,scope.root,'edges',capture_clauses=True,factory=_CaptureSink)
    try:
        if not encoder.prepare(seconds=5):raise AssertionError('Tiny encoder incomplete')
        base=encoder.proof_base()
    finally:encoder.close()
    return VerifiedNativeClauses(base['clauses'],base['variables'],fingerprint=problem.fingerprint,
                                 target=scope.target,initial=scope.root)


class PurePersistenceTests(unittest.TestCase):
    def test_completed_derivation_survives_serialization_and_checks_again(self):
        first=example();first.queue_import(bundle_for(first,[[1]]));finish(first)
        self.assertEqual(first.projected,[(1,)])
        second=example();second.queue_import(json.loads(json.dumps(first.snapshot())));finish(second)
        self.assertEqual(second.projected,[(1,)])
        self.assertGreater(second.stats['session_persist_verification_visits'],0)
        for values in itertools.product((False,True),repeat=2):
            original=all(any(values[abs(lit)-1]==(lit>0) for lit in row) for row in first.base.clauses)
            strengthened=original and all(any(values[abs(lit)-1]==(lit>0) for lit in row) for row in second.projected)
            self.assertEqual(original,strengthened)

    def test_exact_base_auxiliary_clause_survives_and_preserves_every_assignment(self):
        def make():return VerifiedNativeClauses(((1,3),(-1,3),(-3,2)),({0:1,1:2},),
            fingerprint='aux-cnf',target='edges',initial=(3,))
        first=make();first.queue_import(bundle_for(first,[[3]]));finish(first)
        self.assertFalse(first.projected);self.assertEqual(first.reusable,[(3,)])
        second=make();second.queue_import(json.loads(json.dumps(first.snapshot())));finish(second)
        self.assertEqual(second.stats['session_persist_auxiliary_clauses'],1)
        for values in itertools.product((False,True),repeat=3):
            admitted=lambda rows:all(any(values[abs(lit)-1]==(lit>0) for lit in row) for row in rows)
            self.assertEqual(admitted(second.base.clauses),admitted(second.base.clauses+tuple(second.reusable)))
        payload=bundle_for(make(),[[4]])
        with self.assertRaises(ValueError):make().queue_import(payload)

    def test_root_binding_malformed_and_scope_dependent_claims_rejected(self):
        for mutate in [lambda x:x['binding'].__setitem__('target','gold'),
                       lambda x:x['binding'].__setitem__('base_sha256','forged'),
                       lambda x:x['binding'].__setitem__('variables_sha256','forged'),
                       lambda x:x.__setitem__('proof',[[99]]),lambda x:x.__setitem__('proof',[[True]]),
                       lambda x:x.__setitem__('proof',[[]]),lambda x:x.__setitem__('version',True)]:
            store=example();payload=bundle_for(store,[[1]]);mutate(payload)
            with self.assertRaises(ValueError):store.queue_import(payload)
            self.assertFalse(store.projected)
        store=example();store.queue_import(bundle_for(store,[[2]]))
        with self.assertRaisesRegex(ValueError,'unverified'):finish(store)
        self.assertFalse(store.projected)

    def test_stop_resumes_exact_checker_cursor_and_unknown_never_imports(self):
        stopped=example();stopped.queue_import(bundle_for(stopped,[[1]]));calls=[0]
        def check():
            calls[0]+=1
            if calls[0]==4:raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):stopped.advance(check,ticks=100)
        self.assertTrue(stopped.importing);self.assertFalse(stopped.projected)
        finish(stopped);plain=example();plain.queue_import(bundle_for(plain,[[1]]));finish(plain)
        self.assertEqual(stopped.stats['session_persist_verification_visits'],plain.stats['session_persist_verification_visits'])
        limited=example(max_visits=1);limited.queue_import(bundle_for(limited,[[1]]))
        with self.assertRaisesRegex(ValueError,'verification bound'):finish(limited)
        self.assertFalse(limited.projected)

    def test_native_file_reads_are_incremental_bounded_and_reject_unknown_selectors(self):
        class Fake:pass
        store=example();fake=Fake()
        with tempfile.TemporaryFile() as stream:
            fake.prfile=stream;stream.write(b'99 0\n2 0\n1 0\n');stream.flush();position=stream.tell()
            with patch.object(persistence.os,'pread',create=True,side_effect=lambda fd,n,offset:self._pread(stream,n,offset)):
                store.harvest(fake);finish(store)
                self.assertEqual(stream.tell(),position)
                self.assertEqual(store.projected,[(1,)])
                before=store.stats['session_persist_read_bytes'];store.harvest(fake)
                self.assertEqual(store.stats['session_persist_read_bytes'],before)
                with patch.object(persistence,'MAX_LOG_BYTES',1):self.assertTrue(store.monitor_log(fake))
            self.assertIsNotNone(store.disable_reason)
            self.assertGreaterEqual(store.stats['session_persist_rejected_clauses'],2)
            self.assertLessEqual(before,persistence.READ_BYTES)

    @staticmethod
    def _pread(stream,n,offset):
        old=stream.tell();stream.seek(offset)
        try:return stream.read(n)
        finally:stream.seek(old)

    def test_windows_explicit_logger_rejected_before_factory_creation(self):
        with patch.object(persistence.sys,'platform','win32'),patch('pysat.solvers.Glucose4') as native:
            with self.assertRaisesRegex(ValueError,'Windows'):example().create_solver(((1,2),),logging=True)
            native.assert_not_called()
        with patch.object(persistence.sys,'platform','linux'),patch.object(persistence.platform,'machine',return_value='x86_64'),patch.object(persistence.platform,'python_implementation',return_value='CPython'),patch.object(persistence.sys,'version_info',(3,12)),patch.object(persistence.importlib.metadata,'version',return_value='future'):
            self.assertFalse(persistence.capability()[0])

    def test_pending_import_cannot_be_applied_or_exported_and_payload_cap_is_enforced(self):
        store=example();store.queue_import(bundle_for(store,[[1]]))
        with self.assertRaisesRegex(ValueError,'partially'):store.apply(object())
        with self.assertRaisesRegex(ValueError,'Unverified'):store.snapshot()
        with patch.object(persistence,'MAX_SNAPSHOT_BYTES',10):
            with self.assertRaisesRegex(ValueError,'byte bound'):example().queue_import(bundle_for(example(),[[1]]))


@unittest.skipUnless(HAS_SAT,'ordinary nonlogging SAT package needed')
class SessionImportTests(unittest.TestCase):
    def test_cross_platform_resume_checks_subset_without_starting_logger(self):
        problem,_=fixture();scope=WorkScope.for_problem(problem,'edges');original=PersistentSatSession(problem,scope)
        store=problem_store(problem,scope)
        # A source placement clause is independently RUP; this fixture chiefly
        # verifies source-root reconstruction and the no-logger resume boundary.
        row=next(row for row in store.base.clauses if row and all(abs(v) in store.projection for v in row))
        payload=original.snapshot();payload['native_clauses']=bundle_for(store,[list(row)])
        payload['digest']=digest({k:v for k,v in payload.items() if k!='digest'});original.close()
        before=copy.deepcopy(payload)
        from pysat.solvers import Glucose4
        created=[]
        def factory(*args,**kwargs):
            self.assertFalse(kwargs.get('with_proof',False));created.append(kwargs)
            return Glucose4(*args,**kwargs)
        with patch.object(persistence.sys,'platform','win32'),patch('pysat.solvers.Glucose4',side_effect=factory):
            restored=PersistentSatSession.restore(problem,payload,'edges',native_persistence=False)
            try:
                answer=restored.advance(seconds=5)
                self.assertIn(answer['status'],('edge_perfect','solved'))
                self.assertFalse(restored._native_logging);self.assertTrue(created)
                self.assertEqual(payload,before)
            finally:restored.close()

    def test_unavailable_logging_does_not_replace_existing_session_or_checkpoint(self):
        problem,_=fixture();scope=WorkScope.for_problem(problem,'edges');session=PersistentSatSession(problem,scope)
        state={'_sat_session':session};saved=session.snapshot();before=copy.deepcopy(saved)
        try:
            with patch.object(persistence.sys,'platform','win32'):
                with self.assertRaises(ValueError):solve(problem,target='edges',resume=saved,learning_state=state,options={'native_sat_persistence':True})
            self.assertIs(state['_sat_session'],session);self.assertEqual(saved,before)
        finally:session.close()

    def test_first_public_timeout_retains_unverified_tail_then_checks_it_on_fresh_restore(self):
        class Fake:
            def __init__(self):self.prfile=tempfile.TemporaryFile()
            def add_clause(self,row):pass
            def accum_stats(self):return {'decisions':3}
            def delete(self):self.prfile.close()
        problem,_=fixture();scope=WorkScope.for_problem(problem,'edges');fake=Fake()
        with patch.object(persistence,'require_capability'):
            session=PersistentSatSession(problem,scope,native_persistence=True)
        try:
            def interrupted(*args):
                clause=[-session.variables[0][0],-session.variables[1][0]]
                fake.prfile.write((' '.join(map(str,clause))+' 0\n').encode());fake.prfile.flush()
                raise SessionInterrupted('timeout')
            with patch.object(VerifiedNativeClauses,'create_solver',return_value=fake),patch.object(session,'_solve_once',side_effect=interrupted),patch.object(persistence.os,'pread',create=True,side_effect=lambda fd,n,offset:PurePersistenceTests._pread(fake.prfile,n,offset)):
                first=session.advance(seconds=5)
            self.assertEqual(first['status'],'timeout')
            self.assertFalse(first['checkpoint']['native_clauses']['proof'])
            self.assertTrue(first['checkpoint']['native_clauses']['claims'])
            self.assertEqual(first['stats']['session_persist_verified_clauses'],0)
            saved=json.loads(json.dumps(first['checkpoint']))
        finally:session.close()
        restored=PersistentSatSession.restore(problem,saved,'edges',native_persistence=False)
        try:
            for _ in range(20):
                answer=restored.advance(seconds=5)
                if answer['complete']:break
            self.assertIn(answer['status'],('solved','edge_perfect'))
            self.assertGreater(answer['stats']['session_persist_imported_clauses'],0)
            self.assertGreater(answer['stats']['session_persist_verification_visits'],0)
            self.assertEqual(answer['native_persistence']['mode'],'off')
            self.assertFalse(restored._native_logging)
        finally:restored.close()

    def test_logger_rebuild_preserves_owned_work_gold_cuts_and_checked_subset(self):
        class Fake:
            def __init__(self):self.closed=False;self.rows=[]
            def add_clause(self,row):self.rows.append(tuple(row))
            def accum_stats(self):return {'decisions':7}
            def delete(self):self.closed=True
        problem,_=fixture();session=PersistentSatSession(problem,WorkScope.for_problem(problem,'edges'))
        try:
            session.persistence=problem_store(problem,session.scope)
            session._base_clauses=session.persistence.base.clauses
            session.variables=tuple(dict(row) for row in session.persistence.variables)
            old=Fake();new=Fake();session.solver=old;session._native_logging=True;before=copy.deepcopy(session.boxes)
            with patch.object(session.persistence,'create_solver',return_value=new) as create:session._rebuild_without_logging()
            create.assert_called_once_with(session._base_clauses,logging=False)
            self.assertTrue(old.closed);self.assertIs(session.solver,new);self.assertFalse(session._native_logging)
            self.assertEqual(session.boxes,before);self.assertEqual(session._prior_decisions,7)
        finally:session.close()


NATIVE_ENABLED=os.environ.get('DIAMOND_TEST_NATIVE_SAT_PERSISTENCE')=='1' and persistence.capability()[0]
@unittest.skipUnless(NATIVE_ENABLED,'native proof logging runs only in explicitly enabled supported-platform CI')
class NativePersistenceTests(unittest.TestCase):
    def test_interrupted_native_prefix_is_checked_and_reduces_completed_restart_work(self):
        from pysat.solvers import Glucose4
        pigeons,holes=8,7
        var=lambda p,h:1+p*holes+h
        base=[[var(p,h) for h in range(holes)] for p in range(pigeons)]
        base += [[-var(p,h),-var(q,h)] for h in range(holes) for p in range(pigeons) for q in range(p+1,pigeons)]
        mapping=tuple({h:var(p,h) for h in range(holes)} for p in range(pigeons))
        def make():return VerifiedNativeClauses(base,mapping,fingerprint='inventory-8-7',target='edges',initial=[127]*8)
        store=make();producer=store.create_solver(base);interrupted=False
        try:
            for _ in range(3):
                producer.conf_budget(1);interrupted|=producer.solve_limited(expect_interrupt=True) is None
            with patch.object(producer,'get_proof',side_effect=AssertionError('Unbounded proof read used')):
                for _ in range(10000):
                    store.harvest(producer);finish(store)
                    if store.offset>=os.fstat(producer.prfile.fileno()).st_size:break
            self.assertTrue(interrupted);self.assertGreater(len(store.projected),0)
        finally:producer.delete()
        restored=make();restored.queue_import(json.loads(json.dumps(store.snapshot())));finish(restored)
        results=[]
        for learned in (False,True):
            solver=restored.create_solver(base,logging=False) if learned else Glucose4(bootstrap_with=base)
            try:
                self.assertFalse(solver.solve());results.append(solver.accum_stats())
            finally:solver.delete()
        self.assertLess(results[1]['conflicts'],results[0]['conflicts'],results)
        print(json.dumps({'native_persistence_restart':results,'projected':len(restored.projected),'verification_visits':restored.stats['session_persist_verification_visits']}),flush=True)

    def test_actual_session_logging_and_finite_solution_preservation(self):
        problem,_=fixture();scope=WorkScope.for_problem(problem,'edges');session=PersistentSatSession(problem,scope,native_persistence=True)
        try:
            self.assertTrue(session.prepare(lambda:None,5))
            session._activate();session._solve_once(lambda:None,time.monotonic()+5)
            session._maintain_persistence(lambda:None);saved=json.loads(json.dumps(session.snapshot()))
        finally:session.close()
        restored=PersistentSatSession.restore(problem,saved,'edges',native_persistence=True)
        try:
            for _ in range(100):
                if restored.prepare(lambda:None,5):break
            self.assertIsNotNone(restored.solver)
            seen=[]
            while restored.solver.solve():
                model={lit for lit in restored.solver.get_model() if lit>0}
                codes=tuple(next(code for code,lit in row.items() if lit in model) for row in restored.variables)
                seen.append(codes);restored.solver.add_clause([-restored.variables[c][code] for c,code in enumerate(codes)])
            self.assertEqual(sorted(seen),sorted(brute(problem,'edges')));self.assertEqual(len(seen),72)
        finally:restored.close()


if __name__=='__main__':unittest.main()
