"""Independent finite checks for publication of globally re-proved conflicts."""
import copy,itertools,json,threading,unittest
from unittest.mock import patch
from types import SimpleNamespace
from constraint_shared_conflicts import VerifiedConflictStore,_digest,_disk_bytes,MAX_BYTES,MAX_PENDING,MAX_ATTEMPTS
from constraint_sat_assist import SatAssist
from test_constraint_sat_assist import Fake,HAS_SAT
from test_constraint_dfs import simple_tiles,double_face,tetrahedron_tiles,brute,multiple_loops
from test_constraint_separator import ring
from search_problem import SearchProblem


def finish(store):
    for _ in range(30):
        store.advance(seconds=.1,attempts=4,conflicts=10000)
        if not store.pending_count:return
        if store.status in ('unavailable','unsupported'):return
    raise AssertionError('Tiny verification queue did not finish')


def snapshot_claims(problem,records,target='gold'):
    store=VerifiedConflictStore(problem,target);payload=store.snapshot();store.close()
    payload['records']=records;payload['digest']=_digest(payload);return payload


class SharedConflictSafetyTests(unittest.TestCase):
    def test_offering_and_importing_a_claim_never_publishes_it(self):
        problem=ring(4);store=VerifiedConflictStore(problem)
        try:
            self.assertTrue(store.offer([(0,0),(1,0)]));self.assertEqual(store.records(),[])
            self.assertEqual(store.snapshot()['records'],[]);self.assertEqual(store.pending_count,1)
            receiver=VerifiedConflictStore(problem)
            try:
                payload=snapshot_claims(problem,[[[0,0],[1,0]]])
                self.assertEqual(receiver.import_snapshot(payload),1);self.assertEqual(receiver.records(),[])
                self.assertEqual(receiver.advance(seconds=0),0);self.assertEqual(receiver.records(),[])
            finally:receiver.close()
        finally:store.close()

    def test_malformed_batch_is_atomic_and_bad_digests_or_bindings_are_rejected(self):
        problem=ring(4);store=VerifiedConflictStore(problem)
        try:
            for change in ({'digest':'forged'},{'target':'edges'},{'fingerprint':'other'},
                           {'domains_sha256':'other'},{'shared_conflicts_version':True},
                           {'records':[[[0,0]],[[True,0]]]}):
                payload=snapshot_claims(problem,[[[0,0],[1,0]]]);payload.update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):store.import_snapshot(payload)
                self.assertEqual(store.pending_count,0);self.assertEqual(store.records(),[])
            for record in (None,{},[[0,0],[0,3]],[[0,999]],[[False,0]],[[0,True]]):
                with self.subTest(record=record),self.assertRaises(ValueError):store.offer(record)
            other=ring(4);other.domains[0].pop()
            self.assertFalse(store.matches(other));self.assertFalse(store.matches(problem,'edges'))
            problem.domains[0].pop()
            with self.assertRaisesRegex(ValueError,'changed'):store.advance(seconds=.1)
            with self.assertRaisesRegex(ValueError,'changed'):store.snapshot()
            with self.assertRaisesRegex(ValueError,'changed'):store.records()
        finally:store.close()

    def test_missing_partial_and_cancelled_verifier_never_exports(self):
        problem=ring(4)
        def absent():raise ImportError('SAT unavailable')
        missing=VerifiedConflictStore(problem,factory=absent);missing.offer([(0,0),(1,0)])
        try:
            self.assertEqual(missing.advance(seconds=.05),0);self.assertEqual(missing.status,'unavailable')
            self.assertEqual(missing.records(),[]);self.assertEqual(missing.pending_count,1)
        finally:missing.close()
        store=VerifiedConflictStore(problem,factory=Fake);store.offer([(0,0),(1,0)])
        calls=[0]
        def stop():
            calls[0]+=1
            if calls[0]>=3:raise InterruptedError('stop')
        try:
            with self.assertRaises(InterruptedError):store.advance(stop,seconds=.1)
            self.assertEqual(store.records(),[]);self.assertEqual(store.pending_count,1)
            self.assertFalse(store._oracle.ready)
        finally:store.close()

    def test_unknown_attempts_are_bounded_and_interrupted_false_never_proves(self):
        class LateFalse(Fake):
            def solve_limited(self,**kwargs):
                self.called=True
                if not self.ended.wait(5):raise AssertionError('Verifier monitor did not interrupt')
                return False
            def get_core(self):raise AssertionError('Interrupted answer cannot provide proof')
        store=VerifiedConflictStore(ring(4),factory=LateFalse);store.offer([(0,0),(1,0)])
        # Freeze preparation/main-thread time, then expire the deadline in the
        # real monitor thread. Each call must complete exactly one interrupted
        # query, independently of host load or how long model preparation takes.
        caller=threading.get_ident()
        clock=SimpleNamespace(monotonic=lambda:0.0 if threading.get_ident()==caller else 1.0)
        try:
            with patch('constraint_shared_conflicts.time',clock),patch('constraint_sat_assist.time',clock):
                for attempt in range(MAX_ATTEMPTS):
                    self.assertEqual(store.advance(seconds=.01,attempts=1),0)
                    self.assertEqual(store.stats['shared_conflicts_calls'],attempt+1)
                    self.assertEqual(store.stats['shared_conflicts_unknown'],attempt+1)
                    self.assertEqual(store.pending_count,int(attempt+1<MAX_ATTEMPTS))
            self.assertEqual(store.records(),[]);self.assertEqual(store.pending_count,0)
            self.assertEqual(store.stats['shared_conflicts_unknown'],MAX_ATTEMPTS)
            self.assertEqual(store.stats['shared_conflicts_pending_dropped'],1)
            self.assertEqual(store._oracle.stats['sat_assist_unsat'],0)
            self.assertFalse(store.offer([(0,0),(1,0)]))
        finally:store.close()

    def test_preparation_budget_expiry_does_not_consume_a_query_attempt(self):
        store=VerifiedConflictStore(ring(4),factory=Fake);store.offer([(0,0),(1,0)])
        try:
            # The enclosing budget expires before preparation begins. No query
            # occurred, so this claim must retain all of its retry allowance.
            ticks=iter((0.0,.01))
            with patch('constraint_shared_conflicts.time',SimpleNamespace(monotonic=lambda:next(ticks))), \
                 patch('constraint_sat_assist.time',SimpleNamespace(monotonic=lambda:0.0)):
                self.assertEqual(store.advance(seconds=.01,attempts=1),0)
            self.assertFalse(store._oracle.ready);self.assertFalse(store._oracle.solver.called)
            self.assertEqual(store.pending_count,1);self.assertEqual(store.records(),[])
            self.assertEqual(list(store._pending.values()),[0])
            for name in ('shared_conflicts_calls','shared_conflicts_unknown','shared_conflicts_pending_dropped'):
                self.assertEqual(store.stats[name],0)
        finally:store.close()

    def test_native_stop_retains_unverified_pending_claim(self):
        fake=Fake();store=VerifiedConflictStore(ring(4),factory=lambda:fake)
        store.offer([(0,0),(1,0)])
        def stop():
            if fake.called:raise InterruptedError('native stop')
        try:
            with self.assertRaises(InterruptedError):store.advance(stop,seconds=.1)
            self.assertEqual(store.records(),[]);self.assertEqual(store.pending_count,1)
        finally:store.close()

    def test_partial_model_clause_cap_never_proves_any_record(self):
        problem=ring(4);store=VerifiedConflictStore(problem,factory=Fake)
        store.offer([(0,0),(1,0)])
        store._oracle=SatAssist(problem,store._initial,factory=Fake)
        store._oracle.clauses=1000000
        try:
            self.assertEqual(store.advance(seconds=.1),0)
            self.assertEqual(store.status,'unsupported');self.assertEqual(store.records(),[])
            self.assertFalse(store._oracle.ready);self.assertEqual(store.pending_count,1)
        finally:store.close()

    def test_pending_and_export_bounds_and_snapshot_ownership(self):
        problem=ring(34);store=VerifiedConflictStore(problem)
        try:
            self.assertFalse(store.offer([(cell,0) for cell in range(33)]))
            for a,b in itertools.islice(itertools.combinations(range(34),2),MAX_PENDING+9):store.offer([(a,0),(b,0)])
            self.assertEqual(store.pending_count,MAX_PENDING)
            self.assertEqual(store.stats['shared_conflicts_pending_dropped'],10)
            payload=store.snapshot();payload['records'].append([])
            self.assertEqual(store.records(),[]);self.assertLessEqual(_disk_bytes(store.snapshot()),MAX_BYTES)
            for kwargs in ({'seconds':float('nan')},{'seconds':1},{'seconds':True},{'attempts':0},{'attempts':True},{'conflicts':0}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):store.advance(**kwargs)
        finally:store.close()


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class SharedConflictNativeTests(unittest.TestCase):
    def assert_sound(self,store,solutions):
        for record in store.records():
            self.assertFalse(any(all(board[cell]==code for cell,code in record) for board in solutions))

    def test_tile_inventory_conflict_and_worker_roundtrip_are_reproved(self):
        problem=ring(4);solutions=brute(problem,'edges');store=VerifiedConflictStore(problem)
        receiver=VerifiedConflictStore(problem)
        try:
            for placement in ((0,0),(1,0)):
                self.assertTrue(any(board[placement[0]]==placement[1] for board in solutions))
            store.offer([(0,0),(1,0)]);finish(store)
            self.assertEqual(store.records(),[[[0,0],[1,0]]]);self.assert_sound(store,solutions)
            payload=json.loads(json.dumps(store.snapshot()))
            self.assertEqual(receiver.import_snapshot(payload),1);self.assertEqual(receiver.records(),[])
            finish(receiver);self.assertEqual(receiver.records(),store.records())
            calls=receiver.stats['shared_conflicts_calls']
            self.assertEqual(receiver.import_snapshot(payload),0);self.assertEqual(receiver.pending_count,0)
            self.assertEqual(receiver.stats['shared_conflicts_calls'],calls)
        finally:store.close();receiver.close()

    def test_non_loop_seam_contradictions_preserve_all_independent_solutions(self):
        problem=SearchProblem(simple_tiles(),double_face());solutions=brute(problem,'edges')
        store=VerifiedConflictStore(problem)
        try:
            # Each orientation can occur somewhere in a valid board, but the
            # same rotation is required on these opposite matching triangles.
            clause=[(0,0),(1,4)]
            for cell,code in clause:self.assertTrue(any(board[cell]==code for board in solutions))
            self.assertFalse(any(all(board[c]==v for c,v in clause) for board in solutions))
            self.assertFalse(problem.partial_loop_impossible([0,4]))
            store.offer(clause);finish(store);self.assertGreater(len(store),0);self.assert_sound(store,solutions)
            tetra=SearchProblem(*tetrahedron_tiles());expected=brute(tetra,'edges');second=VerifiedConflictStore(tetra)
            try:
                for cell,row in enumerate(tetra.domains):
                    for code in row:second.offer([(cell,code)])
                finish(second);self.assertGreater(len(second),0);self.assert_sound(second,expected)
                self.assertGreater(second.stats['shared_conflicts_rejected'],0)
            finally:second.close()
        finally:store.close()

    def test_restricted_root_only_private_core_is_not_globally_valid(self):
        problem=ring(4);initial=[sum(1<<code for code in row) for row in problem.domains];initial[0]=1
        private=SatAssist(problem,initial);store=VerifiedConflictStore(problem)
        try:
            self.assertTrue(private.prepare(seconds=.2))
            core=private.query([(1,0)],seconds=.1,conflicts=10000)
            self.assertEqual(core,((1,0),));self.assertEqual(private.drain_candidates(),[core])
            self.assertEqual(private.drain_candidates(),[])
            store.offer(core);finish(store)
            self.assertEqual(store.records(),[]);self.assertEqual(store.stats['shared_conflicts_rejected'],1)
            self.assertTrue(any(board[1]==0 for board in brute(problem,'edges')))
        finally:private.close();store.close()

    def test_forged_but_correctly_rehashed_snapshot_cannot_publish_sat_claim(self):
        problem=ring(4);store=VerifiedConflictStore(problem)
        try:
            payload=snapshot_claims(problem,[[[0,0]],[[0,0],[1,0]]])
            self.assertEqual(store.import_snapshot(payload),2);self.assertEqual(store.records(),[])
            finish(store);self.assertEqual(store.records(),[[[0,0],[1,0]]])
            self.assertEqual(store.stats['shared_conflicts_rejected'],1)
        finally:store.close()

    def test_multiloop_gold_failure_is_not_a_seam_unsat_proof(self):
        problem=SearchProblem(multiple_loops(),double_face());store=VerifiedConflictStore(problem)
        try:
            self.assertFalse(brute(problem,'gold'));self.assertTrue(brute(problem,'edges'))
            store.offer([]);finish(store)
            self.assertEqual(store.records(),[]);self.assertEqual(store.stats['shared_conflicts_rejected'],1)
        finally:store.close()

    def test_global_unsat_empty_core_and_completed_proofs_survive_cancel(self):
        data=simple_tiles();data[1]['segments']=[[[0,2],[1,4]]]
        impossible=SearchProblem(data,double_face());store=VerifiedConflictStore(impossible)
        try:
            self.assertFalse(brute(impossible,'edges'));store.offer([]);finish(store)
            self.assertEqual(store.records(),[[]])
        finally:store.close()
        live=VerifiedConflictStore(ring(4))
        try:
            live.offer([(0,0),(1,0)]);finish(live);before=live.snapshot()
            live.offer([(2,3),(3,3)])
            def stop():raise InterruptedError('stop')
            with self.assertRaises(InterruptedError):live.advance(stop,seconds=.1)
            self.assertEqual(live.snapshot(),before);self.assertEqual(live.pending_count,1)
        finally:live.close()

    def test_subsumption_record_limit_and_disk_byte_bound(self):
        problem=ring(4);store=VerifiedConflictStore(problem,limit=1);solutions=brute(problem,'edges')
        try:
            store.offer([(0,0),(1,0),(2,3)]);finish(store)
            self.assertEqual(store.records(),[[[0,0],[1,0]]])
            self.assertFalse(store.offer([(0,0),(1,0),(3,6)]))
            store.offer([(2,3),(3,3)]);finish(store)
            self.assertEqual(len(store),1);self.assertEqual(store.stats['shared_conflicts_evictions'],1)
            self.assert_sound(store,solutions)
            with patch('constraint_shared_conflicts.MAX_BYTES',400):
                store.offer([(0,6),(3,6)]);finish(store)
                self.assertLessEqual(_disk_bytes(store.snapshot()),400)
        finally:store.close()


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class SharedConflictDFSIntegrationTests(unittest.TestCase):
    def options(self,**changes):
        from search_options import BOOLEAN_OPTIONS
        options={key:False for key in BOOLEAN_OPTIONS}
        options.update(reason_tracking=True,backjumping=True,shared_conflicts=True,native='python')
        options.update(changes);return options

    def test_preverified_nonempty_records_are_normalized_into_private_learning(self):
        import constraint_dfs as dfs
        problem=ring(4);shared=VerifiedConflictStore(problem)
        try:
            shared.offer([(0,0),(1,0)]);finish(shared)
            cache={'_shared_conflicts':shared}
            result=dfs.solve(problem,seconds=5,options=self.options(),learning_state=cache)
            self.assertEqual(result['status'],'solved');self.assertTrue(problem.validate(result['codes'])['valid'])
            self.assertIn(frozenset({(0,0),(1,0)}),cache['_nogoods'].clauses)
            self.assertEqual(result['shared_conflicts']['records'],[[[0,0],[1,0]]])
        finally:shared.close()

    def test_claim_verified_during_dfs_is_normalized_before_application(self):
        import constraint_dfs as dfs
        problem=ring(4);shared=VerifiedConflictStore(problem)
        try:
            shared.import_snapshot(snapshot_claims(problem,[[[0,0],[1,0]]]))
            self.assertEqual(shared.records(),[])
            # Build only the oracle before DFS: the imported claim still has
            # no proof and must be verified through DFS's advance/import path.
            # Bound the preparation and queue completion independently instead
            # of requiring one particular wall-clock slice to finish.
            shared._oracle=SatAssist(problem,shared._initial,'gold')
            for _ in range(30):
                if shared._oracle.prepare(seconds=.25):break
            self.assertTrue(shared._oracle.ready);self.assertEqual(shared.records(),[])
            advance=shared.advance
            def complete_tiny_queue(check,**kwargs):
                verified=0
                for _ in range(30):
                    verified+=advance(check,seconds=.25,attempts=4,conflicts=10000)
                    if not shared.pending_count:return verified
                raise AssertionError('Tiny queue did not finish during DFS integration')
            with patch.object(shared,'advance',side_effect=complete_tiny_queue):
                result=dfs.solve(problem,seconds=5,options=self.options(),learning_state={'_shared_conflicts':shared})
            self.assertEqual(result['status'],'solved');self.assertEqual(shared.records(),[[[0,0],[1,0]]])
            self.assertTrue(problem.validate(result['codes'])['valid'])
        finally:shared.close()

    def test_unverified_forged_global_empty_clause_never_prunes_dfs(self):
        import constraint_dfs as dfs
        problem=ring(4)
        def missing():raise ImportError('verification dependency absent')
        shared=VerifiedConflictStore(problem,factory=missing)
        try:
            shared.import_snapshot(snapshot_claims(problem,[[]]))
            result=dfs.solve(problem,seconds=5,options=self.options(),learning_state={'_shared_conflicts':shared})
            self.assertEqual(result['status'],'solved');self.assertTrue(problem.validate(result['codes'])['valid'])
            self.assertEqual(shared.records(),[]);self.assertEqual(result['shared_conflicts']['records'],[])
            self.assertEqual(shared.pending_count,1)
        finally:shared.close()

    def test_private_restricted_job_failures_are_only_untrusted_global_offers(self):
        import constraint_dfs as dfs
        problem=ring(4);initial=[sum(1<<code for code in row) for row in problem.domains]
        initial[0]=initial[1]=1
        shared=VerifiedConflictStore(problem)
        try:
            # The job root itself is contradictory; an empty local conflict
            # therefore does not establish that the ORIGINAL puzzle is UNSAT.
            result=dfs._solve_core(problem,seconds=5,_initial_domains=initial,
                options=self.options(),learning_state={'_shared_conflicts':shared})
            self.assertEqual(result['status'],'infeasible');self.assertEqual(shared.records(),[])
            self.assertGreater(shared.pending_count,0);finish(shared)
            self.assertEqual(shared.records(),[]);self.assertGreater(shared.stats['shared_conflicts_rejected'],0)
            valid=dfs.solve(problem,seconds=5,options=self.options(),learning_state={'_shared_conflicts':shared})
            self.assertEqual(valid['status'],'solved');self.assertTrue(problem.validate(valid['codes'])['valid'])
        finally:shared.close()

    def test_stop_during_verification_retains_resumable_tree_and_untrusted_queue(self):
        import constraint_dfs as dfs
        problem=ring(4);fake=Fake();shared=VerifiedConflictStore(problem,factory=lambda:fake)
        shared.offer([(0,0),(1,0)])
        try:
            first=dfs.solve(problem,seconds=5,options=self.options(),
                learning_state={'_shared_conflicts':shared},stop=lambda:fake.called)
            self.assertEqual(first['status'],'stopped');self.assertEqual(shared.records(),[])
            self.assertTrue(first['checkpoint']['stack'])
            self.assertNotIn('_reasons',json.dumps(first['checkpoint']))
            self.assertEqual(first['shared_conflicts']['records'],[])
            resumed=dfs.solve(problem,seconds=5,resume=json.loads(json.dumps(first['checkpoint'])),
                options=self.options(shared_conflicts=False))
            self.assertEqual(resumed['status'],'solved');self.assertTrue(problem.validate(resumed['codes'])['valid'])
        finally:shared.close()

    def test_direct_reasons_and_shared_imports_preserve_independent_tiny_solutions(self):
        import constraint_dfs as dfs
        from constraint_reasons import ReasonState
        fixtures=[SearchProblem(*tetrahedron_tiles()),SearchProblem(simple_tiles(),double_face()),
                  SearchProblem(multiple_loops(),double_face())]
        original=dfs._NogoodStore.learn;direct_original=ReasonState.record_cell;seen=[]
        for problem in fixtures:
            for target in ('gold','edges'):
                solutions=brute(problem,target);clauses=[]
                def learned(store,placements,stats):
                    row=tuple(placements);clauses.append(row);return original(store,row,stats)
                def direct(trace,cell,value,deps,rule):
                    count=direct_original(trace,cell,value,deps,rule)
                    if count:seen.append(rule)
                    trace.validate_graph();return count
                options=self.options(native='auto',sat_assist=True,hierarchical_regions=True)
                with patch.object(dfs._NogoodStore,'learn',learned),patch.object(ReasonState,'record_cell',direct):
                    result=dfs.solve(problem,seconds=5,target=target,options=options)
                self.assertTrue(result['complete'])
                if solutions:self.assertIn(tuple(result['codes']),solutions)
                else:self.assertEqual(result['status'],'infeasible')
                for clause in clauses:self.assertFalse(any(all(row[c]==v for c,v in clause) for row in solutions))
                for clause in result['shared_conflicts']['records']:
                    self.assertFalse(any(all(row[c]==v for c,v in clause) for row in solutions))
        self.assertTrue(seen)


if __name__=='__main__':unittest.main()
