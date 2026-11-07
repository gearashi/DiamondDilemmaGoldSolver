"""Independent finite soundness and reuse discriminators for ordered round25 work."""
import itertools,json,random,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from constraint_sat_session import PersistentSatSession,SessionInterrupted
from constraint_work import WorkScope,frontier_snapshot,restore_frontier,CoverageLedger
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,simple_tiles,multiple_loops,double_face,brute
from test_constraint_sat_assist import HAS_SAT


def core_fixture():
    problem=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(problem)
    boxes=[(1,1,1<<code,scope.root[3]) for code in range(12)]
    return problem,scope,boxes


def run_core_fixture(enabled):
    problem,scope,boxes=core_fixture();session=PersistentSatSession(problem,scope,boxes,core_lifting=enabled)
    try:return session.advance(seconds=5)
    finally:session.close()


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class CoreLiftingTests(unittest.TestCase):
    def test_lifted_proof_removes_sibling_cubes_with_less_total_sat_work(self):
        baseline=run_core_fixture(False);lifted=run_core_fixture(True)
        self.assertEqual(baseline['status'],'infeasible');self.assertEqual(lifted['status'],'infeasible')
        self.assertEqual(baseline['stats']['session_native_calls'],12)
        self.assertGreater(lifted['stats']['session_core_boxes_removed'],0)
        self.assertLess(lifted['stats']['session_native_calls']+lifted['stats']['session_core_verifications'],baseline['stats']['session_native_calls'])
        self.assertEqual(lifted['checkpoint']['frontier']['boxes'],[])

    def test_unknown_or_invalid_lift_cannot_remove_remaining_work(self):
        problem,scope,boxes=core_fixture()
        for verify in (False,None):
            session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
            try:
                with patch.object(session,'_verify_lift',return_value=verify):answer=session.advance(seconds=5)
                self.assertEqual(answer['stats']['session_native_calls'],len(boxes))
                self.assertEqual(answer['stats']['session_core_boxes_removed'],0)
            finally:session.close()
        session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
        try:
            # Forge an overbroad suggestion: the independent verifier must find
            # the valid full-root model and refuse to exclude the whole root.
            with patch.object(session,'_lift_candidate',return_value=scope.root):answer=session.advance(seconds=5)
            self.assertEqual(answer['stats']['session_native_calls'],len(boxes))
            self.assertEqual(answer['stats']['session_core_verified'],0)
        finally:session.close()

    def test_interruption_after_cube_proof_preserves_all_unverified_siblings(self):
        problem,scope,boxes=core_fixture();session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
        try:
            with patch.object(session,'_verify_lift',side_effect=SessionInterrupted('stopped')):answer=session.advance(seconds=5)
            self.assertEqual(answer['status'],'stopped')
            self.assertEqual(restore_frontier(answer['checkpoint']['frontier'],scope),tuple(boxes[1:]))
            restored=PersistentSatSession.restore(problem,json.loads(json.dumps(answer['checkpoint'])))
            try:self.assertEqual(restored.advance(seconds=5)['status'],'infeasible')
            finally:restored.close()
        finally:session.close()

    def test_interrupted_primary_call_keeps_attempt_and_decision_counters(self):
        problem,scope,boxes=core_fixture();session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
        class Native:
            def accum_stats(self):return {'decisions':7}
            def delete(self):pass
        try:
            session.solver=Native();session.selector=()
            with patch.object(session,'_native_call',side_effect=SessionInterrupted('stopped')):
                with self.assertRaises(SessionInterrupted):session._solve_once(lambda:None,time.monotonic()+1)
            self.assertEqual(session.stats['session_native_calls'],1)
            self.assertEqual(session.stats['session_decisions'],7)
            self.assertEqual(session.boxes,boxes)
            self.assertEqual(session.result('stopped',time.monotonic())['nodes'],7)
        finally:session.close()

    def test_restored_verifier_keeps_prior_decisions_and_interrupted_attempts(self):
        problem,scope,boxes=core_fixture();session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
        session.stats.update(session_decisions=17,session_core_decisions=11,session_core_verifications=2)
        saved=session.snapshot();session.close();session=PersistentSatSession.restore(problem,saved)
        class Native:
            def accum_stats(self):return {'decisions':5}
            def delete(self):pass
        try:
            session._core_verifier=Native();session.variables=()
            with patch.object(session,'_native_call',side_effect=SessionInterrupted('timeout')):
                with self.assertRaises(SessionInterrupted):session._verify_lift(scope.root,lambda:None,time.monotonic()+1)
            self.assertEqual(session.stats['session_core_verifications'],3)
            self.assertEqual(session.stats['session_core_decisions'],16)
            self.assertEqual(session.stats['session_decisions'],17)
            answer=session.result('timeout',time.monotonic())
            self.assertEqual(answer['nodes'],33);self.assertEqual(answer['stats']['session_total_decisions'],33)
            self.assertEqual(session.boxes,boxes)
            saved=session.snapshot()
        finally:session.close()
        session=PersistentSatSession.restore(problem,saved)
        try:
            session._core_verifier=Native();session.variables=()
            with patch.object(session,'_native_call',return_value=None):self.assertIsNone(session._verify_lift(scope.root,lambda:None,time.monotonic()+1))
            self.assertEqual(session.stats['session_core_decisions'],21)
            self.assertEqual(session.result('timeout',time.monotonic())['nodes'],38)
        finally:session.close()

    def test_owned_cubes_are_immutable_and_core_policy_survives_restart(self):
        problem,scope,boxes=core_fixture();supplied=[list(row) for row in boxes]
        session=PersistentSatSession(problem,scope,supplied,core_lifting=True)
        try:
            supplied[0][0]=scope.root[0]
            self.assertEqual(session.boxes,boxes)
            restored=PersistentSatSession.restore(problem,session.snapshot())
            try:
                self.assertTrue(restored.core_lifting)
                self.assertEqual(restored.boxes,boxes)
            finally:restored.close()
        finally:session.close()

    def test_every_queue_answer_matches_independent_board_enumeration(self):
        rng=random.Random(2501)
        for target in ('edges','gold'):
            problem=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(problem,target)
            solutions=brute(problem,target)
            for _ in range(20):
                # Distinct singleton prefixes form disjoint cubes. Two other
                # cells retain broad orientation domains and nontrivial seams.
                prefixes=rng.sample(list(itertools.product(range(12),repeat=2)),8)
                boxes=[(1<<a,1<<b,scope.root[2],scope.root[3]) for a,b in prefixes]
                expected=[row for row in solutions if any(all(mask>>code&1 for mask,code in zip(box,row)) for box in boxes)]
                session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
                try:answer=session.advance(seconds=5)
                finally:session.close()
                self.assertTrue(answer['complete'])
                if expected:self.assertIn(tuple(answer['codes']),expected)
                else:self.assertEqual(answer['status'],'infeasible')

    def test_independently_verified_lift_enters_coverage_with_owned_scope(self):
        problem,scope,boxes=core_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'coverage.sqlite3';session=PersistentSatSession(problem,scope,boxes,core_lifting=True)
            session.coverage=CoverageLedger(path,scope)
            try:answer=session.advance(seconds=5)
            finally:session.close()
            with CoverageLedger(path,scope) as ledger:
                excluded=ledger.exclusions();solutions=brute(problem,'gold')
                self.assertTrue(any(any(mask.bit_count()>1 for mask in box) for box in excluded))
                self.assertTrue(all(not any(all(mask>>code&1 for mask,code in zip(box,row)) for row in solutions) for box in excluded))
                self.assertEqual(ledger.remaining(boxes)[0],())

def proof_cursor_fixture():
    from constraint_proof import BoundedDPLL
    clauses=((1,2),(1,-2),(-1,2),(-1,-2))
    producer=BoundedDPLL(clauses);assert producer.solve() is False
    return clauses,['1 2 0']*80+producer.proof


def run_proof_cursor_fixture():
    from constraint_proof import RUPChecker,ProofLimit
    from constraint_proof_cursor import ResumableRUPChecker
    clauses,proof=proof_cursor_fixture();cursor=ResumableRUPChecker(clauses,proof)
    slices=0
    while not cursor.finished:
        cursor.advance(ticks=8);slices+=1
        assert slices<100
    assert cursor.result is True
    # The previous checker starts at line zero for each equally small lease.
    repeats=0
    for _ in range(slices):
        old=RUPChecker(clauses,max_visits=8)
        try:old.verify(proof)
        except ProofLimit:pass
        else:raise AssertionError('The bounded baseline unexpectedly completed.')
        repeats+=old.stats['proof_clause_visits']
    return {'slices':slices,'baseline_clause_visits':repeats,'resumed_clause_visits':cursor.stats['proof_clause_visits'],
            'baseline_completed':False,'resumed_completed':cursor.result}


class ResumableProofTests(unittest.TestCase):
    def test_repeated_small_leases_complete_without_rechecking_prefix(self):
        report=run_proof_cursor_fixture()
        self.assertTrue(report['resumed_completed']);self.assertFalse(report['baseline_completed'])
        self.assertLess(report['resumed_clause_visits'],report['baseline_clause_visits'])

    def test_partial_single_clause_propagation_resumes_exactly(self):
        from constraint_proof import RUPChecker
        from constraint_proof_cursor import ResumableRUPChecker
        clauses=[(i,-(i+1)) for i in range(1,30)]+[(30,),(-1,)]
        old=RUPChecker(clauses);self.assertTrue(old.verify(['0']))
        cursor=ResumableRUPChecker(clauses,['0'])
        for _ in range(2000):
            result=cursor.advance(ticks=1)
            if result is not None:break
        self.assertTrue(result)
        self.assertEqual(cursor.stats['proof_clause_visits'],old.stats['proof_clause_visits'])
        self.assertEqual(cursor.stats['proof_propagations'],old.stats['proof_propagations'])

    def test_finite_formulas_match_independent_truth_table_and_old_checker(self):
        from constraint_proof import BoundedDPLL,RUPChecker
        from constraint_proof_cursor import ResumableRUPChecker
        rng=random.Random(2503)
        for _ in range(120):
            clauses=[tuple(v*rng.choice((-1,1)) for v in rng.sample(range(1,4),rng.randrange(1,4))) for _ in range(8)]
            sat=any(all(any(assignment[abs(lit)-1]==(lit>0) for lit in row) for row in clauses) for assignment in itertools.product((False,True),repeat=3))
            producer=BoundedDPLL(clauses);self.assertEqual(producer.solve(),sat)
            proof=producer.proof or ['0'];cursor=ResumableRUPChecker(clauses,proof)
            for _ in range(500):
                result=cursor.advance(ticks=3)
                if result is not None:break
            self.assertEqual(result,RUPChecker(clauses).verify(proof))
            if result:self.assertFalse(sat)

    def test_stop_caps_malformed_and_mutated_inputs_never_admit_partial_proof(self):
        from constraint_proof_cursor import ResumableRUPChecker
        clauses,proof=proof_cursor_fixture();cursor=ResumableRUPChecker(clauses,proof)
        self.assertIsNone(cursor.advance(ticks=3));before=cursor.stats.copy()
        with self.assertRaises(SessionInterrupted):cursor.advance(lambda:(_ for _ in ()).throw(SessionInterrupted('stopped')),ticks=3)
        self.assertEqual(cursor.stats['proof_clause_visits'],before['proof_clause_visits'])
        self.assertFalse(cursor.finished)
        proof[0]='invalid 0'  # Constructor owns its immutable input.
        while not cursor.finished:cursor.advance(ticks=10)
        self.assertTrue(cursor.result)
        limited=ResumableRUPChecker(clauses,['0'],max_visits=1)
        self.assertIsNone(limited.advance());self.assertTrue(limited.limited)
        for invalid in (['bad 0'],['1'],['d 0 0']):
            self.assertFalse(ResumableRUPChecker(clauses,invalid).advance())

    def test_mask_store_uses_retained_checker_without_native_sat_or_early_pruning(self):
        import copy
        from constraint_proof import MaskProofContext
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        problem=SearchProblem(simple_tiles(),double_face());initial=tuple(sum(1<<c for c in row) for row in problem.domains)
        context=MaskProofContext(problem,initial);store=VerifiedMaskConflictStore(problem,resumable_proof_checking=True)
        try:
            certificate=context.prove([(0,1),(1,1)],seconds=2);self.assertIsNotNone(certificate)
            certificate['proof']=[' '.join(map(str,(*context.base[0],0)))]*600+certificate['proof']
            self.assertTrue(store.offer_certificate(certificate))
            self.assertEqual(store.advance(seconds=1),0);self.assertEqual(len(store),0);self.assertIsNone(store._oracle)
            changed=copy.deepcopy(certificate);changed['target']='edges'
            self.assertFalse(store._proofs().verify_incremental(changed,ticks=256))
            for _ in range(100):
                store.advance(seconds=1)
                if len(store):break
            self.assertEqual(len(store),1);self.assertIsNone(store._oracle)
            self.assertGreater(store.stats['mask_proof_resumptions'],1)
            self.assertEqual(store.stats['mask_proofs_verified'],1)
            self.assertTrue(context.verify(certificate))
        finally:store.close();context.close()

def run_symbolic_fixture():
    from constraint_work import subtract_many
    scope=WorkScope('finite-symbolic-fixture','gold',(3,)*160,(3,)*160)
    excluded=[tuple(1<<((number>>axis)&1) for axis in range(8))+(3,)*152 for number in range(255)]
    with tempfile.TemporaryDirectory() as directory:
        ledger=CoverageLedger(Path(directory)/'coverage.sqlite3',scope,symbolic=True)
        try:
            for box in excluded:ledger.record_completed(box,status='infeasible',complete=True)
            baseline_checks=0
            def tick():
                nonlocal baseline_checks
                baseline_checks+=1
            for _ in range(8):
                baseline,done=subtract_many([scope.root],excluded,check=tick)
                assert done
                actual,done=ledger.remaining([scope.root]);assert done
                assert actual==baseline==((2,)*8+(3,)*152,)
            return {'baseline_subtraction_checks':baseline_checks,**ledger.stats,'result_boxes':len(actual),'cells':160,'remaining_assignments':2**152}
        finally:ledger.close()


def run_growing_symbolic_fixture():
    from test_constraint_work import leaves
    scope=WorkScope('growing','gold',(7,)*4,(7,)*4);result={}
    with tempfile.TemporaryDirectory() as directory:
        for enabled in (False,True):
            excluded=[]
            with CoverageLedger(Path(directory)/('symbolic.sqlite3' if enabled else 'flat.sqlite3'),scope,symbolic=enabled) as ledger:
                ledger.remaining([scope.root])
                for value in range(12):
                    box=(1<<(value%3),1<<((value//3)%3),1<<(value%2),7)
                    excluded.append(box);ledger.record_completed(box,status='infeasible',complete=True)
                    actual,done=ledger.remaining([scope.root]);assert done
                    assert leaves(actual)==leaves([scope.root])-leaves(excluded)
                result['symbolic' if enabled else 'baseline']={**ledger.stats,'remaining_assignments':len(leaves(actual))}
    assert result['symbolic']['coverage_symbolic_builds']==1
    assert result['symbolic']['coverage_symbolic_extensions']==12
    assert result['symbolic']['coverage_records_checked']<result['baseline']['coverage_records_checked']
    return result


class SymbolicCoverageTests(unittest.TestCase):
    def test_exact_operations_match_enumerated_arbitrary_domain_sets(self):
        from constraint_symbolic_work import DomainDiagram
        from test_constraint_work import leaves
        rng=random.Random(2506)
        for _ in range(160):
            root=tuple(rng.choice((3,5,7)) for _ in range(rng.randrange(1,5)))
            def rows():
                return [tuple(rng.choice([mask for mask in range(1,value+1) if not mask&~value]) for value in root) for _ in range(rng.randrange(1,9))]
            left,right=rows(),rows();diagram=DomainDiagram(root)
            a=diagram.union(left);b=diagram.union(right)
            for operation,expected in [('union',leaves(left)|leaves(right)),('difference',leaves(left)-leaves(right))]:
                node=diagram.apply(operation,a,b);actual=diagram.boxes(node)
                self.assertEqual(leaves(actual),expected);self.assertEqual(diagram.volume(node),len(expected))
                from constraint_work import disjoint
                self.assertTrue(disjoint(actual))

    def test_compacting_pending_cubes_is_not_reported_as_completed_coverage(self):
        scope=WorkScope('a','gold',(3,3),(3,3))
        with tempfile.TemporaryDirectory() as directory:
            with CoverageLedger(Path(directory)/'coverage.sqlite3',scope,symbolic=True) as ledger:
                self.assertEqual(ledger.remaining([(1,3),(2,3)]),(((3,3),),True))
                self.assertEqual(ledger.stats['coverage_hits'],0)

    def test_growing_queries_avoid_rescanning_already_checked_completed_records(self):
        report=run_growing_symbolic_fixture()
        self.assertEqual(report['baseline']['remaining_assignments'],report['symbolic']['remaining_assignments'])
        self.assertEqual(report['baseline']['coverage_records_checked'],78)
        self.assertEqual(report['symbolic']['coverage_records_checked'],0)

    def test_repeated_ledger_queries_reuse_completed_union(self):
        report=run_symbolic_fixture()
        self.assertEqual(report['coverage_symbolic_builds'],1)
        self.assertEqual(report['coverage_symbolic_reuses'],7)
        self.assertEqual(report['result_boxes'],1)
        self.assertLess(report['coverage_symbolic_apply_steps'],report['baseline_subtraction_checks'])

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_public_sat_runner_uses_symbolic_coverage_and_respects_live_policy(self):
        from constraint_sat_session import solve
        problem,scope,boxes=core_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'coverage.sqlite3'
            with CoverageLedger(path,scope) as ledger:
                for box in boxes[:6]:ledger.record_completed(box,status='infeasible',complete=True)
            state={'_coverage_path':path,'_portable_frontier':frontier_snapshot(scope,boxes)}
            try:
                answer=solve(problem,seconds=5,learning_state=state,options={'shared_coverage':True,'symbolic_coverage':True})
                self.assertEqual(answer['status'],'infeasible')
                self.assertGreater(answer['stats']['coverage_symbolic_builds'],0)
                self.assertTrue(state['_sat_session'].coverage.symbolic)
                solve(problem,seconds=0,learning_state=state,options={'shared_coverage':True,'symbolic_coverage':False})
                self.assertFalse(state['_sat_session'].coverage.symbolic)
            finally:state['_sat_session'].close()

    def test_growing_local_ledger_extends_union_and_live_cap_is_respected(self):
        from test_constraint_work import leaves
        scope=WorkScope('a','gold',(7,)*4,(7,)*4);excluded=[]
        with tempfile.TemporaryDirectory() as directory:
            with CoverageLedger(Path(directory)/'coverage.sqlite3',scope,symbolic=True) as ledger:
                ledger.remaining([scope.root])
                for value in range(12):
                    box=(1<<(value%3),1<<((value//3)%3),1<<(value%2),7)
                    excluded.append(box);ledger.record_completed(box,status='infeasible',complete=True)
                    actual,done=ledger.remaining([scope.root]);self.assertTrue(done)
                    self.assertEqual(leaves(actual),leaves([scope.root])-leaves(excluded))
                self.assertEqual(ledger.stats['coverage_symbolic_builds'],1)
                self.assertEqual(ledger.stats['coverage_symbolic_extensions'],12)
                ledger.max_symbolic_nodes=2
                self.assertEqual(ledger.remaining([scope.root]),((scope.root,),False))

    def test_ledger_record_scan_is_interruptible_before_materializing_all_rows(self):
        scope=WorkScope('a','gold',(7,7),(7,7));checks=0
        with tempfile.TemporaryDirectory() as directory:
            with CoverageLedger(Path(directory)/'coverage.sqlite3',scope,symbolic=True) as ledger:
                for code in range(3):ledger.record_completed((1<<code,1),status='infeasible',complete=True)
                def stop():
                    nonlocal checks
                    checks+=1
                    if checks==2:raise SessionInterrupted('record scan')
                with self.assertRaises(SessionInterrupted):ledger.exclusions(check=stop)
                self.assertEqual(checks,2)
                self.assertIsNone(ledger._symbolic_state)

    def test_scopes_external_commits_local_corruption_and_restart(self):
        from test_constraint_work import leaves
        wide=WorkScope('a','gold',(7,7),(7,7));narrow=WorkScope('a','gold',(7,7),(3,7))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'coverage.sqlite3';first=CoverageLedger(path,wide,symbolic=True)
            second=CoverageLedger(path,wide)
            try:
                first.record_completed((1,3),status='infeasible',complete=True)
                a,done=first.remaining([narrow.root],target_scope=narrow);self.assertTrue(done)
                self.assertEqual(leaves(a),leaves([narrow.root])-leaves([(1,3)]))
                second.record_completed((2,7),status='infeasible',complete=True,source_scope=narrow)
                b,done=first.remaining([narrow.root],target_scope=narrow);self.assertTrue(done)
                self.assertEqual(leaves(b),leaves([narrow.root])-leaves([(1,3),(2,7)]))
                self.assertEqual(first.stats['coverage_symbolic_builds'],2)
                with self.assertRaises(ValueError):first.remaining([wide.root],target_scope=narrow)
                with self.assertRaises(ValueError):first.remaining([wide.root],target_scope=WorkScope('a','edges',(7,7),(7,7)))
                first.db.execute("UPDATE closed SET box='[\"0x7\",\"0x7\"]'");first.db.commit()
                with self.assertRaises(ValueError):first.remaining([wide.root])
            finally:second.close();first.close()

    def test_caps_and_stop_never_replace_pending_work_with_partial_diagram(self):
        from constraint_symbolic_work import DomainDiagram,DiagramLimit
        from test_constraint_work import leaves
        scope=WorkScope('a','gold',(7,7,7),(7,7,7));original=(scope.root,)
        with tempfile.TemporaryDirectory() as directory:
            ledger=CoverageLedger(Path(directory)/'coverage.sqlite3',scope,symbolic=True,max_symbolic_nodes=3)
            try:
                ledger.record_completed((1,1,1),status='infeasible',complete=True)
                self.assertEqual(ledger.remaining(original),(original,False))
                ledger.max_symbolic_nodes=200000
                self.assertEqual(ledger.remaining(original,max_boxes=1),(original,False))
                checks=0
                def stop():
                    nonlocal checks
                    checks+=1
                    if checks==4:raise SessionInterrupted('stopped')
                with self.assertRaises(SessionInterrupted):ledger.remaining(original,check=stop)
                actual,done=ledger.remaining(original);self.assertTrue(done)
                self.assertEqual(leaves(actual),leaves(original)-{(0,0,0)})
            finally:ledger.close()

def run_higher_fixture():
    from constraint_higher_relations import ProjectedRelations
    from constraint_sat_assist import SatAssist
    from test_constraint_sat_domains import fixture,admitted,ordinary_fixpoint
    problem,domains=fixture();root=tuple(sum(1<<c for c in row) for row in problem.domains)
    solutions=brute(problem,'edges');rng=random.Random(2508)
    compiler=SatAssist(problem,root,'edges',witness_pool=True,support_cover=True)
    baseline=SatAssist(problem,root,'edges',witness_pool=True,support_cover=True)
    relation=ProjectedRelations(problem,root,'edges')
    try:
        initial=domains.copy();relation.prune(initial,compiler,seconds=2,max_calls=100)
        assert len(relation.tables)==1
        (scope,cells),rows=next(iter(relation.tables.items()))
        assert set(rows)=={tuple(row[c] for c in cells) for row in admitted(solutions,domains)}
        assert baseline.prepare(seconds=2)
        contexts=0
        for _ in range(200):
            current=domains.copy()
            for cell in cells:
                codes=[code for code in range(12) if domains[cell]>>code&1]
                current[cell]=sum(1<<code for code in rng.sample(codes,rng.randrange(1,len(codes)+1)))
            current=ordinary_fixpoint(problem,current)
            if not all(current):continue
            contexts+=1;candidate=current.copy();relation.prune(candidate,compiler,seconds=2,max_calls=100)
            if all(candidate):candidate=list(compiler.filter_domains(candidate,seconds=1,max_candidates=100))
            expected=baseline.filter_domains(current,seconds=1,max_candidates=100)
            assert admitted(solutions,candidate)==admitted(solutions,expected)==admitted(solutions,current)
            if all(candidate) and all(expected):assert all(candidate[c]==expected[c] for c in cells)
            else:assert all(candidate)==all(expected)
        return {'contexts_after_ordinary_fixpoint':contexts,'baseline':dict(baseline.stats),
                'candidate':dict(relation.stats),'candidate_assistant':dict(compiler.stats),
                'candidate_total_queries':relation.stats['higher_queries']+compiler.stats['sat_assist_domain_calls'],
                'projected_cells':list(cells),'complete_tuples':len(rows),
                'baseline_native_decisions':int(baseline.solver.accum_stats().get('decisions',0)),
                'candidate_native_decisions':int(compiler.solver.accum_stats().get('decisions',0)),
                'baseline_build_steps':baseline.stats['sat_assist_build_steps'],
                'candidate_build_steps':compiler.stats['sat_assist_build_steps']}
    finally:relation.close();compiler.close();baseline.close()


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class HigherRelationTests(unittest.TestCase):
    def setup_relation(self,target='edges',shared=False):
        from constraint_higher_relations import ProjectedRelations
        from constraint_sat_assist import SatAssist
        from constraint_job_sat import SharedSatAssistant
        from test_constraint_sat_domains import fixture
        problem,domains=fixture();root=tuple(sum(1<<c for c in row) for row in problem.domains)
        if shared:
            owner=SharedSatAssistant(problem,target);helper=owner.view(domains);initial=domains
            self.addCleanup(owner.close)
        else:
            helper=SatAssist(problem,root,target);initial=root;self.addCleanup(helper.close)
        relation=ProjectedRelations(problem,initial,target);self.addCleanup(relation.close)
        return problem,domains,helper,relation

    def test_complete_joint_relations_reduce_total_native_proof_queries(self):
        report=run_higher_fixture()
        self.assertEqual(report['contexts_after_ordinary_fixpoint'],135)
        self.assertLess(report['candidate_total_queries'],report['baseline']['sat_assist_domain_calls'])
        self.assertEqual(report['baseline_build_steps'],report['candidate_build_steps'])

    def test_unknown_and_partial_tables_cannot_prune_or_contaminate_shared_helper(self):
        from test_constraint_sat_domains import admitted
        problem,domains,helper,relation=self.setup_relation(shared=True)
        original=domains.copy();relation.prune(domains,helper,seconds=2,max_calls=1)
        self.assertEqual(domains,original);self.assertEqual(len(relation.tables),0)
        self.assertEqual(len(relation.sessions),1)
        native=helper.owner.helper
        with patch.object(native,'_domain_call',return_value=None):relation.prune(domains,helper,seconds=2)
        self.assertEqual(domains,original);self.assertFalse(relation.tables)
        # The projection selector is absent from this unrelated full-root query.
        all_domains=list(native.initial);filtered=native.filter_domains(all_domains,seconds=2,max_candidates=100)
        solutions=brute(problem,'edges')
        self.assertEqual(admitted(solutions,filtered),admitted(solutions,all_domains))
        relation.prune(domains,helper,seconds=2,max_calls=100)
        self.assertTrue(relation.tables);self.assertEqual(admitted(solutions,domains),admitted(solutions,original))

    def test_blocker_committed_before_error_poisoned_session_is_discarded(self):
        from test_constraint_sat_domains import admitted
        problem,domains,helper,relation=self.setup_relation()
        original=domains.copy();post=relation._post;calls=0
        def broken(native,clause):
            nonlocal calls
            post(native,clause);calls+=1
            if calls==2:raise MemoryError('after native blocker, before tuple insertion')
        with patch.object(relation,'_post',side_effect=broken):
            with self.assertRaises(MemoryError):relation.prune(domains,helper,seconds=2,max_calls=100)
        self.assertEqual(domains,original);self.assertFalse(relation.tables)
        self.assertTrue(next(iter(relation.sessions.values()))['poisoned'])
        relation.prune(domains,helper,seconds=2,max_calls=100)
        self.assertTrue(relation.tables)
        (scope,cells),rows=next(iter(relation.tables.items()))
        self.assertEqual(set(rows),{tuple(row[c] for c in cells) for row in admitted(brute(problem,'edges'),original)})

    def test_cap_and_widening_are_safe_and_larger_cap_can_reopen(self):
        problem,domains,helper,relation=self.setup_relation()
        original=domains.copy();relation.prune(domains,helper,seconds=2,max_calls=100,max_models=0)
        self.assertEqual(domains,original);self.assertFalse(relation.tables);self.assertEqual(relation.stats['higher_caps'],1)
        relation.prune(domains,helper,seconds=2,max_calls=100,max_models=256)
        self.assertTrue(relation.tables)
        wide=list(helper.initial);relation.prune(wide,helper,seconds=0)
        self.assertEqual(wide,list(helper.initial))
        calls=relation.stats['higher_queries'];small=domains.copy()
        relation.prune(small,helper,seconds=2,max_calls=100)
        # Exact projection restrictions need no repeat enumeration when only
        # projected cells narrowed. Other refinements may select new cells.
        self.assertEqual(relation.stats['higher_queries'],calls)

    def test_lifetime_blocker_budget_is_shared_and_does_not_affect_plain_queries(self):
        from constraint_higher_relations import ProjectedRelations
        problem,domains,helper,relation=self.setup_relation(shared=True)
        original=domains.copy()
        with patch('constraint_higher_relations.MAX_GUARDED_CLAUSES',2):
            relation.prune(domains,helper,seconds=2,max_calls=100,max_models=0)
            second=ProjectedRelations(problem,helper.initial,'edges');self.addCleanup(second.close)
            second.prune(domains,helper,seconds=2,max_calls=100,max_models=1)
            self.assertEqual(second.stats['higher_queries'],0);self.assertEqual(second.stats['higher_caps'],1)
        self.assertEqual(domains,original);self.assertEqual(helper.owner.helper._higher_projection_budget_used,2)
        self.assertIsNone(helper.query([],seconds=1));self.assertEqual(helper.last_outcome,'sat')

    def test_completed_unsat_table_survives_stop_before_optional_filtering(self):
        problem,domains,helper,relation=self.setup_relation()
        filter_original=relation._filter;count=0
        def stop_after_commit(current,check):
            nonlocal count
            count+=1
            if count==2:raise SessionInterrupted('after complete proof')
            return filter_original(current,check)
        original=domains.copy()
        with patch.object(relation,'_filter',side_effect=stop_after_commit):
            with self.assertRaises(SessionInterrupted):relation.prune(domains,helper,seconds=2,max_calls=100)
        self.assertEqual(domains,original);self.assertEqual(len(relation.tables),1);self.assertFalse(relation.sessions)
        calls=relation.stats['higher_queries'];relation.prune(domains,helper,seconds=0)
        self.assertEqual(relation.stats['higher_queries'],calls);self.assertNotEqual(domains,original)

    def test_strict_budgets_scope_and_gold_solution_preservation(self):
        from constraint_higher_relations import ProjectedRelations
        from test_constraint_sat_domains import admitted
        problem,domains,helper,relation=self.setup_relation(target='gold')
        original=domains.copy();relation.prune(domains,helper,seconds=2,max_calls=100)
        self.assertEqual(admitted(brute(problem,'gold'),domains),admitted(brute(problem,'gold'),original))
        for value in (0,-1,1.5,True):
            with self.assertRaises(ValueError):ProjectedRelations(problem,helper.initial,'gold',limit=value)
        with self.assertRaises(ValueError):relation.prune(domains,helper,arity=3.0)
        with self.assertRaises(SessionInterrupted):relation.prune(domains,helper,lambda:(_ for _ in ()).throw(SessionInterrupted('stop')))
        problem.domains[0]=problem.domains[0][1:]
        with self.assertRaises(ValueError):relation.prune(domains,helper)

    def test_public_dfs_route_reports_projection_work_and_valid_answer(self):
        from constraint_dfs import solve
        from test_constraint_sat_domains import fixture
        problem,domains=fixture();problem.domains=[[c for c in range(12) if mask>>c&1] for mask in domains]
        result=solve(problem,seconds=5,target='edges',options={'higher_order_relations':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'regional_cache':False,'separator_regions':False,
                     'biconnected':False,'strand_connectivity':False,'multi_seam_cuts':False})
        self.assertIn(result['status'],('solved','edge_perfect'))
        self.assertGreater(result['stats']['higher_queries'],0)
        self.assertIn(tuple(result['codes']),brute(problem,'edges'))


def run_conflict_pb_fixture():
    from constraint_conflict_pb import ConflictPBLearner
    from constraint_pb import PBLearner
    from test_constraint_sat_domains import fixture,ordinary_fixpoint,admitted
    problem,domains=fixture();baseline=ordinary_fixpoint(problem,domains);static=PBLearner(problem)
    while True:
        old=baseline.copy();static.prune(baseline,max_values=12);baseline=ordinary_fixpoint(problem,baseline)
        if baseline==old:break
    assert baseline==domains
    learner=ConflictPBLearner(problem);result=baseline.copy();learner.prune(result)
    assert result==[3972,4045,2009,879]
    counts={}
    for target in ('edges','gold'):
        answers=brute(problem,target);assert admitted(answers,result)==admitted(answers,baseline)
        counts[target]=len(admitted(answers,baseline))
    before=dict(learner.stats);repeat=baseline.copy();learner.prune(repeat,max_trials=0)
    assert repeat==result and before['conflict_pb_combinations']==learner.stats['conflict_pb_combinations']
    return {'before':baseline,'after':result,'static':dict(static.stats),'guided':before,
            'retained_proof_lengths':[len(proof) for record in learner.records.values() for proof in record['proofs']],
            'independent_solution_counts':counts,'repeat_new_combinations':learner.stats['conflict_pb_combinations']-before['conflict_pb_combinations']}


class ConflictPBTests(unittest.TestCase):
    def test_existing_fixpoint_misses_two_verified_weighted_cuts(self):
        report=run_conflict_pb_fixture()
        self.assertEqual(report['guided']['conflict_pb_rows_kept'],2)
        self.assertEqual(report['guided']['conflict_pb_trials'],33)
        self.assertEqual(report['repeat_new_combinations'],0)

    def test_random_integer_combinations_preserve_every_base_model(self):
        from constraint_conflict_pb import guided_cut,compact_steps
        from constraint_pb import row,verify_derivation
        rng=random.Random(2510);hits=0
        for _ in range(100):
            base=tuple(row([(c,v,rng.randrange(-3,4)) for c in range(3) for v in range(2)],rng.randrange(-3,5)) for unused in range(7))
            trial=[3,3,3];trial[rng.randrange(3)]=1<<rng.randrange(2)
            hit=guided_cut(base,trial,max_pairs=100,max_new=16)
            if hit is None:continue
            cut,steps=hit;compact=compact_steps(base,steps);self.assertTrue(verify_derivation(base,compact));self.assertEqual(compact[-1][4],cut);hits+=1
            for values in itertools.product(range(2),repeat=3):
                satisfies=lambda item:sum(k for c,v,k in item[0] if values[c]==v)<=item[1]
                if all(satisfies(item) for item in base):self.assertTrue(satisfies(cut))
        self.assertGreater(hits,0)

    def test_scope_reuse_widening_and_cancellation_do_not_leak(self):
        from constraint_conflict_pb import ConflictPBLearner
        from test_constraint_sat_domains import fixture,admitted
        problem,domains=fixture();learner=ConflictPBLearner(problem);learner.prune(domains)
        answers=brute(problem,'edges');rng=random.Random(2511)
        for _ in range(60):
            before=[sum(1<<c for c in row if rng.randrange(3)) for row in problem.domains]
            after=before.copy();learner.prune(after,max_trials=0)
            self.assertEqual(admitted(answers,before),admitted(answers,after))
        root=[sum(1<<c for c in row) for row in problem.domains];wide=root.copy();learner.prune(wide,max_trials=0);self.assertEqual(root,wide)
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==100:raise SessionInterrupted('bounded cancellation')
        fresh=ConflictPBLearner(problem);original=domains.copy()
        with self.assertRaises(SessionInterrupted):fresh.prune(domains,stop)
        self.assertEqual(domains,original);self.assertFalse(any(r['cuts'] for r in fresh.records.values()))
        problem.domains[0]=problem.domains[0][1:]
        with self.assertRaises(ValueError):learner.prune(root,max_trials=0)

    def test_completed_hit_is_not_memoized_before_interrupted_final_replay(self):
        from constraint_conflict_pb import ConflictPBLearner,guided_cut
        from constraint_pb import primitive_rows
        from test_constraint_sat_domains import fixture
        problem,domains=fixture();scope=tuple(enumerate(domains));base=primitive_rows(problem,scope)
        trial=domains.copy();trial[2]=1<<2;hit=guided_cut(base,trial)
        self.assertIsNotNone(hit)
        learner=ConflictPBLearner(problem);original=domains.copy()
        with patch('constraint_conflict_pb.guided_cut',return_value=hit),patch('constraint_conflict_pb.verify_derivation',side_effect=SessionInterrupted('during final replay')):
            with self.assertRaises(SessionInterrupted):learner.prune(domains,max_trials=1)
        self.assertEqual(domains,original);record=next(iter(learner.records.values()))
        self.assertFalse(record['attempted']);self.assertFalse(record['cuts'])
        with patch('constraint_conflict_pb.guided_cut',return_value=hit):learner.prune(domains,max_trials=1)
        self.assertTrue(record['attempted']);self.assertTrue(record['cuts']);self.assertNotEqual(domains,original)

    def test_caps_malformed_rows_and_real_stage_hook(self):
        from constraint_conflict_pb import ConflictPBLearner,guided_cut
        from constraint_round24 import ResearchPruner
        from search_options import normalize_options
        from test_constraint_sat_domains import fixture
        problem,domains=fixture();learner=ConflictPBLearner(problem);before=domains.copy()
        learner.prune(domains,max_pairs=0);self.assertEqual(domains,before)
        with self.assertRaises(ValueError):guided_cut(((((0,0,1.0),),0),),[3])
        for bad in ([-1],[True],[1.0],[1<<481]):
            with self.assertRaises(ValueError):guided_cut((),bad)
        with self.assertRaises(ValueError):guided_cut(((((2,0,1),),0),),[3])
        with self.assertRaises(ValueError):learner.prune(domains,max_trials=True)
        options=normalize_options({'conflict_pb':True});pruner=ResearchPruner(problem,'gold',options)
        pruner.prune(domains,lambda:None,lambda action,reason:action())
        self.assertEqual(domains,[3972,4045,2009,879]);self.assertEqual(pruner.stats['conflict_pb_rows_kept'],2)


def run_hall_rematch_fixture():
    from constraint_incremental_hall import IncrementalHall
    from test_constraint_incremental_hall import exhaustive
    root=(3,3,12,12,48,48);child=(2,3,12,12,48,48);report={}
    for name,enabled in (('baseline',False),('rematch',True)):
        engine=IncrementalHall(rematch_reuse=enabled);engine.filter(root);before=dict(engine.stats)
        answer,matching,failed=engine.filter(child)
        assert answer==exhaustive(child) and failed==-1
        report[name]={'allowed':list(answer),'matching':list(matching),'delta':{k:engine.stats[k]-before[k] for k in engine.stats}}
    assert report['baseline']['allowed']==report['rematch']['allowed']
    return report


class HallRematchTests(unittest.TestCase):
    def test_repaired_component_reuses_unaffected_traversals(self):
        report=run_hall_rematch_fixture()
        self.assertEqual(report['baseline']['delta']['hall_scc_vertices'],6)
        self.assertEqual(report['rematch']['delta']['hall_scc_vertices'],2)
        self.assertEqual(report['rematch']['delta']['hall_scc_reused_vertices'],4)
        self.assertEqual(report['rematch']['delta']['hall_scc_rematched_partial_passes'],1)

    def test_independent_all_matching_enumeration_with_shrink_widen_and_new_seeds(self):
        from constraint_incremental_hall import IncrementalHall
        from test_constraint_incremental_hall import exhaustive
        rng=random.Random(2512)
        for n in range(2,7):
            engine=IncrementalHall(rematch_reuse=True);history=[]
            for iteration in range(140):
                if history and iteration%7==0:rows=rng.choice(history)
                elif history and iteration%3:
                    changed=list(history[-1]);cell=rng.randrange(n);changed[cell]&=~(1<<rng.randrange(n));rows=tuple(changed)
                else:rows=tuple(rng.randrange(1,1<<n) for unused in range(n))
                seed=list(range(n));rng.shuffle(seed)
                result=engine.filter(rows,seed if iteration%4==0 else None)[0]
                self.assertEqual(result,exhaustive(rows),(n,rows));history.append(rows)

    def test_stop_cannot_publish_partial_repair_or_scc(self):
        from constraint_incremental_hall import IncrementalHall
        root=(3,3,12,12,48,48);child=(2,3,12,12,48,48)
        stopped=0
        for limit in range(1,30):
            engine=IncrementalHall(rematch_reuse=True);engine.filter(root);prior=engine.previous;calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==limit:raise SessionInterrupted('during repair')
            try:engine.filter(child,check=check)
            except SessionInterrupted:
                stopped+=1;self.assertIs(engine.previous,prior);self.assertNotIn(child,engine.snapshots)
            self.assertEqual(engine.filter(child)[0],(2,1,12,12,48,48))
        self.assertGreater(stopped,5)


    def test_public_pruner_hook_and_dfs_warm_policy(self):
        from constraint_pruning import Pruner
        from constraint_dfs import solve
        from test_constraint_dfs import board
        tiles=[{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(6)]
        problem=SearchProblem(tiles,board(6,[(a,s,a+1,s) for a in (0,2,4) for s in range(3)]))
        masks=lambda rows:[sum(1<<(3*t) for t in range(6) if row>>t&1) for row in rows]
        engine=Pruner(problem,incremental_hall=True,hall_rematch=True)
        engine.prune_hall(masks((3,3,12,12,48,48)));before=dict(engine.stats)
        child=masks((2,3,12,12,48,48));engine.prune_hall(child)
        self.assertEqual(child,masks((2,1,12,12,48,48)))
        self.assertEqual(engine.stats['hall_scc_vertices']-before['hall_scc_vertices'],2)
        problem=SearchProblem(*tetrahedron_tiles());warm={}
        result=solve(problem,seconds=5,target='edges',learning_state=warm,options={'hall_rematch':True,'native':'python'})
        self.assertIn(result['status'],('solved','edge_perfect'));self.assertTrue(warm['_runtime'].incremental_hall.rematch_reuse)
        result=solve(problem,seconds=5,target='edges',learning_state=warm,options={'hall_rematch':False,'incremental_hall':True,'native':'python'})
        self.assertFalse(warm['_runtime'].incremental_hall.rematch_reuse)


def run_gold_graph_fixture():
    from constraint_gold_graph import PossibleGoldGraph
    from constraint_advanced import AdvancedPruner
    from search_fixtures import planted_gold
    data,board,solution=planted_gold(seed=2513,size=1,extra_edges=3);problem=SearchProblem(data,board)
    root=[sum(1<<c for c in row) for row in problem.domains];contexts=[root.copy()];current=root.copy()
    for index in range(12):
        cell=index%problem.n;code=next(c for c in problem.domains[cell] if current[cell]>>c&1 and c!=solution[cell])
        current=current.copy();current[cell]&=~(1<<code);contexts.append(current)
    contexts+=contexts[:3]
    baseline=AdvancedPruner(problem,options={'native':'python'});candidate=AdvancedPruner(problem,options={'native':'python','reversible_gold_graph':True})
    for context in contexts:
        for method in ('prune_strands','prune_biconnected'):
            before=context.copy();after=context.copy();getattr(baseline,method)(before);getattr(candidate,method)(after)
            assert before==after
            assert all(mask>>code&1 for mask,code in zip(after,solution))
    return {'cells':problem.n,'contexts':len(contexts),'baseline_seam_tests':baseline.stats['gold_graph_seam_tests'],
            'candidate':dict(candidate._possible_gold.stats),'reference_solution':solution,
            'non_claims':['Global component and block traversals remain complete; graph flattening, source binding and snapshot copies are additional operations.',
                          'This compares existing portable Python graph assembly; dense native fast paths may be preferable.']}


class GoldGraphTests(unittest.TestCase):
    def test_affected_seams_and_shared_snapshots_skip_repeated_construction(self):
        report=run_gold_graph_fixture()
        self.assertEqual(report['cells'],10)
        self.assertLess(report['candidate']['gold_graph_seam_tests'],report['baseline_seam_tests'])
        self.assertGreater(report['candidate']['gold_graph_snapshot_hits'],0)

    def test_raw_graph_and_filters_match_independent_construction_and_all_solutions(self):
        from constraint_gold_graph import PossibleGoldGraph
        from constraint_advanced import AdvancedPruner
        from test_constraint_sat_domains import fixture,admitted
        problem,_=fixture();engine=PossibleGoldGraph(problem);baseline=AdvancedPruner(problem,options={'native':'python'})
        candidate=AdvancedPruner(problem,options={'native':'python'});candidate._possible_gold=engine
        answers=brute(problem,'gold');rng=random.Random(25130)
        for iteration in range(120):
            domains=[sum(1<<c for c in row if rng.randrange(3)) for row in problem.domains]
            if not all(domains):continue
            state=engine.update(domains);expected=[];used=[]
            for cell,domain in enumerate(domains):
                pairs={pair for code in range(12) if domain>>code&1 for pair in engine.segments[code]}
                expected.extend((33*cell+a,33*cell+b) for a,b in sorted(pairs));used.append({v for pair in pairs for v in pair})
            for a,sa,b,sb in problem.edges:
                for port in range(11):
                    left=11*sa+port;right=11*sb+10-port
                    if left in used[a] and right in used[b]:expected.append((33*a+left,33*b+right))
            self.assertEqual(state.edges,tuple(expected))
            for method in ('prune_strands','prune_biconnected'):
                before=domains.copy();after=domains.copy();getattr(baseline,method)(before);getattr(candidate,method)(after)
                self.assertEqual(after,before)
                self.assertEqual(admitted(answers,domains),admitted(answers,after))

    def test_partial_update_stop_widening_and_source_binding(self):
        from constraint_gold_graph import PossibleGoldGraph
        from test_constraint_sat_domains import fixture
        problem,_=fixture();root=[sum(1<<c for c in row) for row in problem.domains];child=root.copy();child[0]&=~1
        stopped=0
        for limit in range(1,24):
            engine=PossibleGoldGraph(problem);prior=engine.update(root);calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==limit:raise SessionInterrupted('during graph update')
            try:engine.update(child,check)
            except SessionInterrupted:
                stopped+=1;self.assertIs(engine.previous,prior);self.assertNotIn(tuple(child),engine.snapshots)
            self.assertEqual(engine.update(child).edges,PossibleGoldGraph(problem).update(child).edges)
            self.assertIs(engine.update(root),prior)
        self.assertGreater(stopped,5)
        with self.assertRaises(ValueError):engine.update([-1,*root[1:]])
        segments=list(problem.tiles[0]['segments']);first=list(segments[0]);first[0]=(first[0][0],3);segments[0]=tuple(first);problem.tiles[0]['segments']=segments
        with self.assertRaises(ValueError):engine.update(root)


    def test_public_dfs_route_warm_reuse_and_policy_toggle(self):
        from constraint_dfs import solve
        problem=SearchProblem(*tetrahedron_tiles());warm={};options={'reversible_gold_graph':True,'native':'python','lookahead':False,'pair_lookahead':False}
        first=solve(problem,seconds=5,target='gold',learning_state=warm,options=options)
        self.assertEqual(first['status'],'solved');self.assertGreater(first['stats']['gold_graph_builds'],0)
        graph=warm['_runtime'].advanced._possible_gold;hits=graph.stats['gold_graph_snapshot_hits']
        second=solve(problem,seconds=5,target='gold',learning_state=warm,options=options)
        self.assertEqual(second['status'],'solved');self.assertIs(warm['_runtime'].advanced._possible_gold,graph)
        self.assertGreaterEqual(graph.stats['gold_graph_snapshot_hits'],hits)
        third=solve(problem,seconds=5,target='gold',learning_state=warm,options={**options,'reversible_gold_graph':False})
        self.assertEqual(third['status'],'solved');self.assertIsNone(warm['_runtime'].advanced._possible_gold)
        self.assertIn(tuple(first['codes']),brute(problem,'gold'))

if __name__=='__main__':unittest.main()
