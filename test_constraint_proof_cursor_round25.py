"""Binding, lifecycle and fallback checks for retained incremental RUP cursors."""
import copy
import unittest
from unittest.mock import patch

from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_proof import MaskProofContext
from constraint_proof_cursor import ResumableRUPChecker, _ValidatedRUPBase
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, double_face


class SharedProofBaseTests(unittest.TestCase):
    def test_validated_base_is_shared_but_assumptions_and_proved_lines_are_private(self):
        source=((1,2),(-1,2))
        base=_ValidatedRUPBase(source)
        first=ResumableRUPChecker(base,['0'],extra_clauses=((-2,),))
        second=ResumableRUPChecker(base,['0'],extra_clauses=((2,),))
        self.assertIs(base.clauses,source)
        self.assertIs(first.base,second.base)
        self.assertTrue(first.advance())
        self.assertFalse(second.advance())
        self.assertEqual(source,((1,2),(-1,2)))
        with self.assertRaises(AttributeError):base.clauses=()

    def test_all_construction_paths_validate_and_capture_their_inputs(self):
        for clauses in (((0,),),((True,),),('1 0',)):
            with self.assertRaises(ValueError):_ValidatedRUPBase(clauses)
            with self.assertRaises(ValueError):ResumableRUPChecker(clauses,['0'])
        with self.assertRaises(ValueError):ResumableRUPChecker(((1,),),['0'],extra_clauses=((0,),))
        with self.assertRaises(TypeError):ResumableRUPChecker(((1,),),['0'],trusted=True)
        mutable=[[1],[-1]]
        base=_ValidatedRUPBase(mutable)
        mutable[1][:]=[1]
        self.assertTrue(ResumableRUPChecker(base,['0']).advance())
        self.assertEqual(_ValidatedRUPBase(((1,1),(-1,))).clauses,((1,),(-1,)))


class IncrementalProofBindingTests(unittest.TestCase):
    def fixture(self):
        problem=SearchProblem(simple_tiles(),double_face())
        initial=tuple(sum(1<<code for code in row) for row in problem.domains)
        producer=MaskProofContext(problem,initial)
        self.addCleanup(producer.close)
        certificate=producer.prove([(0,1),(1,1)],seconds=2)
        self.assertIsNotNone(certificate)
        context=MaskProofContext(problem,initial)
        self.addCleanup(context.close)
        return problem,initial,producer,context,certificate

    def finish(self,context,certificate):
        for _ in range(200):
            answer=context.verify_incremental(certificate,ticks=256)
            if answer is not None:return answer
        self.fail('The finite proof did not finish.')

    def test_context_shares_base_between_distinct_record_cursors(self):
        _,_,producer,context,first=self.fixture()
        second=producer.prove([(0,8),(1,8)],seconds=2)
        self.assertIsNotNone(second)
        self.assertIsNone(context.verify_incremental(first,ticks=0))
        self.assertIsNone(context.verify_incremental(second,ticks=0))
        cursors=list(context._verification_cursors.values())
        self.assertEqual(len(cursors),2)
        self.assertIs(cursors[0].base,cursors[1].base)
        self.assertIs(cursors[0].base,context.base)
        self.assertNotEqual(cursors[0].extra_clauses,cursors[1].extra_clauses)
        self.assertTrue(self.finish(context,first))
        self.assertTrue(self.finish(context,second))

    def test_changed_proof_bytes_cannot_borrow_a_checked_prefix(self):
        _,_,_,context,certificate=self.fixture()
        self.assertIsNone(context.verify_incremental(certificate,ticks=1))
        original=next(iter(context._verification_cursors.values()))
        visits=original.stats['proof_clause_visits']
        changed=copy.deepcopy(certificate)
        changed['proof']=['not-an-integer 0']
        self.assertFalse(context.verify_incremental(changed))
        self.assertEqual(original.stats['proof_clause_visits'],visits)
        self.assertFalse(original.finished)
        self.assertTrue(self.finish(context,certificate))
        self.assertEqual(context.stats['proof_imports'],1)

    def test_changed_root_hash_and_problem_binding_do_not_advance_the_cursor(self):
        problem,_,_,context,certificate=self.fixture()
        self.assertIsNone(context.verify_incremental(certificate,ticks=1))
        cursor=next(iter(context._verification_cursors.values()))
        visits=cursor.stats['proof_clause_visits']
        for field,replacement in (('initial',['0x1']*problem.n),('base_sha256','0'*64)):
            changed=copy.deepcopy(certificate);changed[field]=replacement
            self.assertFalse(context.verify_incremental(changed))
            self.assertEqual(cursor.stats['proof_clause_visits'],visits)
            self.assertFalse(cursor.finished)
        original_domains=copy.deepcopy(problem.domains)
        problem.domains[0]=problem.domains[0][1:]
        with self.assertRaises(ValueError):context.verify_incremental(certificate)
        self.assertEqual(cursor.stats['proof_clause_visits'],visits)
        problem.domains=original_domains
        self.assertTrue(self.finish(context,certificate))

    def test_completed_import_is_counted_once_and_close_releases_shared_cursor_base(self):
        _,_,_,context,certificate=self.fixture()
        self.assertTrue(self.finish(context,certificate))
        before=context.stats.copy()
        for _ in range(4):self.assertTrue(context.verify_incremental(certificate))
        self.assertEqual(context.stats,before)
        context.close()
        self.assertFalse(context._verification_cursors)
        self.assertIsNone(context._verification_base)

    def test_total_cap_keeps_claim_pending_until_independent_fallback_verifies_it(self):
        problem,_,_,_,certificate=self.fixture()
        def absent():raise ImportError('Native SAT deliberately unavailable in this regression.')
        store=VerifiedMaskConflictStore(problem,resumable_proof_checking=True,factory=absent)
        self.addCleanup(store.close)
        self.assertTrue(store.offer_certificate(certificate))
        original=ResumableRUPChecker
        def capped(*args,**kwargs):return original(*args,**kwargs,max_visits=0)
        with patch('constraint_proof_cursor.ResumableRUPChecker',side_effect=capped):
            self.assertEqual(store.advance(seconds=1),0)
        self.assertEqual(len(store),0)
        self.assertEqual(store.pending_count,1)
        self.assertFalse(store._certificate_pending)
        self.assertTrue(next(iter(store._proofs()._verification_cursors.values())).limited)
        self.assertIsNone(store._oracle)
        self.assertEqual(store.stats['mask_proofs_verified'],0)
        # Existing ordinary-claim verification falls back to a fresh bounded
        # DPLL proof when the optional native dependency is unavailable.
        self.assertEqual(store.advance(seconds=1),1)
        self.assertEqual(len(store),1)
        self.assertEqual(store.stats['mask_proofs_generated'],1)
        self.assertEqual(store.stats['mask_proofs_verified'],0)

    def test_total_cap_cannot_turn_a_satisfiable_claim_into_an_exclusion(self):
        problem,_,_,_,certificate=self.fixture()
        certificate['record']=[]
        def absent():raise ImportError('Native SAT deliberately unavailable in this regression.')
        store=VerifiedMaskConflictStore(problem,resumable_proof_checking=True,factory=absent)
        self.addCleanup(store.close)
        self.assertTrue(store.offer_certificate(certificate))
        original=ResumableRUPChecker
        def capped(*args,**kwargs):return original(*args,**kwargs,max_visits=0)
        with patch('constraint_proof_cursor.ResumableRUPChecker',side_effect=capped):
            self.assertEqual(store.advance(seconds=1),0)
        self.assertEqual(store.advance(seconds=1),0)
        self.assertEqual(len(store),0)
        self.assertEqual(store.stats['mask_proofs_verified'],0)
        self.assertEqual(store.stats['mask_proofs_generated'],0)


if __name__=='__main__':unittest.main()
