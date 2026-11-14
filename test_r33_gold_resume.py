"""Checkpoint replay, old interval compatibility and concurrent ownership for Gold loss pages."""
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from r27_structure_fixtures import planted
from r32_gold import run_syndrome
from r32_gold_loss import reference_groups
from r32_gold_syndrome import OutsideSyndromes


class CheckpointReplayTests(unittest.TestCase):
    def setUp(self):
        problem, _ = planted(1, 3, 23, False, True)
        reference, _, groups = reference_groups(problem, [3*i for i in range(problem.n)])
        self.args = problem, reference, groups
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'pages.sqlite'

    def query(self, layers, page, path=None):
        return run_syndrome(*self.args, 0, layers, path or self.path, 0, page=page)

    def test_warm_negative_and_positive_read_only_while_another_writer_owns_database(self):
        negative = self.query([3], 2)
        self.assertEqual(negative['status'], 'INFEASIBLE')
        self.assertGreater(negative['new_pages'], 1)
        found = self.query([0], 1)
        self.assertEqual(found['status'], 'PHASE_RELAXATION_FEASIBLE')
        readonly = self.path.as_uri()+'?mode=ro'
        with closing(sqlite3.connect(self.path)) as writer:
            writer.execute('BEGIN IMMEDIATE')
            with patch.object(OutsideSyndromes, '__init__', side_effect=AssertionError('warm search compiled')):
                again = self.query([3], 17, readonly)
                self.assertEqual(again['completed_prefixes'], negative['completed_prefixes'])
                self.assertEqual(again['new_pages'], 0)
                self.assertEqual(again['work'], {})
                witness = self.query([0], 7, readonly)
                self.assertTrue(witness['reused'])
                self.assertEqual(witness['work'], {})
                self.assertEqual(witness['outside_errors'], found['outside_errors'])
                self.assertEqual(witness['next_rank'], found['next_rank'])

    def test_discontiguous_legacy_pages_leave_only_holes_for_search(self):
        expected = self.query([3], 2)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('DELETE FROM r32_loss_pages WHERE lo IN (2,4,10)')
        original = OutsideSyndromes.indexed
        visits = []
        def indexed(query, cap, errors, seconds, start, stop):
            visits.append((start,stop))
            return original(query,cap,errors,seconds,start,stop)
        with patch.object(OutsideSyndromes,'indexed',indexed):
            actual = self.query([3], 8)
        self.assertEqual(actual['status'], expected['status'])
        self.assertEqual(actual['completed_prefixes'], expected['completed_prefixes'])
        self.assertEqual(visits, [(2,6),(10,12)])
        self.assertEqual(self.query([3], 1)['new_pages'], 0)

    def test_partial_and_positive_returns_include_every_new_page(self):
        for final in ('UNKNOWN','PHASE_RELAXATION_FEASIBLE'):
            with self.subTest(final=final):
                path = Path(self.folder.name)/(final+'.sqlite')
                def trial(query, upper, errors, remaining, start, stop):
                    status = 'INFEASIBLE' if start==0 else final
                    return dict(status=status,next_rank=stop if start==0 else start,work={'calls':1})
                with patch.object(OutsideSyndromes,'indexed',trial):
                    result=self.query([3], 1, path)
                self.assertEqual((result['status'],result['work'],result['pages']), (final,{'calls':2},1))
                if final=='PHASE_RELAXATION_FEASIBLE':
                    with patch.object(OutsideSyndromes,'__init__',side_effect=AssertionError('unexpected rebuild')):
                        again=self.query([3],2,path)
                    self.assertEqual(again['work'],{})

    def test_another_worker_can_claim_a_gap_during_compilation(self):
        expected = self.query([3], 2)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('DELETE FROM r32_loss_pages WHERE lo=2')
        initialize = OutsideSyndromes.__init__
        def concurrent(instance, *args, **kwargs):
            with patch.object(OutsideSyndromes,'__init__',initialize):
                second = self.query([3], 5)
                self.assertEqual(second['new_pages'], 1)
            initialize(instance, *args, **kwargs)
        with patch.object(OutsideSyndromes,'__init__',concurrent):
            again = self.query([3], 1)
        self.assertEqual(again['new_pages'],0)
        self.assertEqual(again['completed_prefixes'],expected['completed_prefixes'])


if __name__=='__main__':
    unittest.main()
