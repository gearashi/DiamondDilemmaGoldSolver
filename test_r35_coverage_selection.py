"""Selective reads must see the same committed coverage as the ordered ledger stream."""
import itertools
import sqlite3
from contextlib import closing
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from constraint_work import CoverageLedger, WorkScope, subtract_many


def publish(db, row):
    db.record_completed(row, status='infeasible', complete=True)


class SelectiveCoverageTests(unittest.TestCase):
    def test_arbitrary_masks_caps_scope_change_and_full_index(self):
        rng = Random(3565)
        root = (7, 7, 7, (1 << 130) | 3)
        scope = WorkScope('finite', 'gold', root, root)
        with TemporaryDirectory() as tmp, CoverageLedger(Path(tmp)/'c.sqlite', scope) as db:
            db._coverage_cache_limit = 1024
            for _ in range(30):
                publish(db, (*[rng.randrange(1,8) for _ in root[:3]], rng.choice((1,2,3,1 << 130))))
            for requested in (root, (7, 3, 7, 3), root):
                target = WorkScope('finite', 'gold', root, requested)
                exclusions = db.exclusions(target)
                choices = [[1 << i for i in range(k.bit_length()) if k >> i & 1] for k in requested]
                for point in itertools.product(*choices):
                    for pending, limit in (([point],2), ([point,point],1)):
                        expected = subtract_many(pending, exclusions, max_boxes=limit)
                        self.assertEqual(db.remaining(pending, target_scope=target, max_boxes=limit), expected)
                self.assertLessEqual(db._coverage_bytes, 1024)

    def test_snapshot_interruption_concurrent_changes_and_corruption(self):
        scope = WorkScope('committed','gold',(7,7),(7,7))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'c.sqlite'
            with CoverageLedger(path,scope) as db, CoverageLedger(path,scope) as writer:
                publish(writer,(1,1))
                publish(writer,(2,2))
                db.remaining([(4,4)])
                publish(writer,(4,1))
                self.assertEqual(db.remaining([(4,1)]), ((),True))
                writer.db.execute("DELETE FROM closed WHERE box LIKE '%0x4%'")
                writer.db.commit()
                self.assertEqual(db.remaining([(4,1)]), (((4,1),),True))
                def stop():
                    raise InterruptedError()
                with self.assertRaises(InterruptedError):
                    db.remaining([(4,4)],check=stop)
                self.assertFalse(db.db.in_transaction)
                db.remaining([(4,4)])
                writer.db.execute("UPDATE closed SET evidence='{}'")
                writer.db.commit()
                with self.assertRaisesRegex(ValueError,'corrupt'):
                    db.remaining([(4,4)])

    def test_rollback_with_reused_sequence_for_ordinary_and_symbolic(self):
        scope = WorkScope('rollback','gold',(7,7),(7,7))
        with TemporaryDirectory() as tmp:
            for symbolic in (False,True):
                with CoverageLedger(Path(tmp)/str(symbolic),scope,symbolic=symbolic) as db:
                    publish(db,(1,1))
                    db.remaining([(4,4)])
                    db.db.execute('BEGIN')
                    db.db.execute('DELETE FROM closed')
                    self.assertEqual(db.remaining([(1,1)]), (((1,1),),True))
                    db.db.rollback()
                    self.assertEqual(db.remaining([(1,1)]), ((),True))

    def test_stream_new_record_after_snapshot_and_reused_uncommitted_sequence(self):
        scope = WorkScope('snapshot','gold',(7,7),(7,7))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows'
            with CoverageLedger(path,scope) as db, CoverageLedger(path,scope) as peer:
                publish(peer,(1,1))
                self.assertEqual(db.exclusions(), ((1,1),))
                def commit_during_read():
                    if not getattr(commit_during_read,'done',False):
                        commit_during_read.done = True
                        publish(peer,(2,2))
                self.assertEqual(db.exclusions(check=commit_during_read), ((1,1),))
                self.assertEqual(db.exclusions(), ((1,1),(2,2)))
                self.assertEqual(db.stats['coverage_records_checked'],2)
                row = db.db.execute("SELECT * FROM closed WHERE box LIKE '%0x2%'").fetchone()
                peer.db.execute("DELETE FROM closed WHERE identity=?", (row[0],))
                peer.db.commit()
                for symbolic in (False,True):
                    db.symbolic = symbolic
                    db.remaining([(4,4)])
                    db.db.execute('BEGIN')
                    db.db.execute('INSERT INTO closed VALUES(?,?,?,?,?)',row)
                    self.assertEqual(db.remaining([(2,2)]), ((),True))
                    db.db.rollback()
                    publish(peer,(4,4))
                    self.assertEqual(db.remaining([(2,2)]), (((2,2),),True))
                    self.assertEqual(db.remaining([(4,4)]), ((),True))
                    peer.db.execute("DELETE FROM closed WHERE box LIKE '%0x4%'")
                    peer.db.commit()

    def test_legacy_ledger_and_cold_corruption_outside_pending_hull(self):
        scope = WorkScope('legacy','gold',(7,7),(7,7))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.sqlite'
            with CoverageLedger(Path(tmp)/'seed',scope) as seed:
                seed.record_completed((1,1),status='infeasible',complete=True,
                                      evidence={'kind':'unique_r35_payload_before'})
                row = seed.db.execute('SELECT * FROM closed').fetchone()
            with closing(sqlite3.connect(path)) as old, old:
                old.execute('CREATE TABLE closed(identity TEXT PRIMARY KEY, scope TEXT NOT NULL, box TEXT NOT NULL, evidence TEXT NOT NULL, checksum TEXT NOT NULL)')
                old.execute('INSERT INTO closed VALUES(?,?,?,?,?)',row)
            with CoverageLedger(path,scope) as db:
                self.assertEqual(db.remaining([(4,4)]), (((4,4),),True))
                self.assertEqual(db.exclusions(), ((1,1),))
            # Damage only same-length payload bytes on an isolated, closed SQLite file.
            # The sequence number and its prior, correctly learned axis survive the edit.
            before, after = b'unique_r35_payload_before', b'unique_r35_payload_broken'
            raw = path.read_bytes()
            self.assertEqual((raw.count(before), len(before)), (1, len(after)))
            path.write_bytes(raw.replace(before,after))
            with CoverageLedger(path,scope) as reader:
                with self.assertRaisesRegex(ValueError,'corrupt'):
                    reader.remaining([(4,4)])

    def test_dense_changes_preserve_reordered_queries_and_numeric_widths(self):
        from constraint_work import _CoverageMaskTable
        rng = Random(3572)
        root = ((1 << 96) - 1,) * 64
        table = _CoverageMaskTable(root)
        # Sampled coordinates are identical. Other coordinates repair disjoint
        # anchors, add fresh restrictions, or explicitly return to the full root.
        anchor = tuple(1 << (i % 96) for i in range(64))
        rows = [anchor]
        for _ in range(160):
            current = list(anchor)
            for i in (1,2,3,9,10,17,33,57):
                current[i] = root[i] if rng.randrange(4) == 0 else rng.getrandbits(95) | 1
            rows.append(tuple(current))
        records = [table.encode(row, 16 << 20)[0] for row in rows]
        self.assertTrue(any(record[1] is not None for record in records))
        self.assertTrue(any(record[2].typecode=='H' for record in records))
        for hull in (root, *rows, *((1 << (i%95),)*64 for i in range(65))):
            expected = [row for row in rows if all(a&b for a,b in zip(row,hull))]
            self.assertEqual(list(table.matching(records,hull,lambda:None)), expected)
        before = len(table.ids),len(table.masks),len(table.anchors)
        self.assertIsNone(table.encode((1 << 111,)*64, 1))
        self.assertEqual(before,(len(table.ids),len(table.masks),len(table.anchors)))

    def test_literal_anchor_collisions_and_over_budget_replacements(self):
        from constraint_work import _CoverageMaskTable
        from unittest.mock import patch
        rng = Random(35065)
        root = ((1 << 480) - 1,) * 160
        table = _CoverageMaskTable(root)
        rows = []
        records = []
        with patch('builtins.hash',return_value=0):
            for i in range(24):
                row = [rng.getrandbits(480) | 1 for _ in root] if i % 4 == 0 else list(rows[-1])
                row[3] = root[3] if i % 3 == 0 else (1 << i) | 2
                row = tuple(row)
                records.append(table.encode(row, 1048576)[0])
                rows.append(row)
            before = dict(table.literal_anchors)
            self.assertIsNone(table.encode(tuple(reversed(rows[0])), 8))
            self.assertEqual(table.literal_anchors, before)
        self.assertTrue(any(record[0] is None and record[2] for record in records))
        self.assertTrue(any(record[0] is None and not record[2] for record in records))
        for hull in (root, *rows, *[root[:3]+(1<<i,)+root[4:] for i in range(24)]):
            self.assertEqual(list(table.matching(records,hull,lambda:None)),
                             [row for row in rows if all(a&b for a,b in zip(row,hull))])

    def test_index_wider_than_uint16_and_clipped_rows(self):
        from constraint_work import _CoverageMaskTable
        n=65537
        root=(7,)*n
        table=_CoverageMaskTable(root)
        row=root[:-1]+(1,)
        packed=table.encode(row,4 << 20)[0]
        self.assertIn(packed[2].typecode,('I','Q'))
        self.assertEqual(list(table.matching([None,packed], root, lambda:None)), [row])
        self.assertEqual(list(table.matching([packed], root[:-1]+(2,), lambda:None)), [])
