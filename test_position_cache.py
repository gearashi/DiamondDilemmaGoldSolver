"""Behavioral tests for the persistent exact complete-position cache."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
import time
import unittest

from position_cache import PositionCache, PositionCacheError, pack_position, _payload_digest
from validator import validate_arrangement

ROOT = Path(__file__).resolve().parent


class PositionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic full-edge match with many loops; no downloaded puzzle data.
        cls.tiles = {'tiles': [{'id': f'fixture-{i}',
                  'segments': [[[side, 5], [side, 7]] for side in range(3)]}
                 for i in range(160)]}
        cls.codes = [3 * i for i in range(160)]
        cls.real_report = validate_arrangement(cls.tiles, cls.codes)
        assert cls.real_report['placement_count'] == 160
        assert cls.real_report['unique_tiles']

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='position-cache-test-', dir=ROOT)
        self.directory = Path(self.temp.name).resolve()
        self.assertTrue(self.directory.is_relative_to(ROOT))
        self.path = self.directory / 'positions.sqlite3'

    def tearDown(self):
        # The only recursive cleanup target is this test-created directory.
        self.assertTrue(self.directory.parent == ROOT)
        self.assertTrue(self.directory.name.startswith('position-cache-test-'))
        self.temp.cleanup()

    def compute(self, codes=None):
        return validate_arrangement(self.tiles, self.codes if codes is None else codes)

    def test_exact_repeat_computes_once(self):
        calls = []
        def compute():
            calls.append(True)
            return self.compute()
        with PositionCache(self.path, 'puzzle-v1') as cache:
            first, hit = cache.get_or_compute(self.codes, compute)
            self.assertFalse(hit)
            second, hit = cache.get_or_compute(tuple(self.codes), compute)
            self.assertTrue(hit)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            stats = cache.stats()
            self.assertEqual((stats['unique_entries'], stats['hits'], stats['misses']), (1, 1, 1))
            self.assertEqual((stats['session_hits'], stats['session_misses']), (1, 1))
            self.assertGreater(stats['logical_database_bytes'], 0)

    def test_changed_rotation_and_order_are_distinct_exact_keys(self):
        rotated = list(self.codes)
        rotated[42] += 1
        reordered = list(self.codes)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        calls = []
        with PositionCache(self.path, 'puzzle-v1') as cache:
            for codes in (self.codes, rotated, reordered):
                def compute(codes=codes):
                    calls.append(tuple(codes))
                    return self.compute(codes)
                report, hit = cache.get_or_compute(codes, compute)
                self.assertFalse(hit)
                self.assertEqual(report, self.compute(codes))
            self.assertEqual(cache.stats()['unique_entries'], 3)
            self.assertEqual(len(calls), 3)
            for codes in (self.codes, rotated, reordered):
                _, hit = cache.get_or_compute(codes, lambda: self.fail('exact row was not reused'))
                self.assertTrue(hit)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            rows = connection.execute('SELECT board FROM position_cache_entries').fetchall()
        self.assertEqual(len({row[0] for row in rows}), 3)
        self.assertTrue(all(len(row[0]) == 320 for row in rows))

    def test_reopen_reuses_disk_and_resets_only_session_counters(self):
        with PositionCache(self.path, 'puzzle-v1') as cache:
            cache.get_or_compute(self.codes, self.compute)
        with PositionCache(self.path, 'puzzle-v1') as cache:
            before = cache.stats()
            self.assertEqual((before['unique_entries'], before['misses']), (1, 1))
            self.assertEqual((before['session_hits'], before['session_misses']), (0, 0))
            report, hit = cache.get_or_compute(self.codes, lambda: self.fail('disk result missing'))
            self.assertTrue(hit)
            self.assertEqual(report, self.real_report)
            self.assertEqual(cache.stats()['session_hits'], 1)

    def test_namespaces_are_isolated(self):
        with PositionCache(self.path, 'data-a_topology-a_validator-a') as first:
            first.get_or_compute(self.codes, self.compute)
        with PositionCache(self.path, 'data-a_topology-a_validator-b') as second:
            _, hit = second.get_or_compute(self.codes, self.compute)
            self.assertFalse(hit)
            self.assertEqual(second.stats()['unique_entries'], 1)
            self.assertEqual(second.stats()['hits'], 0)
        with PositionCache(self.path, 'data-a_topology-a_validator-a') as first:
            _, hit = first.get_or_compute(self.codes, lambda: self.fail('wrong namespace result'))
            self.assertTrue(hit)
            self.assertEqual(first.stats()['misses'], 1)

    def test_rejects_noncomplete_or_noninteger_placements(self):
        cases = [self.codes[:-1], self.codes + [0], '0' * 160, bytes(160), None]
        for replacement in (-1, 480, 65536, 0.0, True, '0', None):
            invalid = list(self.codes)
            invalid[0] = replacement
            cases.append(invalid)
        duplicate = list(self.codes)
        duplicate[1] = 1  # Same physical tile as code 0, different rotation.
        cases.append(duplicate)
        with PositionCache(self.path, 'puzzle-v1') as cache:
            for invalid in cases:
                with self.subTest(input=str(invalid)[:40]):
                    with self.assertRaises(ValueError):
                        cache.get_or_compute(invalid, lambda: self.fail('invalid input reached validator'))
            self.assertEqual(cache.stats()['unique_entries'], 0)

    def test_packed_key_is_exact_little_endian_uint16(self):
        packed = pack_position(self.codes)
        self.assertEqual(len(packed), 320)
        self.assertEqual(struct.unpack('<160H', packed), tuple(self.codes))
        self.assertEqual(packed[:4], b'\x00\x00\x03\x00')

    def test_failed_callback_rolls_back_without_counters_or_entry(self):
        def fail():
            raise LookupError('validation deliberately failed')
        with PositionCache(self.path, 'puzzle-v1') as cache:
            with self.assertRaisesRegex(LookupError, 'deliberately'):
                cache.get_or_compute(self.codes, fail)
            self.assertEqual(cache.stats()['unique_entries'], 0)
            self.assertEqual(cache.stats()['misses'], 0)
            _, hit = cache.get_or_compute(self.codes, self.compute)
            self.assertFalse(hit)

    def test_malformed_callback_report_cannot_enter_cache(self):
        malformed = [None, {}, {'valid': True}]
        bad = copy.deepcopy(self.real_report)
        bad['valid'] = True
        malformed.append(bad)
        bad = copy.deepcopy(self.real_report)
        bad['component_lengths'][0] += 1
        malformed.append(bad)
        bad = copy.deepcopy(self.real_report)
        bad['optional_metric'] = float('nan')
        malformed.append(bad)
        with PositionCache(self.path, 'puzzle-v1') as cache:
            for report in malformed:
                with self.assertRaises(PositionCacheError):
                    cache.get_or_compute(self.codes, lambda report=report: report)
            self.assertEqual(cache.stats()['unique_entries'], 0)
            self.assertEqual(cache.stats()['misses'], 0)

    def test_returned_report_mutation_does_not_mutate_disk(self):
        with PositionCache(self.path, 'puzzle-v1') as cache:
            report, _ = cache.get_or_compute(self.codes, self.compute)
            report['errors'].append('caller edit')
            report['matched_edges'] = 240
            restored, hit = cache.get_or_compute(self.codes, lambda: self.fail('cache miss'))
            self.assertTrue(hit)
            self.assertEqual(restored, self.real_report)

    def test_corrupt_json_or_integrity_is_recomputed_as_miss(self):
        with PositionCache(self.path, 'puzzle-v1') as cache:
            cache.get_or_compute(self.codes, self.compute)
            with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
                connection.execute("UPDATE position_cache_entries SET report_json='broken JSON'")
            report, hit = cache.get_or_compute(self.codes, self.compute)
            self.assertFalse(hit)
            self.assertEqual(report, self.real_report)
            self.assertEqual(cache.stats()['repairs'], 1)
            self.assertEqual(cache.stats()['unique_entries'], 1)
            self.assertEqual(cache.stats()['misses'], 2)

    def test_malformed_report_with_consistent_digest_is_revalidated(self):
        namespace = 'puzzle-v1'
        with PositionCache(self.path, namespace) as cache:
            cache.get_or_compute(self.codes, self.compute)
            payload = json.dumps({'valid': True})
            digest = _payload_digest(namespace, pack_position(self.codes), payload)
            with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
                connection.execute('UPDATE position_cache_entries SET report_json=?, integrity_sha256=?', (payload, digest))
            report, hit = cache.get_or_compute(self.codes, self.compute)
            self.assertFalse(hit)
            self.assertFalse(report['valid'])
            self.assertEqual(cache.stats()['session_repairs'], 1)

    def test_integrity_binds_report_to_exact_board_not_only_payload(self):
        rotated = list(self.codes)
        rotated[5] += 1
        with PositionCache(self.path, 'puzzle-v1') as cache:
            cache.get_or_compute(self.codes, self.compute)
            with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
                connection.execute('UPDATE position_cache_entries SET board=?', (pack_position(rotated),))
            calls = []
            def compute_rotated():
                calls.append(True)
                return self.compute(rotated)
            report, hit = cache.get_or_compute(rotated, compute_rotated)
            self.assertFalse(hit)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report, self.compute(rotated))

    def test_two_connections_racing_compute_once(self):
        barrier = threading.Barrier(2)
        count_lock = threading.Lock()
        calls = []
        def worker(cache):
            barrier.wait(timeout=5)
            def compute():
                with count_lock:
                    calls.append(True)
                time.sleep(0.03)
                return self.compute()
            return cache.get_or_compute(self.codes, compute)
        with PositionCache(self.path, 'puzzle-v1') as a, PositionCache(self.path, 'puzzle-v1') as b:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, cache) for cache in (a, b)]
                results = [future.result(timeout=10) for future in futures]
            self.assertEqual(sorted(hit for _, hit in results), [False, True])
            self.assertEqual(len(calls), 1)
            self.assertEqual(a.stats()['unique_entries'], 1)
            self.assertEqual(a.stats()['misses'], 1)
            self.assertEqual(a.stats()['hits'], 1)

    def test_closed_cache_rejects_use_and_close_is_idempotent(self):
        cache = PositionCache(self.path, 'puzzle-v1')
        cache.close()
        cache.close()
        with self.assertRaisesRegex(PositionCacheError, 'closed'):
            cache.stats()
        with self.assertRaisesRegex(PositionCacheError, 'closed'):
            cache.get_or_compute(self.codes, self.compute)

    def test_invalid_database_and_namespace_fail_clearly(self):
        for namespace in ('', '  ', None, 1, 'x\0y'):
            with self.assertRaises(ValueError):
                PositionCache(self.path, namespace)
        self.path.write_bytes(b'not a sqlite database')
        with self.assertRaisesRegex(PositionCacheError, 'Cannot open position cache'):
            PositionCache(self.path, 'puzzle-v1')

    def test_unknown_schema_does_not_silently_reuse_results(self):
        with PositionCache(self.path, 'puzzle-v1'):
            pass
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute('PRAGMA user_version=987')
        with self.assertRaisesRegex(PositionCacheError, 'Unsupported'):
            PositionCache(self.path, 'puzzle-v1')


if __name__ == '__main__':
    unittest.main()
