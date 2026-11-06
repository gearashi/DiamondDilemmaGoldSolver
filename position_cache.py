"""Persistent exact full-board validation-result cache for Diamond Dilemma Gold.

The primary key is (namespace, 320-byte little-endian uint16 placement), not a
hash of the board. Every input must use each of the 160 physical tiles once;
code = 3 * tile_index + rotation, and only rotations 0..2 are allowed.

The caller supplies a namespace covering the puzzle data, topology, and
validator implementation. ``get_or_compute(codes, compute)`` invokes the
zero-argument ``compute()`` only on a miss and returns ``(report, hit)``.
The callback must validate the same placement. Cached ``valid=True`` remains
a hint: a solver must independently revalidate before declaring a solution.

SQLite serializes lookup + compute + store in one transaction, so concurrent
processes do not compute the same missing board twice. Do not recursively
access this cache from a compute callback. WAL supports concurrent readers;
no entry eviction occurs. Counts are per namespace and count only successful
API returns, with separate counts for this PositionCache instance.
"""
from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path
import sqlite3
import struct
import threading
from typing import Any, Callable, Iterable

TILE_COUNT = 160
ORIENTATION_COUNT = 480
BOARD_BYTES = TILE_COUNT * 2
_SCHEMA_VERSION = 1
_MAX_REPORT_BYTES = 2_000_000


class PositionCacheError(RuntimeError):
    """The database, callback report, or cache lifecycle is invalid."""


def pack_position(codes: Iterable[int]) -> bytes:
    """Encode an exact complete placement; reject duplicates and coercions."""
    if isinstance(codes, (str, bytes, bytearray, memoryview)):
        raise ValueError('placement must be an iterable of 160 integer codes')
    try:
        values = list(codes)
    except TypeError as exc:
        raise ValueError('placement must be an iterable of 160 integer codes') from exc
    if len(values) != TILE_COUNT:
        raise ValueError(f'placement must contain exactly {TILE_COUNT} codes')
    normalized = []
    for index, code in enumerate(values):
        if isinstance(code, bool) or not isinstance(code, numbers.Integral):
            raise ValueError(f'placement code {index} must be an integer, not a coercible value')
        code = int(code)
        if not 0 <= code < ORIENTATION_COUNT:
            raise ValueError(f'placement code {index} must be in [0, {ORIENTATION_COUNT})')
        normalized.append(code)
    if len({code // 3 for code in normalized}) != TILE_COUNT:
        raise ValueError('placement must use every physical tile exactly once')
    return struct.pack('<160H', *normalized)


def _integer(value: Any, low: int, high: int, field: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{field} must be an integer in [{low}, {high}]')
    return value


def _validate_report(report: Any) -> dict:
    """Reject malformed/internally inconsistent JSON validation results."""
    if not isinstance(report, dict):
        raise ValueError('validation report must be an object')
    for key in ('valid', 'unique_tiles', 'all_closed'):
        if type(report.get(key)) is not bool:
            raise ValueError(f'{key} must be a boolean')
    for key in ('placement_count', 'tile_count'):
        _integer(report.get(key), TILE_COUNT, TILE_COUNT, key)
    if not report['unique_tiles']:
        raise ValueError('report disagrees with the complete unique placement')
    _integer(report.get('total_edges'), 240, 240, 'total_edges')
    matched = _integer(report.get('matched_edges'), 0, 240, 'matched_edges')
    segments = _integer(report.get('total_segments'), 1, 10000, 'total_segments')
    components = _integer(report.get('line_components'), 1, segments, 'line_components')
    lengths = report.get('component_lengths')
    if not isinstance(lengths, list) or len(lengths) != components:
        raise ValueError('component_lengths must describe every line component')
    for length in lengths:
        _integer(length, 1, segments, 'component length')
    if sum(lengths) != segments:
        raise ValueError('component lengths do not sum to total_segments')
    errors = report.get('errors')
    if not isinstance(errors, list) or any(not isinstance(e, str) or not e for e in errors):
        raise ValueError('errors must be a list of nonempty strings')
    unmatched = report.get('unmatched_edges')
    if not isinstance(unmatched, list) or len(unmatched) != 240 - matched:
        raise ValueError('unmatched_edges does not agree with matched_edges')
    seen = set()
    for edge in unmatched:
        if not isinstance(edge, dict):
            raise ValueError('each unmatched edge must be an object')
        a = _integer(edge.get('cell_a'), 0, 159, 'cell_a')
        b = _integer(edge.get('cell_b'), 0, 159, 'cell_b')
        sa = _integer(edge.get('side_a'), 0, 2, 'side_a')
        sb = _integer(edge.get('side_b'), 0, 2, 'side_b')
        key = tuple(sorted(((a, sa), (b, sb))))
        if a == b or key in seen:
            raise ValueError('unmatched_edges contains a repeated or self edge')
        seen.add(key)
        point_sets = []
        for field in ('points_a', 'reversed_points_b'):
            points = edge.get(field)
            if not isinstance(points, list):
                raise ValueError(f'{field} must be a list')
            for point in points:
                _integer(point, 1, 11, field)
            if len(set(points)) != len(points):
                raise ValueError(f'{field} repeats an endpoint')
            point_sets.append(set(points))
        if point_sets[0] == point_sets[1]:
            raise ValueError('an unmatched edge has equal endpoint sets')
    if report['all_closed'] != (matched == 240):
        raise ValueError('all_closed does not agree with the complete edge match count')
    if report['valid']:
        if errors or matched != 240 or components != 1 or not report['all_closed']:
            raise ValueError('valid=True contradicts the validation details')
    elif not errors:
        raise ValueError('an invalid report must explain its validation errors')
    return report


def _serialize_report(report: Any) -> str:
    _validate_report(report)
    try:
        payload = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f'validation report is not finite JSON: {exc}') from exc
    if len(payload.encode('utf-8')) > _MAX_REPORT_BYTES:
        raise ValueError('validation report exceeds the cache payload limit')
    return payload


def _payload_digest(namespace: str, board: bytes, payload: str) -> str:
    """Integrity digest; board identity still uses the complete BLOB key."""
    namespace_bytes = namespace.encode('utf-8')
    return hashlib.sha256(struct.pack('<I', len(namespace_bytes)) + namespace_bytes + board + payload.encode('utf-8')).hexdigest()


class PositionCache:
    """SQLite-backed exact placement cache, safe across threads/processes."""

    def __init__(self, path: str | Path, namespace: str, *, timeout: float = 30.0):
        if not isinstance(namespace, str) or not namespace.strip() or '\0' in namespace or len(namespace) > 1024:
            raise ValueError('namespace must be a nonempty string of at most 1024 characters without NUL')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError('timeout must be a positive number of seconds at most 300')
        self.path = Path(path).expanduser().resolve()
        self.namespace = namespace
        self._lock = threading.RLock()
        self._closed = False
        self._session_hits = self._session_misses = self._session_repairs = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(str(self.path), timeout=timeout, isolation_level=None, check_same_thread=False)
            self._connection.execute(f'PRAGMA busy_timeout={int(timeout * 1000)}')
            self._connection.execute('PRAGMA journal_mode=WAL')
            self._connection.execute('PRAGMA synchronous=FULL')
            self._initialize()
        except (sqlite3.Error, OSError) as exc:
            connection = getattr(self, '_connection', None)
            if connection is not None:
                connection.close()
            raise PositionCacheError(f'Cannot open position cache {self.path}: {exc}') from exc
        except Exception:
            connection = getattr(self, '_connection', None)
            if connection is not None:
                connection.close()
            raise

    def _initialize(self) -> None:
        connection = self._connection
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, _SCHEMA_VERSION):
            raise PositionCacheError(f'Unsupported position cache schema version {version}')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        expected = {'position_cache_entries', 'position_cache_stats'}
        if tables - expected:
            raise PositionCacheError('Position cache path contains an unrelated SQLite database')
        connection.execute('BEGIN IMMEDIATE')
        try:
            connection.execute('''CREATE TABLE IF NOT EXISTS position_cache_entries (
                namespace TEXT NOT NULL,
                board BLOB NOT NULL CHECK(length(board) = 320),
                report_json TEXT NOT NULL,
                integrity_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY(namespace, board)
            ) WITHOUT ROWID''')
            connection.execute('''CREATE TABLE IF NOT EXISTS position_cache_stats (
                namespace TEXT PRIMARY KEY,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0,
                repairs INTEGER NOT NULL DEFAULT 0
            ) WITHOUT ROWID''')
            actual = {row[1] for row in connection.execute('PRAGMA table_info(position_cache_entries)')}
            if actual != {'namespace', 'board', 'report_json', 'integrity_sha256', 'created_at'}:
                raise PositionCacheError('Position cache entry schema is invalid')
            actual_stats = {row[1] for row in connection.execute('PRAGMA table_info(position_cache_stats)')}
            if actual_stats != {'namespace', 'hits', 'misses', 'repairs'}:
                raise PositionCacheError('Position cache statistics schema is invalid')
            connection.execute('INSERT OR IGNORE INTO position_cache_stats(namespace) VALUES (?)', (self.namespace,))
            connection.execute(f'PRAGMA user_version={_SCHEMA_VERSION}')
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise PositionCacheError('Position cache is closed')

    def get_or_compute(self, codes: Iterable[int], compute: Callable[[], dict]) -> tuple[dict, bool]:
        """Return (report, hit), recomputing absent or corrupted result rows.

        A corrupt payload is repaired only after a successful callback result.
        Callback exceptions propagate unchanged and the transaction rolls back.
        Do not call this cache from inside the callback. A returned report is a
        new JSON object; caller mutations cannot change the stored result.
        """
        board = pack_position(codes)
        if not callable(compute):
            raise TypeError('compute must be a zero-argument callable')
        with self._lock:
            self._ensure_open()
            connection = self._connection
            try:
                connection.execute('BEGIN IMMEDIATE')
                row = connection.execute('SELECT report_json, integrity_sha256 FROM position_cache_entries WHERE namespace=? AND board=?', (self.namespace, board)).fetchone()
                corrupt = False
                if row is not None:
                    payload, digest = row
                    try:
                        if not isinstance(payload, str) or not isinstance(digest, str) or len(payload.encode('utf-8')) > _MAX_REPORT_BYTES:
                            raise ValueError('invalid payload storage')
                        if _payload_digest(self.namespace, board, payload) != digest:
                            raise ValueError('payload integrity check failed')
                        report = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON constant {value}')))
                        _validate_report(report)
                    except (TypeError, ValueError, UnicodeError, RecursionError):
                        corrupt = True
                    else:
                        connection.execute('UPDATE position_cache_stats SET hits=hits+1 WHERE namespace=?', (self.namespace,))
                        connection.commit()
                        self._session_hits += 1
                        return report, True
                report = compute()
                try:
                    payload = _serialize_report(report)
                except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
                    raise PositionCacheError(f'Validation callback returned a malformed report: {exc}') from exc
                connection.execute('''INSERT INTO position_cache_entries(namespace, board, report_json, integrity_sha256)
                    VALUES (?, ?, ?, ?) ON CONFLICT(namespace, board) DO UPDATE SET
                    report_json=excluded.report_json, integrity_sha256=excluded.integrity_sha256,
                    created_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')''',
                    (self.namespace, board, payload, _payload_digest(self.namespace, board, payload)))
                connection.execute('UPDATE position_cache_stats SET misses=misses+1, repairs=repairs+? WHERE namespace=?', (int(corrupt), self.namespace))
                connection.commit()
                self._session_misses += 1
                self._session_repairs += int(corrupt)
                return json.loads(payload), False
            except sqlite3.Error as exc:
                if connection.in_transaction:
                    connection.rollback()
                raise PositionCacheError(f'Position cache transaction failed for {self.path}: {exc}') from exc
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def stats(self) -> dict:
        """Namespace-lifetime counts plus counts for this open instance.

        A hit is a successful cached return; a miss is a successfully computed
        return. Repairs are the subset of misses replacing corrupt rows. Failed
        callbacks/transactions do not increment counts. Logical bytes include
        every namespace in this database, not only the active namespace.
        """
        with self._lock:
            self._ensure_open()
            try:
                connection = self._connection
                entries = connection.execute('SELECT COUNT(*) FROM position_cache_entries WHERE namespace=?', (self.namespace,)).fetchone()[0]
                hits, misses, repairs = connection.execute('SELECT hits, misses, repairs FROM position_cache_stats WHERE namespace=?', (self.namespace,)).fetchone()
                page_count = connection.execute('PRAGMA page_count').fetchone()[0]
                page_size = connection.execute('PRAGMA page_size').fetchone()[0]
                return {'namespace': self.namespace, 'unique_entries': entries, 'hits': hits, 'misses': misses,
                        'repairs': repairs, 'session_hits': self._session_hits,
                        'session_misses': self._session_misses, 'session_repairs': self._session_repairs,
                        'logical_database_bytes': page_count * page_size,
                        'counter_semantics': 'successful returns; lifetime namespace counters and current-instance session counters; repairs are included in misses'}
            except sqlite3.Error as exc:
                raise PositionCacheError(f'Cannot read position cache statistics: {exc}') from exc

    def close(self) -> None:
        """Close the connection; persisted entries remain available on reopen."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> 'PositionCache':
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
