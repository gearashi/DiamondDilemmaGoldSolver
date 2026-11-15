"""Measure real SQLite payloads passed to the verifier at the normal remaining-work entry point."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from bench_r35_coverage_selectivity import benchmark
from constraint_symbolic_work import DomainDiagram
from constraint_work import CoverageLedger, WorkScope, subtract_many


class MeasuredLedger(CoverageLedger):
    checked_bytes = 0

    def _verified_coverage_row(self, target, *columns):
        self.checked_bytes += sum(len(column.encode()) for column in columns)
        return super()._verified_coverage_row(target, *columns)


class AxisLedger(MeasuredLedger):
    """Competing partial index. Possible hits still fetch and verify the full proof."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._projections = {}
        self._projection_limit = 4096

    def _selective_remaining(self, source, target, hull, max_boxes, check):
        from r34_coverage_index import scope_id, version
        # The version and all rows must come from the same committed snapshot.
        self.db.execute('BEGIN')
        try:
            group = scope_id(self.db, target)
            epoch, through = version(self.db, group)
            key = (target, group, epoch)
            if self._coverage_key != key:
                self._projections.clear()
                self._coverage_key = key
            skip = []
            for seq, axes in self._projections.items():
                check()
                if any(not (hull[axis] & mask) for axis, mask in axes):
                    skip.append(seq)
            sql = """SELECT s.seq,c.scope,c.box,c.evidence,c.checksum
                FROM coverage_sequence s JOIN closed c ON c.identity=s.identity
                WHERE s.scope_id=? AND s.seq<=?"""
            params = [group, through]
            if skip:
                sql += " AND s.seq NOT IN (SELECT value FROM json_each(?))"
                params.append(json.dumps(skip))
            cursor = self.db.execute(sql + " ORDER BY s.seq", params)
            def records():
                for seq, scope_json, box_json, evidence_json, checksum in cursor:
                    check()
                    row = self._verified_coverage_row(target, scope_json, box_json, evidence_json, checksum)
                    if row is None:
                        continue
                    if seq not in self._projections and len(self._projections) < self._projection_limit:
                        axes = [i for i, mask in enumerate(row) if mask != target.root[i]]
                        axes.sort(key=lambda i: row[i].bit_count() / target.root[i].bit_count())
                        if axes:
                            self._projections[seq] = tuple((i, row[i]) for i in axes[:4])
                    yield row
            try:
                return subtract_many(source, records(), max_boxes=max_boxes, check=check)
            finally:
                cursor.close()
        finally:
            self.db.rollback()

def measured(problem, covers, queries):
    scope = WorkScope.for_problem(problem)
    out = {}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'coverage.sqlite'
        with CoverageLedger(path, scope) as writer:
            for box, evidence in covers:
                writer.record_completed(box, status='infeasible', complete=True, evidence=evidence)
        for label, batches in queries.items():
            out[label] = {}
            flat = []
            for mode in ('stream', 'four_axes', 'prefix', 'symbolic'):
                engine = AxisLedger if mode == 'four_axes' else MeasuredLedger
                with engine(path, scope, symbolic=mode == 'symbolic') as reader:
                    if mode == 'stream':
                        reader._coverage_cache_limit = 0
                    elapsed = 0
                    for i, pending in enumerate(batches):
                        start = perf_counter()
                        actual, done = reader.remaining(pending)
                        elapsed += perf_counter() - start
                        assert done
                        if mode == 'stream':
                            flat.append(actual)
                        elif mode in ('four_axes','prefix'):
                            assert actual == flat[i]
                        else:
                            diagram = DomainDiagram(scope.root, max_nodes=200000)
                            assert diagram.union(actual) == diagram.union(flat[i])
                    out[label][mode] = dict(queries=len(batches), rows=reader.stats['coverage_records_checked'],
                                            payload_bytes=reader.checked_bytes, seconds=round(elapsed,4),
                                            indexed_sequences=len(getattr(reader, '_projections', reader._coverage_prefix)),
                                            retained_bytes=reader._coverage_bytes)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(json.loads(args.tiles.read_bytes()), measured)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
