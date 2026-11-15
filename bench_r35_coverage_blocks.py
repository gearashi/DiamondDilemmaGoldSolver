"""Compare shared verified axis blocks on many fixed-cell proofs, including mixed boards."""
import argparse
import json
from pathlib import Path
from sys import getsizeof
from tempfile import TemporaryDirectory

from bench_r35_coverage_consumers import dense_proofs
from bench_r35_coverage_reads import MeasuredLedger
from r34_coverage_index import scope_id, version


class BlockLedger(MeasuredLedger):
    width = 8

    def _use_checked_coverage(self, target, hull, check, consume):
        self.db.execute('BEGIN')
        try:
            group = scope_id(self.db, target)
            epoch, through = version(self.db, group)
            if self._coverage_key != (target, group, epoch):
                self._coverage_key = target, group, epoch
                self._coverage_prefix = []
                self._coverage_after = 0
                self._coverage_bytes = 0
                self._block_ids = {}
                self._blocks = []

            def records():
                possible = {}
                for ids in self._coverage_prefix:
                    check()
                    if ids is None:
                        continue
                    for ident in ids:
                        if ident not in possible:
                            possible[ident] = all(hull[i] & mask for i, mask in self._blocks[ident])
                        if not possible[ident]:
                            break
                    else:
                        row = list(target.root)
                        for ident in ids:
                            for i, mask in self._blocks[ident]:
                                row[i] = mask
                        yield tuple(row)
                if self._coverage_after == through:
                    return
                cursor = self.db.execute("""SELECT s.seq,c.scope,c.box,c.evidence,c.checksum
                    FROM coverage_sequence s JOIN closed c ON c.identity=s.identity
                    WHERE s.scope_id=? AND s.seq>? AND s.seq<=? ORDER BY s.seq""",
                    (group,self._coverage_after,through))
                can_extend = True
                try:
                    for seq,*columns in cursor:
                        check()
                        row = self._verified_coverage_row(target,*columns)
                        if can_extend:
                            ids = None
                            new = []
                            if row is not None:
                                parts = []
                                for start in range(0,len(row),self.width):
                                    block = tuple((i,row[i]) for i in range(start,min(len(row),start+self.width))
                                                  if row[i] != target.root[i])
                                    if not block:
                                        continue
                                    ident = self._block_ids.get(block)
                                    if ident is None:
                                        ident = len(self._blocks) + len(new)
                                        new.append((ident,block))
                                    parts.append(ident)
                                ids = tuple(parts)
                            size = 16 + getsizeof(ids)
                            for ident,block in new:
                                size += 112 + getsizeof(ident) + getsizeof(block)
                                size += sum(getsizeof(pair) + getsizeof(pair[0]) + getsizeof(pair[1]) for pair in block)
                            can_extend = self._coverage_bytes + size <= self._coverage_cache_limit
                            if can_extend:
                                for ident,block in new:
                                    self._block_ids[block] = ident
                                    self._blocks.append(block)
                                self._coverage_prefix.append(ids)
                                self._coverage_bytes += size
                                self._coverage_after = seq
                        if row is not None:
                            yield row
                finally:
                    cursor.close()
            stream = records()
            try:
                return consume(stream)
            finally:
                stream.close()
        finally:
            self.db.rollback()


def mixed_families(tiles, directory, family_count, splits, *, engines=None):
    from constraint_work import WorkScope
    from gpu_domain_runner import split_cubes
    from search_problem import SearchProblem
    from validator import validate_arrangement
    problem = SearchProblem(tiles)
    scope = WorkScope.for_problem(problem)
    codes = json.loads((Path(__file__).parent/'docs/R29_INCUMBENT.json').read_bytes())['codes']
    families = []
    path = directory/f'mixed-{family_count}.sqlite'
    with MeasuredLedger(path,scope) as db:
        for f in range(family_count):
            perm = codes[f:] + codes[:f]
            root = tuple(mask if i < 4 else 1 << perm[i] for i,mask in enumerate(scope.root))
            # Cell 4 takes a distinct physical tile in each family.
            assert all(root[4] & old[0].root[4] == 0 for old in families)
            report = validate_arrangement(tiles,perm,problem.board)
            seam = next(e for e in report['unmatched_edges'] if min(e['cell_a'],e['cell_b']) >= 4)
            assert seam['points_a'] != seam['reversed_points_b']
            families.append((WorkScope.for_problem(problem,root=root),split_cubes([root],splits),seam))
        for j in range(0,splits,2):
            for family,boxes,seam in families:
                db.record_completed(boxes[j],status='infeasible',complete=True,source_scope=family,
                                    evidence={'kind':'fixed_mismatch','seam':seam})
    out = {}
    if engines is None:
        engines = [(n, MeasuredLedger if n == 0 else BlockLedger) for n in (0,1,4,8,16,32)]
    for width, engine in engines:
        if engine is BlockLedger:
            BlockLedger.width = width
        runs = []
        for cap in (65536,4194304):
            with engine(path,scope) as db:
                db._coverage_cache_limit = cap
                # Wide consumers require the same positive proof order, including
                # the interleaving of families, on both the first and warm reads.
                expected = tuple(rows[j] for j in range(0,splits,2) for _,rows,_ in families)
                for _ in range(2):
                    assert db.exclusions() == expected
                for idx in (0,family_count-1):
                    rows = families[idx][1]
                    for j in (0,1,splits-2,splits-1):
                        answer = () if j % 2 == 0 else (rows[j],)
                        assert db.remaining([rows[j]]) == (answer,True)
                runs.append(dict(cap=cap,rows=db.stats['coverage_records_checked'],payload=db.checked_bytes,
                                 retained=db._coverage_bytes))
        out[str(width)] = runs
    return out


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--mixed',action='store_true')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    tiles=json.loads(args.tiles.read_bytes())
    result={}
    with TemporaryDirectory() as tmp:
        if args.mixed:
            for families,splits in ((8,64),(32,16),(128,2)):
                result[str(families)] = mixed_families(tiles,Path(tmp),families,splits)
        else:
            for width in (1,4,8,16,32):
                BlockLedger.width=width
                case=Path(tmp)/str(width)
                case.mkdir()
                result[str(width)]=dense_proofs(tiles,case,engine=BlockLedger,budgets=(1048576,4194304))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
