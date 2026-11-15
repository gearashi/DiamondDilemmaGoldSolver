"""Compare packed mask IDs and verified proof-root factoring on actual SQLite consumers."""
import argparse
import json
from pathlib import Path
from sys import getsizeof
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bench_r35_coverage_blocks import mixed_families
from bench_r35_coverage_consumers import dense_proofs, budget_matrix, sat_slices
from bench_r35_coverage_reads import MeasuredLedger
from constraint_work import intersect
from r34_coverage_index import scope_id, version


class SourceLedger(MeasuredLedger):
    """Store each checked source root once; represent descendants by sparse coordinate deltas."""
    def _use_checked_coverage(self, target, hull, check, consume):
        self.db.execute('BEGIN')
        try:
            group = scope_id(self.db,target)
            epoch, through = version(self.db,group)
            key = target,group,epoch
            if self._coverage_key != key:
                self._coverage_key = key
                self._coverage_after = self._coverage_bytes = 0
                self._coverage_prefix = []
                self._roots = {target.root:target.root}
            def records():
                valid_roots = {}
                for item in self._coverage_prefix:
                    check()
                    if item is None:
                        continue
                    root,delta = item
                    ident = id(root)
                    if ident not in valid_roots:
                        valid_roots[ident] = all(a & b for a,b in zip(root,hull))
                    if valid_roots[ident] and all(mask & hull[i] for i,mask in delta):
                        row = list(root)
                        for i,mask in delta:
                            row[i]=mask
                        yield tuple(row)
                if self._coverage_after == through:
                    return
                sql = """SELECT s.seq,c.scope,c.box,c.evidence,c.checksum
                    FROM coverage_sequence s JOIN closed c ON c.identity=s.identity
                    WHERE s.scope_id=? AND s.seq>? AND s.seq<=? ORDER BY s.seq"""
                cursor = self.db.execute(sql,(group,self._coverage_after,through))
                can_extend = True
                try:
                    for seq,*columns in cursor:
                        check()
                        row = self._verified_coverage_row(target,*columns)
                        if can_extend:
                            entry=None
                            new_root=None
                            size=16
                            if row is not None:
                                # The original payload passed checksum and scope validation above.
                                candidate=intersect(tuple(int(v,16) for v in json.loads(columns[0])['root']),target.root)
                                root=self._roots.get(candidate)
                                if root is None:
                                    root=new_root=candidate
                                    size += 96 + getsizeof(root) + sum(getsizeof(v) for v in root)
                                delta=tuple((i,v) for i,v in enumerate(row) if v!=root[i])
                                entry=root,delta
                                size+=getsizeof(entry)+getsizeof(delta)
                                size+=sum(getsizeof(p)+getsizeof(p[0])+getsizeof(p[1]) for p in delta)
                            can_extend=(self._coverage_bytes+size <= self._coverage_cache_limit)
                            if can_extend:
                                if new_root is not None:
                                    self._roots[new_root]=new_root
                                self._coverage_prefix.append(entry)
                                self._coverage_bytes+=size
                                self._coverage_after=seq
                        if row is not None:
                            yield row
                finally:
                    cursor.close()
            stream=records()
            try:
                return consume(stream)
            finally:
                stream.close()
        finally:
            self.db.rollback()


def run(tiles,directory):
    engines = [('mask_ids',MeasuredLedger),('shared_root',SourceLedger)]
    results = {'dense':{},'sparse':{},'sat':{},'mixed':{}}
    for label,engine in engines:
        case=directory/label
        case.mkdir()
        results['dense'][label]=dense_proofs(tiles,case,engine=engine,budgets=(65536,262144,1048576,4194304))
        with patch('bench_r35_coverage_consumers.MeasuredLedger',engine):
            results['sparse'][label]=budget_matrix(tiles,case)
            results['sat'][label]=sat_slices(case,4194304)
    for families,splits in ((8,64),(32,16),(128,2)):
        results['mixed'][str(families)]=mixed_families(tiles,directory,families,splits,engines=engines)
    return results


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--large',action='store_true')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    with TemporaryDirectory() as tmp:
        tiles=json.loads(args.tiles.read_bytes())
        root=Path(tmp)
        if args.large:
            from constraint_work import _CoverageMaskTable
            result={}
            for count in (1024,16384):
                shared=root/str(count)
                shared.mkdir()
                for limit in (0,256):
                    with patch.object(_CoverageMaskTable,'anchor_limit',limit):
                        result[f'{count}/{limit}']=dense_proofs(tiles,shared,count=count,
                            budgets=(262144,1048576,4194304,16777216))
        else:
            result=run(tiles,root)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
