"""Adverse original-scope masks and the actual two-worker CPU queue."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bench_r35_coverage_reads import MeasuredLedger
from bench_r35_parked_cache import original_residual
from constraint_work import WorkScope, _CoverageMaskTable
from search_problem import SearchProblem


def entropy(tiles, folder):
    rng = Random(3565)
    fixed, witness = original_residual(tiles)
    scope = WorkScope.for_problem(SearchProblem(tiles))
    restricted = WorkScope.for_problem(fixed).root
    a, b = witness['cell_a'], witness['cell_b']
    centers = [tuple(rng.getrandbits(v.bit_length()) & v or v for v in scope.root) for _ in range(8)]
    output = {}
    for pattern in ('unique', 'clustered', 'interleaved'):
        path = folder/(pattern + '.sqlite')
        with MeasuredLedger(path, scope) as writer:
            for i in range(512):
                if pattern == 'unique':
                    row = [rng.getrandbits(v.bit_length()) & v or v for v in scope.root]
                else:
                    row = list(centers[i//64 if pattern == 'clustered' else i%8])
                    axis = next(j for j in range(1 + i%150, 160) if j not in (a,b))
                    row[axis] = rng.getrandbits(row[axis].bit_length()) & row[axis] or row[axis]
                row[a], row[b] = restricted[a], restricted[b]
                assert writer.record_completed(row, status='infeasible', complete=True,
                    evidence={'kind':'fixed_mismatch','seam':witness})
        outside = list(scope.root)
        outside[a] ^= restricted[a]
        query = tuple(outside)
        for budget in (262144, 4194304, 8388608, 16777216):
            modes = [False,True] if hasattr(_CoverageMaskTable,'literal_fallback') else [False]
            for use_literal in modes:
                selector = patch.object(_CoverageMaskTable,'literal_fallback',use_literal) if len(modes)>1 else nullcontext()
                with selector, MeasuredLedger(path,scope) as reader:
                    reader._coverage_cache_limit = budget
                    for _ in range(5):
                        assert reader.remaining([query]) == ((query,),True)
                    assert tuple(reader.iter_exclusions()) == reader.exclusions()
                    output[f'{pattern}/{budget}/{use_literal}'] = dict(rows=len(reader._coverage_prefix),
                        reads=reader.stats['coverage_records_checked'],bytes=reader._coverage_bytes)
    return output


def memory():
    """Charge new integers to their cached rows; exclude the shared immutable target root."""
    import gc
    import tracemalloc
    root=((1<<480)-1,)*160
    results={}
    for mode in ('unique','eight_families','singleton'):
        rng=Random(35065)
        size=0
        # Allocate seed descriptions before tracing; make independent fresh values below.
        centers=[tuple(rng.getrandbits(480) or 1 for _ in root) for _ in range(8)]
        tracemalloc.start()
        table=_CoverageMaskTable(root)
        records=[]
        for i in range(4096):
            if mode=='unique':
                row=tuple(rng.getrandbits(480) or 1 for _ in root)
            elif mode=='eight_families':
                row=[int.from_bytes(v.to_bytes(60,'little'),'little') for v in centers[i%8]]
                row[i%160]=rng.getrandbits(480) or 1
                row=tuple(row)
            else:
                row=tuple(1 << ((j+(i%8 if j%5==0 else 0))%480) for j in range(160))
            encoded=table.encode(row,8388608-size)
            if encoded is None:
                break
            record,used=encoded
            records.append(record)
            size+=used
        del row,encoded
        gc.collect()
        actual,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[mode]=dict(rows=len(records),estimated=size,traced_current=actual,traced_peak=peak)
    return results


def parallel(folder, cap):
    from constraint_parallel import solve
    from test_constraint_dfs import board, brute
    from test_constraint_parallel import legacy_root, OPTIONS
    from gpu_domain_runner import split_cubes
    problem = SearchProblem([{'id':i,'segments':[[[0,2],[1,10]],[[0,10],[1,2]]]} for i in range(5)],
        board(5,[(i,1,(i+1)%5,0) for i in range(5)]))
    assert not brute(problem,'gold') and len(brute(problem,'edges'))==120
    path = folder/f'cpu-{cap}.sqlite'
    scope = WorkScope.for_problem(problem)
    with MeasuredLedger(path,scope) as writer:
        for row in split_cubes([scope.root],32)[::2]:
            writer.record_completed(row,status='infeasible',complete=True,evidence={'kind':'finite_enumeration'})
    instances = []
    def reader(*args, **kwargs):
        db = MeasuredLedger(*args,**kwargs)
        db._coverage_cache_limit = cap
        instances.append(db)
        return db
    with patch('constraint_parallel.CoverageLedger',side_effect=reader):
        result = solve(problem, seconds=120,workers=2,resume=legacy_root(problem),
            options={**OPTIONS,'shared_coverage':True},learning_state={'_coverage_path':path})
    assert result['status']=='infeasible' and result['complete'] and result['parallel']['pending_jobs']==0, result
    db, = instances
    return dict(checked=db.stats['coverage_records_checked'],payload=db.checked_bytes,
        solved_jobs=result['stats']['parallel_jobs_completed'], workers=result['stats']['parallel_workers_started'],
        nodes=result['nodes'],pending=result['parallel']['pending_jobs'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--memory-only',action='store_true')
    args=cli.parse_args()
    if args.memory_only:
        args.output.write_text(json.dumps(memory(),indent=2)+'\n',encoding='utf8')
        raise SystemExit
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        report={'entropy':entropy(json.loads(args.tiles.read_bytes()),folder),
                'cpu':{str(cap):parallel(folder,cap) for cap in (0,4194304)}}
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
