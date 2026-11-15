"""Check public solver consumers and the memory crossover of bounded coverage reuse."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bench_r35_coverage_reads import MeasuredLedger
from bench_r35_parked_cache import original_residual
from constraint_work import WorkScope, frontier_snapshot
from gpu_domain_runner import split_cubes


def budget_matrix(tiles,folder):
    p,seam = original_residual(tiles)
    scope = WorkScope.for_problem(p)
    boxes = split_cubes([scope.root],128)
    path = folder/'budget.sqlite'
    with MeasuredLedger(path,scope) as db:
        for row in boxes[::2]:
            db.record_completed(row,status='infeasible',complete=True,evidence={'kind':'fixed_mismatch','seam':seam})
    out=[]
    for cap in (0,4096,32768,65536,4194304,16777216):
        with MeasuredLedger(path,scope) as db:
            db._coverage_cache_limit=cap
            for i,row in enumerate(boxes):
                assert db.remaining([row]) == ((() if i%2==0 else (row,)),True)
            out.append(dict(cap=cap,checked=db.stats['coverage_records_checked'],payload=db.checked_bytes,
                            retained=db._coverage_bytes))
    return out


def dense_proofs(tiles, folder, *, engine=MeasuredLedger, budgets=(0,65536,1048576,4194304,16777216,33554432)):
    """Many closures share 156 fixed cells, read from the unrestricted original root."""
    from search_problem import SearchProblem
    original = SearchProblem(tiles)
    restricted, seam = original_residual(tiles)
    scope = WorkScope.for_problem(original)
    subroot = WorkScope.for_problem(restricted).root
    source = WorkScope.for_problem(original, root=subroot)
    boxes = split_cubes([subroot], 1024)
    assert len(boxes) == 1024
    path = folder/'dense.sqlite'
    with MeasuredLedger(path,scope) as writer:
        for box in boxes[::2]:
            writer.record_completed(box,status='infeasible',complete=True,source_scope=source,
                                    evidence={'kind':'fixed_mismatch','seam':seam})
    reports = []
    # Check both a covered query and an uncovered sibling at the beginning,
    # middle and end of a large proof stream, then reopen for each budget.
    for cap in budgets:
        with engine(path,scope) as db:
            db._coverage_cache_limit=cap
            for i in (1,0,2,3,512,513,1022,1023,1):
                expected = () if i%2==0 else (boxes[i],)
                assert db.remaining([boxes[i]]) == (expected,True)
            reports.append(dict(cap=cap,checked=db.stats['coverage_records_checked'],
                                payload=db.checked_bytes,estimated_bytes=db._coverage_bytes))
    return reports


def sat_slices(folder, cap):
    from constraint_sat_session import solve
    from test_constraint_learning_round25 import core_fixture
    p,scope,boxes=core_fixture()
    path=folder/'sat.sqlite'
    with MeasuredLedger(path,scope) as db:
        for row in boxes[:4]:
            db.record_completed(row,status='infeasible',complete=True,evidence={'kind':'duplicate_tile'})
    instances=[]
    def reader(*args,**kwargs):
        ledger=MeasuredLedger(*args,**kwargs)
        ledger._coverage_cache_limit=cap
        instances.append(ledger)
        return ledger
    work={'_coverage_path':path,'_portable_frontier':frontier_snapshot(scope,boxes)}
    closes=[]
    with patch('constraint_sat_session.CoverageLedger',side_effect=reader):
        for _ in range(12):
            stop=[False]
            base=closes[-1] if closes else 0
            def report(event):
                if event['stats']['session_cubes_closed']>base:
                    stop[0]=True
            result=solve(p,seconds=30,progress=report,stop=lambda:stop[0],learning_state=work,
                         options={'shared_coverage':True})
            closes.append(result['stats']['session_cubes_closed'])
            if result['status']=='infeasible':
                break
    try:
        assert result['status']=='infeasible' and len(instances)==1
        ledger=instances[0]
        return dict(closes=closes,native=result['stats']['session_native_calls'],
                    decisions=result['stats']['session_decisions'],checked=ledger.stats['coverage_records_checked'],
                    payload=ledger.checked_bytes)
    finally:
        work['_sat_session'].close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--dense-only',action='store_true')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        tiles=json.loads(args.tiles.read_bytes())
        if args.dense_only:
            args.output.write_text(json.dumps(dense_proofs(tiles,root),indent=2)+'\n',encoding='utf8')
            raise SystemExit
        out={'budgets':budget_matrix(tiles,root)}
        for label,cap in [('stream',0),('cached',4194304)]:
            run=root/label
            run.mkdir()
            out[label]=sat_slices(run,cap)
        assert all(out['stream'][k]==out['cached'][k] for k in ('closes','native','decisions'))
        args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
