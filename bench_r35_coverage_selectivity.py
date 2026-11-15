"""Screen selective lookup of genuine GPU/DFS proof products before changing storage."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from bench_r35_parked_cache import original_residual
from constraint_work import CoverageLedger, WorkScope, contains, intersect, subtract_many
from gpu_domain_runner import inventory_coverage, inventory_screen, split_cubes
from search_problem import SearchProblem


def selectors(root,row,count):
    axes = [i for i in range(len(root)) if row[i] != root[i]]
    return sorted(axes,key=lambda i:(row[i].bit_count()/root[i].bit_count(),i))[:count]


def case(problem,covers,queries):
    scope=WorkScope.for_problem(problem)
    with TemporaryDirectory() as path:
        file=Path(path)/'coverage.sqlite'
        with CoverageLedger(file,scope) as db:
            for box,evidence in covers:
                db.record_completed(box,status='infeasible',complete=True,evidence=evidence)
        with CoverageLedger(file,scope) as db:
            records=db.exclusions()
            # Measure actual serialized SQLite column bytes delivered to Python.
            sizes={tuple(int(x,16) for x in json.loads(body)):size for body,size in db.db.execute(
                "SELECT box,length(identity)+length(scope)+length(box)+length(evidence)+length(checksum) FROM closed")}
            results={}
            for label,query_list in queries.items():
                cuts_by_width={str(n):Counter() for n in (0,1,2,4)}
                for pending in query_list:
                    exact,_=subtract_many(pending,records)
                    hull=list(pending[0])
                    for row in pending[1:]:
                        for axis,mask in enumerate(row):
                            hull[axis] |= mask
                    if not contains(scope.root,hull):
                        raise ValueError('query escapes benchmark scope')
                    for n in (0,1,2,4):
                        selected=[box for box in records if all(box[a]&hull[a] for a in selectors(scope.root,box,n))]
                        result,done=subtract_many(pending,selected)
                        assert done and result==exact
                        counter=cuts_by_width[str(n)]
                        counter.update(queries=1,fetched=len(selected),payload_bytes=sum(sizes[box] for box in selected),
                                       exact_overlaps=sum(any(intersect(box,q) is not None for q in pending) for box in selected))
                results[label]=cuts_by_width
    return dict(records=len(records),scope=problem.fingerprint,queries=results)


def hall_case(problem,evaluate):
    scope=WorkScope.for_problem(problem)
    split=split_cubes([scope.root],128)
    live,closed=inventory_screen(split,problem.n)
    proofs=[(box,{'kind':'physical_inventory_hall','cells':list(cells)}) for box,cells in inventory_coverage(closed,scope.root)]
    queries={'wide':[[scope.root]],'whole_frontier':[split],'one_live':[[r] for r in live],
             'one_closed':[[b] for b,_ in closed]}
    return evaluate(problem,proofs,queries)


def benchmark(tiles,evaluate=case):
    from test_gpu_domain_round25 import precise_reason_fixture, word_boundary_fixture
    original=SearchProblem(tiles)
    restricted,witness=original_residual(tiles)
    boxes=split_cubes([WorkScope.for_problem(restricted).root],128)
    # Every partial assignment here contains the same independently immovable
    # mismatched seam. Publish half the disjoint branches, preserving a remainder.
    proofs=[(box,{'kind':'independent_fixed_mismatch','seam':witness}) for box in boxes[::2]]
    return {'original_hall':hall_case(original,evaluate),'six_piece_hall':hall_case(precise_reason_fixture(),evaluate),
            'word_boundary_hall':hall_case(word_boundary_fixture(),evaluate),'fixed_scope':evaluate(restricted,proofs,{
            'wide':[[WorkScope.for_problem(restricted).root]],'whole_frontier':[boxes],
            'one_box':[[row] for row in boxes],'eight_box':[boxes[i:i+8] for i in range(0,len(boxes),8)]})}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(benchmark(json.loads(args.tiles.read_bytes())),indent=2)+'\n',encoding='utf8')
