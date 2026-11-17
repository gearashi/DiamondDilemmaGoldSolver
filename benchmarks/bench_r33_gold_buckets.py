"""Measure suffix ownership and projected group costs on equal original-puzzle ranges."""
import argparse
import json
from math import comb
from pathlib import Path
from time import perf_counter

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


class CountLookups(dict):
    """Count actual projection lookups while keeping table values unchanged."""

    def __init__(self, source):
        super().__init__(source)
        self.visits = 0

    def get(self, key, default=None):
        self.visits += 1
        return super().get(key, default)


def measure(problem, reference, groups, required, cap, errors, start, stop, seek, prepay):
    query = OutsideSyndromes(problem, reference, groups, required,
                             seek_ordered_tails=seek, use_bucket_price=prepay)
    query.tables = [(rhs, cols, CountLookups(lookup)) for rhs, cols, lookup in query.tables]
    # A warm table separates traversal work from the independently measured
    # anchor selection and suffix construction. A timer never closes a range.
    query.indexed(cap, errors, seconds=float('inf'), start=start, stop=start+1)
    for _, _, cost in query.tables:
        cost.visits = 0
    before = perf_counter()
    result = query.indexed(cap, errors, seconds=float('inf'), start=start, stop=stop)
    work = result['work']
    return dict(seek=seek, prepay=prepay, status=result['status'], next_rank=result['next_rank'],
                witness=result.get('outside_errors'), wall_seconds=round(perf_counter()-before,5),
                group_lookups=sum(cost.visits for _, _, cost in query.tables),
                skipped=work.get('unowned_tail_visits_skipped',0), unowned=work.get('unowned_tail_visits',0),
                tails=work.get('indexed_tails',0), joint=work.get('joint_queries',0),
                prefixes=work.get('prefixes',0), bucket_prices=work.get('bucket_prices_reused',0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='data/tiles.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    problem = SearchProblem(read_json(args.tiles))
    codes = read_json('docs/R29_INCUMBENT.json')['codes']
    reference, _, groups = reference_groups(problem, codes)
    required = inventory(problem).required
    n = len(OutsideSyndromes.layout(reference, groups, required)[2])
    cases = []
    for cap, errors, width in ((14,4,4096),(15,5,4096),(16,6,2048),(17,7,512)):
        size = comb(n-2,errors-2)
        for start in sorted({0,size//2,size-width}):
            stop = min(size,start+width)
            rows = [measure(problem,reference,groups,required,cap,errors,start,stop,seek,prepay)
                    for seek,prepay in ((False,False),(True,False),(True,True),(False,True))]
            outcome = ('status','next_rank','witness','tails','joint','prefixes')
            assert all(tuple(r[k] for k in outcome)==tuple(rows[0][k] for k in outcome) for r in rows)
            if rows[0]['status']=='INFEASIBLE':
                assert rows[0]['unowned']==rows[1]['skipped']
            cases.append(dict(cap=cap, errors=errors, start=start, stop=stop, rows=rows))
    record = dict(fingerprint=problem.fingerprint, reference_matches=len(reference),cases=cases)
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
