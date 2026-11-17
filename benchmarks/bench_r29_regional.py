"""Reproducible work comparisons for binary, paging and proof improvements."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from r27_structure_fixtures import planted
from r28_geo_regions import paged_trie
from r28_logic_region import regional_interpolant
from r28_logic_sat import checked_interpolant
from r28_logic_tail import binary_search, cut_bound
from r29_logic_binary import priced_cut
from r29_logic_proof import proof_interpolants
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_models import pair_problem


def benchmark_binary(samples):
    p = SearchProblem(*tetrahedron_tiles())
    rng = Random(29)
    stats = Counter(samples=samples)
    for _ in range(samples):
        rows = [rng.sample(r,2) for r in p.domains]
        scores = [p.validate(list(b))['matched_edges'] for b in product(*rows)
                  if len({v//3 for v in b}) == p.n]
        old = cut_bound(p,rows)
        new = priced_cut(p,rows)
        if not scores:
            if new['codes'] is not None:
                raise ArithmeticError("False physical witness.")
            stats['inventory_infeasible'] += 1
            continue
        best = max(scores)
        stats['feasible'] += 1
        stats['frustrated'] += bool(old['frustrated'])
        stats['baseline_exact_upper'] += old['upper'] == best
        stats['baseline_optimal_witness'] += old['score'] == best
        stats['tightened'] += old['upper'] > new['upper']
        stats['contracted_scopes'] += new['work']['variables'] < p.n
        stats['new_exact_upper'] += new['upper'] == best
        stats['new_optimal_witness'] += new['score'] == best
        stats['exact_frustrated'] += bool(old['frustrated']) and new['upper'] == best
        # Separate prices and residual-tie effects from inventory contraction.
        no_contract = priced_cut(p,rows,contract=False)
        cold = priced_cut(p,rows,contract=False,iterations=1,max_ties=1)
        stats['price_and_tie_additional_upper'] += cold['upper'] > no_contract['upper']
        stats['tie_recovers_optimal_witness'] += (cold['score'] != best and
             priced_cut(p,rows,contract=False,iterations=1)['score'] == best)
        for result in (new,no_contract,cold):
            if not best <= result['upper'] <= old['upper']:
                raise ArithmeticError("Certified bound contradicted exhaustive inventory.")
            if result['codes'] is not None and p.validate(result['codes'])['matched_edges'] != result['score']:
                raise ArithmeticError("Stored witness differs from independent score.")
    return dict(stats)


def page_run(problem, parts, ds, db, budget=100000, page=5):
    work, cursor = Counter(), None
    while True:
        result = paged_trie(problem,parts,ds,cache_path=db,cursor=cursor,max_steps=page,table_work=budget)
        cursor = result['cursor']
        work.update(result.get('work',{}))
        work['pages'] += 1
        if result['complete']:
            return dict(solutions=result['solutions'],checks=result['checks'],work=dict(work))


def benchmark_pages():
    parts = ((0,1,2),(3,4,5),(6,7,8,9))
    out = {}
    with TemporaryDirectory() as temp:
        for separate in (False,True):
            p,ds = planted(1,5,97,uniform_ports=True,split=separate)
            db = Path(temp)/str(separate)
            old = page_run(p,parts,ds,None)
            new = page_run(p,parts,ds,db,17)
            warm = page_run(p,parts,ds,db,17)
            narrowed = list(ds)
            c = next(c for c,m in enumerate(ds) if m.bit_count()>1)
            narrowed[c] ^= narrowed[c] & -narrowed[c]
            child = page_run(p,parts,narrowed,db)
            child_old = page_run(p,parts,narrowed,None)
            if len({v['solutions'] for v in (old,new,warm)}) != 1 or child['solutions']!=child_old['solutions']:
                raise ArithmeticError("Regional count changed on resume or related domain.")
            out['split' if separate else 'gold'] = dict(old=old,new=new,warm=warm,related=child)
    return out


def benchmark_proof():
    width = 6
    left, right = [],[]
    for i,bit in enumerate(product((0,1),repeat=width)):
        (left if sum(bit)%2 else right).append([v if x else -v for v,x in enumerate(bit,1)])
    # Count native SAT calls in the old boundary enumeration; both independent
    # semantic checks are included on the proof side as well.
    from unittest.mock import patch
    from pysat.solvers import Glucose4
    calls = Counter()
    real_solve = Glucose4.solve
    def counted(self,*a,**kw):
        calls["calls"] += 1
        return real_solve(self,*a,**kw)
    with patch.object(Glucose4,'solve',counted):
        base = checked_interpolant(left,right,range(1,width+1))
        old_calls = calls.pop('calls')
        proof = proof_interpolants([left[:len(left)//2],left[len(left)//2:],right],[(0,1),(0,)],
                                  max_work=2000000)
        from r29_logic_proof import reuse_boundary
        proof_calls = calls['calls']
        if not reuse_boundary(left,right+[[width+21]],proof['boundaries'][0]):
            raise ArithmeticError("Checked parity boundary failed on related region.")
        reuse_calls = calls['calls']-proof_calls
    if proof['status'] != base['status'] or not proof.get('boundaries'):
        raise ArithmeticError("No validated boundary on parity control.")
    return dict(cube_calls=old_calls,cube_clauses=len(base['clauses']),proof_calls=proof_calls,reuse_calls=reuse_calls,
                boundaries=len(proof['boundaries']),method=proof['method'],proof_builds=proof['proof_builds'],
                proof_work=proof['work'])


def benchmark_implications():
    stats = Counter()
    rng = Random(63)
    p = pair_problem(multiple_loops=True)
    for _ in range(500):
        rows = [rng.sample(r,2) for r in p.domains]
        old, _, baseline = binary_search(p,rows,'gold',reuse_graph=False)
        new, _, work = binary_search(p,rows,'gold')
        if old != new:
            raise ArithmeticError("Loop-cut SCC result disagrees.")
        stats['cases'] += 1
        stats['fewer'] += work < baseline
        stats['baseline_work'] += baseline
        stats['current_work'] += work
    return dict(stats)


def benchmark_actual(tiles, incumbent):
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
    p = SearchProblem(read(tiles))
    board = read(incumbent)['codes']
    rng = Random(29)
    out = {}
    for width in (8,32,160):
        cells = rng.sample(range(p.n),width)
        rows = [[v] for v in board]
        # Three pieces compete cyclically within small groups; keep the
        # incumbent legal and leave leftover cells binary in rotation.
        for i,c in enumerate(cells):
            source = cells[i//3*3+(i+1)%3] if i//3*3+2 < width else c
            rows[c].append(3*(board[source]//3)+(board[source]%3+1)%3)
        old,new = cut_bound(p,rows),priced_cut(p,rows)
        if not p.validate(board)['matched_edges'] <= new['upper'] <= old['upper']:
            raise ArithmeticError("Incumbent escaped certified binary upper.")
        scc,_,work = binary_search(p,rows,'gold')
        proof = regional_interpolant(p,rows,range(16),proof=True,subregions=[range(4)],proof_work=250000)
        out[str(width)] = dict(old=old['upper'],new=new['upper'],score=new['score'],work=new['work'],
                               scc=scc,scc_work=work,proof_status=proof['status'],
                               proof_builds=proof.get('proof_builds'),boundaries=len(proof.get('boundaries',())))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=600)
    parser.add_argument('--tiles',type=Path)
    parser.add_argument('--incumbent',type=Path)
    args = parser.parse_args()
    result = dict(binary=benchmark_binary(args.samples),implications=benchmark_implications(),
                  paging=benchmark_pages(),proof=benchmark_proof())
    if args.tiles and args.incumbent:
        result['actual'] = benchmark_actual(args.tiles,args.incumbent)
    print(json.dumps(result,indent=2))
