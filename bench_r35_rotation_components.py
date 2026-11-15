"""Compare exact whole-component reuse against cached numeric factors and structural plans."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_rotations import RotationOracle
from r30_fresh import fresh_start
from search_problem import SearchProblem


def cases(model,incumbent,seed):
    rng=np.random.default_rng(seed)
    adjacent=[]
    for count in (1,2,3,4)*3:
        row=incumbent.copy()
        for _ in range(count):
            a,b=rng.choice(model.free,2,replace=False)
            row[[a,b]]=row[[b,a]]
        adjacent.append(row)
    fresh=[fresh_start(model,rng) for _ in range(6)]
    return dict(adjacent=[(v,59049) for v in adjacent],fresh=[(v,59049) for v in fresh],
                repeated=[(3*(incumbent//3)+(incumbent+s)%3,59049) for s in range(3)],
                caps=[(v,c) for v in (incumbent,fresh[0]) for c in (1,7,81,59049)])


def run(model,queries,mode):
    oracle=RotationOracle(model,components=mode!='factor')
    if oracle.component_cache:
        oracle.component_cache.max_cells={'single':0,'size4':4,'size8':8,'size16':16,'size32':32}.get(mode,160)
        oracle.component_cache.normalize=mode!='absolute'
        oracle.component_cache.frames=int(mode[6:]) if mode.startswith('frames') else (3 if mode.startswith('vertex') else 0)
        oracle.component_cache.vertices=int(mode[6:]) if mode.startswith(('vertex','lazies')) else 0
        oracle.component_cache.defer=mode.startswith('lazies')
    answers=[]
    firsts=[]
    timed=perf_counter()
    for board,cap in queries:
        result=oracle.optimum(board,max_entries=cap,ties=True,limit=4)
        if result is None:
            answers.append(None)
            firsts.append(None)
            continue
        score,items=result
        boards=list(items)
        for codes in boards:
            arr=model.legal_board(codes)
            if not np.array_equal(board//3,arr//3) or model.matches(arr).sum()!=score:raise AssertionError('Independent physical witness failed.')
        def digest(value):return hashlib.sha256(json.dumps(value).encode()).hexdigest()
        answers.append((score,len(boards),digest(boards)))
        firsts.append(digest(boards[0]))
    cache=oracle.component_cache
    keep=('numeric_keys','singleton_responses','component_keys','component_hits','component_factors_reused',
          'component_reindex','component_frame_hits','component_vertex_hits','component_tuples_encoded','component_tuples_decoded','messages','hits','avoided','plan_builds','plan_hits','threshold_rejected')
    return dict(seconds=perf_counter()-timed,work={k:oracle.work[k] for k in keep},answers=answers,firsts=firsts,
                stored_components=len(cache.rows) if cache else 0,component_bytes=cache.bytes if cache else 0)


def plateau(model,source,folder):
    from contextlib import closing
    from functools import partial
    import sqlite3
    from unittest.mock import patch
    from r29_plateau import walk
    folder.mkdir(parents=True,exist_ok=True)
    result={}
    for enabled in (False,True):
        name='components' if enabled else 'factors'
        db=folder/(name+'.sqlite')
        if db.exists():raise ValueError('Use a fresh directory to compare real plateau decisions.')
        with patch('r29_plateau.RotationOracle',partial(RotationOracle,components=enabled)):
            answer=walk(model.problem,source,db,207,208,max_swaps=1,tie_limit=1,widest=True,
                        incremental=True,max_centers=100)
        physical=[]
        with closing(sqlite3.connect(db)) as conn:
            for k,row in conn.execute('SELECT key,result FROM optima ORDER BY key'):
                item=json.loads(row)
                report=model.problem.validate(item['codes'])
                if report['matched_edges']!=item['score'] or report['line_components']!=item['components']:
                    raise ArithmeticError('Physical saved rotation failed independent validation.')
                physical.append((k,item['score'],item['components'],item['codes']))
        result[name]=dict(candidates=answer['swap_candidates'],optima=len(physical),score=answer['score'],
                          regions=answer['closed'],work=answer['rotation_work'],
                          witness_sha256=hashlib.sha256(json.dumps(physical).encode()).hexdigest())
    for key in ('candidates','optima','score','regions','witness_sha256'):
        assert result['components'][key]==result['factors'][key]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plateau-only',type=Path,help='Compare both modes on a complete original one-swap floor-207 neighborhood.')
    p.add_argument('--seeds',type=int,nargs='+',default=[851,7407])
    p.add_argument('--modes',nargs='+',default=['factor','single','size4','size8','size16','size32','absolute','whole'])
    args=p.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    board=model.legal_board(read_json(args.board)['codes'])
    if args.plateau_only:
        witness=plateau(model,board,args.plateau_only)
        atomic_json(args.output,witness)
        print(witness)
        return
    out={}
    for seed in args.seeds:
        for name,queries in cases(model,board,seed).items():
            group=out.setdefault(str(seed)+'-'+name,{})
            expected=None
            for mode in args.modes:
                rec=run(model,queries,mode)
                if expected is None:expected=rec['answers']
                elif expected!=rec['answers']:raise ArithmeticError((name,seed,mode,'Score, tie order or cap result differs.'))
                group[mode]=rec
                print(seed,name,mode,rec['work']['numeric_keys'],rec['work']['component_hits'],rec['work']['singleton_responses'],flush=True)
                atomic_json(args.output,dict(input=model.problem.fingerprint,groups=out))


if __name__=='__main__':main()
