"""Reproduce physical-core coverage and warm score-floor comparisons on an incumbent."""
import argparse
from contextlib import closing
import itertools
import json
import math
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import read_json, run_shell
from r29_move_cores import learn, load_cores
from r29_plateau import walk
from search_problem import SearchProblem


def skipped_sets(n, size, cores):
    """Inclusion-exclusion counts any support missing at least one core exactly once."""
    return sum((-1)**(k+1) * math.comb(n-len(set().union(*group)),size)
               for k in range(1,len(cores)+1) for group in itertools.combinations(cores,k))


def benchmark(problem, board, database, target, floor):
    model = BoardModel(problem)
    with closing(sqlite3.connect(database)) as db:
        cores = load_cores(db,model,board,target)
    if len(cores)>14:
        raise ValueError("Use at most 14 non-subsumed cores for this exact inclusion-exclusion benchmark.")
    with TemporaryDirectory() as tmp:
        cold = run_shell(problem,board,2,target,Path(tmp)/"cold.sqlite")
        # Use a snapshot: previously closed shell cursors must not mask the work comparison.
        snapshot = Path(tmp)/"cores.sqlite"
        with closing(sqlite3.connect(database)) as src, closing(sqlite3.connect(snapshot)) as dst:
            for name in ('physical_cores','physical_core_pins'):
                schema = src.execute("SELECT sql FROM sqlite_master WHERE name=?",(name,)).fetchone()
                if schema:
                    dst.execute(schema[0])
                    dst.executemany("INSERT INTO "+name+" VALUES (?,?,?)",src.execute("SELECT * FROM "+name))
            dst.commit()
        warm = run_shell(problem,board,2,target,snapshot)
        if (cold['hits'],cold['tested'],cold['complete']) != (warm['hits'],warm['tested'],True):
            raise RuntimeError("Core filter changed shell result.")
        args = dict(max_centers=1,tie_limit=1)
        db_path = Path(tmp)/"graph.sqlite"
        first = walk(problem,board,db_path,floor,target,**args)
        second = walk(problem,board,db_path,floor-1,target,**args)
        control = walk(problem,board,Path(tmp)/"coldwalk.sqlite",floor-1,target,**args)
        if (second['score'],second['open'],second['closed'])!=(control['score'],control['open'],control['closed']):
            raise RuntimeError("Floor cache changed the one-center search.")
    def summary(row):
        keys = ['score','components','open','closed','solved','reused','star_evaluations','cached_stars']
        return {key:row[key] for key in keys}
    # Physical selection counts use only movable cells when the board has pins.
    free = set(map(int,model.free))
    effective = [core & free for core in cores]
    return dict(target=target, floor=floor, core_sizes=list(map(len,cores)),
                core_skipped_two=warm.get('core_skips',0), unfiltered_two=cold['tested'],
                core_skipped_six_supports=skipped_sets(len(free),6,effective) if len(free)>=6 else 0,
                floor_first=summary(first), floor_second=summary(second), floor_second_cold=summary(control))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--incumbent',required=True)
    p.add_argument('--database',required=True)
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--floor',type=int,default=207)
    p.add_argument('--learn', action='store_true', help='Prepare two cores before the comparison.')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    problem = SearchProblem(read_json(args.tiles))
    board = read_json(args.incumbent)['codes']
    if args.learn:
        model = BoardModel(problem)
        for seed,seconds,queries in ((91,2,34),(18,3,42)):
            order = np.random.default_rng(seed).permutation(model.n).tolist()
            learn(model,board,args.database,args.target,order, seconds=seconds,max_queries=queries)
    result = benchmark(problem,board,args.database,args.target,args.floor)
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
