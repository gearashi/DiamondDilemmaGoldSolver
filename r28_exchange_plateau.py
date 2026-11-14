"""Explore distinct physical one-swap centers while preserving a verified score floor."""
import argparse
import itertools
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from r28_exchange import ExchangeBound, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from r27_incumbent import BoardModel, rotation_alternatives
from search_problem import SearchProblem


def walk(problem, codes, database, floor, goal, *, max_centers=20, seconds=0, tie_limit=8):
    if not 0 <= floor < goal <= len(problem.edges) or tie_limit<1:
        raise ValueError("Invalid plateau floor, goal or tie limit.")
    model = BoardModel(problem)
    start = model.legal_board(codes)
    oracle = RotationOracle(model)
    free = [i for i in range(model.n) if i not in problem.fixed]
    pairs = np.array(list(itertools.combinations(free,2)),dtype=int).reshape(-1,2)
    started = time.monotonic()
    fingerprint = domain_binding(model) + f"/{floor}/{goal}/{tie_limit}"
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS centers (scope TEXT, perm TEXT, closed INT, score INT,"
        " comps INT, codes TEXT, PRIMARY KEY(scope,perm))")
    connection.execute("CREATE TABLE IF NOT EXISTS optimum (scope TEXT, perm TEXT, score INT, comps INT, codes TEXT,"
                       "PRIMARY KEY(scope,perm))")
    def identity(board):
        return ",".join(map(str,np.asarray(board)//3))
    def insert(board,score,comps):
        connection.execute("INSERT OR IGNORE INTO centers VALUES (?,?,?,?,?,?)",
                           (fingerprint,identity(board),0,score,comps,json.dumps(list(map(int,board)))))
    original = problem.validate(start.tolist())
    if original["matched_edges"]<floor:
        raise ValueError("Starting score is below plateau floor.")
    insert(start,original["matched_edges"],original["line_components"])
    connection.commit()
    work = dict(centers=0,star=0,solved=0,reused=0,exact_fallback=0)
    try:
        while work["centers"]<max_centers and (not seconds or time.monotonic()-started<seconds):
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT perm,codes,score FROM centers WHERE scope=? AND closed=0 "
                "ORDER BY score DESC, comps ASC,rowid LIMIT 1",(fingerprint,)).fetchone()
            if not row or row[2]>=goal:
                connection.commit()
                break
            key, board = row[0],json.loads(row[1])
            bound = ExchangeBound(model,board)
            for offset in range(0,len(pairs),512):
                cells = pairs[offset:offset+512]
                bounds = bound.bounds(cells,(1,0))
                work["star"] += int((bounds<floor).sum())
                for moved in cells[bounds>=floor]:
                    seed = bound.witness_seed(moved,(1,0))
                    if seed is None:
                        continue
                    perm = identity(seed)
                    saved = connection.execute("SELECT score,comps,codes FROM optimum WHERE scope=? AND perm=?",
                                               (fingerprint,perm)).fetchone()
                    if saved is None:
                        answer = oracle.optimum(seed,ties=True,limit=tie_limit)
                        if answer is None:
                            work["exact_fallback"] += 1
                            ans = rotation_alternatives(model,seed,k=1,seconds=120)
                            if not ans or not ans[0][1]["exact"]:
                                raise RuntimeError("Rotation optimum incomplete; resume the center.")
                            optimum,choices = ans[0][1]["matched_edges"],[ans[0][0].tolist()]
                        else:
                            optimum,choices = answer
                        if optimum<floor:
                            comps,candidate = None,None
                        else:
                            comps,candidate = None,None
                            for child in choices:
                                model.legal_board(child)
                                report = problem.validate(child)
                                if report["matched_edges"]!=optimum:
                                    raise RuntimeError("Independent plateau scoring failed.")
                                if comps is None or report["line_components"]<comps:
                                    comps,candidate = report["line_components"],child
                        saved = optimum,comps,json.dumps(candidate)
                        connection.execute("INSERT INTO optimum VALUES (?,?,?,?,?)",(fingerprint,perm,*saved))
                        work["solved"] += 1
                    else:
                        work["reused"] += 1
                    score,comps,encoded = saved
                    if score>=floor:
                        insert(json.loads(encoded),score,comps)
            connection.execute("UPDATE centers SET closed=1 WHERE scope=? AND perm=?",(fingerprint,key))
            connection.commit()
            work["centers"] += 1
        score,comps,encoded = connection.execute("SELECT score,comps,codes FROM centers WHERE scope=? "
            "ORDER BY score DESC,comps ASC LIMIT 1",(fingerprint,)).fetchone()
        counts = dict(connection.execute("SELECT closed,count(*) FROM centers WHERE scope=? GROUP BY closed",
                                         (fingerprint,)))
        gold = problem.validate(json.loads(encoded))["valid"]
        return dict(work,objective="matched_edges",target=goal,gold=gold,closed=counts.get(1,0),open=counts.get(0,0),score=score,components=comps,
                    codes=json.loads(encoded),rotation_work=oracle.work)
    finally:
        connection.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles",default="data/tiles.json")
    p.add_argument("--incumbent",required=True)
    p.add_argument("--database",required=True)
    p.add_argument("--floor",type=int,default=207)
    p.add_argument("--target",type=int,default=208,help="Matched-edge threshold. At 240, inspect the Gold flag to verify one loop.")
    p.add_argument("--centers",type=int,default=20)
    p.add_argument("--seconds",type=float,default=0)
    p.add_argument("--output",type=Path)
    args = p.parse_args()
    result = walk(SearchProblem(read_json(args.tiles)),read_json(args.incumbent)["codes"],
                  args.database,args.floor,args.target,max_centers=args.centers,seconds=args.seconds)
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
