"""Learn threshold-specific physical identities that cannot all remain fixed."""
from contextlib import closing
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import domain_binding, read_json
from search_problem import SearchProblem


def scope_key(model, board):
    tiles = model.legal_board(board) // 3
    return domain_binding(model) + ":" + hashlib.sha256(tiles.astype("<i4").tobytes()).hexdigest()


def load_cores(connection, model, board, target):
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='physical_cores'").fetchone() is None:
        return []
    rows = connection.execute("SELECT fixed FROM physical_cores WHERE binding=? AND target<=?",
                              (scope_key(model, board),target))
    candidates = {frozenset(json.loads(row[0])) for row in rows}
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='physical_core_pins'").fetchone():
        tiles = model.legal_board(board) // 3
        for pins, in connection.execute("SELECT pins FROM physical_core_pins WHERE binding=? AND target<=?",
                                        (domain_binding(model),target)):
            pairs = json.loads(pins)
            if all(tiles[cell] == tile for cell,tile in pairs):
                candidates.add(frozenset(cell for cell,_ in pairs))
    result = []
    for candidate in sorted(candidates,key=lambda c:(len(c),sorted(c))):
        if not any(other <= candidate for other in result):
            result.append(candidate)
    return result


def permitted_sets(cells, cores):
    """Vectorized hitting-set screen before donors or rotations are considered."""
    keep = np.ones(len(cells), dtype=bool)
    for core in cores:
        if not keep.any():
            break
        keep &= np.isin(cells, list(core)).any(axis=1)
    return keep


def decide(model, board, fixed, target, seconds, *, table=False):
    """With physical identities fixed on these cells, can any arrangement reach target?"""
    from ortools.sat.python import cp_model
    board = model.legal_board(board)
    fixed = set(fixed)
    if not 0 <= target <= len(model.problem.edges) or seconds <= 0 or not fixed.issubset(range(model.n)):
        raise ValueError("Invalid physical core query.")
    locked = set(board[list(fixed)] // 3) if fixed else set()
    cp = cp_model.CpModel()
    codes, physical, values = [], [], []
    for cell in range(model.n):
        allowed = [int(c) for c in np.flatnonzero(model.legal[cell])
                   if (c//3 == board[cell]//3 if cell in fixed else c//3 not in locked)]
        values.append(allowed)
        v = cp.new_int_var_from_domain(cp_model.Domain.from_values(allowed), "")
        codes.append(v)
        if cell not in fixed:
            piece = cp.new_int_var(0,model.n-1,"")
            cp.add_division_equality(piece,v,3)
            physical.append(piece)
        cp.add_hint(v, int(board[cell]))
    if physical:
        cp.add_all_different(physical)
    terms = []
    for a,sa,b,sb in model.problem.edges:
        left = [int(model.masks[x,sa]) for x in values[a]]
        right = [int(model.reverse[model.masks[y,sb]]) for y in values[b]]
        common = set(left).intersection(right)
        if not common:
            continue
        if len(set(left+right)) == 1:
            terms.append(1)
            continue
        matched = cp.new_bool_var("")
        if table:
            partners = {}
            for y,mask in zip(values[b],right):
                partners.setdefault(mask,[]).append(y)
            cp.add_allowed_assignments([codes[a],codes[b]],
                [(x,y) for x,mask in zip(values[a],left) for y in partners.get(mask,())]).only_enforce_if(matched)
        else:
            # Channel each physical orientation to its edge-mask class once per side.
            ids = {mask:i for i,mask in enumerate(sorted(set(left+right)))}
            ends = []
            for cell, masks in ((a,left),(b,right)):
                if len(set(masks)) == 1:
                    ends.append(ids[masks[0]])
                else:
                    val = cp.new_int_var(0,len(ids)-1,"")
                    cp.add_allowed_assignments([codes[cell],val],[(x,ids[m]) for x,m in zip(values[cell],masks)])
                    ends.append(val)
            cp.add(ends[0]==ends[1]).only_enforce_if(matched)
        terms.append(matched)
    cp.add(sum(terms)>=target)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    status = solver.solve(cp)
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        witness = [int(solver.value(v)) for v in codes]
        model.legal_board(witness)
        if (any(witness[i]//3!=board[i]//3 for i in fixed)
                or model.problem.validate(witness)['matched_edges']<target):
            raise RuntimeError("Physical core decision disagrees with independent scoring.")
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError(solver.solution_info())
    return solver.status_name(status)


def learn(model, board, database, target, order, *, seconds=2, initial_free=(), max_queries=None):
    """Deletion-minimize in the supplied order. Unknowns keep an assumption in the core."""
    board = model.legal_board(board)
    fixed = set(range(model.n)) - set(initial_free)
    initial = decide(model,board,fixed,target,seconds)
    work = dict(queries=1, infeasible=0, unknown=0, feasible=0)
    if initial != 'INFEASIBLE':
        return dict(work,status=initial,stored=False)
    for cell in order:
        if cell not in fixed:
            continue
        if max_queries is not None and work['queries']>=max_queries:
            break
        status = decide(model,board,fixed-{cell},target,seconds)
        work['queries'] += 1
        if status=='INFEASIBLE':
            fixed.remove(cell)
            work['infeasible'] += 1
        elif status=='UNKNOWN':
            work['unknown'] += 1
        else:
            work['feasible'] += 1
    # Rebuild with direct compatible-pair tables for an independent confirmation.
    checked = decide(model,board,fixed,target,seconds*3,table=True)
    work["confirmation_queries"] = 1
    if checked!='INFEASIBLE':
        return dict(work,status=checked,stored=False)
    encoded = json.dumps(sorted(fixed))
    binding = scope_key(model,board)
    with closing(sqlite3.connect(database,timeout=60)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS physical_cores(binding TEXT, target INTEGER,
                   fixed TEXT, PRIMARY KEY(binding,target,fixed))""")
        db.execute("""CREATE TABLE IF NOT EXISTS physical_core_pins(binding TEXT,target INTEGER,
                   pins TEXT,PRIMARY KEY(binding,target,pins))""")
        pins = json.dumps([(cell,int(board[cell]//3)) for cell in sorted(fixed)])
        db.execute("INSERT OR IGNORE INTO physical_core_pins VALUES (?,?,?)",(domain_binding(model),target,pins))
        db.execute("INSERT OR IGNORE INTO physical_cores VALUES (?,?,?)",(binding,target,encoded))
    return dict(work,status=checked,stored=True,held=len(fixed),free=sorted(set(range(model.n))-fixed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles", default="data/tiles.json")
    p.add_argument("--incumbent",help="Starting board for learning new cores.")
    p.add_argument("--database",required=True)
    p.add_argument("--target",type=int,default=208)
    p.add_argument("--seconds-per-query",type=float,default=3)
    p.add_argument("--queries",type=int,default=24)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--verify-release",action="store_true",
                   help="Recheck and install the published movement cores for the release center.")
    args = p.parse_args()
    model = BoardModel(SearchProblem(read_json(args.tiles)))
    if args.verify_release:
        root = Path(__file__).resolve().parent / 'docs'
        record = read_json(root/'R29_PHYSICAL_CORES.json')
        centers = read_json(root/'R29_EXCHANGE_CENTERS.json')
        if domain_binding(model)!=centers['domain'] or args.target!=record['target']:
            raise ValueError("The published movement cores require the release domains and score target.")
        tiles = next(row['pieces'] for row in centers['centers'] if row['sha256']==record['center'])
        board = [3*t+int(np.flatnonzero(model.legal[c,3*t:3*t+3])[0]) for c,t in enumerate(tiles)]
        result = [learn(model,board,args.database,args.target,[],seconds=args.seconds_per_query,
                        initial_free=free) for free in record['free_sets']]
        print(json.dumps(result))
        return
    if not args.incumbent:
        p.error("--incumbent is required for learning new cores.")
    board = read_json(args.incumbent)['codes']
    order = np.random.default_rng(args.seed).permutation(model.n).tolist()
    print(json.dumps(learn(model,board,args.database,args.target,order,seconds=args.seconds_per_query,
                           max_queries=args.queries)))


if __name__=="__main__":
    main()
