"""Search from independently generated full boards and retain verified improvements across restarts."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel, assignment, polish_step
from r28_exchange import domain_binding, read_json
from search_problem import SearchProblem


def fresh_start(model, rng):
    """Choose a fresh legal physical assignment, respecting any fixed cells."""
    n = model.n
    weights = rng.integers(0, 1000000, size=(n,n,3), dtype=np.int32)
    allowed = model.legal.reshape(n,n,3)
    weights[~allowed] = -1
    pieces = assignment(weights.max(axis=2), allowed.any(axis=2))
    return 3*pieces + weights[np.arange(n),pieces].argmax(axis=1)


def _consider(best, codes, validation):
    rank=(int(validation['valid']),validation['matched_edges'],-validation['line_components'])
    if best is None or rank>(int(best['gold']),best['score'],-best['components']):
        return dict(codes=list(map(int,codes)),score=rank[1],components=-rank[2],gold=bool(rank[0]))
    return best


def _sample_start(model, seed, backend, replicas, dispatches):
    """Initial sampler and its exact post-initialization CPU RNG state."""
    rng = np.random.default_rng(seed)
    problem = model.problem
    rows = [fresh_start(model,rng).tolist()]
    proposals = 0
    if backend != 'cpu':
        from gpu_sampling_resume_round25 import SamplerSession
        session = SamplerSession(problem, seed=seed, backend=backend, replicas=replicas, target='gold',
                                 experiments={'replica_exchange':True,'move_tabu':True})
        try:
            while session.launches<dispatches and not session.done:
                session.advance(60,max_dispatches=dispatches-session.launches)
            session.finish_phase()
            report = session.report()
            if session.solution is not None:
                rows.append(session.solution)
            rows.extend(r['codes'] for r in report['top_samples'] if not r['domain_violations'])
            proposals = report['move_proposals']
        finally:
            session.close()
    best = None
    for codes in rows:
        model.legal_board(codes)
        best=_consider(best,codes,problem.validate(codes))
    return dict(seed=seed,initial=best['score'],gpu_proposals=proposals,repair_rounds=0,
                current=best['codes'],rng=rng.bit_generator.state,best=best)


def _advance(model,state,limit):
    board=model.legal_board(state['current'])
    rng=np.random.default_rng()
    rng.bit_generator.state=state['rng']
    while state['repair_rounds']<limit and not state['best']['gold']:
        board,validation=polish_step(model,board,rng)
        state['best']=_consider(state['best'],board,validation)
        state['repair_rounds']+=1
    state.update(current=board.tolist(),rng=rng.bit_generator.state)


def _result(state):
    return dict(seed=state['seed'],initial=state['initial'],gpu_proposals=state['gpu_proposals'],
                repair_rounds=state['repair_rounds'],**state['best'])


def one_start(model, seed, backend, replicas, dispatches, rounds):
    """Run a standalone seed through its requested repair ceiling."""
    state=_sample_start(model,seed,backend,replicas,dispatches)
    _advance(model,state,rounds)
    return _result(state)


def search(problem, database, *, seed=0, starts=1, backend='cpu', replicas=512,
           dispatches=64, rounds=32, output=None, progress=None, batch=16):
    """Commit sampler and repair batches separately. Shorter ceilings can reuse earlier checkpoints."""
    if min(seed,starts,rounds,dispatches)<0 or batch<1 or not 1<=replicas<=131072 or backend not in ('cpu','cuda','webgpu'):
        raise ValueError("Invalid independent search configuration.")
    model = BoardModel(problem)
    sampling = [backend,replicas,dispatches] if backend!='cpu' else [backend]
    identity=[1,domain_binding(model),sampling]
    scope = hashlib.sha256(json.dumps([*identity,rounds]).encode()).hexdigest()
    path_scope = hashlib.sha256(json.dumps([*identity,'repair-trajectory-v1','gold']).encode()).hexdigest()
    def publish(row):
        if output:
            atomic_json(output,dict(codes=row['codes'],matched_edges=row['score'],
                                    line_components=row['components'],valid=row['gold'],seed=row['seed']))
    done=reused=repair_work=samples=proposals=reused_rounds=reused_samples=0
    with closing(sqlite3.connect(database,timeout=120)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS r30_starts(scope TEXT,seed INTEGER,score INTEGER,components INTEGER,gold INTEGER,result TEXT NOT NULL,PRIMARY KEY(scope,seed))")
        db.execute('CREATE INDEX IF NOT EXISTS r30_starts_best ON r30_starts(scope,gold DESC,score DESC,components,seed)')
        db.execute("""CREATE TABLE IF NOT EXISTS r30_trajectory(
                   scope TEXT,seed INTEGER,rounds INTEGER,gold INTEGER,state TEXT NOT NULL,
                   PRIMARY KEY(scope,seed,rounds))""")
        db.execute("CREATE INDEX IF NOT EXISTS r30_trajectory_gold ON r30_trajectory(scope) WHERE gold=1")
        def current_best():
            row=db.execute('SELECT state FROM r30_trajectory WHERE scope=? AND gold=1 LIMIT 1',(path_scope,)).fetchone()
            if row:
                return _result(json.loads(row[0]))
            row = db.execute('SELECT result FROM r30_starts WHERE scope=? ORDER BY gold DESC,score DESC,components,seed LIMIT 1',(scope,)).fetchone()
            return json.loads(row[0]) if row else None
        best=current_best()
        while (not starts or done+reused<starts) and not (best and best['gold']):
            index=seed+done+reused
            first_query=True
            while True:
                db.execute('BEGIN IMMEDIATE')
                best=current_best()
                if best and best['gold']:
                    db.commit()
                    break
                old=db.execute('SELECT 1 FROM r30_starts WHERE scope=? AND seed=?',(scope,index)).fetchone()
                if old:
                    next_seed=db.execute('SELECT min(s.seed+1) FROM r30_starts s WHERE s.scope=? AND s.seed>=? '
                                         'AND NOT EXISTS(SELECT 1 FROM r30_starts t WHERE t.scope=s.scope AND t.seed=s.seed+1)',
                                         (scope,index)).fetchone()[0]
                    reused+=min(next_seed-index,starts-done-reused) if starts else next_seed-index
                    db.commit()
                    break
                row=db.execute('SELECT state FROM r30_trajectory WHERE scope=? AND seed=? AND rounds<=? '
                               'ORDER BY rounds DESC LIMIT 1',(path_scope,index,rounds)).fetchone()
                if row is None:
                    state=_sample_start(model,index,backend,replicas,dispatches)
                    samples+=1
                    proposals+=state['gpu_proposals']
                    changed=True
                else:
                    state=json.loads(row[0])
                    before=state['repair_rounds']
                    if first_query:
                        reused_rounds+=before
                        reused_samples+=1
                    if before<rounds:
                        limit=min(rounds,(before//batch+1)*batch)
                        _advance(model,state,limit)
                    repair_work+=state['repair_rounds']-before
                    changed=state['repair_rounds']!=before
                first_query=False
                if changed:
                    db.execute('INSERT INTO r30_trajectory VALUES(?,?,?,?,?)',
                               (path_scope,index,state['repair_rounds'],int(state['best']['gold']),json.dumps(state)))
                finished=state['best']['gold'] or state['repair_rounds']==rounds
                if finished:
                    item=_result(state)
                    db.execute('INSERT INTO r30_starts VALUES(?,?,?,?,?,?)',
                               (scope,index,item['score'],item['components'],item['gold'],json.dumps(item)))
                    done+=1
                db.commit()
                if finished:
                    if progress:
                        progress({k:v for k,v in item.items() if k!='codes'})
                    best=current_best()
                    if output:
                        db.execute('BEGIN IMMEDIATE')
                        publish(current_best())
                        db.commit()
                    break
        best=current_best()
        if best and output:
            db.execute('BEGIN IMMEDIATE')
            best=current_best()
            publish(best)
            db.commit()
    return dict(new_starts=done,reused_starts=reused,best=best,scope=scope,new_sampler_starts=samples,
                new_gpu_proposals=proposals,new_repair_rounds=repair_work,reused_sampler_starts=reused_samples,
                reused_repair_rounds=reused_rounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--backend',choices=('cpu','cuda','webgpu'),default='cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--starts',type=int,default=1,help='Number of consecutive seeds, zero for unlimited.')
    parser.add_argument('--rounds',type=int,default=32)
    parser.add_argument('--dispatches',type=int,default=64)
    parser.add_argument('--replicas',type=int,default=512)
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True,exist_ok=True)
    result = search(SearchProblem(read_json(args.tiles)),args.database,seed=args.seed,starts=args.starts,
                    backend=args.backend,replicas=args.replicas,dispatches=args.dispatches,rounds=args.rounds,
                    output=args.output,progress=lambda row: print(json.dumps(row),flush=True))
    print(json.dumps(result))


if __name__=='__main__':
    main()
