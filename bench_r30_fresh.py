"""Compare validated independent starts without importing any release-center certificate."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np

from constraint_symmetry import proper_rotations
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, domain_binding, read_json
from r29_exchange_clusters import StarClusters
from r29_exchange_route import existing_balls, owner_copies, physical_hash, transpositions
from r30_bounds import (RotationRegions, budget_upper, cell_mask, defect_cores,
                         defect_packings, issued_pair_anchors)
from r30_cycles import Predecessors
from r30_fresh import search
from search_problem import SearchProblem


def scope_key(parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def load_starts(problem, database, config, temporary):
    """Allow the old hash only when the entire original config reproduces it."""
    domain=domain_binding(BoardModel(problem))
    backend=config['backend']
    sampling=[backend] if backend=='cpu' else [backend,config['replicas'],config['dispatches']]
    current=scope_key([1,domain,sampling,config['rounds']])
    old=scope_key([1,domain,backend,config['replicas'],config['dispatches'],config['rounds']])
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as source:
        with closing(sqlite3.connect(temporary)) as dest:
            source.backup(dest)
            scopes={r[0] for r in dest.execute("SELECT DISTINCT scope FROM r30_starts")}
            key=current if current in scopes else old
            if key not in scopes:
                raise ValueError("No matching independent run in "+str(database))
            starts=[json.loads(row[0]) for row in dest.execute(
                "SELECT result FROM r30_starts WHERE scope=? AND seed BETWEEN 0 AND 3 ORDER BY seed",(key,))]
            if [r['seed'] for r in starts]!=[0,1,2,3]:
                raise ValueError("The comparison requires all four independent seeds.")
            if key!=current:
                dest.execute("UPDATE r30_starts SET scope=? WHERE scope=?",(current,key))
                dest.commit()
    repeat=search(problem,temporary,seed=0,starts=4,**config)
    assert repeat['scope']==current and (repeat['new_starts'],repeat['reused_starts'])==(0,4)
    return starts,dict(old_key=key,key=current,rekeyed_scratch_copy=key!=current,
                       new_starts=repeat['new_starts'],reused_starts=repeat['reused_starts'])


def nearest(center, release, rotations):
    inverse=np.argsort(center)
    distances=[]
    for row in release:
        for i,rot in enumerate(rotations):
            oriented=np.empty(len(center),int)
            oriented[list(rot.cells)]=row['pieces']
            permutation=inverse[oriented]
            moves=transpositions(permutation)
            mismatch=int(np.count_nonzero(oriented!=center))
            distances.append((moves,mismatch,row['sha256'],i))
    moves,displaced,owner,rot=min(distances)
    return dict(minimum_swaps=moves,displaced_for_minimum_swaps=displaced,
                minimum_displaced=min(d[1] for d in distances),release_center=owner,rotation_index=rot)


def measure(problem, board, supports, release, rotations):
    model=BoardModel(problem)
    b=ExchangeBound(model,board)
    scoped=Predecessors(b,208)
    anchors=issued_pair_anchors(b,208,4)
    owners=list(owner_copies(model,existing_balls(model,208),b.tiles,8))
    if scoped.certified or anchors or owners:
        raise ValueError("Expected an independent center outside all three released scopes.")
    clusters=StarClusters(b)
    regions=RotationRegions(b,2)
    cores=defect_cores(b,208)
    packs=defect_packings(b)
    # The strongest witness packing remains sound on every tested footprint.
    # It also avoids running maximum-coverage optimization 256 times per support.
    best=max(packs,key=len,default=())
    counts={}
    for size,sets in supports.items():
        stats=Counter()
        for cells in sets:
            mask=cell_mask(cells)
            core=any(not held&mask for held in cores)
            base=clusters.support_upper_twice(cells)
            corr=base-regions.discount(mask)
            limit=budget_upper(b,[best],4,cells)
            stats.update(dict(supports=1,base_star_closed=base<416,
                              regional_closed=corr<416,regional_tighter=corr<base,
                              required_movement_closed=core,defect_budget_closed=limit<208,
                              combined_closed=(core or corr<416 or limit<208),
                              early_defect_spares_star=(core or limit<208)))
        counts[size]=dict(stats)
    return dict(physical_center=physical_hash(b.tiles),nearest=nearest(b.tiles,release,rotations),
                release_predecessors=scoped.certified,release_anchors=len(anchors),
                release_ball_overlaps=len(owners),defect_cores=len(cores),
                disjoint_defect_witnesses=len(best),star_upper=b.total//2,
                whole_four_swap_budget_upper=budget_upper(b,[best],4),
                support_results=counts)


def run(problem, db_cpu, db_cuda, samples, seconds):
    if samples<1 or seconds<=0:
        raise ValueError("A positive sample count and root budget are required.")
    from constraint_models import solve
    root=Path(__file__).resolve().parent
    release=read_json(root/'docs/R29_EXCHANGE_CENTERS.json')['centers']
    rotations=proper_rotations(problem.board)
    rng=np.random.default_rng(1901)
    sets={k:[sorted(map(int,rng.choice(problem.n,k,replace=False))) for _ in range(samples)] for k in (6,7,8)}
    configs={}
    with TemporaryDirectory() as t:
        for backend,db,dispatches in [('cpu',db_cpu,64),('cuda',db_cuda,256)]:
            cfg=dict(backend=backend,replicas=512,dispatches=dispatches,rounds=64)
            starts,resume=load_starts(problem,db,cfg,Path(t)/(backend+'.sqlite'))
            outputs=[]
            for row in starts:
                board=row['codes']
                report=problem.validate(board)
                if (report['matched_edges'],report['line_components'],report['valid'])!=(row['score'],row['components'],row['gold']):
                    raise ArithmeticError("Independent board validation disagrees with the checkpoint.")
                outputs.append(dict(row,**measure(problem,board,sets,release,rotations)))
            configs[backend]=dict(configuration=cfg,resume=resume,starts=outputs)
    # A fresh, incumbent-free Gold query. Every physically legal placement can
    # be tried; the solver may apply its usual root symmetry reduction.
    control=solve(problem,engine='cp-sat',seconds=seconds,seed=17,target='gold')
    if control['codes']:
        if not problem.validate(control['codes'])['valid']:
            raise ArithmeticError("Invalid unrestricted Gold witness.")
    return dict(domain=domain_binding(BoardModel(problem)),support_seed=1901,samples_per_size=samples,
                support_sha=scope_key(sets),objective=208,proper_rotations=len(rotations),released_centers=len(release),
                configurations=configs,unrestricted_gold_control=dict(engine='cp-sat',seconds=seconds,seed=17,
                status=control['status'],complete=control['complete'],nodes=control['nodes'],validation=control['validation']))


def trajectory_extension(problem, directory):
    """Measure checkpoint-backed 64-to-320 repair extensions on real inputs."""
    directory.mkdir(parents=True,exist_ok=True)
    result={}
    def counters(run):
        return {k:v for k,v in run.items() if k not in ('best','scope')}
    for backend,dispatches in [('cpu',64),('cuda',256)]:
        db=directory/('continued-'+backend+'.sqlite')
        opts=dict(seed=0,starts=4,backend=backend,replicas=512,dispatches=dispatches)
        records=[]
        for budget in (64,320,320):
            run=search(problem,db,rounds=budget,**opts)
            with closing(sqlite3.connect(db)) as conn:
                rows=[json.loads(r[0]) for r in conn.execute(
                    "SELECT result FROM r30_starts WHERE scope=? ORDER BY seed",(run['scope'],))]
            for row in rows:
                report=problem.validate(row['codes'])
                if (report['matched_edges'],report['line_components'],report['valid'])!=(row['score'],row['components'],row['gold']):
                    raise ArithmeticError('Invalid extended independent board.')
            records.append(dict(rounds=budget,scope=run['scope'],work=counters(run),
                                boards=[dict(seed=r['seed'],score=r['score'],components=r['components'],
                                       gold=r['gold'],rounds=r['repair_rounds'],
                                       physical_hash=physical_hash(np.asarray(r['codes'])//3)) for r in rows]))
        result[backend]=records
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    p.add_argument('--cpu-db',type=Path,default=Path('audit/r30/independent-cpu.sqlite'))
    p.add_argument('--cuda-db',type=Path,default=Path('audit/r30/independent.sqlite'))
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extend',type=Path,help='Run a 64/320/320 trajectory test in this checkpoint directory.')
    a=p.parse_args()
    problem=SearchProblem(read_json(a.tiles))
    result=run(problem,a.cpu_db,a.cuda_db,a.samples,a.seconds)
    if a.extend:
        result['trajectory_extension']=trajectory_extension(problem,a.extend)
    atomic_json(a.output,result)
