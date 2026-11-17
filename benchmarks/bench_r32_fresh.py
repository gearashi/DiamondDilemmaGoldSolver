"""Paired Gold phase and contour experiments. Reads the released original tile inventory."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r31_fresh import Assemblies, Engine, physical_bonds
from r32_fresh import fresh
from r32_fresh_geometry import projected, skeletons
from r32_fresh_phases import fit, lift_cube, phase_cube, SignedRepair
from r32_fresh_skeleton import parity_seed
from r32_gold_bonds import inventory
from search_problem import SearchProblem


def compare(problem, section, database, seeds, rounds):
    model=BoardModel(problem)
    required=inventory(problem).required
    forbidden=inventory(problem).forbidden
    results=[]
    def add(name, seed, call):
        start=monotonic()
        row=call()
        summary={k:v for k,v in row.items() if k not in ('codes','phases','patterns')}
        if 'codes' in row:
            board=model.legal_board(row['codes'])
            validation=problem.validate(board.tolist())
            summary['raw']=validation['matched_edges']
            summary['components']=validation['line_components']
            summary['gold']=validation['valid']
            bonds=physical_bonds(model,board)
            summary['required']=len(required&bonds)
            summary['forbidden']=len(forbidden&bonds)
            f=fit(problem,required,{0:0},bonds-forbidden,effort=.3,hint=row.get('phases'))
            summary['final_refit']={k:f[k] for k in ('status','kept','ceiling','calls','branches') if k in f}
        if 'patterns' in row:
            summary['skeletons_found']=len(row['patterns']) if 'codes' not in row else 1
        entry=dict(name=name,seed=seed,seconds=round(monotonic()-start,3),result=summary)
        results.append(entry)
        print(name,seed,summary.get('raw'),summary.get('signed'),summary.get('final_refit'),entry['seconds'],flush=True)
    if section in ('cpu','cuda'):
        base=Engine(problem,bonds=True,cycle_budget=3)
        legacy=Engine(problem,bonds=False,cycle_budget=3)
        legacy.assemblies=Assemblies(legacy.model,gold_bonds=False)
        for seed in seeds:
            config=dict(backend=section,rounds=rounds,replicas=512,dispatches=256,
                        sampler_cache=database if section=='cuda' else None)
            add('released-seven',seed,lambda:legacy.run(seed,ranked=True,**config))
            add('eight-only',seed,lambda:base.run(seed,ranked=True,**config))
            add('phase',seed,lambda:fresh(problem,seed,**config))
            if section=='cpu':
                add('phase-refit',seed,lambda:fresh(problem,seed,refit=8,**config))
                add('phase-assemblies',seed,lambda:fresh(problem,seed,merges=4,**config))
                add('phase-contour',seed,lambda:fresh(problem,seed,contour=True,**config))
    elif section=='geometry':
        for seed in seeds:
            add('sample-contour',seed,lambda:parity_seed(problem,seed))
        for merge in (False,True):
            add('frontier-'+str(merge),0,lambda:skeletons(problem,limit=8,max_nodes=100000,merge=merge))
        for width in (0,1,2):
            for idx in range(1<<width):
                row=fit(problem,required,phase_cube(problem,width,idx),effort=.15)
                results.append(dict(name='phase-cube',width=width,index=idx,
                                    result={k:v for k,v in row.items() if k!='phases'}))
        reference=read_json('docs/R29_INCUMBENT.json')['codes']
        local=range(24)
        donors=sum(7<<(3*(reference[c]//3)) for c in local)
        ds=tuple(donors if c in local else 1<<v for c,v in enumerate(reference))
        for target in (207,208,240):
            for layers in ((0,),tuple(range(6))):
                for regions in ((),(list(local),)):
                    add('slot-projection',target,lambda:projected(problem,ds,layers=layers,
                                                                 target=target,regions=regions,effort=.1))
        p,ds=planted(1,2,19)
        fixed=tuple(1<<(3*t) for t in range(p.n))
        for idx in (0,1):
            for repeat in (0,1):
                row=lift_cube(p,inventory(p).required,phase_cube(p,1,idx),fixed,1,database)
                results.append(dict(name='planted-lift',index=idx,repeat=repeat,result={k:v for k,v in row.items()
                                    if k in ('status','lift_calls','reused','proved_pins','branches')}))
    elif section=='cache':
        phases=fit(problem,required,{0:0},effort=.15)['phases']
        cached=SignedRepair(model,phases,forbidden=forbidden,database=database)
        cold=SignedRepair(model,phases,forbidden=forbidden)
        endpoints=set()
        for trial in (0,1):
            for seed in seeds:
                rng=np.random.default_rng(seed)
                from r31_sampling import ranked_start
                board=Assemblies(model).seed(ranked_start(model,seed),rng)
                protected=Assemblies(model).protected(board)
                before=dict(cached.stats)
                imported=duplicate_boards=0
                own_keys=set()
                local_endpoints=set()
                for _ in range(rounds):
                    cells=np.setdiff1d(model.independent_set(board,rng),protected)
                    key=board.tobytes()
                    duplicate_boards+=int(key in endpoints)
                    local_endpoints.add(key)
                    test=cached.block(board,cells)
                    imported+=len(cached.last_hit_keys-own_keys)
                    own_keys.update(cached.last_hit_keys | cached.last_new_keys)
                    check=cold.block(board,cells)
                    assert np.array_equal(test,check)
                    board=test
                endpoints.update(local_endpoints)
                results.append(dict(cross_seed_boards=duplicate_boards,preexisting_row_keys=imported,name='replay-rows',seed=seed,pass_index=trial,
                                    result={k:v-before.get(k,0) for k,v in cached.stats.items()},
                                    final=cached.score(board)))
    return results


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('section',choices=['cpu','cuda','geometry','cache'])
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--database',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--seeds',type=int,nargs='+',default=[4,5,6,7])
    cli.add_argument('--rounds',type=int,default=64)
    args=cli.parse_args()
    args.database.parent.mkdir(parents=True,exist_ok=True)
    p=SearchProblem(read_json(args.tiles))
    rows=compare(p,args.section,args.database,args.seeds,args.rounds)
    atomic_json(args.output,dict(fingerprint=p.fingerprint,section=args.section,rounds=args.rounds,rows=rows))
