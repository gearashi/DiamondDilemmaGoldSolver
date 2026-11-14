"""Paired original and finite-control checks for the nine Gold proposals."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import sqlite3

from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted,nested_loops
from r28_exchange import read_json
from r28_geo_color import dual_cuts
from r28_geo_sewing import ring_census,side_options
from r30_gold import GoldMatching,search,embedding
from r31_geographic import GeographicConflicts, departure_cores, physical_bonds, ribbon_genus, salvage, tighten_bridge
from r31_gold_connectivity import PhaseInterface,connected_colors
from r31_gold_pricing import price_rings
from r31_gold_switch import sewn_switches
from search_problem import SearchProblem
from test_r28_geometry import phases,seams
from test_r30_exactgold import bridge_fixture
from validator import reverse_mask


def counts(row):
    return {k:v for k,v in row.items() if k not in ('codes','pairs','phases','discovered_type_sets',
            'reusable_type_sets','physical_cutsets','witness','witnesses','geographic_cores')}


def mutations(problem, rows):
    for i,j in combinations(range(len(rows)),2):
        a,b=rows[i]
        c,d=rows[j]
        if len({v//3 for v in (a,b,c,d)})!=4:continue
        for q,r in (((a,c),(b,d)),((a,d),(b,c))):
            if any(int(problem.masks[3*(u//3),u%3])!=reverse_mask(int(problem.masks[3*(v//3),v%3])) for u,v in (q,r)):
                continue
            yield tuple(sorted(tuple(sorted(pair)) for pair in [v for k,v in enumerate(rows) if k not in (i,j)]+[q,r]))


def run(tiles,center,fresh=()):
    import json
    p,ds=planted(1,0,23,False,True)
    all_ds=domain_bits(p)
    rows=seams(p)
    cc=GeographicConflicts(p,all_ds)
    first=list(mutations(p,rows))
    cache_training=0
    for row in first:
        if ribbon_genus(p,row):
            cc.learn(row)
            cache_training+=1
    holdout={other for row in first for other in mutations(p,row)}-set(first)
    saved=0
    for row in holdout:
        if any(c<=set(row) for c in cc.cores):
            mate=dict(row)
            mate.update((b,a) for a,b in row)
            assert embedding(p,all_ds,[mate[i] for i in range(3*p.n)]) is None
            saved+=1
    bad=next(r for r in first if ribbon_genus(p,r))
    required=[z for z in rows if tuple(sorted(z)) in set(bad)][:2]
    recovered=salvage(p,ds,required,bad,seconds=3)
    result={'geometry':dict(training_matchings=cache_training,cores=len(cc.cores),
                            core_sizes=list(map(len,cc.cores)),holdout_matchings=len(holdout),
                            avoided_full_embeddings=saved,full_pairs=len(rows)),
            'salvage':counts(recovered)}
    cases=[('planted',*planted(1,3,23,False,True)),('split',*planted(1,3,23,True,True)),
           ('nested',*nested_loops()),('large_planted',*planted(4,4,73))]
    result['geographic']={}
    for label,source,dom in cases:
        result['geographic'][label]={k:counts(search(source,domains=dom,geography=True,
            seconds=6,learned_geometry=k!='old',channel=k=='channel')) for k in ('old','learn','channel')}
        result['geographic'][label]['regions']={k:counts(connected_colors(source,dom,k,seconds=4))
                                                 for k in ('parents','flow','lazy')}
    allowed,banned,fixed,pins=bridge_fixture(p)
    result['fractional']={}
    for name,opts in (('baseline',{}),('lp',{'fraction':True})):
        result['fractional'][name]=counts(search(p,pins=pins,bonds=fixed,forbidden=banned,seconds=4,**opts))
    cuts=search(p,pins=pins,bonds=fixed,forbidden=banned,seconds=4)['physical_cutsets']
    checker=GoldMatching(p,all_ds,pins,fixed,(),banned)
    result['bridges']=[tighten_bridge(checker,g) for g in cuts]
    for g in [cuts[1]]:
        result['bridges_result']=counts(search(p,pins=pins,bonds=fixed,forbidden=banned,bridges=[g],seconds=4))
    planted_bonds=GoldMatching(p,ds,None,rows)
    result['planted_tight_cut']=tighten_bridge(planted_bonds,{0})
    path={0}
    for pair in rows:
        links=planted_bonds.transitions[tuple(sorted(pair))]
        if any((a in path)!=(b in path) for a,b in links):
            a,b=next((a,b) for a,b in links if (a in path)!=(b in path))
            path.add(b if a in path else a)
            break
    result['nested_path_cuts']=counts(search(p,domains=ds,bonds=rows,geography=True,bridges=[[0],path],seconds=4))
    result['import_bridge']=counts(search(p,pins=pins,bonds=fixed,forbidden=banned,
                                           bridges=result['bridges_result']['bridge_hints'],seconds=4))
    root=SearchProblem(read_json(tiles))
    center_codes=read_json(center)['codes']
    def mine(codes):
        r=departure_cores(root,physical_bonds(root,codes),trials=256,seconds=4)
        return dict(lower_bound=r['lower_bound'],upper=len(r.get('repair_bonds',())),
                    minimum=r.get('minimum_breaks'),witnesses=r['disjoint_witnesses'],
                    distinct_cores=len(r['cores']),core_sizes=dict(Counter(map(len,r['cores']))))
    result['binding']=root.fingerprint
    result['departures']={'center':mine(center_codes)}
    for name in fresh:
        with sqlite3.connect(Path(name).resolve().as_uri()+'?mode=ro',uri=True) as db:
            states=list(db.execute('SELECT seed,state FROM r30_trajectory WHERE rounds=320 ORDER BY seed'))
        result['departures'][Path(name).stem]=[
            dict(seed=seed,score=(st:=json.loads(state))['best']['score'],**mine(st['best']['codes']))
            for seed,state in states]
    result['pricing']={}
    orig=GoldMatching(root,domain_bits(root),None,[])
    options=[set() for _ in range(3*root.n)]
    for a,b in orig.mate:
        options[a].add(b)
        options[b].add(a)
    for name,source,opts in (('planted',p,side_options(p,ds)),('original',root,options)):
        result['pricing'][name]=dict(old=ring_census(source,opts,max_work=100000),
                                      priced=price_rings(source,opts,max_work=100000))
    interface=PhaseInterface(p)
    q=interface.project({},(0,2))
    result['phase']={'cold':q,'warm':interface.project({},(0,2)),
                     'conditional':interface.project({0:1},(2,)),
                     'original':PhaseInterface(root).project({},(0,1),max_states=512)}
    from ortools.sat.python import cp_model
    from r28_geo_color import signed_phase_model
    m,phase_vars=signed_phase_model(root)
    phase_solver=cp_model.CpSolver()
    phase_solver.parameters.num_search_workers=1
    phase_solver.parameters.max_time_in_seconds=10
    status=phase_solver.solve(m)
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        pins={i:phase_solver.value(x) for i,x in enumerate(phase_vars) if i>=12}
        inventory=PhaseInterface(root)
        exact=inventory.project(pins,(0,1,2,3),512)
        result['phase']['real_residual']=dict(pins=pins,boundary=[0,1,2,3],result=exact)
        result['phase']['real_child_queries']=[]
        for z in (0,1):
            child=inventory.project(pins|{0:z},(1,2),512)
            check,_=signed_phase_model(root,pins=pins|{0:z})
            sr=phase_solver.solve(check)
            cp_status=phase_solver.status_name(sr)
            assert sr in (cp_model.INFEASIBLE,cp_model.OPTIMAL,cp_model.FEASIBLE)
            assert bool(child['patterns'])==(sr!=cp_model.INFEASIBLE)
            result['phase']['real_child_queries'].append(dict(value=z,check=cp_status,**child))
    base={u:v for a,b in rows for u,v in ((a,b),(b,a))}
    a,b=0,7
    c,d=base[a],base[b]
    alt=base|{a:d,d:a,b:c,c:b}
    grouped=sewn_switches(p,ds,[(u,v) for u,v in alt.items() if u<v],phases(p),[(a,b)])
    result['switches']=dict(counts(grouped),verified=len(grouped['witnesses']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--center',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--fresh',action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from io_utils import atomic_json
    atomic_json(args.output,run(args.tiles,args.center,args.fresh))
