"""Reproduce threshold, donor reuse, lazy endpoint, and simultaneous bridge checks."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from r27_incumbent import BoardModel
from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r29_gold import phase_search
from r30_exact_routes import CycleModel, cycle_type, learn, rules_in
from r30_gold import GoldMatching, search, restriction_core, core_applies
from search_problem import SearchProblem
from test_r30_exactgold import bridge_fixture, pairing_oracle


def small(record):
    return {k:v for k,v in record.items() if k not in ('codes','pairs','phases','discovered_type_sets','reusable_type_sets','physical_cutsets')}


def run(tiles,center, seconds, relaxation):
    p=SearchProblem(read_json(tiles))
    b=read_json(center)['codes']
    m=BoardModel(p)
    cells=[18,30,44,51,62,146]
    pins=list(zip(cells,[30,44,51,18,146,62]))
    r={}
    with TemporaryDirectory() as temp:
        db=Path(temp)/'rules.sqlite'
        for cycles in ((4,2),(3,3)):
            q=CycleModel(m,b,cells,cycles)
            r[str(cycles)]={key:small(q.solve(target=208,seconds=seconds,**flags))
                            for key,flags in (("independent",{}),("contact",{"contact":True}),
                                              ("contact_predecessors",{"contact":True,"predecessors":True}))}
            assert r[str(cycles)]["contact_predecessors"]["predecessor_cuts"]==len(cycles)
            assert all(v['status']=='INFEASIBLE' for v in r[str(cycles)].values())
        q=CycleModel(m,b,cells,(4,2))
        r["predecessors_unknown"]=small(q.solve(target=208,predecessors=True,seconds=1e-9))
        assert r["predecessors_unknown"]["status"]=="UNKNOWN"
        learned=learn(q,pins,204,seconds=seconds,max_delete=4,database=db)
        assert learned['status']=='CERTIFIED'
        rule=learned['rule']
        matches=skipped=0
        for row in permutations(cells):
            if cycle_type([cells.index(j) for j in row])!=(4,2):
                continue
            ds=dict(zip(cells,row))
            if all(ds[i]==j for i,j in rule['pins']):
                matches+=1
                answer=q.solve(target=204,pins=ds.items(),rules=rules_in(db,q.binding))
                assert answer['status']=='INFEASIBLE'
                skipped+=answer['cp_calls']==0
        r['learned_real_204']=dict(learned,matching_endpoints=matches,zero_call_hits=skipped)
        r['real_pool_feasible_204']=small(q.solve(target=204,seconds=seconds))
        r['pool_bound_208']=learn(q,pins,208,seconds=seconds,max_delete=6)
    r['full_family_trial']={}
    for cycles in ((4,2),(3,3)):
        full=CycleModel(m,b,None,cycles)
        result=full.solve(target=208,seconds=seconds)
        r['full_family_trial'][str(cycles)]=small(result)
    positives={}
    psmall,ds=planted(1,0,23,False,True)
    identity=[3*i for i in range(psmall.n)]
    for k in ((4,2),(3,3)):
        q=CycleModel(BoardModel(psmall),identity,list(range(6)),k)
        positives[str(k)]=small(q.solve(target=len(psmall.edges),seconds=seconds,predecessors=True))
        assert positives[str(k)]['gold'] and positives[str(k)]['predecessor_cuts']==0
    gold={}
    earlier=read_json(Path(__file__).parent/'docs/R29_GOLD_RESULTS.json')['endpoint_rules']
    for name in ('five','twelve'):
        pins={int(k):v for k,v in earlier[name].items()}
        first=search(p,pins=pins,matching=False,seconds=seconds)
        warm=search(p,pins=pins,matching=False,seconds=seconds,type_cuts=first['discovered_type_sets'])
        assert first['status']==warm['status']=='INFEASIBLE'
        previous=phase_search(p,pins=pins,width=2 if name=='five' else 3,cp=False,warm_cuts=False)
        integrated=[]
        for flip in (0,1):
            effect=phase_search(p,pins={i:v^flip for i,v in pins.items()},cp=False)
            assert effect['status']=='INFEASIBLE' and effect['reason']=='endpoint-crossings'
            integrated.append(dict(upper=effect['cut']['upper'],subsets=effect['metrics']['subsets'],
                                   cp_calls=effect['metrics']['cp_calls']))
        assert previous['status']=='INFEASIBLE'
        gold[name]=dict(phase_entry_previous=previous['metrics']['subsets'],phase_entry_reuse=integrated,
                        cold=small(first),warm=small(warm),
                        learned_cut_widths=list(map(len,first['discovered_type_sets'])),
                        previous_subsets=earlier['pairs' if name=='five' else 'triples']['subsets'])
    allowed,banned,fixed,pins=bridge_fixture(psmall)
    base=GoldMatching(psmall,domain_bits(psmall),pins,[],[],[])
    independent=pairing_oracle(base,allowed,fixed)
    bridge=search(psmall,pins=pins,bonds=fixed,forbidden=banned,seconds=seconds)
    bridge_model=GoldMatching(psmall,domain_bits(psmall),pins,fixed,(),banned)
    capacities=[]
    for group in bridge["physical_cutsets"]:
        capacities.append(bridge_model.crossing_capacity(group,seconds))
        bridge_model.strand_cut(group)
    core=restriction_core(psmall,pins=pins,bonds=fixed,forbidden=banned,seconds=seconds,max_delete=4)
    assert core_applies(psmall,None,pins,fixed,banned,core)
    gold['competing_bridges']=dict(oracle=dict(independent),coupled=small(bridge),successive_capacities=capacities,
        core_recheck=core['status'],core_queries=core['queries'],core_entries=len(core['restrictions']),
        restricted_entries=len(fixed)+len(pins)+len(banned))
    gold['planted_embedding']=small(search(psmall,domains=ds,geography=True,seconds=seconds))
    if relaxation:
        gold['unrestricted']=small(search(p,matching=True,seconds=seconds,max_rounds=20))
    return dict(scope=p.fingerprint,cells=cells,seconds_per_query=seconds,routes=r,
                planted_route=positives,gold=gold)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--incumbent',default='docs/R29_INCUMBENT.json')
    cli.add_argument('--seconds',type=float,default=5)
    cli.add_argument('--include-root-gold',action='store_true')
    cli.add_argument('--output',type=Path)
    a=cli.parse_args()
    results=run(a.tiles,a.incumbent,a.seconds,a.include_root_gold)
    if a.output:
        from io_utils import atomic_json
        atomic_json(a.output,results)
    else:
        print(json.dumps(results,indent=2))
