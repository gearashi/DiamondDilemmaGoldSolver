"""Independent physical circuit enumerations and comparisons to geographic circuit/region models."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from random import Random

from r27_structure_filters import bits,domain_bits
from r27_structure_fixtures import planted,nested_loops
from r28_exchange import read_json
from r28_geo_color import signed_phase_model
from r30_gold import GoldMatching,search
from r31_geographic import physical_bonds
from r31_gold_connectivity import connected_colors
from r32_gold_bonds import endpoints,loop_groups
from search_problem import SearchProblem


def family(p):
    from ortools.sat.python import cp_model
    d=domain_bits(p)
    true=set(physical_bonds(p,[3*t for t in range(p.n)]))
    m,x=signed_phase_model(p,bonds=true)
    solver=cp_model.CpSolver()
    assert solver.solve(m)==cp_model.OPTIMAL
    phases=[solver.value(v) for v in x]
    check=GoldMatching(p,d,dict(enumerate(phases)),(),physical_support=False)
    options=set(true)
    used=set()
    for pair1,pair2 in combinations(sorted(true),2):
        if used.intersection(pair1+pair2):
            continue
        a,b=pair1;c,e=pair2
        if not check.inputs.sides[a][0]:
            continue
        accepted=False
        for trial in (((a,c),(b,e)),((a,e),(b,c))):
            trial={tuple(sorted(pair)) for pair in trial}
            if not trial<=check.mate.keys():
                continue
            for u,v in trial:
                mask,left=check.inputs.sides[u];_,right=check.inputs.sides[v]
                if left^right^phases[u//3]^phases[v//3]^(mask.bit_count()%2):
                    break
            else:
                options.update(trial);used.update(pair1+pair2);accepted=True
            if accepted:break
    return phases,set(check.mate)-options,options,true


def complete_models(p,phases,banned,mode):
    from ortools.sat.python import cp_model
    solver=cp_model.CpSolver()
    check=GoldMatching(p,domain_bits(p),dict(enumerate(phases)),(),forbidden=banned,physical_support=False,tour_mode=mode)
    found=set()
    while solver.solve(check.m)==cp_model.OPTIMAL:
        chosen=frozenset(k for k,v in check.mate.items() if solver.value(v))
        found.add(chosen)
        check.m.add_bool_or([~check.mate[k] for k in chosen])
        assert len(found)<=512
    return found


def enumerate_family(seed):
    from bench_r35_physical_sat import solve_fixed
    q,_=planted(1,10,seed,uniform_ports=True)
    phases,banned,_,true=family(q)
    raw=complete_models(q,phases,banned,None)
    arcs=endpoints(q)
    slots=[tuple(k for k in range(1,12) if (d,k) in arcs) for d in range(3*q.n)]
    good={pairs for pairs in raw if not loop_groups(arcs,slots,pairs,q.total_segments)}
    assert frozenset(true) in good
    stats={}
    for mode in ('raw','contract','sparse'):
        found=complete_models(q,phases,banned,mode)
        assert found==good
        stats[mode]=len(found)
    check=GoldMatching(q,domain_bits(q),dict(enumerate(phases)),(),forbidden=banned,physical_support=False)
    sat=solve_fixed(check,phases,(),limit=513)
    assert sat['complete'] and {frozenset(map(tuple,model)) for model in sat['models']}==good
    return dict(seed=seed,all_sewings=len(raw),independent_connected=len(good),cp=stats,
                sat={k:v for k,v in sat.items() if k!='models'})


def link_region_placements(check,choices):
    """Couple selected complement-region placements to the very same physical sewing."""
    for a,s,b,t in check.problem.edges:
        for v,first in choices[a].items():
            x=3*(v//3)+(s-v%3)%3
            for w,second in choices[b].items():
                y=3*(w//3)+(t-w%3)%3
                physical=check.mate.get(tuple(sorted((x,y))))
                row=[~first,~second]
                if physical is not None:
                    row.append(physical)
                check.m.add_bool_or(row)

def coupled_region(p,d,method,check):
    from ortools.sat.python import cp_model
    from r28_geo_color import _sphere
    from r31_gold_connectivity import _post_regions
    assert tuple(d)==tuple(check.domains) and check.blank_projection is None
    corner,_=_sphere(p)
    choices,_,_=_post_regions(p,d,method,corner,check.m)
    link_region_placements(check,choices)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=5
    status=solver.solve(check.m)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        codes=[next(v for v,x in row.items() if solver.value(x)) for row in choices]
        assert p.validate(codes)['valid']
    return dict(status=solver.status_name(status),branches=solver.num_branches,boards=int(
        status in (cp_model.FEASIBLE,cp_model.OPTIMAL)))


def region_case(p,d):
    from ortools.sat.python import cp_model
    from constraint_circuit import build_circuit_model
    out={}
    for label,mode,channel in [('lazy',None,False),('direct','raw',False),('contract','contract',False),
                                ('channel',None,True),('combined_channel','contract',True)]:
        result=search(p,domains=d,tour_mode=mode,channel=channel,geography=True,seconds=5,max_rounds=40)
        out[label]={k:v for k,v in result.items() if k in ('status','cp_calls','branches','strand_cuts','physical_circuit')}
        if result['status']=='GOLD':assert p.validate(result['codes'])['valid']
    for mode in ('parents','flow'):
        for coupled in (False,True):
            physical=GoldMatching(p,d,None,(),tour_mode='contract') if coupled else None
            result=coupled_region(p,d,mode,physical) if coupled else connected_colors(p,d,mode,seconds=5)
            out[mode+('_with_tour' if coupled else '')]={k:v for k,v in result.items() if k in ('status','branches','boards')}
    geographic=build_circuit_model(p,[list(bits(v)) for v in d],contract_fixed=True)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=5
    status=solver.solve(geographic.model)
    out['geographic_circuit']=dict(status=solver.status_name(status),branches=solver.num_branches)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):assert p.validate([solver.value(v) for v in geographic.codes])['valid']
    return out


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--save-tours',type=Path)
    cli.add_argument('--root-seconds',type=float,default=20)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    result=dict(enumeration=[enumerate_family(seed) for seed in (35,71,103,157)],geographic_controls={})
    for name,(q,d) in [('planted',planted(1,3,35)),('split',planted(1,3,35,split=True)),('five_loops',nested_loops())]:
        result['geographic_controls'][name]=region_case(q,d)
    result['physical_controls']=[]
    for seed in (35,71,103,157):
        q,d=planted(1,6,seed,uniform_ports=True)
        entry={'seed':seed}
        for mode in (None,'raw','contract','sparse'):
            res=search(q,domains=d,tour_mode=mode,project_blanks=True,seconds=10,max_rounds=40)
            entry[mode or 'lazy']={k:v for k,v in res.items() if k in ('status','cp_calls','branches','physical_circuit','strand_cuts')}
        result['physical_controls'].append(entry)
    reference=read_json('docs/R29_INCUMBENT.json')['codes']
    result['geographic_controls']['original_all_tiles_rotate']=region_case(p,tuple(7<<(3*(v//3)) for v in reference))
    d=domain_bits(p)
    fixed=read_json('docs/R35_NONBLANK_SEWING.json')
    params=dict(domains=d,pins=dict(enumerate(fixed['phases'])),bonds=list(map(tuple,fixed['nonblank'])),
                project_blanks=True,seconds=10)
    result['original_nonblank']={}
    for geographic in (False,True):
        for mode in (None,'raw','contract','sparse'):
            res=search(p,**params,tour_mode=mode,geography=geographic)
            result['original_nonblank'][str(geographic)+'_'+str(mode)]={k:v for k,v in res.items()
                if k in ('status','cp_calls','branches','physical_circuit','strand_cuts','blank_root_trials')}
    result['original_root']={}
    for mode in ('raw','contract','sparse'):
        check=GoldMatching(p,d,None,(),tour_mode=mode)
        assert not check.m.validate()
        result['original_root'][mode]=check.tour_stats
    result['original_projected_trace']={}
    tours={}
    for mode in (None,'raw','contract','sparse'):
        res=search(p,domains=d,project_blanks=True,tour_mode=mode,max_rounds=4,seconds=args.root_seconds)
        result['original_projected_trace'][mode or 'lazy']={k:v for k,v in res.items()
            if k in ('status','cp_calls','branches','strand_cuts','physical_circuit','blank_phase_failures')}
        if res['status']=='CONNECTED_SEWING':
            from r30_gold import embedding
            from r31_geographic import ribbon_genus
            arcs=endpoints(p)
            slots=[tuple(k for k in range(1,12) if (dart,k) in arcs) for dart in range(3*p.n)]
            pairs=list(map(tuple,res['pairs']))
            assert not loop_groups(arcs,slots,pairs,p.total_segments)
            mate=dict(pairs);mate.update({b:a for a,b in pairs})
            codes=embedding(p,d,[mate[i] for i in range(3*p.n)])
            if codes is not None:assert p.validate(codes)['valid']
            tours[mode]=dict(phases=res['phases'],pairs=pairs,genus=ribbon_genus(p,pairs),codes=codes)
            result['original_projected_trace'][mode].update(genus=tours[mode]['genus'],geographic=codes is not None)
    if args.save_tours:args.save_tours.write_text(json.dumps(tours,separators=(',',':'))+'\n',encoding='utf-8')
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
