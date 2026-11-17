"""Benchmark the existing phase projection and grouped compilation on original plus partial pins."""
import argparse
import json
from itertools import product
from pathlib import Path
from random import Random

from ortools.sat.python import cp_model
from bench_r35_phase_lattice import original_cases
from r28_exchange import read_json
from r28_geo_color import signed_phase_model
from r31_gold_connectivity import PhaseInterface
from r35_phase_frontier import compile_projection
from search_problem import SearchProblem


def oracle(p,pins,boundary,relations=(),euler=True):
    patterns=[]
    for mask in range(1<<len(boundary)):
        values=dict(pins); bits={t:(mask>>i)&1 for i,t in enumerate(boundary)}
        if any(t in pins and pins[t]!=v for t,v in bits.items()):continue
        values.update(bits)
        model,x=signed_phase_model(p,euler=euler,pins=values)
        for u,v,flip in relations:model.add(x[u]!=x[v] if flip else x[u]==x[v])
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=5
        status=solver.solve(model)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):patterns.append(mask)
        elif status!=cp_model.INFEASIBLE:raise TimeoutError('Independent native oracle did not complete.')
    return patterns


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json');parser.add_argument('--output',type=Path)
    args=parser.parse_args();p=SearchProblem(read_json(args.tiles));x=PhaseInterface(p)
    results=[]
    for seed,source in ((28,'raw'),(35,'contract')):
        phases=read_json('docs/R35_CONNECTED_TOURS.json')[source]['phases']
        order=list(range(len(phases)));Random(seed).shuffle(order)
        for free in (0,8,20,40,60,80,120,160):
            pins={t:phases[t] for t in order[free:]};boundary=tuple(order[:2])
            x.cache.clear()
            old=x.project(pins,boundary,max_states=10000,method='legacy')
            new=compile_projection(x,pins,boundary,max_states=10000)
            split=compile_projection(x,pins,boundary,max_states=10000,split=True)
            expected=oracle(p,pins,boundary)
            for result in (old,new,split):
                if result['status']=='COMPLETE':assert expected==result['patterns'],(expected,result)
            row=dict(seed=seed,free=free,baseline=old,frontier=new,split=split,native=expected)
            results.append(row)
            print(seed,free,old['status'],old['transitions'],new,flush=True)
    actual=[]
    for name,rel in original_cases(p):
        if 'historical_core_delete' in name and name['historical_core_delete'] not in (0,27,50):continue
        new=compile_projection(x,{},(0,1),relations=rel,euler=False,max_states=10000)
        expected=oracle(p,{},(0,1),rel,False)
        if new['status']=='COMPLETE':assert new['patterns']==expected
        actual.append(dict(scope=name,frontier=new,native=expected))
    data=dict(controlled=results,archived=actual)
    if args.output:args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(data))

if __name__=='__main__':main()
