"""Compare geographic tile-neighbor conditions with physical matching and corner support."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
from unittest.mock import patch

from r27_structure_filters import Work,domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r28_geo_sewing import PosePacking
from r30_gold import GoldMatching,search
from r31_geographic import physical_bonds
from r32_gold_bonds import gold_matching_supports,inventory
from r35_corner_chains import filter_options
from r35_physical_adjacency import filter_neighbors
from search_problem import SearchProblem


def residuals(p,reference):
    rules=inventory(p)
    choices=sorted(set(physical_bonds(p,reference))&rules.possible)
    result=[]
    for depth in (4,12,24,48,96,144):
        stats=Counter()
        for seed in range(10):
            bonds=Random(seed).sample(choices,depth)
            fixed=dict(bonds)
            fixed.update({b:a for a,b in bonds})
            rows=[set() for _ in range(3*p.n)]
            for a,b in rules.possible:
                if fixed.get(a,b)==b and fixed.get(b,a)==a:
                    rows[a].add(b);rows[b].add(a)
            supported=gold_matching_supports(p,rows,Work(2000000))
            if supported is None:
                stats['prior_matching_closed']+=1
                continue
            pairs={(a,b) for a,row in enumerate(supported) for b in row if a<b}
            required={(a,b) for a,b in pairs if len(supported[a])==1 or len(supported[b])==1}
            live,work=filter_neighbors(p,pairs,required)
            with_cut,extra=filter_neighbors(p,pairs,required,connect=True,work_limit=5000000)
            corner=filter_options(p,pairs,required)[0]
            stats['local_closed']+=not live
            stats['newly_closed_over_corner']+=bool(corner) and not live
            stats['connectivity_only_closed']+=bool(live) and not with_cut
            stats['undecided_connectivity']+=extra.get('separator_unknown',0)
            stats['retained_mates_removed']+=len(pairs-live) if live else 0
            stats['distinct_from_corners_on_both_survivors']+=len(corner-live) if live and corner else 0
            for field in ('fixed_capacity','neighbor_hall'):
                stats[field]+=work.get(field,0)
            if live:
                stats['independently_infeasible_pose_checks']+=sum(
                    PosePacking(p).check(required|{pair},max_steps=10000)['feasible'] is False
                    for pair in pairs-live)
        result.append(dict(pins=depth,**stats))
    return result


def native(p):
    import wgpu
    from dfs_gpu import DFS,_physical_gold_table,fill_order
    from dfs_webgpu import WebGPUDFS
    from r35_physical_adjacency import geographic_inventory
    order=fill_order(p.neighbors)
    a,b=map(int,order[:2])
    side=list(p.neighbors[a]).index(b);rev=int(p.sides[a,side])
    exclusions=sorted(geographic_inventory(p).forbidden-inventory(p).forbidden)
    prefixes=[[3*(x//3)+(side-x%3)%3,3*(y//3)+(rev-y%3)%3]
              for pair in exclusions for x,y in (pair,pair[::-1])]
    apis=[('CUDA',DFS,{})]+[(adapter.info['backend_type'],WebGPUDFS,dict(adapter=adapter))
                            for adapter in wgpu.gpu.enumerate_adapters_sync()
                            if adapter.info['adapter_type']=='DiscreteGPU' and
                            adapter.info['backend_type'] in ('Vulkan','D3D12','Metal')]
    result={}
    for label,factory,kwargs in apis:
        runs=[]
        for rule in (inventory,geographic_inventory):
            _physical_gold_table.cache_clear()
            with patch('r35_physical_adjacency.geographic_inventory',rule):
                engine=factory(p.masks,p.neighbors,p.sides,n=len(prefixes),order=order,prefixes=prefixes,
                               gold_data=p.tiles,seed=91,**kwargs)
            try:
                engine.step(16)
                engine.verify()
                counters=engine.counters.get().sum(axis=1)
                runs.append(dict(exhausted=int((engine.states.get()==2).sum()),tries=int(counters[0]),
                                 placements=int(counters[1]),backtracks=int(counters[2])))
            finally:
                if hasattr(engine,'close'):engine.close()
        assert runs[1]['exhausted']==len(prefixes) and runs[1]['tries']==0 and runs[0]['tries']>0,runs
        result[label]=runs
    _physical_gold_table.cache_clear()
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--native',action='store_true')
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    rules=inventory(p)
    live,stats=filter_neighbors(p,rules.possible,rules.required,connect=True,work_limit=5000000)
    original=GoldMatching(p,domain_bits(p),None,(),neighbor_mode='local')
    assert not original.m.validate()
    out=dict(root=dict(before=len(rules.possible),after=len(live),removed=sorted(rules.possible-live),
                       **stats,**{'cp_pair_limits':original.neighbor_stats['pair_bounds']}),
             residuals=residuals(p,read_json('docs/R29_INCUMBENT.json')['codes']))
    out['planted']=[]
    for seed in (35,71,103,157):
        q,d=planted(1,3,seed)
        entry={'seed':seed}
        for mode in (None,'local','connectivity','channel'):
            res=search(q,domains=d,geography=True,neighbor_mode=mode if mode!='channel' else None,
                       channel=mode=='channel',seconds=10,max_rounds=40)
            entry[mode or 'prior']={k:res[k] for k in ('status','cp_calls','branches') if k in res}
            if res['status']=='GOLD':assert q.validate(res['codes'])['valid']
        out['planted'].append(entry)
    if args.native:
        out['native_prefixes_prior_then_trial']=native(p)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
