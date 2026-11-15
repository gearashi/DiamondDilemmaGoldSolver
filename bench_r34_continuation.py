"""Separate physical-device table reuse and phase-trajectory extension controls."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from r28_exchange import read_json
from search_problem import SearchProblem


def device(problem):
    from gpu_domain_round24 import DomainCUDA
    import cupy as cp
    from test_constraint_dfs import tetrahedron_tiles,brute
    from test_gpu_domain_round24 import enumerate_engine
    from gpu_domain_runner import split_cubes
    builds=[]
    original=DomainCUDA._tables
    def track(engine):
        table=original(engine)
        builds.append(table.nbytes)
        return table
    with patch.object(DomainCUDA,'_tables',track):
        engines=[]
        try:
            for target,event in (('gold',False),('gold',False),('edges',False),('gold',True)):
                engines.append(DomainCUDA(problem,target=target,event_propagation=event))
            assert len(builds)==3
            a,b=engines[:2]
            assert a.tables is b.tables and a.state.data.ptr != b.state.data.ptr
            cp.testing.assert_array_equal(a.tables,b.tables)
            pristine=b.state.copy()
            a.step(32)
            cp.testing.assert_array_equal(b.state,pristine)
            a.close()
            b.step(32)
        finally:
            for e in engines:e.close()
    p=SearchProblem(*tetrahedron_tiles())
    cubes=split_cubes([tuple(sum(1<<c for c in row) for row in p.domains)],2)
    streams=[cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    instances=[]
    try:
        for stream in streams:
            with stream:
                instances.append(DomainCUDA(p,cubes,target='gold'))
        assert instances[0].tables is instances[1].tables
        expected=set(map(tuple,brute(p,'gold')))
        for e,stream in zip(instances,streams):
            with stream:assert set(map(tuple,enumerate_engine(e,8)))==expected
    finally:
        for e in instances:e.close()
    return dict(engines=4,builds=len(builds),shared_upload_bytes_avoided=builds[0],distinct_layout_bytes=builds,
                independent_finite_gold_solutions=len(expected),independent_streams=2)


def trajectory(problem,backend='cpu'):
    from r32_fresh import run_fresh_seeds, fresh
    opts=dict(rounds=16,cube_bits=2,refit=8,merges=0,backend=backend,replicas=128,dispatches=3,effort=.15,contour=False)
    restored=[]
    def observe(*args,**kw):
        saved=kw.get('saved')
        restored.append(saved['cursor'] if saved else 0)
        return fresh(*args,**kw)
    with tempfile.TemporaryDirectory() as temp, patch('r32_fresh.fresh',observe):
        hot=Path(temp)/'prior.sqlite'
        cold=Path(temp)/'cold.sqlite'
        run_fresh_seeds(problem,hot,0,1,opts)
        extended=run_fresh_seeds(SearchProblem(problem.tiles,problem.board),hot,0,1,dict(opts,rounds=64))
        baseline=run_fresh_seeds(SearchProblem(problem.tiles,problem.board),cold,0,1,dict(opts,rounds=64))
        def state(path):
            with closing(sqlite3.connect(path)) as db:
                from r34_signed_snapshot import expand
                return expand(db,json.loads(db.execute("SELECT state FROM r34_trajectories WHERE cursor=64").fetchone()[0]))
        left,right=state(hot),state(cold)
        assert left==right and extended['best']==baseline['best'],{
            k:(left[k],right[k]) for k in left if left[k]!=right[k]}
        run_fresh_seeds(problem,hot,0,1,dict(opts,rounds=8,cube_bits=1))
        assert restored==[0,16,0,0],restored
        return dict(backend=backend,initial_rounds=16,extended_rounds=64,repeated_rounds_avoided=16,new_rounds=48,
                    same_rng_phases_and_board=True,score=extended['best']['score'],work=extended['best']['work'],
                    snapshot_darts=state(hot)['dart_prices']['count'],snapshot_json_bytes=len(
                        json.dumps(state(hot)['dart_prices'],separators=(',',':')).encode()))


def price_storage(problem):
    from r32_fresh import run_fresh_seeds
    from r34_signed_snapshot import expand
    opts=dict(rounds=256,cube_bits=2,refit=8,merges=0,backend='cpu',replicas=128,dispatches=3,effort=.15,contour=False)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'phases.sqlite'
        run_fresh_seeds(problem,path,0,1,opts)
        with closing(sqlite3.connect(path)) as db:
            raw=[json.loads(row[0]) for row in db.execute('SELECT state FROM r34_trajectories ORDER BY cursor')]
            stored=int(db.execute('SELECT sum(length(payload)) FROM r34_price_blobs').fetchone()[0])
            blobs=db.execute('SELECT count(*) FROM r34_price_blobs').fetchone()[0]
            keys=sum(len(json.dumps({'dart_price_blob':row['dart_price_blob']})) for row in raw)
            expanded=[expand(db,row)['dart_prices'] for row in raw]
            earlier=sum(len(json.dumps(row)) for row in expanded)
            assert len(raw)==32 and blobs==len({row['checksum'] for row in expanded})
            return dict(cursors=len(raw),unique_images=blobs,inline_price_json_bytes=earlier,
                        shared_price_json_bytes=stored,reference_json_bytes=keys)


def classic(problem,apis=None,dxc=None):
    from contextlib import nullcontext
    from dfs_gpu import DFS,fill_order
    from dfs_webgpu import WebGPUDFS
    from r32_gold_bonds import inventory
    import numpy as np
    import wgpu

    if dxc is not None:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        set_instance_extras(dx12_compiler='Dxc',dxc_path=str(dxc.resolve(strict=True)))
    order=fill_order(problem.neighbors)
    a,b=map(int,order[:2])
    side=list(problem.neighbors[a]).index(b)
    mate=int(problem.sides[a,side])
    rules=inventory(problem)
    prefixes=[[3*(x//3)+(side-x%3)%3,3*(y//3)+(mate-y%3)%3]
              for x,y in sorted(rules.forbidden-rules.loops)[:64]]
    assert prefixes
    factories=[('CUDA',DFS,{})]
    for adapter in wgpu.gpu.enumerate_adapters_sync():
        backend=adapter.info['backend_type']
        if adapter.info['adapter_type']=='DiscreteGPU' and backend in ('Vulkan','D3D12','Metal'):
            factories.append((backend,WebGPUDFS,dict(adapter=adapter)))
    results={}
    for label,factory,kwargs in factories:
        if apis and label not in apis:continue
        runs=[]
        for enabled in (False,True):
            control=nullcontext() if enabled else patch('dfs_gpu._physical_gold_table',return_value=np.zeros(1,np.uint32))
            with control:
                gpu=factory(problem.masks,problem.neighbors,problem.sides,n=len(prefixes),order=order,
                            prefixes=prefixes,gold_data=problem.tiles,seed=91,**kwargs)
            try:
                gpu.step(16)
                gpu.verify()
                counters=gpu.counters.get().sum(axis=1)
                runs.append(dict(exhausted=int((gpu.states.get()==2).sum()),tries=int(counters[0]),
                                 placements=int(counters[1]),backtracks=int(counters[2])))
            finally:
                if hasattr(gpu,'close'):gpu.close()
        assert runs[1]['exhausted']==len(prefixes) and runs[1]['tries']==0 and runs[0]['tries']>0,runs
        results[label]=runs
    return dict(prefixes=len(prefixes),baseline_then_trial=results)


def seam_reuse(problem):
    from r31_defect_bands import solve
    from r27_incumbent import BoardModel
    import numpy as np
    board=read_json(Path(__file__).parent/'docs'/'R29_INCUMBENT.json')['codes']
    board_model=BoardModel(problem)
    matched=board_model.matches(np.asarray(board))
    cells=sorted({cell for i,ok in enumerate(matched) if not ok for cell in problem.edges[i][::2]})[:8]
    rows=[[v] for v in board]
    for cell in cells:
        rows[cell]=sorted(3*(board[k]//3)+r for k in cells for r in range(3))
    summary=[]
    keys=('status','calls','structural_builds','structural_replays','counter_builds','counter_replays',
          'shared_core_checks','shared_core_skips','decisions','conflicts')
    with tempfile.TemporaryDirectory() as temp:
        for index,(target,width) in enumerate([(208,64),(208,63),(209,62),(207,64)]):
            hot=solve(problem,Path(temp)/'shared.sqlite',domains=rows,target=target,width=width)
            cold=solve(problem,Path(temp)/('cold%d.sqlite'%index),domains=rows,target=target,width=width)
            assert hot['status']==cold['status']
            if hot['witness']:
                assert problem.validate(hot['witness']['codes'])['matched_edges']>=target
            summary.append(dict(target=target,width=width,shared={k:hot.get(k,0) for k in keys},
                                cold={k:cold.get(k,0) for k in keys}))
    assert summary[1]['shared']['structural_replays']==1 and summary[1]['shared']['counter_replays']==1
    return dict(original_cells=cells,comparisons=summary)


def geographic_reuse(problem):
    from r32_gold_loss import geographic,reference_groups
    from r30_gold import search
    board=read_json(Path(__file__).parent/'docs'/'R29_INCUMBENT.json')['codes']
    ref,cert,groups=reference_groups(problem,board)
    fixed=[1<<v for v in board]
    wide=list(fixed)
    for i in range(4):
        wide[i]|=1<<(3*(board[i]//3)+(board[i]+1)%3)
    def ask(path,domains,pins=None):
        return geographic(problem,ref,groups,cert.rows,[],0,len(ref),None,20,path,domains,pins=pins)
    with tempfile.TemporaryDirectory() as temp,patch('r30_gold.search',wraps=search) as native:
        shared=Path(temp)/'child.sqlite'
        parent=ask(shared,wide)
        assert parent['status']=='INFEASIBLE',parent
        saved=[]
        for i in range(4):
            child=wide.copy()
            child[i]=fixed[i]
            count=native.call_count
            hot=ask(shared,child)
            assert native.call_count==count
            cold=ask(Path(temp)/('cold%d.sqlite'%i),child)
            assert native.call_count==count+1 and cold['status']==hot['status']=='INFEASIBLE'
            saved.append(cold.get('cp_calls',0))
        count=native.call_count
        changed=ask(shared,fixed,pins={1:1})
        assert native.call_count==count+1 and changed['status']=='INFEASIBLE'
        return dict(descendants=4,geographic_models_avoided=4,cp_calls_avoided=sum(saved),distinct_pins_recomputed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('device','trajectory','classic','seams','geographic','prices'))
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--sampler-backend',choices=['cpu','cuda','webgpu'],default='cpu')
    parser.add_argument('--api',action='append',choices=['CUDA','Vulkan','D3D12','Metal'])
    parser.add_argument('--dxc',type=Path,help='Optional installed Windows SDK dxcompiler.dll for DX12 comparison.')
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    if args.experiment=='trajectory':
        print(json.dumps(trajectory(problem,args.sampler_backend)))
        raise SystemExit
    if args.experiment=='classic':
        print(json.dumps(classic(problem,args.api,args.dxc)))
        raise SystemExit
    print(json.dumps({'device':device,'trajectory':trajectory,'classic':classic,'seams':seam_reuse,'geographic':geographic_reuse,'prices':price_storage}[args.experiment](problem)))
