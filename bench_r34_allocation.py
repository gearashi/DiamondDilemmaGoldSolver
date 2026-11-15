"""Run matched exact residuals and fresh discovery trials on actual backends."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from constraint_coordinator_round26 import AllocationPolicy
from constraint_work import WorkScope,frontier_snapshot
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r31_fresh import Engine
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_gpu_domain_round25 import precise_reason_fixture


def mathematical_answers(problem):
    model=BoardModel(problem)
    out=set()
    for row in itertools.product(*problem.domains):
        if len(set(c//3 for c in row))!=model.n:
            continue
        if model.matches(np.array(row)).all() and problem.validate(row)['valid']:
            out.add(row)
    return out


def residuals(original):
    incumbent=read_json('docs/R29_INCUMBENT.json')['codes']
    for i in range(4):
        p=precise_reason_fixture()
        p.domains[i]=p.domains[i][::2]
        yield 'prism-'+str(i),p
    for i in range(4):
        p=SearchProblem(original.tiles,original.board)
        p.domains=[[c] for c in incumbent]
        for k in (i,39+i,80+i):
            p.domains[k]=[3*(incumbent[k]//3)+r for r in range(3)]
        yield 'original-'+str(i),p
    for n in (1,4):
        for i in range(4):
            tiles,geometry,board=planted_gold(3430+i,size=n,extra_edges=3)
            p=SearchProblem(tiles,geometry)
            p.domains=[[c] for c in board]
            for k in np.linspace(0,p.n-1,3+i%2).astype(int):
                p.domains[k]=[3*(board[k]//3)+r for r in range(3)]
            yield f'planted-{n}-{i}',p


def solve(backend,p,seconds,*,resume=None):
    opts={}
    args=dict(seconds=seconds,seed=34,target='gold',resume=resume,options=opts)
    if backend=='cpu':
        from constraint_dfs import solve as call
    elif backend=='sat':
        from constraint_sat_session import solve as call
    else:
        from gpu_domain_runner import solve as call
        opts.update(gpu_joint_support=True,gpu_event_propagation=True,gpu_precise_reasons=True,
                    gpu_watched_clauses=True,gpu_global_matching=True,gpu_support_reuse=True)
        args.update(replicas=128,backend='cuda')
    t=time.perf_counter()
    result=call(p,**args)
    return result,time.perf_counter()-t


def schedule(records):
    summaries={}
    arms=('cpu','sat','gpu')
    # Shadow evaluation consumes only observations from the backend a policy
    # actually selected on previous residuals. Held-out cases supply outcomes.
    for policy in ('released','stratified','cpu','sat','gpu'):
        stats={'proof':[],'discovery':[]}
        old=AllocationPolicy()
        strat={}
        for sample in records:
            obj=sample['goal']
            box=tuple(int(v,16) for v in sample['box'])
            # Scope size and density are available before dispatch. No result
            # labels are available at this choice.
            key=(obj,len(box)>80,sum(v.bit_count()>1 for v in box)>10)
            if policy=='released':
                selected=old.choose(box,arms,'cpu')
            elif policy=='stratified':
                data=strat.setdefault(key,{a:[] for a in arms})
                unseen=[a for a in arms if not data[a]]
                if unseen:
                    selected=unseen[0]
                else:
                    selected=max(arms,key=lambda a:sum(x[0] for x in data[a])/sum(x[1] for x in data[a]))
            else:
                selected=policy
            row=sample['warm'][selected]
            success=row['certified']
            old.observe(box,selected,row['seconds'],bool(success))
            if policy=='stratified':
                strat[key][selected].append((success,row['seconds']))
            stats[obj].append((selected,success,row['seconds']))
        summaries[policy]={k:dict(arms=[v[0] for v in rows],certified=sum(v[1] for v in rows),
                                   seconds=sum(v[2] for v in rows)) for k,rows in stats.items()}
    return summaries


def exact(problem,path):
    records=[]
    for index,(name,p) in enumerate(residuals(problem)):
        answers=mathematical_answers(p)
        arms=['cpu','gpu','sat']
        arms=arms[index%3:]+arms[:index%3]
        record=dict(name=name,goal='discovery' if name.startswith('planted-') else 'proof',answers=len(answers),
                    box=[hex(v) for v in WorkScope.for_problem(p,'gold').root],cold={},warm={})
        for phase,order in [('cold',arms),('warm',arms[::-1])]:
            for backend in order:
                r,t=solve(backend,p,5)
                status=r['status']
                if status=='solved':
                    assert tuple(r['codes']) in answers and p.validate(r['codes'])['valid']
                if status=='infeasible':
                    assert not answers and r['complete']
                record[phase][backend]=dict(status=status,certified=int(status in ('solved','infeasible')),
                                           seconds=t,decisions=r.get('nodes'),checkpoint_bytes=len(
                                               json.dumps(r['checkpoint']).encode()) if r.get('checkpoint') else 0)
        records.append(record)
        atomic_json(path,dict(records=records,policy=schedule(records)))


def discovery(problem,path):
    records=[]
    for seed in (340,341,342,343):
        for backend in (('cpu','cuda') if seed%2==0 else ('cuda','cpu')):
            args=dict(backend=backend,replicas=512,dispatches=16,rounds=24)
            engine=Engine(problem,bonds=True,joint=True,cycle_budget=3)
            cache=path.with_name('allocation-sampling.sqlite') if backend=='cuda' else None
            for phase in ('cold','warm'):
                t=time.perf_counter()
                row=engine.run(seed,sampler_cache=cache,**args)
                elapsed=time.perf_counter()-t
                model=engine.model
                check=model.problem.validate(row['codes'])
                assert check['matched_edges']==row['score'] and check['valid']==row['gold']
                records.append(dict(backend=backend,seed=seed,phase=phase,score=row['score'],gold=row['gold'],
                                    seconds=elapsed,work=row['work']))
                atomic_json(path,records)


def transfer(problem,path):
    from constraint_symbolic_transfer import engine_remaining
    record={}
    # Use a genuine unfinished unrestricted residual. Timeouts certify no negative.
    for source in ('cpu','gpu'):
        first,elapsed=solve(source,problem,.35)
        row={'cold_seconds':elapsed,'initial_status':first['status']}
        if first['status'] not in ('timeout','stopped') or not first.get('checkpoint'):
            record[source]=row
            continue
        cp=first['checkpoint']
        native,t=solve(source,problem,.35,resume=cp)
        row.update(native_seconds=t,native_status=native['status'],checkpoint_bytes=len(json.dumps(cp).encode()))
        t=time.perf_counter()
        scope,boxes=engine_remaining(problem,source,cp,target='gold')
        portable=frontier_snapshot(scope,boxes)
        row.update(boxes=len(boxes),export_seconds=time.perf_counter()-t,
                   portable_bytes=len(json.dumps(portable).encode()))
        dest='cpu' if source=='gpu' else 'gpu'
        if dest=='gpu':
            from gpu_domain_runner import solve as gpu_solve
            t=time.perf_counter()
            other=gpu_solve(problem,seconds=.35,seed=34,target='gold',replicas=128,backend='cuda',
                            work_frontier=portable,options=dict(gpu_joint_support=True,gpu_event_propagation=True,
                            gpu_precise_reasons=True,gpu_watched_clauses=True,gpu_global_matching=True,gpu_support_reuse=True))
            elapsed=time.perf_counter()-t
        else:
            from constraint_dfs import _solve_core
            if not boxes:
                raise ArithmeticError("Expected a live portable frontier")
            # CPU workers claim one portable box at a time. The other boxes
            # remain owned by the caller and are not marked complete.
            t=time.perf_counter()
            other=_solve_core(problem,seconds=.35,seed=34,target='gold',_initial_domains=boxes[0],options={})
            elapsed=time.perf_counter()-t
            row['unassigned_portable_boxes']=len(boxes)-1
        row.update(destination=dest,destination_seconds=elapsed,destination_status=other['status'])
        record[source]=row
        atomic_json(path,record)


def migrated_boxes(problem,path):
    from constraint_dfs import _solve_core
    from constraint_symbolic_transfer import engine_remaining
    from gpu_domain_runner import solve as gpu
    source,_=solve('gpu',problem,.35)
    if source['complete']:
        raise ValueError("Expected an unfinished unrestricted GPU root.")
    _,boxes=engine_remaining(problem,'gpu',source['checkpoint'])
    rows=[]
    for index in np.linspace(0,len(boxes)-1,min(12,len(boxes))).astype(int):
        box=boxes[index]
        scope=WorkScope.for_problem(problem,'gold',box)
        control=frontier_snapshot(scope,[box])
        row={'index':int(index),'assigned_cells':sum(v.bit_count()==1 for v in box)}
        for name in (('cpu','gpu') if len(rows)%2==0 else ('gpu','cpu')):
            t=time.perf_counter()
            if name=='cpu':
                r=_solve_core(problem,seconds=.7,seed=34,target='gold',_initial_domains=box,options={})
            else:
                r=gpu(problem,seconds=.7,seed=34,target='gold',replicas=1,backend='cuda',
                      work_frontier=control,options=dict(gpu_joint_support=True,gpu_event_propagation=True,
                      gpu_precise_reasons=True,gpu_watched_clauses=True,gpu_global_matching=True,gpu_support_reuse=True))
            row[name]=dict(status=r['status'],seconds=time.perf_counter()-t,decisions=r['nodes'])
            if r['status']=='solved':
                assert problem.validate(r['codes'])['valid'] and all(box[i] & (1<<code) for i,code in enumerate(r['codes']))
        if all(row[a]['status'] in ('infeasible','solved') for a in ('cpu','gpu')):
            assert row['cpu']['status']==row['gpu']['status']
        rows.append(row)
        atomic_json(path,dict(source_boxes=len(boxes),rows=rows))


def hall_control(problem,path):
    from unittest.mock import patch
    from gpu_domain_runner import inventory_screen,split_cubes
    root=WorkScope.for_problem(problem,'gold').root
    boxes=split_cubes([root],128)
    live,closed=inventory_screen(boxes,problem.n)
    # Independent pigeonhole replay with explicit physical-tile sets.
    for box,cells in closed:
        tiles={j//3 for c in cells for j in range(box[c].bit_length()) if box[c]>>j&1}
        assert len(tiles)<len(cells)
    report=dict(total=len(boxes),host_survivors=len(live),host_proofs=len(closed))
    def run_once():
        r,t=solve('gpu',problem,.45)
        cp=r['checkpoint']
        return dict(status=r['status'],lanes=r['stats'].get('gpu_domain_lanes'),
                    state_bytes=r['stats'].get('gpu_domain_state_bytes'),seconds=t,
                    open_queue=len(cp['queue']),host_closed=cp.get('host_hall_closed',0),
                    completed=cp['completed_lanes'],checkpoint_bytes=len(json.dumps(cp).encode()))
    with patch('gpu_domain_runner.inventory_screen',side_effect=lambda cubes,n:(cubes,[])):
        report['before']=run_once()
    report['after']=run_once()
    atomic_json(path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('exact','discovery','transfer','boxes','screen'))
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-split',action='store_true',help='Reproduce the allocation matrix before the host Hall filter.')
    args=p.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    from contextlib import nullcontext
    from unittest.mock import patch
    old=patch('gpu_domain_runner.inventory_screen',side_effect=lambda cubes,n:(cubes,[]))
    with old if args.legacy_split else nullcontext():
        {'exact':exact,'discovery':discovery,'transfer':transfer,'boxes':migrated_boxes,
         'screen':hall_control}[args.mode](problem,args.output)


if __name__=='__main__':
    main()
