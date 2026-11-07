"""Completed local repair work comparisons; native CP search is never serialized."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from benchmark_provenance import source_snapshot,repository_snapshot,require_assertions
from benchmark_repair_round25 import _runtime_snapshot,native_work


def ordinary_finite():
    from unittest.mock import patch
    from ortools.sat.python import cp_model as cp
    from constraint_lns import _solve_patch,_Control,score
    from constraint_repair_service_round26 import RetainedRepairService
    from search_problem import SearchProblem
    from search_fixtures import planted_gold
    from test_constraint_dfs import tetrahedron_tiles
    from test_retained_repair_round26 import brute
    rows=[]
    for label in ('tetrahedron','controlled160'):
        if label=='tetrahedron':problem=SearchProblem(*tetrahedron_tiles());initial=[0,3,6,9]
        else:
            tiles,board,initial=planted_gold(2617,size=4,extra_edges=12)
            problem=SearchProblem(tiles,board,fixed={cell:initial[cell] for cell in range(4,160)})
        alternate=initial.copy();alternate[0],alternate[1]=initial[1],initial[0]
        contexts=[(current.copy(),cells) for current in (initial,alternate) for cells in ([0,1],[1,2],[0,2,3])]
        expected=[max((value for value,_ in brute(problem,current,cells)),default=None) for current,cells in contexts]
        variants={}
        for mode in ('existing_local_models','retained_warm','retained_restore_after3'):
            models=[];old_init=cp.CpModel.__init__;owner=None;observed=[];states=[];details=[]
            def model_init(model,*args,**kwargs):
                old_init(model,*args,**kwargs);models.append(model)
            try:
                with native_work() as native,patch.object(cp.CpModel,'__init__',model_init):
                    # Preserve native_work model accounting while retaining model
                    # references only for this finite construction-term audit.
                    if mode!='existing_local_models':owner=RetainedRepairService(problem)
                    for index,(current,cells) in enumerate(contexts):
                        if mode=='retained_restore_after3' and index==3:
                            saved=json.loads(json.dumps(owner.snapshot()));states.append(len(json.dumps(saved).encode()));owner.close();owner=RetainedRepairService(problem,resume=saved)
                        diagnostic={};kwargs={} if owner is None else {'retained_service':owner}
                        candidate,status=_solve_patch(problem,current,cells,cp_model=cp,control=_Control(10,None),seconds=5,seed=2617+index,target='edges',diagnostics=diagnostic,**kwargs)
                        result=None if candidate is None else score(problem,candidate)['matched_edges']
                        if status not in ('OPTIMAL','INFEASIBLE') or result!=expected[index]:raise RuntimeError('Repair comparison did not complete the independently checked query.')
                        if candidate is not None:
                            if any(candidate[c]!=current[c] for c in range(problem.n) if c not in cells) or score(problem,candidate)['domain_violations']:raise RuntimeError('Repair result escaped local scope.')
                        observed.append({'status':status,'matched_edges':result});details.append(diagnostic)
                    if owner is not None:
                        calls=owner.stats['retained_native_calls'];before=owner.snapshot();candidate,status=_solve_patch(problem,*contexts[0],cp_model=cp,control=_Control(5,None),seconds=2,seed=9,target='edges',retained_service=owner)
                        if owner.stats['retained_native_calls']!=calls or (None if candidate is None else score(problem,candidate)['matched_edges'])!=expected[0]:raise RuntimeError('Completed retained record was not reused.')
                # model_init is deliberately outside native_work's construction
                # hook; count exactly the captured constructor calls here.
                native['model_constructions']=len(models)
                variants[mode]={'queries':observed,'native':dict(native),'constructed_variables':sum(len(m.proto.variables) for m in models),
                  'constructed_constraints':sum(len(m.proto.constraints) for m in models),'query_diagnostics':details,
                  'retained_stats':dict(owner.stats) if owner is not None else {},'checkpoint_bytes':states}
            finally:
                if owner is not None:owner.close()
        assert variants['existing_local_models']['queries']==variants['retained_warm']['queries']==variants['retained_restore_after3']['queries']
        assert variants['retained_warm']['native']['model_constructions']<variants['existing_local_models']['native']['model_constructions']
        assert variants['retained_restore_after3']['native']['model_constructions']<variants['existing_local_models']['native']['model_constructions']
        rows.append({'case':label,'fixture':{'cells':problem.n,'free_cells':4,'provenance':'synthetic planted Gold on actual160-cell geometry, all but4fixed' if label=='controlled160' else 'independent tiny tetrahedron'},
          'unique_contexts':len(contexts),'same_completed_optima':True,'variants':variants,
          'limitation':'Existing controller caching already skips repeated exact contexts. The extra cache check is lifecycle evidence; the comparison benefit is reused construction across six different contexts. All model terms, branches and calls remain separately reported.'})
    return rows


def main(argv=None):
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent;output=Path(args.output).resolve()
    if output.suffix.lower()!='.json' or output.exists() or not output.parent.is_dir():raise ValueError('A fresh JSON output in an existing directory is required.')
    before=source_snapshot(root);runtime_before=_runtime_snapshot();repo_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo_before.get('available') or repo_before.get('dirty') or not repo_before.get('commit')):raise RuntimeError('Frozen evidence requires a clean readable commit.')
    # Import/load every fixture only after the provenance baseline.
    ordinary=ordinary_finite()
    from benchmark_gold_repair_round26 import finite as gold_finite
    gold=gold_finite()
    after=source_snapshot(root);runtime_after=_runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime_before!=runtime_after or repo_before!=repo_after:raise RuntimeError('Source/runtime/Git changed; discard retained-repair evidence.')
    result={'schema':'diamond-retained-repair-round26-evidence-v1','status':'passed','family':'F17','evidence_stage':args.evidence_stage,
      'ordinary':ordinary,'gold_masters':gold,'limitations':['Local completed query/model reuse is not global puzzle coverage.','No native CP-SAT active search or learned-state serialization is claimed.','Construction savings may accompany more model terms or search branches; no elapsed-time winner or original Gold result is inferred.'],
      'provenance':{'sources_before':before,'sources_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repo_before,'repository_after':repo_after}}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','family':'F17','ordinary_cases':len(ordinary),'evidence_stage':args.evidence_stage}));return result

if __name__=='__main__':main()
