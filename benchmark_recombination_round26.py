"""Completed-neighborhood F16 work comparisons; CPU only, no timing winner."""
import argparse,json
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions


def finite():
    from unittest.mock import patch
    from test_gpu_recombination_round26 import interacting_fixture,independent_candidates,finish_model
    from test_constraint_dfs import tetrahedron_tiles,board
    from search_problem import SearchProblem
    from sampling_crossover_round25 import CycleCrossover
    from sampling_crossover_round26 import OptimizingCrossover
    from gpu_sampling import gold_score
    problem,parents=interacting_fixture();archive=[{'codes':row} for row in parents]
    old=CycleCrossover(problem,gold_aware=True);old_result=old.propose(archive)
    improved,_,scorings=finish_model(problem,parents)
    expected=independent_candidates(problem,parents)
    maximum=max(gold_score(problem,list(row))['gold_objective'] for row in expected)
    assert old_result['solution'] is None and improved.best['gold'] and improved.best['metrics']['gold_objective']==maximum
    interacting={'neighborhood_boards':len(expected),'old_single_cycle_stats':old.stats,'old_found_gold':False,
      'exact_interacting_stats':improved.stats,'independent_scores_including_parent_seed':scorings,'found_gold':True}
    resumed,_,resumed_scorings=finish_model(problem,parents,lease=1,resume=True)
    assert resumed.best==improved.best and resumed.stats==improved.stats
    reuse={'cold_model_stats':improved.stats,'resumed_model_stats':resumed.stats,'cold_independent_scores':scorings,'resumed_independent_scores':resumed_scorings,
      'limitation':'Standalone model stats exclude repeated static construction during restore; the controller records rebuilds/component visits separately.'}
    problem=SearchProblem(*tetrahedron_tiles());parents=((0,4,7,10),(1,3,7,10),(1,4,6,9))
    neighborhoods={}
    for name,selected,rotations in (('pair',parents[:2],False),('three_parent',parents,False),('all_rotations',parents[:2],True)):
        model,_,count=finish_model(problem,selected,rotations);expected=independent_candidates(problem,selected,rotations)
        maximum=max((gold_score(problem,list(row))['gold_objective'],int(problem.validate(row)['valid'])) for row in expected)
        assert (model.best['metrics']['gold_objective'],int(model.best['gold']))==maximum
        neighborhoods[name]={'complete':True,'boards_in_defined_neighborhood':len(expected),'best_objective':maximum,'stats':model.stats,'independent_scores_including_parent_seed':count}
    n=5;tiles=[{'id':i,'segments':[[[0,2],[1,10]],[[0,10],[1,2]]]} for i in range(n)]
    problem=SearchProblem(tiles,board(n,[(i,1,(i+1)%n,0) for i in range(n)]));parents=(tuple(3*i for i in range(n)),(3,0,9,6,12))
    with patch('sampling_crossover_round26.loop_cut',return_value=None):baseline,_,base_scores=finish_model(problem,parents)
    cuts,_,cut_scores=finish_model(problem,parents)
    assert cuts.best['metrics']==baseline.best['metrics'] and not cuts.best['gold'] and cuts.stats['recombination_nodes']<baseline.stats['recombination_nodes']
    cut_comparison={'baseline_stats':baseline.stats,'cut_stats':cuts.stats,'baseline_scores':base_scores,'cut_scores':cut_scores,'same_completed_best':True}
    problem=SearchProblem(*tetrahedron_tiles());archive=[{'codes':row} for row in ((0,4,7,10),(1,3,7,10),(1,4,6,9))]
    controller=OptimizingCrossover(problem,gold_aware=True,max_nodes=16,max_parents=3);restores=0
    for iteration in range(300):
        controller.propose(archive)
        if all(record['stage']==4 for record in controller.pairs.values()):break
        if iteration%3==2:controller=OptimizingCrossover(problem,gold_aware=True,max_nodes=16,max_parents=3,resume=json.loads(json.dumps(controller.snapshot())));restores+=1
    assert all(record['stage']==4 for record in controller.pairs.values())
    return {'interacting_cycles':interacting,'completed_progress':reuse,'expanded_neighborhoods':neighborhoods,'conditional_gold_cuts':cut_comparison,
      'adaptive_controller':{'complete_configured_schedule':True,'restores':restores,'stats':controller.stats},
      'limitations':['All neighborhood counts describe the defined parental model only, not original puzzle coverage.','Stronger neighborhoods can require more work while finding a better candidate.','Gold cuts are exact safe bounds on the sampling objective; only independent full validation claims Gold.','Cache eviction and interrupted unfinished factor calculations can repeat work; counters disclose actual completed and retried operations.','No GPU workload, original Gold solution or elapsed-time superiority is claimed.']}


def main(argv=None):
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    output=Path(args.output).resolve();root=Path(__file__).resolve().parent
    if output.suffix.lower()!='.json' or output.exists() or not output.parent.is_dir():raise ValueError('A fresh JSON output in an existing directory is required.')
    before=source_snapshot(root);runtime_before=runtime_snapshot();repo_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo_before.get('available') or repo_before.get('dirty') or not repo_before.get('commit')):raise RuntimeError('Frozen evidence requires a readable clean Git commit.')
    result=finite()
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime_before!=runtime_after or repo_before!=repo_after:raise RuntimeError('Source/runtime/Git identity changed; discard recombination evidence.')
    report={'schema':'diamond-recombination-round26-evidence-v1','status':'passed','evidence_stage':args.evidence_stage,'family':'F16','result':result,
      'provenance':{'sources_before':before,'sources_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repo_before,'repository_after':repo_after}}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','family':'F16','evidence_stage':args.evidence_stage,'source_runtime_stable':True}));return report

if __name__=='__main__':main()
