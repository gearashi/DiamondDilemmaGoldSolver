"""Finite CPU-only checks for bounded neighborhood repair and cancellation."""
import threading
import time
import unittest
from unittest.mock import PropertyMock, patch
from types import SimpleNamespace

import constraint_lns as lns
from search_problem import SearchProblem
from test_constraint_dfs import board, double_face, simple_tiles, multiple_loops, tetrahedron_tiles


class RepairTests(unittest.TestCase):
    def test_gold_certificate_and_multiloop_are_distinct(self):
        problem = SearchProblem(simple_tiles(), double_face())
        result = lns.repair(problem,[0,3],seconds=1)
        self.assertEqual(result['status'],'solved')
        self.assertTrue(result['validation']['valid'])
        loops = SearchProblem(multiple_loops(),double_face())
        edge = lns.repair(loops,[0,3],seconds=1,target='edges')
        self.assertEqual(edge['status'],'edge_perfect')
        gold = lns.repair(loops,[0,3],seconds=2,patch_sizes=(2,),max_attempts=3)
        self.assertFalse(gold['complete'])
        self.assertNotEqual(gold['status'],'infeasible')
        self.assertGreaterEqual(gold['stats']['equal_alternatives'],1)
        self.assertGreater(gold['stats']['loop_cuts'],0)
        self.assertFalse(gold['validation']['valid'])

    def test_native_patch_preserves_outside_inventory_fixed_cells_and_scores(self):
        tiles, geometry = tetrahedron_tiles()
        problem = SearchProblem(tiles,geometry,fixed={0:0})
        current = [0,4,6,9]
        candidate, native = lns._solve_patch(problem,current,[1],cp_model=lns._load_cp_model(),
            control=lns._Control(2,None),seconds=1,seed=17,target='gold')
        self.assertIn(native,('OPTIMAL','FEASIBLE'))
        self.assertEqual(candidate,[0,3,6,9])
        self.assertTrue(problem.validate(candidate)['valid'])
        self.assertEqual(candidate[0],0)
        self.assertEqual({candidate[cell]//3 for cell in [1]}, {current[cell]//3 for cell in [1]})

    def test_bounded_local_failure_never_claims_global_infeasible(self):
        # Tile 0 is fixed in a legal orientation, while tile 1 has the wrong
        # asymmetric endpoint pattern for any globally matched arrangement.
        tiles = simple_tiles(); tiles[1]['segments'] = [[[0,2],[1,4]]]
        problem = SearchProblem(tiles,double_face(),fixed={0:0})
        result = lns.repair(problem,[0,3],seconds=2,patch_sizes=(1,),max_attempts=2)
        self.assertFalse(result['complete'])
        self.assertEqual(result['scope'],'local_repair_only')
        self.assertNotEqual(result['status'],'infeasible')
        self.assertGreaterEqual(result['best_score'], lns.score(problem,[0,3])['matched_edges'])

    def test_native_infeasible_patch_is_only_a_local_result(self):
        tiles = [{'id':0,'segments':[[[0,2],[0,10]]]},
                 {'id':1,'segments':[[[0,3],[0,9]]]}]
        problem = SearchProblem(tiles,board(2,[(0,0,1,0)]),fixed={0:0})
        result = lns.repair(problem,[0,3],seconds=2,patch_sizes=(1,),max_attempts=1)
        self.assertEqual(result['stats']['native_statuses'],{'INFEASIBLE':1})
        self.assertEqual(result['status'],'inconclusive')
        self.assertFalse(result['complete'])

    def test_progress_stats_are_independent_snapshots(self):
        problem = SearchProblem(multiple_loops(),double_face())
        events = []
        lns.repair(problem,[0,3],seconds=2,patch_sizes=(2,),max_attempts=2,progress=events.append)
        self.assertEqual(events[0]['stats']['native_statuses'],{})
        self.assertEqual(sum(events[-1]['stats']['native_statuses'].values()),2)

    def test_patch_expands_through_matched_neighbors_and_includes_bad_domains(self):
        import random
        tiles, geometry = tetrahedron_tiles()
        problem = SearchProblem(tiles,geometry)
        codes = [0,3,6,9]
        selected = lns._select_patch(problem,codes,lns.score(problem,codes),3,random.Random(3),'gold')
        self.assertEqual(len(selected),3)
        boundary_tiles = [{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(2)]
        boundary = SearchProblem(boundary_tiles,board(2,[(0,0,1,0)]))
        result = lns.repair(boundary,[1,4],seconds=2,patch_sizes=(1,),max_attempts=1)
        self.assertEqual(result['domain_violations'],0)
        self.assertEqual(result['status'],'solved')

    def test_zero_budget_and_stop_never_start_native_search(self):
        problem = SearchProblem(simple_tiles(),double_face())
        with patch.object(lns,'_load_cp_model',side_effect=AssertionError('No solver should start')):
            self.assertEqual(lns.repair(problem,[1,3],seconds=0)['status'],'timeout')
            self.assertEqual(lns.repair(problem,[1,3],seconds=2,stop=lambda:True)['status'],'stopped')
        event = threading.Event()
        progress = []
        def notify(value):
            progress.append(value)
            if value['event']=='repair_attempt': event.set()
        with patch.object(lns,'_solve_patch',wraps=lns._solve_patch):
            result = lns.repair(problem,[1,3],seconds=2,stop=event,progress=notify)
        self.assertEqual(result['status'],'stopped')
        self.assertTrue(all(value['phase']=='repair' for value in progress))

    def test_missing_ortools_is_visible_and_nonfatal(self):
        problem = SearchProblem(simple_tiles(),double_face())
        events = []
        with patch.object(lns,'_load_cp_model',side_effect=ImportError('optional dependency')):
            result = lns.repair(problem,[1,3],seconds=2,progress=events.append)
        self.assertEqual(result['status'],'skipped')
        self.assertIn('OR-Tools',result['reason'])
        self.assertFalse(result['complete'])
        self.assertEqual(events[-1]['event'],'repair_skipped')

    def test_native_stop_watcher_calls_stop_search_and_releases_thread(self):
        requested, interrupted = threading.Event(), threading.Event()
        control = lns._Control(2,requested)
        fake = SimpleNamespace(stop_search=interrupted.set)
        with lns._watch(control,time.monotonic()+2,fake):
            requested.set()
            self.assertTrue(interrupted.wait(1))
        self.assertEqual(control.reason,'stopped')
        self.assertFalse(any(thread.name=='diamond-repair-stop' and thread.is_alive() for thread in threading.enumerate()))

    def test_attempt_time_budget_includes_model_build_before_native_search(self):
        problem = SearchProblem(simple_tiles(),double_face())
        cp_model = lns._load_cp_model()
        model = cp_model.CpModel()
        maximize = model.maximize
        now = [100.0]
        def build_objective(objective):
            maximize(objective)
            # Real model construction consumes the local budget while the
            # overall run remains live. This is independent of OS clock ticks.
            now[0] += 1.0
        with patch.object(lns,'time',SimpleNamespace(monotonic=lambda:now[0])), \
             patch.object(cp_model,'CpModel',return_value=model) as build_model, \
             patch.object(model,'maximize',side_effect=build_objective) as build_step, \
             patch.object(cp_model.CpSolver,'solve',side_effect=AssertionError('Expired model must not run')) as native:
            control = lns._Control(10,None)
            result, status = lns._solve_patch(problem,[1,3],[0,1],cp_model=cp_model,
                control=control,seconds=.5,seed=0,target='gold')
            self.assertFalse(control.cancelled())
        build_model.assert_called_once_with()
        build_step.assert_called_once()
        native.assert_not_called()
        self.assertIsNone(result)
        self.assertEqual(status,'MODEL_BUDGET')

    def test_native_candidate_with_incorrect_objective_is_rejected(self):
        tiles, geometry = tetrahedron_tiles()
        problem = SearchProblem(tiles,geometry,fixed={0:0})
        cp_model = lns._load_cp_model()
        # A real native solve supplies the candidate. Its deliberately corrupt
        # score must fail the independent CPU check before any board is accepted.
        with patch.object(cp_model.CpSolver,'objective_value',new_callable=PropertyMock,return_value=-1000) as objective:
            with self.assertRaisesRegex(RuntimeError,'independent CPU seam scoring'):
                lns._solve_patch(problem,[0,4,6,9],[1],cp_model=cp_model,
                    control=lns._Control(5,None),seconds=2,seed=17,target='gold')
        objective.assert_called_once_with()

    def test_diverse_archive_deduplicates_and_preserves_best(self):
        best={'codes':[0,3,6,9],'objective':20}
        near={'codes':[1,3,6,9],'objective':20}
        far={'codes':[9,6,3,0],'objective':19}
        archive=lns.diverse_archive([best,best,near,far],capacity=2)
        self.assertEqual(archive,[best,far])
        self.assertIsNot(archive[0]['codes'],best['codes'])

    def test_loop_aware_repair_can_join_identical_signatures_into_gold(self):
        tiles=[{'id':i,'segments':[[[0,2],[1,2]],[[0,10],[1,10]],[[2,2],[2,10]]]} for i in range(2)]
        problem=SearchProblem(tiles,double_face())
        self.assertEqual(problem.validate([0,3])['line_components'],3)
        result=lns.repair(problem,[0,3],seconds=2,patch_sizes=(2,),max_attempts=1)
        self.assertEqual(result['status'],'solved')
        self.assertTrue(problem.validate(result['best_board'])['valid'])
        self.assertGreaterEqual(result['stats']['loop_cuts'],1)

    def test_conclusive_attempts_are_skipped_but_timeouts_are_retryable(self):
        problem=SearchProblem(simple_tiles(),double_face())
        for status,expected in [('INFEASIBLE',1),('UNKNOWN',3),('MODEL_BUDGET',3),('FEASIBLE',3)]:
            with self.subTest(status=status), patch.object(lns,'_load_cp_model',return_value=object()), patch.object(lns,'_select_patch',return_value=[0,1]), patch.object(lns,'_solve_patch',return_value=(None,status)) as search:
                result=lns.repair(problem,[1,3],seconds=2,max_attempts=3,adaptive=False)
            self.assertEqual(search.call_count,expected)
            self.assertEqual(result['stats']['attempt_cache_hits'],3-expected)
            self.assertFalse(result['complete'])
            self.assertNotEqual(result['status'],'infeasible')

    def test_adaptive_schedule_expands_proven_stalls_and_backs_off_timeouts(self):
        tiles,geometry=tetrahedron_tiles();problem=SearchProblem(tiles,geometry)
        sizes=[]
        def select(problem,current,metrics,size,rng,target):
            sizes.append(size);return list(range(size))
        with patch.object(lns,'_load_cp_model',return_value=object()), patch.object(lns,'_select_patch',side_effect=select), patch.object(lns,'_solve_patch',side_effect=[(None,'INFEASIBLE'),(None,'UNKNOWN'),(None,'INFEASIBLE')]):
            result=lns.repair(problem,[0,4,6,9],seconds=2,patch_sizes=(1,2,3),max_attempts=3,cache_attempts=False)
        self.assertEqual(sizes,[1,2,1])
        self.assertEqual(result['stats']['size_changes'],2)

    def test_invalid_seed_inventory_fixed_placements_and_limits(self):
        problem = SearchProblem(simple_tiles(),double_face(),fixed={0:0})
        for codes,options in (([0,0],{}),([1,3],{}),([0,3],{'seconds':float('inf')}),
                             ([0,3],{'patch_sizes':()}),([0,3],{'per_attempt_seconds':0}),
                             ([0,3],{'max_attempts':0}),([0,3],{'stop':42})):
            with self.subTest(codes=codes,options=options), self.assertRaises(ValueError):
                lns.repair(problem,codes,**options)


if __name__=='__main__': unittest.main()
