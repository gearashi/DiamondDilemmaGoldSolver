"""Independent finite neighborhood oracles and durable host lifecycle; no GPU."""
import itertools,json,random,unittest
from copy import deepcopy
from unittest.mock import patch
from sampling_crossover_round26 import OptimizingCrossover,Neighborhood,closure_components
from sampling_crossover_round25 import relative_cycles,_digest
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,board
from test_gpu_crossover_reference_round25 import complementary_fixture
from gpu_sampling import gold_score,score_board


def independent_candidates(problem,parents,rotate=False):
    # Independent graph connectivity for equal physical-ownership blocks.
    adjacency={cell:set() for cell in range(problem.n)};first=parents[0]
    for row in parents[1:]:
        for cell,code in enumerate(row):
            other=next(c for c,v in enumerate(first) if v//3==code//3)
            adjacency[cell].add(other);adjacency[other].add(cell)
    pending=set(adjacency);groups=[]
    while pending:
        root=min(pending);pending.remove(root);seen={root};todo=[root]
        for vertex in todo:
            for other in adjacency[vertex]-seen:seen.add(other);pending.discard(other);todo.append(other)
        groups.append(sorted(seen))
    answers=set()
    for chosen in itertools.product(range(len(parents)),repeat=len(groups)):
        codes=[None]*problem.n
        for cells,index in zip(groups,chosen):
            for cell in cells:codes[cell]=parents[index][cell]
        possibilities=[range(3*(code//3),3*(code//3)+3) if rotate else (code,) for code in codes]
        for candidate in itertools.product(*possibilities):
            if any(code not in problem.domains[c] for c,code in enumerate(candidate)):continue
            if any(candidate[c]!=v for c,v in problem.fixed.items()):continue
            if any(problem.neighbors[c,s]<0 and int(problem.masks[code,s]) for c,code in enumerate(candidate) for s in range(3)):continue
            answers.add(candidate)
    return answers


def finish_model(problem,parents,rotate=False,gold=True,lease=50,resume=False):
    model=Neighborhood(problem,tuple(sorted(parents)),rotate,gold)
    scores={};counter=[0]
    def score(codes):
        if codes not in scores:
            metrics=(gold_score if gold else score_board)(problem,list(codes));counter[0]+=1
            scores[codes]={'metrics':metrics,'gold':bool(problem.validate(codes)['valid'])}
        return scores[codes]
    model.seed({row:score(row) for row in parents})
    while not model.complete:
        model.advance(lease,score)
        if resume:model=Neighborhood(problem,tuple(sorted(parents)),rotate,gold,state=json.loads(json.dumps(model.snapshot())))
    return model,scores,counter[0]


def interacting_fixture():
    n=8
    tiles=[{'id':i,'segments':[[[0,i+1],[1,12-((i+1)%n+1)]]]} for i in range(n)]
    problem=SearchProblem(tiles,board(n,[(c,1,(c+1)%n,0) for c in range(n)]))
    left=[3*i for i in range(n)];right=left.copy()
    for a,b in ((0,1),(2,3)):left[a],left[b]=left[b],left[a]
    for a,b in ((4,5),(6,7)):right[a],right[b]=right[b],right[a]
    return problem,(tuple(left),tuple(right))


class CompleteRecombinationTests(unittest.TestCase):
    def test_interacting_cycles_find_gold_missing_from_every_single_cycle_move(self):
        problem,parents=interacting_fixture();self.assertTrue(problem.validate([3*i for i in range(8)])['valid'])
        cycles=relative_cycles(*parents);self.assertEqual(len(cycles),4)
        for base,donor in (parents,parents[::-1]):
            for cycle in cycles:
                child=list(base)
                for cell in cycle:child[cell]=donor[cell]
                self.assertFalse(problem.validate(child)['valid'])
        model,_,_=finish_model(problem,parents)
        self.assertTrue(model.best['gold']);self.assertIn(tuple(model.best['codes']),independent_candidates(problem,parents))
        self.assertGreater(model.stats['recombination_bound_prunes'],0)
    def test_all_bounded_two_three_parent_and_rotation_neighborhoods_match_oracle(self):
        problem=SearchProblem(*tetrahedron_tiles());rng=random.Random(2616)
        for count,rotate,gold in itertools.product((2,3),(False,True),(False,True)):
            for _ in range(3):
                parents=[]
                while len(parents)<count:
                    pieces=list(range(problem.n));rng.shuffle(pieces);row=tuple(3*t+rng.randrange(3) for t in pieces)
                    if row not in parents:parents.append(row)
                expected=independent_candidates(problem,parents,rotate);model,_,_=finish_model(problem,parents,rotate,gold)
                score=lambda row: ((gold_score if gold else score_board)(problem,list(row))['gold_objective' if gold else 'objective'],int(problem.validate(row)['valid']))
                maximum=max(map(score,expected)) if expected else None
                actual=None if model.best is None else (model.best['metrics']['gold_objective' if gold else 'objective'],int(model.best['gold']))
                self.assertEqual(actual,maximum)
                if model.best is not None:self.assertIn(tuple(model.best['codes']),expected)
    def test_rotation_only_and_three_parent_scope_are_both_real(self):
        problem=SearchProblem(*tetrahedron_tiles())
        parents=((0,4,7,10),(1,3,7,10),(1,4,6,9))
        for row in parents:self.assertFalse(problem.validate(row)['valid'])
        model,_,_=finish_model(problem,parents)
        self.assertTrue(model.best['gold']);self.assertEqual(len(closure_components(parents)),4)
        pair,_,_=finish_model(problem,parents[:2],rotate=True)
        self.assertTrue(pair.best['gold'])
        self.assertGreater(len(independent_candidates(problem,parents[:2],True)),len(independent_candidates(problem,parents[:2],False)))
    def test_gold_cuts_preserve_optimum_and_reduce_completed_work(self):
        n=5;tiles=[{'id':i,'segments':[[[0,2],[1,10]],[[0,10],[1,2]]]} for i in range(n)]
        problem=SearchProblem(tiles,board(n,[(i,1,(i+1)%n,0) for i in range(n)]))
        parents=(tuple(3*i for i in range(n)),(3,0,9,6,12))
        with patch('sampling_crossover_round26.loop_cut',return_value=None):baseline,_,_=finish_model(problem,parents)
        optimized,_,_=finish_model(problem,parents)
        self.assertEqual(optimized.best['metrics'],baseline.best['metrics']);self.assertFalse(optimized.best['gold'])
        self.assertGreater(optimized.stats['recombination_gold_cuts'],0)
        self.assertGreater(optimized.stats['recombination_cut_bounds'],0)
        self.assertLess(optimized.stats['recombination_nodes'],baseline.stats['recombination_nodes'])
    def test_sliced_restores_keep_completed_bound_factors_and_branch_work(self):
        problem,parents=interacting_fixture();cold,_,_=finish_model(problem,parents)
        resumed,_,_=finish_model(problem,parents,lease=1,resume=True)
        self.assertEqual(resumed.best,cold.best);self.assertEqual(resumed.stats,cold.stats)


class OptimizingControllerTests(unittest.TestCase):
    def test_controller_adapts_only_after_completion_and_explores_all_parent_pairs(self):
        problem=SearchProblem(*tetrahedron_tiles());parents=((0,4,7,10),(1,3,7,10),(1,4,6,9));archive=[{'codes':row} for row in parents]
        solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=1,max_parents=3)
        for _ in range(300):
            solver.propose(archive)
            if all(row['stage']==4 for row in solver.pairs.values()):break
        self.assertTrue(all(row['stage']==4 for row in solver.pairs.values()))
        self.assertGreater(solver.stats['recombination_rotation_models'],0);self.assertGreater(solver.stats['recombination_multiparent_models'],0)
        self.assertGreater(solver.stats['recombination_adaptive_expansions'],0);self.assertGreater(solver.stats['recombination_outcome_reuses'],0)
        nodes=solver.stats['recombination_nodes'];solver.propose(archive);self.assertEqual(solver.stats['recombination_nodes'],nodes)
    def test_public_resume_injection_and_duplicate_child_scoring(self):
        problem,archive,_=complementary_fixture();solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=1)
        solution=None
        for _ in range(60):
            result=solver.propose(archive)
            if result['solution'] is not None:
                solution=result['solution'];self.assertTrue(problem.validate(solution)['valid']);solver.record_injection()
                with self.assertRaises(RuntimeError):solver.record_injection()
            solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=1,resume=json.loads(json.dumps(solver.snapshot())))
            if all(row['stage']==2 for row in solver.pairs.values()):break
        self.assertIsNotNone(solution);self.assertGreater(solver.stats['recombination_parent_cache_hits'],0)
        self.assertGreater(solver.stats['recombination_models_reused'],0)
    def test_stop_boundaries_preserve_completed_work_and_pending_offer(self):
        problem,archive,_=complementary_fixture()
        for boundary in (1,5,12,30,70):
            solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=8);polls=0
            def stop():
                nonlocal polls
                polls+=1
                if polls==boundary:raise InterruptedError('test stop')
            try:solver.propose(archive,stop)
            except InterruptedError:pass
            saved=solver.snapshot();before=solver.stats['recombination_nodes']
            solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=8,resume=json.loads(json.dumps(saved)))
            self.assertEqual(solver.stats['recombination_nodes'],before)
            for _ in range(40):
                result=solver.propose(archive)
                if result['candidate'] is not None:self.assertTrue(all(v in problem.domains[c] for c,v in enumerate(result['candidate'])))
                if result['solution'] is not None:self.assertTrue(problem.validate(result['solution'])['valid'])
                if all(row['stage']==2 for row in solver.pairs.values()):break
            self.assertTrue(all(row['stage']==2 for row in solver.pairs.values()))
    def test_scope_types_tampering_and_source_changes_are_rejected(self):
        problem,archive,_=complementary_fixture();solver=OptimizingCrossover(problem,max_nodes=1);solver.propose(archive);snapshot=solver.snapshot()
        for mutate in (lambda s:s['state']['stats'].__setitem__('recombination_nodes',True),lambda s:s['state'].__setitem__('pending',1),lambda s:s['state']['models'][0]['state'].__setitem__('complete',True)):
            bad=deepcopy(snapshot);mutate(bad);bad['digest']=_digest({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaises(ValueError):OptimizingCrossover(problem,max_nodes=1,resume=bad)
        with self.assertRaises(ValueError):OptimizingCrossover(problem,max_nodes=2,resume=snapshot)
        problem.domains[0]=problem.domains[0][1:]
        with self.assertRaises(ValueError):solver.propose(archive)
        with self.assertRaises(ValueError):OptimizingCrossover(problem,max_nodes=1,resume=snapshot)
    def test_small_cache_bounds_disclose_eviction_and_no_global_coverage(self):
        problem=SearchProblem(*tetrahedron_tiles());archive=[{'codes':row} for row in ((0,4,7,10),(1,3,7,10),(1,4,6,9))]
        solver=OptimizingCrossover(problem,max_models=1,history_limit=2,max_nodes=4,max_parents=3)
        for _ in range(12):solver.propose(archive)
        self.assertLessEqual(len(solver.models),1);self.assertLessEqual(len(solver.history),2)
        self.assertGreater(solver.stats['recombination_models_evicted'],0);self.assertGreater(solver.stats['recombination_history_evictions'],0)
        self.assertNotIn('exhausted_boards',solver.stats)


class RecombinationLifecycleTests(unittest.TestCase):
    def test_pending_publication_survives_stop_and_is_independently_rechecked(self):
        problem,archive,_=complementary_fixture();solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=64)
        def stop():
            if solver.pending_offer is not None:raise InterruptedError('before publication')
        with self.assertRaises(InterruptedError):solver.propose(archive,stop)
        self.assertIsNotNone(solver.pending_offer)
        restored=OptimizingCrossover(problem,gold_aware=True,max_nodes=64,resume=json.loads(json.dumps(solver.snapshot())))
        result=restored.propose(archive);self.assertTrue(problem.validate(result['solution'])['valid'])
        self.assertGreater(restored.stats['recombination_offer_score_checks'],0)
        restored.record_injection()
    def test_actual160_geometry_candidate_preserves_inventory_and_scope(self):
        from search_fixtures import planted_gold
        data,mesh,witness=planted_gold(261616,size=4);problem=SearchProblem(data,mesh)
        left=witness.copy();right=witness.copy()
        left[0],left[1]=left[1],left[0];right[2],right[3]=right[3],right[2]
        solver=OptimizingCrossover(problem,gold_aware=True,max_nodes=32)
        result=solver.propose([{'codes':left},{'codes':right}])
        self.assertEqual(problem.n,160);self.assertIsNotNone(result['candidate'])
        self.assertTrue(problem.validate(result['solution'])['valid'])
        self.assertEqual(sorted(code//3 for code in result['candidate']),list(range(160)))
    def test_malformed_archive_does_not_create_a_model(self):
        problem,archive,_=complementary_fixture();solver=OptimizingCrossover(problem)
        for invalid in ([{'codes':[0,0,6,9]}],[{'codes':[True,3,6,9]}],archive*5,[{}]):
            with self.assertRaises(ValueError):solver.propose(invalid)
            self.assertFalse(solver.models);self.assertFalse(solver.pairs);self.assertFalse(solver.history)


class OptimizerStopCommitTests(unittest.TestCase):
    def test_last_completed_branch_commits_before_stop_poll(self):
        problem,archive,_=complementary_fixture();parents=tuple(tuple(row['codes']) for row in archive)
        model=Neighborhood(problem,parents,False,True)
        def score(codes):return {'metrics':gold_score(problem,list(codes)),'gold':bool(problem.validate(codes)['valid'])}
        def stop():
            if not model.stack:raise InterruptedError('immediately after final branch')
        try:model.advance(10000,score,stop)
        except InterruptedError:pass
        self.assertTrue(model.complete);self.assertFalse(model.stack)
        restored=Neighborhood(problem,parents,False,True,state=json.loads(json.dumps(model.snapshot())))
        self.assertTrue(restored.complete);self.assertEqual(restored.best,model.best)


class RecombinationEvidenceGuardTests(unittest.TestCase):
    def test_fresh_output_and_frozen_dirty_guard_precede_native_work(self):
        import tempfile
        from pathlib import Path
        import benchmark_recombination_round26 as benchmark
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'evidence.json';output.write_text('{}',encoding='utf-8')
            with patch.object(benchmark,'finite',side_effect=AssertionError('No finite work should run')):
                with self.assertRaises(ValueError):benchmark.main(['--output',str(output)])
            fresh=Path(directory)/'fresh.json'
            with patch.object(benchmark,'finite',side_effect=AssertionError('No finite work should run')),patch.object(benchmark,'source_snapshot',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'available':True,'dirty':True,'commit':'abc'}):
                with self.assertRaises(RuntimeError):benchmark.main(['--output',str(fresh),'--evidence-stage','frozen'])
            self.assertFalse(fresh.exists())
    def test_optimized_runtime_is_refused(self):
        import subprocess,sys,tempfile
        from pathlib import Path
        import benchmark_recombination_round26 as benchmark
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'rejected.json'
            result=subprocess.run([sys.executable,'-O',str(Path(benchmark.__file__)),'--output',str(output)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0);self.assertIn('assert',result.stderr.lower());self.assertFalse(output.exists())
