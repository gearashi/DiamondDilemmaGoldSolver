"""CPU-only retained Gold oracle/master finite correctness and lifecycle tests."""
import copy
import itertools
import json
import unittest
from unittest.mock import patch
from constraint_lns import _Control, _Cancelled
from constraint_repair_gold_round26 import GoldRepairPool, _digest
from constraint_repair_round24 import core_guided_repair, fragment_crossover, scoped_repair_failure, matched_fragments
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles, simple_tiles, double_face, multiple_loops, brute
from test_dependencies import requires_ortools


def resign(value):
    value['digest']=_digest({k:v for k,v in value.items() if k!='digest'})
    return value


def hit_optimum(cells,cores):
    for size in range(len(cells)+1):
        for row in itertools.combinations(cells,size):
            if all(set(row)&set(core) for core in cores): return size
    return None


def fragment_optimum(fragments,conflicts):
    best=None
    for bits in range(1<<len(fragments)):
        selected={i for i in range(len(fragments)) if bits>>i&1}
        if any(set(row)<=selected for row in conflicts): continue
        places=[p for i in selected for p in fragments[i]['placements']]
        if len({c for c,v in places})!=len(places) or len({v//3 for c,v in places})!=len(places): continue
        best=max(best or 0,len(places))
    return best


@requires_ortools
class RetainedGoldRepairTests(unittest.TestCase):
    def setUp(self):
        from ortools.sat.python import cp_model
        self.cp=cp_model
        self.problem=SearchProblem(*tetrahedron_tiles())
        self.seed=[0,4,6,9]

    def test_gold_model_reused_across_assumptions_and_complete_results(self):
        p=self.problem; solutions=brute(p,'gold'); pool=GoldRepairPool(p)
        pairs=[(),((0,self.seed[0]),),((0,self.seed[0]),(1,self.seed[1])),((2,self.seed[2]),)]
        for assumptions in pairs:
            answer=pool.oracle(_Control(10,None)).query(assumptions)
            expected=[b for b in solutions if all(b[c]==v for c,v in assumptions)]
            self.assertEqual(answer['status'],'solved' if expected else 'infeasible')
            if expected:self.assertIn(tuple(answer['codes']),solutions)
        before=pool.stats
        for assumptions in pairs:pool.oracle(_Control(10,None)).query(assumptions)
        self.assertEqual(pool.stats['gold_pool_native_calls'],before['gold_pool_native_calls'])
        self.assertEqual(pool.stats['gold_pool_model_builds'],1)
        self.assertGreater(pool.stats['gold_pool_completed_hits']+pool.stats['gold_pool_core_hits'],0)
        saved=json.loads(json.dumps(pool.snapshot())); restored=GoldRepairPool(p,resume=saved)
        for assumptions in pairs:restored.oracle(_Control(10,None)).query(assumptions)
        self.assertEqual(restored.stats['gold_pool_model_builds'],1)
        self.assertIsNone(restored._oracle_model)

    def test_core_claim_subset_reuse_and_unknown_retry_is_not_failure(self):
        pool=GoldRepairPool(self.problem); cp=self.cp
        class UnknownSolver:
            parameters=type('Parameters',(),{})()
            num_branches=0;num_conflicts=0
            def solve(self,model):return cp.UNKNOWN
            def stop_search(self):pass
        with patch.object(cp,'CpSolver',UnknownSolver):
            first=pool.oracle(_Control(10,None)).query([(0,0)])
            second=pool.oracle(_Control(10,None)).query([(0,0)])
        self.assertEqual(first['status'],second['status']);self.assertEqual(first['status'],'unknown')
        self.assertEqual(pool.failures,[]);self.assertEqual(pool.results,[])
        self.assertEqual(pool.stats['gold_pool_native_calls'],2)
        self.assertEqual(pool.stats['gold_pool_native_retries'],1)
        restored=GoldRepairPool(self.problem,resume=json.loads(json.dumps(pool.snapshot())))
        answer=restored.oracle(_Control(10,None)).query([(0,0)])
        self.assertIn(answer['status'],('solved','infeasible'))
        self.assertEqual(restored.stats['gold_pool_native_retries'],2)
        self.assertEqual(restored.stats['gold_pool_model_builds'],2)
        self.assertEqual(restored.pending,[])

    def test_hitting_master_exact_incremental_and_reversed_scope(self):
        pool=GoldRepairPool(self.problem); cells=list(range(4));rows=[]
        for row in ([0,1],[1,2],[2,3],[0,3]):
            rows.append(row)
            selected,work=pool.minimum_hitting_set(rows,cells,_Control(10,None),self.cp,seed=self.seed)
            self.assertEqual(len(selected),hit_optimum(cells,rows))
        calls=pool.stats['gold_pool_master_calls']
        selected,_=pool.minimum_hitting_set(rows,cells,_Control(10,None),self.cp,seed=self.seed)
        self.assertEqual(pool.stats['gold_pool_master_calls'],calls)
        self.assertEqual(pool.stats['gold_pool_master_builds'],1)
        self.assertGreater(pool.stats['gold_pool_master_bound_reuses'],0)
        selected,_=pool.minimum_hitting_set([],cells,_Control(10,None),self.cp,seed=self.seed)
        self.assertEqual(selected,set());self.assertEqual(pool.stats['gold_pool_master_builds'],2)
        saved=pool.snapshot();restored=GoldRepairPool(self.problem,resume=saved)
        selected,_=restored.minimum_hitting_set([],cells,_Control(10,None),self.cp,seed=self.seed)
        self.assertEqual(selected,set());self.assertEqual(restored.stats['gold_pool_master_builds'],2)

    def test_fragment_master_exact_constraints_bound_and_rebuild(self):
        pool=GoldRepairPool(self.problem)
        fragments=[{'source':0,'placements':[(0,0)]},{'source':0,'placements':[(1,3)]},
                   {'source':1,'placements':[(0,3)]},{'source':1,'placements':[(1,0)]}]
        conflicts=[]
        for new in (None,[0,1],[2,3],[0]):
            if new is not None:conflicts.append(new)
            chosen,work=pool.fragment_master(fragments,conflicts,_Control(10,None),self.cp)
            self.assertEqual(sum(len(fragments[i]['placements']) for i in chosen),fragment_optimum(fragments,conflicts))
        self.assertEqual(pool.stats['gold_pool_master_builds'],1)
        self.assertGreater(pool.stats['gold_pool_master_bound_reuses'],0)
        previous=pool.stats['gold_pool_master_calls'];restored=GoldRepairPool(self.problem,resume=pool.snapshot())
        restored.fragment_master(fragments,conflicts,_Control(10,None),self.cp)
        self.assertEqual(restored.stats['gold_pool_master_calls'],previous)

    def test_actual_controllers_share_one_gold_model_and_preserve_optima(self):
        pool=GoldRepairPool(self.problem)
        first=core_guided_repair(self.problem,self.seed,seconds=10,retained_service=pool)
        self.assertEqual(first['status'],'solved',first)
        self.assertEqual(first['minimum_changes'],min(sum(a!=b for a,b in zip(self.seed,row)) for row in brute(self.problem,'gold')))
        calls=pool.stats['gold_pool_native_calls']; masters=pool.stats['gold_pool_master_calls']
        second=core_guided_repair(self.problem,self.seed,seconds=10,retained_service=pool)
        self.assertEqual(second['minimum_changes'],first['minimum_changes'])
        self.assertEqual(pool.stats['gold_pool_native_calls'],calls)
        # Different intermediate hitting-set histories may legitimately differ;
        # final completed master/query records remain reusable.
        self.assertLessEqual(pool.stats['gold_pool_master_calls']-masters,first['stats']['repair_completed_masters'])
        third=scoped_repair_failure(self.problem,first['best_board'],[],seconds=10,retained_service=pool)
        self.assertEqual(third['status'],'solved');self.assertEqual(pool.stats['gold_pool_model_builds'],1)
        crossed=fragment_crossover(self.problem,[self.seed,first['best_board']],max_fragment=2,seconds=10,retained_service=pool)
        self.assertEqual(crossed['status'],'solved',crossed)
        self.assertEqual(pool.stats['gold_pool_model_builds'],1)
        resumed=core_guided_repair(self.problem,self.seed,seconds=0,resume=first['checkpoint'],retained_service=pool)
        self.assertEqual(resumed['status'],'solved')

    def test_master_unknown_and_interruption_resume_without_false_optimum(self):
        pool=GoldRepairPool(self.problem);cp=self.cp
        class UnknownSolver:
            parameters=type('Parameters',(),{})()
            num_branches=0
            def solve(self,model):return cp.UNKNOWN
            def stop_search(self):pass
        with patch.object(cp,'CpSolver',UnknownSolver):
            for _ in range(2):
                answer=pool.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),cp,seed=self.seed)
                self.assertEqual(answer,(None,'unknown'))
        self.assertFalse(pool.masters[0]['optimal']);self.assertTrue(pool.masters[0]['retry'])
        self.assertEqual(pool.stats['gold_pool_master_calls'],2);self.assertEqual(pool.stats['gold_pool_master_retries'],1)
        saved=pool.snapshot();restored=GoldRepairPool(self.problem,resume=saved)
        class InterruptedSolver(UnknownSolver):
            def solve(self,model):raise _Cancelled()
        with patch.object(cp,'CpSolver',InterruptedSolver),self.assertRaises(_Cancelled):
            restored.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),cp,seed=self.seed)
        self.assertEqual(restored.stats['gold_pool_master_interrupted'],1)
        selected,_=restored.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),cp,seed=self.seed)
        self.assertEqual(len(selected),1);self.assertEqual(restored.stats['gold_pool_master_retries'],3)
        self.assertEqual(restored.stats['gold_pool_master_builds'],2)

    def test_core_learning_clause_cap_rebuilds_without_unsafe_reuse(self):
        pool=GoldRepairPool(self.problem,result_limit=1)
        solutions=brute(self.problem,'gold')
        for assumptions in (((0,1),),((1,0),),((2,0),)):
            self.assertFalse(any(all(board[c]==v for c,v in assumptions) for board in solutions))
            answer=pool.oracle(_Control(10,None)).query(assumptions)
            self.assertEqual(answer['status'],'infeasible')
            self.assertLessEqual(len(pool.failures),1);self.assertLessEqual(len(pool._posted_cores),1)
        self.assertGreater(pool.stats['gold_pool_core_clauses_added'],0)
        self.assertGreater(pool.stats['gold_pool_model_evictions'],0)
        self.assertGreater(pool.stats['gold_pool_result_evictions'],0)

    def test_finite_comparator_has_independent_actual160_and_all_costs(self):
        from benchmark_gold_repair_round26 import finite
        result=finite()
        gold=result['gold_queries']
        self.assertEqual(gold['repeated_contexts_native_calls'],0)
        self.assertLess(gold['retained_with_one_restore']['stats']['gold_pool_model_builds'],gold['cold']['models'])
        self.assertEqual(result['controlled160']['cells'],160)
        self.assertEqual(result['controlled160']['enumerated_assignments'],162)
        self.assertGreater(result['controlled160']['gold_boards'],0)
        self.assertEqual(gold['cold']['native']['statuses'].get('UNKNOWN',0),0)

    def test_source_mutation_and_callback_error_do_not_publish_work(self):
        from pathlib import Path
        pool=GoldRepairPool(self.problem);original=Path.read_bytes
        def changed(path):
            value=original(path)
            return value+b'changed' if path.name=='constraint_circuit.py' else value
        with patch.object(Path,'read_bytes',changed),self.assertRaises(ValueError):pool.oracle(_Control(10,None))
        before=pool.stats
        def fail():raise OSError('callback failure')
        with self.assertRaisesRegex(OSError,'callback failure'):pool.oracle(_Control(10,fail))
        self.assertEqual(pool.stats['gold_pool_native_calls'],before['gold_pool_native_calls'])
        self.assertFalse(pool.pending);self.assertFalse(pool.failures)

    def test_strict_snapshot_and_live_semantic_binding(self):
        pool=GoldRepairPool(self.problem)
        pool.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),self.cp,seed=self.seed)
        original=pool.snapshot()
        for mutate in (lambda v:v.update(version=True),lambda v:v['stats'].__setitem__('gold_pool_master_calls',True),
                       lambda v:v['masters'][0].__setitem__('optimal',1),lambda v:v['masters'][0].__setitem__('bound',.5),
                       lambda v:v['masters'][0]['context'].__setitem__('seed',[0,0,0,0])):
            bad=copy.deepcopy(original);mutate(bad);resign(bad)
            with self.assertRaises((ValueError,TypeError)):GoldRepairPool(self.problem,resume=bad)
        detached=pool.snapshot();detached['masters'].clear();self.assertTrue(pool.masters)
        self.problem.domains[0]=tuple(self.problem.domains[0][1:])
        with self.assertRaises(ValueError):pool.oracle(_Control(10,None))

    def test_stop_before_work_and_master_eviction_are_explicit(self):
        pool=GoldRepairPool(self.problem,model_limit=1,result_limit=1)
        with self.assertRaises(_Cancelled):pool.oracle(_Control(10,lambda:True))
        self.assertEqual(pool.stats['gold_pool_model_builds'],0)
        pool.minimum_hitting_set([[0]],list(range(4)),_Control(10,None),self.cp,seed=self.seed)
        other=[3,1,6,9]
        pool.minimum_hitting_set([[1]],list(range(4)),_Control(10,None),self.cp,seed=other)
        self.assertEqual(len(pool.masters),1);self.assertEqual(pool.stats['gold_pool_master_evictions'],1)
        self.assertEqual(len(pool._native_masters),1)
        pool.close()
        with self.assertRaises(RuntimeError):pool.snapshot()

    def test_completed_master_commits_before_next_stop(self):
        pool=GoldRepairPool(self.problem)
        value,_=pool.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),self.cp,seed=self.seed)
        saved=pool.snapshot();restored=GoldRepairPool(self.problem,resume=saved)
        with self.assertRaises(_Cancelled):restored.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,lambda:True),self.cp,seed=self.seed)
        calls=restored.stats['gold_pool_master_calls']
        again,_=restored.minimum_hitting_set([[0,1]],list(range(4)),_Control(10,None),self.cp,seed=self.seed)
        self.assertEqual(value,again);self.assertEqual(restored.stats['gold_pool_master_calls'],calls)

if __name__=='__main__':unittest.main()
