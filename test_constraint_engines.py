"""Independent finite regression checks for the four optional search extensions."""
from collections import defaultdict
from copy import deepcopy
from itertools import product
import os
import unittest
from unittest.mock import patch

import constraint_dfs as dfs
import constraint_native_search as native
from constraint_branching import ConnectionBranching
from search_options import BOOLEAN_OPTIONS, DEFAULT_OPTIONS
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import simple_tiles, double_face, tetrahedron_tiles, multiple_loops
from test_constraint_models import pair_problem, brute
import test_constraint_learning as learning_tests
from constraint_lcg import LCGStore
from constraint_branching import from_native_frame

FLAGS=('gold_oracle','connection_branching','lcg','native_search')

def options(**enabled):
    result=dict.fromkeys(BOOLEAN_OPTIONS,False)
    result.update(enabled)
    result['native']='compiled' if result['native_search'] else 'python'
    return result


def ambiguous_planted():
    """Keep a planted Gold loop but remove the unique seam colors.

    The two occupied endpoints of each side become {2,10}; reverse side order
    is preserved. Unlike the usual fixture, seam propagation cannot recover
    the whole planted assignment before the oracle is reached.
    """
    data,board,reference=planted_gold(seed=91,size=1,extra_edges=100)
    for tile in data['tiles']:
        ports=defaultdict(set)
        for segment in tile['segments']:
            for side,position in segment:ports[side].add(position)
        remap={side:dict(zip(sorted(points),(2,10))) for side,points in ports.items()}
        for segment in tile['segments']:
            for point in segment:point[1]=remap[point[0]][point[1]]
    problem=SearchProblem(data,board)
    if not problem.validate(reference)['valid']:raise AssertionError('Remapping lost the planted Gold loop.')
    problem.domains=[list(reference) for _ in range(problem.n)]
    return problem,reference


def pending_assignments(checkpoint):
    """Enumerate pending raw boards independently of DFS partition helpers."""
    frames=checkpoint['stack'];pending=set()
    for index,frame in enumerate(frames):
        domains=[int(value,16) for value in frame['domains']]
        if frame['cell']<0:
            if index!=len(frames)-1:raise AssertionError('Unexpanded ancestor')
            branches=[domains]
        else:
            masks=([int(value,16) for value in frame['choice_masks']]
                   if 'choice_masks' in frame else [1<<code for code in frame['choices']])
            branches=[]
            for mask in masks[frame['next']:]:
                child=domains.copy();child[frame['cell']]=mask;branches.append(child)
        for branch in branches:
            candidates=[tuple(code for code in range(mask.bit_length()) if mask>>code&1) for mask in branch]
            boards=set(product(*candidates))
            if pending&boards:raise AssertionError('Pending branches overlap')
            pending|=boards
    return pending


class IntegratedEnginesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:native.load_library();cls.native_available=True
        except (OSError,ValueError,RuntimeError):
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            cls.native_available=False

    def require_native(self):
        if not self.native_available:self.skipTest('Build the optional native search library first.')

    def test_native_rejects_before_python_reasons_and_counts_once(self):
        self.require_native();problem=pair_problem()
        # Both cells demand the same physical tile but retain two rotations.
        # Initial domains are nonempty; the C inventory fixpoint must reject.
        problem.domains=[[0,1],[0,1]]
        self.assertFalse(brute(problem,'gold'))
        active=options(native_search=True,lcg=True,reason_tracking=True)
        with patch.object(dfs.LCGState,'reconstruct',side_effect=AssertionError('Python evaluator must be bypassed')):
            result=dfs.solve(problem,seconds=10,options=active)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['nodes'],1)
        self.assertEqual(result['stats']['backtracks'],1)
        self.assertEqual(result['stats']['native_fast_rejects'],1)
        self.assertEqual(result['stats']['native_search_nodes'],1)

    def test_native_prefilter_with_both_reason_systems_preserves_finite_truth(self):
        self.require_native()
        for problem in (pair_problem(),pair_problem(multiple_loops=True),
                        SearchProblem(*tetrahedron_tiles())):
            for target in ('gold','edges'):
                expected=brute(problem,target)
                active=options(native_search=True,lcg=True,reason_tracking=True,
                               backjumping=True,connection_branching=True)
                result=dfs.solve(problem,seconds=10,target=target,options=active)
                self.assertTrue(result['complete'],result)
                if expected:self.assertIn(tuple(result['codes']),expected)
                else:self.assertEqual(result['status'],'infeasible')
                self.assertGreater(result['stats']['native_fast_nodes'],0)

    def test_native_certificate_skips_duplicate_pass_but_stronger_rules_requeue(self):
        self.require_native()
        first=dfs.solve(pair_problem(),seconds=10,options=options(native_search=True))
        self.assertEqual(first['status'],'solved')
        self.assertEqual(first['stats']['native_calls'],0)
        self.assertEqual(first['stats']['native_basic_passes_skipped'],first['stats']['native_search_nodes'])
        problem=learning_tests.LearningIntegrationTests().backjump_fixture()
        # Two cells reserve tiles A/B. Hall propagation removes A/B from the
        # other two, a deduction not performed by cheap owner/sole-cell rules.
        problem.domains=[list(range(6)),list(range(6)),list(range(12)),list(range(12))]
        later=dfs.solve(problem,seconds=10,seed=4,target='edges',
                        options=options(native_search=True,lcg=True,backjumping=True))
        self.assertEqual(later['status'],'edge_perfect')
        self.assertGreater(later['stats']['native_basic_passes_skipped'],0)
        self.assertGreater(later['stats']['native_calls'],0)
        self.assertGreater(later['stats']['native_arcs_checked'],0)

    def test_asserting_jump_stop_commits_exact_frontier_and_cross_backend_resume(self):
        self.require_native()
        def pending_native(frames):
            ordinary=[from_native_frame(frame) for frame in frames]
            for frame in ordinary:
                frame['domains']=[hex(v) for v in frame['domains']]
                if 'choice_masks' in frame:frame['choice_masks']=[hex(v) for v in frame['choice_masks']]
            return pending_assignments({'stack':ordinary})
        for use_native in (False,True):
            problem=learning_tests.LearningIntegrationTests().backjump_fixture();requested=[False];captured=[]
            original=LCGStore.asserting_frontier
            def assertion(owner,frames,learned,**kwargs):
                replacement=original(owner,frames,learned,**kwargs)
                if replacement is not None and len(replacement)<len(frames):
                    before=pending_native(frames);after=pending_native(replacement)
                    allowed={row for row in before if not all(mask>>row[cell]&1 for cell,mask in learned.literals)}
                    self.assertLess(after,before)
                    self.assertLessEqual(allowed,after)
                    self.assertTrue(all(not problem.validate(list(row))['valid'] for row in before-after))
                    captured.append((before,after));requested[0]=True
                return replacement
            with patch.object(dfs.ConnectivityPruner,'prune',return_value=False), \
                 patch.object(problem,'partial_loop_impossible',return_value=False), \
                 patch.object(LCGStore,'asserting_frontier',assertion):
                first=dfs.solve(problem,seconds=10,seed=2,stop=lambda:requested[0],
                                options=options(native_search=use_native,lcg=True,backjumping=True))
            self.assertTrue(captured)
            self.assertEqual(first['status'],'stopped')
            self.assertEqual(pending_assignments(first['checkpoint']),captured[-1][1])
            self.assertGreater(first['stats']['lcg_asserting_jumps'],0)
            saved=deepcopy(first['checkpoint'])
            self.assertTrue(all(not any(key.startswith('_') for key in frame) for frame in saved['stack']))
            for end_native in (False,True):
                # Turn learning and backjumping OFF: narrowed pending domains
                # must stand alone without serializing a learned-clause graph.
                active=options(native_search=end_native)
                checked=dfs.solve(problem,seconds=0,resume=saved,options=active)
                self.assertEqual(checked['checkpoint']['stack'],saved['stack'])
                completed=dfs.solve(problem,seconds=10,resume=saved,options=active)
                self.assertEqual(completed['status'],'infeasible')
            self.assertEqual(saved,first['checkpoint'])

    def test_all_four_controls_default_off(self):
        self.assertTrue(all(DEFAULT_OPTIONS[name] is False for name in FLAGS))

    def test_all_sixteen_combinations_match_independent_finite_truth(self):
        tetra,board=tetrahedron_tiles()
        fixtures=[pair_problem(),pair_problem(multiple_loops=True),pair_problem(boundary=True),
                  pair_problem(fixed={0:0,1:0}),SearchProblem(tetra,board),
                  SearchProblem(tetra,board,{0:0})]
        for target in ('gold','edges'):
            for index,problem in enumerate(fixtures):
                expected=brute(problem,target)
                for bits in product((False,True),repeat=4):
                    enabled=dict(zip(FLAGS,bits))
                    if enabled['native_search'] and not self.native_available:continue
                    with self.subTest(target=target,fixture=index,enabled=enabled):
                        result=dfs.solve(problem,seconds=10,seed=31,target=target,options=options(**enabled))
                        self.assertTrue(result['complete'],result)
                        if expected:
                            self.assertIn(tuple(result['codes']),expected)
                            self.assertEqual(result['status'],'solved' if result['validation']['valid'] else 'edge_perfect')
                        else:self.assertEqual(result['status'],'infeasible')
                        if enabled['native_search'] and result['nodes']:
                            # Empty initial domains legitimately finish before
                            # loading either control loop.
                            self.assertEqual(result['search_loop_backend'],'compiled')

    def test_actual_oracle_receives_narrowed_planted_domains(self):
        for use_native in (False,True):
            if use_native and not self.native_available:continue
            problem,reference=ambiguous_planted();calls=[]
            original=dfs.GoldOracle.query
            def query(owner,domains,check=lambda:None,**kwargs):
                calls.append((tuple(domains),kwargs['seconds']))
                return original(owner,domains,check,**kwargs)
            with self.subTest(native=use_native),patch.object(dfs.GoldOracle,'query',query):
                result=dfs.solve(problem,seconds=10,seed=7,options=options(gold_oracle=True,native_search=use_native))
            self.assertEqual(result['status'],'solved',result)
            self.assertTrue(problem.validate(result['codes'])['valid'])
            self.assertGreater(result['stats']['gold_oracle_calls'],0)
            self.assertTrue(calls);self.assertTrue(all(budget==.5 for _,budget in calls))
            self.assertTrue(all(sum(mask.bit_count() for mask in domains)<=100 for domains,_ in calls))
            allowed=sum(1<<code for code in reference)
            self.assertTrue(all(all(mask>0 and not mask&~allowed for mask in domains) for domains,_ in calls))

    def test_stopped_branch_resumes_across_both_control_backends(self):
        self.require_native()
        problem=SearchProblem(*tetrahedron_tiles());expected=brute(problem,'gold');pending=[]
        for start_native in (False,True):
            stopped=[False];original=ConnectionBranching.choose
            def choose(owner,*args,**kwargs):
                plan=original(owner,*args,**kwargs);stopped[0]=True;return plan
            active=options(connection_branching=True,lcg=True,native_search=start_native)
            with patch.object(ConnectionBranching,'choose',choose):
                first=dfs.solve(problem,seconds=10,seed=43,options=active,stop=lambda:stopped[0])
            self.assertEqual(first['status'],'stopped');self.assertFalse(first['complete'])
            self.assertTrue(first['checkpoint']['stack'])
            coverage=pending_assignments(first['checkpoint']);pending.append(coverage)
            self.assertTrue(expected&coverage)
            original_checkpoint=deepcopy(first['checkpoint'])
            for end_native in (False,True):
                resumed_options=options(connection_branching=True,lcg=True,native_search=end_native)
                unchanged=dfs.solve(problem,seconds=0,resume=first['checkpoint'],options=resumed_options)
                self.assertEqual(unchanged['checkpoint']['stack'],first['checkpoint']['stack'])
                result=dfs.solve(problem,seconds=10,resume=first['checkpoint'],options=resumed_options)
                self.assertEqual(result['status'],'solved');self.assertIn(tuple(result['codes']),expected&coverage)
                self.assertEqual(first['checkpoint'],original_checkpoint)
        self.assertEqual(pending[0],pending[1])

    def test_stop_after_valid_candidate_keeps_conclusive_result(self):
        for use_native in (False,True):
            if use_native and not self.native_available:continue
            for use_oracle in (False,True):
                problem=pair_problem();stopped=[False];candidates=[]
                def progress(value):
                    if value.get('event')=='candidate':
                        candidates.append(tuple(value['codes']));stopped[0]=True
                result=dfs.solve(problem,seconds=10,stop=lambda:stopped[0],progress=progress,
                                 options=options(native_search=use_native,gold_oracle=use_oracle,lcg=True,connection_branching=True))
                self.assertEqual(result['status'],'solved');self.assertTrue(result['complete'])
                self.assertTrue(stopped[0]);self.assertEqual(candidates,[tuple(result['codes'])])
                self.assertNotIn('checkpoint',result)

    def test_requested_stop_and_zero_budget_never_claim_exhaustion(self):
        problem=pair_problem()
        for use_native in (False,True):
            if use_native and not self.native_available:continue
            active=options(gold_oracle=True,connection_branching=True,lcg=True,native_search=use_native)
            for seconds,stop,status in ((0,None,'timeout'),(10,lambda:True,'stopped')):
                result=dfs.solve(problem,seconds=seconds,stop=stop,options=active)
                self.assertEqual(result['status'],status);self.assertFalse(result['complete'])
                self.assertTrue(result['checkpoint']['stack'])
                self.assertEqual(result['nodes'],0)

    def test_native_choice_honors_python_selection_and_missing_build(self):
        problem=pair_problem();active=options(native_search=True);active['native']='python'
        with patch.object(native,'load_library',side_effect=AssertionError('must not load native loop')):
            result=dfs.solve(problem,seconds=10,options=active)
        self.assertEqual(result['status'],'solved');self.assertEqual(result['search_loop_backend'],'python')
        self.assertIn('Python',result['search_loop_fallback_reason'])
        active['native']='auto'
        with patch.object(native,'load_library',side_effect=OSError('search library missing')):
            result=dfs.solve(problem,seconds=10,options=active)
        self.assertEqual(result['status'],'solved');self.assertEqual(result['search_loop_backend'],'python')
        self.assertIn('search library missing',result['search_loop_fallback_reason'])
        if self.native_available:
            active['native']='compiled'
            with patch.object(native,'load_library',side_effect=OSError('search library missing')):
                with self.assertRaisesRegex(RuntimeError,'build_native.py'):dfs.solve(problem,seconds=10,options=active)

    def test_resume_deeper_than_native_capacity_preserves_pending_work(self):
        self.require_native();problem=SearchProblem(*tetrahedron_tiles())
        stopped=[False];original=ConnectionBranching.choose
        def choose(owner,*args,**kwargs):
            plan=original(owner,*args,**kwargs);stopped[0]=True;return plan
        with patch.object(ConnectionBranching,'choose',choose):
            first=dfs.solve(problem,seconds=10,seed=43,options=options(connection_branching=True),
                            stop=lambda:stopped[0])
        self.assertEqual(first['status'],'stopped')
        saved=deepcopy(first['checkpoint']);self.assertGreater(len(saved['stack']),1)
        expected=brute(problem,'gold')&pending_assignments(saved);self.assertTrue(expected)
        def limited(n,count,**kwargs):return native.NativeSearch(n,count,max_depth=1),None
        with patch.object(dfs,'select_native_search',limited):
            result=dfs.solve(problem,seconds=10,resume=saved,
                             options=options(native_search=True,connection_branching=True,lcg=True))
        self.assertEqual(result['status'],'solved');self.assertIn(tuple(result['codes']),expected)
        self.assertEqual(result['search_loop_backend'],'python')
        self.assertRegex(result['search_loop_fallback_reason'].lower(),'capacity|depth')
        self.assertEqual(saved,first['checkpoint'])

    def test_native_capacity_fallback_continues_exact_pending_work(self):
        self.require_native();problem=pair_problem();expected=brute(problem,'gold')
        def limited(n,count,**kwargs):return native.NativeSearch(n,count,max_depth=1),None
        with patch.object(dfs,'select_native_search',limited):
            result=dfs.solve(problem,seconds=10,options=options(native_search=True,connection_branching=True,lcg=True))
        self.assertEqual(result['status'],'solved');self.assertIn(tuple(result['codes']),expected)
        self.assertEqual(result['search_loop_backend'],'python')
        self.assertIn('capacity',result['search_loop_fallback_reason'])
        self.assertGreater(result['stats']['native_search_nodes'],0)

    def test_all_four_enabled_emit_real_engine_counters(self):
        self.require_native()
        # Oracle disabled by its scheduling gate only in this test so the
        # connection branch and trail actually execute before a Gold witness.
        problem=SearchProblem(*tetrahedron_tiles())
        with patch.object(dfs.GoldOracle,'should_query',return_value=False):
            result=dfs.solve(problem,seconds=10,options=options(**dict.fromkeys(FLAGS,True)))
        self.assertEqual(result['status'],'solved')
        self.assertEqual(result['search_loop_backend'],'compiled')
        self.assertGreater(result['stats']['native_search_nodes'],0)
        self.assertGreater(result['stats']['native_search_branches'],0)
        self.assertGreater(result['stats']['lcg_rebases'],0)
        self.assertGreater(result['stats']['lcg_nodes'],0)

if __name__=='__main__':unittest.main()
