"""Integrated R23 feature combinations, including resumable proof scope."""
import copy,random,unittest
from unittest.mock import patch
from constraint_deep import CertificateStore
from constraint_dfs import solve
from search_options import EXPERIMENTAL_OPTIONS,ROUND25_OPTIONS
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute

class IntegratedCandidatesTests(unittest.TestCase):
    def options(self):
        # Freeze this historical combination to pre-Round25 options. New
        # engine-specific controls need separate tests with compatible engines
        # and prerequisites; blindly enabling them changes the regression.
        historical=tuple(key for key in EXPERIMENTAL_OPTIONS if key not in ROUND25_OPTIONS)
        return {**dict.fromkeys(historical,True),'native':'python','lcg':True,'sat_assist':True,'shared_conflicts':True,'reason_tracking':True,'connection_branching':True}
    def test_combinations_preserve_small_gold_feasibility(self):
        problem,_=fixture();solutions=brute(problem,'gold');rng=random.Random(23017)
        for sample in range(12):
            candidate=copy.deepcopy(problem)
            masks=[rng.randrange(1,4096) for _ in range(4)]
            candidate.domains=[[code for code in row if masks[c]>>code&1] for c,row in enumerate(candidate.domains)]
            expected=bool(admitted(solutions,masks))
            for native in ('python','auto'):
                options=self.options();options['native']=native;options['native_search']=native=='auto'
                result=solve(candidate,seconds=5,seed=sample,options=options)
                self.assertIn(result['status'],('solved','infeasible'))
                self.assertEqual(result['status']=='solved',expected,(masks,result))
                if expected:self.assertTrue(candidate.validate(result['codes'])['valid'])
    def test_new_group_and_proof_counters_resume(self):
        problem,_=fixture();options=self.options();events=[]
        first=solve(problem,seconds=5,options=options,progress=events.append,stop=lambda:len(events)>0)
        if first['status']=='stopped':
            restored=solve(problem,seconds=5,resume=first['checkpoint'],options=options)
            self.assertEqual(restored['status'],'solved');self.assertTrue(restored['validation']['valid'])
        else:self.assertEqual(first['status'],'solved')
    def test_empty_domain_combination_is_a_completed_failure(self):
        problem,_=fixture();problem.domains[0]=[]
        result=solve(problem,seconds=5,options=self.options())
        self.assertEqual(result['status'],'infeasible')
    def test_stop_after_last_rejected_frame_preserves_completed_exhaustion(self):
        problem,_=fixture()
        for cell in range(3):problem.domains[cell]=list(range(6))
        requested=[False]
        def capture(store,domains,check):
            requested[0]=True;check()
        with patch.object(CertificateStore,'capture',capture):
            result=solve(problem,seconds=5,options={'native':'python','proof_certificates':True},stop=lambda:requested[0])
        self.assertTrue(requested[0]);self.assertEqual(result['status'],'infeasible')
        self.assertNotIn('checkpoint',result)

    def test_symmetry_cache_cannot_escape_assumptions(self):
        problem,_=fixture();learning={};options=self.options()
        solve(problem,seconds=5,options=options,learning_state=learning)
        options['piece_symmetry']=False
        with self.assertRaisesRegex(ValueError,'symmetry assumptions'):
            solve(problem,seconds=1,options=options,learning_state=learning)

class HigherRelationEmptyDomainTests(unittest.TestCase):
    def test_empty_original_domains_complete_without_constructing_projected_relations(self):
        for target in ('gold','edges'):
            problem,_=fixture();problem.domains[0]=[]
            with patch('constraint_dfs.ProjectedRelations',side_effect=AssertionError('Empty scope reached projection constructor')):
                result=solve(problem,seconds=5,target=target,options={'native':'python','higher_order_relations':True})
            self.assertEqual(result['status'],'infeasible');self.assertTrue(result['complete'])
            self.assertNotIn('checkpoint',result)

    def test_empty_owned_resume_keeps_valid_original_root_and_completes(self):
        from constraint_dfs import _solve_core
        from constraint_parallel import _root_copy
        for target in ('gold','edges'):
            problem,_=fixture();original=copy.deepcopy(problem.domains)
            template=_solve_core(problem,seconds=0,target=target,options={'native':'python'})['checkpoint']
            domains=[sum(1<<code for code in row) for row in problem.domains];domains[1]=0
            saved=_root_copy(template,domains);before=copy.deepcopy(saved)
            with patch('constraint_dfs.ProjectedRelations',side_effect=AssertionError('Empty owned scope reached projection constructor')):
                result=solve(problem,seconds=5,target=target,resume=saved,
                             options={'native':'python','higher_order_relations':True})
            self.assertEqual(result['status'],'infeasible');self.assertTrue(result['complete'])
            self.assertNotIn('checkpoint',result);self.assertEqual(problem.domains,original)
            self.assertEqual(saved,before)

if __name__=='__main__':unittest.main()
