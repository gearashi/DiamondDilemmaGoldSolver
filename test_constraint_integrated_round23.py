"""Integrated R23 feature combinations, including resumable proof scope."""
import copy,random,unittest
from unittest.mock import patch
from constraint_deep import CertificateStore
from constraint_dfs import solve
from search_options import EXPERIMENTAL_OPTIONS
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute

class IntegratedCandidatesTests(unittest.TestCase):
    def options(self):
        return {**dict.fromkeys(EXPERIMENTAL_OPTIONS,True),'native':'python','lcg':True,'sat_assist':True,'shared_conflicts':True,'reason_tracking':True,'connection_branching':True}
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

if __name__=='__main__':unittest.main()
