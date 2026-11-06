"""Independent solution coverage through SAT deductions and CPU checkpoints."""
import copy
import os
import unittest
from unittest.mock import patch
import constraint_dfs as dfs
from constraint_sat_assist import SatAssist
from search_options import BOOLEAN_OPTIONS
from test_constraint_dfs import brute
from test_constraint_engines import pending_assignments
from test_constraint_sat_domains import fixture
try:
    import pysat
    HAS_SAT=True
except ImportError:
    HAS_SAT=False


def problem_and_solutions(target):
    problem,masks=fixture()
    problem.domains=[[c for c in row if masks[i]&(1<<c)] for i,row in enumerate(problem.domains)]
    return problem,set(brute(problem,target))


def options(native=False,**extra):
    result=dict.fromkeys(BOOLEAN_OPTIONS,False)
    result.update(native='compiled' if native else 'python',native_search=native,
                  sat_assist=True,reason_tracking=True,lcg=True,backjumping=True,
                  connection_branching=True)
    result.update(extra)
    return result


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class SatDomainIntegrationTests(unittest.TestCase):
    def native_available(self):
        engine,reason=dfs.select_native_search(4,12)
        if engine is None:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':self.fail(str(reason))
            return False
        engine.close();return True

    def test_actual_sat_consequences_preserve_both_targets_and_learned_scope(self):
        prepare=SatAssist.prepare;filter_domains=SatAssist.filter_domains
        for target in ('gold','edges'):
            for native in (False,True):
                if native and not self.native_available():continue
                problem,solutions=problem_and_solutions(target);self.assertTrue(solutions)
                learning={};captured=[]
                def ready(helper,check=lambda:None,**kwargs):return prepare(helper,check,seconds=1.)
                def checked(helper,domains,check=lambda:None,**kwargs):
                    before=tuple(domains)
                    result=filter_domains(helper,domains,check,seconds=1.,conflicts=10000,max_candidates=48)
                    retained={row for row in solutions if all(d&(1<<c) for d,c in zip(before,row))}
                    self.assertTrue(all(all(d&(1<<c) for d,c in zip(result,row)) for row in retained))
                    captured.append((before,result));return result
                with self.subTest(target=target,native=native),patch.object(SatAssist,'prepare',ready),patch.object(SatAssist,'filter_domains',checked):
                    result=dfs.solve(problem,seconds=10,target=target,options=options(native),learning_state=learning)
                self.assertTrue(result['complete']);self.assertIn(tuple(result['codes']),solutions)
                self.assertTrue(captured)
                self.assertTrue(any(before!=after for before,after in captured))
                self.assertGreater(result['stats']['sat_assist_domain_prunes'],0)
                for clause in learning['_nogoods'].clauses:
                    self.assertTrue(all(not all(row[c]==code for c,code in clause) for row in solutions))
                if native:self.assertEqual(result['search_loop_backend'],'compiled')
                learning['_runtime'].sat.close()

    def test_stop_after_scoped_reductions_and_resume_with_assistance_disabled(self):
        prepare=SatAssist.prepare;filter_domains=SatAssist.filter_domains
        for native in (False,True):
            if native and not self.native_available():continue
            problem,solutions=problem_and_solutions('gold');stopped=[False];learning={};captured=[]
            def ready(helper,check=lambda:None,**kwargs):return prepare(helper,check,seconds=1.)
            def pause(helper,domains,check=lambda:None,**kwargs):
                before=tuple(domains)
                result=filter_domains(helper,domains,check,seconds=1.,conflicts=10000,max_candidates=48)
                if result!=before:captured.append(result);stopped[0]=True
                return result
            with patch.object(SatAssist,'prepare',ready),patch.object(SatAssist,'filter_domains',pause):
                result=dfs.solve(problem,seconds=10,options=options(native),learning_state=learning,stop=lambda:stopped[0])
            self.assertTrue(captured);self.assertEqual(result['status'],'stopped')
            pending=pending_assignments(result['checkpoint'])
            self.assertLessEqual(solutions,pending)
            for resume_native in (False,True):
                if resume_native and not self.native_available():continue
                resumed=dfs.solve(problem,seconds=10,resume=copy.deepcopy(result['checkpoint']),
                                  options=options(resume_native,sat_assist=False))
                self.assertIn(tuple(resumed['codes']),solutions)
            learning['_runtime'].sat.close()


if __name__=='__main__':unittest.main()
