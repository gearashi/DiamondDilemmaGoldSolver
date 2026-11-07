import json,unittest
from constraint_advanced import AdvancedPruner
from test_constraint_advanced import ConditionalGlobalSupportTests

class ContinuationTests(unittest.TestCase):
    def test_cut_union_resumes_without_repeating_completed_vectors(self):
        p,d=ConditionalGlobalSupportTests.fixture();counts=[];answers=[]
        for enabled in (False,True):
            helper=AdvancedPruner(p,options={'native':'python','resumable_reasoning':enabled});helper.cut_matching_work_limit=20
            for _ in range(10):
                answer=helper._cut_support(d,(0,1),lambda:None)
                if answer is not None:break
            self.assertIsNotNone(answer);answers.append(answer);counts.append(helper.stats['cut_matching_checks'])
        self.assertEqual(answers[0],answers[1]);self.assertEqual(counts,[8,5])
    def test_restart_keeps_partial_union_private_and_reuses_completed_states(self):
        p,d=ConditionalGlobalSupportTests.fixture();helper=AdvancedPruner(p,options={'native':'python','resumable_reasoning':True})
        helper.cut_matching_work_limit=20
        self.assertIsNone(helper._cut_support(d,(0,1),lambda:None));self.assertFalse(helper._cut_cache)
        saved=json.loads(json.dumps(helper.reasoning_snapshot()))
        other=AdvancedPruner(p,options={'native':'python','resumable_reasoning':True});other.restore_reasoning(saved)
        answer=other._cut_support(d,(0,1),lambda:None)
        expected=AdvancedPruner(p,options={'native':'python'})._cut_support(d,(0,1),lambda:None)
        self.assertEqual(answer,expected);self.assertGreater(other.stats['cut_continuation_states_reused'],0)
        saved['target']='edges'
        with self.assertRaises(ValueError):other.restore_reasoning(saved)
    def test_stop_every_checkpoint_keeps_exact_result(self):
        p,d=ConditionalGlobalSupportTests.fixture();expected=AdvancedPruner(p,options={'native':'python'})._cut_support(d,(0,1),lambda:None)
        for at in range(1,75):
            helper=AdvancedPruner(p,options={'native':'python','resumable_reasoning':True});calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==at:raise InterruptedError()
            try:helper._cut_support(d,(0,1),check)
            except InterruptedError:pass
            other=AdvancedPruner(p,options={'native':'python','resumable_reasoning':True});other.restore_reasoning(json.loads(json.dumps(helper.reasoning_snapshot())))
            self.assertEqual(other._cut_support(d,(0,1),lambda:None),expected)

if __name__=='__main__':unittest.main()
