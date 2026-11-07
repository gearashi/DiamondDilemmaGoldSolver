import unittest
from constraint_context_policy_round26 import ContextPolicy, ContextualPropagation, features, profile_options
from constraint_scheduling import PropagationScheduler
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, double_face

class ContextPolicyTests(unittest.TestCase):
    def test_only_complete_labels_and_held_out_exclusion(self):
        model=ContextPolicy();near=(.2,.2,.3,.4,.5);far=(.9,.9,.8,.7,.6)
        self.assertFalse(model.observe(near,'base',nodes=0,complete=False,case='timeout'))
        for case,vector in [('train-a',near),('train-b',far)]:
            for profile in ('base','impact','support','constructive'):
                good=(profile=='support' if case=='train-a' else profile=='impact')
                self.assertTrue(model.observe(vector,profile,nodes=2 if good else 30,complete=True,case=case))
        self.assertEqual(model.choose(near,explore=False),'support')
        self.assertEqual(model.choose(far,explore=False),'impact')
        self.assertEqual(model.choose(near,explore=False,exclude_case='train-a'),'impact')
        self.assertFalse(model.observe(near,'support',nodes=0,complete=True,case='train-a'))
        self.assertEqual(ContextPolicy(model.snapshot()).snapshot(),model.snapshot())
        corrupt=model.snapshot();corrupt['records'][0]['features'][0]=float('nan')
        with self.assertRaises(ValueError):ContextPolicy(corrupt)

    def test_bounded_model_and_contexts(self):
        model=ContextPolicy()
        for i in range(550):model.observe((.1,.2,.3,.4,.5),'base',nodes=i,complete=True,case=str(i))
        self.assertEqual(len(model.records),512)
        policy=ContextualPropagation(limit=3)
        for i in range(30):policy.observe(str(i),[7,56],removed=0,proofs=0,complete=True)
        self.assertEqual(len(policy.records),3)

    def test_context_retry_and_useful_pruning_over_cost(self):
        policy=ContextualPropagation()
        for _ in range(4):policy.observe('region',[7,56],removed=0,proofs=0,complete=True)
        self.assertEqual([policy.allow('region',[7,56]) for _ in range(4)],[False,False,False,True])
        self.assertTrue(policy.allow('region',[1,8]))
        policy.observe('region',[7,56],removed=1,proofs=0,complete=True)
        self.assertTrue(all(policy.allow('region',[7,56]) for _ in range(12)))

    def test_force_and_interruption_never_train_or_skip_required_checks(self):
        scheduler=PropagationScheduler(incremental=False,contextual=True)
        stats={};calls=[]
        for _ in range(4):scheduler.run('optional',[7,56],lambda:None,stats,optional=True,force=True)
        scheduler.run('optional',[7,56],lambda:calls.append('skip'),stats,optional=True)
        scheduler.run('optional',[7,56],lambda:calls.append('proof'),stats,optional=True,force=True)
        scheduler.run('optional',[7,56],lambda:calls.append('mandatory'),stats)
        self.assertEqual(calls,['proof','mandatory'])
        before=dict(scheduler.contextual.records)
        def fail():raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):scheduler.run('new',[7],fail,stats,optional=True)
        self.assertEqual(dict(scheduler.contextual.records),before)
        scheduler.run('unfinished',[7],lambda:None,stats,optional=True,complete=lambda:False)
        self.assertNotIn(('unfinished',3,2),scheduler.contextual.records)

    def test_profiles_and_features_only_guide_search(self):
        problem=SearchProblem(simple_tiles(),double_face())
        domains=[sum(1<<code for code in row) for row in problem.domains]
        self.assertTrue(all(0<=x<=1 for x in features(problem,domains)))
        original={'native':'python','lookahead':False}
        for profile in ('base','impact','support','constructive'):
            options=profile_options(original,profile)
            self.assertEqual(options['native'],'python')
        self.assertFalse(original['lookahead'])

if __name__=='__main__':unittest.main()
