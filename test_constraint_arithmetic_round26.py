import itertools,random,unittest
from constraint_pb import row,filter_rows
from constraint_arithmetic_round26 import apply_rule,replay,normalized_literals,IndexedPBPropagation,ArithmeticPBLearner
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute


def holds(cut,assignment):return sum(a*assignment[c,v] for c,v,a in cut[0])<=cut[1]

class ArithmeticRuleTests(unittest.TestCase):
    def test_all_rules_preserve_every_boolean_model_of_random_rows(self):
        rng=random.Random(260704)
        for _ in range(120):
            source=row([(i,0,rng.randrange(-6,8)) for i in range(5)],rng.randrange(-5,16));literals,rhs=normalized_literals(source)
            rules=[('divide',2),('divide',3),('saturate',None)]+[('weaken',i) for i in range(len(source[0]))]
            for count in range(1,len(literals)+1):
                cover=list(range(count))
                if sum(literals[i][3] for i in cover)>rhs and rhs>=0:
                    rules.append(('cover',{'cover':cover,'lift':list(range(count,len(literals)))}));break
            for rule,args in rules:
                consequence=apply_rule(source,rule,args)
                for bits in itertools.product((0,1),repeat=5):
                    assignment={(i,0):v for i,v in enumerate(bits)}
                    if holds(source,assignment):self.assertTrue(holds(consequence,assignment),(source,rule,args,consequence,bits))
                self.assertEqual(replay([source],[{'parents':[0],'rule':rule,'args':args,'row':consequence}])[-1],consequence)
        source=row([(0,0,5),(1,0,4),(2,0,3)],8)
        self.assertEqual(apply_rule(source,'cover',{'cover':[0,1],'lift':[2]}),row([(0,0,1),(1,0,1)],1))

    def test_strict_types_forged_steps_and_stop_never_admit(self):
        source=row([(0,0,2),(1,0,3)],3)
        for rule,args in [('divide',True),('divide',0),('weaken',-1),('cover',{'cover':[True],'lift':[]}),('cover',{'cover':[0],'lift':[0]})]:
            with self.assertRaises(ValueError):apply_rule(source,rule,args)
        with self.assertRaises(ValueError):replay([source],[{'parents':[0],'rule':'divide','args':2,'row':((),-1)}])
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==4:raise InterruptedError
        with self.assertRaises(InterruptedError):apply_rule(row([(0,0,5),(1,0,4),(2,0,3)],8),'cover',{'cover':[0,1],'lift':[2]},stop)

    def test_incremental_queue_same_fixpoint_and_resume_widening(self):
        rows=[row([(i,0,sign),(i+1,0,sign)],sign) for i in range(5) for sign in (-1,1)]
        helper=IndexedPBPropagation(rows);initial=[3]*6;initial[0]=1
        expected=initial.copy()
        while filter_rows(expected,rows,max_passes=1):pass
        current=initial.copy();helper.filter(current);self.assertEqual(current,expected)
        for start in ([3]*6,[2,3,3,3,3,1],initial):
            expected=list(start)
            while filter_rows(expected,rows,max_passes=1):pass
            actual=list(start);helper.filter(actual);self.assertEqual(actual,expected)
        for cutoff in range(1,24):
            worker=IndexedPBPropagation(rows);current=initial.copy();calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==cutoff:raise InterruptedError
            try:worker.filter(current,stop)
            except InterruptedError:self.assertEqual(current,initial)
            worker.filter(current);self.assertEqual(current,[1,2,1,2,1,2])

    def test_existing_pb_learner_uses_queue_without_new_axioms_and_exports_checked_reasons(self):
        from constraint_pb import PBLearner
        from constraint_deduction_chain_round26 import verify_chain
        import json
        p,_=fixture();initial=[3972,4045,4061,879]
        for discover in (False,True):
            old=PBLearner(p,discover=discover);new=PBLearner(p,discover=discover,indexed=True)
            a=initial.copy();b=initial.copy()
            old.prune(a,max_values=12);new.prune(b,max_values=12)
            # Queue runs to an exact fixpoint; iterate the original row scan to
            # that same fixed point for the semantic comparison.
            while old.prune(a,max_values=12):pass
            while new.prune(b,max_values=12):pass
            self.assertEqual(a,b)
            self.assertGreater(new.queue_stats['arithmetic_rows_tested'],0)
            for certificate in new.drain_deduction_chains(target='edges',limit=128):verify_chain(p,json.loads(json.dumps(certificate)),target='edges')

    def test_plain_and_conflict_pb_export_original_integer_derivations(self):
        from constraint_pb import PBLearner
        from constraint_conflict_pb import ConflictPBLearner
        from constraint_deduction_chain_round26 import verify_chain
        import json
        p,initial=fixture();answers=brute(p,'edges')
        for cls in (PBLearner,ConflictPBLearner):
            learner=cls(p,explain=True);current=initial.copy()
            if cls is PBLearner:learner.prune(current,max_values=12)
            else:learner.prune(current)
            certificates=list(learner.drain_deduction_chains(target='edges',limit=128))
            for cert in certificates:
                guards=verify_chain(p,json.loads(json.dumps(cert)),target='edges')
                self.assertFalse(any(all(mask>>row[cell]&1 for cell,mask in guards) for row in answers))
            if cls is ConflictPBLearner:self.assertTrue(certificates)

    def test_actual_puzzle_scope_replay_and_all_solution_preservation(self):
        p,_=fixture();solutions={target:brute(p,target) for target in ('edges','gold')};rng=random.Random(260705);helper=ArithmeticPBLearner(p)
        for _ in range(25):
            initial=[sum(1<<code for code in row if rng.random()<.6) for row in p.domains];current=initial.copy()
            helper.prune(current,max_values=12)
            for target,boards in solutions.items():self.assertEqual(admitted(boards,initial),admitted(boards,current),target)
            from constraint_deduction_chain_round26 import verify_chain
            import json
            for certificate in helper.drain_deduction_chains(target='edges'):
                verify_chain(p,json.loads(json.dumps(certificate)),target='edges')
        for record in helper.records.values():self.assertEqual(replay(record['base'],record['steps']),record['rows'])
        self.assertGreater(helper.stats['arithmetic_divisions'],0)

if __name__=='__main__':unittest.main()
