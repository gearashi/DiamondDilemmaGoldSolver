"""Finite independent relations and complete tiny puzzle witnesses for R23."""
import copy,itertools,json,random,tempfile,unittest
from pathlib import Path
from constraint_lookahead import CompletedProbeCache,grouped_constructive_lookahead
from constraint_lcg import LCGStore,LearnedClause
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_sat_assist import SatAssist
from constraint_scheduling import PropagationScheduler,STAT_KEYS
from test_constraint_sat_assist import HAS_SAT,GoldFixtureOracle,Fake
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute


def clause(store,literals):return LearnedClause(store.binding,tuple(literals),None,0,0,0)


class ProbeImplicationTests(unittest.TestCase):
    def test_narrower_context_reuses_by_intersection_and_detects_conflict(self):
        cache=CompletedProbeCache();stats={};calls=[]
        def probe(domains,cell):calls.append(1);domains[1]&=3;return True
        first=[3,7];cache.run('same',first,0,probe,lambda:None,stats)
        second=[1,6];self.assertTrue(cache.run('same',second,0,probe,lambda:None,stats))
        self.assertEqual(second,[1,2]);self.assertEqual(len(calls),2)
        impossible=[1,4];self.assertFalse(cache.run('same',impossible,0,probe,lambda:None,stats))
        self.assertEqual(stats['lookahead_subset_hits'],2)
        fresh=[1,6];cache.run('different',fresh,0,probe,lambda:None,stats)
        self.assertEqual(len(calls),3)

    def test_failed_wider_probe_reused_only_under_narrowing(self):
        cache=CompletedProbeCache();stats={};calls=[]
        def probe(domains,cell):calls.append(1);return False
        for row in ([3,7],[1,3]):self.assertFalse(cache.run('same',list(row),0,probe,lambda:None,stats))
        self.assertEqual(len(calls),1)
        cache.run('same',[7,7],0,probe,lambda:None,stats);self.assertEqual(len(calls),2)

    def test_group_union_over_wide_domains_is_exact_on_finite_relations(self):
        rng=random.Random(2312)
        for _ in range(40):
            allowed={row for row in itertools.product(range(8),range(4),range(3)) if rng.randrange(4)}
            original=[255,15,7];domains=original.copy();stats={}
            def probe(trial,cell):
                rows=[row for row in allowed if trial[cell]>>row[cell]&1]
                if not rows:return False
                for c in range(3):trial[c]=sum(1<<v for v in {row[c] for row in rows})
                return True
            grouped_constructive_lookahead(domains,0,[15,240],probe,lambda:None,lambda mask:None,stats)
            expected=[sum(1<<v for v in {row[c] for row in allowed}) for c in range(3)]
            self.assertEqual(domains,expected);self.assertEqual(stats['group_lookahead_probes'],2)

    def test_group_interruption_never_commits_incomplete_union(self):
        domains=[255,7];stats={};failed=[]
        def probe(trial,cell):
            if trial[cell]==15:return False
            trial[1]=1;raise InterruptedError
        with self.assertRaises(InterruptedError):
            grouped_constructive_lookahead(domains,0,[15,240],probe,lambda:None,failed.append,stats)
        self.assertEqual(domains,[240,7]);self.assertEqual(failed,[15])
        self.assertEqual(stats['group_lookahead_complete'],0)

    def test_invalid_group_partitions_never_run(self):
        for groups in ([3,3],[3,4],[1,2,4],[-1,16],[True,14]):
            with self.assertRaises(ValueError):
                grouped_constructive_lookahead([15,7],0,groups,lambda *_:self.fail('ran'),lambda:None,lambda _:None,{})


class ClauseCompletionTests(unittest.TestCase):
    def test_resolution_can_eliminate_an_unfixed_pivot(self):
        store=LCGStore('fixture','gold',[3,3,3])
        store.learn(clause(store,[(0,1),(1,1)]));store.learn(clause(store,[(0,2),(2,1)]))
        self.assertEqual(store.resolve(),1);self.assertIn(((1,1),(2,1)),store.clauses)
        domains=[3,1,3];store.prune(domains,lambda:None)
        self.assertEqual(domains[2],2)

    def test_noncovering_masks_do_not_resolve(self):
        store=LCGStore('fixture','gold',[7,3,3])
        store.learn(clause(store,[(0,1),(1,1)]));store.learn(clause(store,[(0,2),(2,1)]))
        self.assertEqual(store.resolve(),0)
        self.assertEqual(len(store.clauses),2)

    def test_vivification_drops_a_redundant_literal_by_completed_replay(self):
        store=LCGStore('fixture','gold',[3]*3)
        store.learn(clause(store,[(0,1),(1,1),(2,1)]))
        store.learn(clause(store,[(0,1),(1,2)]))
        self.assertGreater(store.strengthen(max_tests=2),0)
        self.assertIn(((0,1),(2,1)),store.clauses)
        self.assertGreater(store.stats['lcg_vivification_literals_saved'],0)

    def test_finite_truth_tables_preserved_under_both_rules(self):
        rng=random.Random(2308)
        for _ in range(120):
            store=LCGStore('fixture','gold',[7]*4)
            for attempt in range(8):
                cells=rng.sample(range(4),rng.randrange(1,5))
                store.learn(clause(store,[(c,rng.randrange(1,7)) for c in cells]))
            expected={row for row in itertools.product(range(3),repeat=4)
                      if not any(all(mask>>row[c]&1 for c,mask in terms) for terms in store.clauses)}
            store.resolve(max_pairs=64,max_new=16);store.strengthen(max_tests=32)
            observed={row for row in itertools.product(range(3),repeat=4)
                      if not any(all(mask>>row[c]&1 for c,mask in terms) for terms in store.clauses)}
            self.assertEqual(observed,expected)

    def test_incomplete_replay_and_interruption_cannot_add_clause(self):
        store=LCGStore('fixture','gold',[3]*3)
        store.learn(clause(store,[(0,1),(1,1),(2,1)]));before=store.clauses.copy()
        self.assertEqual(store.strengthen(max_tests=1,max_steps=1),0);self.assertEqual(before,store.clauses)
        def stop():raise InterruptedError
        with self.assertRaises(InterruptedError):store.resolve(stop)
        self.assertEqual(before,store.clauses)


class SatReuseTests(unittest.TestCase):
    def helper(self,*,native=False,factory=None,initial=None):
        problem,domains=fixture();initial=initial or [sum(1<<code for code in row) for row in problem.domains]
        helper=SatAssist(problem,initial,**({} if native else {'factory':factory or (lambda:GoldFixtureOracle(problem,initial))}))
        self.addCleanup(helper.close);self.assertTrue(helper.prepare(seconds=3));return problem,domains,helper

    def test_resume_one_candidate_at_a_time_reaches_same_supports_without_replay(self):
        problem,domains,helper=self.helper()
        for _ in range(60):filtered=helper.filter_domains(domains,seconds=1,max_candidates=1)
        self.assertEqual(filtered,(3460,1737,1177,301))
        self.assertLessEqual(helper.stats['sat_assist_domain_calls'],sum(mask.bit_count() for mask in domains))
        self.assertGreater(helper.stats['sat_assist_domain_resumes'],0)
        calls=helper.stats['sat_assist_domain_calls']
        helper.filter_domains(domains,seconds=1,max_candidates=48)
        self.assertEqual(helper.stats['sat_assist_domain_calls'],calls)
        self.assertEqual(admitted(brute(problem,'edges'),domains),admitted(brute(problem,'edges'),filtered))

    def test_mask_cores_are_valid_in_original_scope_and_preserve_widened_sibling(self):
        problem,domains,helper=self.helper()
        helper.filter_domains(domains,seconds=1,max_candidates=48)
        cores=helper.drain_domain_candidates();self.assertTrue(cores)
        solutions=brute(problem,'edges')
        for core in cores:self.assertFalse(any(all(mask>>row[c]&1 for c,mask in core) for row in solutions))
        widened=helper.filter_domains(helper.initial,seconds=1,max_candidates=48)
        self.assertEqual(admitted(solutions,helper.initial),admitted(solutions,widened))
        self.assertFalse(helper.drain_candidates())

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_actual_model_witnesses_skip_redundant_candidate_queries(self):
        problem,domains,helper=self.helper(native=True)
        filtered=helper.filter_domains(domains,seconds=2,max_candidates=48)
        self.assertEqual(filtered,(3460,1737,1177,301))
        self.assertGreater(helper.stats['sat_assist_model_supports'],0)
        self.assertLess(helper.stats['sat_assist_domain_calls'],sum(mask.bit_count() for mask in domains))
        self.assertEqual(helper.stats['sat_assist_model_invalid'],0)

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_native_mask_core_rejects_only_proved_conjunction(self):
        problem,domains,helper=self.helper(native=True)
        record=[(0,3),(1,3)] # both cells would consume physical tile zero
        core=helper.query_masks(record,seconds=1,conflicts=10000)
        self.assertIsNotNone(core);self.assertEqual(helper.last_outcome,'unsat')
        solutions=brute(problem,'edges')
        self.assertFalse(any(all(mask>>row[c]&1 for c,mask in core) for row in solutions))
        calls=helper.stats['sat_assist_mask_calls'];helper.query_masks([(0,1),(1,1)],seconds=1)
        self.assertEqual(helper.stats['sat_assist_mask_calls'],calls)


class MaskPersistenceTests(unittest.TestCase):
    def helper(self,initial=None,factory=None):
        problem,_=fixture();root=initial or [sum(1<<c for c in row) for row in problem.domains]
        helper=VerifiedMaskConflictStore(problem,initial=initial,factory=factory or (lambda:GoldFixtureOracle(problem,root)))
        self.addCleanup(helper.close);return helper

    def test_cross_worker_claim_reproved_and_scoped_units_work(self):
        first=self.helper();first.offer([(0,3),(1,3)])
        self.assertEqual(first.advance(seconds=1),1)
        second=self.helper();self.assertEqual(second.import_snapshot(first.snapshot()),1)
        self.assertEqual(second.records(),());self.assertEqual(second.pending_count,1)
        second.advance(seconds=1);self.assertEqual(len(second.records()),1)
        domains=list(second.initial);domains[0]=3
        self.assertTrue(second.prune(domains));self.assertFalse(domains[1]&3)
        self.assertGreater(second.stats['mask_conflicts_prunes'],0)

    def test_persisted_local_claims_cannot_escape_job_root(self):
        original=self.helper();initial=list(original.initial);initial[0]=3
        job=self.helper(initial=initial);job.offer([(1,3)]);job.advance(seconds=1)
        self.assertTrue(job.records())
        with self.assertRaisesRegex(ValueError,'effective root'):original.import_snapshot(job.snapshot())
        same=self.helper(initial=initial)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'proofs.json';job.save(path);same.load(path)
        self.assertFalse(same.records());same.advance(seconds=1);self.assertTrue(same.records())
        # Same literal is NOT a theorem under original domains.
        original.offer([(1,3)]);original.advance(seconds=1)
        self.assertFalse(original.records());self.assertGreater(original.stats['mask_conflicts_rejected'],0)

    def test_forged_claims_and_unknown_never_prune(self):
        helper=self.helper();helper.offer([(0,3)])
        other=self.helper();payload=helper.snapshot();other.import_snapshot(payload)
        other.advance(seconds=1);self.assertFalse(other.records())
        unknown=self.helper(factory=Fake);unknown.offer([(0,3),(1,3)])
        unknown.advance(seconds=.01);self.assertFalse(unknown.records())
        self.assertEqual(unknown.pending_count,1)
        domains=list(unknown.initial);self.assertFalse(unknown.prune(domains))
        payload['target']='edges'
        with self.assertRaises(ValueError):other.import_snapshot(payload)


class DeductionSchedulerTests(unittest.TestCase):
    def test_useful_proofs_without_domain_changes_reset_cooldown(self):
        clock=iter(range(0,100000000,1000000));scheduler=PropagationScheduler(incremental=False,clock=lambda:next(clock))
        stats=dict.fromkeys(STAT_KEYS,0);proofs=[0]
        for _ in range(3):scheduler.run('p',[7],lambda:None,stats,optional=True)
        self.assertGreater(scheduler.records['p']['cooldown'],0)
        scheduler.run('p',[7],lambda:proofs.__setitem__(0,proofs[0]+1),stats,optional=True,force=True,proof_yield=lambda:proofs[0])
        self.assertEqual(scheduler.records['p']['cooldown'],0)
        self.assertEqual(stats['propagation_proofs_produced'],1)

    def test_budgets_respond_to_observed_deductions(self):
        scheduler=PropagationScheduler(incremental=False);stats={}
        for _ in range(4):scheduler.run('barren',[7],lambda:None,stats)
        for _ in range(4):
            domains=[7];scheduler.run('productive',domains,lambda:domains.__setitem__(0,1),stats)
        self.assertEqual(scheduler.budget('barren',.1),.05)
        self.assertEqual(scheduler.budget('productive',.1),.2)
        self.assertEqual(stats['propagation_values_removed'],8)
