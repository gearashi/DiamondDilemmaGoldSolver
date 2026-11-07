import copy,itertools,json,tempfile,unittest
from unittest.mock import patch
from constraint_projection_certificate_round26 import ProjectionCertificates,_digest
from constraint_higher_relations import ProjectedRelations
from constraint_sat_assist import SatAssist
from constraint_proof import ProofLimit
from test_constraint_separator import ring
from test_constraint_dfs import brute
from test_constraint_sat_domains import fixture,admitted
from test_constraint_sat_assist import HAS_SAT

class ProjectionCertificateTests(unittest.TestCase):
    def context(self,n=3):
        problem=ring(n);root=tuple(sum(1<<c for c in row) for row in problem.domains);service=ProjectionCertificates(problem,root,'gold');self.addCleanup(service.close)
        witnesses=brute(problem,'edges');cells=tuple(range(n));rows=tuple(sorted(witnesses))
        return problem,root,cells,rows,service
    def test_independent_full_projection_and_adversarial_omissions(self):
        problem,root,cells,rows,service=self.context();cert=service.produce(root,cells,rows,rows,seconds=2)
        self.assertIsNotNone(cert);self.assertEqual(service.verify(json.loads(json.dumps(cert)))[1],rows)
        for edit in ('omit','witness','row','proof','scope','type'):
            bad=copy.deepcopy(cert)
            if edit=='omit':bad['rows'].pop();bad['witnesses'].pop()
            elif edit=='witness':bad['witnesses'][0]=bad['witnesses'][1]
            elif edit=='row':bad['rows'][0][0]=bad['rows'][0][1]
            elif edit=='proof':bad['proof']=[]
            elif edit=='scope':bad['scope'][0]|=1<<30
            else:bad['rows'][0][0]=True
            bad['digest']=_digest({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaises((ValueError,ProofLimit),msg=edit):service.verify(bad)
        self.assertIsNone(service.produce(root,cells,rows,rows,seconds=0))
        def stop():raise ProofLimit('external')
        with self.assertRaisesRegex(ProofLimit,'external'):service.produce(root,cells,rows,rows,check=stop,seconds=1)
    def test_durable_complete_import_rechecks_and_cancellation_is_not_corruption(self):
        problem,root,cells,rows,service=self.context();cert=service.produce(root,cells,rows,rows,seconds=2)
        with tempfile.TemporaryDirectory() as directory:
            service.configure_cache_dir(directory);self.assertTrue(service.save(cert))
            cold=ProjectionCertificates(problem,root,'gold');self.addCleanup(cold.close);cold.configure_cache_dir(directory)
            self.assertEqual(cold.load()[0][1],rows);self.assertGreater(cold.stats['relation_certificate_checker_visits'],0)
            def stop():raise ValueError('external callback')
            with self.assertRaisesRegex(ValueError,'external callback'):cold.load(stop)
            self.assertEqual(cold.store.stats['reasoning_cache_invalid'],0)
    def test_pure_certificate_path_does_not_import_pysat(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='pysat' or name.startswith('pysat.'):raise ImportError('optional native package deliberately absent')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            problem,root,cells,rows,service=self.context();cert=service.produce(root,cells,rows,rows,seconds=2)
            self.assertIsNotNone(cert);self.assertEqual(service.verify(cert)[1],rows)

    @unittest.skipUnless(HAS_SAT,'python-sat unavailable')
    def test_shared_gold_theory_growth_keeps_safe_cover_not_false_exactness(self):
        from test_constraint_separator import permutation_ring
        from constraint_knowledge import KnowledgeStore
        problem=permutation_ring([(0,1,2)]*3);root=tuple(sum(1<<c for c in row) for row in problem.domains)
        helper=SatAssist(problem,root,'gold');relation=ProjectedRelations(problem,root,'gold',certificates=True,guided=True)
        self.addCleanup(helper.close);self.addCleanup(relation.close);current=list(root)
        relation.prune(current,helper,seconds=2,max_calls=1,projection_cells=(0,1,2));session=next(iter(relation.sessions.values()));witness=next(iter(session['witnesses'].values()))
        store=KnowledgeStore(problem);store.add(problem.gold_nogood(witness));helper.add_gold_witnesses(store)
        self.assertTrue(any(all(witness[c]==v for c,v in cut) for cut in helper._gold_known))
        relation.prune(current,helper,seconds=2,max_calls=100,projection_cells=(0,1,2));self.assertTrue(relation.tables)
        key,rows=next(iter(relation.tables.items()));observed=relation.table_witnesses[key]
        cert=relation.certificates.produce(*key,rows,[observed[row] for row in rows],relation.table_gold[key],seconds=2)
        self.assertIsNotNone(cert);self.assertEqual(relation.certificates.verify(cert)[1],rows)
        # Weaker attempted theory with an omitted row is simply uncertifiable.
        self.assertIsNone(relation.certificates.produce(*key,rows[:-1],[observed[row] for row in rows[:-1]],(),seconds=2))

    @unittest.skipUnless(HAS_SAT,'python-sat unavailable')
    def test_actual_partial_enumeration_cold_resume_avoids_repeated_models(self):
        problem=ring(4);root=tuple(sum(1<<c for c in row) for row in problem.domains);solutions=brute(problem,'edges')
        with tempfile.TemporaryDirectory() as directory:
            helper=SatAssist(problem,root,'edges');first=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);first.configure_cache_dir(directory)
            self.addCleanup(helper.close);self.addCleanup(first.close)
            current=list(root);first.prune(current,helper,seconds=2,max_calls=3,projection_cells=(0,1,2),max_models=64);self.assertFalse(first.tables);self.assertEqual(first.stats['higher_models'],3);self.assertEqual(current,list(root));first.flush()
            nexthelper=SatAssist(problem,root,'edges');second=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);self.addCleanup(nexthelper.close);self.addCleanup(second.close);second.configure_cache_dir(directory)
            current=list(root);second.prune(current,nexthelper,seconds=2,max_calls=100,projection_cells=(0,1,2),max_models=64)
            self.assertTrue(second.tables);key,rows=next(iter(second.tables.items()));expected={tuple(r[c] for c in key[1]) for r in solutions}
            self.assertEqual(set(rows),expected);self.assertEqual(first.stats['higher_models']+second.stats['higher_models'],len(rows));self.assertEqual(second.certificates.stats['relation_observed_rows'],3)
            # Explicitly allow a longer bounded completeness proof for this
            # test; ordinary application slices remain short and unknown-safe.
            second.certificate_attempted.clear();self.assertTrue(second.persist_complete(seconds=2))
            third=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);self.addCleanup(third.close);third.configure_cache_dir(directory)
            self.assertEqual(third.tables,second.tables);self.assertEqual(third.stats['higher_queries'],0)
    @unittest.skipUnless(HAS_SAT,'python-sat unavailable')
    def test_partial_consequences_use_separate_queries_not_missing_rows(self):
        problem,domains=fixture();root=tuple(sum(1<<c for c in row) for row in problem.domains);solutions=brute(problem,'edges')
        helper=SatAssist(problem,root,'edges');self.addCleanup(helper.close);self.assertTrue(helper.prepare(seconds=2))
        relation=ProjectedRelations(problem,root,'edges',partial=True);self.addCleanup(relation.close)
        original=domains.copy();relation.prune(domains,helper,seconds=2,max_calls=1)
        self.assertFalse(relation.tables);self.assertGreater(relation.stats['higher_partial_queries'],0)
        # Public slices are intentionally only10ms; concurrent CI work can
        # leave them unknown. Finish bounded support work explicitly rather
        # than mistaking a scheduling-dependent short slice for a discriminator.
        for _ in range(8):
            if relation.stats['higher_partial_prunes']:break
            relation._partial_filter(domains,helper,lambda:None,2)
        self.assertGreater(relation.stats['higher_partial_prunes'],0,'Incomplete native support discriminator after eight explicit2s leases')
        self.assertEqual(admitted(solutions,domains),admitted(solutions,original))
        # No selector assumption is used by the ordinary query; a widened
        # sibling retains every original edge completion.
        widened=list(root);relation.prune(widened,helper,seconds=0);self.assertEqual(widened,list(root))

if __name__=='__main__':unittest.main()
