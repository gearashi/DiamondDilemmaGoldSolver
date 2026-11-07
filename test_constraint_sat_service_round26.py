from test_constraint_sat_assist import HAS_SAT
import copy,itertools,json,multiprocessing,tempfile,threading,time,unittest
from pathlib import Path
from constraint_witness_round26 import SemanticWitnessPool
from constraint_sat_service_round26 import SATServiceOwner,SATServiceClient,_Backend
from constraint_sat_assist import STAT_KEYS
from constraint_knowledge import KnowledgeStore
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,multiple_loops,brute
from test_constraint_separator import ring


def query_child(client,queue):
    try:
        result=[]
        for _ in range(8):result.append(client.query([(0,0),(1,0)],seconds=.2))
        queue.put((result,client.last_outcome))
    except BaseException as exc:queue.put(('error',repr(exc)))
    finally:client.close()


class SemanticWitnessTests(unittest.TestCase):
    def test_index_supports_equal_complete_model_scan_across_contexts_and_cuts(self):
        problem=SearchProblem(multiple_loops(),double_face());boards=brute(problem,'edges');pool=SemanticWitnessPool(problem)
        for board in boards:pool.add(board)
        for left,right in itertools.product(range(1,64),repeat=2):
            domains=(left,right);supported=[0,0];pool.supports(domains,supported)
            expected=[0,0]
            for board in boards:
                if all(mask>>v&1 for mask,v in zip(domains,board)):
                    for c,v in enumerate(board):expected[c]|=1<<v
            self.assertEqual(supported,expected)
        cuts=[list(enumerate(board)) for board in boards[:3]];pool.add_cuts(cuts);supported=[0,0];pool.supports((63,63),supported)
        expected=[0,0]
        for board in boards:
            if board not in pool.blocked:
                for c,v in enumerate(board):expected[c]|=1<<v
        self.assertEqual(supported,expected);self.assertGreater(pool.stats['witness_cut_invalidations'],0)
        with self.assertRaises(ValueError):pool.add_cuts([[(0,0)]])

    def test_saved_boards_revalidated_and_forged_tail_is_atomic(self):
        problem=SearchProblem(simple_tiles(),double_face());boards=brute(problem,'edges');pool=SemanticWitnessPool(problem)
        for row in boards[:4]:pool.add(row)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'boards.json';pool.save(path);other=SemanticWitnessPool(problem);other.load(path)
            self.assertEqual(set(other.rows),set(pool.rows));self.assertGreaterEqual(other.stats['witness_semantic_validations'],4)
        bad=pool.snapshot();bad['boards'][-1]=[0,0]
        import hashlib
        bad['digest']=hashlib.sha256(json.dumps({k:v for k,v in bad.items() if k!='digest'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        other=SemanticWitnessPool(problem)
        with self.assertRaises(ValueError):other.import_snapshot(bad)
        self.assertFalse(other.rows)
        problem.tiles[0]['segments']=list(problem.tiles[0]['segments']);problem.tiles[0]['segments'][0]=((0,3),(1,8))
        with self.assertRaises(ValueError):pool.supports((63,63),[0,0])

    def test_verified_symmetry_and_bounded_coverage_retention(self):
        problem=ring(4);boards=brute(problem,'edges');pool=SemanticWitnessPool(problem,limit=2);pool.add(boards[0])
        self.assertTrue(pool.transport(boards[0],[1,2,3,0],[0]*4))
        for row in boards[1:8]:pool.add(row)
        self.assertEqual(len(pool.rows),2);self.assertGreater(pool.stats['witness_evictions'],0)
        with self.assertRaises(ValueError):pool.transport(boards[0],[1,0,2,3],[0]*4)
        with self.assertRaises(ValueError):pool.transport(boards[0],[1,2,3,0],[1,0,0,0])


class SharedSATServiceTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_two_spawned_workers_share_one_model_and_completed_query(self):
        problem=SearchProblem(simple_tiles(),double_face());context=multiprocessing.get_context('spawn')
        with SATServiceOwner(problem,'edges') as owner:
            first=owner.client();self.assertTrue(first.prepare(seconds=.2));first.close()
            queue=context.Queue();workers=[context.Process(target=query_child,args=(owner.client(),queue)) for _ in range(2)]
            for process in workers:process.start()
            try:
                results=[queue.get(timeout=15) for _ in workers]
                for rows,status in results:self.assertEqual(status,'unsat');self.assertEqual(rows,[((0,0),(1,0))]*8)
            finally:
                for process in workers:
                    process.join(timeout=10)
                    if process.is_alive():process.terminate();process.join()
                    self.assertEqual(process.exitcode,0)
                queue.close()
            counts=owner.counters()[0];self.assertEqual(counts['service']['sat_service_models'],1)
            self.assertEqual(counts['service']['sat_service_query_calls'],1);self.assertEqual(counts['service']['sat_service_cache_hits'],15)

    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_job_masks_batch_scope_and_unknown_do_not_become_conflicts(self):
        problem=SearchProblem(simple_tiles(),double_face());backend=_Backend(problem,'edges')
        try:
            response=backend.request([{'kind':'query','domains':(63,63),'max_candidates':16}],0)
            self.assertEqual(response['answers'][0]['outcome'],'unknown');self.assertFalse(backend.cache)
            with self.assertRaises(ValueError):backend.request([{'kind':'query','domains':(63,63),'max_candidates':16},{'kind':'query','domains':(64,63),'max_candidates':16}])
            self.assertEqual(backend.stats['sat_service_requests'],1)
            backend.request([{'kind':'prepare','domains':(63,63),'max_candidates':16}],.2)
            requests=[{'kind':'query','domains':(1<<a,1<<b),'max_candidates':16} for a,b in itertools.islice(itertools.product(range(6),repeat=2),16)]
            answers=backend.request(requests,.25)['answers'];solutions=set(brute(problem,'edges'))
            for request,answer in zip(requests,answers):
                board=tuple(mask.bit_length()-1 for mask in request['domains'])
                self.assertEqual(answer['outcome'],'sat' if board in solutions else 'unsat')
            client=SATServiceClient(backend,backend.identity(),(1,1));self.assertEqual(client.query([],seconds=.2),());client.close()
        finally:backend.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_views_preserve_scope_and_attribute_each_server_operation_once(self):
        problem=SearchProblem(simple_tiles(),double_face());backend=_Backend(problem,'edges')
        first=SATServiceClient(backend,backend.identity(),(63,63));second=first.view((1,63))
        try:
            self.assertTrue(first.matches(problem,'edges'))
            self.assertTrue(second.matches(problem,'edges',(1,63)))
            self.assertFalse(second.matches(problem,'gold'))
            self.assertFalse(second.matches(problem,'edges',(63,63)))
            for bad in ((3,63),(True,63),(1,-1)):
                with self.assertRaises(ValueError):second.view(bad)
            first.prepare(seconds=.2);first.query([(0,0),(1,0)],seconds=.2)
            second.query([(1,0)],seconds=.2)
            totals=backend.counters()
            for label,attribute in (('service','service_stats'),('witness','witness_stats')):
                for key,total in totals[label].items():
                    self.assertEqual(getattr(first,attribute)[key]+getattr(second,attribute)[key],total,(label,key))
            self.assertEqual(first.service_stats['sat_service_models'],1)
            self.assertEqual(second.service_stats['sat_service_models'],0)
            self.assertEqual(second.service_stats['sat_service_cache_hits'],1)
        finally:first.close();second.close();backend.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_gold_cut_rpc_is_pollable_and_pending_completion_is_reused(self):
        problem=SearchProblem(multiple_loops(),double_face());backend=_Backend(problem,'gold')
        entered=threading.Event();release=threading.Event()
        class Blocked:
            calls=0
            def add_cuts(self,records):
                self.calls+=1;entered.set()
                if not release.wait(5):raise RuntimeError('Test did not release Gold RPC.')
                return backend.add_cuts(records)
        proxy=Blocked();client=SATServiceClient(proxy,backend.identity(),backend.root);source=KnowledgeStore(problem)
        source.add(list(enumerate(brute(problem,'edges')[0])))
        def stop():
            if entered.is_set():release.set();raise InterruptedError('stop')
        try:
            with self.assertRaises(InterruptedError):client.add_gold_witnesses(source,stop)
            self.assertEqual(client.add_gold_witnesses(source),1);self.assertEqual(proxy.calls,1)
            self.assertEqual(client.stats['sat_assist_gold_verified'],1)
            self.assertEqual(client.service_stats['sat_service_models'],1)
        finally:release.set();client.close();backend.close()

    @staticmethod
    def response():
        delta=dict.fromkeys(STAT_KEYS,0);delta['sat_assist_calls']=1
        return {'binding':('p','edges',(3,3)),'answers':[{'outcome':'unsat','core':((0,1),)}],
                'ready':True,'status':'ready','reason':None,'sat_delta':delta,'clauses':3}

    def test_cancelled_client_retains_bounded_pending_result_but_never_returns_it_to_stop(self):
        entered=threading.Event();release=threading.Event();response=self.response()
        class Blocked:
            calls=0
            def request(self,requests,seconds,conflicts):
                self.calls+=1;entered.set()
                if not release.wait(5):raise RuntimeError('Test did not release query RPC.')
                return response
        proxy=Blocked();client=SATServiceClient(proxy,('p','edges',(3,3)),(3,3))
        def stop():
            if entered.is_set():release.set();raise InterruptedError('stop')
        try:
            with self.assertRaises(InterruptedError):client.query_masks([(0,1)],stop,seconds=.2)
            self.assertEqual(client.last_outcome,'not_ready');self.assertEqual(client.stats['sat_assist_calls'],0)
            self.assertEqual(client.query_masks([(0,1)],seconds=.2),((0,1),));self.assertEqual(proxy.calls,1)
            self.assertEqual(client.stats['sat_assist_calls'],1)
            client.close()
            with self.assertRaises(RuntimeError):client.prepare()
        finally:release.set();client.close()

    def completed_client(self, *, error=None):
        from concurrent.futures import Future
        from types import SimpleNamespace
        future=Future()
        if error is None:future.set_result(self.response())
        else:future.set_exception(error)
        client=SATServiceClient(SimpleNamespace(),('p','edges',(3,3)),(3,3))
        client._executor=SimpleNamespace(shutdown=lambda **kw:None) # No submit API: resubmission must fail.
        client._future=future
        request={'kind':'query','domains':(1,3),'max_candidates':16}
        client._key=json.dumps(['request',[request],.2,256],sort_keys=True)
        self.addCleanup(client.close)
        return client,future

    def test_stop_after_completed_result_keeps_future_and_unapplied_deltas(self):
        client,future=self.completed_client();checks=0
        def stop():
            nonlocal checks
            checks+=1
            if checks==3:raise InterruptedError('Stop at completed-result publication')
        with self.assertRaisesRegex(InterruptedError,'publication'):client.query_masks([(0,1)],stop,seconds=.2)
        self.assertIs(client._future,future);self.assertIsNotNone(client._key)
        self.assertEqual(client.stats['sat_assist_calls'],0);self.assertEqual(client.last_outcome,'not_ready')
        self.assertEqual(client.query_masks([(0,1)],seconds=.2),((0,1),))
        self.assertIsNone(client._future);self.assertEqual(client.stats['sat_assist_calls'],1)

    def test_completed_gold_cut_response_is_returned_once_after_stop(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        problem=SearchProblem(multiple_loops(),double_face());source=KnowledgeStore(problem)
        source.add(list(enumerate(brute(problem,'edges')[0])))
        binding=(problem.fingerprint,'gold',tuple(sum(1<<v for v in row) for row in problem.domains))
        client=SATServiceClient(SimpleNamespace(),binding,binding[2]);self.addCleanup(client.close)
        response=self.response();response.update(binding=binding,added=1)
        future=Future();future.set_result(response);client._future=future
        client._key=json.dumps(['add_cuts',source.records()],sort_keys=True)
        client._executor=SimpleNamespace(shutdown=lambda **kw:None)
        checks=0
        def stop():
            nonlocal checks
            checks+=1
            if checks==4:raise InterruptedError('completed cut publication')
        with self.assertRaisesRegex(InterruptedError,'cut publication'):client.add_gold_witnesses(source,stop)
        self.assertIs(client._future,future);self.assertEqual(client.stats['sat_assist_calls'],0)
        self.assertEqual(client.add_gold_witnesses(source),1)
        self.assertEqual(client.stats['sat_assist_calls'],1);self.assertIsNone(client._future)

    def test_rpc_timeout_error_is_not_polled_as_an_unfinished_wait(self):
        error=TimeoutError('RPC itself failed');client,_=self.completed_client(error=error)
        with self.assertRaisesRegex(TimeoutError,'RPC itself failed'):client.query_masks([(0,1)],seconds=.2)
        self.assertIsNone(client._future);self.assertIsNone(client._key)
        self.assertEqual(client.stats['sat_assist_calls'],0)



class PhysicalWitnessTransportTests(unittest.TestCase):
    def test_identical_raw_piece_transport_and_pairing_near_miss(self):
        from test_constraint_separator import ring
        from constraint_witness_round26 import SemanticWitnessPool
        problem=ring(4);pool=SemanticWitnessPool(problem);board=(0,3,6,9);pool.add(board)
        self.assertGreater(pool.expand_piece_symmetries(board,limit=2),0)
        for row in pool.rows:self.assertTrue(problem.validate(row)['valid'])
        self.assertTrue(pool.transport(board,range(4),(0,)*4,piece_map=(1,0,2,3))) if (3,0,6,9) not in pool.rows else None
        with self.assertRaises(ValueError):pool.transport(board,range(4),(0,)*4,piece_map=(1,0,2,3),piece_offsets=(1,0,0,0))
        problem.fixed[0]=0;problem.domains[0]=[0]
        with self.assertRaises(ValueError):pool.expand_piece_symmetries(board)

if __name__=='__main__':unittest.main()
