import json,os,sys,time,unittest
from unittest.mock import patch
from constraint_isolated_proof_round26 import IsolatedRUPProducer,ROOT
from constraint_proof_worker_round26 import identity,validate
from constraint_proof import MaskProofContext
from constraint_proof_indexed_round26 import IndexedRUPChecker
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face

FORMULA=[[1,2],[1,-2],[-1,2],[-1,-2]]

def await_proof(owner,formula,seconds=5,backend='g4'):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        proof=owner.produce_rup(formula,seconds=.03,backend=backend)
        if proof is not None:return proof
        if owner.stats['rejected']:return None
    return None

def fake(mode='valid',delay=0):
    code="""import json,sys,time,hashlib
from pathlib import Path
source=hashlib.sha256(Path('constraint_proof_worker_round26.py').read_bytes()).hexdigest()
print(json.dumps({'ready':1}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 time.sleep(DELAY)
 result={'version':1,'identity':request['identity'],'proof':['2 0','0'],'python_sat':'1.9.dev15','source':source,'decisions':2}
 if MODE=='wrong-root':result['identity']='wrong'
 if MODE=='wrong-source':result['source']='wrong'
 if MODE=='bad-proof':result['proof']=['0']
 print(json.dumps({'identity':request['identity'],'result':result}),flush=True)
""".replace('DELAY',repr(delay)).replace('MODE',repr(mode))
    return [sys.executable,'-B','-c',code]

class IsolatedProtocolTests(unittest.TestCase):
    def test_repeated_pending_queries_share_one_job_and_result(self):
        owner=IsolatedRUPProducer(fake(delay=.08));self.addCleanup(owner.close)
        self.assertIsNone(owner.produce_rup(FORMULA,seconds=.001))
        proof=await_proof(owner,FORMULA)
        self.assertEqual(proof,['2 0','0']);self.assertEqual(owner.stats['requests'],1)
        prior=owner.stats['ready_hits']
        self.assertEqual(owner.produce_rup(FORMULA,seconds=.01),proof)
        self.assertEqual(owner.stats['ready_hits'],prior+1)
        check=IndexedRUPChecker(FORMULA,proof)
        while not check.finished:check.advance()
        self.assertTrue(check.result)

    def test_cancellation_is_prompt_and_exact_request_can_finish_later(self):
        owner=IsolatedRUPProducer(fake(delay=.1));self.addCleanup(owner.close)
        owner.produce_rup(FORMULA,seconds=.001)
        def cancel():raise RuntimeError('stopped')
        started=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'stopped'):owner.produce_rup(FORMULA,cancel,seconds=2)
        self.assertLess(time.monotonic()-started,.1)
        self.assertIsNotNone(await_proof(owner,FORMULA));self.assertEqual(owner.stats['requests'],1)

    def test_wedged_start_and_reply_are_bounded_and_closed(self):
        for code,timeout in [("import time;time.sleep(5)",.1),
             ("import time;print('{\"ready\":1}',flush=True);time.sleep(5)",.5)]:
            owner=IsolatedRUPProducer([sys.executable,'-B','-c',code],startup_timeout=.1,reply_grace=.05)
            try:
                owner.produce_rup(FORMULA,seconds=.001)
                end=time.monotonic()+2
                while not owner.failed and time.monotonic()<end:time.sleep(.01)
                self.assertTrue(owner.failed)
                self.assertIsNone(owner.produce_rup(FORMULA,seconds=.001))
                if owner.process is not None:owner.process.wait(timeout=2);self.assertIsNotNone(owner.process.poll())
            finally:owner.close()

    def test_close_cancels_only_the_owned_optional_process(self):
        owner=IsolatedRUPProducer(fake(delay=5))
        owner.produce_rup(FORMULA,seconds=.03)
        end=time.monotonic()+2
        while owner.process is None and time.monotonic()<end:time.sleep(.01)
        started=time.monotonic();owner.close()
        self.assertLess(time.monotonic()-started,.3)
        if owner.process is not None:owner.process.wait(timeout=2);self.assertIsNotNone(owner.process.poll())
        self.assertIsNone(owner.produce_rup(FORMULA,seconds=.01))

    def test_wrong_input_or_source_never_returns_proof(self):
        for mode in ('wrong-root','wrong-source'):
            owner=IsolatedRUPProducer(fake(mode));self.addCleanup(owner.close)
            self.assertIsNone(await_proof(owner,FORMULA));self.assertGreater(owner.stats['rejected'],0)
        request={'version':1,'clauses':[[True]],'backend':'g4','seconds':.2,'conflicts':10}
        request['identity']=identity(request)
        with self.assertRaises(ValueError):validate(request)

    def test_valid_transport_does_not_replace_independent_proof_checking(self):
        problem=SearchProblem(simple_tiles(),double_face())
        context=MaskProofContext(problem,[63,63],indexed=True);context.isolated_producer=True
        owner=IsolatedRUPProducer(fake('bad-proof'));self.addCleanup(owner.close);self.addCleanup(context.close)
        with patch('constraint_isolated_proof_round26.provider',return_value=owner):
            for _ in range(20):
                answer=context.prove([],seconds=.05)
                self.assertIsNone(answer)
                if owner.stats['ready_hits']:break
            self.assertGreater(owner.stats['ready_hits'],0)
            self.assertGreater(context.stats['proof_checks'],0)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_ISOLATED_RUP')=='1','optional real isolated native producer')
class RealProducerTests(unittest.TestCase):
    def test_real_native_query_and_mask_proof_are_independently_verified(self):
        owner=IsolatedRUPProducer();self.addCleanup(owner.close)
        proof=await_proof(owner,FORMULA,seconds=25)
        self.assertIsNotNone(proof,owner.stats)
        checker=IndexedRUPChecker(FORMULA,proof)
        while not checker.finished:checker.advance()
        self.assertTrue(checker.result)
        problem=SearchProblem(simple_tiles(),double_face())
        context=MaskProofContext(problem,[63,63],indexed=True);context.isolated_producer=True
        self.addCleanup(context.close)
        with patch('constraint_isolated_proof_round26.provider',return_value=owner):
            end=time.monotonic()+15;answer=None
            while time.monotonic()<end and answer is None:answer=context.prove([(0,1),(1,1)],seconds=.05)
        self.assertIsNotNone(answer,owner.stats);self.assertTrue(context.verify(answer))
        self.assertEqual(owner.stats['requests'],2)

if __name__=='__main__':unittest.main()
