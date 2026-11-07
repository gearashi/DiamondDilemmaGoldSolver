"""Opted-in hardware checks for experimental sampling transitions."""
import os
import unittest
import numpy as np
import gpu_sampling as sampling
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,double_face,multiple_loops

class _Hardware:
    def read(self,engine):
        if self.backend=='cuda':return engine.boards.get().T,engine.best.get().T,engine.meta.get()
        current=np.frombuffer(engine.queue.read_buffer(engine.buffers[0]),np.uint32).reshape(engine.n,engine.r).T.copy()
        best=np.frombuffer(engine.queue.read_buffer(engine.buffers[1]),np.uint32).reshape(engine.n,engine.r).T.copy()
        meta=np.frombuffer(engine.queue.read_buffer(engine.buffers[2]),np.uint32).reshape(engine.meta_rows,engine.r).copy()
        return current,best,meta
    def test_all_sampling_combinations_are_independently_scored(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board,fixed={0:0})
        for flags in range(1,8):
            switches={name:bool(flags&bit) for name,bit in [('ejection_chains',1),('move_tabu',2),('replica_exchange',4)]}
            for gold in (False,True):
                with self.subTest(flags=flags,gold=gold):
                    engine=sampling._make_sampler(problem,*sampling._inputs(problem,gold_aware=gold),16,440+self.backend.count('a'),self.backend,experiments=switches,
                        allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
                    try:
                        for steps in (1,7,64):
                            engine.launch(steps)
                            if flags&4:engine.exchange(steps%2)
                            current,best,meta=self.read(engine)
                            for lane in range(engine.r):
                                metric=sampling.gold_score if gold else sampling.score_board
                                field='gold_objective' if gold else 'objective'
                                self.assertEqual(int(meta[1,lane]),metric(problem,current[lane])[field])
                                self.assertEqual(int(meta[2,lane]),metric(problem,best[lane])[field])
                                self.assertEqual(current[lane,0],0);self.assertGreaterEqual(meta[2,lane],meta[1,lane])
                        stats=engine.experiment_stats()
                        if flags&1:self.assertGreater(stats['chain_proposals'],0)
                        if flags&4:self.assertGreater(stats['exchange_attempts'],0)
                    finally:engine.close()
    def test_equal_energy_exchange_swaps_current_boards_and_keeps_temperatures(self):
        problem=SearchProblem(multiple_loops(),double_face())
        engine=sampling._make_sampler(problem,*sampling._inputs(problem),8,443,self.backend,experiments={'replica_exchange':True},
            allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
        try:
            current=np.array([[0,3] if lane%2==0 else [3,0] for lane in range(8)],np.uint32)
            _,_,meta=self.read(engine);meta[1:3,:]=sampling.score_board(problem,current[0])['objective'];meta[3,:]=np.arange(8,dtype=np.uint32)
            if self.backend=='cuda':
                engine.boards[:]=engine.cp.asarray(current.T);engine.best[:]=engine.cp.asarray(current.T);engine.meta[:]=engine.cp.asarray(meta)
            else:
                engine.queue.write_buffer(engine.buffers[0],0,current.T.copy());engine.queue.write_buffer(engine.buffers[1],0,current.T.copy());engine.queue.write_buffer(engine.buffers[2],0,meta)
            engine.exchange(0);after,_,new=self.read(engine)
            np.testing.assert_array_equal(after,current[np.arange(8)^1]);np.testing.assert_array_equal(new[3],meta[3])
            self.assertEqual(engine.experiment_stats()['exchange_accepts'],4)
        finally:engine.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA hardware test')
class CUDASamplingRound24(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','Optional WebGPU hardware test')
class WebGPUSamplingRound24(_Hardware,unittest.TestCase):backend='webgpu'

if __name__=='__main__':unittest.main()
