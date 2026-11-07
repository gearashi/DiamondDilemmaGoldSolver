"""Actual compiled guidance versus an independent unpacked proposal oracle."""
import json,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import gpu_sampling as sampling
from gpu_sampling_resume_round25 import SamplerSession
from search_problem import SearchProblem
from test_constraint_dfs import board,double_face,multiple_loops,simple_tiles,tetrahedron_tiles
from test_gpu_chains_reference_round25 import guided_cycle_reference

ROOT=Path(__file__).parent
GUIDED={'obstruction_chains':True}
CUDA_PROBE=r'''
extern "C" __global__ void guided_probe(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int count,int edges,int gold_total){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;
 unsigned random=m[lane];int cells[6],old[6],next[6];int before=total(t,b,n,r,f,edges,lane);
 guided_cycle(t,b,m,n,r,f,edges,gold_total,int(m[r+lane]),lane,random,count,cells,old,next);
 for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
 for(int i=0;i<count;i++)b[cells[i]*r+lane]=next[i];
 if(total(t,b,n,r,f,edges,lane)>before)m[21*r+lane]++;
 m[lane]=random;m[3*r+lane]=count;m[4*r+lane]=cells[0];
}
'''
WEBGPU_PROBE=r'''
@compute @workgroup_size(64)
fn guided_probe(@builtin(global_invocation_id) id:vec3<u32>){
 let lane=id.x;let r=p.replicas;if lane>=r{return;}random_state=lane_data[lane];
 var cells:array<u32,6>;var old:array<u32,6>;var next:array<u32,6>;let before=total(lane);
 guided_cycle(lane,p.steps,i32(lane_data[r+lane]),&cells,&old,&next);
 for(var c=0u;c<p.n;c++){best_boards[c*r+lane]=at(c,lane);}
 for(var i=0u;i<p.steps;i++){boards[cells[i]*r+lane]=next[i];}
 if total(lane)>before {lane_data[21u*r+lane]++;}
 lane_data[lane]=random_state;lane_data[3u*r+lane]=p.steps;lane_data[4u*r+lane]=cells[0];
}
'''

def fixtures():
    tiles,shape=tetrahedron_tiles()
    ring=SearchProblem([{'id':str(i),'segments':[[[0,2],[1,10]]]} for i in range(40)],board(40,[(i,1,(i+1)%40,0) for i in range(40)]))
    ring.domains[31]=[3*32]
    return [(SearchProblem(simple_tiles(),double_face(),fixed={0:0}),1),(SearchProblem(multiple_loops(),double_face()),2),
            (SearchProblem(tiles,shape,fixed={0:0}),3),(ring,6)]

class GuidedHostTests(unittest.TestCase):
    def test_direct_flag_implies_chain_and_rejects_unknown(self):
        self.assertEqual(sampling.sampling_flags(GUIDED),9)
        self.assertEqual(sampling.sampling_flags({'obstruction_chains':False,'move_tabu':True}),2)
        with self.assertRaises(ValueError):sampling.sampling_flags({'obstruction_chain':True})
    def test_legacy_hybrid_forwards_guidance_without_durable_mode(self):
        p=SearchProblem(multiple_loops(),double_face())
        final={'status':'infeasible','complete':True,'nodes':0,'stats':{},'elapsed_seconds':0.}
        with patch.object(sampling,'sample',return_value={'top_samples':[]}) as sample,patch('constraint_dfs.solve',return_value=final):
            sampling.solve(p,seconds=1,repair_fraction=0,options={'obstruction_chains':True})
        self.assertEqual(sample.call_args.kwargs['experiments'],GUIDED)
    def test_durable_policy_binds_guidance(self):
        p=SearchProblem(multiple_loops(),double_face())
        result=sampling.solve(p,seconds=1,stop=lambda:True,options={'durable_hybrid':True,'obstruction_chains':True})
        self.assertTrue(result['checkpoint']['sampling_policy']['experiments']['obstruction_chains'])
        with self.assertRaisesRegex(ValueError,'saved replicas/seed/policy'):
            sampling.solve(p,seconds=1,stop=lambda:True,resume=result['checkpoint'],options={'durable_hybrid':True})

class _Hardware:
    def engine(self,p,*,gold=True,flags=None,replicas=32):
        return sampling._make_sampler(p,*sampling._inputs(p,gold_aware=gold),replicas,2534,self.backend,experiments=GUIDED if flags is None else flags,
            allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def probe(self,engine,count):
        if self.backend=='cuda':
            kernel=engine.cp.RawKernel((ROOT/'sampling_round24.cu').read_text()+CUDA_PROBE,'guided_probe',options=('--std=c++17',))
            kernel(((engine.r+127)//128,),(128,),(engine.boards,engine.best,engine.meta,engine.table,*map(np.int32,(engine.n,engine.r,engine.f,count,engine.e,engine.gold_total))))
            engine.cp.cuda.get_current_stream().synchronize()
        else:
            module=engine.gpu.create_shader_module(code=(ROOT/'sampling_round24.wgsl').read_text()+WEBGPU_PROBE)
            pipeline=engine.gpu.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'guided_probe'})
            entries=[{'binding':i,'resource':{'buffer':b}} for i,b in enumerate(engine.buffers)]+[{'binding':4,'resource':{'buffer':engine.params}}]
            group=engine.gpu.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=entries)
            engine.queue.write_buffer(engine.params,0,np.array([engine.n,engine.r,engine.f,count,1,engine.e,engine.gold_total,engine.flags],np.uint32))
            encoder=engine.gpu.create_command_encoder();compute=encoder.begin_compute_pass();compute.set_pipeline(pipeline);compute.set_bind_group(0,group);compute.dispatch_workgroups((engine.r+63)//64);compute.end();engine.queue.submit([encoder.finish()])
    def test_compiled_helper_matches_independent_unpacked_reference(self):
        for p,count in fixtures():
            for gold in (False,True):
                with self.subTest(n=p.n,count=count,gold=gold):
                    engine=self.engine(p,gold=gold)
                    try:
                        before=engine.snapshot_arrays();free=[c for c in range(p.n) if c not in p.fixed]
                        expected=[guided_cycle_reference(p,list(map(int,before[0][:,lane])),free,int(before[2][0,lane]),count,gold_aware=gold) for lane in range(engine.r)]
                        self.probe(engine,count);after=engine.snapshot_arrays()
                        for lane,row in enumerate(expected):
                            self.assertEqual(after[0][:,lane].tolist(),row['board']);self.assertEqual(int(after[2][0,lane]),row['random_state'])
                            self.assertEqual(int(after[2][4,lane]),row['cells'][0])
                            for offset,key in enumerate(('guided_proposals','guided_candidate_checks','guided_obstruction_starts','guided_base_improvements')):
                                self.assertEqual(int(after[2][18+offset,lane]),row['counters'][key])
                    finally:engine.close()
    def test_full_moves_exchange_and_layouts_are_independently_scored(self):
        tiles,shape=tetrahedron_tiles();p=SearchProblem(tiles,shape,fixed={0:0})
        for flags in (GUIDED,{**GUIDED,'move_tabu':True},{**GUIDED,'replica_exchange':True},{**GUIDED,'move_tabu':True,'replica_exchange':True}):
            for gold in (False,True):
                engine=self.engine(p,gold=gold,flags=flags)
                try:
                    self.assertEqual(engine.meta_rows,22)
                    for steps in (1,7,64):
                        engine.launch(steps);engine.exchange(steps%2);current,best,meta=engine.snapshot_arrays()
                        for lane in range(engine.r):
                            scorer=sampling.gold_score if gold else sampling.score_board;field='gold_objective' if gold else 'objective'
                            self.assertEqual(int(meta[1,lane]),scorer(p,current[:,lane])[field]);self.assertEqual(int(meta[2,lane]),scorer(p,best[:,lane])[field])
                            self.assertEqual(int(current[0,lane]),0);self.assertGreaterEqual(meta[2,lane],meta[1,lane])
                    self.assertGreater(engine.experiment_stats()['guided_candidate_checks'],0)
                finally:engine.close()
    def test_full160_guided_scores_and_native_resume(self):
        from search_fixtures import planted_gold
        data,geometry,witness=planted_gold(2534,size=4,extra_edges=12);p=SearchProblem(data,geometry)
        self.assertEqual(p.n,160);self.assertTrue(p.validate(witness)['valid'])
        kw={'backend':self.backend,'replicas':4,'seed':2534,'gold_aware':True,
            'experiments':{**GUIDED,'move_tabu':True,'replica_exchange':True},'allow_software':os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1'}
        full=SamplerSession(p,**kw);prefix=tail=None
        try:
            full.advance(30,max_dispatches=4);expected=full.snapshot();arrays=full.engine.snapshot_arrays()
            for lane in range(4):
                self.assertEqual(int(arrays[2][1,lane]),sampling.gold_score(p,arrays[0][:,lane])['gold_objective'])
                self.assertEqual(int(arrays[2][2,lane]),sampling.gold_score(p,arrays[1][:,lane])['gold_objective'])
            prefix=SamplerSession(p,**kw);prefix.advance(30,max_dispatches=2);saved=json.loads(json.dumps(prefix.snapshot()));prefix.close();prefix=None
            tail=SamplerSession(p,resume=saved,**kw);tail.advance(30,max_dispatches=2);actual=tail.snapshot()
            self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['experiment_stats'],actual['experiment_stats'])
            self.assertGreater(actual['experiment_stats']['guided_candidate_checks'],0)
        finally:
            full.close()
            if prefix:prefix.close()
            if tail:tail.close()
    def test_durable_guidance_restores_all_counters_and_metadata(self):
        p=SearchProblem(multiple_loops(),double_face());kw={'backend':self.backend,'replicas':32,'seed':2534,'gold_aware':True,
            'experiments':{**GUIDED,'move_tabu':True,'replica_exchange':True},'allow_software':os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1'}
        first=SamplerSession(p,**kw);split=restored=None
        try:
            first.advance(30,max_dispatches=18);expected=first.snapshot()
            split=SamplerSession(p,**kw);split.advance(30,max_dispatches=9);saved=json.loads(json.dumps(split.snapshot()));split.close();split=None
            restored=SamplerSession(p,resume=saved,**kw);restored.advance(30,max_dispatches=9);actual=restored.snapshot()
            self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['archive'],actual['archive']);self.assertEqual(expected['experiment_stats'],actual['experiment_stats'])
        finally:
            first.close()
            if split:split.close()
            if restored:restored.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA hardware')
class CUDAChains(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','Optional native or explicit CI software WebGPU')
class WebGPUChains(_Hardware,unittest.TestCase):backend='webgpu'
if __name__=='__main__':unittest.main()
