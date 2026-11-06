"""Scoped CPU assistance, endgame ownership, and connectivity-aware heuristics."""
import json
import os
import random
import unittest
from unittest.mock import patch
import numpy as np
from gpu_cpu_assist import CPUAssist
from gpu_reasoning import prepare_policy
from dfs_gpu import DFS, fill_order
from test_gpu_reasoning import varied_gold_fixture
from test_gpu_gold import gold_fixture
from test_replica_resize import host_dfs
from systematic_jobs import JobLedger
from validator import orientation_masks
import gpu_sampling
import constraint_lns


def assisted_host(data,n=1,features=None):
    from geometry import build_board
    board=build_board();gpu=host_dfs(n)
    gpu._prepare_host(np.asarray(orientation_masks(data),np.uint16),board.neighbor_cells,board.neighbor_sides,n,None,None,[])
    gpu._prepare_gold(data);gpu._prepare_tables()
    prepare_policy(gpu,features or {'cpu_endgame':True},data)
    gpu._upload_policy=lambda:None
    return gpu


class HybridHostTests(unittest.TestCase):
    def test_cpu_propagation_scope_unknown_and_reuse(self):
        data,board=varied_gold_fixture();service=CPUAssist(data,'fixture')
        assignment=list(range(0,480,3));assignment[-3:]=[-1]*3
        before=assignment.copy()
        self.assertEqual(service.probe(assignment,seconds=0)['status'],'unknown')
        proof=service.probe(assignment,seconds=10)
        self.assertEqual(proof['status'],'propagated');self.assertGreater(proof['removed'],0)
        for cell,mask in enumerate(proof['domains']):self.assertTrue(mask&(1<<(3*cell)))
        self.assertEqual(service.probe(assignment,seconds=0),proof)
        self.assertEqual(assignment,before)
        loops,_=gold_fixture(False);closed=CPUAssist(loops,'other')
        a,_,b,_=board.edges[0];partial=[-1]*160;partial[a]=3*a;partial[b]=3*b
        self.assertEqual(closed.probe(partial,seconds=1)['status'],'infeasible')

    def test_endgame_leaves_just_completed_gpu_board_for_pending_dispatch(self):
        data,board=gold_fixture();gpu=assisted_host(data,features={'cpu_endgame':True,'cpu_probe':True})
        gpu.load_prefixes([list(3*gpu.order)])
        # This is the real boundary between placing the last cell and the next
        # shader iteration setting state1. There is no frame160 to transfer.
        gpu.states[0]=0;gpu.cpu_assist.interval=1
        self.assertEqual(int(gpu.depths.get()[0]),160)
        self.assertEqual(gpu.cpu_assist.before_step(gpu),[])
        self.assertFalse(gpu.cpu_assist.entries)
        self.assertEqual(gpu.cpu_assist.stats['cpu_prefix_probes'],0)
        self.assertEqual(int(gpu.states.get()[0]),0)

    def test_cpu_matching_proves_a_shortage_beyond_small_gpu_hall_sets(self):
        data,board=gold_fixture();holes=[]
        for cell in range(160):
            if all(cell not in board.neighbor_cells[other] and not set(board.neighbor_cells[cell])&set(board.neighbor_cells[other]) for other in holes):holes.append(cell)
            if len(holes)==5:break
        self.assertEqual(len(holes),5)
        # Removing cells from one existing loop cannot create a proper loop.
        # Four available pieces keep mask80 on each side; the fifth has mask136
        # and cannot occupy any hole against the fixed surrounding signatures.
        tile=data['tiles'][holes[-1]]
        tile['segments']=[[[side,{5:4,7:8}[position]] for side,position in segment]
                          for segment in tile['segments']]
        service=CPUAssist(data,'five-cell-Hall')
        assignment=[-1 if cell in holes else 3*cell for cell in range(160)]
        self.assertFalse(service.problem.partial_loop_impossible(assignment))
        # Every hole has four physical-piece supports, so availability and
        # pair/triple Hall shortages alone cannot reject this initial state.
        available=[3*piece+rotation for piece in holes for rotation in range(3)]
        for cell in holes:
            supported={code//3 for code in available if all(
                int(service.problem.masks[code,side])==80 for side in range(3))}
            self.assertEqual(supported,set(holes[:4]))
        result=service.probe(assignment,seconds=10)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(service.stats['cpu_prefix_failures'],1)

    def test_endgame_freeze_resume_resize_and_pending_certificate(self):
        data,board=varied_gold_fixture();order=fill_order(board.neighbor_cells)
        features={'cpu_endgame':True,'dynamic_mrv':True}
        gpu=assisted_host(data,2,features);gpu.cpu_assist.interval=1
        # Two disjoint tiny tails. The first is the known Gold arrangement;
        # changing a root rotation would mismatch seams, so use a second empty
        # pool lane and exercise resize while the CPU owns the only active job.
        rows=np.full((1,160),-1,np.int16);rows[0,:157]=(3*order[:157])
        lengths=np.array([157],np.int16);ledger=JobLedger.empty(1,2);ledger.refill(gpu,rows,lengths)
        held=gpu._pause_cpu_owned();self.assertEqual(held,[(0,0)]);gpu._restore_cpu_owned(held)
        self.assertEqual(int(gpu.depths.get()[0]),157)
        # A zero-progress exact slice must retain a real resumable checkpoint.
        gpu.cpu_assist.seconds=1e-9
        held=gpu._pause_cpu_owned();gpu._restore_cpu_owned(held)
        self.assertEqual(len(gpu.cpu_assist.entries),1)
        entry=next(iter(gpu.cpu_assist.entries.values()));self.assertIn('checkpoint',entry)
        saved=gpu.snapshot_checkpoint();saved.update(ledger.fields())
        malformed=dict(saved);state=json.loads(str(saved['cpu_assist_state']));state['entries'][0]['owner_prefix']=[479]
        malformed['cpu_assist_state']=np.array(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'orphaned'):
            JobLedger.resume(assisted_host(data,1,features),malformed,rows,lengths)
        restored=assisted_host(data,1,features)
        JobLedger.resume(restored,saved,rows,lengths)
        self.assertEqual(len(restored.cpu_assist.entries),1)
        next(iter(restored.cpu_assist.entries.values()))['slice_seconds']=2
        for _ in range(10):
            held=restored._pause_cpu_owned();restored._restore_cpu_owned(held)
            if int(restored.states.get()[0])==1:break
        else:self.fail('Tiny CPU endgame did not finish')
        restored.verify()
        self.assertTrue(restored.cpu_assist.problem.validate(restored.boards[:,0].get().astype(int).tolist())['valid'])
        self.assertEqual(int(restored.depths.get()[0]),160)
        self.assertEqual(len(restored.drain_cpu_candidates()),1)
        self.assertFalse(restored.cpu_assist.entries)
        # The ordinary pending checkpoint is self-contained and readable.
        resumed=assisted_host(data,1,features);resumed.restore_checkpoint(restored.snapshot_checkpoint());resumed.verify()
        self.assertEqual(int(resumed.states.get()[0]),1)

    def test_cpu_owned_paused_job_survives_shrink_retire_and_refill(self):
        data,board=gold_fixture();order=fill_order(board.neighbor_cells)
        gpu=assisted_host(data,2,{'cpu_endgame':True});gpu.cpu_assist.interval=1
        rows=np.full((2,160),-1,np.int16)
        rows[0,:157]=3*order[:157];rows[1,:158]=3*order[:158];rows[1,0]+=1
        lengths=np.array([157,158],np.int16)
        ledger=JobLedger.empty(2,2);ledger.refill(gpu,rows,lengths)
        held=gpu._pause_cpu_owned();gpu._restore_cpu_owned(held)
        self.assertEqual(held,[(1,0)])
        gpu.cpu_assist.seconds=1e-9
        held=gpu._pause_cpu_owned();gpu._restore_cpu_owned(held)
        saved=gpu.snapshot_checkpoint();saved.update(ledger.fields())
        smaller=assisted_host(data,1,{'cpu_endgame':True})
        resized=JobLedger.resume(smaller,saved,rows,lengths)
        self.assertEqual(resized.paused_count,1)
        entry=next(iter(smaller.cpu_assist.entries.values()))
        self.assertEqual(entry['owner_prefix'],rows[1,:158].astype(int).tolist())
        self.assertIsNone(smaller.cpu_assist._entry_lane(smaller,entry))
        # Simulate a completed proof for the other active job. This test checks
        # ownership bookkeeping; it makes no feasibility claim about that job.
        smaller.reject_current_subtree(0,smaller.boards[:,0].get().astype(int).tolist())
        resized.retire(smaller.states.get());resized.refill(smaller,rows,lengths)
        self.assertEqual(resized.paused_count,0)
        self.assertEqual(smaller.cpu_assist._entry_lane(smaller,entry),0)
        smaller.verify();resized.validate(smaller.states.get())
        replay=smaller.snapshot_checkpoint();replay.update(resized.fields())
        larger=assisted_host(data,3,{'cpu_endgame':True})
        restored=JobLedger.resume(larger,replay,rows,lengths)
        self.assertEqual(len(larger.cpu_assist.entries),1)
        self.assertEqual(larger.cpu_assist._entry_lane(larger,next(iter(larger.cpu_assist.entries.values()))),0)
        larger.verify();restored.validate(larger.states.get())

    def test_forced_current_frame_cannot_shift_away_from_its_unique_value(self):
        from gpu_reasoning import pack_frame
        data,board=varied_gold_fixture();gpu=assisted_host(data,features={'forced':True})
        for depth in range(157,160):
            gpu=assisted_host(data,features={'forced':True});gpu.load_prefixes([list(3*gpu.order[:depth])])
            cell=int(gpu.order[depth]);partial=gpu.boards[:,0].get()
            known=[side for side in range(3) if partial[int(gpu.neighbors[cell,side])]>=0]
            needs=[int(gpu.host_reverse[partial[int(gpu.neighbors[cell,side])],int(gpu.neighbor_sides[cell,side])]) for side in known[:2]]
            if len(known)==1:
                key=known[0]*gpu.colors+needs[0];first=int(gpu.host_single_offsets[key]);length=int(gpu.host_single_offsets[key+1])-first
            else:
                pair={(0,1):0,(0,2):1,(1,2):2}[tuple(known[:2])];key=(pair*gpu.colors+needs[0])*gpu.colors+needs[1]
                first=int(gpu.host_pair_offsets[key]);length=int(gpu.host_pair_offsets[key+1])-first
            if length<=1:continue
            shift=int(np.flatnonzero(gpu.host_pool[first:first+length]==3*cell)[0])
            pf,ps=pack_frame(first,shift,cell)
            gpu.firsts[depth,0]=pf;gpu.shifts[depth,0]=ps;gpu.lengths[depth,0]=length|32768;gpu.cursors[depth,0]=0
            gpu.verify()
            saved=gpu.snapshot_checkpoint();saved={key:value.copy() for key,value in saved.items()}
            saved['shifts'][depth,0]=pack_frame(first,(shift+1)%length,cell)[1]
            with self.assertRaisesRegex(ValueError,'saved shift'):gpu.read_checkpoint(saved)
            break
        else:self.fail('Fixture lacks a nontrivial forced bucket')

    def test_assistance_rejects_foreign_geometry_and_unbounded_resume_slice(self):
        from gpu_reasoning import prepare_policy
        data,board=gold_fixture();engine=assisted_host(data)
        permutation=np.roll(np.arange(160),1);inverse=np.argsort(permutation)
        engine.neighbors=permutation[engine.neighbors[inverse]]
        engine.neighbor_sides=engine.neighbor_sides[inverse]
        with self.assertRaisesRegex(ValueError,'geometry differs'):
            prepare_policy(engine,{'cpu_probe':True},data)
        service=CPUAssist(data,'scope')
        partial=list(range(0,480,3));partial[-1]=-1
        for amount in (float('inf'),float('nan'),-1,0,1000,True):
            state={'version':1,'fingerprint':'scope','entries':[{'assignment':partial,'owner_prefix':[], 'slice_seconds':amount}]}
            with self.assertRaisesRegex(ValueError,'bounded scheduling'):service.restore(state)
        state['entries'][0]={'assignment':partial,'owner_prefix':[],'solved':list(range(0,480,3))}
        with self.assertRaisesRegex(ValueError,'ordinary pending'):service.restore(state)

    def test_cpu_rejection_keeps_ancestors_and_rejects_changed_scope(self):
        data,board=varied_gold_fixture();gpu=assisted_host(data)
        gpu.load_prefixes([list(3*gpu.order[:158])])
        original=gpu.boards[:,0].get().astype(int).tolist()
        wrong=original.copy();wrong[int(gpu.order[0])]=-1
        with self.assertRaisesRegex(ValueError,'no longer'):gpu.reject_current_subtree(0,wrong)
        gpu.reject_current_subtree(0,original);gpu.verify();self.assertEqual(int(gpu.states.get()[0]),2)

    def test_gold_objective_distinguishes_equal_seams(self):
        data,board=gold_fixture()
        from search_problem import SearchProblem
        problem=SearchProblem(data)
        codes=list(range(0,480,3));good=gpu_sampling.gold_score(problem,codes)
        self.assertEqual(good['closed_proper_loops'],0)
        broken=next((trial for rot in range(1,3) for cell in range(160)
                     if (trial:=codes[:cell]+[codes[cell]+rot]+codes[cell+1:]) and problem.partial_loop_impossible(trial)),None)
        self.assertIsNotNone(broken)
        bad=gpu_sampling.gold_score(problem,broken)
        self.assertEqual(good['matched_edges'],bad['matched_edges'])
        self.assertGreater(good['gold_objective'],bad['gold_objective'])
        archive=constraint_lns.diverse_archive([{'codes':broken,**bad},{'codes':codes,**good}],rank_key='gold_objective')
        self.assertEqual(archive[0]['codes'],codes)

    def test_dependency_chain_can_import_a_remote_piece(self):
        from search_problem import SearchProblem
        data,board=varied_gold_fixture();problem=SearchProblem(data)
        current=list(range(0,480,3));current[0],current[159]=current[159],current[0]
        metrics=constraint_lns.score(problem,current)
        geometric=constraint_lns._select_patch(problem,current,metrics,2,random.Random(3),'gold')
        dependent=constraint_lns._select_patch(problem,current,metrics,2,random.Random(3),'gold',dependency_chains=True)
        self.assertEqual(geometric,[0,64]);self.assertEqual(dependent,[0,159])
        self.assertNotIn(159,problem.neighbors[0])
        import importlib.util
        if importlib.util.find_spec('ortools') is None:self.skipTest('OR-Tools optional repair dependency unavailable')
        candidate,status=constraint_lns._solve_patch(problem,current,dependent,cp_model=constraint_lns._load_cp_model(),
            control=constraint_lns._Control(5,None),seconds=3,seed=3,target='gold')
        self.assertIsNotNone(candidate);self.assertTrue(problem.validate(candidate)['valid'])
        self.assertEqual(metrics['matched_edges'],234)


class SamplingGoldMixin:
    def test_gpu_gold_scores_match_independent_components_after_many_moves(self):
        from search_problem import SearchProblem
        from test_constraint_dfs import multiple_loops,double_face,tetrahedron_tiles
        fixtures=[(multiple_loops(),double_face()),tetrahedron_tiles()]
        for data,board in fixtures:
            problem=SearchProblem(data,board)
            table,free=gpu_sampling._inputs(problem,gold_aware=True)
            sampler=gpu_sampling._make_sampler(problem,table,free,16,47,self.backend,allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
            try:
                for steps in (0,1,17,64):
                    if steps:sampler.launch(steps)
                    boards,scores=sampler.top(16)
                    for codes,value in zip(boards,scores):self.assertEqual(gpu_sampling.gold_score(problem,codes)['gold_objective'],int(value))
            finally:sampler.close()


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','CUDA hardware opt-in')
class GoldSamplingCUDA(SamplingGoldMixin,unittest.TestCase):backend='cuda'


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU hardware opt-in')
class GoldSamplingWebGPU(SamplingGoldMixin,unittest.TestCase):backend='webgpu'


if __name__=='__main__':unittest.main()
