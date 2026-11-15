"""Original signed pose frontiers, portable ownership, and exact native startup controls."""
from pathlib import Path
import json
import os
import unittest
from unittest.mock import patch

from constraint_search import run_solver
from constraint_work import WorkScope, frontier_snapshot,restore_frontier
from r35_pose_frontiers import FrontierPoses,prepare_frontier,validate_receiver
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


class SyntheticFrontiers(unittest.TestCase):
    def test_held_literal_implications_remain_correct_on_widened_siblings(self):
        from types import SimpleNamespace
        from itertools import product
        from r35_pose_departures import indexed_rows
        n=160
        image=[(c,1<<(3*c)) for c in range(n)]
        other=image.copy()
        other[0]=(2,1<<6)
        other[1]=(1,1<<4)
        other[2]=(0,1<<1)
        images=(tuple(image),tuple(other))
        receiver=FrontierPoses(SimpleNamespace(n=n))
        receiver.checked=True
        receiver.catalogue=(set().union(*images),images,indexed_rows(((0,1),)),n)
        full=(1<<(3*n))-1
        for a,b,c in product((1,2,3),(8,16,24),(64,128,192)):
            before=[a,b,c]+[full]*(n-3)
            after,facts=receiver.prune(before)
            boards=[v for v in product((0,1),(3,4),(6,7))
                    if all(m & (1<<k) for m,k in zip(before,v))
                    and v[:2]!=(0,3) and v[1:]!=(4,6)]
            if facts and facts['over']:
                self.assertFalse(boards)
            else:
                self.assertTrue(all(all(m & (1<<k) for m,k in zip(after,v)) for v in boards))
            self.assertEqual(before[:3],[a,b,c])

    def test_explicit_override_can_turn_implicit_policy_off_or_on(self):
        from types import SimpleNamespace
        problem=SimpleNamespace(reference_pose_cuts=True)
        with patch('constraint_dfs.solve',side_effect=lambda item,**kw:{'applied':item.reference_pose_cuts}):
            self.assertFalse(run_solver(problem,'dfs',reference_pose_cuts=False)['applied'])
            self.assertTrue(problem.reference_pose_cuts)
            problem.reference_pose_cuts=False
            self.assertTrue(run_solver(problem,'dfs',reference_pose_cuts=True)['applied'])
            self.assertFalse(problem.reference_pose_cuts)

    def test_legacy_physical_counter_restore_keeps_scope_and_rejects_unknowns(self):
        from copy import deepcopy
        from constraint_sat_session import PersistentSatSession
        from constraint_work import digest
        from test_gpu_domain_round25 import joint_hall_fixture
        problem=joint_hall_fixture()
        scope=WorkScope.for_problem(problem,'gold')
        session=PersistentSatSession(problem,scope,[scope.root])
        try:saved=session.snapshot()
        finally:session.close()
        # Old original-physical models emitted just the populated keys.
        saved['stats'].pop('physical_core_units',None)
        saved['stats'].update(physical_forbidden_clauses=11520,physical_required_clauses=7680)
        def resign(data):data['digest']=digest({k:v for k,v in data.items() if k!='digest'})
        resign(saved)
        restored=PersistentSatSession.restore(problem,saved)
        try:
            after=restored.snapshot()
            self.assertEqual(after['frontier'],saved['frontier'])
            self.assertEqual(after['stats']['physical_required_clauses'],7680)
            self.assertEqual(after['stats']['session_cubes_closed'],0)
        finally:restored.close()
        for edit in (None,dict(wrong_counter=5),dict(physical_required_clauses=-1),dict(physical_required_clauses=True)):
            bad=deepcopy(saved)
            if edit is None:bad['stats']=[]
            else:bad['stats'].update(edit)
            resign(bad)
            with self.assertRaisesRegex(ValueError,'Malformed native-session counters'):PersistentSatSession.restore(problem,bad)

    def test_api_rejects_unsupported_receivers_before_loading_any_engine(self):
        for alg,opts,target,resume in [('sat',{},'gold',None),('dfs',{},'edges',None),
                ('dfs',{'shared_coverage':True},'gold',None),('dfs',{'lcg':True},'gold',None),
                ('dfs',{'reason_tracking':True},'gold',None),('dfs',{'symbolic_frontier':True},'gold',None),
                ('dfs',{},'gold',{'parallel_version':1}),('gpu-domains',{},'gold',{'schema':'diamond-symbolic-coordinator-v1'})]:
            with self.assertRaises(ValueError):validate_receiver(alg,opts,target,resume)


class FrontierCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=next((p for p in (Path('data/tiles.json'),Path('../data/tiles.json')) if p.is_file()),None)
        if path is None:raise unittest.SkipTest('Original prepared artwork unavailable')
        cls.data=json.loads(path.read_bytes())

    def setUp(self):
        self.p=SearchProblem(self.data,physical_gold_support=True)
        self.record=active_catalog(self.p)
        self.seed=self.record['codes']
        self.scope=WorkScope.for_problem(self.p,'gold')

    def pieces(self,cells):
        ds=list(self.scope.root)
        for c in cells:ds[c]=1<<self.seed[c]
        return ds

    def test_empty_root_and_nonmatching_singleton_leave_frontier_intact(self):
        with patch('r35_repair_seeds.active_catalog',side_effect=AssertionError('open root has no unit')):
            self.assertEqual(prepare_frontier(self.p,None)[0],None)
        input_box=self.pieces([5]);input_box[5]=1<<((self.seed[5]+1)%480)
        reduced,stats=FrontierPoses(self.p).prune(input_box)
        self.assertEqual(reduced,input_box)
        self.assertIsNone(stats)

    def test_checked_literals_transport_under_all_ten_proper_rotations(self):
        from constraint_symmetry import proper_rotations
        receiver=FrontierPoses(self.p)
        for rotate in proper_rotations(self.p.board):
            transformed=rotate.apply(self.seed)
            ds=list(self.scope.root)
            a,b=rotate.cells[5],rotate.cells[102]
            ds[a]=1<<transformed[a]
            out,stats=receiver.prune(ds)
            self.assertFalse(stats['over'])
            self.assertFalse(out[b]>>transformed[b]&1)
            self.assertTrue(ds[b]>>transformed[b]&1)
            # Widen the last cube after a different-domain sibling with the
            # identical singleton signature had no need for this particular cut.
            narrow=ds.copy();narrow[b]&=~(1<<transformed[b])
            self.assertFalse(receiver.prune(narrow)[0][b]>>transformed[b]&1)
            self.assertFalse(receiver.prune(ds)[0][b]>>transformed[b]&1)
            ds[b]=1<<transformed[b]
            self.assertTrue(receiver.prune(ds)[1]['over'])

    def test_guard_sets_are_distinct_from_static_cover_and_geometry_is_bound(self):
        from r35_pose_departures import indexed_rows,propagate,reference_images
        guards=self.record['post_master']
        images=reference_images(tuple(self.seed))
        base=indexed_rows(tuple(tuple(row) for row in self.record['rows']))
        for guard in guards:
            ds=self.pieces(guard)
            _,old=propagate(ds,images,base,65)
            self.assertFalse(old['over'])
            self.assertTrue(FrontierPoses(self.p).prune(ds)[1]['over'])
        for which in ('art','adjacency','masks'):
            p=SearchProblem(self.data,physical_gold_support=True)
            if which=='masks':p.masks[0,0]^=1
            elif which=='adjacency':p.neighbors[0,0]=int(p.neighbors[0,1])
            else:p.tiles[0]['segments'],p.tiles[1]['segments']=p.tiles[1]['segments'],p.tiles[0]['segments']
            self.assertIsNone(FrontierPoses(p).prune(self.pieces(guards[0]))[1],which)

    def test_public_cuda_skips_only_proved_cells_and_retains_original_scope(self):
        solved=self.pieces([5,102])
        untouched=list(self.scope.root)
        untouched[5]=1<<((self.seed[5]+1)%480)
        portable=frontier_snapshot(self.scope,[solved,untouched])
        recent,work=prepare_frontier(self.p,portable)
        self.assertEqual(restore_frontier(recent,self.scope),(tuple(untouched),))
        self.assertEqual(work['reference_pose_closed'],1)
        self.assertEqual(recent['scope'],portable['scope'])
        # A fresh imported frontier narrows only its owned work. All negative
        # lanes can finish before a CUDA package or device is needed.
        self.p.reference_pose_cuts=True
        dead=frontier_snapshot(self.scope,[solved])
        with patch('gpu_domain_round24.DomainCUDA',side_effect=AssertionError('unexpected device creation')):
            ans=run_solver(self.p,'gpu-domains',work_frontier=dead,seconds=10,replicas=1)
        self.assertEqual((ans['status'],ans['nodes'],ans['stats']['reference_pose_closed']),('infeasible',0,1))
        self.assertEqual(ans['checkpoint']['frontier']['boxes'],[])
        self.assertEqual(prepare_frontier(self.p,portable,lambda:True)[0],portable)

    def test_parallel_and_persistent_sat_prepare_only_portable_jobs(self):
        gone=self.pieces([5,102])
        rest=list(self.scope.root);rest[5]=1<<((self.seed[5]+1)%480)
        mixed=frontier_snapshot(self.scope,[gone,rest])
        all_gone=frontier_snapshot(self.scope,[gone])
        for algorithm,options in [('parallel-dfs',{}),('sat',{'persistent_sat':True})]:
            params=dict(options=options,seconds=10,reference_pose_cuts=True)
            if algorithm=='parallel-dfs':params['workers']=2
            done=run_solver(self.p,algorithm,resume=all_gone,**params)
            self.assertEqual((done['status'],done['nodes'],done['stats']['reference_pose_closed']),('infeasible',0,1))
            self.assertEqual(done['stats'].get('session_model_builds',done['stats'].get('parallel_workers_started')),0)
            ticks=iter((False,False))
            stopped=run_solver(self.p,algorithm,resume=mixed,stop=lambda:next(ticks,True),**params)
            self.assertFalse(stopped['complete'])
            saved=stopped['checkpoint']
            if algorithm=='sat':
                self.assertEqual(restore_frontier(saved['frontier'],self.scope),(tuple(rest),))
            else:
                self.assertEqual(len(saved['jobs']),1)
                self.assertEqual([int(v,16) for v in saved['jobs'][0]['region']],rest)
            with patch('r35_pose_frontiers.prepare_frontier',side_effect=AssertionError('opaque continuation cannot be screened')):
                again=run_solver(self.p,algorithm,resume=saved,stop=lambda:True,**params)
            self.assertEqual(again['status'],'stopped')
            if algorithm=='sat':self.assertEqual(again['checkpoint']['frontier'],saved['frontier'])
            else:self.assertEqual(again['checkpoint']['jobs'],saved['jobs'])
            for invalid in ({}, {'resume':mixed,'work_frontier':mixed}, {'resume':[]}, {'resume':dict(mixed,digest='wrong')}, {'resume':{'version':99}}):
                with self.assertRaises(ValueError):run_solver(self.p,algorithm,**invalid,**params)

    @unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Requires native CUDA')
    def test_existing_live_cuda_resume_keeps_its_unfinished_native_owner(self):
        from gpu_domain_round24 import DomainCUDA
        root=self.pieces([5,102])
        pending=frontier_snapshot(self.scope,[root])
        did_step=[]
        original=DomainCUDA.step
        def tick(engine,*args,**kwargs):
            value=original(engine,*args,**kwargs)
            did_step.append(True)
            return value
        with patch.object(DomainCUDA,'step',tick):
            paused=run_solver(self.p,'gpu-domains',work_frontier=pending,seconds=15,replicas=1,stop=lambda:bool(did_step))
        self.assertEqual(paused['status'],'stopped')
        self.assertTrue(paused['checkpoint']['native'])
        self.p.reference_pose_cuts=True
        with patch('r35_pose_frontiers.prepare_frontier',side_effect=AssertionError('tried to replace an active native owner')):
            end=run_solver(self.p,'gpu-domains',resume=paused['checkpoint'],seconds=30,replicas=1)
        self.assertEqual(end['status'],'infeasible')
        self.assertGreater(end['stats']['support_checks'],0)
