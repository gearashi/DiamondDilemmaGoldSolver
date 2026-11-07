"""Round25 exact GPU mechanisms; independent finite enumeration is authoritative."""
import copy,itertools,json,os,unittest
from unittest.mock import patch
import numpy as np
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import BASE_COUNTERS,COUNTERS,pack_native,solve,split_cubes
from constraint_work import WorkScope,frontier_snapshot,digest
from search_problem import SearchProblem
from validator import reverse_mask,validate_arrangement
from test_constraint_dfs import board,simple_tiles,multiple_loops,double_face,tetrahedron_tiles
from test_gpu_domain_round24 import enumerate_engine


def joint_hall_fixture():
    _,geometry=tetrahedron_tiles()
    tiles=[{'id':str(i),'segments':[[[side,2],[side,10]] for side in range(3)]} for i in range(4)]
    problem=SearchProblem(tiles,geometry);problem.domains=[[0,3],[0,6,9],[0,6,9],[0,6,9]]
    return problem

def repeated_neighbor_fixture():
    tiles=[{'id':'A','segments':[[[s,p],[s,12-p]] for s,p in enumerate((2,3,4))]},
           {'id':'B','segments':[[[s,p],[s,12-p]] for s,p in enumerate((2,4,3))]}]
    problem=SearchProblem(tiles,double_face());problem.domains=[[0,1,2],[3,4,5]]
    return problem

def word_boundary_fixture():
    n=12;geometry=board(n,[(a,s,a+1,s) for a in range(0,n,2) for s in range(3)])
    tiles=[{'id':str(i),'segments':[[[s,2],[s,10]] for s in range(3)]} for i in range(n)]
    problem=SearchProblem(tiles,geometry)
    problem.domains=[list(range(30,36)),list(range(30,36))]+[[3*(cell-2)] for cell in range(2,n)]
    return problem

def planted160_fixture():
    from search_fixtures import planted_gold
    data,geometry,witness=planted_gold(2526,size=4,extra_edges=12)
    problem=SearchProblem(data,geometry);problem.domains=[[code] for code in witness]
    for cell in (0,17,93):problem.domains[cell]=list(range(3*(witness[cell]//3),3*(witness[cell]//3)+3))
    return problem

def fork_gold_fixture():
    n=12;geometry=board(n,[(cell,1,(cell+1)%n,0) for cell in range(n)])
    tiles=[{'id':str(i),'segments':[[[0,6],[1,6]]]} for i in range(n)]
    problem=SearchProblem(tiles,geometry);problem.domains=[[3*cell] for cell in range(n-2)]+[[30,33],[30,33]]
    return problem


def precise_reason_fixture():
    # Seed2529 trial789, independently found six-cell triangular-prism case.
    tiles=[{'id': 0, 'segments': [[[2, 9], [1, 2]], [[2, 3], [1, 10]], [[0, 2], [0, 10]]]}, {'id': 1, 'segments': [[[1, 2], [2, 3]], [[2, 9], [0, 10]], [[1, 10], [0, 2]]]}, {'id': 2, 'segments': [[[2, 3], [2, 9]], [[1, 9], [0, 9]], [[1, 3], [0, 3]]]}, {'id': 3, 'segments': [[[0, 10], [0, 2]], [[1, 2], [2, 2]], [[2, 10], [1, 10]]]}, {'id': 4, 'segments': [[[1, 10], [0, 2]], [[0, 10], [2, 9]], [[1, 2], [2, 3]]]}, {'id': 5, 'segments': [[[0, 10], [0, 2]], [[1, 3], [2, 2]], [[2, 10], [1, 9]]]}]
    geometry=board(6,[[0, 0, 1, 0], [1, 1, 2, 0], [2, 1, 0, 1], [3, 0, 4, 0], [4, 1, 5, 0], [5, 1, 3, 1], [0, 2, 3, 2], [1, 2, 4, 2], [2, 2, 5, 2]])
    problem=SearchProblem(tiles,geometry);problem.domains=[[2, 6, 12, 13], [1, 2, 5, 8, 11, 12, 15, 16], [0, 3, 7, 8, 9], [3, 4, 8, 9, 10, 15], [1, 10, 11, 12, 15, 16], [2, 3, 7, 9, 12, 15]]
    return problem


def independent_solutions(problem,target):
    found=[]
    for codes in itertools.product(*problem.domains):
        if len({code//3 for code in codes})!=problem.n:continue
        report=validate_arrangement(problem.tiles,list(codes),problem.board)
        if report['valid'] or target=='edges' and report['matched_edges']==len(problem.edges):found.append(codes)
    return set(found)

def independent_star_support(problem,domains,cell,code):
    # Explicit Cartesian orientation assignments, not Hall masks or GPU tables.
    neighbors=sorted(set(int(x) for x in problem.neighbors[cell] if x>=0))
    for chosen in itertools.product(*(domains[other] for other in neighbors)):
        if len({code//3,*[value//3 for value in chosen]})!=len(neighbors)+1:continue
        mapping=dict(zip(neighbors,chosen));ok=True
        for side,other in enumerate(problem.neighbors[cell]):
            if other<0:ok &= problem.masks[code,side]==0
            else:ok &= problem.masks[code,side]==reverse_mask(problem.masks[mapping[int(other)],int(problem.sides[cell,side])])
        if ok:return True
    return False

class JointSupportHostTests(unittest.TestCase):
    def test_reference_discriminator_is_not_independent_seam_support(self):
        problem=joint_hall_fixture();self.assertEqual(len(independent_solutions(problem,'edges')),6)
        self.assertFalse(independent_star_support(problem,problem.domains,0,0))
        self.assertTrue(independent_star_support(problem,problem.domains,0,3))
        for side,other in enumerate(problem.neighbors[0]):
            self.assertTrue(any(problem.masks[0,side]==reverse_mask(problem.masks[value,int(problem.sides[0,side])]) for value in problem.domains[other]))
    def test_repeated_neighbor_requires_one_orientation_for_all_seams(self):
        problem=repeated_neighbor_fixture();self.assertEqual(independent_solutions(problem,'edges'),set())
        for code in problem.domains[0]:
            self.assertFalse(independent_star_support(problem,problem.domains,0,code))
            for side in range(3):self.assertTrue(any(problem.masks[code,side]==reverse_mask(problem.masks[value,side]) for value in problem.domains[1]))
    def test_capping_each_support_set_at_neighbor_count_preserves_every_matching(self):
        universe=range(4);sets=[tuple(tile for tile in universe if mask>>tile&1) for mask in range(16)]
        checked=0
        for count in (1,2,3):
            for choices in itertools.product(sets,repeat=count):
                exists=any(len(set(row))==count for row in itertools.product(*choices))
                capped=any(len(set(row))==count for row in itertools.product(*(row[:count] for row in choices)))
                self.assertEqual(exists,capped);checked+=1
        self.assertEqual(checked,4368)

    def test_feature_is_boolean_and_memory_is_bounded_before_device_import(self):
        problem=joint_hall_fixture()
        with self.assertRaisesRegex(ValueError,'boolean'):DomainCUDA(problem,joint_support=1)
        for enabled in (False,True):
            with self.assertRaisesRegex(ValueError,'budget'):DomainCUDA(problem,joint_support=enabled,max_memory_bytes=1)
    def test_already_stopped_runner_passes_no_gpu_work_for_enabled_feature(self):
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(joint_hall_fixture(),options={'gpu_joint_support':True},stop=lambda:True)
        factory.assert_not_called();self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['stats']['joint_exclusions'],0)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round25 exact tests')
class JointSupportCUDATests(unittest.TestCase):
    def test_complete_solutions_match_independent_domains_enumeration(self):
        problems=[SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),SearchProblem(*tetrahedron_tiles()),joint_hall_fixture(),repeated_neighbor_fixture(),word_boundary_fixture()]
        for problem in problems:
            for target in ('edges','gold'):
                expected=independent_solutions(problem,target)
                for learning,strands in itertools.product((False,True),repeat=2):
                    with self.subTest(n=problem.n,target=target,learning=learning,strands=strands):
                        engine=DomainCUDA(problem,target=target,joint_support=True,learning=learning,incremental_strands=strands)
                        try:
                            found=enumerate_engine(engine,17)
                            self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                        finally:engine.close()
    def test_joint_hall_and_repeated_seams_reduce_real_decisions(self):
        for problem in (joint_hall_fixture(),repeated_neighbor_fixture()):
            records=[];expected=independent_solutions(problem,'edges')
            for enabled in (False,True):
                engine=DomainCUDA(problem,target='edges',joint_support=enabled)
                try:self.assertEqual(set(enumerate_engine(engine,17)),expected);records.append(engine.stats())
                finally:engine.close()
            self.assertGreater(records[1]['joint_exclusions'],0)
            self.assertGreater(records[1]['joint_word_scans'],0)
            if problem.n==4:self.assertGreater(records[1]['joint_tile_visits'],0)
            else:self.assertEqual(records[1]['joint_tile_visits'],0)  # All conjoined masks are empty.
            self.assertLess(records[1]['decisions'],records[0]['decisions'])
    def test_every_primitive_native_resume_and_remaining_solution_ownership(self):
        problem=SearchProblem(*tetrahedron_tiles());expected=independent_solutions(problem,'gold');seen=set();engine=DomainCUDA(problem,joint_support=True)
        try:
            for iteration in range(3000):
                states=engine.step(1);cubes=engine.remaining_domain_cubes()
                frontier_snapshot(WorkScope.for_problem(problem),cubes)
                remaining={answer for answer in expected if any(all(mask&(1<<value) for mask,value in zip(cube,answer)) for cube in cubes)}
                self.assertEqual(remaining,expected-seen)
                for item in engine.candidates():
                    answer=tuple(item['codes']);self.assertNotIn(answer,seen);seen.add(answer);engine.acknowledge([item['lane']])
                if iteration%17==0:
                    saved=engine.checkpoint();replacement=DomainCUDA(problem,joint_support=True);replacement.restore(saved);engine.close();engine=replacement
                if all(states==2):break
            else:self.fail('Finite complete enumeration stalled')
            self.assertEqual(seen,expected)
        finally:engine.close()
    def test_all160_cells_preserve_controlled_gold_witness(self):
        problem=planted160_fixture();expected=independent_solutions(problem,'gold')
        self.assertEqual(problem.n,160);self.assertTrue(expected)
        engine=DomainCUDA(problem,joint_support=True)
        try:
            engine.step(7);snapshot=engine.checkpoint();engine.restore(snapshot)
            found=enumerate_engine(engine,256)
            self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
        finally:engine.close()

    def test_native_split_run_preserves_exact_arrays_and_counter_carry(self):
        problem=joint_hall_fixture();first=DomainCUDA(problem,target='edges',joint_support=True);second=DomainCUDA(problem,target='edges',joint_support=True)
        try:
            first.state[first.offsets['joint_counters'],0]=np.uint32(0xffffffff)
            first.step(7);saved=first.checkpoint();second.restore(saved)
            first.step(29);second.step(29)
            self.assertTrue(np.array_equal(first.checkpoint()['state'],second.checkpoint()['state']))
            self.assertGreater(first.stats()['joint_support_checks'],2**32-1)
            self.assertEqual(first.stats(),second.stats())
        finally:first.close();second.close()
    def test_runner_legacy_resume_feature_change_and_new_native_resume(self):
        problem=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(problem);engine=DomainCUDA(problem)
        try:
            engine.step(5)
            payload={'version':1,'scope':scope.payload(),'queue':[],'native':pack_native(engine),'totals':dict.fromkeys(BASE_COUNTERS,0),
                     'completed_lanes':0,'active_stats':engine.stats(),'frontier':frontier_snapshot(scope,engine.remaining_domain_cubes())}
        finally:engine.close()
        payload['digest']=digest(payload);before=copy.deepcopy(payload)
        result=solve(problem,seconds=5,replicas=1,resume=payload,options={'gpu_joint_support':True})
        self.assertEqual(payload,before);self.assertEqual(result['status'],'solved');self.assertTrue(result['validation']['valid'])
        self.assertEqual(result['stats']['gpu_domain_portable_restart'],1)
        checks=iter((False,True,True));stopped=solve(problem,replicas=1,options={'gpu_joint_support':True},stop=lambda:next(checks,True))
        saved=json.loads(json.dumps(stopped['checkpoint']));resumed=solve(problem,seconds=5,replicas=1,resume=saved,options={'gpu_joint_support':True})
        self.assertEqual(resumed['status'],'solved');self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],0)
        self.assertEqual(saved['native']['schema'],'diamond-domain-cuda-v2')

class PropagatedForkHostTests(unittest.TestCase):
    def test_already_cancelled_preparation_never_allocates(self):
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(fork_gold_fixture(),options={'gpu_propagated_forks':True},stop=lambda:True)
        factory.assert_not_called();self.assertEqual(result['status'],'stopped')
        self.assertFalse(result['checkpoint']['preparing']);self.assertEqual(result['stats']['fork_children'],0)
    def test_invalid_flag_rejected_before_allocation(self):
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            with self.assertRaisesRegex(ValueError,'boolean'):solve(fork_gold_fixture(),options={'gpu_propagated_forks':1})
        factory.assert_not_called()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round25 exact tests')
class PropagatedForkCUDATests(unittest.TestCase):
    def prepare(self,engine,quantum=17):
        for iteration in range(10000):
            states,ready=engine.propagate_root(quantum)
            if ready:return states
        self.fail('Bounded root propagation stalled')
    def test_complete_partitions_preserve_all_solutions_and_root_facts(self):
        for factory,target in [(fork_gold_fixture,'gold'),(joint_hall_fixture,'edges'),(word_boundary_fixture,'edges'),(repeated_neighbor_fixture,'edges')]:
            for joint,strands in itertools.product((False,True),repeat=2):
                problem=factory();expected=independent_solutions(problem,target);engine=DomainCUDA(problem,target=target,joint_support=joint,incremental_strands=strands)
                child=None
                try:
                    states=self.prepare(engine)
                    self.assertEqual(engine.stats()['decisions'],0)
                    if all(states==2):self.assertEqual(expected,set());continue
                    cubes=split_cubes(engine.remaining_domain_cubes(),8)
                    child=engine.fork_root(cubes);data=child.checkpoint()['state']
                    self.assertTrue(np.all(data[1:3]==0));self.assertEqual(child.stats()['decisions'],0)
                    parent=engine.checkpoint()['state']
                    for name,length in [('board',problem.n),('owner',problem.n),('parent',16*problem.n),('size',16*problem.n)]:
                        offset=engine.offsets[name];self.assertTrue(np.all(data[offset:offset+length]==parent[offset:offset+length]))
                    child.restore(child.checkpoint())
                    found=enumerate_engine(child,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                finally:
                    if child is not None:child.close()
                    engine.close()
    def test_reuses_real_root_strand_joins_with_no_lost_gold_completions(self):
        problem=fork_gold_fixture();expected=independent_solutions(problem,'gold');self.assertEqual(len(expected),2)
        raw=DomainCUDA(problem,split_cubes([WorkScope.for_problem(problem).root],8));parent=DomainCUDA(problem);child=None
        try:
            self.assertEqual(set(enumerate_engine(raw,17)),expected);old=raw.stats()
            self.prepare(parent);before=parent.stats();self.assertGreater(before['strand_joins'],0)
            child=parent.fork_root(split_cubes(parent.remaining_domain_cubes(),8))
            self.assertEqual(set(enumerate_engine(child,17)),expected)
            self.assertLess(before['strand_joins']+child.stats()['strand_joins'],old['strand_joins'])
        finally:
            if child is not None:child.close()
            parent.close();raw.close()
    def test_preparation_primitive_resume_is_exact_and_frontier_preserves_witnesses(self):
        problem=fork_gold_fixture();expected=independent_solutions(problem,'gold');engine=DomainCUDA(problem)
        try:
            for step in range(500):
                states,ready=engine.propagate_root(1)
                cubes=engine.remaining_domain_cubes();frontier_snapshot(WorkScope.for_problem(problem),cubes)
                self.assertTrue(all(any(all(mask&(1<<code) for mask,code in zip(row,answer)) for row in cubes) for answer in expected))
                if step%7==0:
                    saved=engine.checkpoint();replacement=DomainCUDA(problem);replacement.restore(saved)
                    self.assertTrue(np.array_equal(saved['state'],replacement.checkpoint()['state']));engine.close();engine=replacement
                if ready:break
            else:self.fail('Preparation did not reach its barrier')
            before=engine.checkpoint()['state'];engine.propagate_root(100)
            self.assertTrue(np.array_equal(before,engine.checkpoint()['state']))
            with self.assertRaisesRegex(ValueError,'cover'):engine.fork_root([tuple(1<<(row.bit_length()-1) for row in cubes[0])])
            with self.assertRaisesRegex(ValueError,'overlap|cover'):engine.fork_root([cubes[0],cubes[0]])
        finally:engine.close()
    def test_full160_root_connectivity_rebases_and_restores(self):
        problem=planted160_fixture();expected=independent_solutions(problem,'gold');parent=DomainCUDA(problem,joint_support=True);child=None
        try:
            self.prepare(parent,256);self.assertGreater(parent.stats()['strand_joins'],0)
            child=parent.fork_root(parent.remaining_domain_cubes());child.restore(child.checkpoint())
            self.assertEqual(set(enumerate_engine(child,256)),expected)
        finally:
            if child is not None:child.close()
            parent.close()
    def test_stop_after_fork_preserves_native_state_and_future_work_counts(self):
        problem=fork_gold_fixture();options={'gpu_propagated_forks':True};forked=[False];original=DomainCUDA.fork_root
        def fork(engine,*args,**kwargs):
            child=original(engine,*args,**kwargs);forked[0]=True;return child
        with patch.object(DomainCUDA,'fork_root',fork):
            paused=solve(problem,replicas=4,options=options,stop=lambda:forked[0])
        self.assertEqual(paused['status'],'stopped');self.assertFalse(paused['checkpoint']['preparing'])
        self.assertEqual(paused['stats']['fork_children'],4)
        resumed=solve(problem,replicas=4,options=options,resume=paused['checkpoint'])
        direct=solve(problem,replicas=4,options=options)
        self.assertEqual(resumed['codes'],direct['codes'])
        self.assertEqual(resumed['stats'],direct['stats'])
        self.assertEqual(resumed['checkpoint']['native'],direct['checkpoint']['native'])

    def test_prepared_owner_enters_coverage_only_after_all_children_exhaust(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from constraint_work import CoverageLedger
        n=4;geometry=board(n,[(cell,1,(cell+1)%n,0) for cell in range(n)])
        tiles=[{'id':str(i),'segments':[[[0,6],[1,6]]]} for i in range(n)]
        problem=SearchProblem(tiles,geometry);problem.domains=[[0,3,6] for _ in range(n)]
        problem.domains[0].append(1)  # Unsupported rotation removed before the fork.
        self.assertEqual(independent_solutions(problem,'gold'),set())
        scope=WorkScope.for_problem(problem);options={'gpu_propagated_forks':True,'gpu_joint_support':True,'shared_coverage':True}
        forked=[False];original=DomainCUDA.fork_root
        def fork(engine,*args,**kwargs):
            child=original(engine,*args,**kwargs);forked[0]=True;return child
        with TemporaryDirectory() as directory:
            path=Path(directory)/'coverage.sqlite3';learning={'_coverage_path':str(path)}
            with patch.object(DomainCUDA,'fork_root',fork):
                paused=solve(problem,replicas=4,options=options,learning_state=learning,stop=lambda:forked[0])
            self.assertEqual(paused['status'],'stopped');self.assertGreater(paused['stats']['fork_root_removals'],0)
            self.assertFalse(paused['checkpoint']['preparing'])
            with CoverageLedger(path,scope) as ledger:self.assertEqual(ledger.exclusions(),())
            resumed=solve(problem,seconds=10,replicas=4,options=options,learning_state=learning,resume=paused['checkpoint'])
            self.assertEqual(resumed['status'],'infeasible');self.assertTrue(resumed['complete'])
            with CoverageLedger(path,scope) as ledger:self.assertEqual(ledger.exclusions(),(scope.root,))

    def test_runner_stopped_preparation_and_forked_resume_keep_exact_policy(self):
        problem=fork_gold_fixture();options={'gpu_propagated_forks':True};events=[]
        original=DomainCUDA.propagate_root;calls=[0]
        def one(engine,work):calls[0]+=1;return original(engine,1)
        with patch.object(DomainCUDA,'propagate_root',one):
            first=solve(problem,replicas=4,options=options,stop=lambda:calls[0]>=3)
        saved=json.loads(json.dumps(first['checkpoint']));self.assertTrue(saved['preparing'])
        original_digest=digest(saved)
        resumed=solve(problem,seconds=10,replicas=4,options=options,resume=saved)
        self.assertEqual(resumed['status'],'solved');self.assertEqual(digest(saved),original_digest)
        self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],0)
        self.assertGreater(resumed['stats']['fork_reused_unions'],0)
        self.assertFalse(resumed['checkpoint']['preparing'])
        same=solve(problem,seconds=10,replicas=4,options=options,resume=resumed['checkpoint'])
        self.assertEqual(same['codes'],resumed['codes']);self.assertEqual(same['stats']['fork_children'],resumed['stats']['fork_children'])
        off=solve(problem,seconds=10,replicas=4,resume=saved)
        self.assertEqual(off['status'],'solved');self.assertEqual(off['stats']['gpu_domain_portable_restart'],1)

class EventPropagationHostTests(unittest.TestCase):
    def test_static_consumer_index_covers_original_domain_tiles_across_words(self):
        problem=planted160_fixture();engine=DomainCUDA.__new__(DomainCUDA)
        engine.problem=problem;engine.n=problem.n;engine.nc=3*problem.n;engine.words=(engine.nc+31)//32;engine.event_propagation=True
        table=engine._tables();offset=25*engine.n+engine.nc*33+9*engine.nc*engine.words;words=(engine.n+31)//32
        index=table[offset:].reshape(engine.n,words)
        for tile in range(engine.n):
            cells={cell for cell in range(engine.n) if int(index[tile,cell//32])>>(cell%32)&1}
            expected={cell for cell,row in enumerate(problem.domains) if any(code//3==tile for code in row)}
            self.assertEqual(cells,expected)

    def test_policy_and_memory_fail_before_device_allocation(self):
        problem=fork_gold_fixture()
        with self.assertRaisesRegex(ValueError,'boolean'):DomainCUDA(problem,event_propagation=1)
        with self.assertRaisesRegex(ValueError,'budget'):DomainCUDA(problem,event_propagation=True,max_memory_bytes=1)
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(problem,options={'gpu_event_propagation':True},stop=lambda:True)
        factory.assert_not_called();self.assertEqual(result['stats']['event_enqueues'],0)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round25 exact tests')
class EventPropagationCUDATests(unittest.TestCase):
    def test_complete_independent_enumeration_in_all_reasoning_modes(self):
        fixtures=[lambda:SearchProblem(simple_tiles(),double_face()),lambda:SearchProblem(multiple_loops(),double_face()),lambda:SearchProblem(*tetrahedron_tiles()),joint_hall_fixture,repeated_neighbor_fixture,word_boundary_fixture,fork_gold_fixture]
        for factory in fixtures:
            problem=factory()
            for target in ('edges','gold'):
                expected=independent_solutions(problem,target)
                for joint,learning,strands in itertools.product((False,True),repeat=3):
                    with self.subTest(n=problem.n,target=target,joint=joint,learning=learning,strands=strands):
                        engine=DomainCUDA(problem,target=target,event_propagation=True,joint_support=joint,learning=learning,incremental_strands=strands)
                        try:
                            found=enumerate_engine(engine,19)
                            self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                        finally:engine.close()
    def test_each_primitive_resume_preserves_remaining_solutions_and_exact_arrays(self):
        problem=joint_hall_fixture();expected=independent_solutions(problem,'edges');seen=set();engine=DomainCUDA(problem,target='edges',event_propagation=True,joint_support=True)
        try:
            for iteration in range(4000):
                states=engine.step(1);cubes=engine.remaining_domain_cubes();frontier_snapshot(WorkScope.for_problem(problem,'edges'),cubes)
                remaining={answer for answer in expected if any(all(mask&(1<<value) for mask,value in zip(cube,answer)) for cube in cubes)}
                self.assertEqual(remaining,expected-seen)
                for item in engine.candidates():
                    answer=tuple(item['codes']);self.assertNotIn(answer,seen);seen.add(answer);engine.acknowledge([item['lane']])
                if iteration%11==0:
                    saved=engine.checkpoint();replacement=DomainCUDA(problem,target='edges',event_propagation=True,joint_support=True);replacement.restore(saved)
                    self.assertTrue(np.array_equal(saved['state'],replacement.checkpoint()['state']));engine.close();engine=replacement
                if all(states==2):break
            else:self.fail('Finite event queue did not finish')
            self.assertEqual(seen,expected)
        finally:engine.close()
    def test_queue_padding_and_false_fixed_point_rejected_with_valid_hash(self):
        import hashlib
        engine=DomainCUDA(joint_hall_fixture(),target='edges',event_propagation=True)
        try:
            for mutation in ('padding','fixed_point','changed'):
                saved=engine.checkpoint()
                if mutation=='padding':saved['state'][engine.offsets['dirty_cells'],0]|=np.uint32(1<<12)
                else:
                    saved['state'][6,0]=2
                    if mutation=='changed':
                        saved['state'][engine.offsets['dirty_cells'],0]=0;saved['state'][5,0]=1
                saved['state_sha256']=hashlib.sha256(saved['state'].tobytes()).hexdigest()
                with self.assertRaisesRegex(ValueError,'padding|pending'):engine.restore(saved)
        finally:engine.close()
    def test_event_policy_resume_and_prepared_forked_fullsize_gold(self):
        problem=planted160_fixture();expected=independent_solutions(problem,'gold');parent=DomainCUDA(problem,joint_support=True,event_propagation=True);child=None
        try:
            for _ in range(1000):
                states,ready=parent.propagate_root(23)
                if ready:break
            else:self.fail('Fullsize event propagation stalled')
            saved=parent.checkpoint();parent.restore(saved)
            child=parent.fork_root(parent.remaining_domain_cubes());child.restore(child.checkpoint())
            self.assertEqual(set(enumerate_engine(child,256)),expected)
        finally:
            if child is not None:child.close()
            parent.close()
        problem=fork_gold_fixture();options={'gpu_event_propagation':True,'gpu_propagated_forks':True,'gpu_joint_support':True}
        checks=iter((False,True,True));paused=solve(problem,replicas=4,options=options,stop=lambda:next(checks,True))
        saved=json.loads(json.dumps(paused['checkpoint']));self.assertEqual(saved['native']['schema'],'diamond-domain-cuda-v3')
        resumed=solve(problem,seconds=10,replicas=4,options=options,resume=saved)
        self.assertEqual(resumed['status'],'solved');self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],0)
        old=solve(problem,seconds=10,replicas=4,options={'gpu_propagated_forks':True,'gpu_joint_support':True},resume=saved)
        self.assertEqual(old['status'],'solved');self.assertEqual(old['stats']['gpu_domain_portable_restart'],1)

class PreciseReasonHostTests(unittest.TestCase):
    def test_policy_bounds_and_memory_reject_before_gpu(self):
        problem=precise_reason_fixture()
        for args in ({'precise_reasons':1},{'reason_slots':8},{'precise_reasons':True,'reason_slots':7},{'precise_reasons':True,'reason_slots':4096}):
            with self.assertRaises(ValueError):DomainCUDA(problem,**args)
        with self.assertRaisesRegex(ValueError,'1,010,844 bytes'):
            DomainCUDA(planted160_fixture(),joint_support=True,event_propagation=True,precise_reasons=True,max_memory_bytes=1)
        self.assertLessEqual(1010844*512,512*1024*1024)
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(problem,options={'gpu_precise_reasons':True},stop=lambda:True)
        factory.assert_not_called();self.assertEqual(result['stats']['precise_hits'],0)
    def test_independent_prism_fixture_has_no_completion(self):
        self.assertEqual(independent_solutions(precise_reason_fixture(),'edges'),set())

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round25 exact tests')
class PreciseReasonCUDATests(unittest.TestCase):
    def test_actual_prism_decisions_reduce_and_saturation_falls_back(self):
        records=[];problem=precise_reason_fixture();expected=independent_solutions(problem,'edges')
        for enabled,slots in ((False,None),(True,None),(True,8)):
            engine=DomainCUDA(problem,target='edges',precise_reasons=enabled,reason_slots=slots)
            try:self.assertEqual(set(enumerate_engine(engine,17)),expected);records.append(engine.stats())
            finally:engine.close()
        self.assertEqual([row['decisions'] for row in records],[21,17,21])
        self.assertEqual([row['support_checks'] for row in records],[509,458,509])
        self.assertGreater(records[2]['precise_fallbacks'],0);self.assertGreater(records[2]['precise_dropped'],0)
    def test_complete_satisfiable_and_unsatisfiable_enumeration_in_feature_combinations(self):
        for factory,target in [(joint_hall_fixture,'edges'),(repeated_neighbor_fixture,'edges'),(word_boundary_fixture,'edges'),(fork_gold_fixture,'gold'),(lambda:SearchProblem(*tetrahedron_tiles()),'gold'),(precise_reason_fixture,'edges')]:
            problem=factory();expected=independent_solutions(problem,target)
            for joint,event,slots in itertools.product((False,True),(False,True),(8,None)):
                engine=DomainCUDA(problem,target=target,joint_support=joint,event_propagation=event,precise_reasons=True,reason_slots=slots)
                try:
                    found=enumerate_engine(engine,19);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                finally:engine.close()
    def test_learning_and_incremental_strand_disabled_modes_preserve_completions(self):
        for factory,target in ((joint_hall_fixture,'edges'),(fork_gold_fixture,'gold')):
            problem=factory();expected=independent_solutions(problem,target)
            for learning,strands in itertools.product((False,True),repeat=2):
                engine=DomainCUDA(problem,target=target,precise_reasons=True,reason_slots=8,learning=learning,incremental_strands=strands)
                try:
                    found=enumerate_engine(engine,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
                finally:engine.close()

    def test_cached_reasons_and_root_proofs_are_independently_entailed_at_every_primitive(self):
        for slots in (8,None):
            problem=joint_hall_fixture();expected=independent_solutions(problem,'edges');engine=DomainCUDA(problem,target='edges',precise_reasons=True,reason_slots=slots)
            seen=set()
            try:
                for iteration in range(4000):
                    states=engine.step(1);saved=engine.checkpoint();data=saved['state'];depth=int(data[1,0]);phase=int(data[6,0])
                    for cell in range(problem.n):
                        zeros=sum(int(data[engine.offsets['precise_zero']+cell*engine.words+w,0])<<(32*w) for w in range(engine.words))
                        self.assertFalse(any(zeros&(1<<answer[cell]) for answer in expected-seen))
                    keys=data[engine.offsets['precise_keys']:engine.offsets['precise_keys']+engine.reason_slots,0]
                    for slot,key in enumerate(keys):
                        if key in (0,0xffffffff):continue
                        cell,code=divmod(int(key)-1,engine.nc)
                        why=sum(int(data[engine.offsets['precise_why']+slot*engine.reason_words+w,0])<<(32*w) for w in range(engine.reason_words))
                        if phase not in (3,4):self.assertEqual(why>>depth,0)
                        assumptions=[(int(data[engine.offsets['frame_cell']+level,0]),int(data[engine.offsets['frame_code']+level,0])) for level in range(problem.n) if why>>level&1]
                        self.assertFalse(any(answer[cell]==code and all(answer[c]==value for c,value in assumptions) for answer in expected-seen))
                    cubes=engine.remaining_domain_cubes();frontier_snapshot(WorkScope.for_problem(problem,'edges'),cubes)
                    remaining={answer for answer in expected if any(all(mask&(1<<value) for mask,value in zip(cube,answer)) for cube in cubes)}
                    self.assertEqual(remaining,expected-seen)
                    for item in engine.candidates():seen.add(tuple(item['codes']));engine.acknowledge([item['lane']])
                    if iteration%13==0:
                        saved=engine.checkpoint();replacement=DomainCUDA(problem,target='edges',precise_reasons=True,reason_slots=slots);replacement.restore(saved)
                        self.assertTrue(np.array_equal(saved['state'],replacement.checkpoint()['state']));engine.close();engine=replacement
                    if all(states==2):break
                else:self.fail('Finite precise enumeration stalled')
                self.assertEqual(seen,expected)
            finally:engine.close()
    def test_collisions_saturation_undo_and_uncached_redeletion_preserve_state(self):
        problem=precise_reason_fixture();engine=DomainCUDA(problem,target='edges',precise_reasons=True,reason_slots=8)
        previous=set();history=set();collision=False;uncached_redelete=False
        try:
            for iteration in range(5000):
                states=engine.step(1);saved=engine.checkpoint();data=saved['state'];engine.restore(saved)
                removed=set(int(x) for x in data[engine.offsets['trail']:engine.offsets['trail']+int(data[2,0]),0])
                keys={int(x) for x in data[engine.offsets['precise_keys']:engine.offsets['precise_keys']+engine.reason_slots,0] if x not in (0,0xffffffff)}
                hashes=[(key*2654435761)&7 for key in keys];collision|=len(hashes)!=len(set(hashes))
                for encoded in removed-previous:
                    cell,code=divmod(encoded,engine.nc);zero=int(data[engine.offsets['precise_zero']+cell*engine.words+(code//32),0])>>(code%32)&1
                    if encoded in history and encoded+1 not in keys and not zero:uncached_redelete=True
                history|=removed;previous=removed
                if all(states==2):break
            else:self.fail('Finite collision enumeration stalled')
            self.assertTrue(collision);self.assertTrue(uncached_redelete)
            stats=engine.stats();self.assertGreater(stats['backjumps'],0);self.assertGreater(stats['precise_invalidations'],0)
            self.assertGreater(stats['precise_dropped'],0);self.assertGreater(stats['precise_fallbacks'],0)
        finally:engine.close()
    def test_malformed_cache_scope_key_and_count_rejected(self):
        import hashlib
        engine=DomainCUDA(precise_reason_fixture(),target='edges',precise_reasons=True,reason_slots=8)
        try:
            for mutation in ('root','zero','count','key'):
                saved=engine.checkpoint();data=saved['state']
                if mutation=='root':data[engine.offsets['precise_root'],0]=0
                elif mutation=='zero':data[engine.offsets['precise_zero'],0]=np.uint32(1<<31)
                elif mutation=='count':data[engine.offsets['precise_count'],0]=5
                else:data[engine.offsets['precise_keys'],0]=np.uint32(engine.n*engine.nc+1);data[engine.offsets['precise_count'],0]=1
                saved['state_sha256']=hashlib.sha256(data.tobytes()).hexdigest()
                with self.assertRaisesRegex(ValueError,'Precise|precise'):engine.restore(saved)
        finally:engine.close()
    def test_fullsize_root_zero_proofs_fork_resume_and_policy_change(self):
        problem=planted160_fixture();expected=independent_solutions(problem,'gold');parent=DomainCUDA(problem,joint_support=True,event_propagation=True,precise_reasons=True);child=None
        try:
            for _ in range(1000):
                states,ready=parent.propagate_root(23)
                if ready:break
            else:self.fail('Full160 precise preparation stalled')
            parent.restore(parent.checkpoint());child=parent.fork_root(parent.remaining_domain_cubes());child.restore(child.checkpoint())
            self.assertEqual(set(enumerate_engine(child,256)),expected)
            self.assertEqual(child.state_bytes,1010844)
        finally:
            if child is not None:child.close()
            parent.close()
        problem=joint_hall_fixture();checks=iter((False,True,True));options={'gpu_precise_reasons':True,'gpu_event_propagation':True,'gpu_propagated_forks':True}
        paused=solve(problem,target='edges',replicas=4,options=options,stop=lambda:next(checks,True))
        saved=json.loads(json.dumps(paused['checkpoint']));self.assertEqual(saved['native']['schema'],'diamond-domain-cuda-v4')
        for enabled in (False,True):
            resumed=solve(problem,target='edges',seconds=10,replicas=4,options={**options,'gpu_precise_reasons':enabled},resume=saved)
            self.assertEqual(resumed['status'],'edge_perfect');self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],int(not enabled))

class SharedGPUHostTests(unittest.TestCase):
    def test_bounded_harvest_cursor_visits_all_slots_and_lanes_without_starvation(self):
        # Pure device-read facade tests the sampling cursor, not proof authority.
        class Slice:
            def __init__(self,array):self.array=array
            def get(self):return self.array
        class Device:
            def __init__(self,array):self.array=array
            def __getitem__(self,key):return Slice(self.array[key])
        engine=DomainCUDA.__new__(DomainCUDA);engine.n=12;engine.lanes=2;engine.delivered=[]
        engine.offsets={'clause_length':29,'clause_cell':61,'clause_code':445};data=np.zeros((829,2),np.uint32)
        data[8]=32;data[29:61]=1
        for lane in range(2):
            for index in range(32):data[61+index*12,lane]=index//12+3*lane;data[445+index*12,lane]=3*(index%12)
        engine.state=Device(data);cursor=0;seen=[]
        for _ in range(8):
            rows,cursor,_=engine.learned_claims(cursor,limit=8);seen.extend(tuple(row[0]) for row in rows)
        self.assertEqual(cursor,0);self.assertEqual(len(seen),64);self.assertEqual(len(set(seen)),64)
    def test_immediate_stop_resume_keeps_saved_sharing_work_counters(self):
        from gpu_shared_round25 import GPUClauseExchange
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_pending_round25 import UnknownOracle
        problem=joint_hall_fixture();store=VerifiedConflictStore(problem,'edges');store._oracle=UnknownOracle();store.offer([(0,0)])
        exchange=GPUClauseExchange(problem,'edges',store=store)
        try:
            exchange.sync();payload=solve(problem,target='edges',options={'gpu_shared_clauses':True},stop=lambda:True)['checkpoint']
            payload['sharing']=exchange.snapshot();payload['sharing_stats']=exchange.public_stats();payload['digest']=digest({k:v for k,v in payload.items() if k!='digest'})
            with patch('gpu_domain_round24.DomainCUDA') as factory:
                answer=solve(problem,target='edges',options={'gpu_shared_clauses':True},resume=payload,stop=lambda:True)
            factory.assert_not_called()
            for key,value in exchange.public_stats().items():self.assertEqual(answer['stats'][key],value)
            self.assertEqual(answer['stats']['gpu_shared_verification_calls'],1)
        finally:exchange.close();store.close()

    def test_unknown_budget_and_discarded_identity_survive_exchange_restart(self):
        from gpu_shared_round25 import GPUClauseExchange
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_pending_round25 import UnknownOracle
        problem=joint_hall_fixture();saved=None
        for attempt in range(3):
            store=VerifiedConflictStore(problem,'edges');store._oracle=UnknownOracle()
            exchange=GPUClauseExchange(problem,'edges',store=store,resume=saved)
            try:
                if saved is None:store.offer([(0,0)])
                self.assertFalse(exchange.sync())
                saved=json.loads(json.dumps(exchange.snapshot()));self.assertEqual(saved['verified']['records'],[])
                self.assertEqual(exchange.counters['gpu_shared_verification_calls'],attempt+1)
            finally:exchange.close();store.close()
        store=VerifiedConflictStore(problem,'edges');exchange=GPUClauseExchange(problem,'edges',store=store,resume=saved)
        try:self.assertFalse(store.offer([(0,0)]));self.assertEqual(store.pending_count,0);self.assertTrue(saved['pending']['discarded'])
        finally:exchange.close();store.close()
    def test_pending_capacity_preserves_old_claims_before_harvest(self):
        from gpu_shared_round25 import GPUClauseExchange
        from constraint_shared_conflicts import VerifiedConflictStore,MAX_PENDING
        from test_constraint_shared_pending_round25 import UnknownOracle
        from test_constraint_separator import ring
        problem=ring(34);store=VerifiedConflictStore(problem,'edges');store._oracle=UnknownOracle()
        for a,b in itertools.islice(itertools.combinations(range(34),2),MAX_PENDING):store.offer([(a,0),(b,0)])
        before={tuple(tuple(pair) for pair in row['placements']) for row in store.pending_snapshot()['pending']}
        class Engine:
            delivered=[]
            def learned_claims(self,*args,**kwargs):raise AssertionError('Full pending queue must not be evicted by new harvest')
        exchange=GPUClauseExchange(problem,'edges',store=store)
        try:
            self.assertFalse(exchange.sync(Engine()));after={tuple(tuple(pair) for pair in row['placements']) for row in store.pending_snapshot()['pending']}
            self.assertEqual(before,after);self.assertEqual(store.stats['shared_conflicts_pending_dropped'],0)
        finally:exchange.close();store.close()
    def test_malformed_pending_transport_is_atomic_for_caller_store(self):
        from gpu_shared_round25 import GPUClauseExchange
        from constraint_shared_conflicts import VerifiedConflictStore,_digest
        problem=joint_hall_fixture();donor=GPUClauseExchange(problem,'edges');store=VerifiedConflictStore(problem,'edges')
        try:
            donor.store.offer([(0,0)]);saved=donor.snapshot();saved['pending']['pending'][0]['attempts']=3
            saved['pending']['digest']=_digest(saved['pending']);saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
            before=store.pending_snapshot()
            with self.assertRaisesRegex(ValueError,'attempt'):GPUClauseExchange(problem,'edges',store=store,resume=saved)
            self.assertEqual(store.pending_snapshot(),before);self.assertEqual(store.records(),[])
        finally:donor.close();store.close()
    def test_cancelled_verifier_keeps_untrusted_claim_and_attempt(self):
        from gpu_shared_round25 import GPUClauseExchange,SharingStopped
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_pending_round25 import UnknownOracle
        problem=joint_hall_fixture();store=VerifiedConflictStore(problem,'edges');oracle=UnknownOracle();store._oracle=oracle;store.offer([(0,0)])
        exchange=GPUClauseExchange(problem,'edges',store=store)
        def check():
            if oracle.called:raise SharingStopped()
        try:
            with self.assertRaises(SharingStopped):exchange.sync(check=check)
            saved=exchange.snapshot();self.assertEqual(saved['pending']['pending'][0]['attempts'],0);self.assertEqual(saved['verified']['records'],[])
        finally:exchange.close();store.close()
    def test_stopped_runner_creates_no_gpu_or_native_verifier(self):
        with patch('gpu_domain_round24.DomainCUDA') as factory:
            result=solve(joint_hall_fixture(),target='edges',options={'gpu_shared_clauses':True},stop=lambda:True)
        factory.assert_not_called();self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['stats']['gpu_shared_verification_calls'],0);self.assertIn('sharing',result['checkpoint'])

class SharedGPUVerifierTests(unittest.TestCase):
    def setUp(self):
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional python-sat verifier')
    def test_verified_empty_original_core_finishes_without_gpu_and_records_scope(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from constraint_work import CoverageLedger,restore_frontier
        from test_constraint_shared_conflicts import finish
        from pathlib import Path
        from tempfile import TemporaryDirectory
        problem=precise_reason_fixture();store=VerifiedConflictStore(problem,'edges');store.offer([]);finish(store)
        self.assertEqual(store.records(),[[]])
        try:
            with TemporaryDirectory() as folder:
                path=Path(folder)/'coverage.sqlite3'
                with patch('gpu_domain_round24.DomainCUDA') as factory:
                    answer=solve(problem,target='edges',options={'gpu_shared_clauses':True,'shared_coverage':True},learning_state={'_shared_conflicts':store,'_coverage_path':str(path)})
                factory.assert_not_called();self.assertEqual(answer['status'],'infeasible');self.assertTrue(answer['complete'])
                self.assertEqual(answer['stats']['gpu_verified_root_unsat'],1);self.assertEqual(answer['nodes'],0)
                scope=WorkScope.for_problem(problem,'edges');self.assertEqual(restore_frontier(answer['checkpoint']['frontier'],scope),())
                with CoverageLedger(path,scope) as ledger:self.assertEqual(ledger.exclusions(),(scope.root,))
                self.assertEqual(answer['shared_conflicts']['records'],[[]])
        finally:store.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA round25 exact tests')
class SharedGPUCUDATests(unittest.TestCase):
    def setUp(self):
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional python-sat verifier')
    def verified_store(self,problem):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        store=VerifiedConflictStore(problem,'edges');store.offer([(0,0)]);finish(store)
        self.assertIn([[0,0]],store.records());self.addCleanup(store.close);return store
    def test_actual_local_claims_are_reproved_and_satisfiable_claims_rejected(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        problem=joint_hall_fixture();root=list(WorkScope.for_problem(problem,'edges').root)
        root[1]=(1<<0)|(1<<9);root[2]=root[3]=(1<<0)|(1<<6);source=DomainCUDA(problem,[tuple(root)],target='edges')
        store=VerifiedConflictStore(problem,'edges');expected=independent_solutions(problem,'edges')
        try:
            for _ in range(3000):
                state=source.step(1)
                if any(state==1):break
            else:self.fail('Local source did not reach its first candidate')
            claims,_,_=source.learned_claims(limit=32)
            local_only=[row for row in claims if any(all(answer[c]==v for c,v in row) for answer in expected)]
            self.assertTrue(local_only)
            for row in claims:store.offer(row)
            finish(store);self.assertGreater(store.stats['shared_conflicts_rejected'],0)
            for row in store.records():self.assertFalse(any(all(answer[c]==v for c,v in row) for answer in expected))
            for row in local_only:self.assertNotIn([list(pair) for pair in sorted(row)],store.records())
        finally:source.close();store.close()
    def test_verified_cross_lane_reuse_preserves_all_solutions_and_reduces_work(self):
        from constraint_work import subtract,disjoint
        problem=joint_hall_fixture();all_answers=independent_solutions(problem,'edges');store=self.verified_store(problem)
        root=WorkScope.for_problem(problem,'edges').root;producer=list(root);producer[1]=(1<<0)|(1<<9);producer[2]=producer[3]=(1<<0)|(1<<6)
        parts=list(subtract(root,tuple(producer)));self.assertTrue(disjoint([tuple(producer),*parts]))
        expected={answer for answer in all_answers if not all(mask&(1<<code) for mask,code in zip(producer,answer))}
        self.assertEqual(len(expected),4);stats=[]
        for enabled in (False,True):
            engine=DomainCUDA(problem,parts,target='edges',event_propagation=True,precise_reasons=True)
            try:
                if enabled:
                    result=engine.import_verified_clauses(store);self.assertEqual(result['inserted'],len(parts))
                    self.assertEqual(engine.import_verified_clauses(store)['inserted'],0)
                    engine.restore(engine.checkpoint())
                found=enumerate_engine(engine,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected));stats.append(engine.stats())
            finally:engine.close()
        self.assertLess(stats[1]['decisions'],stats[0]['decisions'])
    def test_verified_gold_clauses_preserve_full_solution_set_through_native_resume(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        problem=fork_gold_fixture();expected=independent_solutions(problem,'gold');store=VerifiedConflictStore(problem,'gold')
        store.offer([(10,30),(11,30)]);finish(store);self.assertTrue(store.records())
        engine=DomainCUDA(problem,joint_support=True,event_propagation=True,precise_reasons=True)
        try:
            self.assertGreater(engine.import_verified_clauses(store)['inserted'],0)
            engine.step(23);saved=engine.checkpoint();engine.restore(saved)
            found=enumerate_engine(engine,17);self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
            with self.assertRaisesRegex(ValueError,'enumeration'):engine.learned_claims()
        finally:engine.close();store.close()

    def test_injection_at_each_live_phase_preserves_unfinished_solutions(self):
        problem=joint_hall_fixture();expected=independent_solutions(problem,'edges');store=self.verified_store(problem)
        source=DomainCUDA(problem,target='edges',event_propagation=True,precise_reasons=True);phases=set()
        try:
            for _ in range(500):
                source.step(1);saved=source.checkpoint();phase=int(saved['state'][6,0])
                if phase not in phases and int(saved['state'][0,0])==0:
                    phases.add(phase);receiver=DomainCUDA(problem,target='edges',event_propagation=True,precise_reasons=True)
                    try:
                        receiver.restore(saved);receiver.import_verified_clauses(store);receiver.restore(receiver.checkpoint())
                        self.assertEqual(set(enumerate_engine(receiver,17)),expected)
                    finally:receiver.close()
                if phases>={0,1,2,3}:break
            self.assertTrue(phases>={0,1,2,3})
        finally:source.close()
    def test_full160_verified_unit_preserves_planted_gold(self):
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        problem=planted160_fixture();expected=independent_solutions(problem,'gold');answer=next(iter(expected))
        wrong=next(code for code in problem.domains[0] if code!=answer[0]);store=VerifiedConflictStore(problem,'gold');store.offer([(0,wrong)]);finish(store)
        self.assertTrue(store.records());engine=DomainCUDA(problem,joint_support=True,event_propagation=True,precise_reasons=True)
        try:
            self.assertGreater(engine.import_verified_clauses(store)['inserted'],0)
            engine.restore(engine.checkpoint());self.assertEqual(set(enumerate_engine(engine,256)),expected)
        finally:engine.close();store.close()

    def test_runner_initial_injection_native_resume_and_pending_transport(self):
        problem=joint_hall_fixture();store=self.verified_store(problem);options={'gpu_shared_clauses':True}
        answer=solve(problem,target='edges',seconds=10,replicas=4,options=options,learning_state={'_shared_conflicts':store})
        self.assertEqual(answer['status'],'edge_perfect');self.assertGreater(answer['stats']['gpu_shared_imports'],0)
        self.assertIn('shared_conflicts',answer);self.assertIn('sharing',answer['checkpoint'])
        snapshot=json.loads(json.dumps(answer['checkpoint']));before=copy.deepcopy(snapshot)
        resumed=solve(problem,target='edges',seconds=10,replicas=4,options=options,resume=snapshot)
        self.assertEqual(resumed['status'],'edge_perfect');self.assertEqual(snapshot,before)
        self.assertEqual(resumed['stats']['gpu_domain_portable_restart'],0)


class LanePoolHostTests(unittest.TestCase):
    def test_queue_and_owner_checkpoint_roundtrip_and_all_or_nothing_validation(self):
        from gpu_lane_pool_round25 import LanePool
        problem=precise_reason_fixture();scope=WorkScope.for_problem(problem,'edges');parts=split_cubes([scope.root],8)
        pool=LanePool(scope,parts);snapshot=pool.snapshot();before=copy.deepcopy(snapshot)
        restored=LanePool.restore(scope,parts,None,snapshot)
        self.assertEqual(restored.snapshot(),snapshot);self.assertEqual(snapshot,before)
        for field,value in [('version',True),('next_id',0),('lanes',[None]),('queued_owners',[999]*8)]:
            bad=copy.deepcopy(snapshot);bad[field]=value;bad['digest']=digest({k:v for k,v in bad.items() if k!='digest'})
            with self.assertRaises(ValueError):LanePool.restore(scope,parts,None,bad)
        self.assertEqual(pool.snapshot(),before)
    def test_cancel_before_allocation_and_policy_change_keep_exact_queue(self):
        from constraint_work import restore_frontier
        problem=precise_reason_fixture();options={'gpu_lane_balance':True}
        with patch('gpu_domain_round24.DomainCUDA',side_effect=AssertionError('Unexpected GPU allocation')):
            first=solve(problem,replicas=4,target='edges',options=options,stop=lambda:True)
            again=solve(problem,replicas=8,target='edges',options=options,resume=first['checkpoint'],stop=lambda:True)
        self.assertEqual(first['checkpoint'],again['checkpoint'])
        self.assertEqual(len(restore_frontier(first['checkpoint']['frontier'],WorkScope.for_problem(problem,'edges'))),4)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA lane ownership tests')
class LanePoolCUDATests(unittest.TestCase):
    def test_selected_replacement_keeps_other_columns_exact_and_counters_retire_once(self):
        from gpu_lane_pool_round25 import LanePool
        problem=precise_reason_fixture();scope=WorkScope.for_problem(problem,'edges');pool=LanePool(scope,split_cubes([scope.root],12))
        engine=DomainCUDA(problem,pool.take_batch(4),target='edges',event_propagation=True,precise_reasons=True)
        try:
            for _ in range(500):
                states=engine.step(1)
                if any(states==2):break
            else:self.fail('Expected finite lane completion')
            old,retired,_=pool.retire(engine,states);self.assertGreater(retired,0)
            self.assertEqual(pool.retire(engine,states),({},0,[]))
            untouched=[i for i,x in enumerate(pool.lanes) if x is not None];before=engine.state[:,untouched].get()
            self.assertTrue(pool.refill(engine));np.testing.assert_array_equal(engine.state[:,untouched].get(),before)
            self.assertGreater(old['support_checks'],0);engine.restore(engine.checkpoint())
            restored=LanePool.restore(scope,pool.queue,engine,pool.snapshot());self.assertEqual(restored.snapshot(),pool.snapshot())
        finally:engine.close()
    def test_donation_preserves_all_remaining_answers_and_unaffected_native_columns(self):
        from gpu_lane_pool_round25 import LanePool
        from constraint_work import disjoint
        for features in ({},{'joint_support':True,'event_propagation':True,'precise_reasons':True}):
            problem=joint_hall_fixture();scope=WorkScope.for_problem(problem,'edges');pool=LanePool(scope,split_cubes([scope.root],4))
            engine=DomainCUDA(problem,pool.take_batch(4),target='edges',**features)
            try:
                for _ in range(2000):
                    states=engine.step(1)
                    self.assertFalse(any(states==1),'Need the donation barrier before first solution')
                    if any(states==2):
                        pool.retire(engine,states);before=engine.state.get();old=pool.donate(engine)
                        if old is not None:break
                else:self.fail('No donated branch')
                touched=[i for i in range(engine.lanes) if not np.array_equal(before[:,i],engine.state[:,i].get())]
                self.assertEqual(len(touched),2)
                self.assertGreater(pool.counters['gpu_lane_donations'],0)
                engine.restore(engine.checkpoint());LanePool.restore(scope,pool.queue,engine,pool.snapshot())
                remaining=engine.remaining_domain_cubes()+pool.queue;self.assertTrue(disjoint(remaining))
                expected=independent_solutions(problem,'edges');found=enumerate_engine(engine,17)
                if pool.queue:
                    other=DomainCUDA(problem,pool.queue,target='edges',**features)
                    try:found+=enumerate_engine(other,17)
                    finally:other.close()
                self.assertEqual(set(found),expected);self.assertEqual(len(found),len(expected))
            finally:engine.close()
    def test_propagated_parent_with_independent_extra_root_copies_only_its_children(self):
        problem=fork_gold_fixture();scope=WorkScope.for_problem(problem,'gold');roots=split_cubes([scope.root],2)
        parent=DomainCUDA(problem,[roots[0]],joint_support=True,event_propagation=True,precise_reasons=True)
        child=None
        try:
            for _ in range(1000):
                states,ready=parent.propagate_root(17)
                if ready:break
            self.assertFalse(any(states==2));parts=split_cubes(parent.remaining_domain_cubes(),2)
            child=parent.fork_root(parts,extra_cubes=[roots[1]])
            self.assertEqual(int(child.state[9,-1].get()),0)
            np.testing.assert_array_equal(child.state[child.offsets['board']:child.offsets['board']+problem.n,-1].get(),np.full(problem.n,0xffffffff,np.uint32))
            child.restore(child.checkpoint());found=enumerate_engine(child,17)
            self.assertEqual(set(found),independent_solutions(problem,'gold'))
        finally:
            if child:child.close()
            parent.close()
    def test_runner_stop_after_donation_resume_and_policy_rebase(self):
        from gpu_lane_pool_round25 import LanePool
        problem=precise_reason_fixture();options={'gpu_lane_balance':True,'gpu_event_propagation':True,'gpu_precise_reasons':True};flag=[False]
        real=LanePool.donate
        def donate(pool,engine):
            value=real(pool,engine)
            if value is not None:flag[0]=True
            return value
        real_step=DomainCUDA.step
        with patch.object(DomainCUDA,'step',lambda engine,work:real_step(engine,1)),patch.object(LanePool,'donate',donate):first=solve(problem,seconds=10,target='edges',replicas=4,options=options,stop=lambda:flag[0])
        self.assertEqual(first['status'],'stopped');self.assertGreater(first['stats']['gpu_lane_donations'],0)
        snapshot=copy.deepcopy(first['checkpoint'])
        for replicas,enabled in [(4,True),(2,True),(4,False)]:
            answer=solve(problem,seconds=10,target='edges',replicas=replicas,options={**options,'gpu_lane_balance':enabled},resume=snapshot)
            self.assertTrue(answer['complete']);self.assertEqual(answer['status'],'infeasible')
            self.assertGreaterEqual(answer['nodes'],first['nodes']);self.assertEqual(snapshot,first['checkpoint'])
            self.assertEqual(answer['stats']['gpu_domain_portable_restart'],int(replicas!=4 or not enabled))
    def test_owner_coverage_waits_for_all_donated_children_and_queued_refill(self):
        import tempfile
        from pathlib import Path
        from constraint_work import CoverageLedger
        from gpu_lane_pool_round25 import LanePool
        problem=precise_reason_fixture();scope=WorkScope.for_problem(problem,'edges');options={'gpu_lane_balance':True,'gpu_propagated_forks':True,'shared_coverage':True,'gpu_event_propagation':True,'gpu_precise_reasons':True}
        real=LanePool.donate;flag=[False]
        def donate(pool,engine):
            value=real(pool,engine)
            if value is not None:flag[0]=True
            return value
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'coverage.sqlite';learning={'_coverage_path':str(path)}
            real_step=DomainCUDA.step
            with patch.object(DomainCUDA,'step',lambda engine,work:real_step(engine,1)),patch.object(LanePool,'donate',donate):first=solve(problem,target='edges',seconds=10,replicas=4,options=options,learning_state=learning,stop=lambda:flag[0])
            self.assertEqual(first['status'],'stopped')
            with CoverageLedger(path,scope) as ledger:self.assertEqual(ledger.exclusions(),())
            answer=solve(problem,target='edges',seconds=10,replicas=4,options=options,learning_state=learning,resume=first['checkpoint'])
            self.assertEqual(answer['status'],'infeasible');self.assertEqual(answer['stats']['gpu_lane_owners_completed'],1)
            with CoverageLedger(path,scope) as ledger:self.assertEqual(ledger.exclusions(),(scope.root,))
            # More queued input roots than hardware lanes exercise actual refills.
            frontier=frontier_snapshot(scope,split_cubes([scope.root],12))
            answer=solve(problem,target='edges',seconds=10,replicas=4,options={'gpu_lane_balance':True},work_frontier=frontier)
            self.assertEqual(answer['status'],'infeasible');self.assertGreaterEqual(answer['stats']['gpu_lane_refills'],8)
    def test_verified_clauses_return_to_actual_reloaded_lanes(self):
        from gpu_shared_round25 import GPUClauseExchange
        from gpu_lane_pool_round25 import LanePool
        from constraint_shared_conflicts import VerifiedConflictStore
        from test_constraint_shared_conflicts import finish
        from test_constraint_sat_assist import HAS_SAT
        if not HAS_SAT:self.skipTest('Optional python-sat verifier')
        problem=joint_hall_fixture();scope=WorkScope.for_problem(problem,'edges');left=list(scope.root);right=list(scope.root)
        left[0]=1;right[0]=1<<3;pool=LanePool(scope,[tuple(left),*split_cubes([tuple(right)],3)])
        store=VerifiedConflictStore(problem,'edges');store.offer([(0,0)]);finish(store)
        engine=DomainCUDA(problem,pool.take_batch(2),target='edges');exchange=GPUClauseExchange(problem,'edges',store=store)
        try:
            exchange.refresh(engine);first_imports=exchange.public_stats()['gpu_shared_imports'];self.assertEqual(first_imports,2)
            for _ in range(100):
                states=engine.step(1)
                if any(states==2):break
            else:self.fail('Global cut did not retire the impossible lane')
            pool.retire(engine,states);self.assertTrue(pool.refill(engine))
            self.assertEqual(int(engine.state[8,0].get()),0)
            exchange.invalidate();self.assertTrue(exchange.needs_initial_refresh(engine));exchange.refresh(engine)
            self.assertGreater(exchange.public_stats()['gpu_shared_imports'],first_imports)
            self.assertGreater(int(engine.state[8,0].get()),0)
            found=enumerate_engine(engine,17)
            if pool.queue:
                other=DomainCUDA(problem,pool.queue,target='edges')
                try:found+=enumerate_engine(other,17)
                finally:other.close()
            self.assertEqual(set(found),independent_solutions(problem,'edges'));self.assertEqual(len(found),6)
        finally:engine.close();exchange.close();store.close()
    def test_full160_witness_and_shared_clause_reapplication(self):
        from gpu_shared_round25 import GPUClauseExchange
        problem=planted160_fixture();answer=solve(problem,seconds=10,replicas=4,options={'gpu_lane_balance':True,'gpu_propagated_forks':True,'gpu_event_propagation':True,'gpu_precise_reasons':True})
        self.assertEqual(answer['status'],'solved');self.assertIn(tuple(answer['codes']),independent_solutions(problem,'gold'))
        exchange=object.__new__(GPUClauseExchange);exchange._refresh_engine=object();exchange._refresh_signature=('old',)
        exchange.invalidate();self.assertIsNone(exchange._refresh_engine);self.assertIsNone(exchange._refresh_signature)

if __name__=='__main__':unittest.main(verbosity=2)

