"""Independent finite GPU deductions and persistent dynamic-stack checks."""
import copy
from itertools import permutations, product
import os
import unittest
import numpy as np
from dfs_gpu import DFS, fill_order
from gpu_reasoning import normalize_features, pack_frame, frame_cells, prepare_policy
from test_gpu_gold import gold_fixture
from validator import orientation_masks, validate_arrangement

# Optional structured finite witnesses for the audited hardware runner.
REASONING_EVIDENCE = []


def varied_gold_fixture(seed=421):
    data, board = gold_fixture()
    rng = np.random.default_rng(seed)
    maps = {}
    for a, sa, b, sb in board.edges:
        pair = rng.choice(np.arange(1,12),2,replace=False).tolist()
        maps[a,sa,5], maps[a,sa,7] = pair
        maps[b,sb,7], maps[b,sb,5] = 12-pair[0],12-pair[1]
    for cell,tile in enumerate(data['tiles']):
        tile['segments'] = [[[side,maps[cell,side,position]] for side,position in segment] for segment in tile['segments']]
    return data,board


def inventory_fixture(shortage=True):
    from geometry import build_board
    board=build_board();holes=[]
    for cell in range(160):
        if all(cell not in board.neighbor_cells[other] and not set(board.neighbor_cells[cell])&set(board.neighbor_cells[other]) for other in holes):holes.append(cell)
        if len(holes)==3:break
    masks=np.zeros((480,3),np.uint16)
    demand={cell:(80 if shortage or index<2 else 136) for index,cell in enumerate(holes)}
    base=np.zeros((160,3),np.uint16)
    for cell in range(160):
        if cell in holes:continue
        for side,other in enumerate(board.neighbor_cells[cell]):
            if other in demand:base[cell,side]=demand[other]
    base[holes[0]]=80;base[holes[1]]=80;base[holes[2]]=0 if shortage else 136
    for cell in range(160):
        for rotation in range(3):masks[3*cell+rotation]=np.roll(base[cell],rotation)
    order=np.array([cell for cell in range(160) if cell not in holes]+holes,np.int16)
    return masks,board,order


def disconnected_open_fixture():
    from geometry import build_board
    board=build_board();tiles=[{'id':str(i),'segments':[]} for i in range(160)]
    group={cell.index for cell in board.cells if cell.face==0}
    for cells in (group,set(range(160))-group):
        edges=[];adjacency={cell:[] for cell in cells}
        for a,sa,b,sb in board.edges:
            if a not in cells or b not in cells:continue
            for position in (5,7):
                edge=len(edges);edges.append((a,(sa,position),b,(sb,12-position)))
                adjacency[a].append(edge);adjacency[b].append(edge)
        used=set();vertices=[min(cells)];path=[];tour=[]
        while vertices:
            cell=vertices[-1]
            while adjacency[cell] and adjacency[cell][-1] in used:adjacency[cell].pop()
            if adjacency[cell]:
                edge=adjacency[cell].pop();used.add(edge);a,ap,b,bp=edges[edge]
                oriented=(a,ap,b,bp) if a==cell else (b,bp,a,ap)
                vertices.append(oriented[2]);path.append(oriented)
            else:
                vertices.pop()
                if path:tour.append(path.pop())
        tour.reverse()
        for edge,following in zip(tour,tour[1:]+tour[:1]):
            assert edge[2]==following[0]
            tiles[edge[2]]['segments'].append([list(edge[3]),list(following[1])])
    interior=[cell for cell in sorted(group) if set(board.neighbor_cells[cell])<=group]
    a=interior[0];b=next(cell for cell in interior if cell!=a and cell not in board.neighbor_cells[a])
    c=next(cell for cell in range(160) if cell not in group and not set(board.neighbor_cells[cell])&group)
    holes=[a,b,c];order=np.array([cell for cell in range(160) if cell not in holes]+holes,np.int16)
    return {'tiles':tiles},board,order


class GPUReasoningHostTests(unittest.TestCase):
    def test_frame_encoding_preserves_every_cell_and_bucket_shift(self):
        for cell in range(160):
            for first in (0,479,480,1919,1920,3359):
                for shift in (0,1,479):
                    a,b=pack_frame(first,shift,cell)
                    self.assertEqual(int(frame_cells(a,b)),cell)
                    self.assertEqual(a&4095,first)
                    self.assertEqual(b&4095,shift)
        with self.assertRaises(ValueError):pack_frame(4096,0,0)
        with self.assertRaises(ValueError):normalize_features({'hall':'yes'})
        with self.assertRaises(ValueError):normalize_features({'hallucinate':True})

    def test_policy_identity_binds_canonical_reduction_and_branching(self):
        data,board=gold_fixture();masks=np.asarray(orientation_masks(data),np.uint16)
        seen=[]
        for features in ({},{'symmetry':True},{'dynamic_mrv':True},{'forced':True}):
            engine=DFS.__new__(DFS)
            engine._prepare_host(masks,board.neighbor_cells,board.neighbor_sides,1,None,None,[])
            engine._prepare_gold(data);prepare_policy(engine,features,data)
            seen.append(engine.fingerprint)
            if features.get('symmetry'):self.assertEqual(int(engine.host_policy[:160].sum()),16)
        self.assertEqual(len(set(seen)),4)

    def test_shared_gold_exclusion_is_independently_verified(self):
        data,board=gold_fixture(False);masks=np.asarray(orientation_masks(data),np.uint16)
        engine=DFS.__new__(DFS)
        engine._prepare_host(masks,board.neighbor_cells,board.neighbor_sides,1,None,None,[])
        engine._prepare_gold(data);prepare_policy(engine,{'shared_gold':True},data)
        engine._upload_policy=lambda:None
        a,_,b,_=board.edges[0]
        self.assertFalse(engine.add_gold_exclusion([(a,3*a)]))
        self.assertTrue(engine.add_gold_exclusion([(a,3*a),(b,3*b)]))
        self.assertFalse(engine.add_gold_exclusion([(a,3*a),(b,3*b)]))
        self.assertEqual(int(engine.host_policy[160]),1)
        with self.assertRaises(ValueError):engine.add_gold_exclusion([(160,0)])

    def test_varied_fixture_is_gold(self):
        for seed in (421,422):
            data,board=varied_gold_fixture(seed)
            self.assertTrue(validate_arrangement(data,list(range(0,480,3)),board)['valid'])


class ReasoningHardwareMixin:
    def make(self,data,board,**kwargs):
        result=self.factory(np.asarray(orientation_masks(data),np.uint16),board.neighbor_cells,board.neighbor_sides,**kwargs)
        if hasattr(result,'close'):self.addCleanup(result.close)
        return result

    def enumerate_tail(self,features,varied):
        data,board=varied_gold_fixture() if varied else gold_fixture()
        order=fill_order(board.neighbor_cells);base=np.arange(160)*3
        expected=set()
        for pieces in permutations(order[-3:]):
            for rotations in product(range(3),repeat=3):
                codes=base.copy();codes[order[-3:]]=[3*int(piece)+r for piece,r in zip(pieces,rotations)]
                if validate_arrangement(data,codes.astype(int).tolist(),board)['valid']:expected.add(tuple(map(int,codes)))
        self.assertTrue(expected)
        gpu=self.make(data,board,n=1,order=order,prefixes=[base[order[:-3]].tolist()],gold_data=data,gpu_features=features)
        shared_cut_loaded=False
        if features.get('shared_gold') and not varied:
            # Exercise a nonempty device clause table using an independently
            # verified proper-loop witness from an alternative rotation.
            for cell in [*order[-3:],*order[:-3]]:
                for rotation in (1,2):
                    trial=base.copy();trial[cell]+=rotation
                    literals=[(i,int(code)) for i,code in enumerate(trial)]
                    if gpu.add_gold_exclusion(literals):shared_cut_loaded=True;break
                if shared_cut_loaded:break
            self.assertTrue(shared_cut_loaded)
        found=set();resumed=False
        for step in range(2000):
            gpu.step(16);gpu.verify()
            if not resumed and int(gpu.states.get()[0])==0 and int(gpu.depths.get()[0])>157:
                saved=gpu.snapshot_checkpoint()
                other=self.make(data,board,n=1,order=order,prefixes=[],gold_data=data,gpu_features=features)
                other.restore_checkpoint(saved);other.verify();gpu=other;resumed=True
            state=int(gpu.states.get()[0])
            if state==1:
                candidate=tuple(map(int,gpu.boards[:,0].get()));self.assertNotIn(candidate,found);found.add(candidate)
                gpu.acknowledge([0]);gpu.verify()
            elif state==2:break
        else:self.fail('Finite tail did not exhaust')
        self.assertEqual(found,expected)
        counters=gpu.counters.get().sum(axis=1).tolist()
        REASONING_EVIDENCE.append({'check':'complete_gold_tail','backend':type(self).__name__,
            'features':features,'varied':varied,'remaining_cells':3,'labelled_candidates':162,
            'expected_gold_solutions':len(expected),'found_gold_solutions':len(found),
            'checkpoint_resumed':resumed,'verified_shared_cut_loaded':shared_cut_loaded,'counters':counters})
        return counters

    def test_each_feature_and_combination_preserves_complete_finite_tail(self):
        configs=[{},*[{name:True} for name in ('remote_check','hall','forced','dynamic_mrv','connectivity','shared_gold','symmetry')],
                 dict.fromkeys(('remote_check','hall','forced','dynamic_mrv','connectivity','shared_gold','symmetry'),True)]
        for features in configs:
            with self.subTest(features=features):self.enumerate_tail(features,True)
        self.enumerate_tail({'dynamic_mrv':True,'forced':True,'remote_check':True,'hall':True,'shared_gold':True},False)

    def test_remote_and_hall_reject_strictly_earlier_than_local_forward_check(self):
        masks,board,order=inventory_fixture(True);counts=[]
        for features in ({},{'remote_check':True},{'hall':True}):
            gpu=self.factory(masks,board.neighbor_cells,board.neighbor_sides,n=1,order=order,
                             prefixes=[(3*order[:-3]).tolist()],gpu_features=features,seed=17)
            if hasattr(gpu,'close'):self.addCleanup(gpu.close)
            for _ in range(100):
                gpu.step(128);gpu.verify()
                if int(gpu.states.get()[0])==2:break
                self.assertNotEqual(int(gpu.states.get()[0]),1)
            else:self.fail('Finite Hall-shortage tail did not exhaust')
            counts.append(int(gpu.counters.get()[0,0]))
        self.assertGreater(counts[0],counts[1]);self.assertGreater(counts[1],counts[2])
        REASONING_EVIDENCE.append({'check':'remote_and_hall_shortage','backend':type(self).__name__,
            'main_candidates_consumed':dict(zip(('baseline','remote_check','hall'),counts)),
            'remaining_cells':3,'gold_or_edge_solutions':0})

    def test_dynamic_mrv_reduces_placement_nodes_and_preserves_every_edge_solution(self):
        masks,board,order=inventory_fixture(False);results=[];consumed=[]
        expected=set()
        from gpu_engine import reverse11
        reversed_masks=reverse11(masks)
        for pieces in permutations(order[-3:]):
            for rotations in product(range(3),repeat=3):
                codes=np.arange(160)*3;codes[order[-3:]]=[3*int(piece)+rotation for piece,rotation in zip(pieces,rotations)]
                if all(masks[codes[a],sa]==reversed_masks[codes[b],sb] for a,sa,b,sb in board.edges):expected.add(tuple(map(int,codes)))
        for features in ({},{'dynamic_mrv':True}):
            gpu=self.factory(masks,board.neighbor_cells,board.neighbor_sides,n=1,order=order,
                             prefixes=[(3*order[:-3]).tolist()],gpu_features=features,seed=17)
            if hasattr(gpu,'close'):self.addCleanup(gpu.close)
            found=set()
            for _ in range(1000):
                gpu.step(128);gpu.verify();state=int(gpu.states.get()[0])
                if state==1:
                    codes=tuple(map(int,gpu.boards[:,0].get()));self.assertNotIn(codes,found);found.add(codes);gpu.acknowledge([0])
                if state==2:break
            else:self.fail('Finite MRV tail did not exhaust')
            self.assertEqual(found,expected);results.append(int(gpu.counters.get()[1,0]));consumed.append(int(gpu.counters.get()[0,0]))
        self.assertLess(results[1],results[0])
        REASONING_EVIDENCE.append({'check':'dynamic_mrv_complete_edge_tail','backend':type(self).__name__,
            'accepted_placements':dict(zip(('baseline','dynamic_mrv'),results)),
            'main_candidates_consumed':dict(zip(('baseline','dynamic_mrv'),consumed)),
            'remaining_cells':3,'expected_edge_solutions':len(expected)})

    def test_optimistic_connectivity_rejects_open_disconnected_strands(self):
        from search_problem import SearchProblem
        data,board,order=disconnected_open_fixture();problem=SearchProblem(data)
        partial=[-1]*160
        for cell in order[:-3]:partial[int(cell)]=3*int(cell)
        self.assertFalse(problem.partial_loop_impossible(partial))
        counts=[]
        for features in ({},{'connectivity':True}):
            gpu=self.make(data,board,n=1,order=order,prefixes=[(3*order[:-3]).tolist()],gold_data=data,gpu_features=features)
            for _ in range(2000):
                gpu.step(128);gpu.verify();state=int(gpu.states.get()[0])
                self.assertNotEqual(state,1)
                if state==2:break
            else:self.fail('Finite disconnected Gold tail did not exhaust')
            counts.append(int(gpu.counters.get()[1,0]))
        self.assertGreater(counts[0],0);self.assertEqual(counts[1],0)
        REASONING_EVIDENCE.append({'check':'optimistic_connectivity_open_strands','backend':type(self).__name__,
            'accepted_placements':dict(zip(('baseline','connectivity'),counts)),
            'already_closed_loop_in_prefix':False,'gold_solutions':0})

    def test_symmetry_prunes_other_macrofaces_and_refuses_legacy_resume(self):
        from constraint_symmetry import proper_rotations
        data,board=gold_fixture();order=fill_order(board.neighbor_cells)
        transformed=next(t.apply(list(range(0,480,3))) for t in proper_rotations(board) if t.faces[0]!=0)
        old=self.make(data,board,n=1,prefixes=[np.asarray(transformed)[order].tolist()],gold_data=data)
        reduced=self.make(data,board,n=1,prefixes=[np.asarray(transformed)[order].tolist()],gold_data=data,gpu_features={'symmetry':True})
        old.step(512);reduced.step(512)
        self.assertEqual(int(old.states.get()[0]),1);self.assertEqual(int(reduced.states.get()[0]),2)
        with self.assertRaisesRegex(ValueError,'differs'):reduced.restore_checkpoint(old.snapshot_checkpoint())


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','CUDA hardware test opt-in')
class ReasoningCUDA(ReasoningHardwareMixin,unittest.TestCase):
    factory=staticmethod(DFS)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU hardware test opt-in')
class ReasoningWebGPU(ReasoningHardwareMixin,unittest.TestCase):
    @staticmethod
    def factory(*args,**kwargs):
        from dfs_webgpu import WebGPUDFS
        return WebGPUDFS(*args,**kwargs,allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')


if __name__=='__main__':unittest.main()

