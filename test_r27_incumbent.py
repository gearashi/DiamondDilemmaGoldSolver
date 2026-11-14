import itertools
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from r27_incumbent import BoardModel, polish, rotation_alternatives
from r27_incumbent_seed import variants, initialize_lanes
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import double_face, simple_tiles
from gpu_sampling import gold_score
from gpu_incremental_sampling_round26 import CACHE_BASE, canonical_labels, metadata_rows


class IncumbentTests(unittest.TestCase):
    def test_assignment_against_all_permutations_and_rotations(self):
        data,board,solution=planted_gold(seed=7,size=1,extra_edges=3)
        p=SearchProblem(data,board)
        m=BoardModel(p)
        rng=np.random.default_rng(9)
        start=np.array(solution)
        cells=m.independent_set(start,rng)[:3]
        self.assertEqual(len(cells),3)
        start[cells]=start[cells[::-1]]
        perms=itertools.permutations(start[cells]//3)
        high=-1
        for tiles in perms:
            for turns in itertools.product(range(3),repeat=3):
                trial=start.copy()
                trial[cells]=3*np.array(tiles)+turns
                high=max(high,p.validate(trial.tolist())['matched_edges'])
        for seed in range(3):
            opt=m.block(start,cells,np.random.default_rng(seed))
            self.assertEqual(p.validate(opt.tolist())['matched_edges'],high)
        with self.assertRaises(ValueError):
            m.block(start,[p.edges[0][0],p.edges[0][2]])
        with self.assertRaises(ValueError):
            m.block(start,[0,0])

    def test_orientations_exact_k_best_and_restricted_domains(self):
        p=SearchProblem(simple_tiles(),double_face())
        p.domains[1]=[3,4]
        m=BoardModel(p)
        scored=[]
        for a,b in itertools.product(range(3),[3,4]):
            row=[a,b]
            scored.append(p.validate(row)['matched_edges'])
        current=[0,4]
        out,info=m.orientations(current)
        self.assertTrue(info['exact'])
        self.assertEqual(info['upper_bound'],max(scored))
        self.assertIn(out[1],p.domains[1])
        try:
            rows=rotation_alternatives(m,current,k=30)
        except ImportError:
            self.skipTest('OR-Tools CP-SAT unavailable')
        self.assertEqual([row['matched_edges'] for _,row in rows],sorted(scored,reverse=True))
        self.assertTrue(all(row['exact'] for _,row in rows))
        self.assertEqual(len({tuple(v) for v,_ in rows}),len(scored))

    def test_bounded_elimination_preserves_attainable_score_and_upper_bound(self):
        data,board,solution=planted_gold(seed=9,size=1,extra_edges=3)
        p=SearchProblem(data,board)
        m=BoardModel(p)
        sub={0,1,2,3}
        start=np.array(solution)
        start[list(sub)]=np.roll(start[list(sub)],1)
        scores=[]
        for r in itertools.product(range(3),repeat=len(sub)):
            trial=start.copy()
            trial[list(sub)]=3*(start[list(sub)]//3)+r
            scores.append(int(m.matches(trial).sum()))
        out,info=m.orientations(start,max_entries=3,movable=sub)
        self.assertGreaterEqual(info['upper_bound'],max(scores))
        self.assertGreaterEqual(info['matched_edges'],int(m.matches(start).sum()))
        self.assertLessEqual(info['matched_edges'],max(scores))
        self.assertEqual(info['matched_edges'],p.validate(out.tolist())['matched_edges'])

    def test_ensemble_checkpoint_matches_uninterrupted(self):
        from bench_r27_incumbent_population import ensemble
        p=SearchProblem(simple_tiles(),double_face())
        m=BoardModel(p)
        for method in ('exchange','flat','cross_entropy','population','niche','penalty'):
            with self.subTest(method=method):
                full=ensemble(m,np.array([0,3]),3,method,proposals=128)
                split=ensemble(m,np.array([0,3]),3,method,proposals=128,interrupt=57)
                resumed=ensemble(m,np.array([0,3]),3,method,proposals=128,resume=split)
                self.assertEqual(resumed['rng'],full['rng'])
                np.testing.assert_array_equal(resumed['boards'],full['boards'])
                self.assertEqual(resumed['accepted'],full['accepted'])
                self.assertEqual(resumed['hist'],full['hist'])

    def test_native_layout_matches_independent_scoring_and_preserves_random_lanes(self):
        p=SearchProblem(simple_tiles(),double_face())
        r=12
        # NumPy stands in for device storage; native CUDA/WebGPU are covered by the bench.
        cp=SimpleNamespace(asarray=np.asarray,cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
        rows=metadata_rows(65,p.total_segments)
        device=SimpleNamespace(cp=cp,r=r,gold_total=p.total_segments,flags=65,
            boards=np.tile([[0],[3]],(1,r)).astype(np.uint32),best=np.tile([[0],[3]],(1,r)).astype(np.uint32),
            meta=np.tile(np.arange(r,dtype=np.uint32),(rows,1)))
        before=[a.copy() for a in (device.boards,device.best,device.meta)]
        result=initialize_lanes(p,device,[1,4],5)
        self.assertEqual(result['seeded_lanes'],9)
        self.assertEqual(result['random_controls'],3)
        np.testing.assert_array_equal(device.boards[:,0],[1,4])
        for a,b in zip((device.boards,device.best,device.meta),before):
            np.testing.assert_array_equal(a[:,9:],b[:,9:])
        for lane in range(9):
            code=device.boards[:,lane].tolist()
            self.assertEqual(device.meta[1,lane],gold_score(p,code)['gold_objective'])
            self.assertEqual(device.meta[2,lane],device.meta[1,lane])
            np.testing.assert_array_equal(device.meta[CACHE_BASE:,lane],canonical_labels(p,code))
        np.testing.assert_array_equal(device.meta[0],before[2][0])

    def test_geometric_crossover_inventory_fixed_domains_and_rotation_dedup(self):
        from bench_r27_incumbent_population import geometric_children
        from constraint_symmetry import proper_rotations
        data,board,gold=planted_gold(seed=9,size=1,extra_edges=3)
        p=SearchProblem(data,board,fixed={0:gold[0]})
        m=BoardModel(p)
        children=geometric_children(m,np.array(gold),np.random.default_rng(4),35)
        rots=proper_rotations(board)
        keys=set()
        for child in children:
            self.assertEqual(child[0],gold[0])
            m.legal_board(child)
            key=min(tuple(r.apply(child.tolist())) for r in rots)
            self.assertNotIn(key,keys)
            keys.add(key)

    def test_hybrid_uses_polished_incumbent_and_restores_cursor(self):
        import gpu_hybrid_resume_round25 as hybrid
        p=SearchProblem(simple_tiles(),double_face())
        # Starting in a valid but imperfect rotation ensures polishing is reached.
        captured=[]
        def stop():
            return len(captured)>=1
        from r27_incumbent import polish_step as actual
        def delayed(model,board,rng):
            captured.append(board)
            # Exercise one real round and pause before native allocation.
            actual(model,board,rng)
            return board,model.problem.validate(board.tolist())
        opts={'durable_hybrid':True,'incumbent_polish':True}
        with patch('r27_incumbent.polish_step',side_effect=delayed):
            interrupted=hybrid.solve(p,seed_board=[0,4],seconds=3,options=opts,stop=stop)
        self.assertEqual(interrupted['checkpoint']['polish_rounds'],1)
        again=hybrid.solve(p,seconds=3,options=opts,resume=interrupted['checkpoint'],stop=lambda:True)
        self.assertEqual(again['checkpoint']['polish_rounds'],1)

    def test_known_prior_sources_only_migrate_if_new_bytes_match(self):
        from r27_incumbent_compat import _COMPATIBLE, compatible_sources
        import hashlib
        from pathlib import Path
        old={name:pair[0] for name,pair in _COMPATIBLE.items()}
        now={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in old}
        self.assertTrue(compatible_sources(old,now))
        self.assertFalse(compatible_sources(old,{**now,'gpu_sampling.py':'future'}))
        self.assertFalse(compatible_sources({**old,'gpu_sampling.py':'unknown'},now))

    def test_previous_windows_crossover_and_repair_bindings(self):
        from r27_incumbent_compat import compatible_sources, _RELEASE_TEXT
        from sampling_crossover_round25 import CycleCrossover, _digest as cycle_digest
        from sampling_crossover_round26 import OptimizingCrossover, _digest as optimize_digest
        from constraint_repair_gold_round26 import GoldRepairPool, _digest as gold_digest
        from pathlib import Path
        import hashlib
        old={name:row[2] for name,row in _RELEASE_TEXT.items()}
        now={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in old}
        self.assertTrue(compatible_sources(old,now))
        p=SearchProblem(simple_tiles(),double_face())
        for cls,digest in ((CycleCrossover,cycle_digest),(OptimizingCrossover,optimize_digest),(GoldRepairPool,gold_digest)):
            controller=cls(p)
            saved=controller.snapshot()
            saved['binding']=controller._identity('windows')
            saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
            restored=cls(p,resume=saved)
            self.assertEqual(controller.binding,restored.binding)

    def test_optional_gold_modules_in_historical_heuristic_identity(self):
        from hashlib import sha256
        from pathlib import Path
        from r27_incumbent_compat import compatible_sources, legacy_sources
        root = Path(__file__).parent
        names = ('constraint_pruning.py', 'search_problem.py', 'r27_structure_filters.py', 'r32_gold_bonds.py')
        current = {n:sha256((root/n).read_bytes()).hexdigest() for n in names}
        previous = legacy_sources(current, 'v112-lf')
        self.assertEqual(set(previous), set(names[:2]))
        self.assertTrue(compatible_sources(previous,current))
        altered = dict(current, **{'r32_gold_bonds.py':'unknown'})
        self.assertFalse(compatible_sources(previous,altered))
        self.assertIsNone(legacy_sources(altered,'v112-lf'))
        self.assertFalse(compatible_sources(previous,dict(current,unreviewed='unknown')))

    def test_assignment_fallback_uses_checked_hungarian(self):
        from r27_incumbent import assignment
        import sys
        with patch.dict(sys.modules, {'ortools.graph.python.linear_sum_assignment':None}):
            result=assignment(np.array([[3,-5],[-2,1]]),np.array([[1,1],[1,1]],bool))
        np.testing.assert_array_equal(result,[0,1])

    def test_seed_variants_keep_fixed_and_domains(self):
        p=SearchProblem(simple_tiles(),double_face(),fixed={0:0})
        p.domains[1]=[3,5]
        choices=variants(p,[0,3],7,17)
        self.assertEqual(len({tuple(v) for v in choices}),len(choices))
        for left,right in choices:
            self.assertEqual(left,0)
            self.assertIn(right,[3,5])
        with self.assertRaises(ValueError):
            variants(p,[0,4],7,17)


if __name__=='__main__':
    unittest.main()

