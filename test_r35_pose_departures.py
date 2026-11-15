"""Finite controls for geographic propagation of repair clauses and distance bounds."""
from itertools import permutations, product
from random import Random
import unittest

from r35_pose_departures import indexed_rows, propagate


class PoseClauseTests(unittest.TestCase):
    def test_fixed_point_across_reference_images(self):
        # The earlier image first deletes pose 0 from cell 1, fixing pose 1.
        # The next image then has both of its forbidden poses fixed.
        image_a=((0,1<<0),(1,1<<1),(2,1<<2))
        image_b=((2,1<<2),(1,1<<0),(0,1<<2))
        narrowed,out=propagate([1<<0,3,1<<2],(image_b,image_a),indexed_rows(((0,1),)),3)
        self.assertTrue(out['over'])
        self.assertEqual(out['reason'],'pose_clause')
        self.assertGreaterEqual(out['passes'],1)

    def test_saturated_distance_removes_only_unsatisfied_reference_poses(self):
        image=tuple((i,1<<i) for i in range(4))
        ds=[1,2,4|32,8|16]
        narrowed,stats=propagate(ds,[image],{},2)
        self.assertEqual(narrowed,[1,2,32,16])
        self.assertEqual(stats['distance_units'],2)
        self.assertFalse(stats['over'])
        self.assertEqual(ds,[1,2,36,24])
        _,conflict=propagate([1,2,4,24],[image],{},2)
        self.assertTrue(conflict['over'])
        self.assertEqual(conflict['reason'],'cell_distance')

    def test_finite_oracle_preserves_every_valid_distinct_tile_and_rotation(self):
        random=Random(8951)
        n=4
        boards=[tuple(3*t+r for t,r in zip(tiles,rots))
                for tiles in permutations(range(n)) for rots in product(range(3),repeat=n)]
        for case in range(80):
            images=[]
            for _ in range(2):
                cells=random.sample(range(n),n)
                codes=random.choice(boards)
                images.append(tuple((cell,1<<codes[cell]) for cell in cells))
            raw_rows=tuple(set(tuple(sorted(random.sample(range(n),random.randrange(2,4)))) for _ in range(3)))
            max_kept=random.randrange(n+1)
            full=(1<<(3*n))-1
            ds=[full if random.random()<.4 else random.choice([pose for image in images for cell,pose in image if cell==i]) |
                (1<<random.randrange(3*n) if random.random()<.5 else 0) for i in range(n)]
            allowed=[]
            for board in boards:
                if any(not mask & (1<<code) for mask,code in zip(ds,board)):
                    continue
                for image in images:
                    same={i for i,(cell,pose) in enumerate(image) if pose==(1<<board[cell])}
                    if len(same)>max_kept or any(same.issuperset(row) for row in raw_rows):
                        break
                else:
                    allowed.append(board)
            reduced,result=propagate(ds,images,indexed_rows(raw_rows),max_kept)
            with self.subTest(case=case):
                if result['over']:
                    self.assertFalse(allowed)
                else:
                    self.assertTrue(all(all(mask&(1<<code) for mask,code in zip(reduced,board)) for board in allowed))
                    self.assertTrue(all((old & new)==new for old,new in zip(ds,reduced)))

    def test_dense_support_against_every_small_bipartite_graph(self):
        from r35_pose_departures import dense_tile_support
        for n in (1, 2, 3, 4):
            # Enumerate unordered rows. Relabeling the cells preserves all possible matchings.
            from itertools import combinations_with_replacement
            every = list(permutations(range(n)))
            for rows in combinations_with_replacement(range(1,1<<n),n):
                poses = [sum(7<<(3*j) for j in range(n) if row>>j & 1) for row in rows]
                if not dense_tile_support(poses):
                    continue
                supported = [0]*n
                for order in every:
                    if all(row>>j & 1 for row,j in zip(rows,order)):
                        for c,j in enumerate(order):
                            supported[c] |= 1<<j
                self.assertEqual(tuple(supported),rows)
        self.assertFalse(dense_tile_support([0,7]))
        self.assertFalse(dense_tile_support([1,2]))

    def test_narrow_crossover_seams_preserve_a_complete_gold_witness(self):
        from r27_structure_filters import domain_bits
        from r35_pose_departures import crossover_seams, reference_images
        from search_problem import SearchProblem
        from test_r35_pose_transfer_authority import catalog, planted_on_codes

        codes = catalog()[0]
        data, board = planted_on_codes(codes)
        problem = SearchProblem(data, board=board, physical_gold_support=True)
        self.assertTrue(problem.validate(codes)["valid"])
        images = reference_images(codes)
        random = Random(8936)
        for count in (5, 40, 120, 160):
            domains = list(domain_bits(problem))
            for cell in random.sample(range(problem.n), count):
                alternatives = {dict(image)[cell] for image in random.sample(images, 3)}
                domains[cell] &= sum(alternatives | {1 << codes[cell]})
            narrowed, result = crossover_seams(problem, domains, images)
            self.assertFalse(result["over"])
            self.assertTrue(all(mask & (1 << code) for mask, code in zip(narrowed, codes)))

    def test_compound_search_keeps_live_domain_reductions_and_validates_cuts(self):
        from unittest.mock import patch
        from r30_gold import search
        from test_r35_pose_transfer_authority import catalog, planted_on_codes
        from search_problem import SearchProblem
        try:
            import ortools
        except ImportError:
            self.skipTest("CP-SAT unavailable")
        codes = catalog()[0]
        data, board = planted_on_codes(codes)
        problem = SearchProblem(data,board)
        domains = [1 << code for code in codes]
        counters = dict(over=False,removed=1)
        with patch("r35_pose_departures.original_pose_departures",return_value=(domains,counters)):
            result = search(problem,geography=True,reference_bound=True,cutsets=[[0]],seconds=15)
            self.assertEqual(result["certified_pose_departures"],counters)
            self.assertEqual((result["status"],tuple(result["codes"])),("GOLD",codes))
            with self.assertRaises(ValueError):
                search(problem,geography=True,reference_bound=True,cutsets=[[]],seconds=2)

    def test_authenticated_rows_reused_only_for_the_same_proof_files_and_mesh(self):
        import json
        import shutil
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        import r35_pose_departures as pose
        import r35_repair_seeds as seeds
        from search_problem import SearchProblem

        data=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if data is None:
            self.skipTest("Original prepared artwork unavailable")
        problem=SearchProblem(json.loads(data.read_bytes()))
        root=Path(pose.__file__).resolve().parent
        seed=json.loads((root/'docs/R29_INCUMBENT.json').read_bytes())['codes']
        ds=[(1<<(3*problem.n))-1]*problem.n
        ds[0]=1<<seed[0]
        names=('R35_REPAIR_PLACEMENTS.json','R35_REPAIR_REFINED.json','R35_REPAIR_REGIONAL.json')
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'docs').mkdir()
            for name in names:
                shutil.copyfile(root/'docs'/name,folder/'docs'/name)
            with patch.object(pose,'__file__',str(folder/'r35_pose_departures.py')), \
                    patch.object(seeds,'__file__',str(folder/'r35_repair_seeds.py')), \
                    patch.object(seeds,'active_catalog',wraps=seeds.active_catalog) as accept:
                for _ in range(2):
                    _,out=pose.original_pose_departures(problem,ds)
                    self.assertIsNotNone(out)
                self.assertEqual(accept.call_count,1)
                problem.masks[0,0]^=1
                self.assertEqual(pose.original_pose_departures(problem,ds),(ds,None))
                self.assertEqual(accept.call_count,1)
                problem.masks[0,0]^=1
                target=folder/'docs'/names[-1]
                before=target.read_bytes()
                (folder/'replaced').write_bytes(b'x'+before[1:])
                (folder/'replaced').replace(target)
                self.assertEqual(pose.original_pose_departures(problem,ds),(ds,None))
                self.assertEqual(accept.call_count,2)
