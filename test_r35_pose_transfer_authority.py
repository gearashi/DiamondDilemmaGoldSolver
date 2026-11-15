"""Independently check the geographic transport of the original repair clauses."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import unittest

from geometry import build_board
from r35_pose_departures import indexed_rows, original_pose_departures, reference_images
from search_fixtures import planted_gold
from search_problem import SearchProblem
from validator import validate_arrangement


DOCS = Path(__file__).resolve().parent / 'docs'
SOURCE = {
    'R35_REPAIR_PLACEMENTS': 'd72d6c147d3af5798f9f53a96832adf1ac82ae7e4fdc9f009c48700ddf020818',
    'R35_REPAIR_REFINED': '3369433c96e41f6af984ba4d8fed42998d26b62538f7920395b1df7a16719397',
    'R35_REPAIR_REGIONAL': '5ae4ed6a4af821eb01d653e11d202aa5122ce73334f1c15114745c17768d4c61',
    'R35_REGIONAL_REPAIR_PROOF': '1e69943dde427ea7b19add2bada26d4f54662a4f8366f5a765995371f7771eed',
}


def catalog():
    docs = {}
    for name, expected in SOURCE.items():
        data = (DOCS / (name + '.json')).read_bytes().strip()
        if sha256(data).hexdigest() != expected:
            raise ValueError('Source receipt changed: ' + name)
        docs[name] = json.loads(data)
    initial, refined, regional, proof = docs.values()
    rows = [*initial['rows'], *refined['extra'], *refined['additional'], *regional['small'], *regional['larger']]
    return tuple(initial['codes']), tuple(map(tuple,rows)), regional['hit'], proof


@lru_cache(maxsize=1)
def oriented_mesh_maps():
    """Reconstruct twin darts from cell corners, then extend each of the 480 root darts."""
    board = build_board()
    incidence = defaultdict(list)
    for c in board.cells:
        for side in range(3):
            u, v = c.vertices[side], c.vertices[(side+1) % 3]
            incidence[min(u,v),max(u,v)].append((c.index,side,u,v))
    across = {}
    for (c,s,u,v),(d,t,w,z) in incidence.values():
        if (u,v)!=(z,w):
            raise ValueError('Inconsistent triangle boundary')
        across[c,s] = d,t
        across[d,t] = c,s
    maps = []
    for d0 in range(len(board.cells)):
        for r0 in range(3):
            assignment = {0:(d0,r0)}
            pending = [0]
            while pending:
                c = pending.pop()
                d, offset = assignment[c]
                for side in range(3):
                    other, back = across[c,side]
                    image, image_back = across[d,(side+offset)%3]
                    new = (image,(image_back-back)%3)
                    if other not in assignment:
                        assignment[other] = new
                        pending.append(other)
                    elif assignment[other]!=new:
                        break
                else:
                    continue
                break
            else:
                if len(assignment)==len(board.cells) and len({d for d,_ in assignment.values()})==len(board.cells):
                    maps.append(tuple(assignment[i] for i in range(len(board.cells))))
    return tuple(maps)


def materialize(codes, rotation):
    result = [None]*len(codes)
    for code,(dest,delta) in zip(codes,rotation):
        tile,turn = divmod(code,3)
        result[dest] = 3*tile+(turn+delta)%3
    return result


def planted_on_codes(codes):
    planted, board, layout = planted_gold(89529,4,extra_edges=0)
    tiles = [None]*len(codes)
    for present,target in zip(layout,codes):
        old, old_turn = divmod(present,3)
        new, new_turn = divmod(target,3)
        # Pair consecutive endpoints around the doubled spanning tree. Its boundary is one
        # noncrossing circuit, so both the raw Gold validator and the signed model apply.
        points=sorted(tuple(point) for arc in planted['tiles'][old]['segments'] for point in arc)
        arcs=[(points[i],points[(i+1)%len(points)]) for i in range(1,len(points),2)]
        segments = [[[(s+old_turn-new_turn)%3,p] for s,p in segment] for segment in arcs]
        tiles[new] = {'id':new, 'segments':segments}
    return {'tiles':tiles}, board


class PoseTransferAuthority(unittest.TestCase):
    def test_every_authenticated_row_in_every_oriented_mesh_map(self):
        codes,rows,hit,proof = catalog()
        self.assertEqual((len(rows),len(set(hit)),proof['lower_bound'],proof['excluded_cap']),(4225,95,95,94))
        self.assertTrue(all(set(row).intersection(hit) for row in rows))
        incidence = defaultdict(set)
        for row in rows:
            mask=sum(1<<c for c in row)
            for c in row:
                incidence[c].add(mask)
        self.assertEqual({c:set(r) for c,r in indexed_rows(rows).items()},dict(incidence))
        expected={rot[0]:rot for rot in oriented_mesh_maps()}
        self.assertEqual(len(expected),10)
        actual=set()
        images=reference_images(codes)
        self.assertEqual(len(images),10)
        for known in images:
            dest,pose=known[0]
            offset=(pose.bit_length()-1-codes[0])%3
            rotation=expected[dest,offset]
            actual.add((dest,offset))
            board=materialize(codes,rotation)
            image=tuple((d,1<<board[d]) for d,_ in rotation)
            self.assertEqual(known,image)
            for row in rows:
                transported=[(rotation[c][0],3*(codes[c]//3)+(codes[c]%3+rotation[c][1])%3) for c in row]
                self.assertEqual([(d,p.bit_length()-1) for d,p in (known[c] for c in row)],transported)
        self.assertEqual(actual,set(expected))

    def test_same_reference_codes_are_geographic_gold_on_alternate_artwork(self):
        codes,rows,_,_ = catalog()
        data,board = planted_on_codes(codes)
        problem=SearchProblem(data,board,physical_gold_support=True)
        for rotation in oriented_mesh_maps():
            target=materialize(codes,rotation)
            domains=[1<<v for v in target]
            full=validate_arrangement(data,target,board)
            self.assertEqual((full['matched_edges'],full['line_components'],full['valid']),(240,1,True))
            self.assertEqual(original_pose_departures(problem,domains),(domains,None))
            self.assertTrue(all(all(target[rotation[c][0]]//3==codes[c]//3 for c in row) for row in rows))

    def test_non_crossing_planted_board_passes_native_geographic_receiver(self):
        try:import ortools
        except ImportError:self.skipTest('Native geographic constructor requires OR-Tools')
        from r30_gold import search
        codes,_,_,_=catalog()
        data,board=planted_on_codes(codes)
        p=SearchProblem(data,board,physical_gold_support=True)
        out=search(p,geography=True,reference_bound=True,domains=[1<<v for v in codes],seconds=30)
        self.assertEqual(out['status'],'GOLD')
        self.assertNotIn('certified_pose_departures',out)
        self.assertEqual(list(codes),out['codes'])
        self.assertTrue(validate_arrangement(data,out['codes'],board)['valid'])

    def test_separate_post_master_rows_refute_all_forty_transported_queries(self):
        try:import pysat.solvers
        except ImportError:self.skipTest('Independent geographic SAT propagation requires python-sat')
        tiles=next((p for p in (DOCS.parent/'data/tiles.json',DOCS.parent.parent/'data/tiles.json') if p.is_file()),None)
        if tiles is None:self.skipTest('Original prepared puzzle unavailable')
        from bench_r35_repair_seeds import independently_allowed,PoseOracle
        from r35_repair_seeds import active_catalog
        p=SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
        codes,_,_,_=catalog()
        extra=json.loads((DOCS/'R35_REPAIR_PROPOSALS.json').read_bytes())['global_cuts']
        self.assertEqual(active_catalog(p)['post_master'],extra)
        self.assertEqual([len(row) for row in extra],[8,6,6,5])
        oracle=PoseOracle(p,independently_allowed(p)[0])
        try:
            for rotation in oriented_mesh_maps():
                image=materialize(codes,rotation)
                for row in extra:
                    held={rotation[c][0] for c in row}
                    self.assertTrue(oracle.replay(image,held))
                    self.assertTrue(all(not oracle.replay(image,held-{c}) for c in held))
                    ds=[(1<<480)-1]*160
                    for c in held:ds[c]=1<<image[c]
                    _,report=original_pose_departures(p,ds)
                    self.assertTrue(report['over'])
                    self.assertEqual(report['reason'],'pose_clause')
                    self.assertEqual(set(map(tuple,report['fixed_poses'])),{(c,image[c]) for c in held})
        finally:
            oracle.close()

    def test_original_sources_and_changed_pairing(self):
        tiles=next((p for p in (DOCS.parent/'data/tiles.json',DOCS.parent.parent/'data/tiles.json') if p.is_file()),None)
        if tiles is None:self.skipTest('Original prepared puzzle unavailable')
        from r35_repair_seeds import active_catalog
        data=json.loads(tiles.read_bytes())
        codes,rows,_,_ = catalog()
        problem=SearchProblem(data,physical_gold_support=True)
        self.assertEqual(tuple(map(tuple,active_catalog(problem)['rows'])),rows)
        for rotation in oriented_mesh_maps():
            image=materialize(codes,rotation)
            full=validate_arrangement(data,image)
            self.assertEqual((full['matched_edges'],full['line_components']),(207,52))
            _,found=original_pose_departures(problem,[1<<x for x in image])
            self.assertTrue(found['over'])
        # Keep every side mask unchanged while switching the interior partners of two strands.
        changed=next(t['segments'] for t in data['tiles'] if len(t['segments'])>=2)
        changed[0][1],changed[1][1]=changed[1][1],changed[0][1]
        changed_problem=SearchProblem(data,physical_gold_support=True)
        self.assertTrue((problem.masks==changed_problem.masks).all())
        ds=[1<<x for x in codes]
        self.assertEqual(original_pose_departures(changed_problem,ds),(ds,None))

    def test_direct_geographic_receiver_validates_optional_inputs(self):
        try:import ortools
        except ImportError:self.skipTest('Native geographic constructor requires OR-Tools')
        tiles=next((p for p in (DOCS.parent/'data/tiles.json',DOCS.parent.parent/'data/tiles.json') if p.is_file()),None)
        if tiles is None:self.skipTest('Original prepared puzzle unavailable')
        from r30_gold import search
        p=SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
        codes,rows,_,_=catalog()
        ds=[(1<<(3*p.n))-1]*p.n
        for c in next(r for r in rows if len(r)==2): ds[c]=1<<codes[c]
        allowed=search(p,geography=True,reference_bound=True,domains=ds,cutsets=[[0]],seconds=2)
        self.assertEqual((allowed['status'],allowed['cp_calls'],allowed['strand_cuts']),('INFEASIBLE',0,1))
        for options in ({'cutsets':[[]]},{'type_cuts':[[]]},{'bridges':[[]]},{'physical_loss':('invalid',)}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                search(p,geography=True,reference_bound=True,domains=ds,seconds=1,**options)
