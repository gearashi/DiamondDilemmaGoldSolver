"""Topology invariants and exact, hand-constructible validation fixtures."""
import unittest
from collections import Counter,defaultdict
from geometry import build_board
from validator import normalize_tiles,orientation_masks,reverse_mask,validate_arrangement


def loop_fixture(board,cycles):
    tiles = [{'id':f'fixture-{i}','segments':[]} for i in range(len(board.cells))]
    for cycle in cycles:
        for position,cell in enumerate(cycle):
            before,after = cycle[position-1],cycle[(position+1)%len(cycle)]
            sa = next(s for s,pair in enumerate(board.neighbors[cell]) if pair[0]==before)
            sb = next(s for s,pair in enumerate(board.neighbors[cell]) if pair[0]==after)
            tiles[cell]['segments'].append([[sa,6],[sb,6]])
    return tiles,[3*i for i in range(len(tiles))]


class GeometryTests(unittest.TestCase):
    def test_counts_and_euler(self):
        for n in (1,2,3,4):
            board = build_board(n)
            self.assertEqual(len(board.cells),10*n*n)
            self.assertEqual(len(board.edges),15*n*n)
            self.assertEqual(len(board.vertices),5*n*n+2)
            self.assertEqual(len(board.vertices)-len(board.edges)+len(board.cells),2)
            self.assertEqual(Counter(c.face for c in board.cells),{f:n*n for f in range(10)})

    def test_neighbors_and_orientation(self):
        board = build_board()
        seen = set()
        for cell in board.cells:
            self.assertEqual(len(set(p[0] for p in board.neighbors[cell.index])),3)
            for side,(other,oside) in enumerate(board.neighbors[cell.index]):
                self.assertNotEqual(other,cell.index)
                self.assertEqual(board.neighbors[other][oside],(cell.index,side))
                a = (cell.vertices[side],cell.vertices[(side+1)%3])
                remote = board.cells[other].vertices
                b = (remote[oside],remote[(oside+1)%3])
                self.assertEqual(a,b[::-1])
                seen.add(tuple(sorted(a)))
        self.assertEqual(len(seen),240)

    def test_all_seams(self):
        board = build_board()
        seams = Counter()
        for a,_,b,_ in board.edges:
            fa,fb = board.cells[a].face,board.cells[b].face
            if fa!=fb:
                seams[tuple(sorted((fa,fb)))] += 1
        expected = {tuple(sorted((i,(i+1)%5))) for i in range(5)}
        expected |= {tuple(sorted((5+i,5+(i+1)%5))) for i in range(5)}
        expected |= {(i,i+5) for i in range(5)}
        self.assertEqual(set(seams),expected)
        self.assertEqual(set(seams.values()),{4})

    def test_vertex_links_are_single_cycles(self):
        board = build_board()
        links = defaultdict(lambda:defaultdict(set))
        for cell in board.cells:
            for pos,center in enumerate(cell.vertices):
                a,b = cell.vertices[(pos+1)%3],cell.vertices[(pos+2)%3]
                links[center][a].add(b)
                links[center][b].add(a)
        for link in links.values():
            self.assertTrue(all(len(neighbors)==2 for neighbors in link.values()))
            reached,pending = set(),[next(iter(link))]
            while pending:
                node = pending.pop()
                if node not in reached:
                    reached.add(node)
                    pending.extend(link[node]-reached)
            self.assertEqual(reached,set(link))

    def test_board_connected(self):
        board = build_board()
        reached,pending = set(),[0]
        while pending:
            cell = pending.pop()
            if cell not in reached:
                reached.add(cell)
                pending.extend(other for other,_ in board.neighbors[cell] if other not in reached)
        self.assertEqual(len(reached),160)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.board = build_board(1)
        self.tiles,self.codes = loop_fixture(self.board,[[0,1,2,3,4,9,8,7,6,5]])

    def test_known_single_loop(self):
        report = validate_arrangement(self.tiles,self.codes,self.board)
        self.assertTrue(report['valid'],report)
        self.assertEqual(report['component_lengths'],[10])
        self.assertEqual(report['matched_edges'],15)

    def test_two_loops_rejected_despite_perfect_edges(self):
        tiles,codes = loop_fixture(self.board,[[0,1,2,3,4],[5,6,7,8,9]])
        report = validate_arrangement(tiles,codes,self.board)
        self.assertFalse(report['valid'])
        self.assertTrue(report['all_closed'])
        self.assertEqual(report['matched_edges'],15)
        self.assertEqual(report['component_lengths'],[5,5])

    def test_duplicate_tile_rejected(self):
        self.codes[0] = self.codes[1]
        report = validate_arrangement(self.tiles,self.codes,self.board)
        self.assertFalse(report['valid'])
        self.assertFalse(report['unique_tiles'])

    def test_noncentral_endpoint_requires_reversal(self):
        a,sa,b,sb = next(e for e in self.board.edges if [e[1],6] in self.tiles[e[0]]['segments'][0])
        for point in self.tiles[a]['segments'][0]:
            if point[0]==sa:
                point[1] = 2
        for point in self.tiles[b]['segments'][0]:
            if point[0]==sb:
                point[1] = 10
        self.assertTrue(validate_arrangement(self.tiles,self.codes,self.board)['valid'])
        for point in self.tiles[b]['segments'][0]:
            if point[0]==sb:
                point[1] = 2
        self.assertFalse(validate_arrangement(self.tiles,self.codes,self.board)['valid'])

    def test_rotation_preserves_positions(self):
        for i,tile in enumerate(self.tiles):
            rotation = i%3
            self.codes[i] += rotation
            for segment in tile['segments']:
                for point in segment:
                    point[0] = (point[0]-rotation)%3
        self.assertTrue(validate_arrangement(self.tiles,self.codes,self.board)['valid'])

    def test_masks_and_reversal(self):
        tile = {'id':'asymmetric','segments':[[[0,1],[1,3]],[[1,11],[2,7]]]}
        masks = orientation_masks([tile])
        self.assertEqual(masks[0],[1,(1<<2)|(1<<10),1<<6])
        self.assertEqual(masks[1],[masks[0][2],masks[0][0],masks[0][1]])
        for mask in range(2048):
            self.assertEqual(reverse_mask(reverse_mask(mask)),mask)
        self.assertEqual(reverse_mask(1),1024)

    def test_crossings_are_not_junctions(self):
        for tile in self.tiles:
            sa,sb = [p[0] for p in tile['segments'][0]]
            tile['segments'] = [[[sa,3],[sb,9]],[[sa,9],[sb,3]]]
        report = validate_arrangement(self.tiles,self.codes,self.board)
        self.assertEqual(report['matched_edges'],15)
        self.assertEqual(report['component_lengths'],[10,10])
        self.assertFalse(report['valid'])

    def test_same_side_segments_accepted(self):
        result = normalize_tiles([{'id':'U','segments':[[[0,1],[0,11]]]}])
        self.assertEqual(len(result[0]['segments']),1)

    def test_invalid_data_and_placements(self):
        self.assertFalse(validate_arrangement(self.tiles,self.codes[:-1],self.board)['valid'])
        self.codes[0] = True
        self.assertFalse(validate_arrangement(self.tiles,self.codes,self.board)['valid'])
        with self.assertRaises(ValueError):
            normalize_tiles([{'id':'bad','segments':[[[0,1],[1,2]],[[0,1],[2,4]]]}])

if __name__ == '__main__':
    unittest.main()
