import unittest
from collections import Counter
from search_fixtures import planted_gold, planted_inventory, signature_pairs
from validator import validate_arrangement


def with_odd_signatures(source, board, codes, corners):
    """Add one endpoint to both sides of every seam around the given mesh vertices.

    Each cell at such a vertex gains two endpoints, joined as one new segment,
    so the board still matches and those seams now carry odd-sized signatures.
    """
    tiles=[{'id':tile['id'],'segments':[[list(point) for point in segment] for segment in tile['segments']]} for tile in source['tiles']]
    used=[[{point for segment in tiles[code//3]['segments'] for side,point in segment if side==(cell_side-code%3)%3}
           for cell_side in range(3)] for code in codes]
    added=[[] for _ in codes]
    for a,sa,b,sb in board.edges:
        shared=set(board.cells[a].vertices)&set(board.cells[b].vertices)
        if len(shared&set(corners))!=1:continue
        point=next(p for p in range(1,12) if p not in used[a][sa])
        used[a][sa].add(point);used[b][sb].add(12-point)
        added[a].append([(sa-codes[a]%3)%3,point]);added[b].append([(sb-codes[b]%3)%3,12-point])
    for cell,points in enumerate(added):
        assert len(points)%2==0
        tiles[codes[cell]//3]['segments'].extend([points[index],points[index+1]] for index in range(0,len(points),2))
    return {'tiles':tiles}


class FixtureTests(unittest.TestCase):
    def test_planted_references_validate_after_tile_shuffle_and_rotation(self):
        for size in (1,2,4):
            for seed in (817,291):
                with self.subTest(size=size,seed=seed):
                    data,board,codes=planted_gold(seed,size)
                    self.assertTrue(validate_arrangement(data,codes,board)['valid'])
                    again,_,reference=planted_gold(seed,size)
                    self.assertEqual((again,reference),(data,codes))

    def test_planted_inventory_keeps_every_side_signature_and_plants_a_matched_board(self):
        for size,extra,corners in ((2,60,()),(4,24,()),(4,240,()),(2,60,(0,5)),(4,24,(0,7,19,40))):
            source,board,codes=planted_gold(291,size,extra)
            if corners:
                source=with_odd_signatures(source,board,codes,corners)
                self.assertEqual(validate_arrangement(source,codes,board)['matched_edges'],len(board.edges))
            odd=sum(len(pair[0])%2 for pair in signature_pairs(source))
            self.assertEqual(odd>0,bool(corners))
            for seed in (3,17):
                with self.subTest(size=size,extra=extra,corners=corners,seed=seed):
                    data,planted_board,reference=planted_inventory(source,seed,size)
                    report=validate_arrangement(data,reference,planted_board)
                    self.assertEqual(report['matched_edges'],report['total_edges'])
                    self.assertTrue(report['unique_tiles'])
                    self.assertEqual(Counter(signature_pairs(data)),Counter(signature_pairs(source)))
                    self.assertNotEqual(data['tiles'],source['tiles'])
                    self.assertTrue(all(tile['segments'] for tile in data['tiles']))
                    again,_,same=planted_inventory(source,seed,size)
                    self.assertEqual((again,same),(data,reference))

    def test_planted_inventory_rejects_tile_sets_that_cannot_pair_up(self):
        source,_,_=planted_gold(291,2)
        with self.assertRaisesRegex(ValueError,'one tile per cell'):
            planted_inventory(source,3,4)
        broken={'tiles':[dict(tile) for tile in source['tiles']]}
        broken['tiles'][0]={'id':'odd','segments':[[[0,1],[0,2]],[[0,3],[0,5]]]}
        with self.assertRaisesRegex(ValueError,'do not pair up'):
            planted_inventory(broken,3,2)

if __name__=='__main__':unittest.main()
