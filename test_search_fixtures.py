import unittest
from search_fixtures import planted_gold
from validator import validate_arrangement

class FixtureTests(unittest.TestCase):
    def test_planted_references_validate_after_tile_shuffle_and_rotation(self):
        for size in (1,2,4):
            for seed in (817,291):
                with self.subTest(size=size,seed=seed):
                    data,board,codes=planted_gold(seed,size)
                    self.assertTrue(validate_arrangement(data,codes,board)['valid'])
                    again,_,reference=planted_gold(seed,size)
                    self.assertEqual((again,reference),(data,codes))

if __name__=='__main__':unittest.main()
