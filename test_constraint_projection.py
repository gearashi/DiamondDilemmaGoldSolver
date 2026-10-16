"""Independent finite checks for orientation/physical-tile bit projections."""
import random
import unittest

from constraint_pruning import (
    _tile_choices, _orientation_choices,
    _gather_tile_choices, _scatter_orientation_choices,
)


def gather_oracle(domain):
    return sum(bool(domain & (7 << (3 * tile))) << tile
               for tile in range((domain.bit_length() + 2) // 3))


def scatter_oracle(tiles):
    return sum(7 << (3 * tile) for tile in range(tiles.bit_length())
               if tiles & (1 << tile))


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        for function in (_tile_choices, _orientation_choices):
            function.cache_clear()
            self.addCleanup(function.cache_clear)

    def test_exhaustive_four_tile_domains_and_small_inventories(self):
        for value in range(4096):
            gathered = gather_oracle(value)
            self.assertEqual(_tile_choices(value), gathered)
            self.assertEqual(_gather_tile_choices(value), gathered)
            scattered = scatter_oracle(value)
            self.assertEqual(_orientation_choices(value), scattered)
            self.assertEqual(_scatter_orientation_choices(value), scattered)

    def test_seeded_model_widths_and_arbitrary_width_fallback(self):
        rng = random.Random(160416)
        widths = (0, 1, 11, 12, 31, 32, 33, 63, 64, 65, 95, 96, 97,
                  127, 128, 129, 159, 160, 161, 255, 256, 479, 480,
                  481, 511, 512, 767, 768, 769, 1024)
        for width in widths:
            for _ in range(32):
                value = rng.getrandbits(width)
                gathered = gather_oracle(value)
                scattered = scatter_oracle(value)
                self.assertEqual(_tile_choices(value), gathered, (width, value))
                self.assertEqual(_orientation_choices(value), scattered, (width, value))
                self.assertEqual(_tile_choices(scattered), value)
                if width <= 480:
                    self.assertEqual(_gather_tile_choices(value), gathered)
                if width <= 160:
                    self.assertEqual(_scatter_orientation_choices(value), scattered)

    def test_each_bit_and_dense_boundaries(self):
        # Include bits beyond the model limits: generic callers must retain them.
        for bit in range(1025):
            value = 1 << bit
            self.assertEqual(_tile_choices(value), 1 << (bit // 3))
            self.assertEqual(_orientation_choices(value), 7 << (3 * bit))
            self.assertEqual(_tile_choices(value | 1), (1 << (bit // 3)) | 1)
        for width in (32, 33, 64, 65, 96, 97, 128, 129, 160, 161,
                      256, 257, 480, 481, 512, 513, 768, 769, 1024):
            value = (1 << width) - 1
            self.assertEqual(_tile_choices(value), gather_oracle(value))
            self.assertEqual(_orientation_choices(value), scatter_oracle(value))

    def test_exact_key_caches_keep_existing_bound_and_eviction_semantics(self):
        for function, oracle in ((_tile_choices, gather_oracle),
                                 (_orientation_choices, scatter_oracle)):
            with self.subTest(function=function.__name__):
                self.assertEqual(function.cache_info().maxsize, 8192)
                self.assertEqual(function(0), 0)
                self.assertEqual(function(0), 0)
                self.assertEqual(function.cache_info().hits, 1)
                for value in range(1, 8194):
                    self.assertEqual(function(value), oracle(value))
                before = function.cache_info()
                self.assertEqual(before.currsize, 8192)
                self.assertEqual(function(0), 0)
                self.assertEqual(function.cache_info().misses, before.misses + 1)
                self.assertEqual(function.cache_info().currsize, 8192)


if __name__ == '__main__':
    unittest.main()
