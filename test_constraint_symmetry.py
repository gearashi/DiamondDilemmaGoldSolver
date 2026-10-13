"""Exact finite tests of the proper-rotation reduction; no puzzle search."""
from dataclasses import replace
import copy
import unittest

from constraint_symmetry import canonical_root_domains, proper_rotations
from geometry import build_board
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, double_face


class SymmetryTests(unittest.TestCase):
    def test_verified_group_has_no_reflections(self):
        for size in (1, 2, 4):
            board = build_board(size)
            rotations = proper_rotations(board)
            self.assertEqual(len(rotations), 10)
            self.assertEqual(len({t.cells for t in rotations}), 10)
            for transform in rotations:
                for cell in board.cells:
                    mapped = [transform.vertices[v] for v in cell.vertices]
                    other = board.cells[transform.cells[cell.index]].vertices
                    self.assertIn(tuple(mapped), [other[offset:]+other[:offset] for offset in range(3)])
            # The equator reflection swapping poles alone would reverse faces;
            # it is deliberately absent from the ten proper rotations.
            self.assertNotIn((1, 0, 2, 3, 4, 5, 6), {t.macro_vertices for t in rotations})

    def test_gold_certificates_and_unique_orbit_representative(self):
        for seed, size in ((17, 1), (29, 2), (31, 4)):
            data, board, reference = planted_gold(seed, size)
            problem = SearchProblem(data, board)
            original_domains = copy.deepcopy(problem.domains)
            original_fingerprint = problem.fingerprint
            for anchor, face in ((0, 0), (problem.n-1, 7)):
                domains, metadata = canonical_root_domains(problem, anchor_tile=anchor, face=face)
                self.assertTrue(metadata['applied'])
                self.assertEqual(metadata['rotations'], 10)
                self.assertEqual(metadata['removed_orientation_values'], 27*size*size)
                candidates = [rotation.apply(reference) for rotation in proper_rotations(board)]
                self.assertEqual(len({tuple(codes) for codes in candidates}), 10)
                for codes in candidates:
                    self.assertTrue(problem.validate(codes)['valid'])
                accepted = [codes for codes in candidates if all(domain & (1 << code) for domain, code in zip(domains, codes))]
                self.assertEqual(len(accepted), 1)
                # Every cell and all three rotations inside the chosen face
                # remain available to the anchor; other tiles remain untouched.
                for cell in board.cells:
                    for code in range(3*problem.n):
                        self.assertEqual(bool(domains[cell.index] & (1 << code)),
                                         code//3 != anchor or cell.face == face)
            self.assertEqual(problem.domains, original_domains)
            self.assertEqual(problem.fingerprint, original_fingerprint)

    def test_fixed_or_narrowed_roots_are_not_restricted(self):
        data, board, reference = planted_gold(17, 1)
        fixed = SearchProblem(data, board, fixed={0: reference[0]})
        domains, metadata = canonical_root_domains(fixed)
        self.assertFalse(metadata['applied'])
        self.assertEqual(domains, [sum(1 << code for code in row) for row in fixed.domains])
        problem = SearchProblem(data, board)
        problem.domains[0].pop()
        domains, metadata = canonical_root_domains(problem)
        self.assertFalse(metadata['applied'])
        self.assertEqual(domains[0].bit_count(), 3*problem.n-1)

    def test_noncanonical_and_corrupted_geometry_are_not_assumed_symmetric(self):
        problem = SearchProblem(simple_tiles(), double_face())
        self.assertFalse(canonical_root_domains(problem)[1]['applied'])
        board = build_board(1)
        cells = list(board.cells)
        cells[0] = replace(cells[0], vertices=tuple(reversed(cells[0].vertices)))
        with self.assertRaises(ValueError):
            proper_rotations(replace(board, cells=tuple(cells)))
        data, board, _ = planted_gold(17, 1)
        problem = SearchProblem(data, board)
        problem.neighbors[0, 0] = -1
        self.assertFalse(canonical_root_domains(problem)[1]['applied'])
        problem = SearchProblem(data, board)
        problem.masks[1, 0] ^= 1
        self.assertFalse(canonical_root_domains(problem)[1]['applied'])

    def test_checks_and_argument_validation(self):
        data, board, _ = planted_gold(17, 1)
        problem = SearchProblem(data, board)
        for args in ({'anchor_tile': True}, {'anchor_tile': 10}, {'face': -1}, {'face': 10}):
            with self.assertRaises(ValueError):
                canonical_root_domains(problem, **args)
        class Stopped(Exception):
            pass
        def check():
            raise Stopped()
        with self.assertRaises(Stopped):
            proper_rotations(board, check)
        with self.assertRaises(Stopped):
            canonical_root_domains(problem, check=check)
        transform = proper_rotations(board)[0]
        for codes in ([], [-1]*10, [True]*10, [30]*10):
            with self.assertRaises(ValueError):
                transform.apply(codes)


if __name__ == '__main__':
    unittest.main()
