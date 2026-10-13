"""Verified proper rotations and an optional fresh-search symmetry reduction.

The ten rotations of the pentagonal bipyramid act freely on its ten macrofaces.
Restricting one labelled tile to one macroface therefore keeps exactly one member
of every full-board rotation orbit. No cell within that face, tile orientation,
or reflected arrangement is fixed. This module never changes a SearchProblem,
its fingerprint, or an existing checkpoint. Callers must persist this choice as
part of fresh-search state and must not silently apply it to a resumed search.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from geometry import Board, build_board


@dataclass(frozen=True)
class ProperRotation:
    macro_vertices: tuple
    vertices: tuple
    cells: tuple
    side_offsets: tuple
    faces: tuple

    def apply(self, codes):
        n = len(self.cells)
        if len(codes) != n or any(type(code) is not int or not 0 <= code < 3*n for code in codes):
            raise ValueError('A rotation needs one valid orientation code per cell.')
        result = [None]*n
        for cell, code in enumerate(codes):
            tile, rotation = divmod(code, 3)
            result[self.cells[cell]] = 3*tile+(rotation+self.side_offsets[cell]) % 3
        return result


def proper_rotations(board, check=lambda: None):
    """Construct and verify all ten orientation-preserving mesh automorphisms.

    Exact barycentric vertex keys eliminate geometric tolerances. Only the
    canonical mesh produced by build_board is accepted; arbitrary topologies
    cannot acquire a symmetry assumption accidentally.
    """
    check()
    if not isinstance(board, Board) or board != build_board(board.subdivision):
        raise ValueError('Proper rotations require the canonical bipyramid mesh.')
    vertex_ids = {key: index for index, key in enumerate(board.vertices)}
    cell_ids = {frozenset(cell.vertices): cell.index for cell in board.cells}
    face_ids = {frozenset(face): index for index, face in enumerate(board.macro_faces)}
    transforms = []
    # Five polar rotations and five half-turns through equatorial axes. A
    # half-turn swaps the poles AND reverses the cyclic order of the equator.
    for swapped in (False, True):
        for shift in range(5):
            check()
            macro = ((1, 0) if swapped else (0, 1)) + tuple(
                2+(shift-i if swapped else shift+i) % 5 for i in range(5))
            vertex_map = tuple(vertex_ids[tuple(sorted((macro[v], weight) for v, weight in key))]
                               for key in board.vertices)
            cell_map, offsets = [], []
            for cell in board.cells:
                check()
                mapped = tuple(vertex_map[v] for v in cell.vertices)
                other = cell_ids[frozenset(mapped)]
                target = board.cells[other].vertices
                delta = target.index(mapped[0])
                if any(mapped[side] != target[(side+delta) % 3] for side in range(3)):
                    raise ValueError('A proposed transform reverses a triangle orientation.')
                cell_map.append(other)
                offsets.append(delta)
            face_map = tuple(face_ids[frozenset(macro[v] for v in face)] for face in board.macro_faces)
            if (len(set(vertex_map)) != len(vertex_map) or len(set(cell_map)) != len(cell_map)
                    or len(set(face_map)) != 10):
                raise ValueError('A proposed transform is not bijective.')
            for cell in board.cells:
                check()
                if board.cells[cell_map[cell.index]].face != face_map[cell.face]:
                    raise ValueError('A proposed transform does not preserve macrofaces.')
                for side, (neighbor, other_side) in enumerate(board.neighbors[cell.index]):
                    mapped_side = (side+offsets[cell.index]) % 3
                    expected = (cell_map[neighbor], (other_side+offsets[neighbor]) % 3)
                    if board.neighbors[cell_map[cell.index]][mapped_side] != expected:
                        raise ValueError('A proposed transform does not preserve oriented seams.')
            transforms.append(ProperRotation(macro, vertex_map, tuple(cell_map), tuple(offsets), face_map))
    signatures = {t.macro_vertices for t in transforms}
    if len(signatures) != 10:
        raise ValueError('The proper-rotation group must have ten distinct elements.')
    for left in transforms:
        check()
        for right in transforms:
            if tuple(left.macro_vertices[right.macro_vertices[v]] for v in range(7)) not in signatures:
                raise ValueError('The proposed transforms are not closed under composition.')
    if any(len({t.faces[face] for t in transforms}) != 10 for face in range(10)):
        raise ValueError('Proper rotations must act freely and transitively on macrofaces.')
    return tuple(transforms)


def canonical_root_domains(problem, *, anchor_tile=0, face=0, check=lambda: None):
    """Return (new integer-bitset domains, metadata) without mutating input.

    This is a NEW-search-only helper. Explicit fixed placements, narrowed initial
    domains, or noncanonical geometry cause a documented no-op. A caller that
    opts in must bind metadata and the returned root domains to its checkpoint.
    The usual puzzle fingerprint alone does not encode this search restriction.
    """
    check()
    if type(anchor_tile) is not int or not 0 <= anchor_tile < problem.n:
        raise ValueError('anchor_tile must identify a physical tile.')
    if type(face) is not int or not 0 <= face < 10:
        raise ValueError('face must be a macroface index from zero through nine.')
    domains = [sum(1 << code for code in set(row)) for row in problem.domains]
    metadata = {'version': 1, 'method': 'proper_rotation_macroface', 'applied': False,
                'anchor_tile': anchor_tile, 'face': face, 'rotations': 0,
                'removed_orientation_values': 0}
    def skip(reason):
        metadata['reason'] = reason
        return domains, metadata
    if problem.fixed:
        return skip('Explicit fixed placements can break board symmetry.')
    full = (1 << (3*problem.n))-1
    if len(domains) != problem.n or any(domain != full for domain in domains):
        return skip('All initial cell domains must contain every orientation code.')
    try:
        rotations = proper_rotations(problem.board, check)
    except ValueError as exc:
        return skip(str(exc))
    if (len(problem.board.cells) != problem.n
            or tuple(tuple(int(v) for v in row) for row in problem.neighbors) != problem.board.neighbor_cells
            or tuple(tuple(int(v) for v in row) for row in problem.sides) != problem.board.neighbor_sides):
        return skip('Problem adjacency differs from the verified mesh.')
    for tile in range(problem.n):
        check()
        for rotation in range(3):
            if any(problem.masks[3*tile+rotation][(side+rotation) % 3] != problem.masks[3*tile][side]
                   for side in range(3)):
                return skip('Orientation codes do not transform by proper triangle rotation.')
    anchor = 7 << (3*anchor_tile)
    for cell in problem.board.cells:
        if cell.face != face:
            domains[cell.index] &= ~anchor
    metadata.update(applied=True, rotations=len(rotations),
                    removed_orientation_values=sum((full ^ domain).bit_count() for domain in domains),
                    reason='Exactly one macroface representative per proper-rotation orbit.')
    metadata['root_domains_sha256'] = hashlib.sha256(json.dumps(domains, separators=(',', ':')).encode()).hexdigest()
    return domains, metadata
