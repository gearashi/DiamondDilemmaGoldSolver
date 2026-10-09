"""Shared exact puzzle model for bounded solver comparisons.

Orientation code = 3*physical_tile + clockwise_rotation. Only a full independent
validation certifies the encoded Gold puzzle; an edge-perfect board may have
several separate loops. Tiny boards use the same cell/side edge convention.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from geometry import build_board
from validator import normalize_tiles, orientation_masks, validate_arrangement


class SearchProblem:
    def __init__(self, data, board=None, fixed=None):
        self.tiles = normalize_tiles(data)
        self.board = board if board is not None else build_board()
        self.n = len(self.tiles)
        if not self.n or len(self.board.cells) != self.n:
            raise ValueError('The board must have exactly one cell per physical tile.')
        self.masks = np.asarray(orientation_masks(self.tiles), dtype=np.uint16)
        self.neighbors = np.full((self.n, 3), -1, dtype=np.int32)
        self.sides = np.full((self.n, 3), -1, dtype=np.int8)
        edges = []
        for edge in self.board.edges:
            if len(edge) != 4 or any(type(v) is not int for v in edge):
                raise ValueError('Edges must be integer (cell,side,cell,side) tuples.')
            a, sa, b, sb = edge
            if not (0 <= a < self.n and 0 <= b < self.n and 0 <= sa < 3 and 0 <= sb < 3) or a == b:
                raise ValueError('Invalid board edge.')
            if self.neighbors[a, sa] >= 0 or self.neighbors[b, sb] >= 0:
                raise ValueError('Each cell side may belong to only one edge.')
            self.neighbors[a, sa], self.sides[a, sa] = b, sb
            self.neighbors[b, sb], self.sides[b, sb] = a, sa
            edges.append((a, sa, b, sb))
        self.edges = tuple(edges)
        if fixed is not None and not isinstance(fixed, dict):
            raise ValueError('Fixed placements must map cell indices to orientation codes.')
        self.fixed = dict(fixed or {})
        for cell, code in self.fixed.items():
            if type(cell) is not int or not 0 <= cell < self.n or type(code) is not int or not 0 <= code < 3*self.n:
                raise ValueError('A fixed cell or orientation code is out of range.')
        fixed_tiles = {code // 3 for code in self.fixed.values()}
        self.domains = []
        for cell in range(self.n):
            choices = [self.fixed[cell]] if cell in self.fixed else [code for code in range(3*self.n) if code//3 not in fixed_tiles]
            self.domains.append([code for code in choices if all(self.neighbors[cell, side] >= 0 or self.masks[code, side] == 0 for side in range(3))])
        self.total_segments = sum(len(tile['segments']) for tile in self.tiles)
        encoded = {'tiles': self.tiles, 'edges': self.edges, 'fixed': sorted(self.fixed.items())}
        self.fingerprint = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate(self, codes):
        # Preserve the independent validator's strict integer/type checks.
        return validate_arrangement(self.tiles, codes, self.board)

    def _closed_components(self, assignment):
        if len(assignment) != self.n or any(type(code) is not int or not -1 <= code < 3*self.n for code in assignment):
            raise ValueError('A partial assignment needs one integer code or -1 per cell.')
        points, cells, parents, degrees = {}, [], [], []
        for cell, code in enumerate(assignment):
            if code < 0:
                continue
            tile, rotation = divmod(code, 3)
            for segment in self.tiles[tile]['segments']:
                index = len(parents)
                parents.append(index)
                degrees.append(0)
                cells.append(cell)
                for side, p in segment:
                    points[cell, (side + rotation) % 3, p] = index
        def find(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x
        for a, sa, b, sb in self.edges:
            if assignment[a] < 0 or assignment[b] < 0:
                continue
            for p in range(1, 12):
                left, right = points.get((a, sa, p)), points.get((b, sb, 12-p))
                if left is not None and right is not None:
                    x, y = find(left), find(right)
                    if x != y:
                        parents[y] = x
                    degrees[left] += 1
                    degrees[right] += 1
        groups = {}
        for index in range(len(parents)):
            groups.setdefault(find(index), []).append(index)
        return [(len(group), sorted({cells[i] for i in group})) for group in groups.values()
                if all(degrees[i] == 2 for i in group)]

    def partial_loop_impossible(self, assignment):
        """Reject a closed proper loop, whose endpoints cannot reach future tiles."""
        return any(size < self.total_segments for size, _ in self._closed_components(assignment))

    def gold_nogood(self, codes):
        """A conjunction of placements that must change to obtain one Gold loop.

        Keeping every cell touched by a closed proper component preserves that
        component, even if those cells also contain other independent segments.
        The caller blocks this conjunction, not each placement individually.
        """
        for size, cells in self._closed_components(codes):
            if size < self.total_segments:
                return [(cell, codes[cell]) for cell in cells]
        return [(cell, code) for cell, code in enumerate(codes) if code >= 0]
