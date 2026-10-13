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
        # Endpoint-to-segment tables belong to this immutable tile model. Domain
        # restrictions and branch assignments never alter them. Each seam pair
        # can then reuse local segment joins without reconstructing point tuples.
        self._segment_counts = tuple(len(tile['segments']) for tile in self.tiles)
        ports = []
        for tile in self.tiles:
            base = [[-1]*11 for _ in range(3)]
            for segment, endpoints in enumerate(tile['segments']):
                for side, position in endpoints:
                    base[side][position-1] = segment
            ports.extend(tuple(tuple(base[(side-rotation)%3]) for side in range(3))
                         for rotation in range(3))
        self._segment_ports = tuple(ports)
        self._seam_joins = {}
        self._seam_cache_limit = 4096
        self._closed_key, self._closed_value = None, ()
        encoded = {'tiles': self.tiles, 'edges': self.edges, 'fixed': sorted(self.fixed.items())}
        self.fingerprint = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate(self, codes):
        # Preserve the independent validator's strict integer/type checks.
        return validate_arrangement(self.tiles, codes, self.board)

    def _closed_components(self, assignment):
        if len(assignment) != self.n or any(type(code) is not int or not -1 <= code < 3*self.n for code in assignment):
            raise ValueError('A partial assignment needs one integer code or -1 per cell.')
        key = tuple(assignment)
        if key == self._closed_key:
            # Return fresh containers: a caller cannot poison the cached cut.
            return [(size, list(cells)) for size, cells in self._closed_value]
        offsets, cells, size = [-1]*self.n, [], 0
        for cell, code in enumerate(assignment):
            if code >= 0:
                count = self._segment_counts[code//3]
                offsets[cell] = size
                cells.extend([cell]*count)
                size += count
        parents, degrees = list(range(size)), [0]*size
        def find(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x
        cache, ports = self._seam_joins, self._segment_ports
        for a, sa, b, sb in self.edges:
            ac, bc = assignment[a], assignment[b]
            if ac < 0 or bc < 0:
                continue
            seam = (ac, sa, bc, sb)
            joins = cache.get(seam)
            if joins is None:
                left, right = ports[ac][sa], ports[bc][sb]
                joins = tuple((segment, right[10-position]) for position, segment in enumerate(left)
                              if segment >= 0 and right[10-position] >= 0)
                if len(cache) >= self._seam_cache_limit:
                    # FIFO eviction loses reuse only; it never removes choices.
                    del cache[next(iter(cache))]
                cache[seam] = joins
            aoffset, boffset = offsets[a], offsets[b]
            for left, right in joins:
                left += aoffset
                right += boffset
                x, y = find(left), find(right)
                if x != y:
                    parents[y] = x
                degrees[left] += 1
                degrees[right] += 1
        groups = {}
        for index in range(size):
            groups.setdefault(find(index), []).append(index)
        result = tuple((len(group), tuple(sorted({cells[i] for i in group}))) for group in groups.values()
                       if all(degrees[i] == 2 for i in group))
        self._closed_key, self._closed_value = key, result
        return [(size, list(cells)) for size, cells in result]

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
