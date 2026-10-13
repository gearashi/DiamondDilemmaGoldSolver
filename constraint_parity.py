"""Exact mod-two mesh constraints for the verified pentagonal bipyramid.

Every tile has an even number of endpoints, so the parity of seam crossings
is a vertex XOR gradient on this triangulated sphere. Summing all seam parities
leaves only odd-degree vertices: the two apices. Their XOR equals the parity of
the total number of gold segments. The rule applies to edge and Gold targets.
Unknown/custom board topologies deliberately disable this optimization.
"""
from dataclasses import dataclass
from geometry import Board, build_board

STAT_KEYS = ('parity_checks', 'parity_prunes', 'parity_failures')

@dataclass(frozen=True)
class ParityTables:
    """Completed, immutable topology deduction; no domains or callbacks."""
    binding: tuple
    enabled: bool
    vertex_count: int
    odd_codes: tuple
    edges: tuple
    odd_vertices: tuple
    total_parity: int


def _binding(problem, check):
    board = problem.board
    if not isinstance(board, Board):
        return (None,)
    masks = []
    for row in getattr(problem, 'masks', ()):
        check()
        masks.append(tuple(map(int, row)))
    return (board, tuple(masks), tuple(tuple(edge) for edge in getattr(problem, 'edges', ())),
            getattr(problem, 'total_segments', None))


class ParityPruner:
    def __init__(self, problem, *, tables=None, check=lambda: None):
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        check()
        binding = _binding(problem, check)
        if tables is not None:
            if not isinstance(tables, ParityTables) or tables.binding != binding:
                raise ValueError('Parity tables differ from the current puzzle topology or masks.')
            self.tables = tables
            self.enabled = tables.enabled
            self.vertex_count, self.odd_codes = tables.vertex_count, tables.odd_codes
            self.edges, self.odd_vertices = tables.edges, tables.odd_vertices
            self.total_parity = tables.total_parity
            return
        self.enabled = False
        self._construct(problem, check)
        self.tables = ParityTables(binding, self.enabled,
            getattr(self, 'vertex_count', 0), tuple(getattr(self, 'odd_codes', ())),
            tuple(getattr(self, 'edges', ())), tuple(getattr(self, 'odd_vertices', ())),
            getattr(self, 'total_parity', 0))

    def _construct(self, problem, check):
        board = problem.board
        if not isinstance(board, Board) or board != build_board(board.subdivision):
            return
        # SearchProblem also checks the actual seam incidence. Use that exact
        # incidence, and accept only the established sphere geometry.
        if tuple(problem.edges) != tuple(board.edges):
            return
        self.vertex_count = len(board.vertices)
        self.odd_codes = [0, 0, 0]
        for code, masks in enumerate(problem.masks):
            check()
            for side, mask in enumerate(masks):
                if int(mask).bit_count() % 2:
                    self.odd_codes[side] |= 1 << code
        degrees = [0] * self.vertex_count
        self.edges = []
        for a, sa, b, sb in problem.edges:
            check()
            vertices = board.cells[a].vertices
            u, v = vertices[sa], vertices[(sa+1)%3]
            degrees[u] += 1
            degrees[v] += 1
            self.edges.append((a, sa, b, sb, u, v))
        self.odd_vertices = [v for v, degree in enumerate(degrees) if degree % 2]
        if len(self.odd_vertices) != 2:
            return
        self.total_parity = problem.total_segments % 2
        self.enabled = True

    def prune(self, domains, check=lambda: None):
        if not self.enabled or any(value == 0 for value in domains):
            return False
        self.stats['parity_checks'] += 1
        parent = list(range(self.vertex_count))
        sizes = [1] * self.vertex_count
        parity = [0] * self.vertex_count
        def find(v):
            value = 0
            while parent[v] != v:
                value ^= parity[v]
                v = parent[v]
            return v, value
        def unite(u, v, required):
            left, x = find(u)
            right, y = find(v)
            if left == right:
                return (x ^ y) == required
            if sizes[left] < sizes[right]:
                left, right = right, left
            parent[right] = left
            parity[right] = x ^ y ^ required
            sizes[left] += sizes[right]
            return True
        def fail():
            self.stats['parity_failures'] += 1
            self.stats['parity_prunes'] += domains[0].bit_count()
            domains[0] = 0
            return True
        unite(*self.odd_vertices, self.total_parity)
        for a, sa, b, sb, u, v in self.edges:
            check()
            for cell, side in ((a, sa), (b, sb)):
                domain = domains[cell]
                odd = domain & self.odd_codes[side]
                if not odd:
                    if not unite(u, v, 0): return fail()
                elif odd == domain:
                    if not unite(u, v, 1): return fail()
        changed = False
        for a, sa, b, sb, u, v in self.edges:
            check()
            left, x = find(u)
            right, y = find(v)
            if left != right:
                continue
            required = x ^ y
            for cell, side in ((a, sa), (b, sb)):
                old = domains[cell]
                allowed = self.odd_codes[side] if required else ~self.odd_codes[side]
                new = old & allowed
                if new != old:
                    domains[cell] = new
                    self.stats['parity_prunes'] += (old ^ new).bit_count()
                    changed = True
                    if not new:
                        self.stats['parity_failures'] += 1
                        return True
        return changed
