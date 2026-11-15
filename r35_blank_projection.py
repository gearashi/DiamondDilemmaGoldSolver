"""Complete blank reciprocal side pairs after constructing the nonblank physical sewing."""
from functools import lru_cache
import time

from r27_structure_filters import _components, bits


class ProjectionTimeout(Exception):
    """The remaining complete cases cannot yet be excluded."""


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise ProjectionTimeout


class BlankProjection:
    def __init__(self, sides, options, required):
        self.sides = sides
        self.blanks = tuple(i for i, (mask, _) in enumerate(sides) if not mask)
        self.owners = sorted({dart // 3 for dart in self.blanks})
        self.required = {pair for pair in required if not sides[pair[0]][0]}
        candidates = {pair for pair in options if not sides[pair[0]][0]}
        fixed = {}
        self.possible = self.required <= candidates
        for a, b in self.required:
            if a in fixed or b in fixed:
                self.possible = False
            fixed[a], fixed[b] = b, a
        self.allowed = {(a, b) for a, b in candidates
                        if fixed.get(a, b) == b and fixed.get(b, a) == a}
        self.possible &= set(self.blanks) == {dart for pair in self.allowed for dart in pair}

    def signed(self, phases):
        return {(a,b) for a,b in self.allowed
                if (self.sides[a][1] ^ phases[a // 3]) == (self.sides[b][1] ^ phases[b // 3])}

    def matching(self, allowed, deadline, stats):
        """Find one full signed blank matching; a completed empty search rules it out."""
        if not self.possible:
            return None
        index = {dart: i for i, dart in enumerate(self.blanks)}
        edges = [set() for _ in index]
        for a,b in allowed:
            i,j = index[a], index[b]
            edges[i].add(j)
            edges[j].add(i)
        check_time(deadline)
        groups = _components(edges)
        if any(len(group) % 2 for group in groups):
            return None
        rows = [sum(1 << j for j in neighbors) for neighbors in edges]

        @lru_cache(maxsize=8192)
        def visit(left):
            check_time(deadline)
            stats['blank_matching_states'] += 1
            if not left:
                return ()
            i = min(bits(left), key=lambda n: (rows[n] & left).bit_count())
            rest = left ^ (1 << i)
            for j in bits(rows[i] & rest):
                tail = visit(rest ^ (1 << j))
                if tail is not None:
                    return (tuple(sorted((self.blanks[i], self.blanks[j]))), *tail)
            return None
        chosen = []
        for group in groups:
            part = visit(sum(1 << i for i in group))
            if part is None:
                return None
            chosen.extend(part)
        return chosen


def lift(problem, domains, nonblank, blanks, signed, deadline, stats):
    """Lift a complete connected nonblank sewing, testing every geographic root pose."""
    if not blanks.possible:
        return None
    neighbor_darts = {}
    for a,b in nonblank:
        if a in neighbor_darts or b in neighbor_darts or not blanks.sides[a][0] or not blanks.sides[b][0]:
            raise ValueError("Expected disjoint nonblank mate pairs.")
        neighbor_darts[a], neighbor_darts[b] = b, a
    if len(neighbor_darts) != 3*problem.n - len(blanks.blanks):
        raise ValueError("A partial nonblank sewing requires further physical search.")

    # One singleton domain determines its tile's only legal root. Otherwise use
    # the first physical tile and every root pose allowed by the current domains.
    root = 0
    starts = None
    for cell, domain in enumerate(domains):
        if domain and domain & (domain - 1) == 0:
            root, rot = divmod(domain.bit_length()-1, 3)
            starts = ((cell, rot),)
            break
    if starts is None:
        starts = ((cell,rot) for cell,domain in enumerate(domains)
                  for rot in range(3) if domain & (1 << (3*root+rot)))

    seen_tiles = {root}
    pending = [root]
    while pending:
        tile = pending.pop()
        for side in range(3):
            mate = neighbor_darts.get(3*tile + side)
            if mate is not None and mate//3 not in seen_tiles:
                seen_tiles.add(mate//3)
                pending.append(mate//3)
    if len(seen_tiles) != problem.n:
        raise ValueError("Disconnected nonblank sewings require strand cuts before lifting.")

    for cell,rot in starts:
        check_time(deadline)
        stats['blank_root_trials'] += 1
        board = [-1] * problem.n
        poses = {}
        todo = [(root,cell,rot)]
        while todo:
            tile,c,r = todo.pop()
            if tile in poses:
                if poses[tile] != (c,r):
                    break
                continue
            code = 3*tile + r
            if board[c] >= 0 or not domains[c] & (1 << code):
                break
            poses[tile] = (c,r)
            board[c] = code
            for s,(neighbor,opposite) in enumerate(problem.board.neighbors[c]):
                a = 3*tile + (s-r)%3
                if a in neighbor_darts:
                    other,side = divmod(neighbor_darts[a],3)
                    todo.append((other,neighbor,(opposite-side)%3))
                elif board[neighbor] >= 0:
                    other,rr = divmod(board[neighbor],3)
                    b = 3*other + (opposite-rr)%3
                    if tuple(sorted((a,b))) not in signed:
                        break
            else:
                continue
            break
        else:
            return board
    return None
