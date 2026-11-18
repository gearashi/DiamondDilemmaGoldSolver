"""Targeted annealing on the matched-seam score, plus an exact completion check.

The kernel maximizes matched seams under the official rules (each physical
tile once, three rotations, no flips). It is a heuristic: it certifies nothing
by itself, and every board it reports is rescored by validator.py.

`AnnealModel` holds the colour tables. `AnnealModel.run` executes Metropolis
moves at one temperature, in the compiled kernel when a C compiler is
available and in the pure-Python kernel otherwise. Both kernels use the same
move set; they do not share a random stream, so equal seeds give different
boards on the two backends.

`completion_check` answers one exact question about a board: keeping every
cell whose three seams already match, can the remaining cells be refilled so
that all seams match? It is a complete search of that restricted question
only, and says nothing about boards that differ in the kept cells.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import random
import time
from pathlib import Path

from build_native import _build_one, library_name
from geometry import build_board
from validator import normalize_tiles, orientation_masks, reverse_mask, validate_arrangement

ROOT = Path(__file__).resolve().parent
MASK64 = (1 << 64) - 1
DEFAULT_TARGETED = 960        # of 1024 swap moves; measured best of 0/256/512/768/960
DEFAULT_T_HOT = 0.5
DEFAULT_T_COLD = 0.09


def _load_native():
    source = ROOT / 'anneal_native.c'
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    folder = ROOT / '.native' / key
    path = folder / library_name('anneal_native')
    if not path.is_file():
        _build_one(stem='anneal_native', output_dir=folder, manifest_name='anneal-build.json')
    dll = ctypes.CDLL(str(path))
    i32p, u16p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint16)
    dll.dd_anneal.argtypes = [ctypes.c_int, ctypes.c_int, u16p, i32p, i32p, i32p, i32p, i32p, i32p,
                              ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.c_double, ctypes.c_int,
                              i32p, i32p, ctypes.POINTER(ctypes.c_int64)]
    dll.dd_anneal.restype = ctypes.c_int
    return dll


class AnnealModel:
    """Immutable colour tables for one tile set and board."""

    def __init__(self, data, board=None, *, native='auto'):
        if native not in ('auto', 'compiled', 'python'):
            raise ValueError('native must be auto, compiled, or python.')
        self.tiles = normalize_tiles(data)
        self.board = board if board is not None else build_board()
        self.n = len(self.tiles)
        if self.n < 2 or len(self.board.cells) != self.n:
            raise ValueError('The board must have exactly one cell per physical tile.')
        masks = orientation_masks(self.tiles)
        values = sorted({int(m) for row in masks for m in row} | {reverse_mask(int(m)) for row in masks for m in row})
        colour = {mask: index for index, mask in enumerate(values)}
        self.k = len(values)
        self.revid = [colour[reverse_mask(mask)] for mask in values]
        # tc[(3*tile+rotation)*3+side]: colour shown on that cell side.
        self.tc = [colour[int(masks[code][side])] for code in range(3 * self.n) for side in range(3)]
        self.nbc, self.nbs = [-1] * (3 * self.n), [-1] * (3 * self.n)
        for a, sa, b, sb in self.board.edges:
            self.nbc[3 * a + sa], self.nbs[3 * a + sa] = b, sb
            self.nbc[3 * b + sb], self.nbs[3 * b + sb] = a, sa
        if min(self.nbc) < 0:
            raise ValueError('Annealing needs a closed board: every cell side must have a neighbour.')
        self.total_edges = len(self.board.edges)
        carriers = [[] for _ in range(self.k)]
        for tile in range(self.n):
            for c in {self.tc[9 * tile + side] for side in range(3)}:
                carriers[c].append(tile)
        self.by_start = [0]
        for row in carriers:
            self.by_start.append(self.by_start[-1] + len(row))
        self.by_tile = [tile for row in carriers for tile in row]
        self.native_error = None
        self._dll = None
        if native != 'python':
            try:
                self._dll = _load_native()
            except (OSError, RuntimeError, AttributeError) as exc:
                if native == 'compiled':
                    raise RuntimeError('The compiled annealing kernel is unavailable: ' + str(exc)) from exc
                self.native_error = str(exc)
        self.backend = 'compiled' if self._dll is not None else 'python'
        if self._dll is not None:
            self._tc = (ctypes.c_uint16 * len(self.tc))(*self.tc)
            self._revid = (ctypes.c_int32 * self.k)(*self.revid)
            self._nbc = (ctypes.c_int32 * len(self.nbc))(*self.nbc)
            self._nbs = (ctypes.c_int32 * len(self.nbs))(*self.nbs)
            self._by_start = (ctypes.c_int32 * len(self.by_start))(*self.by_start)
            self._by_tile = (ctypes.c_int32 * max(1, len(self.by_tile)))(*self.by_tile)

    # -- scoring -----------------------------------------------------------
    def check_codes(self, codes):
        if len(codes) != self.n or any(type(code) is not int or not 0 <= code < 3 * self.n for code in codes):
            raise ValueError('A board needs one integer orientation code per cell.')
        if len({code // 3 for code in codes}) != self.n:
            raise ValueError('Each physical tile must occur exactly once.')

    def score(self, codes):
        """Matched seams, counted directly from the colour tables."""
        self.check_codes(codes)
        return sum(self.tc[3 * codes[a] + sa] == self.revid[self.tc[3 * codes[b] + sb]]
                   for a, sa, b, sb in self.board.edges)

    def validate(self, codes):
        return validate_arrangement(self.tiles, list(codes), self.board)

    def random_board(self, rng):
        order = list(range(self.n))
        rng.shuffle(order)
        return [3 * tile + rng.randrange(3) for tile in order]

    # -- annealing ---------------------------------------------------------
    def run(self, codes, rng_state, iterations, temperature, *, targeted=DEFAULT_TARGETED, best=None):
        """Run `iterations` moves at one temperature.

        Returns (codes, rng_state, score, best_codes, best_score, accepted).
        `best` is an optional (codes, score) pair carried between calls.
        """
        self.check_codes(codes)
        if type(iterations) is not int or iterations < 0:
            raise ValueError('iterations must be a nonnegative integer.')
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite.')
        if type(targeted) is not int or not 0 <= targeted <= 1024:
            raise ValueError('targeted must be an integer from 0 to 1024.')
        rng_state = (int(rng_state) & MASK64) or 0x9E3779B97F4A7C15
        best_codes, best_score = (list(best[0]), int(best[1])) if best else (list(codes), -1)
        if self._dll is None:
            return _python_run(self, list(codes), rng_state, iterations, float(temperature), targeted, best_codes, best_score)
        board = (ctypes.c_int32 * self.n)(*codes)
        kept = (ctypes.c_int32 * self.n)(*best_codes)
        kept_score, state, accepted = ctypes.c_int32(best_score), ctypes.c_uint64(rng_state), ctypes.c_int64(0)
        score = self._dll.dd_anneal(self.n, self.k, self._tc, self._revid, self._nbc, self._nbs,
                                    self._by_start, self._by_tile, board, ctypes.byref(state),
                                    iterations, float(temperature), targeted, kept, ctypes.byref(kept_score),
                                    ctypes.byref(accepted))
        if score < 0:
            raise RuntimeError('The compiled annealing kernel rejected its input or lost its score invariant.')
        return list(board), state.value, score, list(kept), kept_score.value, accepted.value


def _python_run(model, codes, rng_state, iterations, temperature, targeted, best_codes, best_score):
    """Reference kernel: the same moves as anneal_native.c, in plain Python."""
    n, tc, revid, nbc, nbs = model.n, model.tc, model.revid, model.nbc, model.nbs
    rng = random.Random(rng_state)
    tile = [code // 3 for code in codes]
    rot = [code % 3 for code in codes]
    pos = [0] * n
    need = [0] * (3 * n)

    def place(c, t, r):
        tile[c], rot[c], pos[t] = t, r, c
        base = 9 * t + 3 * r
        for side in range(3):
            need[3 * nbc[3 * c + side] + nbs[3 * c + side]] = revid[tc[base + side]]

    def fit(t, r, c):
        base, want = 9 * t + 3 * r, 3 * c
        return (tc[base] == need[want]) + (tc[base + 1] == need[want + 1]) + (tc[base + 2] == need[want + 2])

    for c in range(n):
        place(c, tile[c], rot[c])
    cur = sum(fit(tile[c], rot[c], c) for c in range(n)) // 2
    if best_score < 0 or cur > best_score:
        best_score, best_codes = cur, [3 * tile[c] + rot[c] for c in range(n)]
    accept = [math.exp(-d / temperature) for d in range(8)]
    accepted = 0
    for _ in range(iterations):
        a = rng.randrange(n)
        if rng.randrange(4) == 0:
            t, old = tile[a], rot[a]
            new = (old + 1 + rng.randrange(2)) % 3
            d = fit(t, new, a) - fit(t, old, a)
            if d >= 0 or rng.random() < accept[-d]:
                place(a, t, new); cur += d; accepted += 1
        else:
            if rng.randrange(1024) < targeted:
                side = -1
                for _try in range(4):
                    base = 9 * tile[a] + 3 * rot[a]
                    first = rng.randrange(3)
                    for q in range(3):
                        x = (first + q) % 3
                        if tc[base + x] != need[3 * a + x]:
                            side = x
                            break
                    if side >= 0:
                        break
                    a = rng.randrange(n)
                if side < 0:
                    continue
                want = need[3 * a + side]
                lo, hi = model.by_start[want], model.by_start[want + 1]
                if hi == lo:
                    continue
                b = pos[model.by_tile[rng.randrange(lo, hi)]]
            else:
                b = rng.randrange(n)
            if a == b:
                continue
            ta, tb, oa, ob = tile[a], tile[b], rot[a], rot[b]
            sa = next((x for x in range(3) if nbc[3 * a + x] == b), -1)
            if sa < 0:
                before = fit(ta, oa, a) + fit(tb, ob, b)
                start = rng.randrange(3)
                ra = max(((start + q) % 3 for q in range(3)), key=lambda r: fit(tb, r, a))
                start = rng.randrange(3)
                rb = max(((start + q) % 3 for q in range(3)), key=lambda r: fit(ta, r, b))
                d = fit(tb, ra, a) + fit(ta, rb, b) - before
            else:
                sb = nbs[3 * a + sa]
                shared = tc[9 * ta + 3 * oa + sa] == need[3 * a + sa]
                before = fit(ta, oa, a) + fit(tb, ob, b) - shared
                best_value, ra, rb, start = -1, 0, 0, rng.randrange(9)
                for q in range(9):
                    xa, xb = divmod((start + q) % 9, 3)
                    na, nb = 9 * tb + 3 * xa, 9 * ta + 3 * xb
                    value = sum(tc[na + x] == need[3 * a + x] for x in range(3) if x != sa)
                    value += sum(tc[nb + x] == need[3 * b + x] for x in range(3) if x != sb)
                    value += tc[na + sa] == revid[tc[nb + sb]]
                    if value > best_value:
                        best_value, ra, rb = value, xa, xb
                d = best_value - before
            if d >= 0 or rng.random() < accept[-d]:
                place(a, tb, ra); place(b, ta, rb); cur += d; accepted += 1
        if cur > best_score:
            best_score, best_codes = cur, [3 * tile[c] + rot[c] for c in range(n)]
    final = [3 * tile[c] + rot[c] for c in range(n)]
    return final, rng.getrandbits(64) or 1, cur, best_codes, best_score, accepted


def cooling_temperature(step, steps, hot=DEFAULT_T_HOT, cold=DEFAULT_T_COLD):
    """Geometric cooling: temperature for chunk `step` of `steps`."""
    if steps < 1 or not 0 <= step < steps or not 0 < cold <= hot:
        raise ValueError('Invalid cooling schedule.')
    return hot * (cold / hot) ** (step / steps)


def completion_check(model, codes, *, rings=0, node_limit=20000, seconds=None, cancelled=None):
    """Exact test: can the mismatched part of this board be refilled to match every seam?

    Cells touching a mismatched seam are freed, plus `rings` further layers of
    neighbours. All other cells keep their tile and rotation. The freed cells
    are searched completely (most-constrained cell first) unless `node_limit`,
    the optional time budget `seconds`, or the optional `cancelled()` callback
    ends the search first; all three give 'undetermined'. Result keys: status ('already_matched', 'completion_found',
    'no_completion' or 'undetermined'), free_cells, nodes, and for a found
    completion its codes and validator report.
    """
    model.check_codes(codes)
    if type(rings) is not int or rings < 0 or type(node_limit) is not int or node_limit < 1:
        raise ValueError('rings must be nonnegative and node_limit positive.')
    if seconds is not None and (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not seconds > 0):
        raise ValueError('seconds must be positive when given.')
    n, tc, revid, nbc, nbs = model.n, model.tc, model.revid, model.nbc, model.nbs
    free = set()
    for a, sa, b, sb in model.board.edges:
        if tc[3 * codes[a] + sa] != revid[tc[3 * codes[b] + sb]]:
            free.update((a, b))
    if not free:
        return {'status': 'already_matched', 'free_cells': 0, 'rings': rings, 'nodes': 0}
    for _ in range(rings):
        free |= {nbc[3 * c + side] for c in free for side in range(3)}
    placed = [-1 if c in free else codes[c] for c in range(n)]
    spare = {codes[c] // 3 for c in free}
    nodes = 0

    def options(c):
        wants = [(side, revid[tc[3 * placed[nbc[3 * c + side]] + nbs[3 * c + side]]])
                 for side in range(3) if placed[nbc[3 * c + side]] >= 0]
        if wants:                               # only tiles that carry the first wanted colour
            first = wants[0][1]
            pool = [t for t in model.by_tile[model.by_start[first]:model.by_start[first + 1]] if t in spare]
        else:
            pool = sorted(spare)
        return [3 * t + r for t in pool for r in range(3)
                if all(tc[9 * t + 3 * r + side] == want for side, want in wants)]

    deadline = None if seconds is None else time.monotonic() + seconds

    def search(remaining):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit or (deadline is not None and time.monotonic() > deadline) or (cancelled is not None and cancelled()):
            return None
        if not remaining:
            return True
        chosen, choices = None, None
        for c in remaining:
            if all(placed[nbc[3 * c + side]] < 0 for side in range(3)):
                continue
            found = options(c)
            if not found:
                return False
            if choices is None or len(found) < len(choices):
                chosen, choices = c, found
                if len(found) == 1:
                    break
        if chosen is None:                      # an island with no fixed neighbour
            chosen = min(remaining)
            choices = options(chosen)
        remaining.discard(chosen)
        for code in choices:
            placed[chosen] = code
            spare.discard(code // 3)
            outcome = search(remaining)
            if outcome is not False:
                if outcome is None:
                    placed[chosen] = -1
                    spare.add(code // 3)
                    remaining.add(chosen)
                return outcome
            spare.add(code // 3)
        placed[chosen] = -1
        remaining.add(chosen)
        return False

    outcome = search(set(free))
    result = {'free_cells': len(free), 'rings': rings, 'nodes': min(nodes, node_limit)}
    if outcome is None:
        result['status'] = 'undetermined'
    elif outcome:
        report = model.validate(placed)
        result.update(status='completion_found', codes=list(placed), validation=report)
    else:
        result['status'] = 'no_completion'
    return result


def exact_engine_check(data, codes, *, rings=2, seconds=20, cancelled=None, board=None):
    """The question of completion_check, answered by the app's constraint search.

    The same cells are freed (those touching a mismatched seam, plus `rings`
    layers of neighbours) and all others keep their tile and rotation. The
    constraint search propagates far more than completion_check does, so it
    usually settles a wider ring without branching. Same result keys and
    statuses; 'undetermined' when the time budget or `cancelled()` ends it.
    """
    from search_problem import SearchProblem
    import constraint_dfs
    if type(rings) is not int or rings < 0:
        raise ValueError('rings must be nonnegative.')
    whole = SearchProblem(data, board)
    free = set()
    for edge in whole.validate(codes)['unmatched_edges']:
        free.update((edge['cell_a'], edge['cell_b']))
    if not free:
        return {'status': 'already_matched', 'free_cells': 0, 'rings': rings, 'nodes': 0}
    for _ in range(rings):
        free |= {int(whole.neighbors[cell][side]) for cell in free for side in range(3)}
    problem = SearchProblem(data, board, fixed={cell: codes[cell] for cell in range(whole.n) if cell not in free})
    found = constraint_dfs.solve(problem, seconds=seconds, target='edges', stop=cancelled, options={'native': 'auto'})
    result = {'free_cells': len(free), 'rings': rings, 'nodes': int(found.get('nodes') or 0), 'engine': 'constraint search'}
    if found.get('codes'):
        report = whole.validate(found['codes'])
        if report['matched_edges'] == report['total_edges'] and report['unique_tiles']:
            result.update(status='completion_found', codes=[int(code) for code in found['codes']], validation=report)
            return result
    result['status'] = 'no_completion' if found.get('status') == 'infeasible' and found.get('complete') else 'undetermined'
    return result
