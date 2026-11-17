/* Diamond Dilemma Gold: native annealing kernel for the matched-seam score.
 *
 * One call runs `iters` Metropolis moves at a fixed temperature on one board.
 * The caller (anneal_search.py) owns the cooling schedule, replica exchange,
 * checkpoints and validation. Official rules only: three rotations, no flips.
 *
 * Board state: tile[c], rot[c] for each cell c. code = 3*tile + rot.
 * tc[(3*t+r)*3+s] is the colour id that tile t in rotation r shows on cell
 * side s. need[c*3+s] is the colour that side s of cell c must show to match
 * its neighbour. A seam is matched when both sides agree, so the score is
 * half the number of matched cell sides.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>

#define DD_MAX_CELLS 512
#define DD_MAX_COLOURS 512

typedef struct {
    int n, k;
    const int32_t *nbc, *nbs;      /* neighbour cell and neighbour side, n*3 */
    const int32_t *revid;          /* colour that matches colour i */
    const uint16_t *tc;            /* n*3*3 colours */
    const int32_t *by_start, *by_tile; /* tiles that carry colour i somewhere */
    int32_t tile[DD_MAX_CELLS], rot[DD_MAX_CELLS], pos[DD_MAX_CELLS];
    uint16_t need[DD_MAX_CELLS * 3];
    uint64_t rng;
} dd_state;

static inline uint32_t dd_rand(dd_state *s) {
    s->rng ^= s->rng << 13; s->rng ^= s->rng >> 7; s->rng ^= s->rng << 17;
    return (uint32_t)(s->rng >> 16);
}

static inline int dd_fit(const dd_state *s, int t, int r, int c) {
    const uint16_t *col = s->tc + (3 * t + r) * 3, *nd = s->need + 3 * c;
    return (col[0] == nd[0]) + (col[1] == nd[1]) + (col[2] == nd[2]);
}

static inline void dd_place(dd_state *s, int c, int t, int r) {
    const uint16_t *col = s->tc + (3 * t + r) * 3;
    s->tile[c] = t; s->rot[c] = r; s->pos[t] = c;
    for (int side = 0; side < 3; side++)
        s->need[s->nbc[3 * c + side] * 3 + s->nbs[3 * c + side]] = (uint16_t)s->revid[col[side]];
}

static int dd_score(const dd_state *s) {
    int sides = 0;
    for (int c = 0; c < s->n; c++) sides += dd_fit(s, s->tile[c], s->rot[c], c);
    return sides / 2;
}

/* Returns the final score, or -1 for invalid input.
 * codes: in/out board. best_codes/best_score: in/out best board seen so far
 * (pass best_score < 0 to initialise from the input board).
 * targeted_per_1024: share of swap moves that pick a mismatched side and
 * bring in a tile carrying the colour that side needs.
 * accepted: optional counter of accepted moves. */
int dd_anneal(int n, int k, const uint16_t *tc, const int32_t *revid,
              const int32_t *nbc, const int32_t *nbs,
              const int32_t *by_start, const int32_t *by_tile,
              int32_t *codes, uint64_t *rng_state, int64_t iters, double temperature,
              int targeted_per_1024, int32_t *best_codes, int32_t *best_score,
              int64_t *accepted) {
    static const int next_rot[3][2] = {{1, 2}, {2, 0}, {0, 1}};
    dd_state st; dd_state *s = &st;
    uint32_t accept_at[8];
    int64_t taken = 0;
    if (n < 2 || n > DD_MAX_CELLS || k < 1 || k > DD_MAX_COLOURS || temperature <= 0) return -1;
    s->n = n; s->k = k; s->tc = tc; s->revid = revid; s->nbc = nbc; s->nbs = nbs;
    s->by_start = by_start; s->by_tile = by_tile;
    s->rng = *rng_state ? *rng_state : 0x9E3779B97F4A7C15ULL;
    memset(s->pos, 0xff, sizeof s->pos);
    for (int c = 0; c < n; c++) {
        int t = codes[c] / 3;
        if (codes[c] < 0 || t >= n || s->pos[t] >= 0) return -1;
        s->tile[c] = t; s->rot[c] = codes[c] % 3; s->pos[t] = c;
    }
    for (int c = 0; c < n; c++) dd_place(s, c, s->tile[c], s->rot[c]);
    /* A move changes at most six cell sides, so one table replaces exp(). */
    for (int d = 0; d < 8; d++) {
        double p = exp(-(double)d / temperature);
        accept_at[d] = p >= 1.0 ? 0xffffffffu : (uint32_t)(p * 4294967295.0);
    }
    int cur = dd_score(s);
    if (*best_score < 0 || cur > *best_score) {
        *best_score = cur;
        for (int c = 0; c < n; c++) best_codes[c] = 3 * s->tile[c] + s->rot[c];
    }
    for (int64_t it = 0; it < iters; it++) {
        uint32_t r0 = dd_rand(s);
        int a = (int)(((uint64_t)dd_rand(s) * (uint32_t)n) >> 32);
        if ((r0 & 3) == 0) {                      /* rotate one tile */
            int t = s->tile[a], old = s->rot[a], nr = next_rot[old][(r0 >> 2) & 1];
            int d = dd_fit(s, t, nr, a) - dd_fit(s, t, old, a);
            if (d >= 0 || dd_rand(s) < accept_at[-d]) { dd_place(s, a, t, nr); cur += d; taken++; }
        } else {                                  /* swap two tiles */
            int b;
            if ((int)((r0 >> 2) & 1023) < targeted_per_1024) {
                /* find a mismatched side near a, then a tile that could fix it */
                int side = -1, tries = 0;
                for (; tries < 4; tries++) {
                    const uint16_t *col = s->tc + (3 * s->tile[a] + s->rot[a]) * 3;
                    int first = (int)(dd_rand(s) % 3);
                    for (int q = 0; q < 3; q++) {
                        int x = (first + q) % 3;
                        if (col[x] != s->need[3 * a + x]) { side = x; break; }
                    }
                    if (side >= 0) break;
                    a = (int)(((uint64_t)dd_rand(s) * (uint32_t)n) >> 32);
                }
                if (side < 0) continue;
                int want = s->need[3 * a + side];
                int lo = by_start[want], hi = by_start[want + 1];
                if (hi == lo) continue;
                b = s->pos[by_tile[lo + (int)(((uint64_t)dd_rand(s) * (uint32_t)(hi - lo)) >> 32)]];
            } else {
                b = (int)(((uint64_t)dd_rand(s) * (uint32_t)n) >> 32);
            }
            if (a == b) continue;
            int ta = s->tile[a], tb = s->tile[b], oa = s->rot[a], ob = s->rot[b];
            int sa = -1;
            for (int x = 0; x < 3; x++) if (nbc[3 * a + x] == b) sa = x;
            int d, ra = 0, rb = 0;
            if (sa < 0) {
                int before = dd_fit(s, ta, oa, a) + dd_fit(s, tb, ob, b);
                int fa = -1, fb = -1;
                uint32_t tie = dd_rand(s);
                for (int q = 0; q < 3; q++) {
                    int r = (int)((tie + q) % 3), f = dd_fit(s, tb, r, a);
                    if (f > fa) { fa = f; ra = r; }
                }
                tie >>= 8;
                for (int q = 0; q < 3; q++) {
                    int r = (int)((tie + q) % 3), f = dd_fit(s, ta, r, b);
                    if (f > fb) { fb = f; rb = r; }
                }
                d = fa + fb - before;
            } else {
                /* neighbours: the shared seam depends on both rotations */
                int sb = nbs[3 * a + sa];
                const uint16_t *ca = s->tc + (3 * ta + oa) * 3, *cb = s->tc + (3 * tb + ob) * 3;
                int shared = ca[sa] == s->need[3 * a + sa];
                int before = dd_fit(s, ta, oa, a) + dd_fit(s, tb, ob, b) - shared;
                int bestv = -1;
                uint32_t tie = dd_rand(s);
                for (int q = 0; q < 9; q++) {
                    int idx = (int)((tie + q) % 9), xa = idx / 3, xb = idx % 3;
                    const uint16_t *na = s->tc + (3 * tb + xa) * 3, *nb2 = s->tc + (3 * ta + xb) * 3;
                    int v = 0;
                    for (int x = 0; x < 3; x++) {
                        if (x != sa) v += na[x] == s->need[3 * a + x];
                        if (x != sb) v += nb2[x] == s->need[3 * b + x];
                    }
                    v += na[sa] == (uint16_t)revid[nb2[sb]];
                    if (v > bestv) { bestv = v; ra = xa; rb = xb; }
                }
                d = bestv - before;
            }
            if (d >= 0 || dd_rand(s) < accept_at[-d]) {
                dd_place(s, a, tb, ra); dd_place(s, b, ta, rb); cur += d; taken++;
            }
        }
        if (cur > *best_score) {
            *best_score = cur;
            for (int c = 0; c < n; c++) best_codes[c] = 3 * s->tile[c] + s->rot[c];
        }
    }
    for (int c = 0; c < n; c++) codes[c] = 3 * s->tile[c] + s->rot[c];
    *rng_state = s->rng;
    if (accepted) *accepted += taken;
    return dd_score(s) == cur ? cur : -2;
}
