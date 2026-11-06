// One CUDA thread owns one legal tile permutation. Boards are [cell, replica].


static const int CELLS = 160;
static const int SIDES = 3;
__device__ __forceinline__ unsigned int draw(unsigned int &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
__device__ __forceinline__ int code_at(const short *board, int c, int n, int tid) {
    return (int)board[c * n + tid];
}
__device__ __forceinline__ int incident(
    const short *board, const unsigned short *faces, const unsigned short *reversed,
    const short *neighbors, const unsigned char *other_side,
    int n, int tid, int cell, int code, int replaced_cell, int replaced_code, bool skip_replaced
) {
    int score = 0;
    #pragma unroll
    for (int side = 0; side < SIDES; ++side) {
        int neighbor = neighbors[cell * SIDES + side];
        if (skip_replaced && neighbor == replaced_cell) continue;
        int nc = neighbor == replaced_cell ? replaced_code : code_at(board, neighbor, n, tid);
        score += faces[code * SIDES + side] == reversed[nc * SIDES + other_side[cell * SIDES + side]];
    }
    return score;
}
__device__ __forceinline__ int local_score(
    const short *board, const unsigned short *faces, const unsigned short *reversed,
    const short *neighbors, const unsigned char *other_side,
    int n, int tid, int a, int b, int ca, int cb
) {
    int score = incident(board, faces, reversed, neighbors, other_side, n, tid, a, ca, b, cb, false);
    if (a != b) score += incident(board, faces, reversed, neighbors, other_side, n, tid, b, cb, a, ca, true);
    return score;
}
extern "C" __global__ void score_boards(
    const short *board, const unsigned short *faces, const unsigned short *reversed,
    const short *neighbors, const unsigned char *other_side, int *scores, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int score = 0;
    for (int a = 0; a < CELLS; ++a) {
        int ca = code_at(board, a, n, tid);
        #pragma unroll
        for (int s = 0; s < SIDES; ++s) {
            int b = neighbors[a * SIDES + s];
            if (a < b) {
                int cb = code_at(board, b, n, tid);
                score += faces[ca * SIDES + s] == reversed[cb * SIDES + other_side[a * SIDES + s]];
            }
        }
    }
    scores[tid] = score;
}
extern "C" __global__ void test_delta(
    const short *board, const unsigned short *faces, const unsigned short *reversed,
    const short *neighbors, const unsigned char *other_side, const short *as,
    const short *bs, const short *cas, const short *cbs, int *deltas, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int a = as[tid], b = bs[tid];
    int olda = code_at(board, a, n, tid), oldb = code_at(board, b, n, tid);
    deltas[tid] = local_score(board, faces, reversed, neighbors, other_side, n, tid, a, b, cas[tid], cbs[tid])
                - local_score(board, faces, reversed, neighbors, other_side, n, tid, a, b, olda, oldb);
}

// Fingerprints only select candidates for an exact 160-code comparison.
__device__ __forceinline__ unsigned long long destination_hash(
    const unsigned long long *zobrist, unsigned long long current,
    int a, int b, int olda, int oldb, int newa, int newb
) {
    unsigned long long value = current ^ zobrist[a * 480 + olda] ^ zobrist[a * 480 + newa];
    if (a != b) value ^= zobrist[b * 480 + oldb] ^ zobrist[b * 480 + newb];
    return value;
}
__device__ __forceinline__ bool cache_contains(
    const short *board, const short *cached_boards, const unsigned long long *cached_hashes,
    int count, unsigned long long hash, int a, int b, int newa, int newb, int n, int tid
) {
    for (int slot = 0; slot < count; ++slot) {
        if (cached_hashes[slot * n + tid] != hash) continue;
        bool equal = true;
        for (int c = 0; c < CELLS; ++c) {
            int code = c == a ? newa : (c == b ? newb : board[c * n + tid]);
            if (cached_boards[(slot * CELLS + c) * n + tid] != code) { equal = false; break; }
        }
        if (equal) return true;
    }
    return false;
}
// Deterministic correctness probe, using the same lookup as the search kernel.
extern "C" __global__ void test_cache_probe(
    const short *board, const short *cached_boards, const unsigned long long *cached_hashes,
    const unsigned short *counts, const unsigned long long *zobrist,
    const unsigned long long *current_hashes, const short *as, const short *bs,
    const short *newas, const short *newbs, unsigned char *duplicates,
    unsigned long long *hashes, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int a = as[tid], b = bs[tid];
    unsigned long long hash = destination_hash(zobrist, current_hashes[tid], a, b,
        board[a*n+tid], board[b*n+tid], newas[tid], newbs[tid]);
    hashes[tid] = hash;
    duplicates[tid] = cache_contains(board, cached_boards, cached_hashes, counts[tid],
        hash, a, b, newas[tid], newbs[tid], n, tid);
}

extern "C" __global__ void search_moves(
    short *board, short *bestboard, short *positions,
    const unsigned short *faces, const unsigned short *reversed,
    const short *neighbors, const unsigned char *other_side,
    const int *pair_offsets, const short *pair_codes, int colors,
    const float *guidance, const float *temperatures,
    int *scores, int *bestscores, unsigned int *rng,
    unsigned long long *counters, unsigned char *pending,
    const unsigned long long *zobrist, unsigned long long *current_hashes,
    short *cached_boards, unsigned long long *cached_hashes, short *cached_scores,
    unsigned short *cache_counts, unsigned short *cache_cursors,
    unsigned long long *cache_counters, int cache_slots, int n, int steps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (pending[tid]) return; // A published candidate belongs to the CPU until acknowledged.
    unsigned int state = rng[tid];
    int score = scores[tid], best = bestscores[tid];
    unsigned long long accepted = 0, guided = 0, attempted = 0;
    float temp = temperatures[tid];
    unsigned long long current_hash = current_hashes[tid];
    unsigned long long checks = 0, hits = 0, evictions = 0, inserts = 0;
    int cache_count = cache_counts[tid], cache_cursor = cache_cursors[tid];
    for (int move = 0; move < steps; ++move) {
        ++attempted;
        int a = draw(state) % CELLS;
        if ((draw(state) & 7u) != 0u) {
            for (int trial = 0; trial < 5; ++trial) {
                int ca = code_at(board, a, n, tid);
                if (incident(board, faces, reversed, neighbors, other_side, n, tid, a, ca, -1, 0, false) != SIDES) break;
                a = draw(state) % CELLS;
            }
        }
        int b = draw(state) % CELLS;
        if ((draw(state) & 15u) == 0u) b = a;
        else if ((draw(state) * (1.0f / 4294967296.0f)) < guidance[tid]) {
            int pair = draw(state) % 3;
            int s1 = pair == 2 ? 1 : 0, s2 = pair == 0 ? 1 : 2;
            int n1 = neighbors[a * SIDES + s1], n2 = neighbors[a * SIDES + s2];
            int need1 = reversed[code_at(board, n1, n, tid) * SIDES + other_side[a * SIDES + s1]];
            int need2 = reversed[code_at(board, n2, n, tid) * SIDES + other_side[a * SIDES + s2]];
            int key = (pair * colors + need1) * colors + need2;
            int first = pair_offsets[key], last = pair_offsets[key + 1];
            if (last > first) {
                int candidate = pair_codes[first + draw(state) % (last - first)];
                b = positions[(candidate / 3) * n + tid]; ++guided;
            }
        }
        int olda = code_at(board, a, n, tid), oldb = code_at(board, b, n, tid);
        int old_local = local_score(board, faces, reversed, neighbors, other_side, n, tid, a, b, olda, oldb);
        int newa = (oldb / 3) * 3, newb = (olda / 3) * 3, new_local = -1;
        int ra0 = draw(state) % 3, rb0 = draw(state) % 3;
        if ((draw(state) & 7u) == 0u) {
            newa += ra0; newb = a == b ? newa : newb + rb0;
            new_local = local_score(board, faces, reversed, neighbors, other_side, n, tid, a, b, newa, newb);
        } else if (a == b) {
            int base = newa;
            for (int r = 0; r < 3; ++r) {
                int ca = base + (r + ra0) % 3;
                int value = incident(board, faces, reversed, neighbors, other_side, n, tid, a, ca, -1, 0, false);
                if (value > new_local) { new_local = value; newa = newb = ca; }
            }
        } else {
            bool adjacent = false;
            #pragma unroll
            for (int s = 0; s < SIDES; ++s) adjacent |= neighbors[a * SIDES + s] == b;
            int basea = newa, baseb = newb;
            if (adjacent) {
                for (int r = 0; r < 3; ++r) {
                    int ca = basea + (r + ra0) % 3;
                    for (int q = 0; q < 3; ++q) {
                        int cb = baseb + (q + rb0) % 3;
                        int value = local_score(board, faces, reversed, neighbors, other_side, n, tid, a, b, ca, cb);
                        if (value > new_local) { new_local = value; newa = ca; newb = cb; }
                    }
                }
            } else {
                int besta = -1, bestb = -1;
                for (int r = 0; r < 3; ++r) {
                    int ca = basea + (r + ra0) % 3, cb = baseb + (r + rb0) % 3;
                    int va = incident(board, faces, reversed, neighbors, other_side, n, tid, a, ca, -1, 0, false);
                    int vb = incident(board, faces, reversed, neighbors, other_side, n, tid, b, cb, -1, 0, false);
                    if (va > besta) { besta = va; newa = ca; }
                    if (vb > bestb) { bestb = vb; newb = cb; }
                }
                new_local = besta + bestb;
            }
        }
        int delta = new_local - old_local;
        float u = (draw(state) + 1.0f) * (1.0f / 4294967296.0f);
        // A move already rejected by annealing cannot revisit any search state.
        // Avoid scanning recent fingerprints for these otherwise rejected proposals.
        if (!(delta >= 0 || (temp > 0.0f && u < expf(delta / temp)))) continue;
        unsigned long long proposed_hash = current_hash;
        if (cache_slots) {
            ++checks;
            // The current board is always retained. This exact no-op test needs no table scan.
            if (newa == olda && newb == oldb) { ++hits; continue; }
            proposed_hash = destination_hash(zobrist, current_hash, a, b, olda, oldb, newa, newb);
            if (cache_contains(board, cached_boards, cached_hashes, cache_count,
                               proposed_hash, a, b, newa, newb, n, tid)) {
                ++hits; continue;
            }
        }
        {
            board[a * n + tid] = (short)newa; board[b * n + tid] = (short)newb;
            positions[(oldb / 3) * n + tid] = (short)a;
            positions[(olda / 3) * n + tid] = (short)b;
            score += delta; ++accepted;
            if (cache_slots) {
                current_hash = proposed_hash;
                if (cache_count == cache_slots) ++evictions;
                else ++cache_count;
                cached_hashes[cache_cursor * n + tid] = current_hash;
                cached_scores[cache_cursor * n + tid] = (short)score;
                for (int c = 0; c < CELLS; ++c)
                    cached_boards[(cache_cursor * CELLS + c) * n + tid] = board[c * n + tid];
                cache_cursor = (cache_cursor + 1) % cache_slots;
                ++inserts;
            }
            // Preserve every distinct perfect candidate until the CPU checks its loop.
            // After acknowledgement, no-op proposals must not repeatedly publish the
            // identical already-checked layout and prevent the replica from moving.
            bool publish = score == 240 && best < 240;
            if (score == 240 && best == 240) {
                for (int c = 0; c < CELLS; ++c) {
                    if (bestboard[c * n + tid] != board[c * n + tid]) {
                        publish = true;
                        break;
                    }
                }
            }
            if (score > best || publish) {
                best = score;
                for (int c = 0; c < CELLS; ++c) bestboard[c * n + tid] = board[c * n + tid];
            }
            if (publish) {
                pending[tid] = 1;
                break;
            }
        }
    }
    rng[tid] = state; scores[tid] = score; bestscores[tid] = best;
    counters[tid] += attempted; counters[n + tid] += accepted; counters[2 * n + tid] += guided;
    current_hashes[tid] = current_hash;
    cache_counts[tid] = (unsigned short)cache_count;
    cache_cursors[tid] = (unsigned short)cache_cursor;
    cache_counters[tid] += checks; cache_counters[n + tid] += hits;
    cache_counters[2 * n + tid] += evictions; cache_counters[3 * n + tid] += inserts;
}
