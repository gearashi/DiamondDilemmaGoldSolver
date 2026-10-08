// Constructive exact-edge DFS: persistent private stack per CUDA thread.
__device__ __forceinline__ unsigned int next_random(unsigned int &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
__device__ __forceinline__ bool taken(const unsigned int *used, int tile, int n, int tid) {
    return (used[(tile >> 5) * n + tid] & (1u << (tile & 31))) != 0;
}
__device__ __forceinline__ bool matches(
    const short *board, const unsigned short *faces, const unsigned short *reverse,
    const short *neighbors, const unsigned char *sides, int cell, int code, int n, int tid
) {
    #pragma unroll
    for (int side = 0; side < 3; ++side) {
        int nb = neighbors[cell * 3 + side], other = board[nb * n + tid];
        if (other >= 0 && faces[code * 3 + side] != reverse[other * 3 + sides[cell * 3 + side]]) return false;
    }
    return true;
}
__device__ __forceinline__ void bucket(
    const short *board, const unsigned short *reverse,
    const short *neighbors, const unsigned char *sides,
    const int *single_offsets, const int *pair_offsets, int colors,
    int cell, int n, int tid, int &first, int &length
) {
    int side1 = -1, side2 = -1, need1 = 0, need2 = 0;
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        int nb = neighbors[cell * 3 + s], code = board[nb * n + tid];
        if (code >= 0) {
            int need = reverse[code * 3 + sides[cell * 3 + s]];
            if (side1 < 0) { side1 = s; need1 = need; }
            else if (side2 < 0) { side2 = s; need2 = need; }
        }
    }
    if (side1 < 0) { first = 0; length = 480; }
    else if (side2 < 0) {
        int key = side1 * colors + need1;
        first = single_offsets[key]; length = single_offsets[key + 1] - first;
    } else {
        int pair = side1 == 1 ? 2 : (side2 == 1 ? 0 : 1);
        int key = (pair * colors + need1) * colors + need2;
        first = pair_offsets[key]; length = pair_offsets[key + 1] - first;
    }
}
__device__ __forceinline__ bool possible(
    const short *board, const unsigned int *used, const unsigned short *faces,
    const unsigned short *reverse, const short *neighbors, const unsigned char *sides,
    const int *single_offsets, const int *pair_offsets, const short *pool, int colors,
    int cell, int n, int tid
) {
    int first, length;
    bucket(board, reverse, neighbors, sides, single_offsets, pair_offsets, colors, cell, n, tid, first, length);
    for (int j = 0; j < length; ++j) {
        int code = pool[first + j];
        if (!taken(used, code / 3, n, tid) && matches(board, faces, reverse, neighbors, sides, cell, code, n, tid)) return true;
    }
    return false;
}
extern "C" __global__ void dfs_search(
    short *board, unsigned int *used, short *depths, short *maximum,
    const short *floors, unsigned char *states, unsigned short *firsts, unsigned short *lengths,
    unsigned short *cursors, unsigned short *shifts, unsigned int *rng,
    unsigned long long *counters, const short *order,
    const unsigned short *faces, const unsigned short *reverse,
    const short *neighbors, const unsigned char *sides,
    const int *single_offsets, const int *pair_offsets, const short *pool,
    int colors, int n, int budget
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || states[tid] != 0) return;
    int depth = depths[tid], high = maximum[tid], floor = floors[tid];
    unsigned int random = rng[tid];
    unsigned long long tried = 0, placed = 0, backed = 0;
    for (int work = 0; work < budget; ++work) {
        if (depth == 160) { states[tid] = 1; break; }
        int location = depth * n + tid;
        // 65535 marks a freshly entered stack frame.
        if (cursors[location] == 65535u) {
            int first, length;
            bucket(board, reverse, neighbors, sides, single_offsets, pair_offsets, colors,
                   order[depth], n, tid, first, length);
            firsts[location] = first; lengths[location] = length;
            shifts[location] = length ? next_random(random) % length : 0;
            cursors[location] = 0;
        }
        int length = lengths[location], cursor = cursors[location];
        if (cursor >= length) {
            // Exhaust only this job's subtree. Never visit a sibling prefix owned by another lane.
            if (depth == floor) {
                for (int c = 0; c < 160; ++c) board[c * n + tid] = -1;
                for (int word = 0; word < 5; ++word) used[word * n + tid] = 0;
                depth = 0; states[tid] = 2; ++backed;
                break;
            }
            int cell = order[depth - 1], old = board[cell * n + tid];
            board[cell * n + tid] = -1;
            used[((old / 3) >> 5) * n + tid] &= ~(1u << ((old / 3) & 31));
            --depth; ++backed;
            continue;
        }
        int code = pool[firsts[location] + (cursor + shifts[location]) % length];
        cursors[location] = cursor + 1; ++tried;
        int tile = code / 3, cell = order[depth];
        if (taken(used, tile, n, tid) || !matches(board, faces, reverse, neighbors, sides, cell, code, n, tid)) continue;
        board[cell * n + tid] = code;
        used[(tile >> 5) * n + tid] |= 1u << (tile & 31);
        bool feasible = true;
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            int nb = neighbors[cell * 3 + s];
            if (board[nb * n + tid] < 0 && !possible(board, used, faces, reverse, neighbors,
                sides, single_offsets, pair_offsets, pool, colors, nb, n, tid)) { feasible = false; break; }
        }
        if (!feasible) {
            board[cell * n + tid] = -1;
            used[(tile >> 5) * n + tid] &= ~(1u << (tile & 31));
            continue;
        }
        ++depth; ++placed;
        if (depth > high) high = depth;
        if (depth == 160) { states[tid] = 1; break; }
        cursors[depth * n + tid] = 65535u;
    }
    depths[tid] = depth; maximum[tid] = high; rng[tid] = random;
    counters[tid] += tried; counters[n + tid] += placed; counters[2 * n + tid] += backed;
}
