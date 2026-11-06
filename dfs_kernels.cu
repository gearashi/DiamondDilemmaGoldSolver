// Constructive exact-edge DFS: persistent private stack per CUDA thread.
__device__ __forceinline__ unsigned int next_random(unsigned int &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
__device__ __forceinline__ bool taken(const unsigned int *used, int tile, int n, int tid) {
    return (used[(tile >> 5) * n + tid] & (1u << (tile & 31))) != 0;
}
// The candidate came from bucket(), which covers the first two occupied sides.
// With three sides total, only an occupied third side can remain unchecked.
__device__ __forceinline__ bool matches_after_bucket(
    const short *board, const unsigned short *faces, const unsigned short *reverse,
    const short *neighbors, const unsigned char *sides, int cell, int code, int n, int tid
) {
    if (board[neighbors[cell * 3] * n + tid] < 0 ||
        board[neighbors[cell * 3 + 1] * n + tid] < 0) return true;
    int other = board[neighbors[cell * 3 + 2] * n + tid];
    return other < 0 || faces[code * 3 + 2] == reverse[other * 3 + sides[cell * 3 + 2]];
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
        if (!taken(used, code / 3, n, tid) && matches_after_bucket(board, faces, reverse, neighbors, sides, cell, code, n, tid)) return true;
    }
    return false;
}

// Optional necessary deductions. Every local domain is enumerated from the
// current assigned seams and unused physical pieces; no sampling is a proof.
struct Reasoning {
    const short *board; const unsigned int *used;
    const unsigned short *faces; const unsigned short *reverse;
    const short *neighbors; const unsigned char *sides;
    const int *single_offsets; const int *pair_offsets; const short *pool;
    const unsigned char *partners; const unsigned int *policy;
    int colors, n, tid, flags;
    __device__ bool allowed(int cell, int code) const {
        return !(flags & 32) || code / 3 != 0 || policy[cell];
    }
    __device__ int choices(int cell, int limit, int &singleton_index, int *pieces=nullptr) const {
        int first, length, count=0, unique=0;
        bucket(board, reverse, neighbors, sides, single_offsets, pair_offsets, colors, cell,n,tid,first,length);
        for(int j=0;j<length;++j) {
            int code=pool[first+j];
            if(taken(used,code/3,n,tid)||!allowed(cell,code)||!matches_after_bucket(board,faces,reverse,neighbors,sides,cell,code,n,tid)) continue;
            singleton_index=j; ++count;
            if(pieces) {
                int tile=code/3; bool known=false;
                for(int k=0;k<unique;++k) if(pieces[k]==tile) known=true;
                if(!known) { if(unique==3) return 4; pieces[unique++]=tile; }
            } else if(count>=limit) return count;
        }
        return pieces ? unique : count;
    }
    __device__ bool forward() const {
        for(int cell=0;cell<160;++cell) if(board[cell*n+tid]<0) {
            int ignored=0; if(!choices(cell,1,ignored)) return false;
        }
        return true;
    }
    __device__ bool hall() const {
        // Any inspected Hall shortage is conclusive. A bounded sample of small
        // domains may miss other shortages, but never rejects a supported one.
        int domains[24][3], sizes[24], count=0;
        for(int cell=0;cell<160;++cell) if(board[cell*n+tid]<0) {
            int ignored=0,pieces[3]; int size=choices(cell,480,ignored,pieces);
            if(!size) return false;
            if(size<=3 && count<24) { sizes[count]=size; for(int j=0;j<size;++j) domains[count][j]=pieces[j]; ++count; }
        }
        for(int a=0;a<count;++a) for(int b=a+1;b<count;++b) {
            int values[6], size=0;
            for(int which=0;which<2;++which) {int row=which?b:a;
                for(int k=0;k<sizes[row];++k) {int value=domains[row][k];bool found=false;
                    for(int j=0;j<size;++j) if(values[j]==value) found=true;
                    if(!found) values[size++]=value;}}
            if(size<2) return false;
            if(size>2) continue;
            for(int c=b+1;c<count;++c) {
                int total=size;
                for(int k=0;k<sizes[c];++k) {int value=domains[c][k];bool found=false;
                    for(int j=0;j<size;++j) if(values[j]==value) found=true;
                    if(!found) ++total;}
                if(total<3) return false;
            }
        }
        return true;
    }
    __device__ int choose(const short *order, int &one_index, bool &forced) const {
        int best=-1,minimum=481; forced=false;
        for(int rank=0;rank<160;++rank) {
            int cell=order[rank]; if(board[cell*n+tid]>=0) continue;
            if(best<0) best=cell;
            int index=0, count=choices(cell,minimum,index);
            if(count<minimum) {if((flags&8)||count<=1) best=cell; minimum=count;one_index=index;}
            if(count==0) {forced=false;return cell;}
            if(count==1 && (flags&4)) {forced=true;return cell;}
        }
        return best;
    }
    __device__ bool nonempty(int code,int side) const {
        for(int port=side*11;port<(side+1)*11;++port) if(partners[code*33+port]<33) return true;
        return false;
    }
    __device__ bool connected() const {
        int start=-1;
        for(int c=0;c<160 && start<0;++c) if(board[c*n+tid]>=0)
            for(int side=0;side<3;++side) if(nonempty(board[c*n+tid],side)) {start=c;break;}
        if(start<0) return true;
        // Merge all strands within a cell optimistically, and leave every
        // unknown/unknown seam open. Disconnection still proves impossibility.
        unsigned int seen[5]={0,0,0,0,0}; int queue[160],read=0,write=1;queue[0]=start;
        seen[start>>5]|=1u<<(start&31);
        while(read<write) {int cell=queue[read++],code=board[cell*n+tid];
            for(int side=0;side<3;++side) {int nb=neighbors[cell*3+side],other=board[nb*n+tid];
                if(seen[nb>>5]&(1u<<(nb&31))) continue;
                if(code>=0&&!nonempty(code,side)) continue;
                if(other>=0&&!nonempty(other,sides[cell*3+side])) continue;
                seen[nb>>5]|=1u<<(nb&31);queue[write++]=nb;}}
        for(int cell=0;cell<160;++cell) if(board[cell*n+tid]>=0 && !(seen[cell>>5]&(1u<<(cell&31))))
            for(int side=0;side<3;++side) if(nonempty(board[cell*n+tid],side)) return false;
        return true;
    }
    __device__ bool shared() const {
        for(unsigned i=0;i<policy[160];++i) {
            int offset=161+i*321; bool all=true;
            for(unsigned j=0;j<policy[offset];++j) if(board[policy[offset+1+2*j]*n+tid] != int(policy[offset+2+2*j])) {all=false;break;}
            if(all) return false;
        }
        return true;
    }
};
__device__ __forceinline__ int selected_cell(int depth,int floor,const short*order,
    const unsigned short*firsts,const unsigned short*shifts,int n,int tid,int flags) {
    if(!(flags&12)||depth<floor) return order[depth];
    int at=depth*n+tid;
    return (firsts[at]>>12)|((shifts[at]>>12)<<4);
}

// A degree-two path cannot leave a closed component. Trace only segments
// touching the newly assigned cell; return the latest cell rank in a proper
// cycle, or -1. No connectivity state survives a rollback or checkpoint.
__device__ int gold_closed_rank(const short *board, const unsigned char *partners,
    const short *neighbors, const unsigned char *sides, const short *ranks,
    int start, int n, int tid, int total) {
    int start_code = board[start * n + tid];
    for (int port = 0; port < 33; ++port) {
        int partner = partners[start_code * 33 + port];
        if (partner >= 33 || partner < port) continue;
        int cell = start, incoming = port, latest = ranks[start];
        for (int count = 1; count <= total; ++count) {
            int code = board[cell * n + tid];
            int outgoing = partners[code * 33 + incoming];
            if (outgoing >= 33) break;
            latest = max(latest, (int)ranks[cell]);
            int side = outgoing / 11;
            incoming = sides[cell * 3 + side] * 11 + 10 - outgoing % 11;
            cell = neighbors[cell * 3 + side];
            if (board[cell * n + tid] < 0) break;
            if (cell == start && incoming == port) {
                if (count < total) return latest;
                break;
            }
        }
    }
    return -1;
}
extern "C" __global__ void dfs_search(
    short *board, unsigned int *used, short *depths, short *maximum,
    const short *floors, unsigned char *states, unsigned short *firsts, unsigned short *lengths,
    unsigned short *cursors, unsigned short *shifts, unsigned int *rng,
    unsigned long long *counters, const short *order,
    const unsigned short *faces, const unsigned short *reverse,
    const short *neighbors, const unsigned char *sides,
    const int *single_offsets, const int *pair_offsets, const short *pool,
    const unsigned char *gold_partners, const short *gold_ranks, unsigned char *gold_checked,
    const unsigned int *policy, int flags,
    int gold_total, int colors, int n, int budget
) {
    flags = REASONING_FLAGS;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (states[tid] != 0 && !(gold_total && states[tid] == 1 && gold_checked[tid] != 255)) return;
    int depth = depths[tid], high = maximum[tid], floor = floors[tid];
    unsigned int random = rng[tid];
    Reasoning reasoning={board,used,faces,reverse,neighbors,sides,single_offsets,pair_offsets,pool,gold_partners,policy,colors,n,tid,flags};
    short private_ranks[160]; const short *ranks=gold_ranks;
    if(flags&12) {ranks=private_ranks;for(int i=0;i<depth;++i) private_ranks[selected_cell(i,floor,order,firsts,shifts,n,tid,flags)]=i;}
    bool initial_ok=true;
    if(flags&32) for(int i=0;i<depth;++i) {int cell=selected_cell(i,floor,order,firsts,shifts,n,tid,flags);
        if(!reasoning.allowed(cell,board[cell*n+tid])) {initial_ok=false;break;}}
    if(!initial_ok) { // Canonical prefix restriction, selected only for fresh searches.
        for(int c=0;c<160;++c) board[c*n+tid]=-1;
        for(int word=0;word<5;++word) used[word*n+tid]=0;
        depths[tid]=0;states[tid]=2;return;
    }
    unsigned long long tried = 0, placed = 0, backed = 0;
    // Legacy checkpoints may contain a previously unexamined proper loop.
    // Undo only its proven-impossible subtree; consumed ancestor cursors remain.
    int work = 0;
    if (gold_total && gold_checked[tid] != 255) {
        while (work < budget && gold_checked[tid] != 255) {
            int scan = gold_checked[tid];
            if (scan >= depth) { gold_checked[tid] = 255; break; }
            int closed = gold_closed_rank(board, gold_partners, neighbors, sides, ranks,
                                          selected_cell(scan,floor,order,firsts,shifts,n,tid,flags), n, tid, gold_total);
            ++work;
            if (closed < 0) {
                gold_checked[tid] = scan + 1 >= depth ? 255 : scan + 1;
                continue;
            }
            if (closed < floor) {
                for (int c = 0; c < 160; ++c) board[c * n + tid] = -1;
                for (int word = 0; word < 5; ++word) used[word * n + tid] = 0;
                depth = 0; states[tid] = 2; ++backed; break;
            }
            while (depth > closed) {
                int cell = selected_cell(--depth,floor,order,firsts,shifts,n,tid,flags), old = board[cell * n + tid];
                board[cell * n + tid] = -1;
                used[((old / 3) >> 5) * n + tid] &= ~(1u << ((old / 3) & 31));
                ++backed;
            }
            states[tid] = 0;
            gold_checked[tid] = 0;
        }
    }
    for (; work < budget && states[tid] == 0 && (!gold_total || gold_checked[tid] == 255); ++work) {
        if (depth == 160) { states[tid] = 1; break; }
        int location = depth * n + tid;
        // 65535 marks a freshly entered stack frame.
        if (cursors[location] == 65535u) {
            int first, length, cell=order[depth], one_index=0;bool forced=false;
            if(flags&12) cell=reasoning.choose(order,one_index,forced);
            bucket(board, reverse, neighbors, sides, single_offsets, pair_offsets, colors,
                   cell, n, tid, first, length);
            int shift=forced?one_index:(length ? next_random(random) % length : 0);
            firsts[location] = first | ((flags&12)?((cell&15)<<12):0);
            lengths[location] = length | (forced?32768:0);
            shifts[location] = shift | ((flags&12)?((cell>>4)<<12):0);
            cursors[location] = 0;
        }
        int length = lengths[location]&32767, cursor = cursors[location];
        if (cursor >= ((lengths[location]&32768)?1:length)) {
            // Exhaust only this job's subtree. Never visit a sibling prefix owned by another lane.
            if (depth == floor) {
                for (int c = 0; c < 160; ++c) board[c * n + tid] = -1;
                for (int word = 0; word < 5; ++word) used[word * n + tid] = 0;
                depth = 0; states[tid] = 2; ++backed;
                break;
            }
            int cell = selected_cell(depth-1,floor,order,firsts,shifts,n,tid,flags), old = board[cell * n + tid];
            board[cell * n + tid] = -1;
            used[((old / 3) >> 5) * n + tid] &= ~(1u << ((old / 3) & 31));
            --depth; ++backed;
            continue;
        }
        int code = pool[(firsts[location]&4095) + (cursor + (shifts[location]&4095)) % length];
        cursors[location] = cursor + 1; ++tried;
        int tile = code / 3, cell = selected_cell(depth,floor,order,firsts,shifts,n,tid,flags);
        if (!reasoning.allowed(cell,code) || taken(used, tile, n, tid) || !matches_after_bucket(board, faces, reverse, neighbors, sides, cell, code, n, tid)) continue;
        board[cell * n + tid] = code;
        used[(tile >> 5) * n + tid] |= 1u << (tile & 31);
        bool feasible = !(flags&192) || reasoning.shared();
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (!feasible) break;
            int nb = neighbors[cell * 3 + s];
            if (board[nb * n + tid] < 0 && !possible(board, used, faces, reverse, neighbors,
                sides, single_offsets, pair_offsets, pool, colors, nb, n, tid)) { feasible = false; break; }
        }
        if(feasible && (flags&1) && !reasoning.forward()) feasible=false;
        if(feasible && (flags&2) && !reasoning.hall()) feasible=false;
        if(flags&12) private_ranks[cell]=depth;
        if (feasible && gold_total && gold_closed_rank(board, gold_partners, neighbors, sides,
                ranks, cell, n, tid, gold_total) >= 0) feasible = false;
        if(feasible && gold_total && (flags&16) && !reasoning.connected()) feasible=false;
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
