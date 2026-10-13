/* Exact seam arc consistency over up to 160 triangular cells.
 * All physical-tile exclusions are already present in the immutable support
 * table. Native batches are bounded; Python commits domains before polling.
 */
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
static unsigned low_index(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctzll(value);
#else
    unsigned result=0; while ((value&1)==0) {value>>=1; ++result;} return result;
#endif
}
EXPORT int dd_native_abi(void) { return 1; }
/* state=head,tail,queued; counts=processed_arcs,revised_arcs. */
static int arc_batch(uint64_t *domains, const uint64_t *supports,
    const int32_t *arcs, const int32_t *incoming, int n, int codes,
    int words, int arc_count, int32_t *queue, uint8_t *pending,
    int32_t *state, int budget, uint64_t *counts, int16_t *residues,
    uint8_t *dirty, int32_t *event_arcs, uint64_t *event_domains,
    int event_capacity, int32_t *event_count) {
    if (n<1 || n>160 || codes!=3*n || words!=(codes+63)/64 ||
        arc_count<0 || arc_count>3*n || budget<1) return -2;
    if (event_arcs && (!event_domains || !event_count || event_capacity<budget)) return -2;
    if (event_count) *event_count=0;
    counts[0]=counts[1]=0;
    if (residues || dirty) counts[2]=counts[3]=0;
    if (!state[2]) return 1;
    while (state[2] && budget--) {
        int index=queue[state[0]];
        state[0]=(state[0]+1)%arc_count; --state[2]; pending[index]=0;
        int a=arcs[4*index], sa=arcs[4*index+1];
        int b=arcs[4*index+2], sb=arcs[4*index+3];
        if (a<0 || b<0 || a>=n || b>=n || sa<0 || sb<0 || sa>2 || sb>2 || a==b) return -2;
        int changed=0; uint64_t any=0; ++counts[0];
        for (int w=0;w<words;++w) {
            uint64_t remaining=domains[a*words+w], keep=0;
            while (remaining) {
                uint64_t low=remaining & (0-remaining);
                int code=64*w+(int)low_index(low); remaining^=low;
                if (code>=codes) return -2;
                const uint64_t *allowed=supports+((sa*3+sb)*codes+code)*words;
                /* A remembered support is only a hint: revalidate it after
                 * narrowing, backtracking, cancellation, or a different job. */
                int slot=index*codes+code;
                int previous=residues ? residues[slot] : -1;
                if (previous>=0 && previous<codes) {
                    uint64_t bit=UINT64_C(1)<<(previous%64);
                    if (allowed[previous/64]&domains[b*words+previous/64]&bit) {
                        keep|=low; ++counts[2]; continue;
                    }
                }
                if (residues) ++counts[3];
                for (int j=0;j<words;++j) {
                    uint64_t support=allowed[j]&domains[b*words+j];
                    if (support) {
                        keep|=low;
                        if (residues) residues[slot]=(int16_t)(64*j+low_index(support));
                        break;
                    }
                }
            }
            changed|=(keep!=domains[a*words+w]);domains[a*words+w]=keep;any|=keep;
        }
        if (changed) {
            ++counts[1];
            if (dirty) dirty[a]=1;
            /* Optional bounded trace: at most one revision per processed arc.
             * Record the empty-domain event before returning a contradiction. */
            if (event_arcs) {
                int event=(*event_count)++;
                event_arcs[event]=index;
                for (int w=0;w<words;++w) event_domains[event*words+w]=domains[a*words+w];
            }
            if (!any) return -1;
            for (int j=0;j<3;++j) {
                int next=incoming[3*a+j];
                if (next>=0 && !pending[next]) {
                    if (state[2]>=arc_count) return -2;
                    queue[state[1]]=next;state[1]=(state[1]+1)%arc_count;
                    ++state[2];pending[next]=1;
                }
            }
        }
    }
    return state[2] ? 0 : 1;
}

/* Original ABI remains available for finite baseline comparisons. */
EXPORT int dd_arc_batch(uint64_t *domains, const uint64_t *supports,
    const int32_t *arcs, const int32_t *incoming, int n, int codes,
    int words, int arc_count, int32_t *queue, uint8_t *pending,
    int32_t *state, int budget, uint64_t *counts) {
    return arc_batch(domains,supports,arcs,incoming,n,codes,words,arc_count,
                     queue,pending,state,budget,counts,NULL,NULL,NULL,NULL,0,NULL);
}
EXPORT int dd_arc_batch_cached(uint64_t *domains, const uint64_t *supports,
    const int32_t *arcs, const int32_t *incoming, int n, int codes,
    int words, int arc_count, int32_t *queue, uint8_t *pending,
    int32_t *state, int budget, uint64_t *counts, int16_t *residues) {
    if (!residues) return -2;
    return arc_batch(domains,supports,arcs,incoming,n,codes,words,arc_count,
                     queue,pending,state,budget,counts,residues,NULL,NULL,NULL,0,NULL);
}


/* Reusable Python-owned workspace. Dirty rows are output-only for this batch;
 * trace storage is optional and is never touched on the ordinary path. */
EXPORT int dd_arc_batch_workspace(uint64_t *domains, const uint64_t *supports,
    const int32_t *arcs, const int32_t *incoming, int n, int codes,
    int words, int arc_count, int32_t *queue, uint8_t *pending,
    int32_t *state, int budget, uint64_t *counts, int16_t *residues,
    uint8_t *dirty, int32_t *event_arcs, uint64_t *event_domains,
    int event_capacity, int32_t *event_count) {
    if (!dirty || n<1 || n>160) return -2;
    for (int i=0;i<n;++i) dirty[i]=0;
    return arc_batch(domains,supports,arcs,incoming,n,codes,words,arc_count,
                    queue,pending,state,budget,counts,residues,dirty,
                    event_arcs,event_domains,event_capacity,event_count);
}


/* Iterative Tarjan edge blocks for the optimistic Gold port graph.
 * No recursive C stack and no domain writes. Polling is cooperative every256
 * graph steps; cancellation discards this incomplete certificate entirely. */
#include <stdlib.h>
typedef int (*dd_poll)(void);
EXPORT int dd_cyclic_blocks(int n, int m, const int32_t *edges,
                           int32_t *result, dd_poll poll) {
    if (n<0 || n>5280 || m<0 || m>262144 || (!n && m)) return -2;
    if (poll && poll()) return -3;
    int32_t *memory=(int32_t*)calloc((size_t)(7*n+1+3*m),sizeof(int32_t));
    if (!memory) return -4;
    int32_t *offset=memory,*cursor=offset+n+1,*discovered=cursor+n;
    int32_t *low=discovered+n,*parent=low+n,*stack=parent+n;
    int32_t *marks=stack+n,*adj=marks+n,*pending=adj+2*m;
    int status=1,clock=0,block_id=0,stamp=0,pending_count=0,steps=0;
    for (int eid=0;eid<m;++eid) {
        if ((eid&255)==0 && poll && poll()) {status=-3;goto done;}
        int a=edges[2*eid],b=edges[2*eid+1];
        if (a<0 || b<0 || a>=n || b>=n) {status=-2;goto done;}
        ++offset[a+1];++offset[b+1];result[eid]=-1;
    }
    for (int v=0;v<n;++v) {offset[v+1]+=offset[v];cursor[v]=offset[v];discovered[v]=parent[v]=-1;}
    for (int eid=0;eid<m;++eid) {
        adj[cursor[edges[2*eid]]++]=eid;adj[cursor[edges[2*eid+1]]++]=eid;
    }
    for (int v=0;v<n;++v) cursor[v]=offset[v];
    for (int root=0;root<n;++root) {
        if (discovered[root]>=0 || offset[root]==offset[root+1]) continue;
        discovered[root]=low[root]=clock++;int height=0;stack[height++]=root;
        while (height) {
            if ((++steps&255)==0 && poll && poll()) {status=-3;goto done;}
            int v=stack[height-1],parent_edge=parent[v];
            if (cursor[v]<offset[v+1]) {
                int eid=adj[cursor[v]++];
                if (eid==parent_edge) continue;
                int other=edges[2*eid]^edges[2*eid+1]^v;
                if (discovered[other]<0) {
                    pending[pending_count++]=eid;parent[other]=eid;
                    discovered[other]=low[other]=clock++;stack[height++]=other;
                } else if (discovered[other]<discovered[v]) {
                    pending[pending_count++]=eid;
                    if (discovered[other]<low[v]) low[v]=discovered[other];
                }
                continue;
            }
            --height;
            if (parent_edge>=0) {
                int above=edges[2*parent_edge]^edges[2*parent_edge+1]^v;
                if (low[v]<low[above]) low[above]=low[v];
                if (low[v]>=discovered[above]) {
                    int end=pending_count,vertices=0,eid; ++stamp;
                    do {
                        if (!pending_count) {status=-2;goto done;}
                        eid=pending[--pending_count];
                        int a=edges[2*eid],b=edges[2*eid+1];
                        if (marks[a]!=stamp) {marks[a]=stamp;++vertices;}
                        if (marks[b]!=stamp) {marks[b]=stamp;++vertices;}
                    } while (eid!=parent_edge);
                    if (end-pending_count>=vertices) {
                        for (int i=pending_count;i<end;++i) result[pending[i]]=block_id;
                        ++block_id;
                    }
                }
            }
        }
    }
    if (poll && poll()) status=-3;
done:
    free(memory);return status;
}


/* Exact AllDifferent supports on a balanced cell/physical-tile graph.
 * A checked perfect matching contracts tile->cell edges. A remaining edge
 * belongs to some perfect matching iff its endpoints share a directed SCC.
 * Bitset transitive closure computes that relation for at most160 cells. */
EXPORT int dd_hall_allowed(int n,int words,const uint64_t *rows,uint64_t *allowed,
                          int32_t *matching,int32_t *failed,dd_poll poll) {
    if (n<1 || n>160 || words!=(n+63)/64) return -2;
    int32_t owner[160],cells[160],degree[160],order[160],parents[160],queue[160];
    uint8_t seen[160];uint64_t reach[160*3];
    if (poll && poll()) return -3;
    for (int i=0;i<n;++i) {owner[i]=cells[i]=-1;order[i]=i;degree[i]=0;}
    for (int i=0;i<n;++i) {
        if (n%64 && (rows[i*words+words-1]>>(n%64))) return -2;
        int tile=matching[i];
        if (tile>=0 && tile<n && owner[tile]<0 && (rows[i*words+tile/64]&(UINT64_C(1)<<(tile%64)))) {
            cells[i]=tile;owner[tile]=i;
        }
        for (int w=0;w<words;++w) {uint64_t bits=rows[i*words+w];while(bits) {++degree[i];bits&=bits-1;}}
        int pos=i;
        while (pos>0 && degree[order[pos-1]]>degree[i]) {order[pos]=order[pos-1];--pos;}
        order[pos]=i;
    }
    for (int index=0;index<n;++index) {
        if (poll && poll()) return -3;
        int start=order[index];if(cells[start]>=0) continue;
        for(int i=0;i<n;++i) {parents[i]=-1;seen[i]=0;}
        int head=0,tail=0,free_tile=-1;queue[tail++]=start;seen[start]=1;
        while(head<tail && free_tile<0) {
            int cell=queue[head++];
            for(int w=0;w<words && free_tile<0;++w) {
                uint64_t bits=rows[cell*words+w];
                while(bits) {
                    int tile=64*w+(int)low_index(bits);bits&=bits-1;
                    if(tile>=n) return -2;
                    if(parents[tile]>=0) continue;
                    parents[tile]=cell;int other=owner[tile];
                    if(other<0) {free_tile=tile;break;}
                    if(!seen[other]) {seen[other]=1;queue[tail++]=other;}
                }
            }
        }
        if(free_tile<0) {*failed=start;return 0;}
        while(free_tile>=0) {
            int cell=parents[free_tile],previous=cells[cell];
            cells[cell]=free_tile;owner[free_tile]=cell;free_tile=previous;
        }
    }
    for(int i=0;i<n*words;++i) reach[i]=0;
    for(int cell=0;cell<n;++cell) for(int w=0;w<words;++w) {
        uint64_t bits=rows[cell*words+w];
        while(bits) {int tile=64*w+(int)low_index(bits);bits&=bits-1;int other=owner[tile];
            reach[cell*words+other/64]|=UINT64_C(1)<<(other%64);}
    }
    for(int middle=0;middle<n;++middle) {
        if((middle&15)==0 && poll && poll()) return -3;
        uint64_t bit=UINT64_C(1)<<(middle%64);
        for(int cell=0;cell<n;++cell) if(reach[cell*words+middle/64]&bit)
            for(int w=0;w<words;++w) reach[cell*words+w]|=reach[middle*words+w];
    }
    for(int cell=0;cell<n;++cell) {
        matching[cell]=cells[cell];
        for(int w=0;w<words;++w) {
            uint64_t bits=rows[cell*words+w],keep=0;
            while(bits) {uint64_t bit=bits&(0-bits);int tile=64*w+(int)low_index(bit);bits^=bit;
                if(reach[owner[tile]*words+cell/64]&(UINT64_C(1)<<(cell%64))) keep|=bit;}
            allowed[cell*words+w]=keep;
        }
    }
    if(poll && poll()) return -3;
    return 1;
}
