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
/* Immutable local orientation graph, independent of search/checkpoint state.
 * Each query scans at most 480 codes and 32768 prepared endpoint pairs. */
typedef struct {
    int codes,words;
    int32_t offsets[481];
    uint8_t *ends;
    uint64_t ports[33*8];
    int pair_count;
    uint8_t pairs[2*561];
    uint64_t pair_supports[561*8];
} LocalForests;
EXPORT void *dd_local_forests_create(int codes,int edges,const int32_t *offsets,const uint8_t *ends) {
    LocalForests *f;int pair_ids[33*33];
    if(codes<1||codes>480||edges<0||edges>32768||!offsets||(!ends&&edges)||offsets[0]!=0||offsets[codes]!=edges)return NULL;
    for(int code=0;code<codes;code++)if(offsets[code]<0||offsets[code]>offsets[code+1]||offsets[code+1]>edges)return NULL;
    for(int i=0;i<2*edges;i++)if(ends[i]>=33)return NULL;
    f=(LocalForests *)calloc(1,sizeof(LocalForests));if(!f)return NULL;
    f->ends=(uint8_t *)malloc((size_t)(edges?2*edges:1));if(!f->ends){free(f);return NULL;}
    f->codes=codes;f->words=(codes+63)/64;
    for(int i=0;i<33*33;i++)pair_ids[i]=-1;
    for(int i=0;i<=codes;i++)f->offsets[i]=offsets[i];
    for(int i=0;i<2*edges;i++)f->ends[i]=ends[i];
    for(int code=0;code<codes;code++)for(int i=offsets[code];i<offsets[code+1];i++) {
        uint64_t bit=UINT64_C(1)<<(code%64);
        f->ports[(int)ends[2*i]*f->words+code/64]|=bit;
        f->ports[(int)ends[2*i+1]*f->words+code/64]|=bit;
        int a=ends[2*i],b=ends[2*i+1],pair;
        if(a>b){int t=a;a=b;b=t;}
        pair=pair_ids[a*33+b];
        if(pair<0){pair=f->pair_count++;pair_ids[a*33+b]=pair;f->pairs[2*pair]=(uint8_t)a;f->pairs[2*pair+1]=(uint8_t)b;}
        f->pair_supports[pair*f->words+code/64]|=bit;
    }
    return f;
}
EXPORT void dd_local_forests_destroy(void *ptr) {
    LocalForests *f=(LocalForests *)ptr;if(f){free(f->ends);free(f);}
}
static int local_forest_root(int *parents,int x) {
    while(parents[x]!=x){parents[x]=parents[parents[x]];x=parents[x];}return x;
}
EXPORT int dd_local_forests_calculate(void *ptr,const uint64_t *domain,int32_t *roots,uint64_t *used_out) {
    LocalForests *f=(LocalForests *)ptr;int parents[33],ranks[33]={0},components=0;uint64_t used=0;
    if(!f||!domain||!roots||!used_out)return -2;
    if(f->codes%64 && domain[f->words-1]>>(f->codes%64))return -2;
    for(int p=0;p<33;p++) {
        parents[p]=p;
        for(int w=0;w<f->words;w++)if(domain[w]&f->ports[p*f->words+w]){used|=UINT64_C(1)<<p;components++;break;}
    }
    for(int w=0;w<f->words&&components>1;w++) {
        uint64_t bits=domain[w];
        while(bits&&components>1) {
            int code=w*64+(int)low_index(bits);bits&=bits-1;
            for(int i=f->offsets[code];i<f->offsets[code+1];i++) {
                int a=local_forest_root(parents,f->ends[2*i]),b=local_forest_root(parents,f->ends[2*i+1]);
                if(a==b)continue;
                if(ranks[a]<ranks[b]){int t=a;a=b;b=t;}
                parents[b]=a;if(ranks[a]==ranks[b])ranks[a]++;components--;
            }
        }
    }
    for(int p=0;p<33;p++)roots[p]=local_forest_root(parents,p);
    *used_out=used;return 1;
}

/* Same vertex-biconnectivity certificate as the Python local graph path:
 * preserve first-seen pair order, use the identical prefix proof and Tarjan
 * blocks, then keep bridges and one sorted-vertex cycle per cyclic block. */
EXPORT int dd_local_forests_certificate(void *ptr,const uint64_t *domain,int32_t *out,int32_t *out_count,uint64_t *used_out) {
    LocalForests *f=(LocalForests *)ptr;int32_t pairs[2*561],blocks[561];
    uint64_t used=0,covered=0,groups[561]={0};int order[561],group_count=0,m=0,count=0,ports=0,prefix,status;
    if(!f||!domain||!out||!out_count||!used_out)return -2;
    if(f->codes%64 && domain[f->words-1]>>(f->codes%64))return -2;
    for(int p=0;p<33;p++)for(int w=0;w<f->words;w++)if(domain[w]&f->ports[p*f->words+w]){used|=UINT64_C(1)<<p;ports++;break;}
    for(int pair=0;pair<f->pair_count;pair++)for(int w=0;w<f->words;w++)if(domain[w]&f->pair_supports[pair*f->words+w]) {
        pairs[2*m]=f->pairs[2*pair];pairs[2*m+1]=f->pairs[2*pair+1];m++;break;
    }
    prefix=m<2*ports?m:2*ports;
    status=dd_cyclic_blocks(33,prefix,pairs,blocks,NULL);if(status!=1)return status;
    int complete=prefix>0&&blocks[0]>=0;
    for(int i=0;i<prefix;i++) {
        if(blocks[i]!=blocks[0])complete=0;
        covered|=(UINT64_C(1)<<pairs[2*i])|(UINT64_C(1)<<pairs[2*i+1]);
    }
    if(complete&&covered==used) {
        int first=-1,previous=-1;
        for(int p=0;p<33;p++)if(used&(UINT64_C(1)<<p)) {
            if(first<0)first=p;
            if(previous>=0){out[2*count]=previous;out[2*count+1]=p;count++;}previous=p;
        }
        if(previous>=0){out[2*count]=previous;out[2*count+1]=first;count++;}
    } else {
        if(prefix!=m){status=dd_cyclic_blocks(33,m,pairs,blocks,NULL);if(status!=1)return status;}
        for(int i=0;i<m;i++) {
            int block=blocks[i];
            if(block<0){out[2*count]=pairs[2*i];out[2*count+1]=pairs[2*i+1];count++;}
            else {
                if(block>=561)return -2;
                if(!groups[block])order[group_count++]=block;
                groups[block]|=(UINT64_C(1)<<pairs[2*i])|(UINT64_C(1)<<pairs[2*i+1]);
            }
        }
        for(int i=0;i<group_count;i++) {
            int first=-1,previous=-1;uint64_t vertices=groups[order[i]];
            for(int p=0;p<33;p++)if(vertices&(UINT64_C(1)<<p)) {
                if(first<0)first=p;
                if(previous>=0){if(count>=561)return -2;out[2*count]=previous;out[2*count+1]=p;count++;}previous=p;
            }
            if(previous>=0){if(count>=561)return -2;out[2*count]=previous;out[2*count+1]=first;count++;}
        }
    }
    *used_out=used;*out_count=count;return 1;
}
/* Ordered union-by-rank roots for a bounded immutable graph. All endpoint
 * validation precedes indexing, and incomplete output is never a certificate. */
static int graph_union_root(int32_t *parents,int x) {
    while(parents[x]!=x){parents[x]=parents[parents[x]];x=parents[x];}return x;
}
EXPORT int dd_graph_union_roots(int n,int m,const int32_t *edges,int32_t *roots,dd_poll poll) {
    uint8_t *ranks;int status=1;
    if(n<0||n>5280||m<0||m>262144||(!n&&m)||(!edges&&m)||(!roots&&n))return -2;
    if(poll&&poll())return -3;
    for(int i=0;i<m;i++) {
        if((i&255)==0&&poll&&poll())return -3;
        if(edges[2*i]<0||edges[2*i]>=n||edges[2*i+1]<0||edges[2*i+1]>=n)return -2;
    }
    ranks=(uint8_t *)calloc((size_t)(n?n:1),sizeof(uint8_t));if(!ranks)return -4;
    for(int i=0;i<n;i++)roots[i]=i;
    for(int i=0;i<m;i++) {
        if((i&255)==0&&poll&&poll()){status=-3;goto done_union;}
        int a=graph_union_root(roots,edges[2*i]),b=graph_union_root(roots,edges[2*i+1]);
        if(a==b)continue;
        if(ranks[a]<ranks[b]){int t=a;a=b;b=t;}
        roots[b]=a;if(ranks[a]==ranks[b])ranks[a]++;
    }
    for(int i=0;i<n;i++) {
        if((i&255)==0&&poll&&poll()){status=-3;goto done_union;}
        roots[i]=graph_union_root(roots,i);
    }
    if(poll&&poll())status=-3;
done_union:
    free(ranks);return status;
}
/* Full optimistic strand filter. Prepared geometry is immutable; bounded scratch
 * and direct-mapped exact local forests are private to this workspace. No search
 * domains are modified, and callers discard all output on cancellation. */
#define DD_STRAND_CACHE 1024
typedef struct {
    uint64_t domain[8],used;
    int32_t roots[33];
    uint8_t valid;
} StrandForest;
typedef struct {
    LocalForests *local;
    int n,m;
    int32_t edges[4*480];
    int32_t parents[5280],hosting[160*480],seen[5280];
    uint8_t ranks[5280];
    uint64_t used[160];
    StrandForest cache[DD_STRAND_CACHE];
} StrandWorkspace;
EXPORT void *dd_strands_create(void *local,int n,int m,const int32_t *edges) {
    LocalForests *f=(LocalForests *)local;StrandWorkspace *s;
    if(!f||n<1||n>160||f->codes!=3*n||m<0||m>3*n||(!edges&&m))return NULL;
    for(int i=0;i<m;i++) {
        int a=edges[4*i],sa=edges[4*i+1],b=edges[4*i+2],sb=edges[4*i+3];
        if(a<0||a>=n||b<0||b>=n||sa<0||sa>2||sb<0||sb>2)return NULL;
    }
    s=(StrandWorkspace *)calloc(1,sizeof(StrandWorkspace));if(!s)return NULL;
    s->local=f;s->n=n;s->m=m;
    for(int i=0;i<4*m;i++)s->edges[i]=edges[i];
    return s;
}
EXPORT void dd_strands_destroy(void *ptr) {free(ptr);}
static void strand_join(StrandWorkspace *s,int left,int right) {
    int a=graph_union_root(s->parents,left),b=graph_union_root(s->parents,right);
    if(a==b)return;
    if(s->ranks[a]<s->ranks[b]){int t=a;a=b;b=t;}
    s->parents[b]=a;if(s->ranks[a]==s->ranks[b])s->ranks[a]++;
}
EXPORT int dd_strands_allowed(void *ptr,const uint64_t *domains,uint64_t *allowed,int32_t *failed,dd_poll poll) {
    StrandWorkspace *s=(StrandWorkspace *)ptr;LocalForests *f;
    int common[33],common_count=0,vertices;
    if(!s||!domains||!allowed||!failed)return -2;
    f=s->local;vertices=33*s->n;*failed=-1;
    if(poll&&poll())return -3;
    for(int cell=0;cell<s->n;cell++)
        if(f->codes%64&&domains[cell*f->words+f->words-1]>>(f->codes%64))return -2;
    for(int p=0;p<vertices;p++){s->parents[p]=p;s->ranks[p]=0;s->seen[p]=0;}
    for(int cell=0;cell<s->n;cell++) {
        const uint64_t *domain=domains+cell*f->words;uint64_t hash=UINT64_C(1469598103934665603);
        if(poll&&poll())return -3;
        for(int w=0;w<f->words;w++)hash=(hash^domain[w])*UINT64_C(1099511628211);
        StrandForest *cached=s->cache+(hash&(DD_STRAND_CACHE-1));int hit=cached->valid;
        for(int w=0;w<f->words&&hit;w++)if(cached->domain[w]!=domain[w])hit=0;
        if(!hit) {
            int32_t roots[33];uint64_t used;int status=dd_local_forests_calculate(f,domain,roots,&used);
            if(status!=1)return status;
            /* A completed local forest remains exact even if a later global
             * poll cancels; no partial calculation is inserted. */
            for(int w=0;w<f->words;w++)cached->domain[w]=domain[w];
            for(int p=0;p<33;p++)cached->roots[p]=roots[p];
            cached->used=used;cached->valid=1;
        }
        s->used[cell]=cached->used;
        for(int p=0;p<33;p++)if((cached->used>>p&1)&&cached->roots[p]!=p)
            strand_join(s,33*cell+p,33*cell+cached->roots[p]);
    }
    for(int i=0;i<s->m;i++) {
        if(poll&&poll())return -3;
        int a=s->edges[4*i],sa=s->edges[4*i+1],b=s->edges[4*i+2],sb=s->edges[4*i+3];
        for(int p=0;p<11;p++) {
            int u=11*sa+p,v=11*sb+10-p;
            if((s->used[a]>>u&1)&&(s->used[b]>>v&1))strand_join(s,33*a+u,33*b+v);
        }
    }
    for(int p=0;p<vertices;p++) {
        if((p&255)==0&&poll&&poll())return -3;
        s->parents[p]=graph_union_root(s->parents,p);
    }
    for(int cell=0;cell<s->n;cell++) {
        const uint64_t *domain=domains+cell*f->words;int only=-1,distinct=0;
        if(poll&&poll())return -3;
        for(int p=0;p<33;p++)if(s->used[cell]>>p&1) {
            int root=s->parents[33*cell+p];
            if(only<0){only=root;distinct=1;}else if(root!=only){distinct=2;break;}
        }
        for(int w=0;w<f->words;w++) {
            uint64_t bits=domain[w];
            while(bits) {
                int code=w*64+(int)low_index(bits),host=-1;bits&=bits-1;
                if((code&31)==0&&poll&&poll())return -3;
                if(distinct==1)host=only;
                else for(int i=f->offsets[code];i<f->offsets[code+1];i++) {
                    int root=s->parents[33*cell+f->ends[2*i]];
                    if(host<0)host=root;else if(host!=root){host=-1;break;}
                }
                s->hosting[cell*f->codes+code]=host;
                if(host>=0) {
                    if(cell==0&&s->seen[host]!=1) {
                        if(common_count>=33)return -2;
                        common[common_count++]=host;
                    }
                    s->seen[host]=cell+1;
                }
            }
        }
        if(cell) {
            int count=0;
            for(int i=0;i<common_count;i++)if(s->seen[common[i]]==cell+1)common[count++]=common[i];
            common_count=count;
        }
        if(!common_count){*failed=cell;if(poll&&poll())return -3;return 0;}
    }
    for(int p=0;p<vertices;p++)s->seen[p]=0;
    for(int i=0;i<common_count;i++)s->seen[common[i]]=1;
    for(int cell=0;cell<s->n;cell++) {
        if(poll&&poll())return -3;
        for(int w=0;w<f->words;w++) {
            uint64_t bits=domains[cell*f->words+w],keep=0;
            while(bits) {
                uint64_t bit=bits&(0-bits);int code=w*64+(int)low_index(bit),host=s->hosting[cell*f->codes+code];bits^=bit;
                if(host>=0&&s->seen[host])keep|=bit;
            }
            allowed[cell*f->words+w]=keep;
        }
    }
    if(poll&&poll())return -3;
    return 1;
}
