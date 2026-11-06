/* Iterative DFS control, reversible domain trail, and optional exact basic rules.
 * Python supplies stronger propagation and disjoint exhaustive branch partitions.
 * All callbacks occur at checkpoint-safe boundaries; interrupted propagation is
 * discarded before committing any domains or consuming the pending node. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#define MAX_CELLS 160
#define MAX_CODES 480
#define MAX_DEPTH 4096
#define MAX_TRAIL 4194304u
#define MAX_ARCS 960
#define STATS_COUNT 18
/* Callback process return: -1 interrupted, 0 reject, 1 branch, 2 solved. */
typedef int (*poll_fn)(void *);
typedef int (*process_fn)(void *,uint64_t *,int *,uint64_t *,int *,int *);
typedef int (*event_fn)(void *,int);
typedef struct { size_t index; uint64_t old; } Change;
typedef struct {
    size_t restore_mark,node_mark;
    int cell,count,next;
    uint64_t *choices;
    /* Private certificate only: imports never trust serialized domains. */
    int seeded;
    uint64_t dirty[3];
} Frame;
typedef struct {
    int n,count,words,max_depth,depth,status,active,basic_ready;
    size_t size,trail_count,trail_capacity;
    uint64_t *domains,*work,*choices,*filtered;
    Frame *frames;
    Change *trail;
    uint64_t stats[STATS_COUNT];
    int arc_count,*arcs,*witnesses;
    uint64_t *support;
    int incoming_offsets[MAX_CELLS+1],incoming_arcs[MAX_ARCS];
} Search;
typedef struct {
    int entries[MAX_ARCS],head,tail,count;
    unsigned char pending[MAX_ARCS];
} ArcQueue;
/* stats: nodes, branches, popped frames, written words, undone words,
 * peak trail entries, peak depth, C loop transitions; fast nodes, rejects,
 * narrowed rows, checked arcs, scanned support words, validated residues, polls;
 * enqueued arcs, reverse adjacency visits, seeded nodes. */
static int valid_domains(Search *s,const uint64_t *d) {
    int i,w; uint64_t lastmask = s->count%64 ? (UINT64_C(1)<<(s->count%64))-1 : UINT64_MAX;
    for(i=0;i<s->n;i++) {
        uint64_t any=0; for(w=0;w<s->words;w++) any|=d[(size_t)i*s->words+w];
        if(!any || (d[(size_t)i*s->words+s->words-1]&~lastmask)) return 0;
    } return 1;
}
static int reserve(Search *s,size_t extra) {
    size_t need=s->trail_count+extra,cap; Change *p;
    if(need>MAX_TRAIL) return 0;
    if(need<=s->trail_capacity) return 1;
    cap=s->trail_capacity?s->trail_capacity:1024;
    while(cap<need) { cap*=2; if(cap>MAX_TRAIL)cap=MAX_TRAIL; }
    p=(Change *)realloc(s->trail,cap*sizeof(Change)); if(!p)return 0;
    s->trail=p;s->trail_capacity=cap;return 1;
}
static void write_domain(Search *s,size_t i,uint64_t v) {
    if(s->domains[i]==v)return;
    s->trail[s->trail_count].index=i;s->trail[s->trail_count].old=s->domains[i];s->trail_count++;
    s->domains[i]=v;s->stats[3]++;
    if(s->trail_count>s->stats[5])s->stats[5]=s->trail_count;
}
static void undo(Search *s,size_t mark) {
    while(s->trail_count>mark) { Change c=s->trail[--s->trail_count];s->domains[c.index]=c.old;s->stats[4]++; }
}
static void pop(Search *s) {
    Frame *f=&s->frames[--s->depth];undo(s,f->restore_mark);free(f->choices);memset(f,0,sizeof(*f));s->stats[2]++;
}
static int partition(Search *s,const uint64_t *d,int cell,const uint64_t *choices,int count) {
    int c,w;uint64_t seen[8]={0};
    if(cell<0||cell>=s->n||count<2||count>s->count)return 0;
    for(c=0;c<count;c++) {
        uint64_t any=0;for(w=0;w<s->words;w++) {
            uint64_t v=choices[(size_t)c*s->words+w];
            if((v&~d[(size_t)cell*s->words+w])||(v&seen[w]))return 0;
            any|=v;seen[w]|=v;
        } if(!any)return 0;
    }
    for(w=0;w<s->words;w++)if(seen[w]!=d[(size_t)cell*s->words+w])return 0;
    return 1;
}
/* Residues are hints only: a cached supporting code is tested against the
 * current neighbor every time, so rollback or new checkpoints cannot falsify it. */
static int bit_index(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int result=0;while(!(value&1)) {value>>=1;result++;}return result;
#endif
}
static int has_tile(const uint64_t *d,int tile) {
    int code;for(code=tile*3;code<tile*3+3;code++)if(d[code/64]&(UINT64_C(1)<<(code%64)))return 1;
    return 0;
}
static int sparse_row(Search *s,const uint64_t *d) {
    int w,total=0;if(s->n<32)return 0;
    for(w=0;w<s->words;w++) {
#if defined(__GNUC__) || defined(__clang__)
        total+=__builtin_popcountll(d[w]);
#else
        uint64_t v=d[w];v-=((v>>1)&UINT64_C(0x5555555555555555));
        v=(v&UINT64_C(0x3333333333333333))+((v>>2)&UINT64_C(0x3333333333333333));
        v=(v+(v>>4))&UINT64_C(0x0f0f0f0f0f0f0f0f);
        total+=(int)((v*UINT64_C(0x0101010101010101))>>56);
#endif
        if(total>s->n/2)return 0;
    }return 1;
}
static int restrict_tile(Search *s,uint64_t *d,int tile,int keep) {
    int w,changed=0;uint64_t mask[8]={0};int code;
    (void)s;for(code=tile*3;code<tile*3+3;code++)mask[code/64]|=UINT64_C(1)<<(code%64);
    for(w=0;w<s->words;w++) {uint64_t value=keep?d[w]&mask[w]:d[w]&~mask[w];if(value!=d[w]) {d[w]=value;changed=1;}}
    return changed;
}
static void enqueue_arc(Search *s,ArcQueue *q,int arc) {
    if(q->pending[arc])return;
    q->pending[arc]=1;q->entries[q->tail]=arc;q->tail=(q->tail+1)%MAX_ARCS;q->count++;s->stats[15]++;
}
static void enqueue_incoming(Search *s,int cell,ArcQueue *q) {
    int index;for(index=s->incoming_offsets[cell];index<s->incoming_offsets[cell+1];index++) {
        s->stats[16]++;enqueue_arc(s,q,s->incoming_arcs[index]);
    }
}
static void mark_dirty(Frame *f,int cell) { f->dirty[cell/64]|=UINT64_C(1)<<(cell%64); }
/* Work is scratch. No domain, frame cursor, or consumed-node statistic commits
 * until propagation and the Python callback finish, preserving Stop semantics. */
static int fast_propagate(Search *s,Frame *f,poll_fn poll,void *context) {
    ArcQueue queue;int owner[MAX_CELLS],occurrence[MAX_CELLS],sole[MAX_CELLS];
    int arc,changed,cell,tile,w,steps=0;uint64_t *d=s->work;
    memset(&queue,0,sizeof(queue));s->stats[8]++;
    if(f->seeded) {
        s->stats[17]++;
        for(cell=0;cell<s->n;cell++)if(f->dirty[cell/64]&(UINT64_C(1)<<(cell%64)))enqueue_incoming(s,cell,&queue);
    } else for(arc=0;arc<s->arc_count;arc++)enqueue_arc(s,&queue,arc);
    for(;;) {
        while(queue.count) {
            arc=queue.entries[queue.head];queue.head=(queue.head+1)%MAX_ARCS;queue.count--;queue.pending[arc]=0;
            int *edge=s->arcs+arc*4;uint64_t *source=d+(size_t)edge[0]*s->words;
            const uint64_t *neighbor=d+(size_t)edge[2]*s->words;uint64_t keep[8]={0};int revised=0;
            s->stats[11]++;
            if((steps++&31)==0) {s->stats[14]++;if(poll(context))return -1;}
            for(w=0;w<s->words;w++) {
                uint64_t codes=source[w];while(codes) {
                    int bit=bit_index(codes),code=w*64+bit,residue=s->witnesses[(size_t)arc*s->count+code],found=-1,j;
                    if(residue>=0&&(neighbor[residue/64]&(UINT64_C(1)<<(residue%64)))) {found=residue;s->stats[13]++;}
                    else {
                        const uint64_t *supports=s->support+(((size_t)edge[1]*3+edge[3])*s->count+code)*s->words;
                        for(j=0;j<s->words;j++) {uint64_t present=supports[j]&neighbor[j];s->stats[12]++;if(present) {found=j*64+bit_index(present);break;}}
                        s->witnesses[(size_t)arc*s->count+code]=found;
                    }
                    if(found>=0)keep[w]|=UINT64_C(1)<<bit;
                    codes&=codes-1;
                }
                if(keep[w]!=source[w])revised=1;
            }
            if(revised) {
                uint64_t any=0;for(w=0;w<s->words;w++) {source[w]=keep[w];any|=keep[w];}
                s->stats[10]++;if(!any)return 0;enqueue_incoming(s,edge[0],&queue);
            }
        }
        memset(occurrence,0,sizeof(occurrence));for(tile=0;tile<s->n;tile++)owner[tile]=-1;
        for(cell=0;cell<s->n;cell++) {
            int num=0,last=-1;uint64_t *row=d+(size_t)cell*s->words;
            if((cell&7)==0) {s->stats[14]++;if(poll(context))return -1;}
            /* Enumerate present tile groups only. A group crossing a word
             * boundary is seen twice, but counted once via ascending last. */
            if(!sparse_row(s,row)) {
                for(tile=0;tile<s->n;tile++)if(has_tile(row,tile)) {num++;last=tile;occurrence[tile]++;sole[tile]=cell;}
            } else for(w=0;w<s->words;w++) {
                uint64_t present=row[w];while(present) {
                    int code=w*64+bit_index(present);tile=code/3;
                    if(tile!=last) {num++;last=tile;occurrence[tile]++;sole[tile]=cell;}
                    /* At the start of a word, the tile may start in the prior
                     * word; its remaining bits are positions zero/one. */
                    if(tile*3<w*64)present&=~((UINT64_C(1)<<(tile*3+3-w*64))-1);
                    else present&=~(UINT64_C(7)<<((tile*3)%64));
                }
            }
            if(!num)return 0;
            if(num==1) {if(owner[last]>=0)return 0;owner[last]=cell;}
        }
        changed=0;
        for(tile=0;tile<s->n;tile++) {
            if(!occurrence[tile])return 0;
            if(owner[tile]>=0)for(cell=0;cell<s->n;cell++)if(cell!=owner[tile]&&restrict_tile(s,d+(size_t)cell*s->words,tile,0)) {
                changed=1;s->stats[10]++;enqueue_incoming(s,cell,&queue);
            }
            if(occurrence[tile]==1&&restrict_tile(s,d+(size_t)sole[tile]*s->words,tile,1)) {
                changed=1;s->stats[10]++;enqueue_incoming(s,sole[tile],&queue);
            }
        }
        if(!changed)return 1;
    }
}
API int dd_search_configure(void *ptr,const uint64_t *support,const int *arcs,int arc_count) {
    Search *s=(Search *)ptr;uint64_t *copy;int *edges=NULL,*residues=NULL,i;size_t size,j;
    uint64_t lastmask;
    if(!s||s->active||s->count!=3*s->n||!support||arc_count<0||arc_count>MAX_ARCS||(arc_count&&!arcs))return -1;
    lastmask=s->count%64?(UINT64_C(1)<<(s->count%64))-1:UINT64_MAX;size=(size_t)9*s->count*s->words;
    for(j=s->words-1;j<size;j+=s->words)if(support[j]&~lastmask)return -1;
    for(i=0;i<arc_count;i++)if(arcs[i*4]<0||arcs[i*4]>=s->n||arcs[i*4+1]<0||arcs[i*4+1]>=3||arcs[i*4+2]<0||arcs[i*4+2]>=s->n||arcs[i*4+3]<0||arcs[i*4+3]>=3)return -1;
    copy=(uint64_t *)malloc(size*sizeof(uint64_t));if(!copy)return -2;
    if(arc_count) {
        edges=(int *)malloc((size_t)arc_count*4*sizeof(int));residues=(int *)malloc((size_t)arc_count*s->count*sizeof(int));
        if(!edges||!residues) {free(copy);free(edges);free(residues);return -2;}
        memcpy(edges,arcs,(size_t)arc_count*4*sizeof(int));for(j=0;j<(size_t)arc_count*s->count;j++)residues[j]=-1;
    }
    memcpy(copy,support,size*sizeof(uint64_t));free(s->support);free(s->arcs);free(s->witnesses);
    s->support=copy;s->arcs=edges;s->witnesses=residues;s->arc_count=arc_count;
    memset(s->incoming_offsets,0,sizeof(s->incoming_offsets));
    for(i=0;i<arc_count;i++)s->incoming_offsets[arcs[i*4+2]+1]++;
    for(i=1;i<=s->n;i++)s->incoming_offsets[i]+=s->incoming_offsets[i-1];
    {int cursor[MAX_CELLS];memcpy(cursor,s->incoming_offsets,(size_t)s->n*sizeof(int));
     for(i=0;i<arc_count;i++)s->incoming_arcs[cursor[arcs[i*4+2]]++]=i;}
    for(i=0;i<s->depth;i++)s->frames[i].seeded=0;
    s->basic_ready=0;return 0;
}
API int dd_search_abi(void) { return 1; }
API void *dd_search_create(int n,int count,int max_depth) {
    Search *s;
    if(n<1||n>MAX_CELLS||count<1||count>MAX_CODES||max_depth<1||max_depth>MAX_DEPTH)return NULL;
    s=(Search *)calloc(1,sizeof(Search));if(!s)return NULL;
    s->n=n;s->count=count;s->words=(count+63)/64;s->max_depth=max_depth;s->size=(size_t)n*s->words;
    s->domains=(uint64_t *)calloc(s->size,sizeof(uint64_t));s->work=(uint64_t *)malloc(s->size*sizeof(uint64_t));
    s->filtered=(uint64_t *)malloc(s->size*sizeof(uint64_t));
    s->choices=(uint64_t *)malloc((size_t)count*s->words*sizeof(uint64_t));s->frames=(Frame *)calloc(max_depth,sizeof(Frame));
    if(!s->domains||!s->work||!s->filtered||!s->choices||!s->frames) {free(s->domains);free(s->work);free(s->filtered);free(s->choices);free(s->frames);free(s);return NULL;}
    return s;
}
API void dd_search_destroy(void *ptr) {
    Search *s=(Search *)ptr;int i;if(!s)return;
    for(i=0;i<s->depth;i++)free(s->frames[i].choices);
    free(s->domains);free(s->work);free(s->filtered);free(s->choices);free(s->frames);free(s->trail);free(s->support);free(s->arcs);free(s->witnesses);free(s);
}
API int dd_search_add_frame(void *ptr,const uint64_t *d,int cell,const uint64_t *choices,int count,int next) {
    Search *s=(Search *)ptr;Frame *f;uint64_t *copy=NULL;size_t i;
    if(!s||!d||s->active||s->depth>=s->max_depth||!valid_domains(s,d)||cell< -1||cell>=s->n||count<0||count>s->count||next<0||next>count)return -1;
    if(cell==-1) {if(count||next)return -1;} else if(!choices||!partition(s,d,cell,choices,count))return -1;
    if(s->depth) {
        Frame *parent=&s->frames[s->depth-1];int w;
        if(parent->cell<0||parent->next<1||parent->next>parent->count)return -1;
        for(i=0;i<s->size;i++)if(d[i]&~s->domains[i])return -1;
        for(w=0;w<s->words;w++)if(d[(size_t)parent->cell*s->words+w]&~parent->choices[(size_t)(parent->next-1)*s->words+w])return -1;
        if(!reserve(s,s->size))return -2;
    }
    if(count) {copy=(uint64_t *)malloc((size_t)count*s->words*sizeof(uint64_t));if(!copy)return -2;memcpy(copy,choices,(size_t)count*s->words*sizeof(uint64_t));}
    f=&s->frames[s->depth];f->restore_mark=s->trail_count;
    if(!s->depth)memcpy(s->domains,d,s->size*sizeof(uint64_t));else for(i=0;i<s->size;i++)write_domain(s,i,d[i]);
    f->node_mark=s->trail_count;f->cell=cell;f->count=count;f->next=next;f->choices=copy;s->depth++;
    if((uint64_t)s->depth>s->stats[6])s->stats[6]=s->depth;
    s->status=0;return 0;
}
API int dd_search_depth(void *ptr) { Search *s=(Search *)ptr;return s?s->depth:-1; }
API int dd_search_basic_ready(void *ptr) { Search *s=(Search *)ptr;return s&&s->active&&s->basic_ready; }
API int dd_search_status(void *ptr) { Search *s=(Search *)ptr;return s?s->status:-1; }
API int dd_search_copy_frame(void *ptr,int index,uint64_t *domains,int *cell,uint64_t *choices,int *count,int *next) {
    Search *s=(Search *)ptr;Frame *f;size_t j;
    if(!s||index<0||index>=s->depth||!domains||!cell||!choices||!count||!next)return -1;
    f=&s->frames[index];memcpy(domains,s->domains,s->size*sizeof(uint64_t));
    for(j=s->trail_count;j>f->node_mark;j--) {Change c=s->trail[j-1];domains[c.index]=c.old;}
    *cell=f->cell;*count=f->count;*next=f->next;
    if(f->count)memcpy(choices,f->choices,(size_t)f->count*s->words*sizeof(uint64_t));
    return 0;
}
API int dd_search_decisions(void *ptr,int *cells,uint64_t *choices) {
    Search *s=(Search *)ptr;int i;if(!s||!cells||!choices)return -1;
    for(i=0;i<s->depth-1;i++) {Frame *f=&s->frames[i];if(f->cell<0||f->next<1||f->next>f->count)return -1;
        cells[i]=f->cell;memcpy(choices+(size_t)i*s->words,f->choices+(size_t)(f->next-1)*s->words,s->words*sizeof(uint64_t));}
    return s->depth?s->depth-1:0;
}
API int dd_search_stats(void *ptr,uint64_t *stats) {Search *s=(Search *)ptr;if(!s||!stats)return -1;memcpy(stats,s->stats,8*sizeof(uint64_t));return 0;}
API int dd_search_fast_stats(void *ptr,uint64_t *stats) {Search *s=(Search *)ptr;if(!s||!stats)return -1;memcpy(stats,s->stats+8,7*sizeof(uint64_t));return 0;}
API int dd_search_queue_stats(void *ptr,uint64_t *stats) {Search *s=(Search *)ptr;if(!s||!stats)return -1;memcpy(stats,s->stats+15,3*sizeof(uint64_t));return 0;}
API int dd_search_run(void *ptr,process_fn process,poll_fn poll,event_fn event,void *context,uint64_t max_steps) {
    Search *s=(Search *)ptr;uint64_t steps=0;int rc=0;
    if(!s||!process||!poll||!event||s->active)return -1;
    if(s->status==1||s->status==2)return s->status;
    s->active=1;s->status=0;
    while(s->depth) {
        Frame *f=&s->frames[s->depth-1];
        if((max_steps&&steps>=max_steps)||poll(context)) {s->status=3;break;}
        if(f->cell<0) {
            int cell=-1,count=0,jump=-1,action,fast=1;uint64_t *copy=NULL;size_t i;
            memcpy(s->work,s->domains,s->size*sizeof(uint64_t));
            if(s->support)fast=fast_propagate(s,f,poll,context);
            if(fast<0) {s->status=3;break;}
            s->basic_ready=s->support&&fast;
            if(s->basic_ready)memcpy(s->filtered,s->work,s->size*sizeof(uint64_t));
            action=fast?process(context,s->work,&cell,s->choices,&count,&jump):0;
            s->basic_ready=0;
            if(action<0) {s->status=3;break;}
            if(action>2||jump< -1||jump>s->depth-1) {rc=-3;break;}
            if(action==1) {
                if(jump!=-1||!valid_domains(s,s->work)||!partition(s,s->work,cell,s->choices,count)){rc=-3;break;}
                for(i=0;i<s->size;i++)if(s->work[i]&~s->domains[i]){rc=-3;break;}
                if(rc)break;
                copy=(uint64_t *)malloc((size_t)count*s->words*sizeof(uint64_t));
                if(!copy||!reserve(s,s->size)) {free(copy);s->status=4;break;}
                memcpy(copy,s->choices,(size_t)count*s->words*sizeof(uint64_t));
                for(i=0;i<s->size;i++)write_domain(s,i,s->work[i]);
                f->node_mark=s->trail_count;f->cell=cell;f->count=count;f->next=0;f->choices=copy;
                f->seeded=s->support!=NULL;memset(f->dirty,0,sizeof(f->dirty));
                if(f->seeded)for(i=0;i<s->size;i++)if(s->work[i]!=s->filtered[i])mark_dirty(f,(int)(i/s->words));
            } else {
                if(action==2&&jump!=-1){rc=-3;break;}
                pop(s);if(action==0&&jump>=0)while(s->depth>jump)pop(s);
                if(action==2)s->status=2;else if(!s->depth)s->status=1;
            }
            s->stats[0]++;if(!fast)s->stats[9]++;steps++;s->stats[7]++;
            if(event(context,!fast?4:action==1?2:action==2?3:0)) {if(!s->status)s->status=3;break;}
            if(s->status)break;
        } else if(f->next>=f->count) {
            pop(s);steps++;s->stats[7]++;if(!s->depth)s->status=1;
            if(event(context,1)){if(!s->status)s->status=3;break;}
            if(s->status)break;
        } else {
            int w;Frame *child;size_t mark=s->trail_count;
            if(s->depth>=s->max_depth||!reserve(s,s->words)){s->status=4;break;}
            for(w=0;w<s->words;w++)write_domain(s,(size_t)f->cell*s->words+w,f->choices[(size_t)f->next*s->words+w]);
            f->next++;child=&s->frames[s->depth++];memset(child,0,sizeof(*child));
            child->restore_mark=mark;child->node_mark=s->trail_count;child->cell=-1;
            child->seeded=f->seeded;memcpy(child->dirty,f->dirty,sizeof(child->dirty));mark_dirty(child,f->cell);
            s->stats[1]++;if((uint64_t)s->depth>s->stats[6])s->stats[6]=s->depth;
            steps++;s->stats[7]++;
        }
    }
    if(!rc&&!s->depth&&!s->status)s->status=1;
    s->active=0;return rc?rc:s->status;
}
