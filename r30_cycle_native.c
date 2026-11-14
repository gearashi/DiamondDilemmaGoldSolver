/* Join complete directed physical cycles. For a fixed pair of cycles, a
   remote single-star delta is exact. Nearby inputs can affect only shared
   star factors, whose remaining gain is bounded by each factor's degree. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

#define MAXN 240
typedef struct { unsigned char c[4],k; short gain; } Cycle;
typedef struct { int count; Cycle *cycles; } List;
typedef struct {
    int n,target,base_sum;
    const int *nb,*delta;
    const unsigned char *masks,*base,*free_cell;
    int *reach;
    const short *max_extra; int max_gain;
    List swaps[2*MAXN],triangles[2*MAXN];
} Model;
enum {STREAM,LOOKUP,TRIED,BOUNDED,COUNT};
API int cycle_abi(void) {return 1;}

static int star_value(const Model *q,int c,const int *source) {
    const unsigned char *p=q->masks+source[c]*3*(q->n+1);
    unsigned a=p[source[q->nb[3*c]]], b=p[q->n+1+source[q->nb[3*c+1]]],
             d=p[2*(q->n+1)+source[q->nb[3*c+2]]];
    return (a&b&d)?3:((a&b)|(a&d)|(b&d))?2:(a|b|d)?1:0;
}
static int donor(const Cycle *p,int i){return p->c[(i+1)%p->k];}
/* Each anchored catalogue is sorted by standalone doubled-star gain. */
static int order_gain(const void *left,const void *right) {
    const Cycle *a=left,*b=right;
    if(a->gain!=b->gain)return b->gain-a->gain;
    return memcmp(a->c,b->c,a->k);
}

API void cycle_free(Model *q){
    if(!q)return;
    for(int i=0;i<2*q->n;i++){free(q->swaps[i].cycles);free(q->triangles[i].cycles);}
    free(q->reach);free(q);
}
API Model *cycle_new(int n,int target,const int *nb,const int *delta,
        const unsigned char *masks,const unsigned char *base,const unsigned char *free_cell,const short *max_extra,int right) {
    if(n<2||n>MAXN||(right!=2&&right!=3))return NULL;
    Model *q=calloc(1,sizeof(Model));if(!q)return NULL;
    q->n=n;q->target=2*target;q->nb=nb;q->delta=delta;q->masks=masks;q->base=base;q->free_cell=free_cell;q->max_extra=max_extra;q->max_gain=-100;
    q->reach=calloc((size_t)n*n,sizeof(int));
    if(!q->reach)goto fail;
    for(int a=0;a<n;a++){
        q->base_sum+=base[a];
        q->reach[a*n+a]=1;
        for(int i=0;i<3;i++){
            int b=nb[3*a+i];if(b>=n)continue;
            q->reach[a*n+b]=1;
            for(int j=0;j<3;j++){int c=nb[3*b+j];if(c<n)q->reach[a*n+c]=1;}
        }
    }
    for(int a=0;a<n;a++)if(free_cell[a]){
        List *v=(right==2?q->swaps:q->triangles)+a;
        int cap=right==2?n:n*n;
        v->cycles=malloc((size_t)cap*sizeof(Cycle));if(!v->cycles)goto fail;
        for(int b=0;b<n;b++)if(a!=b&&free_cell[b]){
            if(right==2) {
                Cycle r={{a,b,0,0},2,delta[a*n+b]+delta[b*n+a]};
                v->cycles[v->count++]=r;
            }else for(int c=0;c<n;c++)if(c!=a&&c!=b&&free_cell[c]){
                Cycle r={{a,b,c,0},3,delta[a*n+b]+delta[b*n+c]+delta[c*n+a]};
                v->cycles[v->count++]=r;
            }
        }
        int source[MAXN+1]; for(int c=0;c<=n;c++)source[c]=c;
        for(int j=0;j<v->count;j++){
            Cycle *b=v->cycles+j;
            int same=0,seen[12],ns=0;
            for(int i=0;i<b->k;i++){
                int c=b->c[i];
                for(int u=0;u<i;u++)same|=q->reach[c*n+b->c[u]];
                source[c]=donor(b,i);
            }
            if(same){
                b->gain=0;
                for(int i=0;i<b->k;i++){
                    int c=b->c[i];
                    for(int h=-1;h<3;h++){
                        int x=h<0?c:nb[3*c+h];
                        if(x>=n)continue;
                        int u=0;while(u<ns&&seen[u]!=x)u++;
                        if(u==ns){seen[ns++]=x;b->gain+=star_value(q,x,source)-base[x];}
                    }
                }
            }
            for(int i=0;i<b->k;i++)source[b->c[i]]=b->c[i];
        }
        qsort(v->cycles,v->count,sizeof(Cycle),order_gain);
        if(v->count&&v->cycles[0].gain>q->max_gain)q->max_gain=v->cycles[0].gain;
    }
    return q;
fail:cycle_free(q);return NULL;
}

/* Only stars whose input identities change need repricing. Stamping is
   local to a worker; start/stop calls can be replayed without global state. */
typedef struct {
    int source[MAXN+1],mark[MAXN], touched[32],ns,gen;
} Scratch;
static void assign(const Model *q,Scratch *s,const Cycle *p){
    for(int i=0;i<p->k;i++){
        int c=p->c[i];
        s->source[c]=donor(p,i);
        for(int h=-1;h<3;h++){
            int v=h<0?c:q->nb[3*c+h];
            if(v<q->n&&s->mark[v]!=s->gen){s->mark[v]=s->gen;s->touched[s->ns++]=v;}
        }
    }
}
static int exact_stars(const Model *q,Scratch *s,const Cycle *a,const Cycle *b){
    if(s->gen>=1000000000){memset(s->mark,0,sizeof(s->mark));s->gen=0;}
    ++s->gen;s->ns=0;assign(q,s,a);if(b)assign(q,s,b);
    int val=q->base_sum;
    for(int i=0;i<s->ns;i++){
        int v=s->touched[i];val+=star_value(q,v,s->source)-q->base[v];
    }
    for(int j=0;j<a->k;j++)s->source[a->c[j]]=a->c[j];
    if(b)for(int j=0;j<b->k;j++)s->source[b->c[j]]=b->c[j];
    return val;
}

static int test_pair(const Model*q,Scratch *scratch,const Cycle *a,const Cycle*b,
        uint64_t *metrics,int *out,int cap,int *hits){
    metrics[TRIED]++;
    if(exact_stars(q,scratch,a,b)<q->target)return 0;
    metrics[BOUNDED]++;
    if(*hits>=cap)return 1;
    int *row=out+6*(*hits)++;
    for(int j=0;j<a->k;j++)*row++=a->c[j];
    for(int j=0;j<b->k;j++)*row++=b->c[j];
    return 0;
}

/* Standalone gain + an upper bound on the positive interaction at each
   shared star. For a touched star, the excess over the sum of independent
   effects cannot exceed 3-base minus the two standalone changes. */
static int left_pricing(const Model *q,Scratch *s,const Cycle *a){
    if(s->gen>=1000000000){memset(s->mark,0,sizeof(s->mark));s->gen=0;}
    ++s->gen;s->ns=0;assign(q,s,a);
    int left_sum=q->base_sum;
    for(int i=0;i<s->ns;i++){
        int c=s->touched[i];int x=star_value(q,c,s->source);
        left_sum+=x-q->base[c];
    }
    for(int j=0;j<a->k;j++)s->source[a->c[j]]=a->c[j];
    return left_sum;
}

/* In a star containing renamed cells, each input y is allocated its
   standalone delta plus a nonnegative correction when another changed
   input x is present. The precomputed directed correction tables majorize
   every one of the at most 4! insertion orders. */
static int join_left(const Model*q,Scratch *s,const Cycle *a,int right,
        int certified,uint64_t *v,int *out,int cap,int *hits){
    int n=q->n,flag[MAXN]={0},near[40],nn=0;
    int sum=left_pricing(q,s,a);
    v[STREAM]++;
    for(int i=0;i<a->k;i++)flag[a->c[i]]=2;
    for(int i=0;i<a->k;i++){
        int c=a->c[i];
        for(int h=-1;h<3;h++){
            int z=h<0?c:q->nb[3*c+h];if(z>=n)continue;
            for(int t=-1;t<3;t++){
                int x=t<0?z:q->nb[3*z+t];
                if(x<n&&!flag[x]&&q->free_cell[x]){flag[x]=1;near[nn++]=x;}
            }
        }
    }
    int bonus[MAXN]={0},largest=0,second=0,third=0;
    for(int h=0;h<nn;h++){
        int x=near[h];
        int val=0;
        for(int i=0;i<a->k;i++)val+=q->max_extra[x*n+a->c[i]];
        bonus[x]=val;
        if(val>largest){third=second;second=largest;largest=val;}
        else if(val>second){third=second;second=val;}
        else if(val>third)third=val;
    }
    int global_extra=largest+second+(right==3?third:0);
    if(sum+q->max_gain+global_extra<q->target)return 0;
    for(int h=0;h<(certified?nn:n);h++){
        int x=certified?near[h]:h;
        if(flag[x]==2||!q->free_cell[x])continue;
        List *list=(right==2?(List*)q->swaps:(List*)q->triangles)+x;
        const int prior=sum+bonus[x]+largest+(right==3?second:0);
        for(int j=0;j<list->count;j++){
            const Cycle *b=list->cycles+j;
            if(prior+b->gain<q->target)break;
            int val=sum+b->gain+bonus[x],ok=1,smallest=x;
            for(int k=1;k<right;k++){
                int y=b->c[k];
                if(flag[y]==2 || (y<x&&(!certified||flag[y]))){ok=0;break;}
                if(y<smallest)smallest=y;
                val+=bonus[y];
            }
            if(!ok || val<q->target || (a->k==right&&a->c[0]>smallest))continue;
            v[LOOKUP]++;
            if(test_pair(q,s,a,b,v,out,cap,hits))return 1;
        }
    }return 0;
}

/* Fixed first and second cell of a canonical directed cycle. */
API int cycle_page(Model *q,int start,int stop,int left,int right,int certified,
        uint64_t *stats,int *out,int capacity) {
    if(!q||start<0||stop>q->n*q->n||start>stop||capacity<1||
            !((left==3 && right==3)||(left==4 && right==2)))return -2;
    Scratch s={0};for(int i=0;i<=q->n;i++)s.source[i]=i;
    int hits=0;
    for(int id=start;id<stop;id++){
        int a=id/q->n,b=id%q->n;
        if(a>=b||!q->free_cell[a]||!q->free_cell[b])continue;
        for(int c=a+1;c<q->n;c++)if(b!=c&&q->free_cell[c]){
            Cycle first={{a,b,c,0},left,0};
            if(left==3){
                if(join_left(q,&s,&first,right,certified,stats,out,capacity,&hits))return -1;
            }else for(int d=a+1;d<q->n;d++)if(d!=b&&d!=c&&q->free_cell[d]){
                first.c[3]=d;
                if(join_left(q,&s,&first,right,certified,stats,out,capacity,&hits))return -1;
            }
        }
    }return hits;
}
