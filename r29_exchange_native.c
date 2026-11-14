/* Sparse proof search for three disjoint transpositions.
   Each seam's unrestricted physical compatibility depends on at most two
   distinct swaps. Its exact relaxation is quadratic, even on a triangle. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

typedef struct { int a,b,g,star,nt,t[8],sd[8]; } Swap;
typedef struct { int id,price; } Neighbor;
typedef struct {
    int n,m,base,stars,limit,maxplus; const int *nb;
    const unsigned char *fit,*masks;
    Swap *s; signed char *cross;
    int *degree; Neighbor **neighbors;
    int *owners,owners_n;
} Graph;
enum { PAIRS, CANDIDATES, SEAM_PASS, OWNED, STAR_PASS, PREFIX_OWNED, COUNTERS };

API int route_counter_count(void){return COUNTERS;}

static int by_gain(const void *aa,const void *bb){
    const Swap *a=aa,*b=bb;
    if(a->g!=b->g)return b->g-a->g;
    return a->a!=b->a?a->a-b->a:a->b-b->b;
}
static int by_price(const void *aa,const void *bb){
    const Neighbor *a=aa,*b=bb;
    return a->price!=b->price?b->price-a->price:a->id-b->id;
}
static int contact(const Graph *q,int a,int b){ return q->fit[a*q->n+b]; }
static int change(const Swap *s,int x){ return x==s->a?s->b:x==s->b?s->a:x; }
static int overlap(const Swap *a,const Swap *b){
    return a->a==b->a||a->a==b->b||a->b==b->a||a->b==b->b;
}
static int value(const Graph *q,int c,const int *source){
    int row=source[c]*3*(q->n+1);
    unsigned x=q->masks[row+source[q->nb[3*c]]];
    unsigned y=q->masks[row+(q->n+1)+source[q->nb[3*c+1]]];
    unsigned z=q->masks[row+2*(q->n+1)+source[q->nb[3*c+2]]];
    return (x&y&z)?3:((x&y)|(y&z)|(x&z))?2:(x|y|z)?1:0;
}
API void route_free(Graph *q){
    if(!q)return;
    if(q->neighbors)for(int i=0;i<q->m;i++)free(q->neighbors[i]);
    free(q->neighbors);free(q->degree);free(q->cross);free(q->s);free(q);
}
/* nb, fit, masks and owners are held alive by the calling Python object. */
API Graph *route_new(int n,int m,const int *pairs,const int *nb,const unsigned char *fit,
                     const unsigned char *masks,int target,int *owners,int owners_n){
    Graph *q=calloc(1,sizeof(*q)); if(!q)return NULL;
    q->n=n;q->m=m;q->limit=target;q->nb=nb;q->fit=fit;q->masks=masks;
    q->owners=owners;q->owners_n=owners_n;
    q->s=calloc(m,sizeof(Swap));q->degree=calloc(m,sizeof(int));
    q->neighbors=calloc(m,sizeof(Neighbor*));q->cross=calloc((size_t)m*m,1);
    int *source=calloc(n+1,sizeof(int)),*base=calloc(n,sizeof(int)),*ids=malloc((size_t)n*n*sizeof(int));
    if(!q->s||!q->degree||!q->neighbors||!q->cross||!source||!base||!ids)goto fail;
    for(int i=0;i<=n;i++)source[i]=i;
    for(int c=0;c<n;c++){
        q->stars+=(base[c]=value(q,c,source));
        for(int h=0;h<3;h++){int b=nb[3*c+h];if(b<n && c<b)q->base+=contact(q,c,b);}
    }
    for(int i=0;i<m;i++){
        Swap *s=q->s+i;s->a=pairs[2*i];s->b=pairs[2*i+1];
        source[s->a]=s->b;source[s->b]=s->a;
        for(int z=0;z<2;z++){
            int a=z?s->b:s->a;
            for(int h=-1;h<3;h++){
                int b=h<0?a:nb[a*3+h]; if(b==n)continue;
                if(h>=0 && !(a==s->b && b==s->a))
                    s->g+=contact(q,source[a],source[b])-contact(q,a,b);
                int t=0;while(t<s->nt && s->t[t]!=b)t++;
                if(t==s->nt){s->t[t]=b;s->sd[t]=value(q,b,source)-base[b];s->star+=s->sd[t];s->nt++;}
            }
        }
        source[s->a]=s->a;source[s->b]=s->b;
    }
    qsort(q->s,m,sizeof(Swap),by_gain);
    for(int i=0;i<n*n;i++)ids[i]=-1;
    for(int i=0;i<m;i++){Swap *s=q->s+i;ids[s->a*n+s->b]=ids[s->b*n+s->a]=i;}
    for(int i=0;i<m;i++){
        Swap *s=q->s+i;
        for(int t=0;t<2;t++){
            int a=t?s->b:s->a,ap=change(s,a);
            for(int h=0;h<3;h++){
                int b=nb[3*a+h];if(b>=n||b==ap)continue;
                for(int bp=0;bp<n;bp++){
                    int j=ids[b*n+bp];if(j<=i||bp==a||bp==ap)continue;
                    /* Inclusion-exclusion on this seam, once per i,j seam. */
                    int diff=contact(q,ap,bp)-contact(q,a,bp)-contact(q,ap,b)+contact(q,a,b);
                    q->cross[(size_t)i*m+j]+=diff;q->cross[(size_t)j*m+i]+=diff;
                }
            }
        }
    }
    for(int i=0;i<m;i++){
        signed char *row=q->cross+(size_t)i*m;
        for(int j=0;j<m;j++)if(row[j]>0){q->degree[i]++;if(row[j]>q->maxplus)q->maxplus=row[j];}
        if(!q->degree[i])continue;
        q->neighbors[i]=malloc(q->degree[i]*sizeof(Neighbor));if(!q->neighbors[i])goto fail;
        int t=0;for(int j=0;j<m;j++)if(row[j]>0)q->neighbors[i][t++]=(Neighbor){j,row[j]+q->s[j].g};
        qsort(q->neighbors[i],q->degree[i],sizeof(Neighbor),by_price);
    }
    free(source);free(base);free(ids);return q;
fail:
    free(source);free(base);free(ids);route_free(q);return NULL;
}
/* Owner rows contain donor-at-cell permutations relative to this center, plus
   a radius, or -1 for a previously completed triple-transposition shell. */
static int covered(const Graph *q,const int *source,const int *changed,int count,int left){
    int stride=q->n+14;
    for(int r=0;r<q->owners_n;r++){
        const int *o=q->owners+(size_t)r*stride;int radius=o[q->n],diff=o[q->n+1];
        for(int t=0;t<count;t++){int c=changed[t];diff+=(source[c]!=o[c])-(c!=o[c]);}
        if(radius>=0){if(diff+2*left<=radius)return 1;continue;}
        if(left || diff!=6)continue;
        int ok=1;
        for(int t=0;t<count+o[q->n+1]&&ok;t++){
            int c=t<count?changed[t]:o[q->n+2+t-count];
            /* o maps destination to donor. Find the implied partner among
               changed cells/owner exceptions (at most 18) and check inverse. */
            if(source[c]==o[c])continue;
            int j=source[c]; for(int u=0;u<o[q->n+1];u++){int x=o[q->n+2+u];if(o[x]==source[c]){j=x;break;}}
            if(source[j]!=o[c])ok=0;
        }
        if(ok)return 1;
    }
    return 0;
}
static int prefix_owned(const Graph *q,const Swap *a,const Swap *b){
    for(int r=0;r<q->owners_n;r++){
        const int *o=q->owners+(size_t)r*(q->n+14);
        if(o[q->n]<0)continue;
        int diff=o[q->n+1];
        const Swap *s[2]={a,b};
        for(int z=0;z<2;z++){
            int x=s[z]->a,y=s[z]->b;
            diff+=(y!=o[x])-(x!=o[x])+(x!=o[y])-(y!=o[y]);
        }
        if(diff+2<=o[q->n])return 1;
    }
    return 0;
}
static int consider(const Graph *q,int i,int j,int k,int seam,uint64_t *v,
                    int *out,int cap,int *count,int *src,int *mark,int *single,int *base){
    const Swap *a=q->s+i,*b=q->s+j,*c=q->s+k;
    if(overlap(a,b)||overlap(a,c)||overlap(b,c))return 0;
    v[CANDIDATES]++;
    if(seam<q->limit)return 0;
    v[SEAM_PASS]++;
    const Swap *ss[3]={a,b,c};int ch[6],touched[24],nt=0,score=q->stars;
    for(int z=0;z<3;z++){
        const Swap *s=ss[z];ch[2*z]=s->a;ch[2*z+1]=s->b;src[s->a]=s->b;src[s->b]=s->a;
    }
    int already=covered(q,src,ch,6,0);
    if(already)v[OWNED]++;
    else{
        for(int z=0;z<3;z++){
            const Swap *s=ss[z];score+=s->star;
            for(int t=0;t<s->nt;t++){
                int x=s->t[t];if(!mark[x]++){touched[nt++]=x;single[x]=0;}single[x]+=s->sd[t];
            }
        }
        for(int t=0;t<nt;t++){int x=touched[t];if(mark[x]>1)score+=value(q,x,src)-base[x]-single[x];mark[x]=0;}
    }
    for(int t=0;t<6;t++)src[ch[t]]=ch[t];
    if(already||score<2*q->limit)return 0;
    v[STAR_PASS]++;
    if(*count==cap)return 1;
    for(int t=0;t<6;t++)out[6*(*count)+t]=ch[t];
    (*count)++;return 0;
}
/* Partition triples by their POSITIVE interaction edges: none, one or at
   least two. Every remaining interaction can only decrease the bound. */
API int route_page(Graph *q,int start,int stop,uint64_t *v,int *out,int cap){
    int n=q->n,m=q->m,count=0,target=q->limit-q->base;
    int *area=calloc((size_t)4*(n+1),sizeof(int));if(!area)return -2;
    int *src=area,*mark=src+n+1,*single=mark+n+1,*base=single+n+1;
    for(int c=0;c<=n;c++)src[c]=c;
    for(int c=0;c<n;c++)base[c]=value(q,c,src);
    for(int i=start;i<stop;i++){
        Swap *a=q->s+i;signed char *row=q->cross+(size_t)i*m;
        if(i+1<m && a->g+2*q->s[i+1].g>=target){
            for(int j=i+1;j<m;j++){
                int g=a->g+q->s[j].g;if(g+q->s[j].g<target)break;
                if(row[j]>0||overlap(a,q->s+j))continue;
                if(prefix_owned(q,a,q->s+j)){v[PREFIX_OWNED]++;continue;}
                signed char *rj=q->cross+(size_t)j*m; v[PAIRS]++;
                for(int k=j+1;k<m&&g+q->s[k].g>=target;k++){
                    if(row[k]>0||rj[k]>0)continue;
                    if(consider(q,i,j,k,q->base+g+q->s[k].g+row[j]+row[k]+rj[k],v,out,cap,&count,src,mark,single,base))goto overflow;
                }
            }
        }
        Neighbor *ne=q->neighbors[i];int degree=q->degree[i];
        for(int t=0;t<degree;t++){
            int j=ne[t].id,g=a->g+ne[t].price;signed char *rj=q->cross+(size_t)j*m;
            if(prefix_owned(q,a,q->s+j)){v[PREFIX_OWNED]++;continue;}
            if(i<j && g+q->s[0].g>=target){
                v[PAIRS]++;
                for(int k=0;k<m&&g+q->s[k].g>=target;k++){
                    if(k==i||k==j||row[k]>0||rj[k]>0)continue;
                    if(consider(q,i,j,k,q->base+g+q->s[k].g+row[k]+rj[k],v,out,cap,&count,src,mark,single,base))goto overflow;
                }
            }
            if(g+ne[t].price+q->maxplus<target){if(g+q->s[0].g<target)break;continue;}
            v[PAIRS]++;
            for(int u=t+1;u<degree && g+ne[u].price+q->maxplus>=target;u++){
                int k=ne[u].id;
                if(g+ne[u].price+rj[k]<target || (rj[k]>0&&(i>j||i>k)))continue;
                if(consider(q,i,j,k,q->base+g+ne[u].price+rj[k],v,out,cap,&count,src,mark,single,base))goto overflow;
            }
        }
    }
    free(area);return count;
overflow:
    free(area);return -1;
}
