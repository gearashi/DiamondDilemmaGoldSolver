/* Production two-cycle pricing also supplies component catalogues in the
   disconnected-group audit. q->target uses an exact doubled star threshold. */
#include "r30_cycle_native.c"

API int group_page(Model *q,int start,int stop,int left,int target,
                   uint64_t *stats,int *out,int capacity){
    if(!q||start<0||stop>q->n*q->n||start>stop||capacity<1||
            (left!=2&&left!=3))return -2;
    q->target=target;
    Scratch s={0};
    for(int i=0;i<=q->n;i++)s.source[i]=i;
    int hits=0;
    for(int id=start;id<stop;id++){
        int a=id/q->n,b=id%q->n;
        if(a>=b||!q->free_cell[a]||!q->free_cell[b])continue;
        Cycle first={{a,b,0,0},left,0};
        if(left==2){
            if(join_left(q,&s,&first,2,1,stats,out,capacity,&hits))return -1;
        }else for(int c=a+1;c<q->n;c++)if(b!=c&&q->free_cell[c]){
            first.c[2]=c;
            if(join_left(q,&s,&first,2,1,stats,out,capacity,&hits))return -1;
        }
    }
    return hits;
}
