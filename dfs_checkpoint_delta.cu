__device__ void price(unsigned value,unsigned &ticket,unsigned &size) {
    unsigned change=value&255u,pop=(value>>8)&255u,used=(value>>16)&31u,bits=(value>>21)?16u:9u;
    ticket=size=0;
    if(!change && !used) return;
    unsigned count=change,mode=0,best=((bits+8)*change+31)/32;
    for(unsigned m=1;m<5;m++) {
        unsigned c=m==4?160u:((m&1)?pop:change);
        unsigned cost=((bits+(m<2?8u:0u))*c+31)/32+(m==2 || m==3?5u:0u);
        if(cost<best) {best=cost;mode=m;count=c;}
    }
    size=best+__popc(used);
    ticket=1+used+32*(10*count+mode+(bits==16?5:0));
}
// Per-block prefix sums are bounded to 128 lanes. The following dispatch
// scans at most 1024 block totals, so it needs no cross-block spin barriers.
extern "C" __global__ void price_groups(const short *b,const unsigned *u,short *old,unsigned *ou,
    const unsigned *plan,unsigned *work) {
    unsigned j=threadIdx.x,i=blockIdx.x*128+j,k=plan[1],ticket=0,size=0;
    if(i<k) {
        unsigned lane=(plan[2]&0x80000000u)?(plan[2]&0x7fffffffu)+i:plan[3+i];
        unsigned n=plan[0], changes[5],occupied[5],change=0,pop=0,bad=0,used=0;
        for(unsigned g=0;g<5;++g) {
            unsigned diff=0,live=0;
            for(unsigned j=0;j<32;++j) {
                unsigned at=(32*g+j)*n+lane;
                short now=b[at];
                if(now!=old[at]) {diff|=1u<<j;old[at]=now;}
                if(now!=-1) {live|=1u<<j;bad|=(now<0 || now>=480);}
            }
            changes[g]=diff;occupied[g]=live;change+=__popc(diff);pop+=__popc(live);
            unsigned at=g*n+lane;
            if(u[at]!=ou[at]) {used|=1u<<g;ou[at]=u[at];}
        }
        price(change|(pop<<8)|(used<<16)|(bad<<21),ticket,size);
        if(ticket) {
            unsigned mode=(((ticket-1)>>5)%10)%5;
            if(mode!=4) for(unsigned g=0;g<5;++g)
                work[(4+g)*k+i]=(mode&1u)?occupied[g]:changes[g];
        }
    }
    __shared__ unsigned words[128], dirty[128];
    words[j]=size;dirty[j]=(ticket!=0);__syncthreads();
    for(unsigned d=1;d<128;d*=2) {
        unsigned w=j>=d?words[j-d]:0,t=j>=d?dirty[j-d]:0;__syncthreads();
        words[j]+=w;dirty[j]+=t;__syncthreads();
    }
    if(i<k) {work[4*i]=ticket;work[4*i+1]=words[j]-size;work[4*i+2]=dirty[j]-(ticket!=0);work[4*i+3]=size;}
    if(j==127) {work[9*k+2*blockIdx.x]=words[j];work[9*k+2*blockIdx.x+1]=dirty[j];}
}
extern "C" __global__ void scan_groups(const unsigned *plan,unsigned *work,unsigned *out) {
    unsigned k=plan[1],words=0,dirty=0;
    for(unsigned g=0;g<(k+127)/128;g++) {
        unsigned at=9*k+2*g,w=work[at],d=work[at+1];work[at]=words;work[at+1]=dirty;words+=w;dirty+=d;
    }
    unsigned sparse=k<=65536?1u:2u,dense=(k+1)/2,form=dense<=sparse*dirty?0u:sparse;
    out[0]=0;out[1]=dirty;out[2]=words;out[3]=form?form*dirty:dense;out[4]=form;
}
extern "C" __global__ void pack_groups(const short *b,const unsigned *u,short *old,unsigned *ou,
    const unsigned *plan,const unsigned *work,unsigned *out) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,k=plan[1];
    if(i>=k) return;
    unsigned ticket=work[4*i],form=out[4];
    if(!form && !(i&1)) out[5+i/2]=ticket|(i+1<k?work[4*(i+1)]<<16:0u);
    if(!ticket) return;
    unsigned at=9*k+2*(i/128),before=work[at]+work[4*i+1],idx=work[at+1]+work[4*i+2];
    if(form==1) out[5+idx]=(i<<16)|ticket;
    if(form==2) {out[5+2*idx]=i;out[6+2*idx]=ticket;}
    unsigned v=ticket-1,choice=(v>>5)%10,flags=choice%5+(choice>=5?8:0)+((v&31u)<<4)+((v/320)<<10);
    unsigned lane=(plan[2]&0x80000000u)?(plan[2]&0x7fffffffu)+i:plan[3+i];
    unsigned base=4+out[3]+before;
    unsigned mode=flags&7u,bits=(flags&8u)?16u:9u,used=(flags>>4)&31u,n=plan[0];
    unsigned pos=1+base+(mode==2 || mode==3?5:0),limit=1+base+work[4*i+3],low=0,filled=0;
    for(unsigned group=0;group<5;++group) {
        unsigned mask=mode==4?~0u:work[(4+group)*k+i];
        if(mode==2 || mode==3) out[1+base+group]=mask;
        while(mask) {
            unsigned row=group*32+(__ffs(mask)-1);mask&=mask-1;
            short now=b[row*n+lane];
            unsigned code=(bits==9 && now==-1)?480u:(unsigned short)now;
            unsigned token=mode<2?(row<<bits)|code:code,width=bits+(mode<2?8u:0u);
            low|=token<<filled;
            if(filled+width>=32) {
                if(pos>=limit) {atomicOr(out,1u);return;}
                out[pos++]=low;low=(filled+width==32)?0u:(token>>(32-filled));
            }
            filled=(filled+width)&31u;
        }
    }
    if(filled) {if(pos>=limit) {atomicOr(out,1u);return;}out[pos++]=low;}
    for(unsigned g=0;g<5;++g) if(used&(1u<<g)) {
        if(pos>=limit) {atomicOr(out,1u);return;}out[pos++]=u[g*n+lane];
    }
    if(pos!=limit) atomicOr(out,1u);
}
