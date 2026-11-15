extern "C" __global__ void measure(const unsigned short *a,const unsigned short *b,
    const unsigned short *c,const unsigned short *d,const unsigned *p,unsigned *work) {
    unsigned n=p[0], groups=p[1],row=blockIdx.y,t=threadIdx.x,lane=128*blockIdx.x+t;
    unsigned base=5*(row*groups+blockIdx.x),j=row*n+lane;
    bool dirty=lane<n && (a[j] || b[j] || c[j]!=65535 || d[j]);
    unsigned mask=__ballot_sync(0xffffffffu,dirty);
    if (!(t&31)) work[base+t/32]=mask;
}
extern "C" __global__ void scan(const unsigned *p,unsigned *work,unsigned *meta) {
    unsigned row=blockIdx.x,groups=p[1],n=p[0],seen=0,low=n,high=0;
    for (unsigned g=0;g<groups;g++) {
        unsigned at=5*(row*groups+g);work[at+4]=seen;
        for(unsigned w=0;w<4;w++) {
            unsigned m=work[at+w];
            if (m) {unsigned first=128*g+32*w+__ffs(m)-1;
                if(first<low)low=first;high=128*g+32*w+32-__clz(m);seen+=__popc(m);}
        }
    }
    meta[3*row]=seen;meta[3*row+1]=low;meta[3*row+2]=high;
}
extern "C" __global__ void pack(const unsigned short *a,const unsigned short *b,
    const unsigned short *c,const unsigned short *d,const unsigned *p,const unsigned *work,unsigned *out) {
    unsigned n=p[0],groups=p[1],row=p[2]+blockIdx.y,t=threadIdx.x,lane=128*blockIdx.x+t;
    const unsigned *s=p+3+6*blockIdx.y;
    unsigned mode=s[0],low=s[1],high=s[2],base=s[3],dir=s[4];
    if(!mode || lane>=n)return;
    unsigned at=5*(row*groups+blockIdx.x),w=t/32,m=work[at+w],word=lane/32;
    if(mode==2 && !(t&31) && word>=low/32 && word<=(high-1)/32)out[base+word-low/32]=m;
    unsigned rank;
    if(mode==1) {if(lane<low || lane>=high)return;rank=lane-low;}
    else {
        unsigned bit=1u<<(t&31);
        if(!(m&bit))return;
        rank=work[at+4]+__popc(m&(bit-1));
        for(unsigned k=0;k<w;k++)rank+=__popc(work[at+k]);
        if(mode==3)out[base+rank]=lane;
    }
    unsigned j=row*n+lane,pos=base+dir+2*rank;
    out[pos]=(unsigned)a[j]|((unsigned)b[j]<<16);
    out[pos+1]=(unsigned)c[j]|((unsigned)d[j]<<16);
}
