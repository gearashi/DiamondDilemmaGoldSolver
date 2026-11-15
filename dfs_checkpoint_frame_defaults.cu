extern "C" __global__ void measure(const unsigned short *a,const unsigned short *b,
    const unsigned short *c,const unsigned short *d,const unsigned *p,unsigned *work) {
    unsigned n=p[0], groups=p[1],row=blockIdx.y,t=threadIdx.x,lane=128*blockIdx.x+t;
    unsigned base=9*(row*groups+blockIdx.x),j=row*n+lane;
    unsigned v[4]={0,0,65535,0};
    if(lane<n){v[0]=a[j];v[1]=b[j];v[2]=c[j];v[3]=d[j];}
    bool dirty=v[0] || v[1] || v[2]!=65535 || v[3];
    unsigned mask=__ballot_sync(0xffffffffu,dirty), warp=t/32;
    __shared__ unsigned lows[4][4],highs[4][4];
    if (!(t&31)) work[base+warp]=mask;
    for(unsigned f=0;f<4;f++){
        unsigned lo=dirty?v[f]:65535u,hi=dirty?v[f]:0u;
        for(unsigned distance=16;distance;distance/=2) {
            unsigned x=__shfl_down_sync(0xffffffffu,lo,distance),y=__shfl_down_sync(0xffffffffu,hi,distance);
            lo=min(lo,x);hi=max(hi,y);
        }
        if(!(t&31)){lows[f][warp]=lo;highs[f][warp]=hi;}
    }
    __syncthreads();
    if(t<4) {
        unsigned lo=65535,hi=0;
        for(unsigned w=0;w<4;w++){lo=min(lo,lows[t][w]);hi=max(hi,highs[t][w]);}
        work[base+5+t]=lo|(hi<<16);
    }
}
extern "C" __global__ void scan(const unsigned *p,unsigned *work,unsigned *meta) {
    unsigned row=blockIdx.x,groups=p[1],n=p[0],seen=0,low=n,high=0;
    unsigned mins[4]={65535,65535,65535,65535},maxs[4]={0,0,0,0};
    for (unsigned g=0;g<groups;g++) {
        unsigned at=9*(row*groups+g);work[at+4]=seen;
        for(unsigned w=0;w<4;w++) {
            unsigned m=work[at+w];
            if (m) {low=min(low,128*g+32*w+__ffs(m)-1);high=128*g+32*w+32-__clz(m);seen+=__popc(m);}
            unsigned range=work[at+5+w];mins[w]=min(mins[w],range&65535u);maxs[w]=max(maxs[w],range>>16);
        }
    }
    unsigned base=5+7*row;
    meta[base]=seen;meta[base+1]=low;meta[base+2]=high;
    for(unsigned f=0;f<4;f++)meta[base+3+f]=mins[f]|(maxs[f]<<16);
    if(seen)atomicOr(meta+row/32,1u<<(row%32));
}
extern "C" __global__ void pack(const unsigned short *a,const unsigned short *b,
    const unsigned short *c,const unsigned short *d,const unsigned *p,const unsigned *work,unsigned *out) {
    unsigned n=p[0],groups=p[1],row=p[2]+blockIdx.y,t=threadIdx.x,lane=128*blockIdx.x+t;
    const unsigned *s=p+3+10*blockIdx.y;
    unsigned mode=s[0],low=s[1],high=s[2],base=1+s[3],dir=s[4];
    if(!mode || lane>=n)return;
    unsigned at=9*(row*groups+blockIdx.x),w=t/32,m=work[at+w],word=lane/32;
    if(mode==2 && !(t&31) && word>=low/32 && word<=(high-1)/32)out[base+word-low/32]=m;
    unsigned rank;
    if(mode==1) {if(lane<low || lane>=high)return;rank=lane-low;}
    else {
        unsigned bit=1u<<(t&31);if(!(m&bit))return;
        rank=work[at+4]+__popc(m&(bit-1));
        for(unsigned k=0;k<w;k++)rank+=__popc(work[at+k]);
        if(mode==3)out[base+rank]=lane;
    }
    unsigned j=row*n+lane,v[4]={a[j],b[j],c[j],d[j]},filled=0;
    unsigned long long value=0;
    for(unsigned f=0;f<4;f++){
        unsigned bias=(s[6+f/2]>>(16*(f%2)))&65535u,width=(s[8]>>(5*f))&31u;
        unsigned rel=v[f]-bias;
        if(rel>=(1u<<width)){atomicOr(out,1u);return;}
        if(width)value|=(unsigned long long)rel<<filled;filled+=width;
    }
    unsigned bits=rank*s[9],pos=base+dir+bits/32,shift=bits%32;
    unsigned x=(unsigned)value,y=(unsigned)(value>>32),word0=x<<shift;
    unsigned word1=shift?((y<<shift)|(x>>(32-shift))):y;
    unsigned word2=shift?(y>>(32-shift)):0;
    if(word0)atomicOr(out+pos,word0);
    if(word1)atomicOr(out+pos+1,word1);
    if(word2)atomicOr(out+pos+2,word2);
}
