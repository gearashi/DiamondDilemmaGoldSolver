__device__ void emit_span(unsigned n,unsigned lane,unsigned m,unsigned dest,const unsigned short *first, const unsigned short *length, const unsigned short *cursor, const unsigned short *shift,unsigned *out) {
    unsigned a=m&255, b=(m>>8)&255, c=(m>>16)&255, d=m>>24;
    unsigned prefix=a?b-a+1:0, count=3*prefix+d-c+1, left=0;
    for (unsigned i=0;i<count;i++) {
        unsigned field=prefix?min(i/prefix,3u):3u;
        unsigned row=field==3?c-1+i-3*prefix:a-1+i-field*prefix;
        unsigned src=row*n+lane;
        unsigned val=field==0?first[src]:field==1?length[src]:field==2?shift[src]:cursor[src];
        if (i&1) out[dest+i/2]=left|(val<<16);
        else left=val;
    }
    if(count&1) out[dest+count/2]=left;
}

extern "C" __global__ void gather_spans(const unsigned short *first, const unsigned short *length,
    const unsigned short *cursor, const unsigned short *shift, const unsigned *p, unsigned *out) {
    unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=p[1]) return;
    unsigned flags=p[5], lane=flags&1?p[p[2]+j]:p[2]+j;
    unsigned m=flags&2?p[p[3]+j]:p[3], dest=flags&4?p[p[4]+j]:p[4]*j;
    emit_span(p[0],lane,m,dest,first,length,cursor,shift,out);
}

__device__ unsigned dense_mark(const unsigned *p,unsigned j) {
    if (!(p[5]&16)) return p[p[3]+j];
    unsigned bits=p[6], per=32/bits;
    unsigned code=(p[p[3]+j/per]>>(bits*(j%per)))&((1u<<bits)-1u);
    return p[p[7]+code];
}

extern "C" __global__ void gather_dense_spans(const unsigned short *first, const unsigned short *length, const unsigned short *cursor, const unsigned short *shift,const unsigned *p,unsigned *out) {
    unsigned j=blockIdx.x*32+threadIdx.x;
    unsigned m=j<p[1]?dense_mark(p,j):0;
    unsigned a=m&255, b=(m>>8)&255, c=(m>>16)&255, d=m>>24;
    unsigned width=m?(3*(a?b-a+1:0)+d-c+2)/2:0;
    unsigned sum=width;
    for (unsigned step=1;step<32;step*=2) {
        unsigned other=__shfl_up_sync(0xffffffffu,sum,step);
        if(threadIdx.x>=step) sum+=other;
    }
    if (m) emit_span(p[0],p[2]+j,m,p[p[4]+blockIdx.x]+sum-width,
                     first,length,cursor,shift,out);
}
