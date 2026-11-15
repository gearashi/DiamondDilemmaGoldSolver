// Losslessly transport occupied int16 cells and unchanged used words.
extern "C" __global__ void physical_board(const short *boards,const unsigned int *used,
    const unsigned int *layout,unsigned int *result) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int n=layout[0], k=layout[1];
    if (i>=k) return;
    unsigned int lane=layout[2+i], first=layout[2+k+i], cap=layout[2+2*k+i];
    unsigned int *codes=result+1+10*k+first;
    unsigned int seen=0, pair=0;
    for (unsigned int word=0; word<5; ++word) {
        unsigned int mask=0;
        for (unsigned int bit=0;bit<32;++bit) {
            short value=boards[(32*word+bit)*n+lane];
            if (value==-1) continue;
            mask|=1u<<bit;
            if (seen<cap) {
                unsigned int raw=(unsigned short)value;
                if (!(seen&1u)) pair=raw;
                else { pair|=raw<<16; codes[seen/2]=pair; }
            }
            ++seen;
        }
        result[1+word*k+i]=mask;
        result[1+(word+5)*k+i]=used[word*n+lane];
    }
    if (seen>cap) {atomicOr(result,1u);return;}
    if (seen&1u) codes[seen/2]=pair;
    for (unsigned int w=(seen+1)/2;w<(cap+1)/2;++w) codes[w]=0;
}
