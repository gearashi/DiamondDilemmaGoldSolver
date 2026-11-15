extern "C" __global__ void gather_bands(const unsigned short *f0,const unsigned short *f1,
    const unsigned short *f2,const unsigned short *f3,const unsigned *p,unsigned *out) {
    unsigned word=threadIdx.x+blockIdx.x*blockDim.x;
    const unsigned *row=p+3+4*blockIdx.y;
    unsigned j=2*word;
    if (j>=row[1]) return;
    unsigned field=row[0]/160, base=(row[0]%160)*p[0], at=row[2];
    const unsigned short *src=field==0?f0:field==1?f1:field==2?f2:f3;
    unsigned left=src[base+(at&0x80000000u?(at&0x7fffffffu)+j:p[p[2]+at+j])],right=0;
    if(j+1<row[1]) right=src[base+(at&0x80000000u?(at&0x7fffffffu)+j+1:p[p[2]+at+j+1])];
    out[row[3]+word]=left|(right<<16);
}
