extern "C" __global__ void gather_frame_packet(const unsigned short *first,
    const unsigned short *length, const unsigned short *cursor, const unsigned short *shift,
    const unsigned *layout, unsigned short *packed) {
    unsigned row=blockIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned *window=layout+2+3*row;
    if (col>=window[1]) return;
    unsigned address=window[2];
    unsigned lane=(address&0x80000000u)?(address&0x7fffffffu)+col:layout[address+col];
    unsigned src=row*layout[0]+lane, dest=window[0]+col, pitch=layout[1];
    packed[dest]=first[src];
    packed[pitch+dest]=length[src];
    packed[2*pitch+dest]=cursor[src];
    packed[3*pitch+dest]=shift[src];
}
