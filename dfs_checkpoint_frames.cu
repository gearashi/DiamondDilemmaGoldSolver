extern "C" __global__ void gather_frames(const unsigned short *source,
        const unsigned int *layout, unsigned short *packed) {
    unsigned int row = blockIdx.y, column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int *rect = layout + 2u + 3u * row;
    if (column >= rect[1]) return;
    unsigned int pos = rect[2];
    unsigned int lane = (pos & 0x80000000u) ? (pos & 0x7fffffffu) + column : layout[pos + column];
    packed[rect[0] + column] = source[row * layout[0] + lane];
}
