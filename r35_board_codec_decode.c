/* Exact decoding of the standalone occupancy-mask packet, including damaged int16 values. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#if defined(_MSC_VER)
#include <intrin.h>
static unsigned int low_bit(uint32_t v) { unsigned long i; _BitScanForward(&i,v); return (unsigned int)i; }
#else
static unsigned int low_bit(uint32_t v) { return (unsigned int)__builtin_ctz(v); }
#endif

EXPORT int dd_physical_board_unpack(int k,int packet_size,const uint32_t *starts,const uint32_t *caps,
    const uint32_t *packet,int16_t *board,uint32_t *used) {
    if (k<0 || k>131072 || packet_size < 1+10*k || packet[0]) return 0;
    uint32_t expected=0;
    for(int i=0;i<k;i++) {
        if(caps[i]>160u || starts[i]!=expected)return 0;
        expected+=(caps[i]+1u)/2u;
    }
    if((uint32_t)packet_size!=1u+10u*(uint32_t)k+expected)return 0;
    const uint32_t *masks=packet+1,*codes=packet+1+10*k;
    memset(board,255,(size_t)160*k*sizeof(int16_t));
    memcpy(used,packet+1+5*k,(size_t)5*k*sizeof(uint32_t));
    for (int first=0;first<k;first+=256) {
        int end=first+256<k?first+256:k;
        uint32_t counts[256]={0};
        for(int word=0;word<5;word++) {
            for(int i=first;i<end;i++) {
                uint32_t bits=masks[word*k+i];
                uint32_t rank=counts[i-first];
                while(bits) {
                    if(rank>=caps[i])return 0;
                    unsigned int cell=32u*(unsigned int)word+low_bit(bits);
                    uint16_t raw=(uint16_t)(codes[starts[i]+rank/2]>>(16*(rank&1u)));
                    /* Preserve every physical bit pattern, including negative saved values. */
                    memcpy(board+(size_t)cell*k+i,&raw,sizeof(raw));
                    rank++;
                    bits&=bits-1;
                }
                counts[i-first]=rank;
            }
        }
    }
    return 1;
}

/* A lane owns a word-aligned series of four unsigned 16-bit field slices. */
static uint16_t load_half(const uint32_t *words, unsigned half) {
    return (uint16_t)(words[half / 2] >> (16 * (half & 1)));
}

EXPORT int dd_merge_frame_spans(unsigned n, unsigned k, unsigned size,
    const uint32_t *lanes, const uint32_t *marks, const uint32_t *words,
    uint16_t *firsts, uint16_t *lengths, uint16_t *cursors, uint16_t *shifts) {
    unsigned at = 0;
    if (!n || n > 131072 || k > n) return 0;
    for (unsigned j = 0; j < k; ++j) {
        unsigned lane = lanes[j], m = marks[j], a = m & 255, b = (m >> 8) & 255;
        unsigned c = (m >> 16) & 255, d = m >> 24;
        if (lane >= n || (j && lane <= lanes[j-1]) || !c || d < c || d > 160 ||
            (!a != !b) || (a && (a < c || b < a || b > d))) return 0;
        unsigned init = a ? b-a+1 : 0, cursor = d-c+1;
        unsigned count = (3*init + cursor + 1)/2;
        if (count > size-at) return 0;
        const uint32_t *data = words+at;
        for (unsigned r = 0; r < init; ++r) {
            size_t cell = (size_t)(a-1+r)*n+lane;
            firsts[cell] = load_half(data,r);
            lengths[cell] = load_half(data,r+init);
            shifts[cell] = load_half(data,r+2*init);
        }
        for (unsigned r = 0; r < cursor; ++r) {
            size_t cell = (size_t)(c-1+r)*n+lane;
            cursors[cell] = load_half(data,r+3*init);
        }
        at += count;
    }
    return at == size;
}
