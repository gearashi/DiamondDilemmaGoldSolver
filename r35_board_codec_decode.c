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
