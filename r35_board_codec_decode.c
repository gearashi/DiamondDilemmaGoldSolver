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

/* Decode rows from the measured, bounded nondefault-frame representation.
   The four outputs belong to a fresh generation initialized to their physical defaults. */
EXPORT int dd_restore_default_frames(uint32_t n, uint32_t rows, uint32_t size,
    const uint32_t *desc, const uint32_t *packet, uint16_t *firsts,
    uint16_t *lengths, uint16_t *cursors, uint16_t *shifts) {
    if (!n || n > 131072 || desc[0] != n || desc[1] != (n + 127) / 128 ||
        desc[2] > 160 || rows > 160 - desc[2]) return 0;
    uint32_t consumed = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t *plan = desc + 3 + 10 * row;
        uint32_t mode = plan[0], lo = plan[1], hi = plan[2], base = plan[3];
        uint32_t directory = plan[4], count = plan[5], total = plan[9];
        if (base != consumed) return 0;
        if (!mode) {
            if (count || directory || lo != n || hi || plan[6] || plan[7] || plan[8] || total) return 0;
            continue;
        }
        if (mode > 3 || lo >= hi || hi > n || !count || count > hi-lo) return 0;
        uint32_t expected = mode == 1 ? 0 : mode == 2 ? (hi-1)/32-lo/32+1 : count;
        uint32_t take = mode == 1 ? hi-lo : count;
        uint32_t bias[4], width[4], mask[4], shift[4], bits = 0;
        for (unsigned f = 0; f < 4; ++f) {
            bias[f] = (plan[6+f/2] >> (16*(f&1))) & 65535;
            width[f] = (plan[8] >> (5*f)) & 31;
            if (width[f] > 16) return 0;
            mask[f] = (1u << width[f])-1;
            shift[f] = bits;
            bits += width[f];
        }
        if (plan[8] >> 20 || bits != total || directory != expected) return 0;
        uint32_t stream_bits = take*total;
        uint32_t words = (stream_bits+31)/32;
        if (directory > size-base || words > size-base-directory) return 0;
        const uint32_t *index = packet+base, *values = index+directory;
        if ((stream_bits & 31) && values[words-1] >> (stream_bits & 31)) return 0;
        size_t target = (size_t)(desc[2]+row)*n;
        if (!total && mode == 1) {
            if (count != take || !(bias[0] || bias[1] || bias[2] != 65535 || bias[3])) return 0;
            uint16_t *fields[4] = {firsts, lengths, cursors, shifts};
            for (unsigned f = 0; f < 4; ++f) {
                uint16_t *span = fields[f] + target + lo;
                for (uint32_t j = 0; j < take; ++j) span[j] = (uint16_t)bias[f];
            }
            /* These physically contiguous constant rows have no directory or payload words. */
            continue;
        }
        uint32_t prior = 0, present = 0, word = 0, bitmap = 0;
        for (uint32_t i = 0; i < take; ++i) {
            uint32_t lane = lo+i;
            if (mode == 2) {
                while (!bitmap && word < directory) bitmap = index[word++];
                if (!bitmap) return 0;
                lane = (lo/32 + word-1)*32 + low_bit(bitmap);
                bitmap &= bitmap-1;
            } else if (mode == 3) {
                lane = index[i];
            }
            if (lane < lo || lane >= hi || (i ? lane <= prior : lane != lo) ||
                (i+1 == take && lane+1 != hi)) return 0;
            uint32_t value[4], first_bit = i*total;
            for (unsigned f = 0; f < 4; ++f) {
                uint32_t v = bias[f];
                if (width[f]) {
                    uint32_t at = first_bit+shift[f], off = at&31;
                    uint32_t val = values[at/32] >> off;
                    if (off+width[f] > 32) val |= values[at/32+1] << (32-off);
                    v += val & mask[f];
                    if (v > 65535) return 0;
                }
                value[f] = v;
            }
            uint32_t changed = value[0] || value[1] || value[2] != 65535 || value[3];
            if (!changed && (mode != 1 || i == 0 || i+1 == take)) return 0;
            present += changed;
            firsts[target+lane] = (uint16_t)value[0];
            lengths[target+lane] = (uint16_t)value[1];
            cursors[target+lane] = (uint16_t)value[2];
            shifts[target+lane] = (uint16_t)value[3];
            prior = lane;
        }
        if (present != count || (mode == 2 && (bitmap || word != directory))) return 0;
        consumed += directory+words;
    }
    return consumed == size;
}
