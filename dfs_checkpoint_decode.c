/* Apply physical board and inventory deltas to an independent CPU checkpoint. */
#include <stdint.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static int token_at(const uint32_t *data,unsigned bits_at,unsigned width,unsigned words,uint32_t *value) {
    unsigned pos=bits_at/32,shift=bits_at&31;
    if(pos>=words) return 0;
    uint32_t v=data[pos]>>shift;
    if(shift+width>32) {
        if(pos+1>=words) return 0;
        v|=data[pos+1]<<(32-shift);
    }
    *value=v&((1u<<width)-1);
    return 1;
}

EXPORT int dd_apply_board_delta(int n,int k,int word_count,const int32_t *lanes,
    const uint32_t *flags,const uint32_t *starts,const uint32_t *ends,const uint32_t *data,
    int16_t *board,uint32_t *used) {
    if(n<1 || n>131072 || k<0 || k>n || word_count<0) return 0;
    for(int i=0;i<k;i++) {
        unsigned lane=(unsigned)lanes[i],f=flags[i],mode=f&7u,bits=(f&8u)?16u:9u;
        unsigned used_mask=(f>>4)&31u,count=f>>10,start=starts[i],end=ends[i];
        if(lane>=(unsigned)n || mode>4 || count>160 || start>end || end>(unsigned)word_count) return 0;
        unsigned mask_words=(mode==2 || mode==3)?5u:0u;
        unsigned width=bits+(mode<2?8u:0u),values=(width*count+31)/32;
        unsigned used_words=0;
        for(unsigned m=used_mask;m;m&=m-1) ++used_words;
        if(end-start!=mask_words+values+used_words || (mode==4 && count!=160)) return 0;
        if(mode==1 || mode==3)
            for(unsigned row=0;row<160;row++) board[row*n+lane]=-1;
        unsigned rank=0,prev=0;
        for(unsigned j=0;j<(mask_words?160:count);j++) {
            if(mask_words && !(data[start+j/32]&(1u<<(j&31)))) continue;
            uint32_t token;
            if(rank>=count || !token_at(data+start+mask_words,rank*width,width,values,&token)) return 0;
            unsigned row=mode<2?token>>bits:j;
            if(row>=160 || (mode<2 && rank && row<=prev)) return 0;
            uint32_t v=token&((1u<<bits)-1);
            if(bits==9 && v>480) return 0;
            board[row*n+lane]=(bits==9 && v==480)?-1:(int16_t)(uint16_t)v;
            prev=row;++rank;
        }
        if(rank!=count) return 0;
        unsigned offset=start+mask_words+values;
        for(unsigned row=0;row<5;row++) if(used_mask&(1u<<row)) used[row*n+lane]=data[offset++];
    }
    return 1;
}
