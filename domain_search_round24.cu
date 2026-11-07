// One bounded reasoning primitive per iteration. Persistent state is coalesced
// across lanes. Reasons name active decision levels, under this lane's cube.
struct Context {
 unsigned *s; const unsigned *t; int lanes,lane;
 __device__ unsigned &at(int row){
#if COOPERATIVE
  return s[lane*ROWS+row];
#else
  return s[row*lanes+lane];
#endif
 }
 __device__ int meta(int row){return int(at(META+row));}
 __device__ void put(int row,int value){at(META+row)=unsigned(value);}
 __device__ void bump(int index,unsigned amount=1){unsigned old=at(META+index),next=old+amount;at(META+index)=next;if(next<old)at(META+20+index-11)++;}
 __device__ unsigned domain(int cell,int word){return at(DOMAIN+cell*W+word);}
 __device__ bool has(int cell,int code){return (domain(cell,code>>5)&(1u<<(code&31)))!=0;}
 __device__ int count(int cell){int result=0;for(int word=0;word<W;word++)result+=__popc(domain(cell,word));return result;}
 __device__ int single(int cell){int found=-1;for(int word=0;word<W;word++){unsigned value=domain(cell,word);if(!value)continue;if(found>=0||(value&(value-1)))return -1;found=word*32+__ffs(value)-1;}return found;}
 __device__ void zero(unsigned *why){for(int word=0;word<RW;word++)why[word]=0;}
 __device__ void reason(int cell,unsigned *why){for(int word=0;word<RW;word++)why[word]|=at(REASON+cell*RW+word);}
#if PRECISE_REASONS
 __device__ void precise_bump(int index){int row=PRECISE_COUNTERS+2*index;unsigned old=at(row),next=old+1;at(row)=next;if(next<old)at(row+1)++;}
 __device__ int precise_find(unsigned key){
  unsigned start=(key*2654435761u)&(RS-1);
  for(int probe=0;probe<min(32,RS);probe++){precise_bump(0);int slot=(start+probe)&(RS-1);unsigned stored=at(PRECISE_KEYS+slot);if(stored==key)return slot;if(!stored)return -1;}
  return -1;
 }
 __device__ void precise_invalidate(int cell,int code){
  int slot=precise_find(unsigned(cell*NC+code+1));if(slot<0)return;
  at(PRECISE_KEYS+slot)=0xffffffffu;at(PRECISE_COUNT)--;precise_bump(4);
 }
 __device__ void precise_store(int cell,int code,unsigned *why){
  // Always erase an old identity before a possibly dropped new insertion.
  // Undo also invalidates. Saturation must never revive an earlier reason.
  precise_invalidate(cell,code);if(at(PRECISE_ZERO+cell*W+(code>>5))&(1u<<(code&31)))return;bool empty=true;for(int word=0;word<RW;word++)if(why[word])empty=false;
  if(empty){at(PRECISE_ZERO+cell*W+(code>>5))|=1u<<(code&31);return;}
  if(at(PRECISE_COUNT)>=RS/2){precise_bump(7);return;}
  unsigned key=unsigned(cell*NC+code+1),start=(key*2654435761u)&(RS-1);int vacant=-1;
  for(int probe=0;probe<min(32,RS);probe++){
   precise_bump(0);int slot=(start+probe)&(RS-1);unsigned stored=at(PRECISE_KEYS+slot);
   if(stored==0xffffffffu&&vacant<0)vacant=slot;
   if(!stored){if(vacant<0)vacant=slot;break;}
  }
  if(vacant<0){precise_bump(7);return;}
  for(int word=0;word<RW;word++)at(PRECISE_WHY+vacant*RW+word)=why[word];
  at(PRECISE_KEYS+vacant)=key;at(PRECISE_COUNT)++;precise_bump(3);
 }
 __device__ bool value_reason(int cell,int code,unsigned *why){
  unsigned bit=1u<<(code&31);
  if(!(at(PRECISE_ROOT+cell*W+(code>>5))&bit)||(at(PRECISE_ZERO+cell*W+(code>>5))&bit))return true;
  int slot=precise_find(unsigned(cell*NC+code+1));
  if(slot<0){precise_bump(2);reason(cell,why);return false;}
  precise_bump(1);for(int word=0;word<RW;word++)why[word]|=at(PRECISE_WHY+slot*RW+word);return true;
 }
 __device__ void seam_reason(int cell,int other,int code,int excluded_tile,unsigned *why){
  // All seams to this same neighbor constrain one common orientation. Any
  // removed originally allowed support is justified independently; one cache
  // miss safely falls back to the full cell reason and covers the rest too.
  for(int word=0;word<W;word++){
   precise_bump(5);unsigned lost=at(PRECISE_ROOT+other*W+word)&~domain(other,word);
   for(int side=0;side<3;side++)if(int(t[cell*3+side])==other){
    int os=int(t[3*N+cell*3+side]);lost&=t[25*N+NC*33+((side*3+os)*NC+code)*W+word];
   }
   while(lost){int value=word*32+__ffs(lost)-1;lost&=lost-1;if(value/3==excluded_tile)continue;
    precise_bump(6);if(!value_reason(other,value,why))return;
   }
  }
 }
#endif
#if EVENT_PROPAGATION
 __device__ void event_bump(int index){int row=EVENT_COUNTERS+2*index;unsigned old=at(row),next=old+1;at(row)=next;if(next<old)at(row+1)++;}
 __device__ void dirty(int cell){
  if(cell<0||cell>=N)return;unsigned bit=1u<<(cell&31);int row=DIRTY_CELLS+(cell>>5);
  if(!(at(row)&bit)){at(row)|=bit;event_bump(0);}
 }
 __device__ int pop_dirty(){
  for(int word=0;word<TW;word++){event_bump(3);unsigned bits=at(DIRTY_CELLS+word);if(bits){
   int bit=__ffs(bits)-1;at(DIRTY_CELLS+word)&=~(1u<<bit);event_bump(1);return word*32+bit;
  }}return N;
 }
 __device__ void dirty_owner_consumers(int owner,int tile){
  int offset=25*N+NC*33+9*NC*W+tile*TW;
  for(int word=0;word<TW;word++){
   event_bump(4);unsigned consumers=t[offset+word];
   while(consumers){int cell=word*32+__ffs(consumers)-1;consumers&=consumers-1;if(cell==owner)continue;
    event_bump(2);if(has(cell,tile*3)||has(cell,tile*3+1)||has(cell,tile*3+2))dirty(cell);
   }
  }
 }
#endif
 // Cached supports are positive witnesses, never deductions. Validate each
 // against current domains and immutable seam tables after every rollback.
 __device__ void support_bump(int index){
#if SUPPORT_METRICS
  int row=SUPPORT_COUNTERS+2*index;unsigned old=at(row),next=old+1;at(row)=next;if(next<old)at(row+1)++;
#endif
 }
#if SUPPORT_REUSE
 __device__ unsigned support_key(int cell,int code,int kind){return unsigned((cell*NC+code)*4+kind+1);}
 __device__ int support_slot(unsigned key){return (key*2654435761u)&(SS-1);}
 __device__ void support_put(unsigned key,int a,int b=-1,int c=-1){
  int slot=support_slot(key);unsigned prior=at(SUPPORT_KEYS+slot);
  if(prior&&prior!=key)support_bump(5);
  at(SUPPORT_VALUES+slot*3)=unsigned(a);at(SUPPORT_VALUES+slot*3+1)=unsigned(b);at(SUPPORT_VALUES+slot*3+2)=unsigned(c);
  at(SUPPORT_KEYS+slot)=key;support_bump(4);
 }
#endif
 __device__ bool seam_supported(int cell,int code,int side,int other,int other_side){
  int offset=25*N+NC*33+((side*3+other_side)*NC+code)*W;
#if SUPPORT_REUSE
  unsigned key=support_key(cell,code,side);int slot=support_slot(key);support_bump(0);
  if(at(SUPPORT_KEYS+slot)==key){int value=int(at(SUPPORT_VALUES+slot*3));support_bump(6);
   if(value>=0&&value<NC&&has(other,value)&&(t[offset+(value>>5)]&(1u<<(value&31)))){support_bump(1);return true;}
  }support_bump(2);
#endif
  for(int word=0;word<W;word++){support_bump(3);unsigned supported=domain(other,word)&t[offset+word];if(supported){
#if SUPPORT_REUSE
   support_put(key,word*32+__ffs(supported)-1);
#endif
   return true;
  }}return false;
 }
 __device__ void clause_bump(int index,unsigned amount=1){
#if CLAUSE_METRICS
  int row=CLAUSE_COUNTERS+2*index;unsigned old=at(row),next=old+amount;at(row)=next;if(next<old)at(row+1)++;
#endif
 }
#if WATCHED_CLAUSES
 __device__ void watch_detach(int slot){
  for(int k=0;k<2;k++){int pos=int(at(WATCH_POSITIONS+slot*2+k));
   if(pos>=0&&pos<int(at(CLAUSE_LENGTH+slot))){int cell=int(at(CLAUSE_CELL+slot*N+pos));at(WATCH_CELLS+cell)&=~(1u<<slot);}
   at(WATCH_POSITIONS+slot*2+k)=0xffffffffu;
  }
 }
 __device__ void watch_set(int slot,int first,int second){
  watch_detach(slot);int positions[2]={first,second};
  for(int k=0;k<2;k++)if(positions[k]>=0){int cell=int(at(CLAUSE_CELL+slot*N+positions[k]));
   at(WATCH_POSITIONS+slot*2+k)=positions[k];at(WATCH_CELLS+cell)|=1u<<slot;
  }
 }
 __device__ void watch_replace(int slot){watch_detach(slot);at(WATCH_PENDING)|=1u<<slot;}
 __device__ void watch_wake(int cell){unsigned bits=at(WATCH_CELLS+cell),fresh=bits&~at(WATCH_PENDING);at(WATCH_PENDING)|=bits;clause_bump(4,__popc(fresh));}
#endif
 __device__ void all_decisions(unsigned *why){zero(why);for(int level=0;level<meta(1);level++)why[level>>5]|=1u<<(level&31);}
 __device__ int highest(unsigned *why){for(int word=RW-1;word>=0;word--)if(why[word])return word*32+31-__clz(why[word]);return -1;}
 __device__ void learn(unsigned *why){
  if(!LEARNING)return;int length=0,slot=meta(19)%C;put(19,(meta(19)+1)%C);
#if WATCHED_CLAUSES
  watch_replace(slot);
#endif
  for(int level=0;level<meta(1);level++)if(why[level>>5]&(1u<<(level&31))){
   int cell=int(at(FRAME_CELL+level)),code=int(at(FRAME_CODE+level));
   if(code<0)return;at(CLAUSE_CELL+slot*N+length)=cell;at(CLAUSE_CODE+slot*N+length)=code;length++;
  }
  if(!length)return;at(CLAUSE_LENGTH+slot)=length;put(8,min(C,meta(8)+1));bump(15);
 }
 __device__ void fail(unsigned *why){
#if PROOF_TRANSPORT
  at(PROOF_META)=meta(1);unsigned prior=at(PROOF_META+1);at(PROOF_META+1)=prior+1;if(prior==0xffffffffu)at(PROOF_META+2)++;
#endif
  if(!LEARNING)all_decisions(why);
  int level=highest(why),depth=meta(1);
  if(level<0){if(LEARNING&&depth>0)bump(14,depth);put(0,2);return;}
  // Root restrictions are permanent. A failure supported by fewer current
  // decisions covers every choice of the omitted intervening decisions.
  learn(why);
  for(int word=0;word<RW;word++){
   unsigned value=why[word];if(word==(level>>5))value&=~(1u<<(level&31));
   at(FRAME_FAILURE+level*RW+word)|=value;
  }
  if(level+1<depth)bump(14,depth-level-1);
  put(1,level+1);put(6,3);
 }
 __device__ bool remove(int cell,int code,unsigned *why){
  if(!has(cell,code))return true;
  int trail=meta(2);at(TRAIL+trail)=cell*NC+code;put(2,trail+1);
  at(DOMAIN+cell*W+(code>>5))&=~(1u<<(code&31));
#if PRECISE_REASONS
  precise_store(cell,code,why);
#endif
  for(int word=0;word<RW;word++)at(REASON+cell*RW+word)|=why[word];
  put(5,1);bump(12);
#if WATCHED_CLAUSES
  watch_wake(cell);
#endif
#if GLOBAL_MATCHING
  matching_dirty(cell);
#endif
  int remaining=count(cell);
#if EVENT_PROPAGATION
  for(int side=0;side<3;side++)dirty(int(t[cell*3+side]));
  if(remaining==1)dirty(cell);
#endif
  if(remaining==0){unsigned complete[RW];zero(complete);reason(cell,complete);fail(complete);return false;}
  return true;
 }
 __device__ int find(int node){while(int(at(PARENT+node))!=node)node=int(at(PARENT+node));return node;}
 __device__ bool join(int a,int b){
  bump(16);int x=find(a),y=find(b);
  if(x==y){
   if(int(at(SIZE+x))>=TOTAL_SEGMENTS)return true;
   unsigned why[RW];zero(why);
   for(int cell=0;cell<N;cell++)if(int(at(BOARD+cell))>=0){
    int code=int(at(BOARD+cell)),number=int(t[24*N+code/3]);
    for(int part=0;part<number;part++)if(find(cell*16+part)==x){reason(cell,why);break;}
   }
   fail(why);return false;
  }
  if(at(SIZE+x)<at(SIZE+y)){int swap=x;x=y;y=swap;}
  if(INCREMENTAL_STRANDS){int pos=meta(10);at(UNION_CHILD+pos)=y;at(UNION_PARENT+pos)=x;at(UNION_SIZE+pos)=at(SIZE+x);put(10,pos+1);}
  at(PARENT+y)=x;at(SIZE+x)+=at(SIZE+y);return true;
 }
 __device__ bool trace_all(){
  for(int node=0;node<P;node++){at(PARENT+node)=node;at(SIZE+node)=1;}
  for(int cell=0;cell<N;cell++)if(int(at(BOARD+cell))>=0){
   int code=int(at(BOARD+cell));
   for(int side=0;side<3;side++){
    int other=int(t[cell*3+side]);if(other<=cell||other>=N||int(at(BOARD+other))<0)continue;
    int oc=int(at(BOARD+other)),os=int(t[3*N+cell*3+side]);
    for(int p=0;p<11;p++){
     int a=int(t[25*N+code*33+side*11+p]),b=int(t[25*N+oc*33+os*11+10-p]);
     if(a>=0&&b>=0&&!join(cell*16+a,other*16+b))return false;
    }
   }
  }return true;
 }
 __device__ bool assign(int cell){
  int code=single(cell);if(code<0||int(at(BOARD+cell))>=0)return true;
  int tile=code/3,owner=int(at(OWNER+tile));
  if(owner>=0&&owner!=cell){unsigned why[RW];zero(why);reason(cell,why);reason(owner,why);fail(why);return false;}
  int pos=meta(9);at(ASSIGN_LOG+pos)=cell;put(9,pos+1);at(BOARD+cell)=code;at(OWNER+tile)=cell;
 #if EVENT_PROPAGATION
  dirty_owner_consumers(cell,tile);
 #endif
  if(!GOLD)return true;
  if(!INCREMENTAL_STRANDS)return trace_all();
  for(int part=0;part<16;part++){at(PARENT+cell*16+part)=cell*16+part;at(SIZE+cell*16+part)=1;}
  for(int side=0;side<3;side++){
   int other=int(t[cell*3+side]);if(other<0||other>=N||int(at(BOARD+other))<0)continue;
   int oc=int(at(BOARD+other)),os=int(t[3*N+cell*3+side]);
   for(int p=0;p<11;p++){
    int a=int(t[25*N+code*33+side*11+p]),b=int(t[25*N+oc*33+os*11+10-p]);
    if(a>=0&&b>=0&&!join(cell*16+a,other*16+b))return false;
   }
  }return true;
 }
#if GLOBAL_MATCHING
 __device__ void matching_bump(int index,unsigned amount=1){int row=MATCHING_COUNTERS+2*index;unsigned old=at(row),next=old+amount;at(row)=next;if(next<old)at(row+1)++;}
 __device__ bool matching_bit(int field,int cell){return (at(field+(cell>>5))&(1u<<(cell&31)))!=0;}
 __device__ void matching_dirty(int cell){at(MATCHING_DIRTY+(cell>>5))|=1u<<(cell&31);}
 __device__ int matching_component(int cell){while(int(at(MATCHING_COMPONENTS+cell))!=cell)cell=int(at(MATCHING_COMPONENTS+cell));return cell;}
 __device__ void matching_all_dirty(){for(int word=0;word<TW;word++)at(MATCHING_DIRTY+word)=word==TW-1&&N%32?(1u<<(N%32))-1:0xffffffffu;at(MATCHING_META)=0;}
 __device__ void matching_begin(){at(MATCHING_META)=0;put(6,5);}
 __device__ void matching(){
  int stage=int(at(MATCHING_META)),cursor=int(at(MATCHING_META+1));
  if(stage==0){
   matching_bump(0);for(int word=0;word<TW;word++)at(MATCHING_AFFECTED+word)=0;
   // Domain narrowing cannot join old undirected matching components. Rebuild
   // precisely the union of dirty old components; rollback widens and dirties all.
   for(int cell=0;cell<N;cell++)if(matching_bit(MATCHING_DIRTY,cell)){
    int root=matching_component(cell);for(int other=0;other<N;other++)if(matching_component(other)==root)at(MATCHING_AFFECTED+(other>>5))|=1u<<(other&31);
   }
   for(int word=0;word<TW;word++){matching_bump(8,__popc(at(MATCHING_AFFECTED+word)));at(MATCHING_DIRTY+word)=0;}
   for(int cell=0;cell<N;cell++)if(matching_bit(MATCHING_AFFECTED,cell))at(MATCHING_COMPONENTS+cell)=cell;
   at(MATCHING_META)=1;at(MATCHING_META+1)=0;return;
  }
  if(stage==1){
   while(cursor<N&&!matching_bit(MATCHING_AFFECTED,cursor))cursor++;
   if(cursor==N){at(MATCHING_META)=2;at(MATCHING_META+1)=0;return;}
   for(int word=0;word<TW;word++)at(MATCHING_ADJ+cursor*TW+word)=0;
   for(int code=0;code<NC;code++){matching_bump(2);if(has(cursor,code))at(MATCHING_ADJ+cursor*TW+(code/3>>5))|=1u<<((code/3)&31);}
   int tile=int(at(MATCHING_CELLS+cursor));
   if(tile>=0&&!(at(MATCHING_ADJ+cursor*TW+(tile>>5))&(1u<<(tile&31)))){at(MATCHING_CELLS+cursor)=0xffffffffu;at(MATCHING_OWNERS+tile)=0xffffffffu;}
   matching_bump(1);at(MATCHING_META+1)=cursor+1;return;
  }
  if(stage==2){
   while(cursor<N&&int(at(MATCHING_CELLS+cursor))>=0)cursor++;
   if(cursor==N){at(MATCHING_META)=4;at(MATCHING_META+1)=0;return;}
   for(int word=0;word<TW;word++)at(MATCHING_SEEN+word)=0;
   for(int cell=0;cell<N;cell++)at(MATCHING_PARENT+cell)=0xffffffffu;
   at(MATCHING_SEEN+(cursor>>5))|=1u<<(cursor&31);at(MATCHING_QUEUE)=cursor;
   at(MATCHING_META+4)=0;at(MATCHING_META+5)=1;at(MATCHING_META+6)=cursor;at(MATCHING_META)=3;return;
  }
  if(stage==3){
   int head=int(at(MATCHING_META+4)),tail=int(at(MATCHING_META+5));
   if(head==tail){
    unsigned why[RW];zero(why);for(int cell=0;cell<N;cell++)if(matching_bit(MATCHING_SEEN,cell))reason(cell,why);
    matching_bump(4);fail(why);return;
   }
   int cell=int(at(MATCHING_QUEUE+head));at(MATCHING_META+4)=head+1;
   for(int word=0;word<TW;word++){
    unsigned edges=at(MATCHING_ADJ+cell*TW+word);
    while(edges){int tile=word*32+__ffs(edges)-1;edges&=edges-1;matching_bump(2);int owner=int(at(MATCHING_OWNERS+tile));
     if(owner<0){
      int here=cell,free_tile=tile;
      while(here>=0){int prior=int(at(MATCHING_CELLS+here));at(MATCHING_CELLS+here)=free_tile;at(MATCHING_OWNERS+free_tile)=here;free_tile=prior;here=int(at(MATCHING_PARENT+here));}
      matching_bump(3);at(MATCHING_META)=2;at(MATCHING_META+1)=int(at(MATCHING_META+6))+1;return;
     }
     if(!matching_bit(MATCHING_SEEN,owner)){at(MATCHING_SEEN+(owner>>5))|=1u<<(owner&31);at(MATCHING_PARENT+owner)=cell;at(MATCHING_QUEUE+tail++)=owner;at(MATCHING_META+5)=tail;}
    }
   }return;
  }
  if(stage==4){
   while(cursor<N&&!matching_bit(MATCHING_AFFECTED,cursor))cursor++;
   if(cursor==N){at(MATCHING_META)=5;at(MATCHING_META+2)=0;at(MATCHING_META+3)=0;return;}
   for(int word=0;word<TW;word++)at(MATCHING_REACH+cursor*TW+word)=0;
   at(MATCHING_REACH+cursor*TW+(cursor>>5))|=1u<<(cursor&31);
   for(int word=0;word<TW;word++){
    unsigned edges=at(MATCHING_ADJ+cursor*TW+word);
    while(edges){int tile=word*32+__ffs(edges)-1;edges&=edges-1;int owner=int(at(MATCHING_OWNERS+tile));
     at(MATCHING_REACH+cursor*TW+(owner>>5))|=1u<<(owner&31);
     int a=matching_component(cursor),b=matching_component(owner);if(a!=b)at(MATCHING_COMPONENTS+max(a,b))=min(a,b);
    }
   }at(MATCHING_META+1)=cursor+1;return;
  }
  if(stage==5){
   int k=int(at(MATCHING_META+2)),row=int(at(MATCHING_META+3));
   while(k<N&&!matching_bit(MATCHING_AFFECTED,k))k++;
   if(k==N){at(MATCHING_META)=6;at(MATCHING_META+1)=0;at(MATCHING_META+7)=0;return;}
   while(row<N&&!matching_bit(MATCHING_AFFECTED,row))row++;
   if(row==N){at(MATCHING_META+2)=k+1;at(MATCHING_META+3)=0;return;}
   if(at(MATCHING_REACH+row*TW+(k>>5))&(1u<<(k&31)))for(int word=0;word<TW;word++){at(MATCHING_REACH+row*TW+word)|=at(MATCHING_REACH+k*TW+word);matching_bump(5);}
   at(MATCHING_META+2)=k;at(MATCHING_META+3)=row+1;return;
  }
  // A perfect matching supports edge(c,t) iff its matched owner can reach c
  // in the alternating graph. A failed return path exposes a closed Hall set.
  int tile=int(at(MATCHING_META+7));
  while(cursor<N&&(!matching_bit(MATCHING_AFFECTED,cursor)||tile>=N)){cursor++;tile=0;}
  if(cursor==N){if(meta(5))reset_scan();else fixed_point();return;}
  at(MATCHING_META+1)=cursor;at(MATCHING_META+7)=tile+1;
  if(!(at(MATCHING_ADJ+cursor*TW+(tile>>5))&(1u<<(tile&31))))return;
  matching_bump(6);int owner=int(at(MATCHING_OWNERS+tile));
  if(at(MATCHING_REACH+owner*TW+(cursor>>5))&(1u<<(cursor&31)))return;
  unsigned why[RW];zero(why);for(int cell=0;cell<N;cell++)if(at(MATCHING_REACH+owner*TW+(cell>>5))&(1u<<(cell&31)))reason(cell,why);
  for(int rotation=0;rotation<3;rotation++)if(has(cursor,tile*3+rotation)){matching_bump(7);if(!remove(cursor,tile*3+rotation,why))return;}
 }
#endif
 __device__ void fixed_point(){
#if GLOBAL_MATCHING
  for(int word=0;word<TW;word++)if(at(MATCHING_DIRTY+word)){matching_begin();return;}
#endif
  put(6,2);
 }
 __device__ void reset_scan(){
  put(3,0);put(4,0);put(5,0);put(6,0);put(7,0);
#if WATCHED_CLAUSES
  at(WATCH_PENDING)=0xffffffffu;
#endif
#if EVENT_PROPAGATION
  // Rollback changes both ownership and domains; conservatively reschedule all.
  put(3,N);for(int cell=0;cell<N;cell++)dirty(cell);
#endif
 }
 __device__ void restore(int level){
#if GLOBAL_MATCHING
  matching_all_dirty();
#endif
  int stop=int(at(FRAME_TRAIL+level));
  for(int pos=meta(2)-1;pos>=stop;pos--){int encoded=int(at(TRAIL+pos)),cell=encoded/NC,code=encoded%NC;
#if PRECISE_REASONS
   precise_invalidate(cell,code);
#endif
   at(DOMAIN+cell*W+(code>>5))|=1u<<(code&31);
  }
  put(2,stop);
  if(INCREMENTAL_STRANDS){
   stop=int(at(FRAME_UNION+level));
   for(int pos=meta(10)-1;pos>=stop;pos--){int child=int(at(UNION_CHILD+pos)),parent=int(at(UNION_PARENT+pos));at(PARENT+child)=child;at(SIZE+parent)=at(UNION_SIZE+pos);}
   put(10,stop);
  }
  stop=int(at(FRAME_ASSIGN+level));
  for(int pos=meta(9)-1;pos>=stop;pos--){int cell=int(at(ASSIGN_LOG+pos)),code=int(at(BOARD+cell));at(OWNER+code/3)=0xffffffffu;at(BOARD+cell)=0xffffffffu;}
  put(9,stop);
  for(int cell=0;cell<N;cell++)for(int word=0;word<RW;word++)at(REASON+cell*RW+word)=at(FRAME_REASON+(level*N+cell)*RW+word);
 }
 __device__ void choose_from_frame(int level){
  int cell=int(at(FRAME_CELL+level)),chosen=-1;
  for(int word=0;word<W;word++){unsigned remaining=at(FRAME_REMAINING+level*W+word);if(remaining){chosen=word*32+__ffs(remaining)-1;at(FRAME_REMAINING+level*W+word)&=~(1u<<(chosen&31));break;}}
  if(chosen<0){
   unsigned why[RW];for(int word=0;word<RW;word++)why[word]=at(FRAME_FAILURE+level*RW+word);
   put(1,level);fail(why);return;
  }
  at(FRAME_CODE+level)=chosen;put(1,level+1);bump(11);reset_scan();
  unsigned why[RW];zero(why);why[level>>5]|=1u<<(level&31);
  for(int code=0;code<NC;code++)if(code!=chosen&&!remove(cell,code,why))return;
 }
 __device__ void backtrack(){
  int depth=meta(1);if(depth==0){put(0,2);return;}
  int level=depth-1;restore(level);choose_from_frame(level);
 }
 __device__ void branch(){
  int best=-1,minimum=NC+1;
  for(int cell=0;cell<N;cell++){int size=count(cell);if(size==0){unsigned why[RW];zero(why);reason(cell,why);fail(why);return;}if(size>1&&size<minimum){best=cell;minimum=size;}}
  if(best<0){put(0,1);bump(18);return;}
  int level=meta(1);at(FRAME_CELL+level)=best;at(FRAME_TRAIL+level)=meta(2);at(FRAME_UNION+level)=meta(10);at(FRAME_ASSIGN+level)=meta(9);
  for(int word=0;word<W;word++)at(FRAME_REMAINING+level*W+word)=domain(best,word);
  for(int cell=0;cell<N;cell++)for(int word=0;word<RW;word++)at(FRAME_REASON+(level*N+cell)*RW+word)=at(REASON+cell*RW+word);
  for(int word=0;word<RW;word++)at(FRAME_FAILURE+level*RW+word)=at(REASON+best*RW+word);
  choose_from_frame(level);
 }
 __device__ void clause(){
  int index=meta(7);
#if WATCHED_CLAUSES
  unsigned ready=at(WATCH_PENDING)&(meta(8)>=C?0xffffffffu:((1u<<meta(8))-1));
  if(!ready||!LEARNING){put(6,1);return;}
  index=__ffs(ready)-1;at(WATCH_PENDING)&=~(1u<<index);
#else
  if(index>=meta(8)||!LEARNING){put(6,1);return;}
#endif
  put(7,index+1);int length=int(at(CLAUSE_LENGTH+index)),unknown=0,cell=-1,code=-1,first=-1;unsigned why[RW];zero(why);
#if WATCHED_CLAUSES
  int open=0;
  for(int k=0;k<2;k++){int pos=int(at(WATCH_POSITIONS+index*2+k));if(pos<0||pos>=length)continue;
   int c=int(at(CLAUSE_CELL+index*N+pos)),v=int(at(CLAUSE_CODE+index*N+pos));clause_bump(1);
   if(!has(c,v)){clause_bump(2);return;}
   if(single(c)!=v)open++;
  }
  if(open>=2){clause_bump(2);return;}
#endif
  clause_bump(3);
  for(int pos=0;pos<length;pos++){
   int c=int(at(CLAUSE_CELL+index*N+pos)),v=int(at(CLAUSE_CODE+index*N+pos));clause_bump(0);
   if(!has(c,v)){
#if WATCHED_CLAUSES
    watch_set(index,pos,-1);
#endif
    return;
   }
   if(single(c)!=v){unknown++;cell=c;code=v;if(unknown>1){
#if WATCHED_CLAUSES
    watch_set(index,first,pos);
#endif
    return;
   }first=pos;}else reason(c,why);
  }
#if WATCHED_CLAUSES
  watch_set(index,first,-1);
#endif
  if(unknown==0){fail(why);return;}
  if(remove(cell,code,why))bump(17);
 }

#if JOINT_SUPPORT
 // Hall's condition on a candidate's distinct neighboring cells. Multiple
 // seams to one neighbor constrain the SAME orientation, not separate pieces.
 __device__ void joint_bump(int index){
  int row=JOINT_COUNTERS+2*index;unsigned old=at(row),next=old+1;at(row)=next;if(next<old)at(row+1)++;
 }
 __device__ void joint_support(int cell,int code){
  joint_bump(0);int neighbors[3],count=0;unsigned tiles[3][TW];
#if SUPPORT_REUSE
  int witnesses[3][3],sizes[3]={0,0,0};
#endif
  for(int side=0;side<3;side++){
   int other=int(t[cell*3+side]);if(other<0)continue;
   int other_side=int(t[3*N+cell*3+side]);
   if(other==cell){
    if(t[6*N+code*3+side]!=t[15*N+code*3+other_side]){
     unsigned why[RW];zero(why);joint_bump(2);remove(cell,code,why);return;
    }continue;
   }
   bool present=false;for(int i=0;i<count;i++)if(neighbors[i]==other)present=true;
   if(!present)neighbors[count++]=other;
  }
#if SUPPORT_REUSE
  unsigned cache_key=support_key(cell,code,3);int cache_slot=support_slot(cache_key);support_bump(0);
  if(at(SUPPORT_KEYS+cache_slot)==cache_key){bool valid=true;int previous[3];
   for(int i=0;i<count&&valid;i++){int value=int(at(SUPPORT_VALUES+cache_slot*3+i));support_bump(6);
    if(value<0||value>=NC||value/3==code/3||!has(neighbors[i],value)){valid=false;break;}
    previous[i]=value/3;for(int j=0;j<i;j++)if(previous[j]==previous[i])valid=false;
    for(int side=0;side<3;side++)if(int(t[cell*3+side])==neighbors[i]){
     int os=int(t[3*N+cell*3+side]);if(t[6*N+code*3+side]!=t[15*N+value*3+os])valid=false;
    }
   }
   if(valid){support_bump(1);return;}
  }support_bump(2);
#endif
  for(int i=0;i<count;i++){
   int other=neighbors[i],distinct=0;for(int word=0;word<TW;word++)tiles[i][word]=0;
   // A set with D distinct tiles already satisfies every Hall subset that
   // contains it (subset size <= D). Smaller sets must remain exact.
   for(int word=0;word<W&&distinct<count;word++){
    joint_bump(3);support_bump(3);unsigned supported=domain(other,word);
    for(int side=0;side<3;side++)if(int(t[cell*3+side])==other){
     int other_side=int(t[3*N+cell*3+side]);
     supported&=t[25*N+NC*33+((side*3+other_side)*NC+code)*W+word];
    }
    while(supported&&distinct<count){int other_code=word*32+__ffs(supported)-1; supported&=supported-1;
     joint_bump(4);int tile=other_code/3;unsigned bit=1u<<(tile&31);
     if(tile!=code/3&&!(tiles[i][tile>>5]&bit)){tiles[i][tile>>5]|=bit;
#if SUPPORT_REUSE
      witnesses[i][distinct]=other_code;sizes[i]=distinct+1;
#endif
      distinct++;}
    }
   }
  }
  for(int subset=1;subset<(1<<count);subset++){
   joint_bump(1);int need=__popc(unsigned(subset)),available=0;
   for(int word=0;word<TW&&available<need;word++){
    unsigned combined=0;for(int i=0;i<count;i++)if(subset&(1<<i))combined|=tiles[i][word];
    available+=__popc(combined);
   }
   if(available<need){
    // The candidate itself is the value being refuted, not an active decision
    // assumption. Neighbor domain-removal reasons suffice under this lane root.
    unsigned why[RW];zero(why);for(int i=0;i<count;i++)if(subset&(1<<i)){
#if PRECISE_REASONS
     seam_reason(cell,neighbors[i],code,code/3,why);
#else
     reason(neighbors[i],why);
#endif
    }
    joint_bump(2);remove(cell,code,why);return;
   }
  }
#if SUPPORT_REUSE
  // Hall passed on exact small sets / safely truncated large sets. Find a
  // concrete injective neighbor assignment to certify the next cache hit.
  for(int a=0;a<(count>0?sizes[0]:1);a++)for(int b=0;b<(count>1?sizes[1]:1);b++)for(int c=0;c<(count>2?sizes[2]:1);c++){
   int x=count>0?witnesses[0][a]:-1,y=count>1?witnesses[1][b]:-1,z=count>2?witnesses[2][c]:-1;
   if((count>1&&x/3==y/3)||(count>2&&(x/3==z/3||y/3==z/3)))continue;
   support_put(cache_key,x,y,z);return;
  }
#endif
 }
#endif
 __device__ int prepare_candidate(){
  int cell=meta(3),code=meta(4);
#if EVENT_PROPAGATION
  if(cell>=N){
   cell=pop_dirty();code=0;put(3,cell);put(4,0);
   if(cell>=N){
    // Domain changes can activate a previously scanned learned clause. Drain
    // those clauses before declaring a fixed point, even when no cell is dirty.
    if(meta(5)){put(5,0);put(6,0);put(7,0);}else fixed_point();return -1;
   }
  }
#else
  if(cell>=N){if(meta(5)){reset_scan();return -1;}fixed_point();return -1;}
#endif
  if(!assign(cell))return -1;
  // One existing candidate per primitive, with bounded word-level seam tests.
  while(code<NC&&!has(cell,code))code++;
  if(code>=NC){
#if EVENT_PROPAGATION
   put(3,N);
#else
   put(3,cell+1);
#endif
   put(4,0);return -1;
  }
  put(4,code+1);bump(13);
  int owner=int(at(OWNER+code/3));unsigned why[RW];zero(why);
  if(owner>=0&&owner!=cell){reason(owner,why);remove(cell,code,why);return -1;}
  return cell*NC+code;
 }
 __device__ void scan(){
  int candidate=prepare_candidate();if(candidate<0)return;int cell=candidate/NC,code=candidate%NC;unsigned why[RW];zero(why);
  for(int side=0;side<3;side++){
   int other=int(t[cell*3+side]);
   if(other<0){if(t[6*N+code*3+side]){remove(cell,code,why);return;}continue;}
   int other_side=int(t[3*N+cell*3+side]);
   if(!seam_supported(cell,code,side,other,other_side)){
#if PRECISE_REASONS
    seam_reason(cell,other,code,-1,why);
#else
    reason(other,why);
#endif
    remove(cell,code,why);return;
   }
  }
#if JOINT_SUPPORT
  joint_support(cell,code);
#endif
 }
#if COOPERATIVE
 __device__ bool cooperative_seam(int cell,int code,int side,int other,int other_side,int rank){
  int offset=25*N+NC*33+((side*3+other_side)*NC+code)*W;bool cached=false;
#if SUPPORT_REUSE
  unsigned key=support_key(cell,code,side);int slot=support_slot(key);
  if(rank==0){support_bump(0);if(at(SUPPORT_KEYS+slot)==key){int value=int(at(SUPPORT_VALUES+slot*3));support_bump(6);
   cached=value>=0&&value<NC&&has(other,value)&&(t[offset+(value>>5)]&(1u<<(value&31)));}
   if(cached)support_bump(1);else support_bump(2);
  }
  cached=__shfl_sync(0xffffffffu,int(cached),0)!=0;if(cached)return true;
#endif
  // Every word is read once in parallel; unlike scalar early exit, charge all
  // inspected words. The lowest witness matches scalar deterministic ordering.
  unsigned supported=rank<W?domain(other,rank)&t[offset+rank]:0;
  int best=supported?rank*32+__ffs(supported)-1:NC;
  for(int shift=16;shift;shift>>=1)best=min(best,__shfl_down_sync(0xffffffffu,best,shift));
  best=__shfl_sync(0xffffffffu,best,0);
  if(rank==0){for(int word=0;word<W;word++)support_bump(3);
#if SUPPORT_REUSE
   if(best<NC)support_put(key,best);
#endif
  }return best<NC;
 }
 __device__ void scan_cooperative(int rank){
  int candidate=rank==0?prepare_candidate():-1;candidate=__shfl_sync(0xffffffffu,candidate,0);__syncwarp();
  if(candidate<0)return;int cell=candidate/NC,code=candidate%NC;
  for(int side=0;side<3;side++){
   int other=int(t[cell*3+side]);
   if(other<0){if(t[6*N+code*3+side]){if(rank==0){unsigned why[RW];zero(why);remove(cell,code,why);}return;}continue;}
   int other_side=int(t[3*N+cell*3+side]);
   if(!cooperative_seam(cell,code,side,other,other_side,rank)){
    if(rank==0){unsigned why[RW];zero(why);
#if PRECISE_REASONS
     seam_reason(cell,other,code,-1,why);
#else
     reason(other,why);
#endif
     remove(cell,code,why);
    }return;
   }
  }
#if JOINT_SUPPORT
  if(rank==0)joint_support(cell,code);
#endif
 }
#endif

};
extern "C" __global__ void domain_search(unsigned *state,const unsigned *tables,int lanes,int work){
#if COOPERATIVE
 int lane=(blockIdx.x*blockDim.x+threadIdx.x)/32,rank=threadIdx.x&31;
#else
 int lane=blockIdx.x*blockDim.x+threadIdx.x,rank=0;
#endif
 if(lane>=lanes)return;Context ctx{state,tables,lanes,lane};
 for(int step=0;step<work;step++){
  int status=rank==0?ctx.meta(0):0,phase=rank==0?ctx.meta(6):0;
#if COOPERATIVE
  status=__shfl_sync(0xffffffffu,status,0);phase=__shfl_sync(0xffffffffu,phase,0);
#endif
  if(status!=0)return;
#if COOPERATIVE
  if(phase==1)ctx.scan_cooperative(rank);else
#endif
  if(rank==0){
   if(phase==0)ctx.clause();else if(phase==1)ctx.scan();else if(phase==2)ctx.branch();
#if GLOBAL_MATCHING
   else if(phase==5)ctx.matching();
#endif
   else ctx.backtrack();
  }
#if COOPERATIVE
  __syncwarp();
#endif
 }
}
extern "C" __global__ void domain_ack(unsigned *state,const int *ids,int lanes,int count){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;int lane=ids[index];Context ctx{state,nullptr,lanes,lane};
 // This is an enumeration acknowledgement, not an infeasibility certificate.
 // The exact returned board is retained by the host before its leaf is retired.
 if(ctx.meta(0)!=1)return;ctx.put(0,0);ctx.put(6,4);
 int level=ctx.meta(1)-1;
 if(level<0){ctx.put(0,2);return;}
 for(int parent=0;parent<level;parent++)ctx.at(FRAME_FAILURE+level*RW+(parent>>5))|=1u<<(parent&31);
}

// Root preparation is cooperative: every dispatch has the same primitive bound
// as ordinary search, and stops before creating the first decision frame.
extern "C" __global__ void domain_propagate(unsigned *state,const unsigned *tables,int lanes,int work){
#if COOPERATIVE
 int lane=(blockIdx.x*blockDim.x+threadIdx.x)/32,rank=threadIdx.x&31;
#else
 int lane=blockIdx.x*blockDim.x+threadIdx.x,rank=0;
#endif
 if(lane>=lanes)return;Context ctx{state,tables,lanes,lane};
 for(int step=0;step<work;step++){
  int status=rank==0?ctx.meta(0):0,phase=rank==0?ctx.meta(6):0;
#if COOPERATIVE
  status=__shfl_sync(0xffffffffu,status,0);phase=__shfl_sync(0xffffffffu,phase,0);
#endif
  if(status!=0||phase==2)return;
#if COOPERATIVE
  if(phase==1)ctx.scan_cooperative(rank);else
#endif
  if(rank==0){if(phase==0)ctx.clause();else if(phase==1)ctx.scan();
#if GLOBAL_MATCHING
   else if(phase==5)ctx.matching();
#endif
  }
#if COOPERATIVE
  __syncwarp();
#endif
 }
}

// A shared record has already been proved against the original root by the CPU
// verifier. Refresh at most eight such records; ordinary local learning may
// later evict them from the bounded ring. Never accept an empty core here.
extern "C" __global__ void domain_share(unsigned *state,const unsigned *records,unsigned *counts,int lanes,int number){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes)return;Context ctx{state,nullptr,lanes,lane};
 unsigned inserted=0,comparisons=0;
 if(ctx.meta(0)==0)for(int index=0;index<number;index++){
  const unsigned *record=records+index*65;int length=int(record[0]);bool found=false;
  for(int prior=0;prior<ctx.meta(8)&&!found;prior++){
   comparisons++;if(int(ctx.at(CLAUSE_LENGTH+prior))!=length)continue;bool equal=true;
   // Local learned literals are in decision order; verified ones are sorted.
   for(int p=0;p<length&&equal;p++){
    bool member=false;for(int q=0;q<length;q++){comparisons++;
     if(ctx.at(CLAUSE_CELL+prior*N+q)==record[1+p*2]&&ctx.at(CLAUSE_CODE+prior*N+q)==record[2+p*2]){member=true;break;}
    }if(!member)equal=false;
   }if(equal)found=true;
  }
  if(found)continue;int slot=ctx.meta(19)%C;
#if WATCHED_CLAUSES
  ctx.watch_replace(slot);
#endif
  ctx.put(19,(slot+1)%C);ctx.put(8,min(C,ctx.meta(8)+1));ctx.at(CLAUSE_LENGTH+slot)=length;
  for(int pos=0;pos<length;pos++){ctx.at(CLAUSE_CELL+slot*N+pos)=record[1+pos*2];ctx.at(CLAUSE_CODE+slot*N+pos)=record[2+pos*2];}inserted++;
 }
 if(inserted&&ctx.meta(6)<=2){ctx.put(6,0);ctx.put(7,0);}
 // Matching owns its resumable scratch until completion. Queue a subsequent
 // full clause pass rather than claiming a fixed point with an unscanned import.
 if(inserted&&ctx.meta(6)==5)ctx.put(5,1);
 counts[lane]=inserted;counts[lanes+lane]=comparisons;
}
