__device__ unsigned rnd(unsigned &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
__device__ int legal(const unsigned*t,const unsigned*b,int n,int r,int f,int c,int lane){return t[25*n+f+c*3*n+b[c*r+lane]];}
__device__ int edge(const unsigned*t,const unsigned*b,int n,int r,int c,int side,int lane){
 unsigned nb=t[18*n+c*3+side];if(nb>=n)return 0;
 return t[b[c*r+lane]*3+side]==t[9*n+b[nb*r+lane]*3+t[21*n+c*3+side]];
}
__device__ int total(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int lane){
 int value=0;for(int c=0;c<n;c++){value+=legal(t,b,n,r,f,c,lane)*(edges+1);
 for(int s=0;s<3;s++){unsigned nb=t[18*n+c*3+s];if(nb<n&&nb>c)value+=edge(t,b,n,r,c,s,lane);}}return value;
}
// Score only changed domain terms and incident seams; old board stays intact.
__device__ int change(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int a,int d,unsigned na,unsigned nd,int lane){
 int delta=0;
 for(int which=0;which<(a==d?1:2);which++){
 int c=which?d:a;unsigned old=b[c*r+lane],next=which?nd:na;int base=25*n+f+c*3*n;
 delta+=(int(t[base+next])-int(t[base+old]))*(edges+1);
 for(int side=0;side<3;side++){
 unsigned nb=t[18*n+c*3+side];if(nb>=n||(which&&nb==a))continue;
 unsigned opposite=t[21*n+c*3+side],other=b[nb*r+lane];
 unsigned replacement=nb==a?na:(nb==d?nd:other);
 delta+=int(t[next*3+side]==t[9*n+replacement*3+opposite])-int(t[old*3+side]==t[9*n+other*3+opposite]);
 }}return delta;
}

__device__ unsigned partner(const unsigned*t,int n,int f,unsigned code,unsigned port){return t[25*n+f+3*n*n+1+code*33+port];}
// Count only closed proper components touching a moved cell (a<0: all cells).
// Among touched segments the smallest (cell, local endpoint pair) counts once.
__device__ int loops(const unsigned*t,const unsigned*b,int n,int r,int f,int a,int d,int lane,int total){
 if(!total)return 0;int result=0;
 int roots=a<0?n:(a==d?1:2);
 for(int root=0;root<roots;root++) {int start=a<0?root:(root?d:a);unsigned code=b[start*r+lane];
 for(int port=0;port<33;port++) {unsigned pair=partner(t,n,f,code,port);if(pair>=33||pair<unsigned(port))continue;
 int cell=start,incoming=port,key=start*33+port;
 for(int count=1;count<=total;count++) {
   unsigned outgoing=partner(t,n,f,b[cell*r+lane],incoming);if(outgoing>=33)break;
   int local=cell*33+min(incoming,int(outgoing));if((a<0||cell==a||cell==d)&&local<key)break;
   int side=outgoing/11;unsigned next=t[18*n+cell*3+side];if(next>=unsigned(n))break;
   incoming=t[21*n+cell*3+side]*11+10-outgoing%11;cell=next;
   if(cell==start&&incoming==port){if(count<total)++result;break;}
 }
 }}return result;
}
// Round26 current-board cache: two immutable physical-strand labels/word.
__device__ void ibump(unsigned*m,int r,int lane,int which,unsigned value=1){
 unsigned row=29+2*which,old=m[row*r+lane];m[row*r+lane]=old+value;
 if(m[row*r+lane]<old)m[(row+1)*r+lane]++;
}
__device__ int mtotal(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int edges,int lane,int flags){
 if(flags&192){ibump(m,r,lane,1,n);ibump(m,r,lane,2,edges);}return total(t,b,n,r,f,edges,lane);
}
__device__ int mloops(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int lane,int total,int flags){
 if(!(flags&192))return loops(t,b,n,r,f,-1,-1,lane,total);if(!total)return 0;
 int result=0;unsigned starts=0,steps=0;
 for(int start=0;start<n;start++){unsigned code=b[start*r+lane];
  for(unsigned port=0;port<33;port++){unsigned pair=partner(t,n,f,code,port);if(pair>=33||pair<port)continue;starts++;
   int cell=start,incoming=port,key=start*33+port;
   for(int count=1;count<=total;count++){unsigned outgoing=partner(t,n,f,b[cell*r+lane],incoming);steps++;if(outgoing>=33)break;
    int local=cell*33+min(incoming,int(outgoing));if(local<key)break;unsigned next=t[18*n+cell*3+outgoing/11];if(next>=unsigned(n))break;
    incoming=t[21*n+cell*3+outgoing/11]*11+10-outgoing%11;cell=next;if(cell==start&&incoming==int(port)){if(count<total)result++;break;}
   }
  }
 }ibump(m,r,lane,4,starts);ibump(m,r,lane,5,steps);return result;
}
__device__ unsigned inode(const unsigned*t,int n,int f,unsigned code,unsigned port){
 return t[25*n+f+3*n*n+1+3*n*33+code*33+port];
}
__device__ unsigned ilabel(const unsigned*m,int r,int lane,unsigned node){return (m[(45+node/2)*r+lane]>>(16*(node%2)))&65535u;}
__device__ void iset(unsigned*m,int r,int lane,unsigned node,unsigned value){
 unsigned row=45+node/2,shift=16*(node%2),mask=65535u<<shift;
 m[row*r+lane]=(m[row*r+lane]&~mask)|(value<<shift);
}
__device__ int selected(int cell,int count,const int*cells){for(int i=0;i<count;i++)if(cells[i]==cell)return i;return -1;}
__device__ unsigned virtual_code(const unsigned*b,int r,int lane,int cell,int count,const int*cells,const int*old,bool previous){
 int index=previous?selected(cell,count,cells):-1;return index<0?b[cell*r+lane]:unsigned(old[index]);
}
__device__ int ichange(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int edges,int lane,int count,const int*cells,const int*next){
 int delta=0;unsigned seams=0;
 for(int i=0;i<count;i++){int cell=cells[i];unsigned old=b[cell*r+lane],code=next[i];int base=25*n+f+cell*3*n;
  delta+=(int(t[base+code])-int(t[base+old]))*(edges+1);
  for(int side=0;side<3;side++){unsigned nb=t[18*n+cell*3+side];int index=selected(nb,count,cells);
   if(nb>=unsigned(n)||(index>=0&&index<i))continue;
   unsigned opposite=t[21*n+cell*3+side],other=b[nb*r+lane],replacement=index<0?other:unsigned(next[index]);
   delta+=int(t[code*3+side]==t[9*n+replacement*3+opposite])-int(t[old*3+side]==t[9*n+other*3+opposite]);seams++;
  }
 }
 ibump(m,r,lane,1,2*count);ibump(m,r,lane,2,2*seams);return delta;
}
__device__ int itouched(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int lane,int count,const int*cells){
 unsigned seen[96];int size=0;unsigned reads=0;
 for(int i=0;i<count;i++){unsigned code=b[cells[i]*r+lane];
  for(unsigned port=0;port<33;port++){unsigned pair=partner(t,n,f,code,port);if(pair>=33||pair<port)continue;
   unsigned label=ilabel(m,r,lane,inode(t,n,f,code,port));reads++;if(!label)continue;
   bool found=false;for(int j=0;j<size;j++)if(seen[j]==label)found=true;if(!found)seen[size++]=label;
  }
 }ibump(m,r,lane,3,reads);return size;
}
// mode 0 counts new touched loops; 1 also fills their labels; 2 clears old
// touched closed loops using the pre-proposal virtual board. count<0 is rebuild.
__device__ int icomponents(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int lane,int total,int count,const int*cells,const int*old,int mode){
 if(!total)return 0;int result=0;unsigned starts=0,steps=0,reads=0;
 int roots=count<0?n:count;bool previous=mode==2;
 for(int root=0;root<roots;root++){int start=count<0?root:cells[root];unsigned code=virtual_code(b,r,lane,start,count,cells,old,previous);
  for(unsigned port=0;port<33;port++){unsigned pair=partner(t,n,f,code,port);if(pair>=33||pair<port)continue;
   if(previous){reads++;if(!ilabel(m,r,lane,inode(t,n,f,code,port)))continue;}
   starts++;int cell=start,incoming=port,key=start*33+port;unsigned smallest=0xffffffffu;bool closed=false;
   for(int length=1;length<=total;length++){
    unsigned here=virtual_code(b,r,lane,cell,count,cells,old,previous),outgoing=partner(t,n,f,here,incoming);steps++;if(outgoing>=33)break;
    int local=cell*33+min(incoming,int(outgoing));if(!previous&&(count<0||selected(cell,count,cells)>=0)&&local<key)break;
    smallest=min(smallest,inode(t,n,f,here,incoming));
    unsigned nb=t[18*n+cell*3+outgoing/11];if(nb>=unsigned(n))break;
    incoming=t[21*n+cell*3+outgoing/11]*11+10-outgoing%11;cell=nb;
    if(cell==start&&incoming==int(port)){closed=length<total;break;}
   }
   if(!closed)continue;result++;
   if(mode){cell=start;incoming=port;
    for(int length=1;length<=total;length++){
     unsigned here=virtual_code(b,r,lane,cell,count,cells,old,previous),outgoing=partner(t,n,f,here,incoming);steps++;
     iset(m,r,lane,inode(t,n,f,here,incoming),previous?0:smallest+1);
     unsigned nb=t[18*n+cell*3+outgoing/11];incoming=t[21*n+cell*3+outgoing/11]*11+10-outgoing%11;cell=nb;
     if(cell==start&&incoming==int(port))break;
    }
   }
  }
 }ibump(m,r,lane,3,reads);ibump(m,r,lane,4,starts);ibump(m,r,lane,5,steps);return result;
}
__device__ void irebuild(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int lane,int total){
 for(int word=0;word<(total+1)/2;word++)m[(45+word)*r+lane]=0;
 int cells[6],old[6];icomponents(t,b,m,n,r,f,lane,total,-1,cells,old,1);ibump(m,r,lane,7);
}

// Heuristic placement scores never exclude search work. The complete proposed
// permutation is still evaluated by total()/loops() before acceptance.
__device__ int local_candidate(const unsigned*t,const unsigned*b,int n,int r,int f,int edges,int cell,unsigned code,int lane){
 int value=int(t[25*n+f+cell*3*n+code])*(edges+1);
 for(int side=0;side<3;side++){unsigned nb=t[18*n+cell*3+side];if(nb<unsigned(n))
  value+=int(t[code*3+side]==t[9*n+b[nb*r+lane]*3+t[21*n+cell*3+side]]);}
 return value;
}
__device__ void guided_cycle(const unsigned*t,const unsigned*b,unsigned*m,int n,int r,int f,int edges,int gold_total,int score,int lane,
                             unsigned &random,int count,int*cells,int*old,int*next){
 m[18*r+lane]++;unsigned start=rnd(random)%f;int first=t[25*n+start];
 bool closed_obstruction=gold_total>0&&score/(gold_total+1)==n*(edges+1)+edges&&score<(n*(edges+1)+edges)*(gold_total+1)+gold_total;
 for(int offset=0;offset<f;offset++){
  int cell=t[25*n+(start+offset)%f];bool bad=!legal(t,b,n,r,f,cell,lane);
  for(int side=0;side<3;side++)if(t[18*n+cell*3+side]<unsigned(n)&&!edge(t,b,n,r,cell,side,lane))bad=true;
  if(!bad&&closed_obstruction)for(int port=0;port<33;port++)if(partner(t,n,f,b[cell*r+lane],port)<33){bad=true;break;}
  if(bad){first=cell;m[20*r+lane]++;break;}
 }
 cells[0]=first;old[0]=b[first*r+lane];
 for(int i=0;i<count-1;i++){
  unsigned begin=rnd(random)%f,rotation=rnd(random)%3;int inspected=0,best_value=-1,donor=-1,chosen_code=0;
  for(int offset=0;offset<f&&inspected<16;offset++){
   int candidate=t[25*n+(begin+offset)%f];bool used=false;for(int j=0;j<=i;j++)if(cells[j]==candidate)used=true;
   if(used)continue;inspected++;unsigned tile=b[candidate*r+lane]/3;
   for(int k=0;k<3;k++){unsigned code=tile*3+(rotation+k)%3;m[19*r+lane]++;
    int value=local_candidate(t,b,n,r,f,edges,cells[i],code,lane);
    if(value>best_value){best_value=value;donor=candidate;chosen_code=code;}
   }
  }
  cells[i+1]=donor;old[i+1]=b[donor*r+lane];next[i]=chosen_code;
 }
 unsigned rotation=rnd(random)%3;int best_value=-1;
 for(int k=0;k<3;k++){unsigned code=3*(old[0]/3)+(rotation+k)%3;m[19*r+lane]++;
  int value=local_candidate(t,b,n,r,f,edges,cells[count-1],code,lane);
  if(value>best_value){best_value=value;next[count-1]=code;}
 }
}
// Adaptive temperatures are bounded host policy state, not a pruning rule.
__device__ float exchange_heat(const unsigned*m,int r,int lane,int flags){
 return (flags&32)?__uint_as_float(m[28*r+lane]):(.2f*powf(1.5f,float(lane%8)));
}
extern "C" __global__ void sample_search(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int steps,int phase,int edges,int gold_total,int flags){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;unsigned random=m[lane];
 if(phase==3){if(flags&64){irebuild(t,b,m,n,r,f,lane,gold_total);m[r+lane]=mtotal(t,b,m,n,r,f,edges,lane,flags)*(gold_total+1)+gold_total-mloops(t,b,m,n,r,f,lane,gold_total,flags);}return;}
 if(phase==2){if((flags&64)&&(steps<0||lane==steps)){irebuild(t,b,m,n,r,f,lane,gold_total);ibump(m,r,lane,6);}return;}
 if(phase==0){
  for(int c=0;c<n;c++)b[c*r+lane]=t[24*n+c];
  for(int count=f;count>1;count--){int a=t[25*n+count-1],d=t[25*n+rnd(random)%count];unsigned old=b[a*r+lane];b[a*r+lane]=b[d*r+lane];b[d*r+lane]=old;}
  for(int i=0;i<f;i++){int c=t[25*n+i];b[c*r+lane]=3*(b[c*r+lane]/3)+rnd(random)%3;}
  for(int row=3;row<((flags&192)?45+((flags&64)?(gold_total+1)/2:0):((flags&32)?29:((flags&16)?28:((flags&8)?22:18))));row++)m[row*r+lane]=0;
  m[r+lane]=m[2*r+lane]=mtotal(t,b,m,n,r,f,edges,lane,flags)*(gold_total+1)+gold_total-mloops(t,b,m,n,r,f,lane,gold_total,flags);
  if(flags&32)m[28*r+lane]=__float_as_uint(.2f*powf(1.5f,float(lane%8)));
  for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
  if(flags&64)irebuild(t,b,m,n,r,f,lane,gold_total);
 }else if(f>0){
  int score=m[r+lane],bestscore=m[2*r+lane];unsigned iteration=m[3*r+lane];
  for(int step=0;step<steps;step++){
   int cells[6],old[6],next[6],count=1;
   bool guided=(flags&8)&&rnd(random)%4!=0;
   if(guided){count=f>=3?3+rnd(random)%(min(6,f)-2):f;m[15*r+lane]++;}
   else if((flags&1)&&f>=3&&rnd(random)%4!=0){count=3+rnd(random)%(min(6,f)-2);m[15*r+lane]++;}
   else if(f>=2)count=1+rnd(random)%2;
   // Select distinct positions, then relocate the displaced pieces around
   // the closed chain. Every proposal remains a physical-piece permutation.
   if(guided)guided_cycle(t,b,m,n,r,f,edges,gold_total,score,lane,random,count,cells,old,next);
   else {
   for(int i=0;i<count;i++){
    int chosen=t[25*n+rnd(random)%f];bool duplicate=true;
    while(duplicate){duplicate=false;for(int j=0;j<i;j++)if(cells[j]==chosen)duplicate=true;if(duplicate)chosen=t[25*n+rnd(random)%f];}
    cells[i]=chosen;old[i]=b[chosen*r+lane];
   }
   for(int i=0;i<count;i++)next[i]=3*(old[(i+1)%count]/3)+rnd(random)%3;
   }
   // S=B*(T+1)+T-L, with 0<=L<=T: quotient/remainder recover the
   // exact old domain/seam score and proper-loop count without retracing.
   bool lazy=flags&16,incremental=flags&64;int before,oldloops;
   int base_delta=0,touched=0;
   if(flags&192)ibump(m,r,lane,0);
   if(incremental){base_delta=ichange(t,b,m,n,r,f,edges,lane,count,cells,next);if(gold_total)touched=itouched(t,b,m,n,r,f,lane,count,cells);}
   if(lazy){before=score/(gold_total+1);oldloops=gold_total-score%(gold_total+1);
    m[22*r+lane]++;if(gold_total)m[23*r+lane]++;
   }else if(incremental){before=score/(gold_total+1);oldloops=gold_total-score%(gold_total+1);}
   else {before=mtotal(t,b,m,n,r,f,edges,lane,flags);oldloops=mloops(t,b,m,n,r,f,lane,gold_total,flags);}
   for(int i=0;i<count;i++)b[cells[i]*r+lane]=next[i];
   int candidate_base=incremental?before+base_delta:mtotal(t,b,m,n,r,f,edges,lane,flags);base_delta=candidate_base-before;
   if(guided&&base_delta>0)m[21*r+lane]++;
   bool tabu=false;
   if(flags&2)for(int i=0;i<count;i++)for(int k=0;k<4;k++)
    if(m[(5+k)*r+lane]==unsigned(cells[i]*(3*n)+next[i])&&int(m[(9+k)*r+lane]-iteration)>0)tabu=true;
   float heat=(flags&4)?exchange_heat(m,r,lane,flags):(.25f+1.25f*(1.f-float(iteration%4096)/4096.f));
   bool accept=false,skip=false,predrawn=false;int delta=0;
   // Lnew>=0 makes this an integer upper bound. Tabu aspiration is
   // strictly greater than personal best, and this path consumes no RNG.
   if(lazy&&tabu&&candidate_base*(gold_total+1)+gold_total<=bestscore){skip=true;if(gold_total)m[25*r+lane]++;}
   // Without exchange the negative base difference is the EXACT legacy
   // Metropolis input, independent of loops. Never predraw while tabu:
   // uncertain aspiration could short-circuit the legacy RNG entirely.
   else if(lazy&&!tabu&&!(flags&4)&&base_delta<0){
    predrawn=true;m[27*r+lane]++;
    accept=float(rnd(random)%1000000)<1000000.f*expf(float(base_delta)/heat);
    if(!accept){skip=true;if(gold_total)m[26*r+lane]++;}
   }
   if(!skip){
    int loop_delta=incremental?touched-icomponents(t,b,m,n,r,f,lane,gold_total,count,cells,old,0):oldloops-mloops(t,b,m,n,r,f,lane,gold_total,flags);
    if(lazy&&gold_total)m[24*r+lane]++;
    delta=base_delta*(gold_total+1)+loop_delta;
    float acceptance=(flags&4)?float(delta)/float(gold_total+1):float(base_delta?base_delta:loop_delta);
    if(!predrawn)accept=(!tabu||score+delta>bestscore)&&(acceptance>=0||float(rnd(random)%1000000)<1000000.f*expf(acceptance/heat));
   }
   if(tabu&&!accept)m[14*r+lane]++;
   iteration++;
   if(accept){
    if(incremental&&gold_total){icomponents(t,b,m,n,r,f,lane,gold_total,count,cells,old,2);icomponents(t,b,m,n,r,f,lane,gold_total,count,cells,old,1);}
    score+=delta;m[13*r+lane]++;
    if(flags&2)for(int i=0;i<count;i++)if(old[i]!=next[i]){unsigned slot=m[4*r+lane]%4;m[4*r+lane]++;
     m[(5+slot)*r+lane]=cells[i]*(3*n)+old[i];m[(9+slot)*r+lane]=iteration+8+lane%4;}
    if(score>bestscore||(score==bestscore&&rnd(random)%64==0)){bestscore=score;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}
   }else for(int i=0;i<count;i++)b[cells[i]*r+lane]=old[i];
  }
  m[r+lane]=score;m[2*r+lane]=bestscore;m[3*r+lane]=iteration;
 }m[lane]=random;
}
extern "C" __global__ void replica_exchange(unsigned*b,unsigned*best,unsigned*m,int n,int r,int gold_total,int parity,int flags){
 int pair=blockIdx.x*blockDim.x+threadIdx.x,a=2*pair+parity,d=a+1;
 if(d>=r||a/8!=d/8)return;
 unsigned random=m[a];float ta=exchange_heat(m,r,a,flags),td=exchange_heat(m,r,d,flags);
 float exponent=(1.f/ta-1.f/td)*float(int(m[r+d])-int(m[r+a]))/float(gold_total+1);m[16*r+a]++;
 if(exponent>=0||float(rnd(random)%1000000)<1000000.f*expf(exponent)){
  for(int c=0;c<n;c++){unsigned old=b[c*r+a];b[c*r+a]=b[c*r+d];b[c*r+d]=old;}
  if(flags&64)for(int word=0;word<(gold_total+1)/2;word++){unsigned value=m[(45+word)*r+a];m[(45+word)*r+a]=m[(45+word)*r+d];m[(45+word)*r+d]=value;}
  unsigned old=m[r+a];m[r+a]=m[r+d];m[r+d]=old;m[17*r+a]++;
  for(int row=4;row<13;row++){m[row*r+a]=0;m[row*r+d]=0;}
  for(int which=0;which<2;which++){int lane=which?d:a;if(m[r+lane]>m[2*r+lane]){m[2*r+lane]=m[r+lane];for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}}
 }m[a]=random;
}
