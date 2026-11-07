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
extern "C" __global__ void sample_search(unsigned*b,unsigned*best,unsigned*m,const unsigned*t,int n,int r,int f,int steps,int phase,int edges,int gold_total,int flags){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=r)return;unsigned random=m[lane];
 if(phase==0){
  for(int c=0;c<n;c++)b[c*r+lane]=t[24*n+c];
  for(int count=f;count>1;count--){int a=t[25*n+count-1],d=t[25*n+rnd(random)%count];unsigned old=b[a*r+lane];b[a*r+lane]=b[d*r+lane];b[d*r+lane]=old;}
  for(int i=0;i<f;i++){int c=t[25*n+i];b[c*r+lane]=3*(b[c*r+lane]/3)+rnd(random)%3;}
  m[r+lane]=m[2*r+lane]=total(t,b,n,r,f,edges,lane)*(gold_total+1)+gold_total-loops(t,b,n,r,f,-1,-1,lane,gold_total);
  for(int row=3;row<18;row++)m[row*r+lane]=0;
  for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];
 }else if(f>0){
  int score=m[r+lane],bestscore=m[2*r+lane];unsigned iteration=m[3*r+lane];
  for(int step=0;step<steps;step++){
   int cells[6],old[6],next[6],count=1;
   if((flags&1)&&f>=3&&rnd(random)%4!=0){count=3+rnd(random)%(min(6,f)-2);m[15*r+lane]++;}
   else if(f>=2)count=1+rnd(random)%2;
   // Select distinct positions, then relocate the displaced pieces around
   // the closed chain. Every proposal remains a physical-piece permutation.
   for(int i=0;i<count;i++){
    int chosen=t[25*n+rnd(random)%f];bool duplicate=true;
    while(duplicate){duplicate=false;for(int j=0;j<i;j++)if(cells[j]==chosen)duplicate=true;if(duplicate)chosen=t[25*n+rnd(random)%f];}
    cells[i]=chosen;old[i]=b[chosen*r+lane];
   }
   for(int i=0;i<count;i++)next[i]=3*(old[(i+1)%count]/3)+rnd(random)%3;
   int before=total(t,b,n,r,f,edges,lane),oldloops=loops(t,b,n,r,f,-1,-1,lane,gold_total);
   for(int i=0;i<count;i++)b[cells[i]*r+lane]=next[i];
   int base_delta=total(t,b,n,r,f,edges,lane)-before;
   int loop_delta=oldloops-loops(t,b,n,r,f,-1,-1,lane,gold_total);
   int delta=base_delta*(gold_total+1)+loop_delta;bool tabu=false;
   if(flags&2)for(int i=0;i<count;i++)for(int k=0;k<4;k++)
    if(m[(5+k)*r+lane]==unsigned(cells[i]*(3*n)+next[i])&&int(m[(9+k)*r+lane]-iteration)>0)tabu=true;
   float heat=(flags&4)?(.2f*powf(1.5f,float(lane%8))):(.25f+1.25f*(1.f-float(iteration%4096)/4096.f));
   float acceptance=(flags&4)?float(delta)/float(gold_total+1):float(base_delta?base_delta:loop_delta);
   bool accept=(!tabu||score+delta>bestscore)&&(acceptance>=0||float(rnd(random)%1000000)<1000000.f*expf(acceptance/heat));
   if(tabu&&!accept)m[14*r+lane]++;
   iteration++;
   if(accept){
    score+=delta;m[13*r+lane]++;
    if(flags&2)for(int i=0;i<count;i++)if(old[i]!=next[i]){unsigned slot=m[4*r+lane]%4;m[4*r+lane]++;
     m[(5+slot)*r+lane]=cells[i]*(3*n)+old[i];m[(9+slot)*r+lane]=iteration+8+lane%4;}
    if(score>bestscore||(score==bestscore&&rnd(random)%64==0)){bestscore=score;for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}
   }else for(int i=0;i<count;i++)b[cells[i]*r+lane]=old[i];
  }
  m[r+lane]=score;m[2*r+lane]=bestscore;m[3*r+lane]=iteration;
 }m[lane]=random;
}
extern "C" __global__ void replica_exchange(unsigned*b,unsigned*best,unsigned*m,int n,int r,int gold_total,int parity){
 int pair=blockIdx.x*blockDim.x+threadIdx.x,a=2*pair+parity,d=a+1;
 if(d>=r||a/8!=d/8)return;
 unsigned random=m[a];float ta=.2f*powf(1.5f,float(a%8)),td=.2f*powf(1.5f,float(d%8));
 float exponent=(1.f/ta-1.f/td)*float(int(m[r+d])-int(m[r+a]))/float(gold_total+1);m[16*r+a]++;
 if(exponent>=0||float(rnd(random)%1000000)<1000000.f*expf(exponent)){
  for(int c=0;c<n;c++){unsigned old=b[c*r+a];b[c*r+a]=b[c*r+d];b[c*r+d]=old;}
  unsigned old=m[r+a];m[r+a]=m[r+d];m[r+d]=old;m[17*r+a]++;
  for(int row=4;row<13;row++){m[row*r+a]=0;m[row*r+d]=0;}
  for(int which=0;which<2;which++){int lane=which?d:a;if(m[r+lane]>m[2*r+lane]){m[2*r+lane]=m[r+lane];for(int c=0;c<n;c++)best[c*r+lane]=b[c*r+lane];}}
 }m[a]=random;
}
