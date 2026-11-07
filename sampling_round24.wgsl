// Heuristic permutation sampling only; exact proof search is performed separately.
struct Params { n:u32, replicas:u32, free_count:u32, steps:u32, phase:u32, edges:u32, gold_total:u32, unused1:u32 }
@group(0) @binding(0) var<storage,read_write> boards:array<u32>;
@group(0) @binding(1) var<storage,read_write> best_boards:array<u32>;
@group(0) @binding(2) var<storage,read_write> lane_data:array<u32>;
@group(0) @binding(3) var<storage,read> table:array<u32>;
@group(0) @binding(4) var<uniform> p:Params;
var<private> random_state:u32;
fn random_next()->u32 { random_state^=random_state<<13u; random_state^=random_state>>17u; random_state^=random_state<<5u; return random_state; }
fn at(cell:u32,lane:u32)->u32 { return boards[cell*p.replicas+lane]; }
fn neighbor(cell:u32,side:u32)->u32 { return table[18u*p.n+cell*3u+side]; }
fn legal(cell:u32,lane:u32)->i32 { return i32(table[25u*p.n+p.free_count+cell*3u*p.n+at(cell,lane)]); }
fn edge(cell:u32,side:u32,lane:u32)->i32 {
    let nb=neighbor(cell,side);
    if nb>=p.n { return 0; }
    let a=table[at(cell,lane)*3u+side];
    let b=table[9u*p.n+at(nb,lane)*3u+table[21u*p.n+cell*3u+side]];
    return select(0,1,a==b);
}
fn total(lane:u32)->i32 {
    var value=0;
    for(var cell=0u;cell<p.n;cell++) {
        value+=legal(cell,lane)*i32(p.edges+1u);
        for(var side=0u;side<3u;side++) { let nb=neighbor(cell,side); if nb<p.n && nb>cell { value+=edge(cell,side,lane); } }
    }
    return value;
}
// Old placements stay in storage until a proposal is accepted.
fn score_change(a:u32,b:u32,na:u32,nbcode:u32,lane:u32)->i32 {
    var delta=0;
    for(var which=0u;which<select(2u,1u,a==b);which++) {
        let cell=select(a,b,which==1u); let old=at(cell,lane);let next=select(na,nbcode,which==1u);
        let base=25u*p.n+p.free_count+cell*3u*p.n;
        delta+=(i32(table[base+next])-i32(table[base+old]))*i32(p.edges+1u);
        for(var side=0u;side<3u;side++) {
            let nb=neighbor(cell,side);if nb>=p.n || (which==1u && nb==a) { continue; }
            let opposite=table[21u*p.n+cell*3u+side];let other=at(nb,lane);
            var replacement=other;if nb==a { replacement=na; } else if nb==b { replacement=nbcode; }
            delta+=select(0,1,table[next*3u+side]==table[9u*p.n+replacement*3u+opposite])
                  -select(0,1,table[old*3u+side]==table[9u*p.n+other*3u+opposite]);
        }
    }
    return delta;
}
fn partner(code:u32,port:u32)->u32 {return table[25u*p.n+p.free_count+3u*p.n*p.n+1u+code*33u+port];}
fn loops(lane:u32,a:i32,b:i32)->i32 {
    if p.gold_total==0u {return 0;}
    var result=0;var roots=p.n;
    if a>=0 {roots=select(2u,1u,a==b);}
    for(var root=0u;root<roots;root++) {
        var start=root;if a>=0 {start=u32(select(a,b,root!=0u));}let code=at(start,lane);
        for(var port=0u;port<33u;port++) {let pair=partner(code,port);if pair>=33u||pair<port {continue;}
            var cell=start;var incoming=port;let key=start*33u+port;
            for(var count=1u;count<=p.gold_total;count++) {
                let outgoing=partner(at(cell,lane),incoming);if outgoing>=33u {break;}
                let local=cell*33u+min(incoming,outgoing);
                if (a<0||i32(cell)==a||i32(cell)==b)&&local<key {break;}
                let side=outgoing/11u;let nb=neighbor(cell,side);if nb>=p.n {break;}
                incoming=table[21u*p.n+cell*3u+side]*11u+10u-outgoing%11u;cell=nb;
                if cell==start&&incoming==port {if count<p.gold_total {result++;}break;}
            }
        }
    }
    return result;
}
fn save_best(lane:u32) { for(var cell=0u;cell<p.n;cell++) { best_boards[cell*p.replicas+lane]=at(cell,lane); } }
@compute @workgroup_size(64)
fn sample_search(@builtin(global_invocation_id) id:vec3<u32>) {
 let lane=id.x;let r=p.replicas;if lane>=r {return;}random_state=lane_data[lane];let flags=p.unused1;
 if p.phase==0u {
  for(var cell=0u;cell<p.n;cell++){boards[cell*r+lane]=table[24u*p.n+cell];}
  for(var count=p.free_count;count>1u;count--){let a=table[25u*p.n+count-1u];let b=table[25u*p.n+random_next()%count];let code=at(a,lane);boards[a*r+lane]=at(b,lane);boards[b*r+lane]=code;}
  for(var i=0u;i<p.free_count;i++){let cell=table[25u*p.n+i];boards[cell*r+lane]=3u*(at(cell,lane)/3u)+random_next()%3u;}
  let initial=u32(total(lane)*i32(p.gold_total+1u)+i32(p.gold_total)-loops(lane,-1,-1));lane_data[r+lane]=initial;lane_data[2u*r+lane]=initial;
  for(var row=3u;row<18u;row++){lane_data[row*r+lane]=0u;}save_best(lane);
 } else if p.free_count>0u {
  var score=i32(lane_data[r+lane]);var best_score=i32(lane_data[2u*r+lane]);var iteration=lane_data[3u*r+lane];
  for(var step=0u;step<p.steps;step++){
   var cells:array<u32,6>;var old:array<u32,6>;var next:array<u32,6>;var count=1u;
   if (flags&1u)!=0u && p.free_count>=3u && random_next()%4u!=0u {count=3u+random_next()%(min(6u,p.free_count)-2u);lane_data[15u*r+lane]++;}
   else if p.free_count>=2u {count=1u+random_next()%2u;}
   for(var i=0u;i<count;i++){
    var chosen=table[25u*p.n+random_next()%p.free_count];var duplicate=true;
    while duplicate {duplicate=false;for(var j=0u;j<i;j++){if cells[j]==chosen {duplicate=true;}}if duplicate {chosen=table[25u*p.n+random_next()%p.free_count];}}
    cells[i]=chosen;old[i]=at(chosen,lane);
   }
   for(var i=0u;i<count;i++){next[i]=3u*(old[(i+1u)%count]/3u)+random_next()%3u;}
   let before=total(lane);let old_loops=loops(lane,-1,-1);
   for(var i=0u;i<count;i++){boards[cells[i]*r+lane]=next[i];}
   let base_delta=total(lane)-before;let loop_delta=old_loops-loops(lane,-1,-1);
   let delta=base_delta*i32(p.gold_total+1u)+loop_delta;var tabu=false;
   if (flags&2u)!=0u {for(var i=0u;i<count;i++){for(var k=0u;k<4u;k++){
    if lane_data[(5u+k)*r+lane]==cells[i]*(3u*p.n)+next[i] && i32(lane_data[(9u+k)*r+lane]-iteration)>0 {tabu=true;}
   }}}
   var heat=0.25+1.25*(1.0-f32(iteration%4096u)/4096.0);var acceptance=f32(select(base_delta,loop_delta,base_delta==0));
   if (flags&4u)!=0u {heat=0.2*pow(1.5,f32(lane%8u));acceptance=f32(delta)/f32(p.gold_total+1u);}
   let accept=(!tabu||score+delta>best_score)&&(acceptance>=0.0||f32(random_next()%1000000u)<1000000.0*exp(acceptance/heat));
   if tabu&&!accept {lane_data[14u*r+lane]++;}iteration++;
   if accept {
    score+=delta;lane_data[13u*r+lane]++;
    if (flags&2u)!=0u {for(var i=0u;i<count;i++){if old[i]!=next[i]{let slot=lane_data[4u*r+lane]%4u;lane_data[4u*r+lane]++;lane_data[(5u+slot)*r+lane]=cells[i]*(3u*p.n)+old[i];lane_data[(9u+slot)*r+lane]=iteration+8u+lane%4u;}}}
    if score>best_score || (score==best_score&&random_next()%64u==0u){best_score=score;save_best(lane);}
   } else {for(var i=0u;i<count;i++){boards[cells[i]*r+lane]=old[i];}}
  }
  lane_data[r+lane]=u32(score);lane_data[2u*r+lane]=u32(best_score);lane_data[3u*r+lane]=iteration;
 }lane_data[lane]=random_state;
}
@compute @workgroup_size(64)
fn replica_exchange(@builtin(global_invocation_id) id:vec3<u32>){
 let a=2u*id.x+p.phase;let b=a+1u;let r=p.replicas;if b>=r||a/8u!=b/8u{return;}
 random_state=lane_data[a];let ta=0.2*pow(1.5,f32(a%8u));let tb=0.2*pow(1.5,f32(b%8u));
 let exponent=(1.0/ta-1.0/tb)*f32(i32(lane_data[r+b])-i32(lane_data[r+a]))/f32(p.gold_total+1u);lane_data[16u*r+a]++;
 if exponent>=0.0||f32(random_next()%1000000u)<1000000.0*exp(exponent){
  for(var cell=0u;cell<p.n;cell++){let old=at(cell,a);boards[cell*r+a]=at(cell,b);boards[cell*r+b]=old;}
  let old=lane_data[r+a];lane_data[r+a]=lane_data[r+b];lane_data[r+b]=old;lane_data[17u*r+a]++;
  for(var row=4u;row<13u;row++){lane_data[row*r+a]=0u;lane_data[row*r+b]=0u;}
  for(var which=0u;which<2u;which++){let lane=select(a,b,which!=0u);if lane_data[r+lane]>lane_data[2u*r+lane]{lane_data[2u*r+lane]=lane_data[r+lane];save_best(lane);}}
 }lane_data[a]=random_state;
}
// The prefix contains selected lane IDs; the suffix receives packed row boards.
@group(0) @binding(5) var<storage,read_write> gathered:array<u32>;
@compute @workgroup_size(64)
fn gather_best(@builtin(global_invocation_id) id:vec3<u32>) {
 let count=p.steps;let index=id.x;if index>=count*p.n{return;}
 let selected=index/p.n;let cell=index%p.n;
 gathered[count+index]=best_boards[cell*p.replicas+gathered[selected]];
}
