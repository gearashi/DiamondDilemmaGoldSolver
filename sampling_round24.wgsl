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
// Guidance scores the unchanged board; only the complete closed permutation
// and its independently recomputed total/loop score control acceptance.
fn local_candidate(cell:u32,code:u32,lane:u32)->i32 {
    var value=i32(table[25u*p.n+p.free_count+cell*3u*p.n+code])*i32(p.edges+1u);
    for(var side=0u;side<3u;side++) {
        let nb=neighbor(cell,side);
        if nb<p.n {
            value+=select(0,1,table[code*3u+side]==table[9u*p.n+at(nb,lane)*3u+table[21u*p.n+cell*3u+side]]);
        }
    }
    return value;
}
fn guided_cycle(lane:u32,count:u32,score:i32,cells:ptr<function,array<u32,6>>,old:ptr<function,array<u32,6>>,next:ptr<function,array<u32,6>>) {
    lane_data[18u*p.replicas+lane]++;
    let start=random_next()%p.free_count;var first=table[25u*p.n+start];
    let full_base=i32(p.n*(p.edges+1u)+p.edges);
    let closed_obstruction=p.gold_total>0u && score/i32(p.gold_total+1u)==full_base && score<full_base*i32(p.gold_total+1u)+i32(p.gold_total);
    for(var offset=0u;offset<p.free_count;offset++) {
        let cell=table[25u*p.n+(start+offset)%p.free_count];var bad=legal(cell,lane)==0;
        for(var side=0u;side<3u;side++) {if neighbor(cell,side)<p.n && edge(cell,side,lane)==0 {bad=true;}}
        if !bad && closed_obstruction {
            for(var port=0u;port<33u;port++) {if partner(at(cell,lane),port)<33u {bad=true;break;}}
        }
        if bad {first=cell;lane_data[20u*p.replicas+lane]++;break;}
    }
    (*cells)[0]=first;(*old)[0]=at(first,lane);
    for(var i=0u;i<count-1u;i++) {
        let begin=random_next()%p.free_count;let rotation=random_next()%3u;
        var inspected=0u;var best_value=-1;var donor=0u;var chosen_code=0u;
        for(var offset=0u;offset<p.free_count && inspected<16u;offset++) {
            let candidate=table[25u*p.n+(begin+offset)%p.free_count];var used=false;
            for(var j=0u;j<=i;j++) {if (*cells)[j]==candidate {used=true;}}
            if used {continue;}inspected++;
            let tile=at(candidate,lane)/3u;
            for(var k=0u;k<3u;k++) {
                let code=tile*3u+(rotation+k)%3u;lane_data[19u*p.replicas+lane]++;
                let value=local_candidate((*cells)[i],code,lane);
                if value>best_value {best_value=value;donor=candidate;chosen_code=code;}
            }
        }
        (*cells)[i+1u]=donor;(*old)[i+1u]=at(donor,lane);(*next)[i]=chosen_code;
    }
    let rotation=random_next()%3u;var best_value=-1;
    for(var k=0u;k<3u;k++) {
        let code=3u*((*old)[0]/3u)+(rotation+k)%3u;lane_data[19u*p.replicas+lane]++;
        let value=local_candidate((*cells)[count-1u],code,lane);
        if value>best_value {best_value=value;(*next)[count-1u]=code;}
    }
}
fn exchange_heat(lane:u32)->f32 {
 if (p.unused1&32u)!=0u {return bitcast<f32>(lane_data[28u*p.replicas+lane]);}
 return 0.2*pow(1.5,f32(lane%8u));
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
  for(var row=3u;row<select(select(select(18u,22u,(flags&8u)!=0u),28u,(flags&16u)!=0u),29u,(flags&32u)!=0u);row++){lane_data[row*r+lane]=0u;}
  if (flags&32u)!=0u {lane_data[28u*r+lane]=bitcast<u32>(0.2*pow(1.5,f32(lane%8u)));}save_best(lane);
 } else if p.free_count>0u {
  var score=i32(lane_data[r+lane]);var best_score=i32(lane_data[2u*r+lane]);var iteration=lane_data[3u*r+lane];
  for(var step=0u;step<p.steps;step++){
   var cells:array<u32,6>;var old:array<u32,6>;var next:array<u32,6>;var count=1u;
   var guided=false;if (flags&8u)!=0u {guided=random_next()%4u!=0u;}
   if guided {
    count=p.free_count;if p.free_count>=3u {count=3u+random_next()%(min(6u,p.free_count)-2u);}lane_data[15u*r+lane]++;
   }
   else if (flags&1u)!=0u && p.free_count>=3u && random_next()%4u!=0u {count=3u+random_next()%(min(6u,p.free_count)-2u);lane_data[15u*r+lane]++;}
   else if p.free_count>=2u {count=1u+random_next()%2u;}
   if guided {guided_cycle(lane,count,score,&cells,&old,&next);}
   else {
   for(var i=0u;i<count;i++){
    var chosen=table[25u*p.n+random_next()%p.free_count];var duplicate=true;
    while duplicate {duplicate=false;for(var j=0u;j<i;j++){if cells[j]==chosen {duplicate=true;}}if duplicate {chosen=table[25u*p.n+random_next()%p.free_count];}}
    cells[i]=chosen;old[i]=at(chosen,lane);
   }
   for(var i=0u;i<count;i++){next[i]=3u*(old[(i+1u)%count]/3u)+random_next()%3u;}
   }
   // Integer quotient/remainder recover S=B*(T+1)+T-L exactly.
   let lazy=(flags&16u)!=0u;var before:i32;var old_loops:i32;
   if lazy {
    before=score/i32(p.gold_total+1u);old_loops=i32(p.gold_total)-score%i32(p.gold_total+1u);
    lane_data[22u*r+lane]++;if p.gold_total>0u {lane_data[23u*r+lane]++;}
   } else {before=total(lane);old_loops=loops(lane,-1,-1);}
   for(var i=0u;i<count;i++){boards[cells[i]*r+lane]=next[i];}
   let candidate_base=total(lane);let base_delta=candidate_base-before;
   if guided && base_delta>0 {lane_data[21u*r+lane]++;}
   var tabu=false;
   if (flags&2u)!=0u {for(var i=0u;i<count;i++){for(var k=0u;k<4u;k++){
    if lane_data[(5u+k)*r+lane]==cells[i]*(3u*p.n)+next[i] && i32(lane_data[(9u+k)*r+lane]-iteration)>0 {tabu=true;}
   }}}
   var heat=0.25+1.25*(1.0-f32(iteration%4096u)/4096.0);
   if (flags&4u)!=0u {heat=exchange_heat(lane);}
   var accept=false;var skip=false;var predrawn=false;var delta:i32=0;
   // Lnew>=0 bounds aspiration by an integer. Equality still rejects;
   // a tabu failure must not consume a Metropolis random number.
   if lazy && tabu && candidate_base*i32(p.gold_total+1u)+i32(p.gold_total)<=best_score {
    skip=true;if p.gold_total>0u {lane_data[25u*r+lane]++;}
   } else if lazy && !tabu && (flags&4u)==0u && base_delta<0 {
    predrawn=true;lane_data[27u*r+lane]++;
    accept=f32(random_next()%1000000u)<1000000.0*exp(f32(base_delta)/heat);
    if !accept {skip=true;if p.gold_total>0u {lane_data[26u*r+lane]++;}}
   }
   if !skip {
    let loop_delta=old_loops-loops(lane,-1,-1);
    if lazy && p.gold_total>0u {lane_data[24u*r+lane]++;}
    delta=base_delta*i32(p.gold_total+1u)+loop_delta;
    var acceptance=f32(select(base_delta,loop_delta,base_delta==0));
    if (flags&4u)!=0u {acceptance=f32(delta)/f32(p.gold_total+1u);}
    if !predrawn {accept=(!tabu||score+delta>best_score)&&(acceptance>=0.0||f32(random_next()%1000000u)<1000000.0*exp(acceptance/heat));}
   }
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
 random_state=lane_data[a];let ta=exchange_heat(a);let tb=exchange_heat(b);
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
