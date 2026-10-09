// Heuristic permutation sampling only; exact proof search is performed separately.
struct Params { n:u32, replicas:u32, free_count:u32, steps:u32, phase:u32, edges:u32, unused0:u32, unused1:u32 }
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
fn score_local(a:u32,b:u32,lane:u32)->i32 {
    var value=legal(a,lane)*i32(p.edges+1u);
    for(var side=0u;side<3u;side++) { value+=edge(a,side,lane); }
    if b!=a {
        value+=legal(b,lane)*i32(p.edges+1u);
        for(var side=0u;side<3u;side++) { if neighbor(b,side)!=a { value+=edge(b,side,lane); } }
    }
    return value;
}
fn save_best(lane:u32) { for(var cell=0u;cell<p.n;cell++) { best_boards[cell*p.replicas+lane]=at(cell,lane); } }
@compute @workgroup_size(64)
fn sample_search(@builtin(global_invocation_id) id:vec3<u32>) {
    let lane=id.x; let r=p.replicas;
    if lane>=r { return; }
    random_state=lane_data[lane];
    if p.phase==0u {
        for(var cell=0u;cell<p.n;cell++) { boards[cell*r+lane]=table[24u*p.n+cell]; }
        for(var count=p.free_count;count>1u;count--) {
            let a=table[25u*p.n+count-1u]; let b=table[25u*p.n+random_next()%count];
            let code=at(a,lane); boards[a*r+lane]=at(b,lane); boards[b*r+lane]=code;
        }
        for(var index=0u;index<p.free_count;index++) { let cell=table[25u*p.n+index]; boards[cell*r+lane]=3u*(at(cell,lane)/3u)+random_next()%3u; }
        let score=u32(total(lane)); lane_data[r+lane]=score; lane_data[2u*r+lane]=score; lane_data[3u*r+lane]=0u; save_best(lane);
    } else if p.free_count>0u {
        for(var step=0u;step<p.steps;step++) {
            let a=table[25u*p.n+random_next()%p.free_count]; let b=table[25u*p.n+random_next()%p.free_count];
            let ca=at(a,lane); let cb=at(b,lane); let before=score_local(a,b,lane);
            boards[a*r+lane]=3u*(cb/3u)+random_next()%3u;
            if a!=b { boards[b*r+lane]=3u*(ca/3u)+random_next()%3u; }
            let delta=score_local(a,b,lane)-before;
            let heat=0.25+1.25*(1.0-f32(lane_data[3u*r+lane]%4096u)/4096.0);
            let accept=delta>=0 || f32(random_next()%1000000u)<1000000.0*exp(f32(delta)/heat);
            lane_data[3u*r+lane]+=1u;
            if accept {
                lane_data[r+lane]=u32(i32(lane_data[r+lane])+delta);
                if lane_data[r+lane]>lane_data[2u*r+lane] || (lane_data[r+lane]==lane_data[2u*r+lane] && random_next()%64u==0u) {
                    lane_data[2u*r+lane]=lane_data[r+lane]; save_best(lane);
                }
            } else { boards[a*r+lane]=ca; boards[b*r+lane]=cb; }
        }
    }
    lane_data[lane]=random_state;
}
