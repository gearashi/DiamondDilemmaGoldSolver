@group(0) @binding(0) var<storage,read> firsts:array<u32>;
@group(0) @binding(1) var<storage,read> lengths:array<u32>;
@group(0) @binding(2) var<storage,read> cursors:array<u32>;
@group(0) @binding(3) var<storage,read> shifts:array<u32>;
@group(0) @binding(4) var<storage,read> p:array<u32>;
@group(0) @binding(5) var<storage,read_write> result:array<u32>;

@compute @workgroup_size(64)
fn gather_spans(@builtin(global_invocation_id) id:vec3<u32>) {
    let j=id.x;
    if j>=p[1] {return;}
    let flags=p[5];
    var lane=p[2]+j;
    var m=p[3];
    var dest=p[4]*j;
    if (flags&1u)!=0u {lane=p[p[2]+j];}
    if (flags&2u)!=0u {m=p[p[3]+j];}
    if (flags&4u)!=0u {dest=p[p[4]+j];}
    emit_span(lane,m,dest);
}

fn emit_span(lane:u32,m:u32,dest:u32) {
    let a=m&255u; let b=(m>>8u)&255u; let c=(m>>16u)&255u; let d=m>>24u;
    let pitch=select(0u,b-a+1u,a!=0u);
    let count=3u*pitch+d-c+1u;
    var left:u32;
    for (var i=0u;i<count;i++) {
        let field=select(3u,min(i/max(1u,pitch),3u),pitch!=0u);
        let row=select(a-1u+i-field*pitch,c-1u+i-3u*pitch,field==3u);
        let src=row*p[0]+lane;
        var v:u32;
        switch field {
            case 0u: {v=firsts[src];} case 1u: {v=lengths[src];}
            case 2u: {v=shifts[src];} default: {v=cursors[src];}
        }
        if (i&1u)!=0u {result[dest+i/2u]=left|(v<<16u);} else {left=v&65535u;}
    }
    if (count&1u)!=0u {result[dest+count/2u]=left;}
}

fn dense_mark(j:u32)->u32 {
    if (p[5]&16u)==0u {return p[p[3]+j];}
    let bits=p[6];let per=32u/bits;
    let code=(p[p[3]+j/per]>>(bits*(j%per)))&((1u<<bits)-1u);
    return p[p[7]+code];
}

var<workgroup> starts:array<u32,32>;

@compute @workgroup_size(32)
fn gather_dense_spans(@builtin(global_invocation_id) id:vec3<u32>, @builtin(local_invocation_index) tid:u32,
                      @builtin(workgroup_id) grp:vec3<u32>) {
    // A single lane computes the short local prefix. Each physical lane then
    // owns complete output words, including the last odd half.
    if tid==0u {
        var offset=p[p[4]+grp.x];
        for (var j=0u;j<32u;j++) {
            starts[j]=offset;
            let local=grp.x*32u+j;
            if local<p[1] {
                let m=dense_mark(local);let a=m&255u;let b=(m>>8u)&255u;
                let c=(m>>16u)&255u;let d=m>>24u;
                if m!=0u {offset+=(3u*select(0u,b-a+1u,a!=0u)+d-c+2u)/2u;}
            }
        }
    }
    workgroupBarrier();
    if id.x<p[1] {
        let m=dense_mark(id.x);
        if m!=0u {emit_span(p[2]+id.x,m,starts[tid]);}
    }
}
