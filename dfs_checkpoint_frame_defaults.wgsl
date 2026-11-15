@group(0) @binding(0) var<storage,read> first:array<u32>;
@group(0) @binding(1) var<storage,read> length:array<u32>;
@group(0) @binding(2) var<storage,read> cursor:array<u32>;
@group(0) @binding(3) var<storage,read> shift:array<u32>;
@group(0) @binding(4) var<storage,read> plan:array<u32>;
@group(0) @binding(5) var<storage,read_write> work:array<u32>;
@group(0) @binding(6) var<storage,read_write> out:array<u32>;
var<workgroup> flags:array<atomic<u32>,4>;

@compute @workgroup_size(128)
fn measure(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) t:u32) {
    if t<4u {atomicStore(&flags[t],0u);}
    workgroupBarrier();
    let n=plan[0];let lane=128u*group.x+t;let j=group.y*n+lane;
    if lane<n && (((first[j]|length[j]|shift[j])&65535u)!=0u || (cursor[j]&65535u)!=65535u) {
        atomicOr(&flags[t/32u],1u<<(t&31u));
    }
    workgroupBarrier();
    if t<4u {work[5u*(group.y*plan[1]+group.x)+t]=atomicLoad(&flags[t]);}
}
@compute @workgroup_size(1)
fn scan(@builtin(workgroup_id) group:vec3<u32>) {
    let row=group.x;let groups=plan[1];var seen=0u;var low=plan[0];var high=0u;
    for(var g=0u;g<groups;g++) {
        let pos=5u*(row*groups+g);work[pos+4u]=seen;
        for(var w=0u;w<4u;w++) {
            let m=work[pos+w];
            if m!=0u {
                low=min(low,128u*g+32u*w+firstTrailingBit(m));
                high=128u*g+32u*w+firstLeadingBit(m)+1u;
                seen+=countOneBits(m);
            }
        }
    }
    out[3u*row]=seen;out[3u*row+1u]=low;out[3u*row+2u]=high;
}
@compute @workgroup_size(128)
fn pack(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) t:u32) {
    let n=plan[0];let row=plan[2]+group.y;let lane=128u*group.x+t;
    let pos=3u+6u*group.y;let mode=plan[pos];let low=plan[pos+1u];let high=plan[pos+2u];
    let base=plan[pos+3u];let dir=plan[pos+4u];
    if mode==0u || lane>=n{return;}
    let at=5u*(row*plan[1]+group.x);let w=t/32u;let m=work[at+w];let word=lane/32u;
    if mode==2u && (t&31u)==0u && word>=low/32u && word<=(high-1u)/32u {out[base+word-low/32u]=m;}
    var rank:u32;
    if mode==1u {if lane<low || lane>=high{return;} rank=lane-low;}
    else {
        let bit=1u<<(t&31u);if (m&bit)==0u{return;}
        rank=work[at+4u]+countOneBits(m&(bit-1u));
        for(var k=0u;k<w;k++){rank+=countOneBits(work[at+k]);}
        if mode==3u {out[base+rank]=lane;}
    }
    let src=row*n+lane;let dst=base+dir+2u*rank;
    out[dst]=(first[src]&65535u)|(length[src]<<16u);
    out[dst+1u]=(cursor[src]&65535u)|(shift[src]<<16u);
}
