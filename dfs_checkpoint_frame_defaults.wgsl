@group(0) @binding(0) var<storage,read> first:array<u32>;
@group(0) @binding(1) var<storage,read> length:array<u32>;
@group(0) @binding(2) var<storage,read> cursor:array<u32>;
@group(0) @binding(3) var<storage,read> shift:array<u32>;
@group(0) @binding(4) var<storage,read> plan:array<u32>;
@group(0) @binding(5) var<storage,read_write> work:array<u32>;
@group(0) @binding(6) var<storage,read_write> out:array<atomic<u32>>;
var<workgroup> flags:array<atomic<u32>,4>;
var<workgroup> mins:array<vec4<u32>,128>;
var<workgroup> maxs:array<vec4<u32>,128>;
fn values(j:u32)->vec4<u32>{return vec4<u32>(first[j],length[j],cursor[j],shift[j])&vec4<u32>(65535u);}

@compute @workgroup_size(128)
fn measure(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) t:u32) {
    if t<4u {atomicStore(&flags[t],0u);}
    workgroupBarrier();
    let n=plan[0];let lane=128u*group.x+t;let j=group.y*n+lane;
    var v=vec4<u32>(0u,0u,65535u,0u);if lane<n {v=values(j);}
    let dirty=any(v!=vec4<u32>(0u,0u,65535u,0u));
    if dirty {atomicOr(&flags[t/32u],1u<<(t&31u));}
    mins[t]=select(vec4<u32>(65535u),v,dirty);maxs[t]=select(vec4<u32>(0u),v,dirty);
    workgroupBarrier();
    for(var d=64u;d>0u;d/=2u){
        if t<d {mins[t]=min(mins[t],mins[t+d]);maxs[t]=max(maxs[t],maxs[t+d]);}
        workgroupBarrier();
    }
    if t<4u {
        let at=9u*(group.y*plan[1]+group.x);
        work[at+t]=atomicLoad(&flags[t]);work[at+5u+t]=mins[0][t]|(maxs[0][t]<<16u);
    }
}
@compute @workgroup_size(1)
fn scan(@builtin(workgroup_id) group:vec3<u32>) {
    let row=group.x;let groups=plan[1];var seen=0u;var low=plan[0];var high=0u;
    var lo=vec4<u32>(65535u);var hi=vec4<u32>(0u);
    for(var g=0u;g<groups;g++) {
        let pos=9u*(row*groups+g);work[pos+4u]=seen;
        for(var w=0u;w<4u;w++) {
            let m=work[pos+w];if m!=0u {
                low=min(low,128u*g+32u*w+firstTrailingBit(m));high=128u*g+32u*w+firstLeadingBit(m)+1u;
                seen+=countOneBits(m);
            }
        }
        let ranges=vec4<u32>(work[pos+5u],work[pos+6u],work[pos+7u],work[pos+8u]);
        lo=min(lo,ranges&vec4<u32>(65535u));hi=max(hi,ranges>>vec4<u32>(16u));
    }
    let base=5u+7u*row;
    atomicStore(&out[base],seen);atomicStore(&out[base+1u],low);atomicStore(&out[base+2u],high);
    let bounds=lo|(hi<<vec4<u32>(16u));
    atomicStore(&out[base+3u],bounds.x);atomicStore(&out[base+4u],bounds.y);
    atomicStore(&out[base+5u],bounds.z);atomicStore(&out[base+6u],bounds.w);
    if seen!=0u {atomicOr(&out[row/32u],1u<<(row%32u));}
}
@compute @workgroup_size(128)
fn pack(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) t:u32) {
    let n=plan[0];let row=plan[2]+group.y;let lane=128u*group.x+t;
    let pos=3u+10u*group.y;let mode=plan[pos];let low=plan[pos+1u];let high=plan[pos+2u];
    let base=1u+plan[pos+3u];let dir=plan[pos+4u];
    if mode==0u || lane>=n{return;}
    let at=9u*(row*plan[1]+group.x);let w=t/32u;let m=work[at+w];let word=lane/32u;
    if mode==2u && (t&31u)==0u && word>=low/32u && word<=(high-1u)/32u {atomicStore(&out[base+word-low/32u],m);}
    var rank:u32;
    if mode==1u {if lane<low || lane>=high{return;} rank=lane-low;}
    else {
        let bit=1u<<(t&31u);if (m&bit)==0u{return;}
        rank=work[at+4u]+countOneBits(m&(bit-1u));
        for(var k=0u;k<w;k++){rank+=countOneBits(work[at+k]);}
        if mode==3u {atomicStore(&out[base+rank],lane);}
    }
    let v=values(row*n+lane);var x=0u;var y=0u;var filled=0u;
    for(var f=0u;f<4u;f++){
        let bias=(plan[pos+6u+f/2u]>>(16u*(f%2u)))&65535u;
        let width=(plan[pos+8u]>>(5u*f))&31u;let rel=v[f]-bias;
        if rel>=(1u<<width){atomicOr(&out[0],1u);return;}
        if width!=0u {
            if filled<32u{
                x|=rel<<filled;if filled+width>32u{y|=rel>>(32u-filled);}
            }else{y|=rel<<(filled-32u);}
        }
        filled+=width;
    }
    let bits=rank*plan[pos+9u];let dst=base+dir+bits/32u;let sh=bits%32u;
    let w0=x<<sh;var w1=y;var w2=0u;
    if sh!=0u{w1=(y<<sh)|(x>>(32u-sh));w2=y>>(32u-sh);}
    if w0!=0u{atomicOr(&out[dst],w0);}if w1!=0u{atomicOr(&out[dst+1u],w1);}if w2!=0u{atomicOr(&out[dst+2u],w2);}
}
