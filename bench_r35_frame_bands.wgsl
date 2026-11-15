@group(0) @binding(0) var<storage,read> f0:array<u32>;
@group(0) @binding(1) var<storage,read> f1:array<u32>;
@group(0) @binding(2) var<storage,read> f2:array<u32>;
@group(0) @binding(3) var<storage,read> f3:array<u32>;
@group(0) @binding(4) var<storage,read> p:array<u32>;
@group(0) @binding(5) var<storage,read_write> out:array<u32>;
fn lane(address:u32,j:u32)->u32 {
    if (address&0x80000000u)!=0u {return (address&0x7fffffffu)+j;}
    return p[p[2]+address+j];
}
fn read_field(field:u32,cell:u32)->u32 {
    switch field {case 0u: {return f0[cell];} case 1u: {return f1[cell];} case 2u: {return f2[cell];} default: {return f3[cell];}}
}
@compute @workgroup_size(64)
fn gather_bands(@builtin(global_invocation_id) id:vec3<u32>) {
    let row=3u+4u*id.y; let j=2u*id.x;
    let size=p[row+1u]; if j>=size {return;}
    let field=p[row]/160u; let base=(p[row]%160u)*p[0]; let at=p[row+2u];
    let left=read_field(field,base+lane(at,j))&65535u; var right=0u;
    if j+1u<size {right=read_field(field,base+lane(at,j+1u));}
    out[p[row+3u]+id.x]=left|(right<<16u);
}
