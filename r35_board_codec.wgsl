struct Output {
    failed: atomic<u32>,
    words: array<u32>,
}
@group(0) @binding(0) var<storage,read> physical: array<u32>;
@group(0) @binding(1) var<storage,read> layout_data: array<u32>;
@group(0) @binding(2) var<storage,read_write> output: Output;

@compute @workgroup_size(64)
fn physical_board(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    let n=layout_data[0];let k=layout_data[1];
    if i>=k { return; }
    let lane=layout_data[2u+i];let start=10u*k+layout_data[2u+k+i];
    let cap=layout_data[2u+2u*k+i];
    var seen=0u;
    var pair=0u;
    for (var word=0u;word<5u;word++) {
        var mask=0u;
        for (var bit=0u;bit<32u;bit++) {
            let value=physical[(32u*word+bit)*n+lane]&65535u;
            if value==65535u { continue; }
            mask|=1u<<bit;
            if seen<cap {
                if (seen&1u)==0u {pair=value;}
                else {pair|=value<<16u;output.words[start+seen/2u]=pair;}
            }
            seen++;
        }
        output.words[word*k+i]=mask;
        output.words[(word+5u)*k+i]=physical[(160u+word)*n+lane];
    }
    if seen>cap {atomicOr(&output.failed,1u);return;}
    if (seen&1u)!=0u {output.words[start+seen/2u]=pair;}
    for(var w=(seen+1u)/2u;w<(cap+1u)/2u;w++){output.words[start+w]=0u;}
}
