@group(0) @binding(0) var<storage,read> firsts:array<u32>;
@group(0) @binding(1) var<storage,read> lengths:array<u32>;
@group(0) @binding(2) var<storage,read> cursors:array<u32>;
@group(0) @binding(3) var<storage,read> shifts:array<u32>;
@group(0) @binding(4) var<storage,read> plan:array<u32>;
@group(0) @binding(5) var<storage,read_write> result:array<u32>;

fn field_at(field:u32, pos:u32)->u32 {
    switch field {
        case 0u: {return firsts[pos];}
        case 1u: {return lengths[pos];}
        case 2u: {return cursors[pos];}
        default: {return shifts[pos];}
    }
}

fn lane_at(address:u32,col:u32)->u32 {
    if (address&0x80000000u)!=0u {return (address&0x7fffffffu)+col;}
    return plan[address+col];
}

fn write_pair(field:u32,row:u32,tid:u32) {
    let rect=2u+3u*row;
    let pitch=plan[1];let base=field*pitch+plan[rect];
    let col=2u*tid+(base&1u);
    if col>=plan[rect+1u] {return;}
    let address=plan[rect+2u];
    let left=row*plan[0]+lane_at(address,col);
    var right:u32;
    if col+1u==plan[rect+1u] {
        right=plan[arrayLength(&plan)-160u+row];
    } else {
        right=row*plan[0]+lane_at(address,col+1u);
    }
    // At an odd field boundary, the high half contains the next field's first cell.
    let right_field=field+u32(plan[rect]+col+1u==pitch);
    result[(base+col)/2u]=(field_at(field,left)&65535u)|(field_at(right_field,right)<<16u);
}

@compute @workgroup_size(64)
fn gather_frame_packet(@builtin(global_invocation_id) id:vec3<u32>) {
    write_pair(0u,id.y,id.x);
    write_pair(1u,id.y,id.x);
    write_pair(2u,id.y,id.x);
    write_pair(3u,id.y,id.x);
}
