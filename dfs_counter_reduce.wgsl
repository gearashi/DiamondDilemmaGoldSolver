// Exact modulo-2^64 sum of the three persistent DFS counters.
struct Params { n: u32, groups: u32, reserved0: u32, reserved1: u32 }
@group(0) @binding(0) var<storage, read> metadata: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> a: array<vec2<u32>, 256>;
var<workgroup> b: array<vec2<u32>, 256>;
var<workgroup> c: array<vec2<u32>, 256>;

fn add64(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let low = x.x + y.x;
    return vec2<u32>(low, x.y + y.y + select(0u, 1u, low < x.x));
}
fn read_counter(index: u32, lane: u32) -> vec2<u32> {
    let row = 5u + 2u * index;
    return vec2<u32>(metadata[row * params.n + lane],
                     metadata[(row + 1u) * params.n + lane]);
}
fn reduce_shared(local: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local < stride {
            a[local] = add64(a[local], a[local + stride]);
            b[local] = add64(b[local], b[local + stride]);
            c[local] = add64(c[local], c[local + stride]);
        }
        workgroupBarrier();
    }
}
@compute @workgroup_size(256)
fn reduce_lanes(@builtin(global_invocation_id) global: vec3<u32>,
                @builtin(local_invocation_index) local: u32,
                @builtin(workgroup_id) group: vec3<u32>) {
    a[local] = vec2<u32>(0u); b[local] = vec2<u32>(0u); c[local] = vec2<u32>(0u);
    if global.x < params.n {
        a[local] = read_counter(0u, global.x);
        b[local] = read_counter(1u, global.x);
        c[local] = read_counter(2u, global.x);
    }
    reduce_shared(local);
    if local == 0u {
        partial[group.x * 3u] = a[0];
        partial[group.x * 3u + 1u] = b[0];
        partial[group.x * 3u + 2u] = c[0];
    }
}
@compute @workgroup_size(256)
fn reduce_groups(@builtin(local_invocation_index) local: u32) {
    a[local] = vec2<u32>(0u); b[local] = vec2<u32>(0u); c[local] = vec2<u32>(0u);
    for (var group = local; group < params.groups; group += 256u) {
        a[local] = add64(a[local], partial[group * 3u]);
        b[local] = add64(b[local], partial[group * 3u + 1u]);
        c[local] = add64(c[local], partial[group * 3u + 2u]);
    }
    reduce_shared(local);
    if local == 0u { result[0] = a[0]; result[1] = b[0]; result[2] = c[0]; }
}
