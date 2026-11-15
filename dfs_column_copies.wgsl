@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> selection: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<u32>;

fn coordinate(index: u32, base: u32) -> u32 {
    if (base & 0x80000000u) != 0u {
        return (base & 0x7fffffffu) + index;
    }
    return selection[base + index];
}

fn address(row: u32, column: u32) -> u32 {
    return coordinate(row, selection[3]) * selection[0] + coordinate(column, selection[4]);
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < selection[1] && id.y < selection[2] {
        destination[id.y * selection[1] + id.x] = source[address(id.y, id.x)];
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < selection[1] && id.y < selection[2] {
        destination[address(id.y, id.x)] = source[id.y * selection[1] + id.x];
    }
}

@compute @workgroup_size(64)
fn gather_frames(@builtin(global_invocation_id) id: vec3<u32>) {
    let rect = 2u + 3u * id.y;
    if id.x < selection[rect+1u] {
        let pos = selection[rect+2u];
        var column: u32;
        if (pos & 0x80000000u) != 0u {
            column = (pos & 0x7fffffffu) + id.x;
        } else {
            column = selection[pos + id.x];
        }
        destination[selection[rect] + id.x] = source[id.y * selection[0] + column];
    }
}
