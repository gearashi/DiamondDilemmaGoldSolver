// Exact persistent DFS. One invocation owns one disjoint prefix and private stack.
struct Parameters { n: u32, budget: u32, colors: u32, reserved: u32 }
@group(0) @binding(0) var<storage, read_write> board_used: array<u32>;
@group(0) @binding(1) var<storage, read_write> firsts: array<u32>;
@group(0) @binding(2) var<storage, read_write> lengths: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursors: array<u32>;
@group(0) @binding(4) var<storage, read_write> shifts: array<u32>;
@group(0) @binding(5) var<storage, read_write> metadata: array<u32>;
@group(0) @binding(6) var<storage, read> tables: array<u32>;
@group(0) @binding(7) var<uniform> parameters: Parameters;

fn board(cell: u32, lane: u32) -> i32 { return bitcast<i32>(board_used[cell * parameters.n + lane]); }
fn neighbor(cell: u32, side: u32) -> u32 { return tables[3040u + cell * 3u + side]; }
fn other_side(cell: u32, side: u32) -> u32 { return tables[3520u + cell * 3u + side]; }
fn face(code: u32, side: u32) -> u32 { return tables[160u + code * 3u + side]; }
fn reverse_face(code: u32, side: u32) -> u32 { return tables[1600u + code * 3u + side]; }
fn pair_start() -> u32 { return 4000u + 3u * parameters.colors + 1u; }
fn pool_start() -> u32 { return pair_start() + 3u * parameters.colors * parameters.colors + 1u; }
fn pool(index: u32) -> u32 { return tables[pool_start() + index]; }
fn random_next(value: u32) -> u32 {
    var result = value;
    result ^= result << 13u; result ^= result >> 17u; result ^= result << 5u;
    return result;
}
fn taken(tile: u32, lane: u32) -> bool {
    return (board_used[(160u + (tile >> 5u)) * parameters.n + lane] & (1u << (tile & 31u))) != 0u;
}
fn mark(tile: u32, lane: u32, occupied: bool) {
    let location = (160u + (tile >> 5u)) * parameters.n + lane;
    let bit = 1u << (tile & 31u);
    if occupied { board_used[location] |= bit; } else { board_used[location] &= ~bit; }
}
fn matches(cell: u32, code: u32, lane: u32) -> bool {
    for (var side = 0u; side < 3u; side++) {
        let other = board(neighbor(cell, side), lane);
        if other >= 0 && face(code, side) != reverse_face(u32(other), other_side(cell, side)) { return false; }
    }
    return true;
}
fn bucket(cell: u32, lane: u32) -> vec2<u32> {
    var side1 = -1; var side2 = -1; var need1 = 0u; var need2 = 0u;
    for (var side = 0u; side < 3u; side++) {
        let code = board(neighbor(cell, side), lane);
        if code >= 0 {
            let need = reverse_face(u32(code), other_side(cell, side));
            if side1 < 0 { side1 = i32(side); need1 = need; }
            else if side2 < 0 { side2 = i32(side); need2 = need; }
        }
    }
    if side1 < 0 { return vec2<u32>(0u, 480u); }
    if side2 < 0 {
        let key = 4000u + u32(side1) * parameters.colors + need1;
        return vec2<u32>(tables[key], tables[key + 1u] - tables[key]);
    }
    var pair = 1u;
    if side1 == 1 { pair = 2u; } else if side2 == 1 { pair = 0u; }
    let key = pair_start() + (pair * parameters.colors + need1) * parameters.colors + need2;
    return vec2<u32>(tables[key], tables[key + 1u] - tables[key]);
}
fn possible(cell: u32, lane: u32) -> bool {
    let choice = bucket(cell, lane);
    for (var index = 0u; index < choice.y; index++) {
        let code = pool(choice.x + index);
        if !taken(code / 3u, lane) && matches(cell, code, lane) { return true; }
    }
    return false;
}
fn add_counter(index: u32, lane: u32, amount: u32) {
    let low_index = (5u + 2u * index) * parameters.n + lane;
    let old = metadata[low_index]; let next = old + amount;
    metadata[low_index] = next;
    if next < old { metadata[low_index + parameters.n] += 1u; }
}
@compute @workgroup_size(64)
fn dfs_search(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let lane = invocation.x; let n = parameters.n;
    if lane >= n || metadata[3u * n + lane] != 0u { return; }
    var depth = metadata[lane]; var high = metadata[n + lane]; let floor = metadata[2u * n + lane];
    var random = metadata[4u * n + lane];
    var tried = 0u; var placed = 0u; var backed = 0u;
    for (var work = 0u; work < parameters.budget; work++) {
        if depth == 160u { metadata[3u * n + lane] = 1u; break; }
        let location = depth * n + lane;
        if cursors[location] == 65535u {
            let choice = bucket(tables[depth], lane);
            firsts[location] = choice.x; lengths[location] = choice.y;
            shifts[location] = 0u;
            if choice.y != 0u { random = random_next(random); shifts[location] = random % choice.y; }
            cursors[location] = 0u;
        }
        let length = lengths[location]; let cursor = cursors[location];
        if cursor >= length {
            if depth == floor {
                for (var cell = 0u; cell < 160u; cell++) { board_used[cell * n + lane] = 0xffffffffu; }
                for (var word = 0u; word < 5u; word++) { board_used[(160u + word) * n + lane] = 0u; }
                depth = 0u; metadata[3u * n + lane] = 2u; backed++; break;
            }
            let cell = tables[depth - 1u]; let old = u32(board(cell, lane));
            board_used[cell * n + lane] = 0xffffffffu; mark(old / 3u, lane, false);
            depth--; backed++; continue;
        }
        let code = pool(firsts[location] + (cursor + shifts[location]) % length);
        cursors[location] = cursor + 1u; tried++;
        let tile = code / 3u; let cell = tables[depth];
        if taken(tile, lane) || !matches(cell, code, lane) { continue; }
        board_used[cell * n + lane] = code; mark(tile, lane, true);
        var feasible = true;
        for (var side = 0u; side < 3u; side++) {
            let next_cell = neighbor(cell, side);
            if board(next_cell, lane) < 0 && !possible(next_cell, lane) { feasible = false; break; }
        }
        if !feasible {
            board_used[cell * n + lane] = 0xffffffffu; mark(tile, lane, false); continue;
        }
        depth++; placed++;
        high = max(high, depth);
        if depth == 160u { metadata[3u * n + lane] = 1u; break; }
        cursors[depth * n + lane] = 65535u;
    }
    metadata[lane] = depth; metadata[n + lane] = high; metadata[4u * n + lane] = random;
    add_counter(0u, lane, tried); add_counter(1u, lane, placed); add_counter(2u, lane, backed);
}
