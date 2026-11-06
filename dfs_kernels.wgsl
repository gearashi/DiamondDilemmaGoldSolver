// Exact persistent DFS. One invocation owns one disjoint prefix and private stack.
struct Parameters { n: u32, budget: u32, colors: u32, gold_total: u32, flags: u32, reserved0: u32, reserved1: u32, reserved2: u32 }
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
fn gold_partner(code: u32, port: u32) -> u32 { return tables[pool_start() + 3360u + code * 33u + port]; }
fn policy(index: u32) -> u32 { return tables[pool_start() + 19360u + index]; }
fn selected_cell(depth: u32, floor: u32, lane: u32) -> u32 {
    if (parameters.flags & 12u) == 0u || depth < floor { return tables[depth]; }
    let at = depth*parameters.n + lane;
    return (firsts[at] >> 12u) | ((shifts[at] >> 12u) << 4u);
}
fn allowed(cell: u32, code: u32) -> bool {
    return (parameters.flags & 32u) == 0u || code/3u != 0u || policy(cell) != 0u;
}
fn rank_at(cell:u32, ranks:ptr<function,array<u32,160>>) -> u32 {
    if (parameters.flags&12u)==0u {return tables[pool_start()+19200u+cell];}
    return (*ranks)[cell];
}
// Return the last assigned rank in a closed proper component. Following actual
// paired endpoints preserves same-side U-turns and ignores drawing crossings.
fn gold_closed_rank(start: u32, lane: u32, ranks: ptr<function, array<u32,160>>) -> i32 {
    let start_code = u32(board(start, lane));
    for (var port = 0u; port < 33u; port++) {
        let partner = gold_partner(start_code, port);
        if partner >= 33u || partner < port { continue; }
        var cell = start; var incoming = port; var latest = rank_at(start,ranks);
        for (var count = 1u; count <= parameters.gold_total; count++) {
            let outgoing = gold_partner(u32(board(cell, lane)), incoming);
            if outgoing >= 33u { break; }
            latest = max(latest, rank_at(cell,ranks));
            let side = outgoing / 11u;
            incoming = other_side(cell, side) * 11u + 10u - outgoing % 11u;
            cell = neighbor(cell, side);
            if board(cell, lane) < 0 { break; }
            if cell == start && incoming == port {
                if count < parameters.gold_total { return i32(latest); }
                break;
            }
        }
    }
    return -1;
}
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
// bucket() already covers the first two occupied sides of this candidate.
fn matches_after_bucket(cell: u32, code: u32, lane: u32) -> bool {
    if board(neighbor(cell, 0u), lane) < 0 || board(neighbor(cell, 1u), lane) < 0 { return true; }
    let other = board(neighbor(cell, 2u), lane);
    return other < 0 || face(code, 2u) == reverse_face(u32(other), other_side(cell, 2u));
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
        if !taken(code / 3u, lane) && matches_after_bucket(cell, code, lane) { return true; }
    }
    return false;
}
struct Choices { count: u32, index: u32, pieces: vec3<u32> }
fn choices(cell: u32, lane: u32, limit: u32, collect: bool) -> Choices {
    let bucket_values = bucket(cell,lane); var result: Choices; var unique = 0u;
    for(var j=0u;j<bucket_values.y;j++) {
        let code=pool(bucket_values.x+j);
        if taken(code/3u,lane) || !allowed(cell,code) || !matches_after_bucket(cell,code,lane) {continue;}
        result.count++;result.index=j;
        if collect {
            let tile=code/3u;var known=false;
            for(var k=0u;k<unique;k++) {if result.pieces[k]==tile {known=true;}}
            if !known {if unique==3u {result.count=4u;return result;}result.pieces[unique]=tile;unique++;}
        } else if result.count>=limit {return result;}
    }
    if collect {result.count=unique;}
    return result;
}
fn forward(lane:u32) -> bool {
    for(var cell=0u;cell<160u;cell++) {if board(cell,lane)<0 && choices(cell,lane,1u,false).count==0u {return false;}}
    return true;
}
fn hall(lane:u32) -> bool {
    var domains:array<vec3<u32>,24>;var sizes:array<u32,24>;var count=0u;
    for(var cell=0u;cell<160u;cell++) {if board(cell,lane)<0 {
        let found=choices(cell,lane,480u,true);
        if found.count==0u {return false;}
        if found.count<=3u && count<24u {domains[count]=found.pieces;sizes[count]=found.count;count++;}
    }}
    for(var a=0u;a<count;a++) {for(var b=a+1u;b<count;b++) {
        var values:array<u32,6>;var size=0u;
        for(var which=0u;which<2u;which++) {let row=select(a,b,which!=0u);
            for(var k=0u;k<sizes[row];k++) {let value=domains[row][k];var found=false;
                for(var j=0u;j<size;j++) {if values[j]==value {found=true;}}
                if !found {values[size]=value;size++;}
        }}
        if size<2u {return false;}if size>2u {continue;}
        for(var c=b+1u;c<count;c++) {var total=size;
            for(var k=0u;k<sizes[c];k++) {let value=domains[c][k];var found=false;
                for(var j=0u;j<size;j++) {if values[j]==value {found=true;}}
                if !found {total++;}
            }
            if total<3u {return false;}
        }
    }}
    return true;
}
// cell, unique candidate bucket index, forced flag
fn choose(lane:u32) -> vec3<u32> {
    var best=160u;var minimum=481u;var one_index=0u;
    for(var rank=0u;rank<160u;rank++) {let cell=tables[rank];if board(cell,lane)>=0 {continue;}
        if best==160u {best=cell;}
        let found=choices(cell,lane,minimum,false);
        if found.count<minimum {if (parameters.flags&8u)!=0u || found.count<=1u {best=cell;}minimum=found.count;one_index=found.index;}
        if found.count==0u {return vec3<u32>(cell,0u,0u);}
        if found.count==1u && (parameters.flags&4u)!=0u {return vec3<u32>(cell,found.index,1u);}
    }
    return vec3<u32>(best,one_index,0u);
}
fn nonempty(code:u32,side:u32) -> bool {
    for(var port=side*11u;port<(side+1u)*11u;port++) {if gold_partner(code,port)<33u {return true;}}
    return false;
}
fn connected(lane:u32) -> bool {
    var start=160u;
    for(var c=0u;c<160u && start==160u;c++) {if board(c,lane)>=0 {
        for(var side=0u;side<3u;side++) {if nonempty(u32(board(c,lane)),side) {start=c;break;}}
    }}
    if start==160u {return true;}
    var seen:array<bool,160>;var queue:array<u32,160>;var read=0u;var write=1u;queue[0]=start;seen[start]=true;
    while read<write {let cell=queue[read];read++;let code=board(cell,lane);
        for(var side=0u;side<3u;side++) {let nb=neighbor(cell,side);let other=board(nb,lane);
            if seen[nb] {continue;}if code>=0 && !nonempty(u32(code),side) {continue;}
            if other>=0 && !nonempty(u32(other),other_side(cell,side)) {continue;}
            seen[nb]=true;queue[write]=nb;write++;
    }}
    for(var cell=0u;cell<160u;cell++) {if board(cell,lane)>=0 && !seen[cell] {
        for(var side=0u;side<3u;side++) {if nonempty(u32(board(cell,lane)),side) {return false;}}
    }}
    return true;
}
fn shared_ok(lane:u32) -> bool {
    for(var i=0u;i<policy(160u);i++) {let offset=161u+i*321u;var all=true;
        for(var j=0u;j<policy(offset);j++) {if board(policy(offset+1u+2u*j),lane)!=i32(policy(offset+2u+2u*j)) {all=false;break;}}
        if all {return false;}
    }
    return true;
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
    if lane >= n { return; }
    let state = metadata[3u * n + lane];
    if state != 0u && !(parameters.gold_total != 0u && state == 1u && metadata[11u * n + lane] != 255u) { return; }
    var depth = metadata[lane]; var high = metadata[n + lane]; let floor = metadata[2u * n + lane];
    var random = metadata[4u * n + lane];
    var tried = 0u; var placed = 0u; var backed = 0u;
    var ranks:array<u32,160>;
    if (parameters.flags&12u)!=0u {
        for(var i=0u;i<depth;i++) {ranks[selected_cell(i,floor,lane)]=i;}
    }
    var initial_ok=true;
    if (parameters.flags&32u)!=0u {for(var i=0u;i<depth;i++) {let cell=selected_cell(i,floor,lane);
        if !allowed(cell,u32(board(cell,lane))) {initial_ok=false;break;}}}
    if !initial_ok {
        for(var cell=0u;cell<160u;cell++) {board_used[cell*n+lane]=0xffffffffu;}
        for(var word=0u;word<5u;word++) {board_used[(160u+word)*n+lane]=0u;}
        metadata[lane]=0u;metadata[3u*n+lane]=2u;return;
    }
    // Recheck restored/loaded paths once. A proved bad subtree is unwound to
    // its closing placement; sibling cursors and prefix ownership stay intact.
    var work = 0u;
    if parameters.gold_total != 0u && metadata[11u * n + lane] != 255u {
        while work < parameters.budget && metadata[11u * n + lane] != 255u {
            let scan = metadata[11u * n + lane];
            if scan >= depth { metadata[11u * n + lane] = 255u; break; }
            let closed = gold_closed_rank(selected_cell(scan,floor,lane), lane, &ranks);
            work++;
            if closed < 0 {
                metadata[11u * n + lane] = select(scan + 1u, 255u, scan + 1u >= depth);
                continue;
            }
            if u32(closed) < floor {
                for (var cell = 0u; cell < 160u; cell++) { board_used[cell * n + lane] = 0xffffffffu; }
                for (var word = 0u; word < 5u; word++) { board_used[(160u + word) * n + lane] = 0u; }
                depth = 0u; metadata[3u * n + lane] = 2u; backed++; break;
            }
            while depth > u32(closed) {
                depth--; let cell = selected_cell(depth,floor,lane); let old = u32(board(cell, lane));
                board_used[cell * n + lane] = 0xffffffffu; mark(old / 3u, lane, false); backed++;
            }
            metadata[3u * n + lane] = 0u;
            metadata[11u * n + lane] = 0u;
        }
    }
    for (; work < parameters.budget && metadata[3u * n + lane] == 0u &&
            (parameters.gold_total == 0u || metadata[11u * n + lane] == 255u); work++) {
        if depth == 160u { metadata[3u * n + lane] = 1u; break; }
        let location = depth * n + lane;
        if cursors[location] == 65535u {
            var cell=tables[depth];var one_index=0u;var forced=0u;
            if (parameters.flags&12u)!=0u {let chosen=choose(lane);cell=chosen.x;one_index=chosen.y;forced=chosen.z;}
            let choice = bucket(cell, lane);
            firsts[location] = choice.x; lengths[location] = choice.y | (forced*32768u);
            shifts[location] = 0u;
            if forced!=0u {shifts[location]=one_index;}
            else if choice.y != 0u { random = random_next(random); shifts[location] = random % choice.y; }
            if (parameters.flags&12u)!=0u {firsts[location]|=(cell&15u)<<12u;shifts[location]|=(cell>>4u)<<12u;}
            cursors[location] = 0u;
        }
        let length = lengths[location]&32767u; let cursor = cursors[location];
        if cursor >= select(length,1u,(lengths[location]&32768u)!=0u) {
            if depth == floor {
                for (var cell = 0u; cell < 160u; cell++) { board_used[cell * n + lane] = 0xffffffffu; }
                for (var word = 0u; word < 5u; word++) { board_used[(160u + word) * n + lane] = 0u; }
                depth = 0u; metadata[3u * n + lane] = 2u; backed++; break;
            }
            let cell = selected_cell(depth-1u,floor,lane); let old = u32(board(cell, lane));
            board_used[cell * n + lane] = 0xffffffffu; mark(old / 3u, lane, false);
            depth--; backed++; continue;
        }
        let code = pool((firsts[location]&4095u) + (cursor + (shifts[location]&4095u)) % length);
        cursors[location] = cursor + 1u; tried++;
        let tile = code / 3u; let cell = selected_cell(depth,floor,lane);
        if !allowed(cell,code) || taken(tile, lane) || !matches_after_bucket(cell, code, lane) { continue; }
        board_used[cell * n + lane] = code; mark(tile, lane, true);
        var feasible = (parameters.flags&192u)==0u || shared_ok(lane);
        for (var side = 0u; side < 3u; side++) {
            if !feasible {break;}
            let next_cell = neighbor(cell, side);
            if board(next_cell, lane) < 0 && !possible(next_cell, lane) { feasible = false; break; }
        }
        if feasible && (parameters.flags&1u)!=0u && !forward(lane) {feasible=false;}
        if feasible && (parameters.flags&2u)!=0u && !hall(lane) {feasible=false;}
        if (parameters.flags&12u)!=0u {ranks[cell]=depth;}
        if feasible && parameters.gold_total != 0u && gold_closed_rank(cell, lane, &ranks) >= 0 { feasible = false; }
        if feasible && parameters.gold_total != 0u && (parameters.flags&16u)!=0u && !connected(lane) {feasible=false;}
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
