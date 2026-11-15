"""Pareto fronts for shared donor occupancy at bounded swap distance."""


def frontier(states):
    by_mask = {}
    for (mask, cost, moved), gain in states.items():
        by_mask.setdefault(mask, []).append((cost, -moved, -gain))
    kept = {}
    for mask, rows in by_mask.items():
        # Lower costs come first. At equal cost, visit the largest mover
        # counts first, then largest gains. Only states with enough movers
        # and at least as much gain can dominate the current state.
        previous = {}
        for cost, neg_moved, neg_gain in sorted(rows):
            moved, gain = -neg_moved, -neg_gain
            if any(count >= moved and best >= gain for count, best in previous.items()):
                continue
            kept[mask, cost, moved] = gain
            previous[moved] = max(previous.get(moved, gain), gain)
    return kept


def join_halves(left,right,size,mover_cap,swaps,work,limit):
    by_mask={}
    for (mask,cost,moved),gain in right.items():
        by_mask.setdefault(mask,[]).append((cost,moved,gain))
    full=(1<<size)-1
    result={}
    for (mask,cost,moved),gain in left.items():
        for extra,count,value in by_mask.get(full^mask,()):
            if work['joins']>=limit:
                return None
            work['joins']+=1
            if cost+extra<=2*swaps:
                key=(full,cost+extra,min(mover_cap,moved+count))
                result[key]=max(result.get(key,gain+value),gain+value)
    return frontier(result)
