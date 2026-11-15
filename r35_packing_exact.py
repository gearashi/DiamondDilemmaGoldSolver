"""Optimize small fixed-orientation swap groups with their signed seam interactions."""
from collections import Counter

from r34_swap_packing import vertices


def exact_components(model, board, pairs, codes, weights, selected, by_cell, marginal, max_size=18):
    work = Counter()
    visited = set()
    updated = set(selected)
    for start,incident in enumerate(by_cell):
        if not incident or start in visited:
            continue
        pending = [start]
        visited.add(start)
        moves = set()
        for cell in pending:
            adjacent = [int(c) for c in model.problem.neighbors[cell] if c >= 0 and by_cell[c]]
            for i in by_cell[cell]:
                moves.add(i)
                a,b = pairs[i]
                adjacent.append(b if a==cell else a)
            for c in adjacent:
                if c not in visited:
                    pending.append(c)
                    visited.add(c)
        if len(moves)>max_size:
            work['exact_over_limit_components'] += 1
            continue
        ids = sorted(moves)
        n = len(ids)
        exclusion = [0]*n
        interaction = [[0]*n for _ in ids]
        for i,slot in enumerate(ids):
            a,b = pairs[slot]
            changed = board.copy()
            changed[a],changed[b] = codes[slot]
            for j,other in enumerate(ids[:i]):
                x,y = pairs[other]
                if a in (x,y) or b in (x,y):
                    exclusion[i] |= 1<<j
                    exclusion[j] |= 1<<i
                else:
                    corr = marginal(changed,other)-weights[other]
                    interaction[i][j] = interaction[j][i] = corr
        preferred = [i for i,slot in enumerate(ids) if slot in selected]
        old = sum(weights[ids[i]] for i in preferred)
        old += sum(interaction[i][j] for k,i in enumerate(preferred) for j in preferred[:k])
        best_value,best = old,preferred
        def search(active,adjusted,score,taken):
            nonlocal best_value,best
            work['exact_component_states'] += 1
            if score > best_value:
                best_value,best = score,taken
            remaining = list(vertices(active))
            upper = score+sum(max(0,adjusted[i]) for i in remaining)
            upper += sum(max(0,interaction[i][j]) for k,i in enumerate(remaining) for j in remaining[:k])
            if not remaining or upper <= best_value:
                return
            pivot = max(remaining,key=lambda i:((exclusion[i]&active).bit_count(),adjusted[i]))
            tail = active & ~(1<<pivot)
            search(tail & ~exclusion[pivot],[v+d for v,d in zip(adjusted,interaction[pivot])],
                   score+adjusted[pivot],taken+[pivot])
            search(tail,adjusted,score,taken)
        search((1<<n)-1,[weights[slot] for slot in ids],0,[])
        work['exact_components'] += 1
        if best_value>old:
            work['exact_extra_seams'] += best_value-old
            updated.difference_update(moves)
            updated.update(ids[i] for i in best)
    return sorted(updated),dict(work)
