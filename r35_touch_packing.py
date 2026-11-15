"""Augment a legal pack of swaps with disjoint endpoints using exact current seam scores."""

import numpy as np


def augment(model, board, pairs, codes, weights, initial, mode):
    used = np.zeros(model.n,dtype=bool)
    result = board.copy()
    for index in initial:
        a,b = pairs[index]
        if used[a] or used[b]:
            raise ValueError('A physical endpoint belongs to two base swaps.')
        result[a],result[b] = codes[index]
        used[a] = used[b] = True
    initial_score = int(model.matches(result).sum())
    if mode not in ('neutral','priced','complete','cp'):
        raise ValueError('Unknown contact augmentation mode.')
    # Only candidates disjoint from selected endpoints remain possible. The
    # fixed oriented codes for each swap retain their original physical identity.
    edges = [[] for _ in range(model.n)]
    for a,s,b,t in model.problem.edges:
        edges[a].append((s,b,t))
        if a != b:
            edges[b].append((t,a,s))

    by_cell = [[] for _ in range(model.n)]
    for i,(a,b) in enumerate(pairs):
        by_cell[a].append(i)
        by_cell[b].append(i)

    def marginal(current,i):
        a,b = pairs[i]
        value = 0
        for c,new in zip((a,b),codes[i]):
            for side,neighbor,other in edges[c]:
                if neighbor == c:
                    value += int(model.masks[new,side] == model.reverse[model.masks[new,other]])
                    value -= int(model.masks[current[c],side] == model.reverse[model.masks[current[c],other]])
                elif neighbor == b and c == a:
                    value += int(model.masks[new,side] == model.reverse[model.masks[codes[i][1],other]])
                    value -= int(model.masks[current[c],side] == model.reverse[model.masks[current[b],other]])
                elif neighbor != a or c != b:
                    partner = model.reverse[model.masks[current[neighbor],other]]
                    value += int(model.masks[new,side] == partner)-int(model.masks[current[c],side] == partner)
        return value

    def improve(allow_signed):
        current = result.copy()
        occupied = used.copy()
        chosen = list(initial)
        total = 0
        checked = 0
        pending = range(len(pairs))
        while pending:
            changed = set()
            for i in pending:
                a,b = pairs[i]
                if occupied[a] or occupied[b]:
                    continue
                checked += 1
                value = marginal(current,i)
                if value <= 0 or (not allow_signed and value != weights[i]):
                    continue
                current[a], current[b] = codes[i]
                occupied[a] = occupied[b] = True
                chosen.append(i)
                total += value
                for c in (a,b):
                    changed.update(n for _,n,_ in edges[c])
            # A rejected fixed-code move can acquire a different marginal only
            # when its physical neighbor changes. Earlier candidates have priority.
            pending = sorted({i for c in changed if not occupied[c] for i in by_cell[c]
                              if not occupied[pairs[i][0]] and not occupied[pairs[i][1]]})
        if int(model.matches(current).sum())!=initial_score+total:
            raise ArithmeticError('Packed contact augmentation differs from physical board score.')
        return current,chosen,total,checked

    best = improve(False)
    work = dict(base_swaps=len(initial),neutral_augmented=len(best[1])-len(initial),
                neutral_gain=best[2],neutral_candidates=best[3])
    if mode in ('priced','complete','cp'):
        other = improve(True)
        work.update(priced_augmented=len(other[1])-len(initial),priced_gain=other[2],priced_candidates=other[3])
        if other[2] > best[2]:
            best = other
    if mode in ('complete','cp') and pairs:
        from r35_packing_exact import exact_components
        initial_selected = set(best[1])
        chosen,metrics = exact_components(model,board,pairs,codes,weights,initial_selected,by_cell,marginal)
        work.update(metrics)
        if mode == 'cp' and metrics.get('exact_over_limit_components') and len(pairs)<=512:
            from r35_fixed_swap_cp import exact
            chosen,report = exact(model,board,pairs,codes,chosen,.2)
            work['fixed_cp_calls'] = 1
            work['fixed_cp_branches'] = report['branches']
            work['fixed_cp_optimal'] = int(report['optimal'])
            work['fixed_cp_extra'] = report.get('score',report['prior'])-report['prior']
        rebuilt = board.copy()
        for i in chosen:
            a,b = pairs[i]
            rebuilt[a],rebuilt[b] = codes[i]
        gain = int(model.matches(rebuilt).sum())-initial_score
        if gain < best[2]:
            raise ArithmeticError('Exact component lost a valid swap pack.')
        best = rebuilt,chosen,gain,best[3]
    work['swaps_added'] = len(best[1])-len(initial)
    work['seams_added'] = best[2]
    return best[0],best[1],work
