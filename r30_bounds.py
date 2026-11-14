"""Local defect, rotation agreement and partially specified exchange certificates."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from constraint_connectivity_round26 import assignment_upper_bound


def cell_mask(cells):
    value = 0
    for c in cells:
        value |= 1 << int(c)
    return value


def defect_witnesses(bound):
    """Disjoint seam sets can each force a separate failure at fixed identities."""
    problem = bound.model.problem
    incident = [[] for _ in range(bound.n)]
    singles = []
    ends = []
    for e, (a, s, b, t) in enumerate(problem.edges):
        matches = (bound.masks[bound.tiles[a], :, s, None] ==
                   bound.model.reverse[bound.masks[bound.tiles[b], :, t]])
        footprint = cell_mask((a, b))
        ends.append(footprint)
        if not matches.any():
            singles.append((1 << e, footprint))
        else:
            incident[a].append((e, matches.any(1)))
            incident[b].append((e, matches.any(0)))
    conflicts = []
    for sides in incident:
        local = []
        for width in range(2, len(sides)+1):
            for group in combinations(sides, width):
                edges = cell_mask(e for e, _ in group)
                if any(edges & old == old for old in local):
                    continue
                if not np.logical_and.reduce([rotations for _, rotations in group]).any():
                    local.append(edges)
                    cells = 0
                    for e, _ in group:
                        cells |= ends[e]
                    conflicts.append((edges, cells))
    return singles, sorted(set(conflicts))


def defect_packings(bound, limit=256):
    """Generate verified edge-disjoint packings, keeping a bounded set of alternatives."""
    singles, conflicts = defect_witnesses(bound)
    result = []
    def partition(pos, used, chosen):
        if len(result) >= limit:
            return
        if pos == len(conflicts):
            result.append(tuple(singles) + tuple(chosen))
            return
        witness = conflicts[pos]
        if not used & witness[0]:
            partition(pos+1, used | witness[0], chosen+[witness])
        partition(pos+1, used, chosen)
    # A long list can arise on larger user boards. Greedy order supplies a valid
    # packing even when the alternate orderings exceed the enumeration budget.
    if len(conflicts)>32:
        for seq in (conflicts, conflicts[::-1], sorted(conflicts, key=lambda x:x[1].bit_count())):
            used = 0
            selected = list(singles)
            for witness in seq:
                if not used & witness[0]:
                    used |= witness[0]
                    selected.append(witness)
            result.append(tuple(selected))
    else:
        partition(0, 0, [])
    return sorted(result,key=len,reverse=True)


def defect_cores(bound, target, packings=None):
    required = len(bound.model.problem.edges)-target+1
    if required<=0:
        return (0,)
    cores = set()
    if packings is None:
        packings = defect_packings(bound)
    for packing in packings:
        if len(packing)<required:
            break
        # Select a sufficient subpacking. All choices fit for the three release centers.
        if len(packing)==required:
            groups = (packing,)
        else:
            from itertools import islice
            groups = islice(combinations(packing,required),256)
        for group in groups:
            held = 0
            for _,cells in group:
                held |= cells
            cores.add(held)
    minimal = []
    for mask in sorted(cores,key=lambda m:(m.bit_count(),m)):
        if all(other & mask != other for other in minimal):
            minimal.append(mask)
    return tuple(minimal)


def can_hit_cores(cores, moved, allowed, places):
    """Check whether all remaining movement clauses can fit in the support budget."""
    if places<0:
        return False
    rows = tuple(sorted({c & allowed for c in cores if not c & moved},
                        key=lambda c:(c.bit_count(),c)))
    @lru_cache(None)
    def visit(missing,left):
        if not missing:
            return True
        if not missing[0] or not left:
            return False
        bits = missing[0]
        while bits:
            pick = bits & -bits
            if visit(tuple(row for row in missing if not row & pick),left-1):
                return True
            bits ^= pick
        return False
    return visit(rows,places)


def budget_upper(bound, packings, swaps, allowed=None):
    """Lower-bound persistent defects after <= swaps among allowed cells.

    A seam-disjoint packing starts with one mandatory defect per row. At most
    two physical identities change per swap. Maximize how many distinct rows
    these changes might repair, letting rotations everywhere remain free.
    """
    if swaps < 0:
        raise ValueError("Negative swap budget.")
    allowed = set(map(int,bound.model.free)) if allowed is None else set(allowed)
    if not allowed.issubset(range(bound.n)):
        raise ValueError("Invalid cell.")
    parts = len(bound.model.problem.edges)
    for packing in packings:
        hits = {tuple(i for i,(_,w) in enumerate(packing) if w & (1<<c)) for c in allowed}
        masks = sorted({cell_mask(group) for group in hits if group},key=int.bit_count,reverse=True)
        budget = min(2*swaps,len(masks))
        repairable = 0
        for mask in masks:
            repairable |= mask
        ceiling = min(repairable.bit_count(),sum(mask.bit_count() for mask in masks[:budget]))
        # Exact maximum coverage on small residuals. Limits only weaken the bound.
        if len(masks)<=20 and budget<=4:
            ceiling = 0
            for subset in combinations(masks,budget):
                union = 0
                for mask in subset:
                    union |= mask
                ceiling = max(ceiling,union.bit_count())
        parts = min(parts,len(bound.model.problem.edges)-len(packing)+ceiling)
    return parts


class ResponseGroups:
    """Expected seam bit vectors at a cell, relative to its allowed neighbors."""
    def __init__(self,bound):
        self.bound = bound
        self.table = {}
        self.evaluations = {}

    def groups(self,cell,donors,neighbor_domains=None):
        b = self.bound
        sides = b.model.problem.sides
        layout = []
        for side,nb in enumerate(b.model.problem.neighbors[cell]):
            if nb<0:
                continue
            options = (neighbor_domains or {}).get(int(nb),(int(b.tiles[nb]),))
            layout.append((side,int(sides[cell,side]),tuple(sorted(set(options)))))
        if any(not 0<=j<b.n for _,_,opts in layout for j in opts):
            raise ValueError('Invalid neighbor identity.')
        groups = defaultdict(list)
        for tile in donors:
            if not 0<=tile<b.n:
                raise ValueError('Invalid donor identity.')
            rotations = tuple(np.flatnonzero(b.model.legal[cell,3*tile:3*tile+3]))
            key = (tile,rotations,tuple(layout))
            if key not in self.table:
                rows = []
                for r in rotations:
                    response = []
                    for side,other,options in layout:
                        reverse = b.model.reverse[b.masks[list(options),:,other]]
                        response.extend((reverse==b.masks[tile,r,side]).flat)
                    rows.append(np.packbits(np.asarray(response,dtype=np.uint8)).tobytes())
                self.table[key] = tuple(sorted(rows))
            groups[self.table[key]].append(int(tile))
        return dict(groups)


class RotationRegions:
    """Pack joint star discounts on a fixed physical permutation."""
    def __init__(self,bound,max_group=3,*,rings=False):
        self.bound = bound
        self.factors = {}
        self.footprints = {}
        neighbors = bound.model.problem.neighbors
        groups = {tuple(sorted((int(a),int(b)))) for a,_,b,_ in bound.model.problem.edges}
        if max_group>=3:
            groups.update(tuple(sorted((c,int(a),int(b)))) for c,ns in enumerate(neighbors)
                          for a,b in combinations((v for v in ns if v>=0),2) if a!=b)
        if rings:
            def trace(path):
                c = path[-1]
                for nb in neighbors[c]:
                    nb = int(nb)
                    if nb==path[0] and len(path)>=3:
                        groups.add(tuple(sorted(path)))
                    elif nb>path[0] and nb not in path and len(path)<max_group:
                        trace(path+(nb,))
            for c in range(bound.n):
                trace((c,))
        for group in sorted(groups):
            delta, footprint = self.price(group)
            if delta>0:
                self.factors[group] = delta
                self.footprints[group] = footprint

    def price(self,group):
        b = self.bound
        index = {v:i for i,v in enumerate(group)}
        values = np.array(list(product(range(3),repeat=len(group))),dtype=int)
        total = np.zeros(len(values),int)
        touched = set(group)
        for c in group:
            tile = b.tiles[c]
            for side,nb in enumerate(b.neighbors[c]):
                if nb==b.n:
                    continue
                touched.add(int(nb))
                mask = b.masks[tile,values[:,index[c]],side]
                if nb in index:
                    other = b.model.problem.sides[c,side]
                    total += mask==b.model.reverse[b.masks[b.tiles[nb],values[:,index[nb]],other]]
                else:
                    total += b.support[mask,b.tiles[nb]]
        return int(b.base[list(group)].sum()-total.max()),cell_mask(touched)

    def discount(self,moved=0):
        """Only unchanged complete footprints have the cached base-star values."""
        rows = [(cell_mask(g),v) for g,v in self.factors.items() if not moved & self.footprints[g]]
        if not rows:
            return 0
        rows.sort(key=lambda r:-r[1]/r[0].bit_count())
        if len(rows)>16:
            used = best = 0
            for mask,v in rows:
                if not used & mask:
                    used |= mask
                    best += v
            return best
        @lru_cache(None)
        def pack(index,used):
            if index==len(rows):
                return 0
            mask,v = rows[index]
            other = pack(index+1,used)
            return other if used & mask else max(other,v+pack(index+1,used|mask))
        return pack(0,0)


def _assignment(weights, allowed):
    if len(weights)==0:
        return 0
    if not allowed.any(1).all() or not allowed.any(0).all():
        return None
    shift = int(weights[allowed].min())
    rows = [[int(w)-shift if ok else None for w,ok in zip(row,flags)]
            for row,flags in zip(weights,allowed)]
    result = assignment_upper_bound(rows)
    if not all(allowed[c,j] for c,j in enumerate(result['assignment'])):
        return None
    return result['bound']+len(rows)*shift


def family_upper(clusters, active, *, fixed_donors=None, candidates=None, extra=None):
    """Union of a specified interaction cluster and ANY disjoint isolated changes.

    All extra moved cells must have disjoint closed stars both from the given
    active set and one another. The assignment relaxes that last condition.
    active may itself contain several separate groups. Fixed donor IDs are the
    original positions of the pieces, not physical tile IDs.
    """
    b = clusters.bound
    active = tuple(sorted(set(map(int,active))))
    fixed = dict(fixed_donors or {})
    if (not set(active)<=set(range(b.n)) or not set(fixed)<=set(active)
            or any(i not in range(b.n) for i in fixed.values())
            or (extra is not None and extra<0)):
        raise ValueError("Invalid physical family.")
    candidates = set(map(int,b.model.free)) if candidates is None else set(candidates)
    if not candidates<=set(range(b.n)):
        raise ValueError("Invalid candidate cells.")
    rest = sorted(i for i in candidates-set(active) if not any(clusters.close[i,c] for c in active))
    pool = list(active)+rest
    if not pool:
        return b.total
    if len(set(fixed.values()))!=len(fixed) or not set(fixed.values())<=set(pool):
        return -1
    pos = {v:i for i,v in enumerate(pool)}
    allowed = b.model.legal.reshape(b.n,b.n,3)[pool][:,b.tiles[pool]].any(2)
    for i in active:
        allowed[pos[i],pos[i]] = False
    extra_indices = np.arange(len(active),len(pool),dtype=int)
    rest_mask = np.eye(len(pool),dtype=bool)
    rest_mask |= ~clusters.close[np.ix_(pool,pool)]
    # A remote changed singleton cannot send to a distinct close singleton.
    allowed[len(active):,len(active):] &= rest_mask[len(active):,len(active):]
    for i,j in fixed.items():
        if not allowed[pos[i],pos[j]]:
            return -1
        allowed[pos[i],:] = False
        allowed[:,pos[j]] = False
        allowed[pos[i],pos[j]] = True
    if not allowed.any(1).all():
        return -1
    matrix = np.zeros((len(pool),len(pool)),dtype=int)
    matrix[len(active):] = clusters.delta[np.ix_(rest,pool)]
    const = b.total
    affected = defaultdict(list)
    for i in active:
        for k,v in enumerate(clusters.affected[i]):
            affected[v].append((i,k))
    for v,hits in affected.items():
        if len(hits)==1:
            i,k = hits[0]
            matrix[pos[i]] += clusters.local[i][pool,k]
            continue
        const -= int(b.base[v])
        centers = pool if v in active else [v]
        scores = np.zeros((len(centers),3),dtype=int)
        for side,nb in enumerate(b.neighbors[v]):
            if nb==b.n:
                continue
            possible = [pool[j] for j in np.flatnonzero(allowed[pos[nb]])] if nb in active else [nb]
            masks = b.masks[b.tiles[centers],:,side,None]
            scores += b.support[masks,b.tiles[possible]].max(-1)
        if v in active:
            matrix[pos[v]] += scores.max(1)
        else:
            const += int(scores.max())
    # Penalize extra moved rows and restore at most the specified count.
    penalties = (0,1,2) if extra is not None else (0,)
    best = None
    dirty = np.zeros_like(allowed)
    dirty[extra_indices,:] = True
    np.fill_diagonal(dirty,False)
    for penalty in penalties:
        optimum = _assignment(matrix-penalty*dirty,allowed)
        if optimum is None:
            return -1
        upper = const+optimum+penalty*(extra or 0)
        best = upper if best is None else min(best,upper)
    return best


class RotationForest:
    """Integer upper from two complementary forests with cost-shift consistency."""
    def __init__(self,model):
        self.model = model
        self.cache = {}
        self.nodes = self.hits = 0
        all_edges = tuple(range(len(model.problem.edges)))
        def forest(order):
            parents = list(range(model.n))
            def root(c):
                while parents[c]!=c:
                    parents[c] = parents[parents[c]]
                    c = parents[c]
                return c
            selected = set()
            for e in order:
                a,_,b,_ = model.problem.edges[e]
                x,y = root(a),root(b)
                if x!=y:
                    parents[x] = y
                    selected.add(e)
            return selected
        self.trees = []
        for order in (all_edges,all_edges[::-1]):
            a = forest(order)
            b = forest([e for e in order if e not in a])
            # Any edges outside the two forests are priced independently.
            self.trees.append((sorted(a),sorted(b),sorted(set(all_edges)-a-b)))

    def _price(self,tiles,edges,shift,legal):
        n = len(tiles)
        adjacent = [[] for _ in tiles]
        m = self.model
        for e in edges:
            c,s,d,t = m.problem.edges[e]
            grid = (m.masks[3*tiles[c]:3*tiles[c]+3,s,None] ==
                    m.reverse[m.masks[3*tiles[d]:3*tiles[d]+3,t]]).astype(int)
            adjacent[c].append((d,grid))
            adjacent[d].append((c,grid.T))
        score = 0
        chosen = np.zeros(n,dtype=int)
        seen = set()
        for root in range(n):
            if root in seen:
                continue
            stack = [(root,-1)]
            order = []
            seen.add(root)
            while stack:
                c,parent = stack.pop()
                order.append((c,parent))
                for d,_ in adjacent[c]:
                    if d!=parent:
                        seen.add(d)
                        stack.append((d,c))
            messages = {}
            for c,parent in reversed(order):
                value = np.array(shift[c],dtype=int)
                for d,_ in adjacent[c]:
                    if d!=parent:
                        value += messages[d][0]
                # Only three rotations. Large negative represents excluded
                # rotations after all allowed contributions have been added.
                value[~legal[c]] = value.min()-int(np.ptp(value))-4
                if parent<0:
                    chosen[c] = int(value.argmax())
                    score += int(value.max())
                    continue
                table = next(grid for d,grid in adjacent[c] if d==parent)
                key = (value.tobytes(),table.tobytes())
                item = self.cache.get(key)
                if item is None:
                    relative = table+value[:,None]
                    item = relative.max(0),relative.argmax(0)
                    if len(self.cache)>=4096:
                        self.cache.clear()
                    self.cache[key] = item
                    self.nodes += 1
                else:
                    self.hits += 1
                messages[c] = item
            for c,parent in order:
                if parent>=0:
                    chosen[c] = messages[c][1][chosen[parent]]
        return score,chosen

    def upper(self,tiles,rounds=3):
        if not rounds>0:
            raise ValueError("Positive round budget required.")
        m = self.model
        tiles = np.asarray(tiles)
        if tiles.shape!=(m.n,) or not np.array_equal(np.sort(tiles),np.arange(m.n)):
            raise ValueError("Expected complete physical permutation.")
        legal = m.legal.reshape(m.n,m.n,3)[np.arange(m.n),tiles]
        if not legal.any(1).all():
            return -1
        upper = len(m.problem.edges)
        for first,second,remaining in self.trees:
            remainder = 0
            for e in remaining:
                c,s,d,t = m.problem.edges[e]
                grid = m.masks[3*tiles[c]:3*tiles[c]+3,s,None]==m.reverse[m.masks[3*tiles[d]:3*tiles[d]+3,t]]
                remainder += int(np.any(grid & legal[c,:,None] & legal[d,None,:]))
            shift = np.zeros((m.n,3),int)
            for it in range(rounds):
                lhs,x = self._price(tiles,first,shift,legal)
                rhs,y = self._price(tiles,second,-shift,legal)
                upper = min(upper,remainder+lhs+rhs)
                if np.array_equal(x,y):
                    break
                # Integer cancellation leaves the original objective invariant.
                # Update a subset (deterministic) to avoid synchronous oscillation.
                mismatch = np.flatnonzero(x!=y)
                pick = mismatch[(it%2)::2] if len(mismatch)>1 else mismatch
                shift[pick,x[pick]] -= 1
                shift[pick,y[pick]] += 1
        return upper


def report_family(clusters,cells,target,extra,*,candidates=None,nodes=128):
    """Refine a specified interacting cluster by donor assignments at open stars.

    With no candidates left uncovered, status 'closed' certifies the full
    isolated-remainder family. A budget can only return 'open'.
    """
    cells = tuple(cells)
    seen = closed = 0
    root_upper = None
    bound = clusters.bound
    pool = [c for c in (range(bound.n) if candidates is None else candidates)
            if c in cells or not any(clusters.close[c,a] for a in cells)]
    pool = sorted(set(pool)|set(cells))
    def visit(pins):
        nonlocal seen,closed,root_upper
        if seen>=nodes:
            return False
        seen += 1
        upper = family_upper(clusters,cells,fixed_donors=pins,candidates=candidates,extra=extra)
        if not pins:
            root_upper = upper
        if upper<2*target:
            closed += 1
            return True
        if len(pins)==len(cells):
            return False
        cell = cells[len(pins)]
        for j in pool:
            if j!=cell and j not in pins.values() and not visit(pins|{cell:j}):
                return False
        return True
    status = visit({})
    return dict(closed=status,upper_twice=root_upper,nodes=seen,proved=closed)


def grouped_rotation_bound(bound,cell,donors,*,neighbor_domains=None,response_cache=None):
    """Upper for each donor's central star when neighbor identities may vary.

    Use a representative only for external scoring. Callers retain every donor
    ID and test legal physical assignment/cycle constraints independently.
    """
    donors = tuple(map(int,donors))
    cache = response_cache if response_cache is not None else ResponseGroups(bound)
    if cache.bound is not bound:
        raise ValueError("Response cache belongs to a different physical context.")
    domains = {int(c):tuple(sorted(set(options))) for c,options in (neighbor_domains or {}).items()}
    response = cache.groups(cell,donors,domains)
    scores = {}
    widths = tuple(len(domains.get(int(nb),(int(bound.tiles[nb]),)))
                   for nb in bound.neighbors[cell] if nb!=bound.n)
    if any(w==0 for w in widths):
        return {},len(response)
    for key,options in response.items():
        if (key,widths) in cache.evaluations:
            scores.update(dict.fromkeys(options,cache.evaluations[key,widths]))
            continue
        t = options[0]
        vals = np.zeros(3,dtype=int)
        for s,nb in enumerate(bound.neighbors[cell]):
            if nb==bound.n:
                continue
            neighbors = domains.get(int(nb),(int(bound.tiles[nb]),))
            if not len(neighbors):
                return {},len(response)
            vals += bound.support[bound.masks[t,:,s,None],list(neighbors)].max(1)
        legal = bound.model.legal[cell,3*t:3*t+3]
        value = int(vals[legal].max()) if legal.any() else -1
        cache.evaluations[key,widths] = value
        scores.update(dict.fromkeys(options,value))
    return scores,len(response)


def pattern_distance_upper(bound,allowed,swaps,*,max_nodes=10000):
    """Exact regional one-or-more-swap potential, capped by a defect certificate.

    Score by a fixed-exterior physical-star relaxation with rotations open.
    Cache the whole board contribution as only the allowed physical vector
    changes. A budget boundary returns the initial safe coverage estimate.
    """
    allowed = tuple(sorted(set(map(int,allowed)) & set(map(int,bound.model.free))))
    if swaps<0 or max_nodes<1:
        raise ValueError("Invalid budget.")
    coarse = budget_upper(bound,defect_packings(bound),swaps,allowed)
    identity = tuple(range(len(allowed)))
    frontier = {identity}
    seen = {identity}
    best = bound.total//2
    for level in range(swaps):
        next_level = set()
        for row in frontier:
            for i,j in combinations(range(len(row)),2):
                state = list(row)
                state[i],state[j] = state[j],state[i]
                key = tuple(state)
                if key in seen:
                    continue
                if len(seen)>=max_nodes:
                    return dict(upper=coarse,complete=False,nodes=len(seen),coarse=coarse)
                seen.add(key)
                next_level.add(key)
                # Even a physically illegal intermediate may lead to legal
                # endpoints. Leaving it in this relaxation overestimates safely.
                changed = [k for k in range(len(row)) if key[k]!=k]
                if changed:
                    rank = {k:x for x,k in enumerate(changed)}
                    score = int(bound.bounds([[allowed[k] for k in changed]],
                                             [rank[key[k]] for k in changed])[0])
                    best = max(best,score)
        frontier = next_level
        if not frontier:
            break
    return dict(upper=min(coarse,best),complete=True,nodes=len(seen),coarse=coarse)


def family_case_upper(clusters,active,remaining,*,merge_slots=1,max_parts=64):
    """Cover a cluster plus unrestricted remaining movers by overlap branching.

    The first branch permits only distant singletons. Every violation either
    touches this active cluster or creates another interacting pair in the
    remainder. Promote the relevant changed cell(s) to the active set. A
    truncated decomposition falls back to the parent's universal upper bound.
    """
    active = tuple(sorted(set(active)))
    pool = tuple(sorted(set(remaining)-set(active)))
    b = clusters.bound
    done = {}
    parts = 0
    def solve(cluster,available,slots):
        nonlocal parts
        key = cluster,available,slots
        if key in done:
            return done[key]
        if parts>=max_parts:
            return 2*len(b.model.problem.edges)
        parts += 1
        upper = family_upper(clusters,cluster,candidates=available)
        collisions = [(i,) for i in available if any(clusters.close[i,j] for j in cluster)]
        collisions += [(i,j) for i,j in combinations(available,2) if clusters.close[i,j]]
        if collisions and (slots<=0 or len(collisions)>max_parts-parts):
            return 2*len(b.model.problem.edges)
        for event in collisions:
            more = tuple(sorted(set(cluster)|set(event)))
            left = tuple(i for i in available if i not in event)
            upper = max(upper,solve(more,left,slots-1))
        done[key]=upper
        return upper
    return dict(upper_twice=solve(active,pool,merge_slots),parts=parts)


def issued_pair_anchors(bound,target,extra):
    """Load the checked release list for this exact physical center and domains."""
    from pathlib import Path
    from r29_exchange_route import physical_hash, legal_cell_rotations
    from r28_exchange import domain_binding, read_json
    folder = Path(__file__).parent/'docs'
    manifest = read_json(folder/'R29_EXCHANGE_CENTERS.json')
    binding = domain_binding(bound.model)
    if manifest['domain']!=binding or target<manifest['target']:
        return set()
    current = bound.tiles.tolist()
    key = physical_hash(bound.tiles)
    centers = manifest['centers']
    matched = next((row for row in centers if row['sha256']==key and row['pieces']==current),None)
    perm = range(bound.n)
    if matched is None:
        for transform in legal_cell_rotations(bound.model):
            restored = [current[c] for c in transform]
            matched = next((row for row in centers if row['pieces']==restored),None)
            if matched:
                perm = transform
                break
    if matched is None:
        return set()
    files = ['R30_BOUNDS_ALL_SIZES.json']
    if extra<=4:
        files.append('R30_BOUNDS_RESULTS.json')
    result = set()
    for file in files:
        record = read_json(folder/file).get(matched['sha256'][:16],{})
        if (record.get('domain')!=binding or record.get('target',target+1)>target
                or record.get('extra',extra)<extra):
            continue
        for pair in record['closed_anchors'].split():
            a,b = map(int,pair.split('-'))
            result.add(tuple(sorted((int(perm[a]),int(perm[b])))))
    return result


def main():
    import argparse
    import json
    from r27_incumbent import BoardModel
    from r28_exchange import ExchangeBound, domain_binding, read_json
    from r29_exchange_clusters import StarClusters
    from search_problem import SearchProblem
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cores','rotation','family','budget','responses','overlaps'))
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--incumbent',required=True)
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--cells',nargs='*',type=int,default=[])
    p.add_argument('--candidates',nargs='*',type=int)
    p.add_argument('--extra',type=int,default=4)
    p.add_argument('--unlimited-extra',action='store_true',help='Allow any number of distant movers in family mode.')
    p.add_argument('--nodes',type=int,default=128)
    p.add_argument('--swaps',type=int,default=4)
    args = p.parse_args()
    model = BoardModel(SearchProblem(read_json(args.tiles)))
    bound = ExchangeBound(model,read_json(args.incumbent)['codes'])
    if args.mode=='cores':
        cores = defect_cores(bound,args.target)
        report = dict(binding=domain_binding(model),
                      pieces=bound.tiles.tolist(),target=args.target,
                      cores=[[i for i in range(model.n) if c & (1<<i)] for c in cores])
    elif args.mode=='family':
        report = report_family(StarClusters(bound),args.cells,args.target,None if args.unlimited_extra else args.extra,
                               candidates=args.candidates,nodes=args.nodes)
    elif args.mode=='overlaps':
        if args.candidates is None:
            p.error("Supply the available remainder via --candidates.")
        report = family_case_upper(StarClusters(bound),args.cells,args.candidates,
                                   merge_slots=args.extra,max_parts=args.nodes)
    elif args.mode=='budget':
        if args.candidates is None:
            p.error("Supply allowed cells via --candidates.")
        report = pattern_distance_upper(bound,args.candidates,args.swaps,max_nodes=args.nodes)
    elif args.mode=='rotation':
        region = RotationRegions(bound,6,rings=True)
        tree = RotationForest(model)
        report = dict(star=bound.total//2,regional=(bound.total-region.discount())//2,
                      forest=tree.upper(bound.tiles),messages=tree.nodes,hits=tree.hits)
    else:
        cache = ResponseGroups(bound)
        rows = {}
        for cell in args.cells:
            values,groups = grouped_rotation_bound(bound,cell,range(model.n),response_cache=cache)
            rows[cell]=dict(identities=len(values),signatures=groups,max_star=max(values.values()))
        report=dict(rows=rows,representative_evaluations=len(cache.evaluations))
    print(json.dumps(report))


if __name__=='__main__':
    main()
