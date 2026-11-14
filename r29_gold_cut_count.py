"""Bounded Cut&Count witness search across two geographic regions on the Gold board."""
from collections import defaultdict
from functools import lru_cache
from random import Random

from r27_structure_filters import Work, WorkLimit, bits
from r29_gold_geography import checked_domains


def cut_witness(problem, ds, inside, *, seed=29, trials=4, max_work=1000000):
    """Count consistent strand colorings modulo two with one physical strand pinned.

    A board with k closed loops contributes 2**(k-1) colorings. Random placement weights
    separate connected boards. Positive parity is exact. Zero is always reported unknown.
    """
    ds = checked_domains(problem,ds)
    left = frozenset(inside)
    right = frozenset(range(problem.n))-left
    if not left or not right or left | right != frozenset(range(problem.n)):
        raise ValueError("Expected a nonempty proper geographic subset.")
    if trials < 1:
        raise ValueError("Expected at least one isolation trial.")
    w = Work(max_work)
    rng = Random(seed)
    seams = {}
    for e,(a,s,b,t) in enumerate(problem.edges):
        seams[a,s] = e,False
        seams[b,t] = e,True
    if len(seams) != 3*problem.n:
        raise ValueError("Cut&Count requires a closed surface.")
    block = (3*problem.n).bit_length()

    @lru_cache(maxsize=None)
    def choices(code):
        t,r = divmod(code,3)
        arcs = problem.tiles[t]["segments"]
        masks = [int(m) for m in problem.masks[code]]
        # The canonical seam frame is supplied per cell below.
        out = []
        for coloring in range(0,1<<len(arcs),2 if t==0 else 1):
            w.tick("cut_colorings")
            dark = [0,0,0]
            for j,seg in enumerate(arcs):
                if coloring>>j & 1:
                    for s,p in seg:
                        dark[(s+r)%3] |= 1<<(p-1)
            out.append(tuple(zip(masks,dark)))
        return out

    @lru_cache(maxsize=32000)
    def options(c,code):
        from validator import reverse_mask
        result = []
        for row in choices(code):
            labels = {}
            for s,(m,d) in enumerate(row):
                e,rev = seams[c,s]
                labels[e] = (reverse_mask(m),reverse_mask(d)) if rev else (m,d)
            result.append(labels)
        return result

    def collect(part, domains, weights):
        members = []
        pending = set(part)
        # Grow a contiguous processed region whenever domains permit.
        prior = set()
        while pending:
            c = min(pending,key=lambda x:(domains[x].bit_count(),
                        -sum(b in prior for b,j in problem.board.neighbors[x]),x))
            pending.remove(c)
            prior.add(c)
            members.append(c)
        table = {(0,(),0):(1,0)}
        frontier = set()
        for c in members:
            w.tick("cut_layers")
            local_edges = {seams[c,s][0] for s in range(3)}
            old_edges = sorted(frontier)
            overlap = frontier & local_edges
            frontier ^= local_edges
            new_edges = sorted(frontier)
            new = {}
            for (used,signature,weight),(parity,digest) in table.items():
                opened = dict(zip(old_edges,signature))
                for code in bits(domains[c]):
                    tile = 1 << (code//3)
                    if used & tile:
                        continue
                    step = weights[c,code]
                    board_code = (code+1) << (block*c)
                    for labels in options(c,code):
                        w.tick("cut_transitions")
                        if any(opened[e] != labels[e] for e in overlap):
                            continue
                        key = used | tile,tuple(opened[e] if e in opened else labels[e] for e in new_edges),weight+step
                        p,x = new.get(key,(0,0))
                        new[key] = p ^ parity, x ^ digest ^ (board_code if parity else 0)
            table = {k:v for k,v in new.items() if v != (0,0)}
            w.counters["cut_peak_states"] = max(w.counters["cut_peak_states"],len(table))
        return table

    def oracle(domains, weights):
        a = collect(left,domains,weights)
        b = collect(right,domains,weights)
        inventory = (1<<problem.n)-1
        groups = defaultdict(list)
        for (used,sig,wt),(p,x) in b.items():
            groups[used,sig].append((wt,p,x))
        results = defaultdict(lambda:[0,0])
        for (used,sig,wt),(p,x) in a.items():
            for other,q,y in groups[inventory ^ used,sig]:
                w.tick("cut_joins")
                value = results[wt+other]
                value[0] ^= p & q
                value[1] ^= (x if q else 0) ^ (y if p else 0)
        return results

    try:
        keys = [(c,code) for c,d in enumerate(ds) for code in bits(d)]
        for trial in range(trials):
            weights = {key:rng.randint(1,max(1,2*len(keys))) for key in keys}
            totals = oracle(ds,weights)
            for weight,(odd,digest) in sorted(totals.items()):
                if not odd:
                    continue
                codes = [((digest>>(block*c)) & ((1<<block)-1))-1 for c in range(problem.n)]
                if not all(v>=0 and ds[c]>>v & 1 for c,v in enumerate(codes)) or not problem.validate(codes)["valid"]:
                    # Odd parity remains after at least one child. Recover exactly even on an isolation tie.
                    domains = list(ds)
                    for c,d in enumerate(domains):
                        if d.bit_count() == 1:
                            continue
                        for code in bits(d):
                            domains[c] = 1<<code
                            if oracle(domains,weights).get(weight,[0])[0]:
                                break
                        else:
                            raise ArithmeticError("Lost an odd Cut&Count witness.")
                    codes = [v.bit_length()-1 for v in domains]
                    if not problem.validate(codes)["valid"]:
                        raise ArithmeticError("Cut&Count witness failed independent validation.")
                return dict(exists=True,witness=codes,trial=trial+1,counters=dict(w.counters))
        reason = "zero-parity"
    except WorkLimit:
        reason = "work-limit"
    return dict(exists=None,witness=None,reason=reason,counters=dict(w.counters))
