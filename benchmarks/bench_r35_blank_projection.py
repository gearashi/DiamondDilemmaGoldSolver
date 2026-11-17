"""Compare deferred blanks with complete physical matchings on identical fixed nonblank sewings."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from math import prod
import json
from pathlib import Path
import random

from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r29_gold_phase import _reciprocal
from r30_gold import GoldMatching, embedding, search
from r31_geographic import physical_bonds
from r35_blank_projection import lift
from search_problem import SearchProblem


def matching_completions(blanks, partners):
    if not blanks:
        yield []
        return
    a = min(blanks, key=lambda d: (len(partners[d] & blanks), d))
    for b in sorted(partners[a] & blanks):
        for tail in matching_completions(blanks - {a,b}, partners):
            yield [(min(a,b),max(a,b)), *tail]


def oracle(problem, domains, selected, options, blank_darts):
    """Materialize every complete matching and run the legacy labelled embedding."""
    partners = defaultdict(set)
    for a,b in options:
        partners[a].add(b)
        partners[b].add(a)
    counts = Counter()
    successes = set()
    for extra in matching_completions(set(blank_darts), partners):
        counts['blank_completions'] += 1
        mates = [-1] * (3*problem.n)
        pairs = sorted([*selected,*extra])
        for a,b in pairs:
            mates[a],mates[b] = b,a
        board = embedding(problem,domains,mates)
        if board is not None:
            assert problem.validate(board)['valid'] and physical_bonds(problem,board) == pairs
            successes.add(tuple(board))
            counts['geographic_completions'] += 1
    return counts,successes


def short(result):
    keys = ('status','cp_calls','branches','strand_cuts','partner_variables','blank_pairs_deferred',
            'nonblank_witnesses_checked','blank_matching_states','blank_root_trials','failed_embeddings')
    return {k:result[k] for k in keys if k in result}


def small(seed, negative):
    from test_r28_geometry import phases
    p,_ = planted(1,10,seed)
    full = [3*t for t in range(p.n)]
    physical = physical_bonds(p,full)
    selected = [pair for pair in physical if int(p.masks[3*(pair[0]//3),pair[0]%3])]
    values = phases(p)
    ds = list(domain_bits(p))
    ds[0] = 1
    if negative:
        ds[2] = 1 << 7  # Same physical tile in a rotation incompatible with the pinned root.
    ds = tuple(ds)
    hint = dict(enumerate(values))
    args = dict(domains=ds,pins=hint,bonds=selected,seconds=15,geography=True,max_rounds=8)
    model = GoldMatching(p,ds,hint,selected,project_blanks=True)
    blank = model.blank_projection
    counts,success = oracle(p,ds,selected,blank.signed(values),blank.blanks)
    out = {k:short(search(p,**args,project_blanks=v)) for k,v in [('full',False),('projected',True)]}
    for name,policy in [('legacy_channel',dict(channel=True,learned_geometry=True)),
                        ('full_learned',dict(learned_geometry=True)),
                        ('projected_learned',dict(project_blanks=True,learned_geometry=True))]:
        out[name] = short(search(p,**args,**policy))
    assert out['projected']['status'] == ('GOLD' if success else 'INFEASIBLE')
    return dict(seed=seed, negative=negative, exhaustive=dict(counts), **out)


def components(model, matching):
    parent = list(range(model.strands))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    total = model.strands
    for p in matching:
        for a,b in model.transitions[p]:
            u,v = root(a),root(b)
            if u != v:
                parent[u] = v
                total -= 1
    return total


def connected_control(p, seed=35):
    """Construct a repeatable *abstract* connected sewing for testing the blank lift."""
    from ortools.sat.python import cp_model
    model = GoldMatching(p, domain_bits(p), None, (), project_blanks=True)
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.max_time_in_seconds = 30
    if s.solve(model.m) not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError('Could not prepare the requested physical-only test fixture.')
    values = [s.value(x) for x in model.phase]
    current = [pair for pair,x in model.mate.items() if s.value(x)]

    def label(dart):
        mask, color = model.inputs.sides[dart]
        return mask, color ^ values[dart//3]

    groups = defaultdict(list)
    for idx,(a,b) in enumerate(current):
        if label(a)>label(b):
            a,b = b,a
        current[idx]=(a,b)
        groups[label(a)].append(idx)
    rng = random.Random(seed)
    groups = [indices for indices in groups.values() if len(indices)>1]
    def canonical(pairs):
        return [tuple(sorted(pair)) for pair in pairs]
    initial = tuple(current)
    best = p.total_segments
    for attempt in range(6):
        current = list(initial)
        count = components(model,canonical(current))
        for step in range(12000):
            if count == 1:
                return values,canonical(current),dict(attempt=attempt,step=step,segments=p.total_segments)
            i,j = rng.sample(rng.choice(groups),2)
            a,b = current[i]
            c,d = current[j]
            if label(c)==label(d) and rng.randrange(2):
                c,d = d,c
            trial = canonical(((a,d),(c,b)))
            if not all(k in model.mate for k in trial):
                continue
            old = current[i],current[j]
            current[i],current[j] = (a,d),(c,b)
            new = components(model,canonical(current))
            if new <= count or (count > 4 and rng.random()<.025):
                count = new
            else:
                current[i],current[j] = old
            best = min(best,count)
    raise RuntimeError(f'No connected physical-only fixture; smallest component count was {best}.')


def direct_trace(p, pairs):
    # Independently walk raw tile segments through physically reversed slot IDs.
    mate = dict(pairs)
    mate.update({b:a for a,b in pairs})
    local = {}
    for i,tile in enumerate(p.tiles):
        for (s,k),(t,j) in tile['segments']:
            a,b = (3*i+s,k),(3*i+t,j)
            local[a],local[b] = b,a
    visited = set()
    cycles = []
    for port in local:
        if port in visited:
            continue
        start = port
        length = 0
        while port not in visited:
            other = local[port]
            visited.update((port,other))
            length += 1
            dart,slot = other
            port = (mate[dart],12-slot)
        assert port == start
        cycles.append(length)
    return sorted(cycles)


def count_dense_signed(blanks, values, signed):
    # For this commercial control each color class is complete, or missing one
    # forbidden same-tile pair. Count all completions exactly by inclusion/exclusion.
    total = 1
    counts = []
    for color in (0,1):
        darts = [d for d in blanks.blanks if blanks.sides[d][1]^values[d//3] == color]
        missing = sorted(set(combinations(darts,2))-signed)
        if len(darts)%2 or len(missing)>1:
            raise ValueError('Unexpected dense signed class for the original fixture.')
        gross = prod(range(1,len(darts),2))
        excluded = prod(range(1,len(darts)-2,2)) if missing else 0
        count = gross-excluded
        counts.append(dict(darts=len(darts),missing=missing,completions=count))
        total *= count
    return counts,total


def original(path, fixture):
    p = SearchProblem(read_json(path))
    ds = domain_bits(p)
    if fixture and fixture.exists():
        data = json.loads(fixture.read_text(encoding='utf-8'))
        if data['fingerprint'] != p.fingerprint:
            raise ValueError('Sewing fixture belongs to a different input.')
        phases,pairs = data['phases'],[tuple(x) for x in data['nonblank']]
    else:
        phases,pairs,prep = connected_control(p)
        if fixture:
            fixture.write_text(json.dumps(dict(fingerprint=p.fingerprint,phases=phases,nonblank=sorted(pairs),
                                              preparation=prep),indent=2)+'\n',encoding='utf-8')
    model = GoldMatching(p,ds,None,(),project_blanks=True)
    assert components(model,pairs)==1
    blank=model.blank_projection
    signed=blank.signed(phases)
    witness = blank.matching(signed,float('inf'),Counter())
    assert witness is not None
    sides = model.inputs.sides
    for a,b in [*pairs,*witness]:
        assert _reciprocal(sides[a][0],sides[a][1]^phases[a//3]) == (sides[b][0],sides[b][1]^phases[b//3])
    assert sorted(d for pair in [*pairs,*witness] for d in pair) == list(range(3*p.n))
    circuit = direct_trace(p,[*pairs,*witness])
    assert circuit == [p.total_segments]
    counts,total = count_dense_signed(blank,phases,signed)
    from r31_geographic import ribbon_genus
    from r28_geo_sewing import PosePacking
    genus = ribbon_genus(p,pairs)
    packed = PosePacking(p).check(pairs)
    assert genus>0 and packed['feasible'] is False
    pose_stats=Counter()
    board=lift(p,ds,pairs,blank,signed,float('inf'),pose_stats)
    if board is not None:
        assert p.validate(board)['valid']
    fixed=dict(domains=ds,pins=dict(enumerate(phases)),bonds=pairs,geography=True,max_rounds=6,seconds=60)
    return dict(nonblank_bonds=len(pairs),blank_darts=len(blank.blanks),signed_blank_options=len(signed),
                signed_classes=counts,complete_blank_matchings=total,independent_circuit_lengths=circuit,
                nonblank_ribbon_genus=genus,nonblank_pose_packing=packed,
                geographic_board=board,exhaustive_pose_scan=dict(pose_stats),
                full=short(search(p,**fixed)),projected=short(search(p,**fixed,project_blanks=True)),
                full_learned=short(search(p,**fixed,learned_geometry=True)),
                projected_learned=short(search(p,**fixed,project_blanks=True,learned_geometry=True)),
                legacy_channel=short(search(p,**fixed,channel=True,learned_geometry=True)))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path)
    cli.add_argument('--sewing',type=Path)
    cli.add_argument('--root-rounds',type=int,default=0,
                     help='Also compare this many unrestricted matching calls on the original (60 s per arm).')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    out=dict(planted=[small(seed,neg) for seed in (35,71,103,157) for neg in (False,True)])
    if args.tiles:
        out['original']=original(args.tiles,args.sewing)
        if args.root_rounds:
            p = SearchProblem(read_json(args.tiles))
            out['unrestricted'] = {name:short(search(p,project_blanks=enabled,seconds=60,
                                                      max_rounds=args.root_rounds))
                                   for name,enabled in [('full',False),('projected',True)]}
    text=json.dumps(out,indent=2)+'\n'
    if args.output:
        args.output.write_text(text,encoding='utf-8')
    else:
        print(text)


if __name__=='__main__':
    main()
