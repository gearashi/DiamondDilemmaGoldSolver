"""Global signed physical pairings with loss ownership and projected blank-side matches."""
from itertools import combinations
from math import ceil
from time import monotonic

from r29_gold_phase import checked_pins
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r35_contact_mip import build_model, checked_cycles
from r35_loss_witness import raw_sides, reciprocal
from r35_mip_conditions import euler_row
from r35_signed_mate_support import raw_edges, reciprocal_classes


def sparse_forbidden(pairs):
    """Recognize a matching plus at most one extra forbidden edge."""
    degree = {}
    for a,b in pairs:
        degree[a] = degree.get(a,0)+1
        degree[b] = degree.get(b,0)+1
    crowded = {v for v,n in degree.items() if n>1}
    return not crowded or (max(degree.values())==2 and any(crowded <= set(e) for e in pairs))


def blank_completion(sides, blocked):
    """Construct a perfect matching in a complete graph minus the certified sparse bans."""
    rest = sorted(sides)
    if len(rest)%2:
        return None
    pairs = []
    while len(rest)>4:
        first = rest.pop(0)
        partner = next((b for b in rest if (first,b) not in blocked),None)
        if partner is None:
            return None
        pairs.append((first,partner))
        rest.remove(partner)
    if not rest:
        return pairs
    first = rest[0]
    for other in rest[1:]:
        tail = tuple(x for x in rest[1:] if x!=other)
        finish = [(first,other)] + ([tail] if tail else [])
        if all(pair not in blocked for pair in finish):
            return pairs+finish
    return None


def blank_rows(engine, flags, sides, blocked):
    for active in flags:
        count = sum(active)
        engine.Add(count==2*engine.IntVar(0,len(sides)//2,''))
        by_side = dict(zip(sides,active))
        # An even residual of four or more vertices always admits a matching.
        # For a forbidden residual pair at exactly two, require another vertex.
        for a,b in sorted(blocked):
            engine.Add(count >= 2*(by_side[a]+by_side[b])-1)


def project_blanks(engine, phase, losses, labels, sides, legal, reference, required):
    family = set(sides)
    named = {p for p in reference|required if p[0] in family or p[1] in family}
    if any(a not in family or b not in family for a,b in named):
        return None
    blocked = set(combinations(sorted(sides),2))-legal | named
    if not sparse_forbidden(blocked):
        return None
    incident = {s:[] for s in sides}
    for pair in sorted(named):
        keep = 1-losses[pair] if pair in losses else 1
        if pair not in legal:
            engine.Add(keep==0)
        a,b = pair
        incident[a].append(keep)
        incident[b].append(keep)
    flags = [[],[]]
    for s in sides:
        residual = engine.BoolVar('')
        engine.Add(residual==1-sum(incident[s]))
        dark = engine.BoolVar('')
        color = 1-phase[s//3] if labels[s][1] else phase[s//3]
        engine.Add(dark<=residual)
        engine.Add(dark<=color)
        engine.Add(dark>=residual+color-1)
        flags[1].append(dark)
        flags[0].append(residual-dark)
    blank_rows(engine, flags, sides, blocked)
    return blocked


def recover_bipartite(sides, eligible, labels):
    neighbors = {s:[] for s in sides if labels[s] < reciprocal(labels[s])}
    for a,b in sorted(eligible):
        if a in neighbors:
            neighbors[a].append(b)
        elif b in neighbors:
            neighbors[b].append(a)
    owner = {}
    def augment(a,seen):
        for b in neighbors[a]:
            if b in seen:
                continue
            seen.add(b)
            if b not in owner or augment(owner[b],seen):
                owner[b] = a
                return True
        return False
    if not all(augment(a,set()) for a in sorted(neighbors,key=lambda a:len(neighbors[a]))):
        return None
    return [tuple(sorted((a,b))) for b,a in owner.items()]


def physical_witness(problem, value, pairs, reference, required, groups, cap, allowed, *, sphere=False):
    """Return an independently checked physical mating, including its actual loops."""
    from r35_phase_sewing import strand_components, _lift
    if type(value) is not int or not 0<=value<1<<problem.n or not isinstance(pairs,(tuple,list,set,frozenset)):
        return None
    if len(pairs)!=3*problem.n//2 or any(not isinstance(e,(tuple,list)) or len(e)!=2
                                       or any(type(v) is not int for v in e) for e in pairs):
        return None
    result={tuple(sorted(e)) for e in pairs}
    if sorted(v for p in result for v in p)!=list(range(3*problem.n)) or not required<=result<=allowed:
        return None
    lost=reference-result
    if len(lost)>cap or any(not g&lost for g in groups):
        return None
    labels=[(slots,color^((value>>(i//3))&1)) for i,(slots,color) in enumerate(raw_sides(problem.tiles))]
    if any(labels[a]!=reciprocal(labels[b]) for a,b in result):
        return None
    if sphere:
        table=PhaseInterface(problem)
        if sum(row[(value>>i)&1][-1] for i,row in enumerate(table.rows)) not in table.euler:
            return None
    loops=strand_components(problem,result)
    answer=dict(status='PHYSICAL_MATCHING_FEASIBLE',cap=cap,loss_cost=len(lost),phase=hex(value),
                pairs=sorted(result),strand_components=loops)
    if loops==[problem.total_segments]:
        codes,genus=_lift(problem,result)
        answer.update(codes=codes,genus=genus)
        if codes is not None:
            answer['status']='GOLD'
    return answer


def published_witness(problem, reference, required, groups, cap, allowed, sphere, pins=None):
    import json
    from pathlib import Path
    from r35_phase_sewing import _published_sewings
    if problem.n != 160:
        return None
    try:
        catalog = json.loads((Path(__file__).resolve().parent/'docs/R35_MIXED_MIP_DONOR.json').read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(catalog, dict) or not isinstance(catalog.get('others', []), list):
        return None
    pins = pins or {}
    phases = {}
    seen = set()

    def check(value, record):
        if not isinstance(record, dict):
            return None
        for field in ('first_pairing', 'pairs'):
            pairs = record.get(field)
            if not isinstance(pairs, list) or len(pairs) != 3*problem.n//2:
                continue
            # Historical portfolios sometimes repeat the same ordered full
            # pairing. Only successful raw validation can answer a query.
            key = value, json.dumps(pairs, separators=(',', ':'))
            if key in seen:
                continue
            seen.add(key)
            answer = physical_witness(problem, value, pairs, reference, required, groups, cap, allowed,
                                      sphere=sphere)
            if answer is not None:
                return dict(answer, work={'nodes': 0, 'verified_physical_pairing_reused': 1})
        return None

    for record in [catalog, *catalog.get('others', [])]:
        if not isinstance(record, dict) or record.get('fingerprint') != problem.fingerprint:
            continue
        try:
            original = int(record['phase'], 16)
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= original < 1 << problem.n:
            continue
        values = (original, ((1 << problem.n)-1) ^ original) if pins else (original,)
        for value in values:
            if any((value >> i) & 1 != bit for i, bit in pins.items()):
                continue
            phases[value] = None
            answer = check(value, record)
            if answer is not None:
                return answer
    for value in phases:
        for record in _published_sewings(problem, value):
            answer = check(value, record)
            if answer is not None:
                return answer
    return None

def decide(problem, reference, groups, required, cap, seconds, *, sphere=False, mode='blank', hint=None, pins=None, forbidden=(), published=False, full_hint=True):
    reference, required = set(reference), set(required)
    allowed = inventory(problem).possible-set(forbidden)
    pins = checked_pins(problem.n, {} if pins is None else pins)
    if published:
        replay = published_witness(problem,reference,required,groups,cap,allowed,sphere,pins)
        if replay is not None:
            return replay
    # The signed inventory is invariant under a global color flip. Absolute
    # pins can select either representative of a complementary pair.
    canonical_phase = not pins or pins.get(0) == 0
    engine, phase, loss = build_model(problem,reference,groups,required,cap,checked_cycles(problem,reference,required),
                                     canonical_phase=canonical_phase)
    if sphere:
        euler_row(engine,phase,PhaseInterface(problem))
    for i,value in pins.items():
        engine.Add(phase[i]==value)
    labels, families = reciprocal_classes(problem.tiles)
    modeled = []
    work = dict(explicit_binary_mates=0, continuous_mates=0, blank_projected_sides=0, blank_omitted_edges=0)
    if not required<=allowed:
        engine.Add(False)
    for pair in reference-required-allowed:
        engine.Add(loss[pair]==1)
    for family in families:
        edges = raw_edges(labels,family,allowed)
        legal = {(a,b) for a,b,_ in edges}
        slots = labels[family[0]][0]
        if not slots and mode in ('blank','both'):
            blocked = project_blanks(engine,phase,loss,labels,family,legal,reference,required)
            if blocked is not None:
                modeled.append(('blank',family,blocked))
                work.update(blank_projected_sides=len(family),blank_omitted_edges=len(edges))
                continue
        bipartite = slots != tuple(12-k for k in reversed(slots)) or len(slots)%2
        flow = bipartite and mode in ('flow','both')
        incidence = {s:[] for s in family}
        variables = {}
        for a,b,rhs in edges:
            pair = (a,b)
            if pair in required:
                chosen = 1
            elif pair in loss:
                chosen = 1-loss[pair]
            elif flow:
                chosen = engine.NumVar(0,1,'')
                work['continuous_mates'] += 1
            else:
                chosen = engine.BoolVar('')
                work['explicit_binary_mates'] += 1
            x,y = phase[a//3],phase[b//3]
            if rhs:
                engine.Add(chosen<=x+y)
                engine.Add(chosen<=2-x-y)
            else:
                engine.Add(chosen<=1+x-y)
                engine.Add(chosen<=1-x+y)
            incidence[a].append(chosen)
            incidence[b].append(chosen)
            variables[pair] = chosen
        for options in incidence.values():
            engine.Add(sum(options)==1)
        modeled.append(('flow' if flow else 'binary',family,variables))

    work['variables'] = engine.NumVariables()
    work['rows'] = engine.NumConstraints()
    if hint:
        seed, pairs = hint
        # A full lawful pairing only supplies a warm start; native validation
        # below enforces the current scope, cap and objective.
        variables=[*phase,*loss.values()]
        values=[(seed>>i)&1 for i in range(problem.n)]+[int(pair not in pairs) for pair in loss]
        if full_hint:
            for kind,_,choices in modeled:
                if kind!='blank':
                    for pair, variable in choices.items():
                        if pair not in reference and pair not in required:
                            variables.append(variable)
                            values.append(int(pair in pairs))
        engine.SetHint(variables,values)
    engine.SetSolverSpecificParametersAsString('limits/solutions = 1')
    if seconds:
        engine.SetTimeLimit(max(1,ceil(1000*seconds)))
    start=monotonic()
    status=engine.Solve()
    work.update(nodes=max(0,engine.nodes()),lp_iterations=max(0,engine.iterations()),seconds=monotonic()-start)
    if status==engine.INFEASIBLE:
        return dict(status='INFEASIBLE',cap=cap,work=work)
    if status not in (engine.FEASIBLE,engine.OPTIMAL):
        return dict(status='UNKNOWN',cap=cap,work=work)
    value = sum(int(var.solution_value()>.5)<<i for i,var in enumerate(phase))
    lost = {p for p,var in loss.items() if var.solution_value()>.5}
    retained = (reference-lost)|required
    signed = [(slots,color^((value>>(s//3))&1)) for s,(slots,color) in enumerate(labels)]
    fixed_sides = {s for p in retained for s in p}
    result = set(retained)
    for kind,family,choices in modeled:
        rest = set(family)-fixed_sides
        if kind=='blank':
            for color in (0,1):
                parts = blank_completion((s for s in rest if signed[s][1]==color),choices)
                if parts is None:
                    raise ArithmeticError('Projected blank matching has no physical completion.')
                result.update(parts)
        elif kind=='flow':
            edges = {p for p in choices if p not in lost and p[0] in rest and p[1] in rest
                     and signed[p[0]]==reciprocal(signed[p[1]])}
            pairs = recover_bipartite(rest,edges,signed)
            if pairs is None:
                raise ArithmeticError('A feasible fractional family has no integral matching.')
            result.update(pairs)
        else:
            result.update(p for p,v in choices.items() if p not in reference|required and v.solution_value()>.5)
    answer=physical_witness(problem,value,result,reference,required,groups,cap,allowed,sphere=sphere)
    if answer is None or reference-result!=lost:
        raise ArithmeticError('Native pairing failed complete signed side, retention or loss replay.')
    return dict(answer,work=work)
