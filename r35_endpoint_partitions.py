"""Compile the phase-independent incidence of a physical endpoint partition."""


def compile_endpoint_cut(data, group):
    inside, outside = [], []
    crossings = []
    for tile, pair in enumerate(data.strands):
        counts = []
        occupied = [[],[]]
        for row in pair:
            endpoints = {v for edge in row for v in edge}
            occupied[0].append(bool(endpoints & group))
            occupied[1].append(bool(endpoints - group))
            counts.append(sum((a in group) != (b in group) for a,b in row))
        for opts, uses in zip(occupied,(inside,outside)):
            if all(opts):
                uses.append(True)
            elif any(opts):
                uses.append((tile,opts[1]))
        crossings.append((counts[0],counts[1]-counts[0]))
    return tuple(inside), tuple(outside), tuple(crossings)


def add_compiled_cut(model, phases, coefficients):
    from r30_gold import _occurs

    inside, outside, crossings = coefficients
    def uses(items):
        return [True if item is True else
                (phases[item[0]] if item[1] else ~phases[item[0]]) for item in items]
    occupied = [_occurs(model,uses(row)) for row in (inside,outside)]
    if any(x is False for x in occupied):
        return
    total = sum(base+delta*phases[t] for t,(base,delta) in enumerate(crossings))
    model.add(total>=2).only_enforce_if([x for x in occupied if type(x) is not bool])


def pinned_start_pool(problem):
    """Seed the recurrent singleton partition and recompile it on this physical drawing."""
    from r29_gold_phase import EndpointCuts
    if problem.fingerprint != 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
        return None
    group = frozenset([(20,1,5)])
    cached = getattr(problem,'_r35_start_cut',None)
    if cached is None:
        data = EndpointCuts(problem)
        cached = data, compile_endpoint_cut(data,group)
        problem._r35_start_cut = cached
    data, row = cached
    return dict(data=data,partitions={group},coefficients={group:row})
