"""Indexed phase-cube exclusions and independently validated geographic Gold witnesses."""
import json

from r29_gold_phase import EndpointCuts
from r31_geographic import physical_bonds


def schema(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS r32_phase_cuts (scope TEXT, pins TEXT, PRIMARY KEY(scope,pins))')
    conn.execute("""CREATE INDEX IF NOT EXISTS r34_phase_anchor ON r32_phase_cuts
        (scope, coalesce(json_extract(pins,'$[0][0]'),-1))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS r34_phase_witnesses(
        puzzle TEXT, codes TEXT, phases TEXT, PRIMARY KEY(puzzle,codes))""")


def negative(conn, scope, pins):
    # Both colorings of this complete one-loop geographic problem are equivalent.
    # The first physical tile must be present even when all phase literals reverse.
    anchors = [-1,*pins]
    for start in range(0,len(anchors),400):
        subset = anchors[start:start+400]
        placeholders = ','.join('?' for _ in subset)
        entries = conn.execute(f"""SELECT pins FROM r32_phase_cuts WHERE scope=? AND
            coalesce(json_extract(pins,'$[0][0]'),-1) IN ({placeholders})""", (scope,*subset))
        for raw, in entries:
            core = json.loads(raw)
            if not core:
                return core
            flip = pins[core[0][0]] ^ core[0][1]
            if all(pins.get(tile) == (phase ^ flip) for tile,phase in core):
                return [[tile,phase ^ flip] for tile,phase in core]
    return None


def checked_witness(problem, codes, phases, domains, pins, required=(), forbidden=()):
    """Return the board's requested coloring only if the board realizes every assumption."""
    if (len(codes) != problem.n or len(phases) != problem.n or
            any(type(p) is not int or p not in (0,1) for p in phases) or
            any(type(c) is not int or not 0 <= c < 3*problem.n or not (domain >> c & 1)
                for c,domain in zip(codes,domains))):
        return None
    if not problem.validate(codes)['valid']:
        return None
    bonds = physical_bonds(problem,codes)
    if not set(required).issubset(bonds) or set(forbidden).intersection(bonds):
        return None
    sides = EndpointCuts(problem).sides
    for a,b in bonds:
        mask,start = sides[a]
        _,end = sides[b]
        if (phases[a//3] ^ phases[b//3]) != (start ^ end ^ (mask.bit_count() & 1)):
            return None
    for flip in (0,1):
        if all((phases[t] ^ flip) == v for t,v in pins.items()):
            return dict(status='GOLD',codes=codes,phases=[p ^ flip for p in phases],pairs=bonds)
    return None


def witness(conn, problem, domains, pins, required):
    for codes,phases in conn.execute("SELECT codes,phases FROM r34_phase_witnesses WHERE puzzle=?",
                                     (problem.fingerprint,)):
        row = checked_witness(problem,json.loads(codes),json.loads(phases),domains,pins,required)
        if row is not None:
            return dict(row,reused=True,lift_calls=0,cp_calls=0)
    return None
