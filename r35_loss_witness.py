"""Revalidate a compact phase witness against the destination's physical side labels."""
from collections import Counter


def raw_sides(tiles):
    sides = []
    for tile in tiles:
        positions = [set() for _ in range(3)]
        for segment in tile['segments']:
            for side, slot in segment:
                positions[side].add(slot)
        color = 0
        for slots in positions:
            sides.append((tuple(sorted(slots)), color))
            color ^= len(slots) % 2
        if color:
            raise ValueError('A tile must have an even number of arc endpoints.')
    return sides


def reciprocal(label):
    slots, color = label
    return tuple(12 - k for k in reversed(slots)), color ^ (len(slots) % 2)


def cover_missing(groups, required, failed):
    """Choose the fewest additional compatible bonds that hit every remaining conflict."""
    from functools import lru_cache
    unresolved = [group-required for group in groups if not group&failed]
    if not unresolved:
        return frozenset()
    occurrences = {}
    for i, group in enumerate(unresolved):
        for bond in group:
            occurrences[bond] = occurrences.get(bond,0) | (1<<i)
    if all(bits.bit_count()==1 for bits in occurrences.values()):
        return frozenset(min(g) for g in unresolved)

    @lru_cache(maxsize=None)
    def hits(mask):
        if not mask:
            return ()
        i = min((j for j in range(len(unresolved)) if mask>>j & 1), key=lambda j:len(unresolved[j]))
        choices = [(bond,*hits(mask & ~occurrences[bond])) for bond in sorted(unresolved[i])]
        return min(choices,key=lambda row:(len(row),row))

    return frozenset(hits((1<<len(unresolved))-1))


def physical_price(tiles, reference, groups, required, phase, *, integer=True):
    if type(phase) is not int or not 0 <= phase < 1 << len(tiles):
        return None
    sides = [(slots, color ^ ((phase >> (i // 3)) & 1))
             for i, (slots, color) in enumerate(raw_sides(tiles))]
    counts = Counter(sides)
    for label, count in counts.items():
        other = reciprocal(label)
        if label == other:
            if count % 2:
                return None
        elif ((count != counts[other]) if integer else ((count-counts[other]) % 2)):
            return None
    failed = {p for p in reference | required if sides[p[0]] != reciprocal(sides[p[1]])}
    if required & failed or any(g <= required for g in groups):
        return None
    return len(failed & reference) + len(cover_missing(groups,required,failed)), failed


def save_witness(db, identity, phase):
    db.execute('CREATE TABLE IF NOT EXISTS r35_loss_witnesses('
               'identity TEXT, phase TEXT, PRIMARY KEY(identity,phase))')
    db.execute('INSERT OR IGNORE INTO r35_loss_witnesses VALUES(?,?)', (identity,format(phase,'x')))


def checked_hits(db, identity, problem, reference, groups, required, outside, cap, integer, conditions=None):
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='r35_loss_witnesses'").fetchone()
    if not exists:
        return {}
    found = {}
    for (text,) in db.execute('SELECT phase FROM r35_loss_witnesses WHERE identity=?', (identity,)):
        try:
            if not isinstance(text,str) or len(text) > (problem.n+3)//4:
                continue
            phase = int(text,16)
        except ValueError:
            continue
        result = physical_price(problem.tiles,reference,groups,required,phase,integer=integer)
        if result is None or result[0] > cap or (conditions is not None and not conditions.accepts(phase)):
            continue
        cost, failed = result
        if conditions is not None:cost=conditions.price(phase,cost)
        errors = [pair for pair in outside if pair in failed]
        found.setdefault(len(errors), dict(status='INTEGER_RELAXATION_FEASIBLE' if integer else 'PHASE_RELAXATION_FEASIBLE',
                                           outside_errors=errors, errors=len(errors), cap=cap, loss_cost=cost,
                                           phase=hex(phase), witness_revalidated=True, reused=True))
    return found
