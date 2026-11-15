"""Replay geographic phase exclusions before sharing them between embedding queries."""
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

from r27_structure_filters import domain_bits
from r35_loss_witness import raw_sides
from r35_phase_anchors import narrow


@lru_cache(maxsize=1)
def _published():
    path = Path(__file__).parent / "docs" / "R35_PHASE_ANCHOR_CORES.json"
    if not path.is_file():
        return None, ()
    data = json.loads(path.read_bytes())
    rows = tuple((row["anchor"], tuple(map(tuple, row["pins"]))) for row in data["rules"])
    return data["fingerprint"], rows


def check_rules(problem, pins, deadline, *, candidates=None):
    """Each returned pin assignment and its global complement exclude every board."""
    if candidates is None:
        fingerprint, candidates = _published()
        if fingerprint != problem.fingerprint:
            candidates = ()
    accepted = []
    stats = dict(replayed=0, reused=0, clauses=0, terminal=False)
    if not candidates:
        return accepted, stats
    # A proof on restricted base domains must never leak into wider ones.
    base = tuple(domain_bits(problem))
    mesh = tuple(tuple(zip(map(int, neighbors), map(int, sides)))
                 for neighbors, sides in zip(problem.neighbors, problem.sides))
    scope = (base, mesh, tuple(raw_sides(problem.tiles)))
    old_scope, saved = getattr(problem, "_verified_anchor_rules", (None, None))
    if old_scope != scope:
        saved = set()
        problem._verified_anchor_rules = scope, saved
    for anchor, row in candidates:
        if row in saved:
            stats["reused"] += 1
        else:
            if monotonic() >= deadline:
                break
            result = narrow(problem, base, dict(row), anchors=[anchor],
                            unique=False, project=False, repeat=False, deadline=deadline)
            stats["replayed"] += 1
            if result["status"] != "INFEASIBLE":
                continue
            saved.add(row)
        accepted.append(row)
        stats["clauses"] += 2
        if all(tile in pins for tile, _ in row) and len({pins[tile] ^ bit for tile, bit in row}) <= 1:
            stats["terminal"] = True
            break
    return accepted, stats


def post_rules(model, phase, rows):
    for row in rows:
        for complement in (0, 1):
            model.add_bool_or([~phase[tile] if bit ^ complement else phase[tile] for tile, bit in row])
