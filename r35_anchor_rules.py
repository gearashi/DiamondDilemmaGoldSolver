"""Replay geographic phase exclusions before sharing them between embedding queries."""
from copy import copy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from r27_structure_filters import domain_bits
from r35_loss_witness import raw_sides
from r35_phase_anchors import narrow


@lru_cache(maxsize=1)
def _published():
    path = Path(__file__).parent / "docs" / "R35_PHASE_ANCHOR_CORES.json"
    try:
        data = json.loads(path.read_bytes())
        if data["fingerprint"] != "e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853":
            return None, ()
        rows = []
        for entry in data["rules"]:
            anchor = entry["anchor"]
            pins = tuple((t,v) for t,v in entry["pins"])
            if (type(anchor) is not int or not 0 <= anchor < 160 or not pins
                    or any(type(t) is not int or not 0<=t<160 or type(v) is not int or v not in (0,1)
                           for t,v in pins) or len(dict(pins)) != len(pins)):
                continue
            rows.append((anchor,pins))
        return data["fingerprint"], tuple(rows)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return None, ()


# Generated DIMACS and independent refutation are bound in docs/R35_FIXED_PHASE_PROOF.json.
PROVED_PHASE = 0xd237aba7e91101e88041413e8c589902021e5126
SIGNED_MESH = "3a3e25fc843e5eb6501abb2028150fe4e088befbf50999a0ec62a7ece7c44e03"


# Independent SAT propagation rejected all 480 geographic roots for each pin list.
# The unmodified inputs and counters are in R35_HELDOUT_PHASE_ANCHORS.json.
CHECKED_ANCHORS = frozenset((
    "6893786dd051b167d789a3eeaff898b1802296db4d980531ea1e6c5119f7bf55",
    "471989312e340888274ba8c4ba1f7f1127a4f0d79fd2249def37f69a2c2077ea",
))


def has_proved_mesh(problem):
    if problem.n != 160 or len(problem.edges) != 240:
        return False
    adjacency = tuple(tuple(zip(map(int,cells),map(int,sides)))
                      for cells,sides in zip(problem.neighbors,problem.sides))
    if adjacency != problem.board.neighbors:
        return False
    expected = {(a,s,b,t) for a,row in enumerate(adjacency) for s,(b,t) in enumerate(row) if a<b}
    found = {(a,s,b,t) if a<b else (b,t,a,s) for a,s,b,t in problem.edges}
    if expected != found:
        return False
    mesh = [[list(map(int, adjacent)), list(map(int, sides))]
            for adjacent, sides in zip(problem.neighbors, problem.sides)]
    encoded = json.dumps([raw_sides(problem.tiles), mesh], separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest() == SIGNED_MESH


def project_rule(basis, row):
    """Keep only independent relative signs under the caller's parity equations."""
    if not row:
        return ()
    mark = len(basis.trail)
    first, sign = row[0]
    kept = [row[0]]
    try:
        for tile, bit in row[1:]:
            old = len(basis.trail)
            if not basis.add((1 << first) ^ (1 << tile), sign ^ bit, ()):
                return None
            if len(basis.trail) != old:
                kept.append((tile, bit))
        return tuple(kept)
    finally:
        basis.rollback(mark)


def _query_parity(problem):
    """Give each query a private elimination basis with the same tile-inventory rows."""
    from r29_gold_phase import PhaseBasis, SignedParity
    sides = tuple((sum(1<<(slot-1) for slot in slots),sign) for slots,sign in raw_sides(problem.tiles))
    template = getattr(problem, "_anchor_parity_template", None)
    build = template is None or template.sides != sides
    if build:
        template = SignedParity(problem,sides=sides)
        problem._anchor_parity_template = template
    query = copy(template)
    query.base = PhaseBasis()
    query.base.pivots = template.base.pivots.copy()
    query.base.conflict = template.base.conflict
    return query, build


def check_rules(problem, pins, deadline, *, candidates=None, project=False, required=(), physical_support=False):
    """Each returned pin assignment and its global complement exclude every board."""
    proven = ()
    if candidates is None:
        if has_proved_mesh(problem):
            proven = tuple((t, (PROVED_PHASE >> t) & 1) for t in range(problem.n))
        fingerprint, candidates = _published()
        if not proven and (problem.n != 160 or fingerprint != problem.fingerprint):
            candidates = ()
    accepted = []
    stats = dict(replayed=0, reused=0, clauses=0, terminal=False)
    if not candidates and not proven:
        return accepted, stats
    parity = None
    if project and (required or 0 < len(pins) < problem.n):
        parity, built = _query_parity(problem)
        stats["inventory_builds"] = int(built)
        if physical_support:
            from r32_gold_bonds import inventory
            required = (*required, *inventory(problem).required)
        for a, b in required:
            parity.bond(a, b)
        for tile, value in pins.items():
            parity.pin(tile, value)
        if parity.base.conflict is not None:
            stats.update(terminal=True, reason="signed inventory")
            return accepted, stats

    def include(row):
        if parity is not None:
            shorter = project_rule(parity.base, row)
            if shorter is None:
                return False
            stats["redundant_relative_signs"] = stats.get("redundant_relative_signs", 0) + len(row) - len(shorter)
            row = shorter
        accepted.append(row)
        stats["clauses"] += 2
        if len(row) <= 1 or (all(tile in pins for tile, _ in row) and
                             len({pins[tile] ^ bit for tile, bit in row}) <= 1):
            stats["terminal"] = True
        return stats["terminal"]

    if proven:
        from r35_phase_cube_rules import published_rules
        cubes = published_rules()
        stats["certified_fixed_phase"] = True
        stats["stored_partial_phases"] = len(cubes)
        # The complete phase is covered by any certified subset with matching relative signs.
        complete_covered = any(len({((PROVED_PHASE >> t) & 1) ^ bit for t, bit in row}) <= 1
                               for row in cubes)
        for row in cubes if complete_covered else (*cubes, proven):
            if include(row):
                return accepted, stats
    entries = []
    for anchor, row in candidates:
        signature = sha256(json.dumps([anchor, row], separators=(",", ":")).encode("ascii")).hexdigest()
        entries.append((anchor, row, bool(proven and signature in CHECKED_ANCHORS)))
    if any(not stored for _, _, stored in entries):
        # A proof on restricted base domains must never leak into wider ones.
        base = tuple(domain_bits(problem))
        mesh = tuple(tuple(zip(map(int, neighbors), map(int, sides)))
                     for neighbors, sides in zip(problem.neighbors, problem.sides))
        scope = (base, mesh, tuple(raw_sides(problem.tiles)))
        old_scope, saved = getattr(problem, "_verified_anchor_rules", (None, None))
        if old_scope != scope:
            saved = set()
            problem._verified_anchor_rules = scope, saved
    for anchor, row, stored in entries:
        if stored:
            stats["stored_geographic_rules"] = stats.get("stored_geographic_rules", 0) + 1
        elif row in saved:
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
        if include(row):
            break
    return accepted, stats


def post_rules(model, phase, rows):
    for row in rows:
        for complement in (0, 1):
            model.add_bool_or([~phase[tile] if bit ^ complement else phase[tile] for tile, bit in row])
