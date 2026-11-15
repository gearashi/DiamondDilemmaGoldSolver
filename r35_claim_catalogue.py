"""Reuse individually checked unconditional claims on their exact published mesh."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from r35_geographic_replay import geographic_departures
from r35_induced_contacts import contact_departures
from r35_path_mates import circuit_departures


DOCS = Path(__file__).resolve().parent / "docs"
RECEIPT_HASH = "4ca12cb2345f280a52fb61ebc83f3b79bedea4fe36e92cc235e01ea81457a34d"
KINDS = ("geographic_cores", "edge_perfect_cores", "gold_path_circuits")
LIVE = (geographic_departures, contact_departures, circuit_departures)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode("ascii")).hexdigest()


def _bonds(rows):
    result = []
    for e in rows:
        if not isinstance(e,(list,tuple)) or len(e)!=2 or any(type(side) is not int for side in e):
            raise ValueError("Expected pairs of literal physical side indices")
        result.append(tuple(sorted(e)))
    return tuple(sorted(result))


def _key(kind,hint):
    if kind == "gold_path_circuits":
        return _bonds(hint["path"]), _bonds([hint["mate"]])[0], _bonds(hint["loop"])
    return _bonds(hint)


@lru_cache(maxsize=3)
def _catalogue(kind):
    try:
        manifest = json.loads((DOCS/"R35_CLAIM_CATALOGUE.json").read_bytes())
        if digest(manifest) != RECEIPT_HASH:
            return {}
    except (OSError,ValueError,TypeError,RecursionError):
        return {}
    known = {}
    for source in manifest["catalogues"]:
        if source["kind"] != kind:
            continue
        try:
            data = json.loads((DOCS/source["source"]).read_bytes())
            keys = sorted(_key(kind,row) for row in data[kind])
            if digest(keys) != source["sha256"]:
                continue
        except (OSError,ValueError,TypeError,KeyError,IndexError,RecursionError):
            continue
        for key in keys:
            if kind == "gold_path_circuits":
                path, mate, loop = key
                group = frozenset(path) | (frozenset(loop)-{mate})
            else:
                group = frozenset(key)
            known[key] = group
    return known


def published_scope(problem):
    """Authenticate the full proof mesh even if this caller fixes tiles or narrows poses."""
    if problem.n != 160 or problem.total_segments != 365:
        return False
    if problem.fixed:
        identity = {"tiles":problem.tiles,"edges":problem.edges,"fixed":sorted(problem.fixed.items())}
        if digest(identity) != problem.fingerprint:
            return False
    elif problem.fingerprint != "e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853":
        return False
    from r35_anchor_rules import has_proved_mesh
    full = set(range(3*problem.n))
    if (len(problem.domains) != problem.n or
            any(len(set(d)) != len(d) or not full.issuperset(d) for d in problem.domains)):
        return False
    if not has_proved_mesh(problem) or set(problem.board.edges)!=set(problem.edges):
        return False
    arcs = [sorted(tuple(sorted(map(tuple,segment))) for segment in tile["segments"]) for tile in problem.tiles]
    return (digest(arcs)=="45ec5038bace5d85f931867f735a200d001412aa3347cf743648b379d78fbae9" and
            digest(problem.masks.tolist())=="ed9ffdaf19011396569315abf2add92620a88c4906780191e2874f8fad67c5a2")


def checked_groups(kind,hints):
    """Return all raw unconditional groups, or None to request live proof of this family."""
    catalogue = _catalogue(kind)
    try:
        return {catalogue[_key(kind,hint)] for hint in hints}
    except (KeyError,TypeError,ValueError,IndexError):
        return None


def _applicable(problem,kind,hints,scoped,domains=None):
    if not scoped:
        return None
    groups = checked_groups(kind,hints)
    # A live geometric proof can yield a smaller, stronger core in a warm process.
    if groups and kind == "geographic_cores":
        replay = getattr(problem,"_geographic_replay",None)
        if replay and replay.proofs:
            from r27_structure_filters import domain_bits
            ds = tuple(domains) if domains is not None else domain_bits(problem)
            for scope,core in replay.proofs:
                if (scope is None or all(x & ~y == 0 for x,y in zip(ds,scope))) and any(core < g for g in groups):
                    return None
    return groups


def embedding_claims(problem,domains,geographic,contacts,circuits,*,published):
    from r35_geographic_replay import replay_for
    from r35_induced_contacts import verified_contacts
    from r35_path_mates import verified_circuits
    scoped = published and published_scope(problem)
    if scoped:
        from r29_gold_geography import checked_domains
        domains = checked_domains(problem,domains)
    outputs = []
    reused = {}
    for kind,hints in zip(KINDS,(geographic,contacts,circuits)):
        if not hints:
            outputs.append(())
            continue
        hints = tuple(hints)
        groups = _applicable(problem,kind,hints,scoped,domains)
        if groups is not None:
            result = tuple(sorted(groups,key=lambda r:(len(r),sorted(r))))
            reused[kind] = len(groups)
        elif kind == "geographic_cores":
            result = replay_for(problem).check(domains,hints)
        elif kind == "edge_perfect_cores":
            result = verified_contacts(problem,hints)
        else:
            result = verified_circuits(problem,hints)
        outputs.append(result)
    return (*outputs,reused)


def public_departures(problem,reference,required,geographic_hints,contact_hints,circuit_hints,*,published):
    families = (geographic_hints,contact_hints,circuit_hints)
    scoped = published and any(families) and published_scope(problem)
    required = frozenset(required)
    allowed = frozenset(reference) | required
    outputs = []
    reused = {}
    for kind,raw,hints in zip(KINDS,LIVE,families):
        if not hints:
            outputs.append(())
            continue
        hints = tuple(hints)
        groups = _applicable(problem,kind,hints,scoped)
        if groups is None:
            result = raw(problem,reference,required,hints)
        else:
            eligible = {row for row in groups if row <= allowed}
            result = tuple(sorted({row-required for row in eligible},key=lambda r:(len(r),sorted(r))))
            reused[kind] = len(eligible)
        outputs.append(result)
    return (*outputs,reused)
