"""Authenticated partial geographic phase exclusions on the published signed mesh."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

_CERTIFICATE = Path(__file__).parent / "docs" / "R35_PARTIAL_PHASE_CORES.json"
_CERTIFICATE_HASH = "3e1d1185a64e6c6da895637efb7c1d4cb264a94d54b92a37f1972530478ee68f"


_DIRECT = _CERTIFICATE.with_name("R35_PARTIAL_PHASE_PROOF.json")
_DIRECT_HASH = "5b88e5e26746e18c6a30abe73f73c43550a8a6edc27264a146a7b5f9306909a5"


def _authenticated(document, digest):
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf8")
    except (TypeError, ValueError, RecursionError):
        return False
    return sha256(encoded).hexdigest() == digest


def _conjunction(row):
    return tuple((abs(literal) - 1, int(literal > 0)) for literal in row["pins"])


def certificate_rules(document):
    if _authenticated(document, _CERTIFICATE_HASH):
        return tuple(_conjunction(row) for row in document["rules"])
    return ()


def direct_rule(document):
    if _authenticated(document, _DIRECT_HASH):
        return (_conjunction(document["partial_rule"]),)
    return ()


@lru_cache(maxsize=1)
def published_rules():
    rules = []
    for file, decode in ((_CERTIFICATE, certificate_rules), (_DIRECT, direct_rule)):
        try:
            rules.extend(decode(json.loads(file.read_bytes())))
        except (OSError, ValueError, RecursionError):
            continue
    return tuple(rules)


def certified_rules(problem):
    """Return conjunctions whose two complementary gauges both exclude every board."""
    from r35_anchor_rules import has_proved_mesh
    return published_rules() if has_proved_mesh(problem) else ()
