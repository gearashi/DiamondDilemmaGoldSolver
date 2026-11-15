"""Authenticated partial geographic phase exclusions on the published signed mesh."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

_CERTIFICATE = Path(__file__).parent / "docs" / "R35_PARTIAL_PHASE_CORES.json"
_CERTIFICATE_HASH = "3e1d1185a64e6c6da895637efb7c1d4cb264a94d54b92a37f1972530478ee68f"


def certificate_rules(document):
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf8")
    except (TypeError, ValueError, RecursionError):
        return ()
    if sha256(encoded).hexdigest() != _CERTIFICATE_HASH:
        return ()
    return tuple(tuple((abs(lit)-1, int(lit>0)) for lit in row["pins"]) for row in document["rules"])


@lru_cache(maxsize=1)
def published_rules():
    try:
        return certificate_rules(json.loads(_CERTIFICATE.read_bytes()))
    except (OSError, ValueError, RecursionError):
        return ()


def certified_rules(problem):
    """Return conjunctions whose two complementary gauges both exclude every board."""
    from r35_anchor_rules import has_proved_mesh
    return published_rules() if has_proved_mesh(problem) else ()
