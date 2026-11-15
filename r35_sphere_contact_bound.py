"""Identify the certified reference-loss scope that also requires spherical Euler weights."""
import json
from pathlib import Path

from r32_gold_loss import digest
from verify_r35_loss_euler import reconstruct


RECEIPT = Path(__file__).resolve().parent / 'docs' / 'R35_SPHERE_CONTACT_PROOF.json'
SCOPE = '24919979925a62a4080fdacd356109fc60fb1b159f39e133d8139c134d902d7c'
SPHERE = '6e5b7f0802b0c68c98fb17eb8f7d910f1c4e69ee71051025eb80140926e48756'
CHECKED_PROOFS = {
    '0d8da6bbe9b13345147672123dee7fa05552d2e2c1e14e4837cff14e68b10b56': 32,
}


def checked_sphere_bound(problem, reference, required, physical, charged):
    if problem.fingerprint != 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
        return 0
    try:
        record = json.loads(RECEIPT.read_bytes())
        lower = CHECKED_PROOFS.get(record['proof_files']['proof.vipr']['sha256'])
        if (not lower or record['excluded_cap'] != lower-1 or record['puzzle'] != problem.fingerprint or
            record['sphere_sha256'] != SPHERE):
            return 0
        base = {frozenset(map(tuple,g)) for g in record['physical_groups']}
        if not base <= set(map(frozenset,physical)):
            return 0
        groups = set(base)
        for filename, key in (('R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
                              ('R35_EDGE_CONTACTS.json','edge_perfect_cores')):
            data = json.loads((RECEIPT.parent/filename).read_bytes())
            groups.update(frozenset(map(tuple,g)) for g in data[key])
        source = {g-required for g in groups}
        if digest([problem.fingerprint,sorted(reference),sorted(required),sorted(sorted(g) for g in source)]) != SCOPE:
            return 0
        if charged is not None:
            given = {frozenset(g)-required for g in charged}
            if any(not any(small<=g for small in given) for g in source-given):
                return 0
        weights, allowed, _ = reconstruct(problem)
        return lower if digest([weights,allowed]) == SPHERE else 0
    except (OSError,ValueError,TypeError,KeyError,IndexError):
        return 0
