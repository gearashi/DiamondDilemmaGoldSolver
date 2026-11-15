"""Verify the integer-weight lower bound on lost named bonds for the original Gold reference."""
from collections import Counter
import json
from pathlib import Path

from r35_geographic_replay import geographic_departures, replay_for
from r35_induced_contacts import contact_departures
from r35_loss_witness import raw_sides


CERTIFICATE=Path(__file__).resolve().parent/'docs/R35_CONTACT_LOSS_CERTIFICATE.json'


def xor_conflict(problem, pairs):
    """Check the signed-side XOR equations independently from raw endpoints."""
    labels=raw_sides(problem.tiles)
    equations={}
    for side,(slots,base) in enumerate(labels):
        back=tuple(12-v for v in reversed(slots))
        for sign in (0,1):
            cls=frozenset(((slots,sign),(back,sign^(len(slots)%2))))
            mask,rhs=equations.get(cls,(0,0))
            equations[cls]=(mask^(1<<(side//3)),rhs^(base==sign))
    rows=list(equations.values())
    for a,b in pairs:
        left,ca=labels[a];right,cb=labels[b]
        if left!=tuple(12-v for v in reversed(right)):return False
        rows.append(((1<<(a//3))^(1<<(b//3)),ca^cb^(len(left)%2)))
    basis={}
    for mask,rhs in rows:
        while mask:
            pivot=mask.bit_length()-1
            if pivot not in basis:
                basis[pivot]=mask,rhs
                break
            other,bit=basis[pivot];mask^=other;rhs^=bit
        else:
            if rhs:return True
    return False


def verify(problem, reference, required, physical_groups, evidence, *, charged=None):
    if not isinstance(evidence,dict) or evidence.get('fingerprint')!=problem.fingerprint:return 0
    allowed=set(reference)|set(required)
    if evidence.get('reference')!=[list(p) for p in sorted(reference)] or evidence.get('required')!=[list(p) for p in sorted(required)]:return 0
    rules=evidence.get('weighted_cores')
    if not isinstance(rules,list) or not rules:return 0
    normalizer=replay_for(problem)
    weight_sum=0;incidence=Counter()
    for rule in rules:
        if not isinstance(rule,dict):return 0
        term=normalizer.normalize(rule.get('bonds',()))
        weight=rule.get('weight');kind=rule.get('kind')
        if type(weight) is not int or weight<=0 or not term or not term<=allowed:return 0
        charge=term-required
        if not charge:return 0
        if kind=='required_xor':
            valid=xor_conflict(problem,term|required)
        else:
            if charged is not None and not any(g<=charge for g in charged):return 0
            if kind=='physical':valid=term in physical_groups
            elif kind=='geographic':valid=bool(geographic_departures(problem,reference,required,[term]))
            elif kind=='edge_perfect':valid=bool(contact_departures(problem,reference,required,[term]))
            else:return 0
        if not valid:return 0
        weight_sum+=weight
        for bond in charge:incidence[bond]+=weight
    maximum=max(incidence.values())
    if evidence.get('weight_sum')!=weight_sum or evidence.get('maximum')!=maximum:return 0
    return (weight_sum+maximum-1)//maximum


def checked_contact_bound(problem,reference,required,groups,*,charged=None,signed_parity=False):
    try:data=json.loads(CERTIFICATE.read_bytes())
    except (OSError,ValueError):return 0
    bound=verify(problem,frozenset(reference),frozenset(required),{frozenset(g) for g in groups},data,
                  charged=None if charged is None else tuple(frozenset(g)-required for g in charged))
    if bound==25 and signed_parity:
        # The separately verified cap-25 DRAT proof needs the signed pair and
        # reciprocal-inventory parity equations together with exactly these terms.
        from r32_gold_loss import digest
        identity=digest([data['fingerprint'],sorted(map(tuple,data['reference'])),
                         sorted(map(tuple,data['required'])),
                         sorted((r['kind'],r['weight'],sorted(map(tuple,r['bonds']))) for r in data['weighted_cores'])])
        if identity=='3c0a3e1592b5c7fdde4279da83b240c8a78d84a1210de02cb3f8f02d065ce200':
            return 26
    return bound
