"""Apply the independently verified published physical loss bound to geographic Gold."""
from pathlib import Path

from r32_gold_bonds import inventory
from r32_gold_loss import digest, read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r32_gold_watch import BondWatcher
from r35_loss_bound import REFERENCE_SCOPE, checked_lower_bound


def certified_reference(problem, codes=None):
    """Rebuild the physical scope of the independently checked reference-loss bounds."""
    if problem.fingerprint != 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
        return None
    from hashlib import sha256
    from r35_contact_certificate import CERTIFICATE
    try:
        if codes is None:
            codes=read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
        if len(codes)!=problem.n or any(type(c) is not int or not 0<=c<3*problem.n for c in codes):
            return None
        reference,_,groups=reference_groups(problem,codes)
    except (OSError,ValueError,TypeError,KeyError):
        return None
    packed,required,_=OutsideSyndromes.layout(reference,groups,inventory(problem).required)
    identity=digest([3,problem.fingerprint,sorted(reference),sorted(required),[sorted(g) for g in packed]])
    if identity!=REFERENCE_SCOPE:return None
    # Rebuild the physical identity on every request. The expensive geometry
    # replay is reusable only with the same overlap groups and evidence bytes.
    source=sha256()
    for name in (CERTIFICATE.name,'R35_CONTACT_WIDE_PROOF.json','R35_SPHERE_CONTACT_PROOF.json','R35_GEOGRAPHIC_RESIDUALS.json','R35_EDGE_CONTACTS.json'):
        try:
            evidence=(CERTIFICATE.parent/name).read_bytes()
        except OSError:
            evidence=b''
        source.update(sha256(evidence).digest())
    sphere_input=digest([problem.edges,[tile['segments'] for tile in problem.tiles]])
    key=(identity,frozenset(frozenset(group) for group in groups),source.digest(),sphere_input)
    cached=getattr(problem,'_geographic_loss_reference',None)
    if cached is not None and cached[0]==key:
        return cached[1]
    from r35_contact_certificate import checked_contact_bound
    loss=max(checked_lower_bound(identity,groups),checked_contact_bound(problem,reference,required,groups,signed_parity=True,integer=True,sphere=True))
    result=(frozenset(reference),len(reference)-loss)
    problem._geographic_loss_reference=(key,result)
    return result


def overlap_lower_bound(source, retained_limit, target, required=(), forbidden=()):
    """Minimum lost target bonds implied by the source retention cap and branch literals."""
    from r32_gold_loss import pairset
    required, forbidden = pairset(required), pairset(forbidden)
    # Every side of a physical triangle has exactly one mate in Gold.
    occupied = {side for pair in required for side in pair}
    if len(occupied) != 2*len(required) or required & forbidden:
        return None
    source, target = pairset(source), pairset(target)
    if len(source & required) > retained_limit:
        return None
    eligible = {p for p in target if p not in forbidden and
                (p in required or occupied.isdisjoint(p))}
    shared = source & eligible
    remaining = retained_limit - len(required & (source-target))
    return len(target) - len(eligible-source) - min(remaining,len(shared))


def geographic_loss_cutoff(problem, target, upper, required, forbidden):
    """Use a certified geographic bound only during a full-board geographic search."""
    from r35_claim_catalogue import published_scope
    if upper >= len(target) or not published_scope(problem):
        return None
    proof = certified_reference(problem)
    if proof is None:
        return None
    source, limit = proof
    physical = inventory(problem)
    lower = overlap_lower_bound(source,limit,target,set(required)|physical.required,
                                set(forbidden)|physical.forbidden)
    if lower is not None and lower <= upper:
        return None
    return dict(status='INFEASIBLE',cp_calls=0,branches=0,
                certified_reference_overlap=dict(source_retained_limit=limit,
                    shared_bonds=len(source & set(target)),target_bonds=len(target),
                    minimum_lost=lower,required_inconsistent=lower is None))


def retained_bound(problem,domains,reference,limit,required=()):
    """For a proven bound, count distinct physically entailed joins and all saturated units."""
    scope=frozenset(reference)
    entailed=scope.intersection(tuple(sorted(p)) for p in required)
    poses=BondWatcher(problem,())
    optional={}
    for pair in sorted(scope-entailed):
        left,ls=poses._forced_poses(domains,*pair)
        right,rs=poses._forced_poses(domains,*reversed(pair))
        if left or right:entailed=entailed|{pair}
        else:optional[pair]=(*ls,*rs)
    result=dict(limit=limit,forced=len(entailed),over=len(entailed)>limit,unit_poses=0,forbidden=[])
    if len(entailed)==limit:
        result['forbidden']=sorted(optional)
        for row in optional.values():
            for cell,code in row:
                if domains[cell]>>code&1:
                    domains[cell]^=1<<code
                    result['unit_poses']+=1
    elif len(entailed)<limit:
        from r35_alternative_retention import prune_alternatives
        result.update(prune_alternatives(domains,optional,limit-len(entailed)))
    return result
