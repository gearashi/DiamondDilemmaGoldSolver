"""Apply the independently verified published physical loss bound to geographic Gold."""
from pathlib import Path

from r32_gold_bonds import inventory
from r32_gold_loss import digest, read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r32_gold_watch import BondWatcher
from r35_loss_bound import CHECKED_LOWER_BOUND, REFERENCE_SCOPE


def certified_reference(problem, codes=None):
    """Rebuild the exact mathematical scope that the shipped DRAT proof covers."""
    if problem.fingerprint != 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
        return None
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
    return frozenset(reference),len(reference)-CHECKED_LOWER_BOUND


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
