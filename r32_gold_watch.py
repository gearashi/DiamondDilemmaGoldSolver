"""Replayable location-independent Gold clauses translated into placement restrictions."""
from r32_gold_loss import Certificates, pairset


def placed_bonds(problem, placements):
    loc = dict(placements)
    result = {}
    for a,s,b,t in problem.edges:
        if a not in loc or b not in loc:
            continue
        x,y=loc[a],loc[b]
        u,v=3*(x//3)+(s-x%3)%3,3*(y//3)+(t-y%3)%3
        result[min(u,v),max(u,v)]=((a,x),(b,y))
    return result


def verified_placement(problem, claims, placements):
    if len(dict(placements))!=len(placements) or any(not 0<=c<problem.n or v not in problem.domains[c]
                                                     for c,v in placements):
        return False
    rows=pairset(claims)
    if not rows<=placed_bonds(problem,placements).keys():return False
    return bool(Certificates(problem).conflicts(rows))


class BondWatcher:
    """Unit physical clauses prune a side's mate wherever its other tile is fixed."""
    def __init__(self, problem, claims, check=lambda:None):
        self.problem=problem
        cert=Certificates(problem)
        for row in claims:
            check()
            cert.add([row])
        self.clauses=tuple(sorted(cert.rows,key=lambda row:(len(row),sorted(row))))
        self.applied=self.removed=self.conflicts=0

    def prune(self, domains):
        if not self.clauses:return False
        p=self.problem
        loc={c:m.bit_length()-1 for c,m in enumerate(domains) if m and m.bit_count()==1}
        if not loc:return False
        present=placed_bonds(p,loc)
        if not present and all(len(g)>1 for g in self.clauses):return False
        owners={v//3:(c,v) for c,v in loc.items()}
        original=list(domains)
        for core in self.clauses:
            missing=core-present.keys()
            if not missing:
                domains[min(c for c,_ in present[next(iter(core))])] = 0
                self.conflicts+=1
                break
            if len(missing)!=1:continue
            a,b=next(iter(missing))
            for x,y in ((a,b),(b,a)):
                known=owners.get(x//3)
                if known is None:continue
                c,v=known
                side=(x%3+v%3)%3
                n,t=int(p.neighbors[c,side]),int(p.sides[c,side])
                if n>=0:
                    code=3*(y//3)+(t-y%3)%3
                    if domains[n]>>code&1:
                        domains[n]&=~(1<<code)
                        self.applied+=1
        self.removed+=sum((b^a).bit_count() for a,b in zip(domains,original))
        return original!=domains
