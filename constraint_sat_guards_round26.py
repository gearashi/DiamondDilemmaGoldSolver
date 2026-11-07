"""Reusable per-cell domain guards for an assumption-based native SAT owner.

A retained guard occurs negatively in its defining clause. Omitting its positive
assumption therefore imposes no restriction. Entries are never disabled or
rebound to another mask. Lifetime admission is bounded; after the cap, excluded
placement literals express the same domain directly without allocating a guard.
Native identifiers are deliberately not durable semantic state.
"""
STAT_KEYS=('session_guard_requests','session_guard_creates','session_guard_hits',
           'session_guard_fallback_literals','session_guard_clauses','session_guard_retirements',
           'session_guard_pool_resets')

class DomainGuardPool:
    def __init__(self,root,variables,fresh,post,*,limit=8192,stats=None):
        if type(limit)is not int or not 0<=limit<=65536:raise ValueError('Invalid SAT guard lifetime bound.')
        self.root=tuple(root);self.variables=tuple(dict(row) for row in variables)
        if len(self.root)!=len(self.variables):raise ValueError('SAT guard dimensions differ.')
        used=set()
        for mask,row in zip(self.root,self.variables):
            if type(mask)is not int or mask<=0 or sum(1<<code for code in row)!=mask:raise ValueError('SAT guard domains differ from placement variables.')
            if any(type(code)is not int or code<0 or type(lit)is not int or lit<=0 or lit in used for code,lit in row.items()):raise ValueError('Invalid placement variable map.')
            if len(set(row.values()))!=len(row):raise ValueError('Duplicate placement variables.')
            used.update(row.values())
        self.fresh,self.post,self.limit=fresh,post,limit
        self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        self.entries={};self.poisoned=False

    def assumptions(self,domains):
        if self.poisoned:raise RuntimeError('A failed native guard write requires rebuilding the owner.')
        domains=tuple(domains)
        if len(domains)!=len(self.root) or any(type(m)is not int or m<=0 or m&~r for m,r in zip(domains,self.root)):
            raise ValueError('Guard domains escape the exact SAT root.')
        assumptions=[];meanings={}
        for cell,(domain,root) in enumerate(zip(domains,self.root)):
            if domain==root:continue
            self.stats['session_guard_requests']+=1;key=(cell,domain)
            literal=self.entries.get(key)
            if literal is not None:self.stats['session_guard_hits']+=1
            elif len(self.entries)<self.limit:
                literal=self.fresh()
                self.poisoned=True
                self.post([-literal,*[lit for code,lit in self.variables[cell].items() if domain>>code&1]])
                self.entries[key]=literal;self.poisoned=False
                self.stats['session_guard_creates']+=1;self.stats['session_guard_clauses']+=1
            else:
                for code,lit in self.variables[cell].items():
                    if not domain>>code&1:
                        assumptions.append(-lit);meanings[-lit]=(cell,root&~(1<<code))
                        self.stats['session_guard_fallback_literals']+=1
                continue
            assumptions.append(literal);meanings[literal]=(cell,domain)
        return tuple(assumptions),meanings
