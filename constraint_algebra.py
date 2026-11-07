"""Exact affine overapproximations of allowed seam-port assignments.

Each cell contributes the affine hull of its CURRENT allowed endpoint vectors.
Global seam coordinates identify paired ports. Every actual board satisfies
this system. A candidate inconsistent with it is impossible; a consistent
candidate is merely unknown. All arithmetic is exact GF(2).
"""
from collections import OrderedDict,deque
from constraint_relations import bits

STAT_KEYS=('affine_checks', 'affine_builds', 'affine_equations', 'affine_candidates', 'affine_xors', 'affine_prunes', 'affine_cache_hits', 'affine_incremental_builds', 'affine_full_builds', 'affine_reused_cells')


def add_equation(basis, mask, rhs, check=lambda:None, stats=None):
    row=(mask<<1)|(rhs&1)
    while row>>1:
        if stats is not None:stats['affine_xors']=stats.get('affine_xors',0)+1
        check();pivot=row.bit_length()-1
        if pivot not in basis:basis[pivot]=row;return True
        row^=basis[pivot]
    return row==0


def affine_equations(patterns,width):
    """Equations for the exact affine hull of a finite Boolean vector set."""
    patterns=tuple(dict.fromkeys(patterns))
    if not patterns:return ((0,1),)
    origin=patterns[0];basis={}
    for pattern in patterns[1:]:
        row=pattern^origin
        while row:
            pivot=row.bit_length()-1
            if pivot not in basis:basis[pivot]=row;break
            row^=basis[pivot]
    for pivot in sorted(basis):
        for other in tuple(basis):
            if other>pivot and basis[other]&(1<<pivot):basis[other]^=basis[pivot]
    result=[]
    for free in range(width):
        if free in basis:continue
        vector=1<<free
        for pivot,row in basis.items():
            if row&(1<<free):vector|=1<<pivot
        result.append((vector,(vector&origin).bit_count()%2))
    return tuple(result)


def modular_affine_equations(patterns,width,prime):
    """Reference modular invariant discovery for small finite vector sets.

    This is an exact research helper, not a floating-point nullspace. Other
    primes are not silently accepted. Runtime puzzle propagation below uses
    GF(2); the reference helper lets modular candidates be tested independently.
    """
    if prime not in (2,3,5,7):raise ValueError('Supported prime required.')
    patterns=tuple(tuple(row) for row in patterns)
    if any(len(row)!=width for row in patterns):raise ValueError('Vector width differs.')
    if not patterns:return ((tuple([0]*width),1),)
    origin=patterns[0];rows=[[(a-b)%prime for a,b in zip(row,origin)] for row in patterns[1:]]
    pivots=[];rank=0
    for col in range(width):
        found=next((r for r in range(rank,len(rows)) if rows[r][col]),None)
        if found is None:continue
        rows[rank],rows[found]=rows[found],rows[rank]
        inv=pow(rows[rank][col],-1,prime);rows[rank]=[(v*inv)%prime for v in rows[rank]]
        for r in range(len(rows)):
            if r==rank:continue
            scale=rows[r][col]
            if scale:rows[r]=[(a-scale*b)%prime for a,b in zip(rows[r],rows[rank])]
        pivots.append(col);rank+=1
    result=[]
    for free in range(width):
        if free in pivots:continue
        vector=[0]*width;vector[free]=1
        for r,pivot in enumerate(pivots):vector[pivot]=-rows[r][free]%prime
        result.append((tuple(vector),sum(a*b for a,b in zip(vector,origin))%prime))
    return tuple(result)


class AffinePruner:
    def __init__(self,problem,*,incremental=False,explain=False):
        if type(incremental)is not bool:raise ValueError('Incremental affine policy must be boolean.')
        if type(explain)is not bool:raise ValueError('Invalid affine explanation policy.')
        self.explain=explain;self.pending=deque(maxlen=128)
        self.problem=problem;self.cache=OrderedDict();self.incremental=incremental
        self.original=tuple(sum(1<<code for code in set(row)) for row in problem.domains) if incremental else ()
        self.binding=self._binding() if incremental else None
        self.stats=dict.fromkeys(STAT_KEYS,0)
        from constraint_affine_reasons_round26 import STAT_KEYS as REASON_KEYS
        self.reason_stats=dict.fromkeys(REASON_KEYS,0)
        self.patterns=tuple(sum(int(mask)<<(11*side) for side,mask in enumerate(row)) for row in problem.masks)
        self.coordinates=[[None]*33 for _ in range(problem.n)]
        for edge,(a,sa,b,sb) in enumerate(problem.edges):
            for port in range(11):
                self.coordinates[a][11*sa+port]=11*edge+port
                self.coordinates[b][11*sb+10-port]=11*edge+port
        self.enabled=all(all(v is not None for v in row) for row in self.coordinates)

    def _binding(self):
        return (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),
                tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))

    def _build(self,domains,check):
        self.stats['affine_builds']+=1
        ancestor=None;cost=None
        if self.incremental:
            # A wider saved box proves equations valid in every narrower box.
            # Siblings/widening may reuse only a genuine covering ancestor.
            for prior,session in self.cache.items():
                check()
                if all(not current&~old for current,old in zip(domains,prior)):
                    candidate=(sum(current!=old for current,old in zip(domains,prior)),
                               sum((old&~current).bit_count() for current,old in zip(domains,prior)))
                    if cost is None or candidate<cost:ancestor=(prior,session);cost=candidate
        if ancestor is None:
            basis={};changed=range(len(domains));self.stats['affine_full_builds']+=1
        else:
            prior,session=ancestor;self.stats['affine_incremental_builds']+=1
            changed=[c for c in range(len(domains)) if prior[c]!=domains[c]]
            self.stats['affine_reused_cells']+=len(domains)-len(changed)
            if session['basis'] is None:return None
            # Never mutate an ancestor: exact cached snapshots implement undo.
            basis=session['basis'].copy()
        for cell in changed:
            domain=domains[cell]
            check()
            local=affine_equations((self.patterns[code] for code in bits(domain)),33)
            for mask,rhs in local:
                global_mask=0
                for port in bits(mask):global_mask^=1<<self.coordinates[cell][port]
                self.stats['affine_equations']+=1
                if not add_equation(basis,global_mask,rhs,check,self.stats):return None
        return basis

    def _explain(self,domains,cell,code,check):
        if not self.explain:return
        from constraint_affine_reasons_round26 import produce_affine_conflict
        trial=list(domains)
        if cell is not None:trial[cell]=1<<code
        certificate=produce_affine_conflict(self.problem,trial,check,stats=self.reason_stats)
        if certificate is not None:self.pending.append((tuple(domains),cell,code,certificate))

    def drain_deduction_chains(self,target='gold',limit=8):
        from copy import deepcopy
        from constraint_deduction_chain_round26 import FORMAT
        if target not in ('gold','edges') or type(limit)is not int or not 0<=limit<=128:raise ValueError('Invalid affine export policy.')
        emitted=0;original=[hex(sum(1<<code for code in row)) for row in self.problem.domains]
        while self.pending and emitted<limit:
            item=self.pending[0];root,cell,code,certificate=item
            yield deepcopy({'format':FORMAT,'fingerprint':self.problem.fingerprint,'target':target,'original':original,'root':list(map(hex,root)),
                'decisions':[] if cell is None else [[cell,hex(1<<code)]],'events':[],'terminal':{'rule':'affine','data':{'certificate':certificate}}})
            if self.pending and self.pending[0] is item:self.pending.popleft()
            emitted+=1

    def prune(self,domains,check=lambda:None,*,max_candidates=128):
        if type(max_candidates)is not int or max_candidates<0:raise ValueError('Invalid affine candidate budget.')
        if self.incremental:
            if self._binding()!=self.binding:raise ValueError('Incremental affine puzzle binding changed.')
            if len(domains)!=len(self.original) or any(type(mask)is not int or mask<0 or mask&~old for mask,old in zip(domains,self.original)):
                raise ValueError('Incremental affine domains escape the original root.')
        if not self.enabled or not all(domains):return False
        self.stats['affine_checks']+=1;key=tuple(domains)
        session=self.cache.get(key)
        if session is None:
            basis=self._build(key,check);check()
            session={'basis':basis,'cursor':0,'reduced':key}
            self.cache[key]=session
            while len(self.cache)>4:self.cache.popitem(last=False)
        else:self.stats['affine_cache_hits']+=1;self.cache.move_to_end(key)
        if session['basis'] is None:
            self._explain(key,None,None,check);check();domains[0]=0;return True
        current=list(session['reduced'])
        candidates=[(cell,code) for cell in sorted(range(len(domains)),key=lambda c:(domains[c].bit_count(),c))
                    for code in bits(domains[cell]) if domains[cell].bit_count()>1]
        examined=0
        while session['cursor']<len(candidates) and examined<max_candidates:
            check();cell,code=candidates[session['cursor']];basis=session['basis'].copy()
            consistent=True;pattern=self.patterns[code]
            for port,variable in enumerate(self.coordinates[cell]):
                if not add_equation(basis,1<<variable,(pattern>>port)&1,check,self.stats):consistent=False;break
            # Commit a cursor only after the complete exact check.
            if not consistent:
                self._explain(key,cell,code,check);current[cell]&=~(1<<code)
            session['cursor']+=1;session['reduced']=tuple(current)
            self.stats['affine_candidates']+=1;examined+=1
        check();self.stats['affine_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
