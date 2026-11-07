"""Bounded structural GPU-trail certificate producer. Every output is a claim.

The independent common checker owns authority. Missing learned-clause premises,
work/byte caps and interrupted production return no certificate. Native decision
bits are never used as logical premises, and enumeration history is forbidden.
"""
from __future__ import annotations
import hashlib,json
from validator import reverse_mask

COUNTERS=('gpu_proof_attempts','gpu_proof_events','gpu_proof_support_values','gpu_proof_matching_edges',
 'gpu_proof_loop_trials','gpu_proof_unknown','gpu_proof_cap_hits','gpu_proof_certificates')

class ProductionLimit(Exception):pass

class _Producer:
    def __init__(self,problem,check,stats,limit):self.problem=problem;self.check=check;self.stats=stats;self.left=limit
    def work(self,key,amount=1):
        self.check();self.left-=amount;self.stats[key]+=amount
        if self.left<0:raise ProductionLimit
    def hall(self,domains):
        n=len(domains);owners=[-1]*n;matching=[-1]*n
        for root in range(n):
            queue=[root];seen={root};parent={root:-1};found=False
            for cell in queue:
                tiles=sorted({code//3 for code in values(domains[cell])})
                for tile in tiles:
                    self.work('gpu_proof_matching_edges');owner=owners[tile]
                    if owner<0:
                        here=cell;free=tile
                        while here>=0:
                            prior=matching[here];matching[here]=free;owners[free]=here;free=prior;here=parent[here]
                        found=True;break
                    if owner not in seen:seen.add(owner);queue.append(owner);parent[owner]=cell
                if found:break
            if not found:return sorted(seen)
        return None
    def loop(self,domains):
        self.work('gpu_proof_loop_trials')
        board=[mask.bit_length()-1 if mask and not mask&(mask-1) else -1 for mask in domains]
        if self.problem.partial_loop_impossible(board):return [[int(c),int(v)] for c,v in self.problem.gold_nogood(board)]
        return None
    def removal(self,domains,cell,code,target):
        problem=self.problem;trial=list(domains);trial[cell]=1<<code
        for side,neighbor in enumerate(problem.neighbors[cell]):
            if neighbor<0 and int(problem.masks[code,side]):return 'boundary',{'side':side}
        for neighbor in sorted({int(c) for c in problem.neighbors[cell] if c>=0}):
            seams=[(side,int(problem.sides[cell,side])) for side in range(3) if int(problem.neighbors[cell,side])==neighbor]
            support=0
            for other in values(trial[neighbor]):
                self.work('gpu_proof_support_values')
                if all(int(problem.masks[code,a])==reverse_mask(int(problem.masks[other,b])) for a,b in seams):support|=1<<other
            if not support:return 'seam',{'neighbor':neighbor}
            trial[neighbor]=support
        tile=7<<(3*(code//3))
        for other in range(problem.n):
            if other!=cell:trial[other]&=~tile
        cells=self.hall(trial)
        if cells is not None:return 'hall',{'cells':cells}
        if target=='gold':
            # The loop rule checks only explicit singleton placements, not the
            # seam/Hall trial reductions above, which are not recorded events.
            trial=list(domains);trial[cell]=1<<code;placements=self.loop(trial)
            if placements is not None:return 'loop',{'placements':placements}
        return None

def values(mask):
    while mask:bit=mask&-mask;mask^=bit;yield bit.bit_length()-1

def produce(problem,target,root,decisions,trail,*,check=lambda:None,max_events=2048,max_bytes=1048576,max_work=200000,stats=None):
    """Return (untrusted certificate or None, charged producer work).

    root, decisions and trail are copied native facts supplied by the adapter;
    the verifier still revalidates their full shape and original-input scope.
    """
    if any(type(v)is not int or v<1 for v in (max_events,max_bytes,max_work)) or max_events>2048 or max_bytes>1048576 or max_work>2000000:
        raise ValueError('Invalid bounded proof production limits.')
    n=problem.n;original=[sum(1<<v for v in row) for row in problem.domains]
    if target not in ('gold','edges') or not isinstance(root,(list,tuple)) or len(root)!=n or any(type(v)is not int or v<=0 or v&~allowed for v,allowed in zip(root,original)):
        raise ValueError('Invalid structural proof root.')
    if not isinstance(decisions,(list,tuple)) or len(decisions)>n:raise ValueError('Invalid structural decision prefix.')
    seen=set()
    for row in decisions:
        if not isinstance(row,(list,tuple)) or len(row)!=2:raise ValueError('Invalid structural decision row.')
        c,mask=row
        if type(c)is not int or not 0<=c<n or c in seen or type(mask)is not int or mask<=0 or mask&~root[c]:raise ValueError('Structural decision escapes root.')
        seen.add(c)
    if not isinstance(trail,(list,tuple)) or len(trail)>n*3*n or any(type(v)is not int or not 0<=v<n*3*n for v in trail):raise ValueError('Invalid structural removal trail.')
    stats={} if stats is None else stats
    for key in COUNTERS:stats.setdefault(key,0)
    stats['gpu_proof_attempts']+=1
    worker=_Producer(problem,check,stats,max_work);domains=list(root)
    certificate={'format':'diamond-deduction-chain-v1','fingerprint':problem.fingerprint,'target':target,
        'original':[hex(sum(1<<v for v in row)) for row in problem.domains],'root':[hex(v) for v in root],
        'decisions':[[c,hex(mask)] for c,mask in decisions],'events':[],'terminal':None}
    try:
        check()
        for cell,mask in decisions:domains[cell]&=mask
        size=len(json.dumps(certificate,separators=(',',':')).encode('utf-8'))
        for encoded in trail:
            check();cell,code=divmod(int(encoded),3*problem.n);bit=1<<code
            if not domains[cell]&bit:continue
            if len(certificate['events'])>=max_events:raise ProductionLimit
            reason=worker.removal(domains,cell,code,target)
            if reason is None:stats['gpu_proof_unknown']+=1;return None,stats
            rule,data=reason;event={'rule':rule,'cell':cell,'removed':hex(bit),'data':data}
            size+=len(json.dumps(event,separators=(',',':')).encode('utf-8'))+2
            if size+4096>max_bytes:raise ProductionLimit
            certificate['events'].append(event);domains[cell]&=~bit;stats['gpu_proof_events']+=1
        terminal=None
        for cell,mask in enumerate(domains):
            if not mask:terminal={'rule':'empty','data':{'cell':cell}};break
        if terminal is None:
            cells=worker.hall(domains)
            if cells is not None:terminal={'rule':'hall','data':{'cells':cells}}
        if terminal is None and target=='gold':
            placements=worker.loop(domains)
            if placements is not None:terminal={'rule':'loop','data':{'placements':placements}}
        if terminal is None:stats['gpu_proof_unknown']+=1;return None,stats
        certificate['terminal']=terminal;check()
        if len(json.dumps(certificate,sort_keys=True).encode('utf-8'))>max_bytes:raise ProductionLimit
        stats['gpu_proof_certificates']+=1;return certificate,stats
    except ProductionLimit:
        stats['gpu_proof_cap_hits']+=1;return None,stats
