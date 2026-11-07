"""Bounded contextual scheduling from completed observations only.

These predictions select search order or optional effort. They never remove a
placement, establish a proof, or turn a timeout into a training label. Contexts
use scale-free domain and seam features. CPU nodes are compared only with CPU
nodes; runtime engine allocation keeps its own separate policy.
"""
from collections import OrderedDict
import copy
import math

PROFILES = ('base', 'impact', 'support', 'constructive')
MAX_RECORDS = 512

def features(problem, domains):
    widths = [v.bit_count() for v in domains]
    n = len(widths)
    if not n or any(not w for w in widths):
        raise ValueError('Context needs nonempty domains.')
    free = sum(w > 1 for w in widths)
    largest = max(widths)
    fixed = set(i for i,w in enumerate(widths) if w == 1)
    seams = sum((a in fixed) != (b in fixed) for a,sa,b,sb in problem.edges)
    tile_width = sum(len({code//3 for code in range(mask.bit_length()) if mask>>code&1}) for mask in domains)
    return (free/n, math.log2(largest+1)/math.log2(3*n+1),
            sum(widths)/(n*largest), seams/max(1,len(problem.edges)), tile_width/max(1,sum(widths)))


def _vector(row):
    if (not isinstance(row,(list,tuple)) or len(row)!=5
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in row)):
        raise ValueError('Invalid contextual feature vector.')
    return tuple(float(v) for v in row)


class ContextPolicy:
    def __init__(self, payload=None):
        self.records=[]; self.turn=0
        if payload is not None:
            if not isinstance(payload,dict) or set(payload)!={'version','turn','records'} or payload['version']!=1:
                raise ValueError('Invalid contextual model.')
            if type(payload['turn']) is not int or payload['turn']<0 or not isinstance(payload['records'],list) or len(payload['records'])>MAX_RECORDS:
                raise ValueError('Invalid contextual model size.')
            for row in payload['records']:
                if not isinstance(row,dict) or set(row)!={'features','profile','nodes','case'}:
                    raise ValueError('Invalid completed residual label.')
                _vector(row['features'])
                if (row['profile'] not in PROFILES or type(row['nodes']) is not int or row['nodes']<0
                        or not isinstance(row['case'],str) or len(row['case'])>160):
                    raise ValueError('Invalid completed residual work.')
            self.records=copy.deepcopy(payload['records']); self.turn=payload['turn']

    def observe(self, vector, profile, *, nodes, complete, case):
        vector=_vector(vector)
        if type(complete) is not bool or profile not in PROFILES or type(nodes) is not int or nodes<0 or not isinstance(case,str) or len(case)>160:
            raise ValueError('Invalid contextual observation.')
        if not complete:return False
        # A repeated checkpoint/result cannot multiply a completed case label.
        identity=(case,profile)
        if any((r['case'],r['profile'])==identity for r in self.records):return False
        self.records.append({'features':list(vector),'profile':profile,'nodes':nodes,'case':case})
        if len(self.records)>MAX_RECORDS:del self.records[0]
        return True

    def choose(self, vector, *, explore=True, exclude_case=None):
        vector=_vector(vector);self.turn+=1
        rows=[r for r in self.records if r['case']!=exclude_case]
        by={p:[r for r in rows if r['profile']==p] for p in PROFILES}
        if explore and (self.turn%5==0 or not rows):
            return min(PROFILES,key=lambda p:(len(by[p]),PROFILES.index(p)))
        scored=[]
        for profile in PROFILES:
            if not by[profile]:continue
            near=sorted(by[profile],key=lambda r:sum((a-b)**2 for a,b in zip(vector,r['features'])))[:8]
            distances=[1/(.02+sum((a-b)**2 for a,b in zip(vector,r['features']))) for r in near]
            prediction=sum(w*math.log1p(r['nodes']) for w,r in zip(distances,near))/sum(distances)
            scored.append((prediction,PROFILES.index(profile),profile))
        return min(scored)[2] if scored else 'base'

    def snapshot(self):return {'version':1,'turn':self.turn,'records':copy.deepcopy(self.records)}


def profile_options(options, profile):
    if profile not in PROFILES:raise ValueError('Unknown contextual strategy.')
    out=dict(options)
    if profile=='impact':out.update(impact_branching=True, failure_guided=True)
    elif profile=='support':out.update(value_ordering=True, objective_ordering=True, ordering_goal='conflict')
    elif profile=='constructive':out.update(lookahead=True,pair_lookahead=True,group_lookahead=True,connection_branching=True)
    return out


class ContextualPropagation:
    """Per-context retry/effort policy; interruption never enters the model."""
    def __init__(self, limit=128):
        if type(limit) is not int or not 1<=limit<=1024:raise ValueError('Invalid context capacity.')
        self.limit=limit;self.records=OrderedDict();self.turns={}

    @staticmethod
    def key(name, domains):
        widths=[v.bit_count() for v in domains]
        n=max(1,len(widths));free=sum(v>1 for v in widths)
        return (name,min(3,4*free//n),min(5,max(widths,default=0).bit_length()))

    def allow(self,name,domains):
        key=self.key(name,domains)
        if key not in self.records:return True
        row=self.records[key];self.records.move_to_end(key)
        calls,useful,removed=row
        tick=self.turns.get(key,0)+1;self.turns[key]=tick
        # Failed predictions always receive another full opportunity. A context
        # with any demonstrated pruning is retained regardless of elapsed cost.
        return calls<4 or useful>0 or tick%4==0

    def observe(self,name,domains,*,removed,proofs,complete):
        if not complete:return
        key=self.key(name,domains);row=self.records.setdefault(key,[0,0,0])
        row[0]+=1;row[1]+=int(removed>0 or proofs>0);row[2]+=removed+proofs
        self.records.move_to_end(key)
        while len(self.records)>self.limit:
            old,_=self.records.popitem(last=False);self.turns.pop(old,None)
