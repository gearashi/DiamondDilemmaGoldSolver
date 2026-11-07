"""Bounded trusted-local continuation of complete constructive and pair probes.

Only completed branch outcomes enter support unions. A stopped individual probe
may be retried from its last sound narrowed domains by CompletedProbeCache.
These plans are application-local state, never imported proof authority.
"""
from collections import OrderedDict
import copy,json

STAT_KEYS=('probe_plan_builds','probe_plan_resumes','probe_plan_completed_skips','probe_plan_evictions',
 'probe_plan_callbacks_reused','probe_partial_saves','probe_partial_resumes')


def _freeze(value):
    if isinstance(value,(tuple,list)):return tuple(_freeze(v) for v in value)
    if value is None or type(value)in(str,int,bool,float):return value
    raise ValueError('Invalid local probe context.')


def _domains(value,source=None):
    if not isinstance(value,(tuple,list)) or not value or any(type(m)is not int or m<0 for m in value):raise ValueError('Invalid probe-plan domains.')
    result=tuple(value)
    if source is not None and (len(result)!=len(source) or any(m&~old for m,old in zip(result,source))):raise ValueError('Probe-plan domains escape input.')
    return result


def _positive(trial,before,cell,mask):
    trial=_domains(trial,before)
    if not all(trial) or trial[cell]&~mask:raise ValueError('Positive probe outcome escapes its branch.')
    return trial


class ProbePlans:
    def __init__(self,limit=16,max_bytes=2*1024*1024):
        if type(limit)is not int or not 0<=limit<=128 or type(max_bytes)is not int or not 0<=max_bytes<=16*1024*1024:raise ValueError('Invalid probe-plan cache bound.')
        self.limit,self.max_bytes=limit,max_bytes;self.records=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
    def _get(self,kind,context,domains,cell,groups,neighbors=(),*,cover=True):
        before=_domains(domains);context=_freeze(context)
        if type(cell)is not int or not 0<=cell<len(before):raise ValueError('Invalid probe-plan cell.')
        groups=tuple(groups);seen=0
        for mask in groups:
            if type(mask)is not int or mask<=0 or mask&~before[cell] or seen&mask:raise ValueError('Invalid probe partition.')
            seen|=mask
        if kind in ('single','pair') and any(mask&(mask-1) for mask in groups):raise ValueError('Singleton plans require singleton groups.')
        if seen!=before[cell]:raise ValueError('Probe plan must cover its entire choice domain.')
        key=(kind,context,before,cell,groups,tuple(neighbors));record=self.records.get(key)
        if record is None and cover:
            # A complete old partition also covers a narrower current box.
            # Keeping its wider successful outcomes can only weaken deductions.
            for prior,candidate in reversed(self.records.items()):
                if prior[0]==kind and prior[1]==context and prior[3]==cell and prior[5]==tuple(neighbors) and all(not now&~old for now,old in zip(before,prior[2])):
                    key=prior;record=candidate;break
        if record is not None:self.records.move_to_end(key);self.stats['probe_plan_resumes']+=1;return key,record
        record={'kind':kind,'context':context,'before':before,'cell':cell,'groups':groups,'neighbors':tuple(neighbors),'results':[],'done':False}
        self.records[key]=record;self.stats['probe_plan_builds']+=1;self._trim(key);return key,record
    def _trim(self,current):
        while self.records and (len(self.records)>self.limit or len(json.dumps(list(self.records.values()),separators=(',',':')).encode())>self.max_bytes):
            self.records.popitem(last=False);self.stats['probe_plan_evictions']+=1
    def constructive(self,context,domains,cell,groups,probe,check,on_failure,stats,*,grouped=False):
        check();key,state=self._get('group' if grouped else 'single',context,domains,cell,groups)
        before=state['before'];groups=state['groups'];results=state['results'];was=tuple(domains)
        if state['done']:self.stats['probe_plan_completed_skips']+=1
        while True:
            # Retry only notifications whose callback did not finish. Learned
            # stores are idempotent; an interrupted callback is not assumed done.
            for index,result in enumerate(results):
                if result['possible']:continue
                mask=groups[index];removed=domains[cell]&mask;domains[cell]&=~mask
                metric='group_lookahead_prunes' if grouped else 'lookahead_prunes';stats[metric]=stats.get(metric,0)+removed.bit_count()
                if not result['notified']:
                    on_failure(mask if grouped else mask.bit_length()-1);result['notified']=True
            if len(results)==len(groups):break
            check();mask=groups[len(results)];trial=list(before);trial[cell]=mask
            name='group_lookahead_probes' if grouped else 'lookahead_probes';stats[name]=stats.get(name,0)+1
            possible=probe(trial,cell)
            if type(possible)is not bool:raise ValueError('Probe result must be boolean.')
            narrowed=_positive(trial,before,cell,mask) if possible else ()
            results.append({'possible':possible,'domains':narrowed,'notified':possible})
            if not possible:
                name='group_lookahead_failures' if grouped else 'lookahead_failures';stats[name]=stats.get(name,0)+1
            # Completed result commits before cancellation can discard it.
            self._trim(key)
        supports=[0]*len(before)
        for result in results:
            if result['possible']:
                for c,mask in enumerate(result['domains']):supports[c]|=mask
        for c,mask in enumerate(supports):
            removed=domains[c]&~mask;domains[c]&=mask
            metric='group_lookahead_prunes' if grouped else 'lookahead_common_prunes';stats[metric]=stats.get(metric,0)+removed.bit_count()
        if not state['done']:
            name='group_lookahead_complete' if grouped else 'lookahead_complete_passes';stats[name]=stats.get(name,0)+1
        state['done']=True;return tuple(domains)!=was
    def pairs(self,context,domains,cell,choices,singleton_probe,pair_probe,check,on_failure,on_failed_first,stats,*,neighbors=(),max_pairs=16):
        if type(max_pairs)is not int or max_pairs<0:raise ValueError('Invalid pair-probe budget.')
        check();key,state=self._get('pair',context,domains,cell,[1<<v for v in choices],neighbors)
        before=state['before'];was=tuple(domains);results=state['results'];used=0
        for i,mask in enumerate(state['groups']):
            code=mask.bit_length()-1
            if i==len(results):
                check();trial=list(before);trial[cell]=mask;stats['pair_lookahead_singletons']=stats.get('pair_lookahead_singletons',0)+1
                possible=singleton_probe(trial,cell)
                if type(possible)is not bool:raise ValueError('Invalid singleton outcome.')
                trial=_positive(trial,before,cell,mask) if possible else ()
                eligible=[c for c,m in enumerate(trial) if c!=cell and 2<=m.bit_count()<=4]
                other=min(eligible,key=lambda c:(c not in neighbors,trial[c].bit_count(),c)) if eligible else None
                results.append({'possible':possible,'domains':trial,'other':other,'seconds':tuple(v for v in range(trial[other].bit_length()) if trial[other]>>v&1) if other is not None else (),
                                'outcomes':[],'notified':False})
            row=results[i];other=row['other']
            for j,outcome in enumerate(row['outcomes']):
                if not outcome['possible'] and not outcome['notified']:
                    on_failure(cell,code,other,row['seconds'][j]);outcome['notified']=True
            if row['possible'] and other is not None:
                while len(row['outcomes'])<len(row['seconds']):
                    if used>=max_pairs:self._trim(key);return tuple(domains)!=was
                    check();second=row['seconds'][len(row['outcomes'])];trial=list(row['domains']);trial[other]=1<<second
                    used+=1;stats['pair_lookahead_probes']=stats.get('pair_lookahead_probes',0)+1
                    possible=pair_probe(trial,other)
                    if type(possible)is not bool:raise ValueError('Invalid pair outcome.')
                    if possible:_positive(trial,row['domains'],other,1<<second)
                    row['outcomes'].append({'possible':possible,'notified':possible})
                    if not possible:
                        stats['pair_lookahead_failures']=stats.get('pair_lookahead_failures',0)+1
                        on_failure(cell,code,other,second);row['outcomes'][-1]['notified']=True
            failed=not row['possible'] or other is not None and not any(v['possible'] for v in row['outcomes'])
            if failed:
                removed=domains[cell]&mask;domains[cell]&=~mask
                if not row['notified']:
                    on_failed_first(code);row['notified']=True
                    stats['pair_lookahead_parent_prunes']=stats.get('pair_lookahead_parent_prunes',0)+bool(removed)
            self._trim(key)
        if not state['done']:stats['pair_lookahead_complete_passes']=stats.get('pair_lookahead_complete_passes',0)+1
        else:self.stats['probe_plan_completed_skips']+=1
        state['done']=True;return tuple(domains)!=was
    def snapshot(self):return {'version':1,'limit':self.limit,'max_bytes':self.max_bytes,'records':copy.deepcopy(list(self.records.values())),'stats':dict(self.stats)}
    @classmethod
    def restore(cls,payload):
        if type(payload)is not dict or set(payload)!={'version','limit','max_bytes','records','stats'} or type(payload['version'])is not int or payload['version']!=1:raise ValueError('Invalid probe-plan schema.')
        obj=cls(payload['limit'],payload['max_bytes']);records=payload['records']
        if type(records)is not list or len(records)>obj.limit or len(json.dumps(records,separators=(',',':')).encode())>obj.max_bytes:raise ValueError('Probe-plan snapshot exceeds bounds.')
        for raw in records:
            if type(raw)is not dict or set(raw)!={'kind','context','before','cell','groups','neighbors','results','done'} or raw['kind'] not in ('single','group','pair') or type(raw['done'])is not bool:raise ValueError('Invalid probe plan.')
            before=_domains(raw['before']);cell=raw['cell'];groups=tuple(raw['groups']);context=_freeze(raw['context']);neighbors=tuple(raw['neighbors'])
            if any(type(c)is not int or not -1<=c<len(before) for c in neighbors):raise ValueError('Invalid probe neighborhood.')
            if (raw['kind'],context,before,cell,groups,neighbors) in obj.records:raise ValueError('Duplicate probe-plan scope.')
            key,state=obj._get(raw['kind'],context,before,cell,groups,neighbors,cover=False)
            if state['results']:raise ValueError('Duplicate probe-plan scope.')
            if type(raw['results'])is not list or len(raw['results'])>len(groups):raise ValueError('Invalid completed probe count.')
            for i,result in enumerate(raw['results']):
                required={'possible','domains','notified'} if raw['kind']!='pair' else {'possible','domains','notified','other','seconds','outcomes'}
                if type(result)is not dict or set(result)!=required or type(result['possible'])is not bool or type(result['notified'])is not bool:raise ValueError('Invalid completed probe row.')
                trial=_positive(result['domains'],before,cell,groups[i]) if result['possible'] else ()
                if not result['possible'] and result['domains'] not in ([],()):raise ValueError('Negative probe retained narrowing.')
                row={**result,'domains':trial}
                if raw['kind']=='pair':
                    other=result['other'];seconds=tuple(result['seconds'])
                    if other is not None and (type(other)is not int or not 0<=other<len(before) or other==cell or not result['possible']):raise ValueError('Invalid pair second cell.')
                    expected=tuple(v for v in range(trial[other].bit_length()) if trial[other]>>v&1) if other is not None else ()
                    if seconds!=expected or any(type(v)is not int for v in seconds) or type(result['outcomes'])is not list or len(result['outcomes'])>len(seconds):raise ValueError('Invalid pair coverage.')
                    if any(type(v)is not dict or set(v)!={'possible','notified'} or any(type(x)is not bool for x in v.values()) for v in result['outcomes']):raise ValueError('Invalid pair outcome.')
                    row['seconds']=seconds
                    if raw['done'] and len(row['outcomes'])!=len(seconds):raise ValueError('Incomplete pair marked complete.')
                state['results'].append(row)
            if raw['done'] and len(state['results'])!=len(groups):raise ValueError('Incomplete constructive plan marked complete.')
            state['done']=raw['done']
        if type(payload['stats'])is not dict or set(payload['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in payload['stats'].values()):raise ValueError('Invalid probe-plan counters.')
        obj.stats=dict(payload['stats']);return obj
