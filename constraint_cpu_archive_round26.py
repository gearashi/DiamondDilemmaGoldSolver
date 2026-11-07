"""Trusted-local persistence for job-bound CPU clauses, reasons and probe caches.

This module is called only from the local application cache. Its records are
NOT accepted by portable job/proof import and are NOT independently proved
certificates. The existing LCG sound-reason contract remains unchanged.
"""
from collections import OrderedDict
from pathlib import Path
from constraint_reasoning_store_round26 import LocalReasoningStore,encoded
from constraint_lcg import LCGStore,LCGState,LearnedClause,Fact,STAT_KEYS,_LazyParents,_SharedReason
from constraint_lookahead import CompletedProbeCache

MAX_PARENT_LINKS=2_000_000
_SOURCE_IMPORTS={}

class ReasonCacheLimit(ValueError):pass


def _integer(value,low=0,high=None):
    if type(value)is not int or value<low or high is not None and value>high:
        raise ValueError('Malformed local CPU integer.')
    return value


def _tuple(value):
    if isinstance(value,list):return tuple(_tuple(x) for x in value)
    if value is None or type(value)in(str,int,bool,float):return value
    raise ValueError('Unsupported probe context value.')


def _domains(values,root):
    if type(values)is not list or len(values)!=len(root) or any(type(x)is not int or x<0 or x&~r for x,r in zip(values,root)):
        raise ValueError('Local CPU domains escape owned root.')
    return values


def dump_state(state):
    if type(state)is not LCGState:raise ValueError('Expected a local LCG state.')
    facts=state.facts[:max(state.current,default=-1)+1]
    if sum(len(tuple(f.parents)) for f in facts)>MAX_PARENT_LINKS:raise ReasonCacheLimit('Local reason link cap.')
    return {'binding':state.binding,'max_facts':state.max_facts,'domains':state.domains,
        'facts':[[f.cell,f.mask,f.level,list(f.parents),f.rule,f.decision,isinstance(f.parents,_LazyParents)] for f in facts],
        'current':state.current,'level':state.level,'enabled':state.enabled,'failure':state.failure,
        'budget_skips':state.budget_skips}


def restore_state(payload,initial,fingerprint,target):
    keys={'binding','max_facts','domains','facts','current','level','enabled','failure','budget_skips'}
    if type(payload)is not dict or set(payload)!=keys:raise ValueError('Malformed local reason graph.')
    obj=LCGState(initial,fingerprint,target,max_facts=payload['max_facts'])
    if _tuple(payload['binding'])!=obj.binding:raise ValueError('Local reason scope differs.')
    domains=_domains(payload['domains'],initial);rows=payload['facts']
    if type(rows)is not list or not len(initial)<=len(rows)<=obj.max_facts:raise ValueError('Local fact count differs.')
    facts=[];links=0;shared={}
    for index,row in enumerate(rows):
        if type(row)is not list or len(row)!=7:raise ValueError('Malformed local fact.')
        cell,mask,level,parents,rule,decision,lazy=row
        _integer(cell,0,len(initial)-1);_integer(mask);_integer(level)
        if mask&~initial[cell] or type(parents)is not list or type(rule)is not str or len(rule)>256 or type(decision)is not bool or type(lazy)is not bool:
            raise ValueError('Malformed local fact fields.')
        links+=len(parents)
        if links>MAX_PARENT_LINKS or len(set(parents))!=len(parents):raise ValueError('Local reason links exceed bound or repeat.')
        for p in parents:_integer(p,0,index-1)
        if index<len(initial) and row!=[index,initial[index],0,[],'root',False,False]:raise ValueError('Local fact roots differ.')
        inherited=max((facts[p].level for p in parents),default=0)
        if decision:
            if len(parents)!=1 or level<=inherited or facts[parents[0]].cell!=cell or mask&~facts[parents[0]].mask:
                raise ValueError('Malformed local decision dependency.')
        elif level!=inherited:raise ValueError('Malformed local inherited level.')
        refs=tuple(parents)
        if lazy:
            shared.setdefault(refs,_SharedReason(refs,inherited));refs=_LazyParents(shared[refs])
        facts.append(Fact(cell,mask,level,refs,rule,decision))
    current=payload['current']
    if type(current)is not list or len(current)!=len(initial):raise ValueError('Malformed local current facts.')
    for cell,index in enumerate(current):
        _integer(index,0,len(facts)-1)
        if facts[index].cell!=cell:raise ValueError('Current fact has another cell.')
    level=_integer(payload['level']);skips=_integer(payload['budget_skips'])
    if type(payload['enabled'])is not bool or any(f.level>level for f in facts):raise ValueError('Invalid local level or state.')
    if payload['enabled'] and any(facts[index].mask!=domains[cell] for cell,index in enumerate(current)):
        raise ValueError('Live reason facts differ from domains.')
    failure=payload['failure']
    if failure is not None:
        if type(failure)is not list:raise ValueError('Invalid local conflict cells.')
        for c in failure:_integer(c,0,len(initial)-1)
    obj.domains=domains.copy();obj.facts=facts;obj.current=current.copy();obj.level=level
    obj.enabled=payload['enabled'];obj.failure=failure;obj.budget_skips=skips;obj.validate_graph();return obj


class CPUReasoningArchive:
    def __init__(self,directory,fingerprint,target,initial,*,original=None,max_bytes=64*1024*1024,max_entries=128):
        self.initial=tuple(initial);self.fingerprint=fingerprint;self.target=target
        LCGStore(fingerprint,target,self.initial)  # Reuse strict scope validation.
        self.local=LocalReasoningStore(directory,{'fingerprint':fingerprint,'target':target,
            'original':list(original if original is not None else initial),'initial':list(initial)},
            max_bytes=max_bytes,max_entries=max_entries)

    @staticmethod
    def sources(component):
        # Completed stages, probe results and learned clauses depend on their
        # producing rules, including optional imports nested inside functions.
        # Bind transitive local code; a root-file hash alone cannot detect a
        # changed leaf propagator whose completed stage would otherwise skip.
        import ast,hashlib
        root=Path(__file__).parent;pending=['constraint_cpu_archive_round26','constraint_dfs']
        seen=set();paths=set()
        while pending:
            name=pending.pop()
            if name in seen:continue
            seen.add(name);path=root/(name+'.py')
            if not path.is_file():continue
            paths.add(path);raw=path.read_bytes();identity=hashlib.sha256(raw).hexdigest()
            cached=_SOURCE_IMPORTS.get(str(path))
            if cached is None or cached[0]!=identity:
                imports=set()
                for node in ast.walk(ast.parse(raw.decode('utf-8-sig'))):
                    if isinstance(node,ast.Import):imports.update(item.name.split('.')[0] for item in node.names)
                    elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
                cached=(identity,tuple(sorted(imports)));_SOURCE_IMPORTS[str(path)]=cached
            pending.extend(cached[1]);native=root/(name+'.c')
            if native.is_file():paths.add(native)
        return sorted(paths)

    def save_lcg(self,store,check=lambda:None):
        if not store.matches(self.fingerprint,self.target,self.initial):raise ValueError('Local LCG store scope differs.')
        payload={'limit':store.limit,'literal_limit':store.literal_limit,'clauses':store.clauses,
            'stats':store.stats,'utility':[[clause,list(value)] for clause,value in store._utility.items()],
            'learning_tick':store._learning_tick,'use_serial':store._use_serial,
            'prune_snapshot':store._prune_snapshot,'prune_pending':sorted(store._prune_pending)}
        return self.local.save('lcg',(),payload,sources=self.sources('lcg'),check=check)

    def load_lcg(self,check=lambda:None):
        p=self.local.load('lcg',(),sources=self.sources('lcg'),check=check)
        if p is None:return None
        try:
            if type(p)is not dict or set(p)!={'limit','literal_limit','clauses','stats','utility','learning_tick','use_serial','prune_snapshot','prune_pending'}:raise ValueError('Malformed LCG record.')
            obj=LCGStore(self.fingerprint,self.target,self.initial,limit=p['limit'],literal_limit=p['literal_limit'])
            rows=p['clauses']
            if type(rows)is not list or len(rows)>obj.limit or sum(map(len,rows))>obj.literal_limit:raise ValueError('Local clauses exceed bound.')
            for row in rows:
                row=_tuple(row)
                if tuple(sorted(row))!=row or len({cell for cell,_ in row})!=len(row):raise ValueError('Noncanonical local clause.')
                if not obj.learn(LearnedClause(obj.binding,row,None,0,0,0)):raise ValueError('Redundant local clause.')
            if len(obj.clauses)!=len(rows):raise ValueError('Subsumed local clause.')
            stats=p['stats']
            if type(stats)is not dict or set(stats)!=set(STAT_KEYS):raise ValueError('Local counter schema differs.')
            for value in stats.values():_integer(value)
            obj.stats=stats.copy();obj._learning_tick=_integer(p['learning_tick']);obj._use_serial=_integer(p['use_serial'])
            utility={}
            for row,value in p['utility']:
                clause=_tuple(row)
                if clause not in obj.clauses or clause in utility or len(value)!=3:raise ValueError('Local utility clause differs.')
                score,tick,recent=value;_integer(score,0,65535);_integer(tick,0,obj._learning_tick);_integer(recent,0,obj._use_serial)
                utility[clause]=tuple(value)
            obj._utility=utility
            if p['prune_snapshot'] is not None:obj._prune_snapshot=tuple(_domains(p['prune_snapshot'],self.initial))
            if type(p['prune_pending'])is not list:raise ValueError('Malformed pending local clauses.')
            obj._prune_pending=set(_integer(i,0,len(obj.clauses)-1) for i in p['prune_pending'])
            return obj
        except (ValueError,TypeError,KeyError,IndexError):self.local.stats['reasoning_cache_invalid']+=1;return None

    def save_probes(self,cache,check=lambda:None):
        if type(cache)is not CompletedProbeCache:raise ValueError('Expected completed probe cache.')
        payload={'limit':cache.limit,'max_slots':cache.max_slots,
                 'entries':[[scope,source,possible,narrowed] for (scope,source),(possible,narrowed,_) in cache.entries.items()],
                 'plans':cache.plans.snapshot(),'partial':[[scope,source,narrowed] for (scope,source),narrowed in cache.partial.items()]}
        return self.local.save('probes',(),payload,sources=self.sources('probes'),check=check)

    def load_probes(self,check=lambda:None):
        p=self.local.load('probes',(),sources=self.sources('probes'),check=check)
        if p is None:return None
        try:
            if type(p)is not dict or not {'limit','max_slots','entries'}<=set(p)<={'limit','max_slots','entries','plans','partial'}:raise ValueError('Malformed probe record.')
            obj=CompletedProbeCache(p['limit'],p['max_slots'])
            if type(p['entries'])is not list or len(p['entries'])>obj.limit:raise ValueError('Probe cache exceeds count.')
            for row in p['entries']:
                if type(row)is not list or len(row)!=4:raise ValueError('Malformed completed probe.')
                scope,source,possible,narrowed=row;scope=_tuple(scope);source=tuple(_domains(source,self.initial))
                if type(possible)is not bool or not all(source):raise ValueError('Malformed probe outcome.')
                if possible:
                    narrowed=tuple(_domains(narrowed,source))
                    if not all(narrowed):raise ValueError('Completed positive probe is empty.')
                elif narrowed!=[]:raise ValueError('Negative probe retained unsupported narrowing.')
                else:narrowed=()
                key=(scope,source)
                if key in obj.entries:raise ValueError('Duplicate completed probe.')
                slots=len(source)+len(narrowed);obj.slots+=slots
                if obj.slots>obj.max_slots:raise ValueError('Probe cache exceeds slots.')
                obj.entries[key]=(possible,narrowed,slots)
            if 'plans' in p:
                from constraint_probe_plans_round26 import ProbePlans
                obj.plans=ProbePlans.restore(p['plans'])
                for record in obj.plans.records.values():_domains(list(record['before']),self.initial)
            if 'partial' in p:
                if type(p['partial'])is not list or len(p['partial'])>min(obj.limit,16):raise ValueError('Partial-probe count exceeds cap.')
                for scope,source,narrowed in p['partial']:
                    source=tuple(_domains(source,self.initial));narrowed=tuple(_domains(narrowed,source));key=(_tuple(scope),source)
                    if key in obj.partial:raise ValueError('Duplicate partial probe.')
                    obj.partial[key]=narrowed
                if sum(len(row) for row in obj.partial.values())>obj.max_slots:raise ValueError('Partial-probe slots exceed cap.')
            return obj
        except (ValueError,TypeError,KeyError,IndexError):self.local.stats['reasoning_cache_invalid']+=1;return None

    def save_reason(self,key,state,check=lambda:None):
        if state.binding!=(self.fingerprint,self.target,self.initial):raise ValueError('Reason graph scope differs.')
        try:payload=dump_state(state)
        except ReasonCacheLimit:return False
        return self.local.save('reason',key,payload,sources=self.sources('reason'),check=check)

    def load_reason(self,key,check=lambda:None):
        p=self.local.load('reason',key,sources=self.sources('reason'),check=check)
        if p is None:return None
        try:return restore_state(p,self.initial,self.fingerprint,self.target)
        except (ValueError,TypeError,KeyError,IndexError):self.local.stats['reasoning_cache_invalid']+=1;return None

    def save_propagation(self,key,cursor,check=lambda:None):
        from constraint_propagation_cursor_round26 import PropagationCursor
        if type(cursor)is not PropagationCursor:raise ValueError('Expected local propagation cursor.')
        _domains(list(cursor.basic.initial),self.initial)
        return self.local.save('propagation',key,cursor.snapshot(),sources=self.sources('propagation'),check=check)

    def load_propagation(self,key,arcs,incoming,tile_bits,stage_names,check=lambda:None):
        from constraint_propagation_cursor_round26 import PropagationCursor
        p=self.local.load('propagation',key,sources=self.sources('propagation'),check=check)
        if p is None:return None
        try:
            cursor=PropagationCursor.restore(p,arcs,incoming,tile_bits,stage_names)
            _domains(list(cursor.basic.initial),self.initial);return cursor
        except (ValueError,TypeError,KeyError,IndexError):
            self.local.stats['reasoning_cache_invalid']+=1;return None
