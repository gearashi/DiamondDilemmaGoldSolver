"""Durable recursive separator conditioning and exact correlated messages (F10).

Tasks are keyed by complete interior domains; outside constraints are absent
from their transition semantics. Parent joins reintroduce every seam, physical
resource and Gold path dependency. Descendants are decomposed again after each
conditioning. No component is assumed independent merely because geometry split.
"""
from collections import OrderedDict
from itertools import combinations,product
from pathlib import Path
import json
from fractions import Fraction
from constraint_region_compiler_round26 import _domains,_digest,_source,_json_size,MAX_BYTES
from constraint_region_incremental import source_binding
from constraint_region_interfaces_round26 import ParametricEnvelope,signature_messages
from constraint_region_certificate import _Envelope
from constraint_region_network import assignment_signature,joined_signature
from constraint_pruning import values
from constraint_separator import _Budget
from validator import reverse_mask
from constraint_region_messages_round26 import SymbolicMessageJoin,STAT_KEYS as SYMBOLIC_KEYS

STAT_KEYS=('recursive_region_queries','recursive_region_completed','recursive_region_unknown',
 'recursive_region_tasks','recursive_region_task_reuses','recursive_region_conditionings',
 'recursive_region_redecompositions','recursive_region_leaf_transitions','recursive_region_leaf_states',
 'recursive_region_message_edges','recursive_region_join_attempts','recursive_region_join_merges',
 'recursive_region_outward_links','recursive_region_prunes','recursive_region_saved','recursive_region_loaded',
 'recursive_region_io_failures','recursive_region_cross_query_hits')+SYMBOLIC_KEYS


def _pack(value):
    if value is None or type(value) in (int,str,bool):return value
    if isinstance(value,Fraction):return {'f':[value.numerator,value.denominator]}
    if type(value)is tuple:return {'t':[_pack(x) for x in value]}
    if type(value)is list:return {'l':[_pack(x) for x in value]}
    if type(value)is dict:return {'d':[[_pack(k),_pack(v)] for k,v in value.items()]}
    raise ValueError('Unsupported tree continuation value.')

def _unpack(value,depth=0):
    if depth>50:raise ValueError('Tree continuation nesting cap.')
    if value is None or type(value) in (int,str,bool):return value
    if type(value)is not dict or len(value)!=1:raise ValueError('Malformed tree continuation.')
    kind,rows=next(iter(value.items()))
    if kind not in ('t','l','d','f') or type(rows)is not list or len(rows)>200000:raise ValueError('Malformed tree sequence.')
    if kind=='f':
        if len(rows)!=2 or any(type(v)is not int for v in rows) or rows[1]<=0:raise ValueError('Invalid exact rational continuation.')
        return Fraction(*rows)
    if kind=='d':
        result={}
        for row in rows:
            if type(row)is not list or len(row)!=2:raise ValueError('Malformed tree mapping.')
            key,val=(_unpack(x,depth+1) for x in row)
            try:
                if key in result:raise ValueError('Duplicate tree key.')
                result[key]=val
            except TypeError as error:raise ValueError('Unhashable tree key.') from error
        return result
    result=[_unpack(x,depth+1) for x in rows]
    return tuple(result) if kind=='t' else result


def _components(owner,cells,removed=()):
    pending=set(cells)-set(removed);parts=[]
    while pending:
        start=min(pending);pending.remove(start);todo=[start];part=[]
        while todo:
            cell=todo.pop();part.append(cell)
            for nxt in sorted(owner.neighbors[cell]&pending):pending.remove(nxt);todo.append(nxt)
        parts.append(tuple(sorted(part)))
    return tuple(sorted(parts))


def _plan(owner,cells,domains,leaf_size,check):
    if len(cells)<=leaf_size:return None
    parts=_components(owner,cells)
    if len(parts)>1:return ((),parts)
    best=None
    # Candidate cutsets include all singletons and adjacent pairs; fallback
    # balanced message trees remain exact when no inexpensive separator exists.
    candidates=[(c,) for c in cells]
    candidates.extend((a,b) for a,b in combinations(cells,2) if b in owner.neighbors[a])
    for cut in candidates:
        check();count=1
        for c in cut:count*=domains[c].bit_count()
        if count>64:continue
        parts=_components(owner,cells,cut)
        if len(parts)<2:continue
        score=(max(map(len,parts)),count,len(cut),cut)
        if best is None or score<best[0]:best=(score,cut,parts)
    if best is not None:return best[1:]
    ordered=[];pending=set(cells);todo=[min(cells)]
    while pending:
        if not todo:todo.append(min(pending))
        cell=todo.pop(0)
        if cell not in pending:continue
        pending.remove(cell);ordered.append(cell);todo.extend(sorted(owner.neighbors[cell]&pending))
    middle=len(ordered)//2
    return (), (tuple(sorted(ordered[:middle])),tuple(sorted(ordered[middle:])))


class RecursiveRegionSession:
    def __init__(self,owner,domains,*,leaf_size=4,max_tasks=2048,max_states=20000,max_steps=200000,resume=None,completed=None):
        self.owner=owner;self.envelope=_Envelope(owner);self.problem=owner.problem
        self.domains=_domains(self.problem,domains);self.semantic=source_binding(owner)
        if any(type(v)is not int or v<1 for v in (leaf_size,max_tasks,max_states,max_steps)):raise ValueError('Invalid recursive regional policy.')
        self.policy=(leaf_size,max_tasks,max_states,max_steps)
        source=Path(__file__).read_bytes()
        self.binding=_digest((self.semantic,self.domains,self.policy,_source(),source.hex()))
        self.original=tuple(sum(1<<v for v in set(row)) for row in self.problem.domains)
        self.stats=dict.fromkeys(STAT_KEYS,0);self.tasks=[];self.index={};self.leaves={};self.joiners={}
        self.complete=False;self.blocked=False;self.total_steps=0
        self.shared={} if completed is None else completed
        self.root=self._task(tuple(range(self.problem.n)),self.domains)
        if resume is not None:self._restore(resume)
    def _task(self,cells,domains):
        key=(tuple(cells),tuple(domains[c] for c in cells))
        if key in self.index:self.stats['recursive_region_task_reuses']+=1;return self.index[key]
        if len(self.tasks)>=self.policy[1]:raise _Budget
        ident=len(self.tasks);self.index[key]=ident
        self.tasks.append({'cells':tuple(cells),'domains':tuple(domains[c] for c in cells),'plan':None,
          'planned':False,'done':False,'messages':{},'condition':0,'child':0,'current':None,
          'members':(),'join':None,'links':[]})
        self.stats['recursive_region_tasks']+=1
        if key in self.shared:
            # Only this live stage owns these exact, completed envelope messages.
            self.tasks[ident]['messages']=self.shared[key];self.tasks[ident]['done']=True
            self.tasks[ident]['planned']=True;self.stats['recursive_region_cross_query_hits']+=1
        return ident
    def _full_domains(self,task):
        domains=list(self.original)
        for c,d in zip(task['cells'],task['domains']):domains[c]=d
        return domains
    def _prepare(self,ident,check):
        task=self.tasks[ident];domains=self._full_domains(task);cells=task['cells']
        plan=_plan(self.owner,cells,domains,self.policy[0],check)
        if plan is None:task['planned']=True;task['plan']=None;return
        cut,parts=plan;conditions=[]
        for codes in product(*(tuple(values(domains[c])) for c in cut)) if cut else [()]:
            check();signature=assignment_signature(self.envelope,cut,codes,domains,check) if cut else None
            if cut and signature is None:continue
            narrowed=list(domains);used={v//3 for v in codes};fixed=dict(zip(cut,codes))
            for cell in cells:
                if cell in fixed:narrowed[cell]=1<<fixed[cell];continue
                kept=[]
                for code in values(domains[cell]):
                    if code//3 in used:continue
                    if any(other in fixed and self.owner.masks[code][side]!=reverse_mask(self.owner.masks[fixed[other]][other_side])
                           for other,side,other_side,_ in self.owner.adj[cell]):continue
                    kept.append(code)
                narrowed[cell]=sum(1<<code for code in kept)
            if any(not narrowed[c] for c in cells):continue
            children=[self._task(part,narrowed) for part in parts]
            entry=None if not cut else {signature:{'count':1,'supports':{c:1<<v for c,v in fixed.items()},'witness':fixed}}
            conditions.append((tuple(cut),entry,tuple(children)))
        check();task['plan']=conditions;task['planned']=True
        self.stats['recursive_region_redecompositions']+=1
        self.stats['recursive_region_conditionings']+=len(conditions) if cut else 0
    def _leaf(self,ident,check):
        task=self.tasks[ident];domains=self._full_domains(task)
        if ident not in self.leaves:self.leaves[ident]=ParametricEnvelope(self.envelope,task['cells'],domains,max_states=self.policy[2],max_steps=self.policy[3])
        leaf=self.leaves[ident];before=(leaf.session.steps,leaf.session.states)
        try:done=leaf.advance(1,check)
        finally:
            self.stats['recursive_region_leaf_transitions']+=leaf.session.steps-before[0]
            self.stats['recursive_region_leaf_states']+=leaf.session.states-before[1]
        if done:
            diagram=leaf.view(domains,check,fixed_chains=False)
            messages,work=signature_messages(diagram,check,max_entries=self.policy[3])
            check();task['messages']=messages;task['done']=True
            self.stats['recursive_region_message_edges']+=work
    @staticmethod
    def _accumulate(target,source):
        for sig,entry in source.items():
            if sig not in target:target[sig]={'count':0,'supports':{},'witness':dict(entry['witness'])}
            old=target[sig];old['count']+=entry['count']
            for c,d in entry['supports'].items():old['supports'][c]=old['supports'].get(c,0)|d
    def _step(self,ident,check):
        task=self.tasks[ident]
        if task['done']:return
        if not task['planned']:self._prepare(ident,check);return
        if task['plan'] is None:self._leaf(ident,check);return
        if task['condition']==len(task['plan']):task['done']=True;return
        cut,initial,children=task['plan'][task['condition']]
        if task['current'] is None:
            task['current']={} if initial is None else {s:{'count':e['count'],'supports':dict(e['supports']),'witness':dict(e['witness'])} for s,e in initial.items()}
            task['members']=cut
        if task['child']==len(children):
            self._accumulate(task['messages'],task['current']);task['condition']+=1;task['child']=0;task['current']=None;task['members']=();return
        child_id=children[task['child']];child=self.tasks[child_id]
        if not child['done']:self._step(child_id,check);return
        if not task['members']:
            task['current']={s:{'count':e['count'],'supports':dict(e['supports']),'witness':dict(e['witness'])} for s,e in child['messages'].items()}
            task['members']=child['cells'];task['child']+=1;return
        if task['join'] is None:
            task['join']={'symbolic':True}
            self.joiners[ident]=SymbolicMessageJoin(self.envelope,task['members'],task['current'],child['cells'],child['messages'],self._full_domains(task),max_nodes=self.policy[2])
        joiner=self.joiners[ident];before=dict(joiner.stats)
        try:done=joiner.advance(check)
        finally:
            for name,value in joiner.stats.items():self.stats[name]+=value-before[name]
            self.stats['recursive_region_join_attempts']+=joiner.stats['recursive_region_symbolic_geometry_tests']-before['recursive_region_symbolic_geometry_tests']
        if done:
            task['current']=joiner.result
            task['links'].extend((merged,ls,child_id,rs) for merged,ls,rs in joiner.links)
            self.stats['recursive_region_outward_links']+=len(joiner.links)
            self.stats['recursive_region_join_merges']+=max(0,len(joiner.links)-len(joiner.result))
            task['members']=tuple(sorted((*task['members'],*child['cells'])))
            task['child']+=1;task['join']=None;del self.joiners[ident]
    def advance(self,lease=1024,check=lambda:None):
        if type(lease)is not int or lease<1:raise ValueError('Invalid recursive regional lease.')
        if source_binding(self.owner)!=self.semantic:raise ValueError('Recursive region semantic source changed.')
        for _ in range(lease):
            check()
            if self.complete or self.blocked:return self.complete
            if self.total_steps>=self.policy[3]:self.blocked=True;return False
            try:self._step(self.root,check)
            except _Budget:self.blocked=True;return False
            self.total_steps+=1;self.complete=self.tasks[self.root]['done']
        return self.complete
    def result(self,check=lambda:None):
        if not self.complete:return None
        table=self.tasks[self.root]['messages'];supports=[0]*self.problem.n;rows=0;witness=None
        for signature,entry in table.items():
            check();rows+=entry['count']
            for cell,mask in entry['supports'].items():supports[cell]|=mask
            if witness is None:witness=tuple(entry['witness'][c] for c in range(self.problem.n))
        if witness is not None and self.owner.target=='gold' and not self.problem.validate(witness).get('valid'):raise RuntimeError('Recursive message Gold witness invalid.')
        return {'complete':True,'rows':rows,'supports':tuple(supports),'witness':witness}
    def snapshot(self):
        body={'schema':1,'binding':self.binding,'tasks':self.tasks,'stats':self.stats,'steps':self.total_steps,
              'complete':self.complete,'blocked':self.blocked,'leaves':{i:leaf.snapshot() for i,leaf in self.leaves.items()},'joiners':{i:joiner.snapshot() for i,joiner in self.joiners.items()}}
        packed=_pack(body);record={'body':packed,'sha256':_digest(packed)}
        if _json_size(record)>MAX_BYTES:raise _Budget
        return record
    def _restore(self,record):
        if type(record)is not dict or set(record)!={'body','sha256'} or record['sha256']!=_digest(record['body']) or _json_size(record)>MAX_BYTES:raise ValueError('Invalid recursive local checkpoint.')
        body=_unpack(record['body'])
        if type(body)is not dict or set(body)!={'schema','binding','tasks','stats','steps','complete','blocked','leaves','joiners'} or body['schema']!=1 or body['binding']!=self.binding:raise ValueError('Recursive checkpoint source/scope differs.')
        tasks=body['tasks']
        if type(tasks)is not list or not 1<=len(tasks)<=self.policy[1] or type(body['steps'])is not int or not 0<=body['steps']<=self.policy[3]:raise ValueError('Invalid recursive task limits.')
        if any(type(body[k])is not bool for k in ('complete','blocked')) or body['complete'] and body['blocked']:raise ValueError('Invalid recursive status.')
        if set(body['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in body['stats'].values()):raise ValueError('Invalid recursive counters.')
        index={}
        for i,task in enumerate(tasks):
            if set(task)!=set(self.tasks[0]):raise ValueError('Invalid recursive task schema.')
            cells=task['cells'];domains=task['domains']
            if type(cells)is not tuple or not cells or cells!=tuple(sorted(set(cells))) or any(type(c)is not int or not 0<=c<self.problem.n for c in cells) or len(domains)!=len(cells):raise ValueError('Invalid recursive cells.')
            if any(type(d)is not int or d<0 or d&~self.original[c] for c,d in zip(cells,domains)):raise ValueError('Invalid recursive task domain.')
            if any(type(task[k])is not bool for k in ('done','planned')) or any(type(task[k])is not int or task[k]<0 for k in ('condition','child')):raise ValueError('Invalid recursive cursor.')
            if task['done'] and not task['planned']:raise ValueError('Completed recursive task is unplanned.')
            if task['plan'] is not None:
                if type(task['plan'])is not list or task['condition']>len(task['plan']):raise ValueError('Invalid conditional cursor.')
                if task['condition']==len(task['plan']):
                    if task['child']!=0 or task['current'] is not None or task['join'] is not None:raise ValueError('Completed conditions retain a pending child.')
                elif task['child']>len(task['plan'][task['condition']][2]):raise ValueError('Conditional child index is out of range.')
                if task['done'] and task['condition']!=len(task['plan']):raise ValueError('Completed task skipped a condition.')
            key=(cells,domains)
            if key in index:raise ValueError('Duplicate recursive task.')
            index[key]=i
            if task['plan'] is not None:
                for cut,initial,children in task['plan']:
                    for child in children:
                        if type(child)is not int or not 0<=child<len(tasks) or len(tasks[child]['cells'])>=len(cells):raise ValueError('Invalid recursive dependency DAG.')
            for signature,entry in task['messages'].items():
                if len(signature)!=4 or type(entry['count'])is not int or entry['count']<1 or set(entry['supports'])!=set(cells) or set(entry['witness'])!=set(cells):raise ValueError('Invalid recursive message.')
                if any(type(v)is not int or not domains[cells.index(c)]>>v&1 for c,v in entry['witness'].items()):raise ValueError('Invalid recursive witness scope.')
        if body['complete']!=tasks[0]['done']:raise ValueError('Recursive completion flag differs from owned root.')
        if tasks[0]['cells']!=tuple(range(self.problem.n)) or tasks[0]['domains']!=self.domains:raise ValueError('Recursive root scope differs.')
        leaves={}
        for i,snapshot in body['leaves'].items():
            if type(i)is not int or not 0<=i<len(tasks) or tasks[i]['plan'] is not None:raise ValueError('Invalid recursive leaf.')
            task=tasks[i];full=list(self.original)
            for c,d in zip(task['cells'],task['domains']):full[c]=d
            leaves[i]=ParametricEnvelope(self.envelope,task['cells'],full,max_states=self.policy[2],max_steps=self.policy[3],resume=snapshot)
            if task['done'] and not leaves[i].session.complete:raise ValueError('Completed leaf has an unfinished compiler.')
        joiners={}
        for i,snapshot in body['joiners'].items():
            if type(i)is not int or not 0<=i<len(tasks):raise ValueError('Invalid symbolic tree join owner.')
            task=tasks[i]
            if task['join']!={'symbolic':True} or task['plan'] is None or task['condition']>=len(task['plan']):raise ValueError('Symbolic tree join has no pending condition.')
            children=task['plan'][task['condition']][2]
            if task['child']>=len(children):raise ValueError('Symbolic tree join has no right child.')
            child=tasks[children[task['child']]];full=list(self.original)
            for c,d in zip(task['cells'],task['domains']):full[c]=d
            joiners[i]=SymbolicMessageJoin(self.envelope,task['members'],task['current'],child['cells'],child['messages'],full,max_nodes=self.policy[2],state=snapshot)
        if {i for i,task in enumerate(tasks) if task['join'] is not None}!=set(joiners):raise ValueError('Missing durable symbolic join.')
        self.joiners=joiners
        self.tasks=tasks;self.index=index;self.leaves=leaves;self.stats=body['stats'];self.total_steps=body['steps'];self.complete=body['complete'];self.blocked=body['blocked']


class RecursiveRegions:
    def __init__(self,owner,limit=4):
        self.owner=owner;self.semantic=source_binding(owner);self.limit=limit;self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0);self.last={};self.solution=None;self.directory=None;self.completed_messages=OrderedDict()
    def configure_cache_dir(self,path):self.directory=None if path is None else Path(path)/'recursive-regions-v1'
    def prune(self,domains,check=lambda:None,*,lease=2048,**policy):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Recursive message cache source changed.')
        _domains(self.owner.problem,domains)
        self.solution=None;key=(tuple(domains),tuple(sorted(policy.items())));self.stats['recursive_region_queries']+=1
        if key not in self.cache:
            session=RecursiveRegionSession(self.owner,domains,completed=self.completed_messages,**policy)
            if self.directory is not None:
                path=self.directory/(session.binding+'.json')
                try:
                    if path.is_file() and path.stat().st_size<=MAX_BYTES:
                        session._restore(json.loads(path.read_text(encoding='utf-8')));self.last[key]=dict(session.stats);self.stats['recursive_region_loaded']+=1
                except (OSError,ValueError,RecursionError,OverflowError,TypeError,_Budget):self.stats['recursive_region_io_failures']+=1
            self.cache[key]=session
        session=self.cache[key];self.cache.move_to_end(key)
        try:
            was=session.complete;session.advance(lease,check);result=session.result(check)
            if result is None:self.stats['recursive_region_unknown']+=1;return False
            self.stats['recursive_region_completed']+=int(not was)
            updated=[a&b for a,b in zip(domains,result['supports'])];check()
            self.stats['recursive_region_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,updated))
            changed=list(domains)!=updated;domains[:]=updated;self.solution=result['witness'];return changed
        finally:
            for task in session.tasks:
                if task['done']:
                    message_key=(task['cells'],task['domains']);self.completed_messages[message_key]=task['messages'];self.completed_messages.move_to_end(message_key)
            while len(self.completed_messages)>64:self.completed_messages.popitem(last=False)
            old=self.last.get(key,{})
            for name,value in session.stats.items():self.stats[name]+=max(0,value-old.get(name,0))
            self.last[key]=dict(session.stats)
            if self.directory is not None:
                try:
                    from io_utils import atomic_json
                    self.directory.mkdir(parents=True,exist_ok=True);atomic_json(self.directory/(session.binding+'.json'),session.snapshot());self.stats['recursive_region_saved']+=1
                except (OSError,ValueError,RecursionError,OverflowError,TypeError,_Budget):self.stats['recursive_region_io_failures']+=1
            while len(self.cache)>self.limit:
                old,_=self.cache.popitem(last=False);self.last.pop(old,None)
