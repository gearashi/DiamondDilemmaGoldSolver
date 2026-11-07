"""Reusable ordered whole-constraint Gold DAGs (F09).

Only completed exact diagrams expose counts/supports. Rank reduction is a
separate existential mode. Checkpoints are bounded trusted-local continuations,
not portable mathematical certificates. A Stop inside a transition can retry
that transition; completed transitions are never intentionally replayed.
"""
from collections import OrderedDict, defaultdict
from hashlib import sha256
from pathlib import Path
import json,os
from constraint_pruning import values
from constraint_region_compact import CompactFactors, GoldCompactFactors, CompactLimit
from constraint_region_incremental import source_binding
from constraint_region_rank import cut_representatives
from constraint_separator import _Budget

STAT_KEYS = ('regional_order_candidates', 'regional_order_cost_evaluations',
 'regional_compiler_queries', 'regional_compiler_completed', 'regional_compiler_unknown',
 'regional_compiler_cache_hits', 'regional_compiler_resumes', 'regional_compiler_transition_attempts',
 'regional_compiler_transitions', 'regional_compiler_states', 'regional_compiler_merges',
 'regional_compiler_rank_rows', 'regional_compiler_rank_kept', 'regional_compiler_rank_entries',
 'regional_compiler_rank_xors', 'regional_compiler_backward_edges', 'regional_compiler_prunes',
 'regional_compiler_saved', 'regional_compiler_loaded', 'regional_compiler_io_failures',
 'regional_compiler_native_arcs', 'regional_compiler_native_constraints',
 'regional_compiler_model_builds','regional_compiler_model_placements','regional_compiler_model_seam_tests',
 'regional_compiler_model_forget_checks','regional_compiler_model_pair_splices')
MAX_BYTES=16*1024*1024

def _json_size(value):
    return len(json.dumps(value,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8'))

def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

_SOURCE_IMPORTS={}
_SOURCE_SEEDS=('constraint_region_compiler_round26','constraint_region_interfaces_round26',
 'constraint_region_tree_round26','constraint_region_messages_round26','constraint_region_proof_round26',
 'constraint_region_integer_round26','constraint_region_branch_price_round26','constraint_connectivity_round26',
 'constraint_regions_round26','constraint_region_models','search_problem')

def _source():
    """Bind the transitive local algorithms, including nested optional imports.

    Import discovery is cached by content hash, never by modification time.
    Adjacent C implementations are included when a local wrapper owns one.
    This is conservative binding, not mathematical certificate authority.
    """
    import ast
    root=Path(__file__).parent;pending=list(_SOURCE_SEEDS);seen=set();result={}
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=root/(name+'.py')
        if not path.is_file():continue
        raw=path.read_bytes();digest=sha256(raw).hexdigest();result[path.name]=digest
        cached=_SOURCE_IMPORTS.get(name)
        if cached is None or cached[0]!=digest:
            imports=set()
            for node in ast.walk(ast.parse(raw.decode('utf-8-sig'))):
                if isinstance(node,ast.Import):imports.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
            cached=(digest,tuple(sorted(imports)));_SOURCE_IMPORTS[name]=cached
        pending.extend(cached[1])
        native=root/(name+'.c')
        if native.is_file():result[native.name]=sha256(native.read_bytes()).hexdigest()
    return tuple(sorted(result.items()))

def _tuple(value,depth=0):
    if depth>6:raise ValueError('Regional state nesting cap.')
    if type(value) is list:return tuple(_tuple(v,depth+1) for v in value)
    if type(value) is not int:raise ValueError('Invalid compiler state scalar.')
    return value

def _domains(problem, domains):
    masks=tuple(domains)
    if len(masks)!=problem.n or any(type(d)is not int or d<0 or
        d & ~sum(1<<v for v in set(problem.domains[c])) for c,d in enumerate(masks)):
        raise ValueError('Regional compiler domain scope is invalid.')
    return masks

def choose_order(problem, domains, check=lambda:None, stats=None):
    """Greedy frontier cost considers resources, seam alternatives and ports.

    Compare both orientations with natural order using the same maximum/cumulative
    score, so a heuristic never silently worsens that structural estimate.
    This estimate is not a promised runtime or state-count bound.
    """
    domains=_domains(problem,domains);stats={} if stats is None else stats
    tile_sets=[{v//3 for v in values(d)} for d in domains]
    incident=[set() for _ in domains]
    edge_cost=[]
    for eid,(a,sa,b,sb) in enumerate(problem.edges):
        check();incident[a].add(eid);incident[b].add(eid)
        masks={int(problem.masks[v,sa]) for v in values(domains[a])}
        ports=max((m.bit_count() for m in masks),default=0)
        edge_cost.append((max(1,len(masks)).bit_length(),ports))
    def walk(order=None):
        remaining=[0]*problem.n;seen=[False]*problem.n
        for choices in tile_sets:
            for tile in choices:remaining[tile]+=1
        pending=set(range(problem.n));chosen=set();live=seam=ports=0;answer=[];costs=[]
        while pending:
            def step_cost(cell):
                check();stats['regional_order_cost_evaluations']=stats.get('regional_order_cost_evaluations',0)+1
                resources=live
                for tile in tile_sets[cell]:
                    resources+=int(not seen[tile] and remaining[tile]>1)-int(seen[tile] and remaining[tile]==1)
                next_seam=seam;next_ports=ports
                for eid in incident[cell]:
                    a,sa,b,sb=problem.edges[eid]
                    if a==b:continue
                    other=b if a==cell else a;sign=-1 if other in chosen else 1
                    next_seam+=sign*edge_cost[eid][0];next_ports+=sign*edge_cost[eid][1]
                return resources+next_seam+next_ports,next_ports,next_seam,resources
            cell=min(pending,key=lambda c:(step_cost(c),domains[c].bit_count(),c)) if order is None else order[len(answer)]
            score=step_cost(cell);live=score[3];seam=score[2];ports=score[1]
            for tile in tile_sets[cell]:seen[tile]=True;remaining[tile]-=1
            chosen.add(cell);pending.remove(cell);answer.append(cell);costs.append(score)
        return tuple(answer),(max(costs,default=(0,0,0,0)),sum(c[0] for c in costs))
    greedy,_=walk()
    candidates=tuple(dict.fromkeys((tuple(range(problem.n)),greedy,tuple(reversed(greedy)))))
    stats['regional_order_candidates']=stats.get('regional_order_candidates',0)+len(candidates)
    return min(candidates,key=lambda order:(walk(order)[1],order))

class RegionalCompilerSession:
    """Resumable, bounded exact DAG construction, optional existential row basis."""
    def __init__(self, owner, domains, *, order=None, existence_only=False,
                 max_states=20000,max_steps=200000,max_ports=128,resume=None):
        self.owner=owner;self.problem=owner.problem;self.domains=_domains(self.problem,domains)
        if type(existence_only)is not bool or any(type(x)is not int or x<0 for x in
                (max_states,max_steps,max_ports)):raise ValueError('Invalid regional compiler policy.')
        self.stats=dict.fromkeys(STAT_KEYS,0)
        order=choose_order(self.problem,self.domains,stats=self.stats) if order is None else tuple(order)
        self.policy=(existence_only,max_states,max_steps,max_ports)
        self.binding=_digest((source_binding(owner),_source(),self.domains,order,self.policy))
        self.semantic=source_binding(owner)
        factory=GoldCompactFactors if owner.target=='gold' else CompactFactors
        self.model=factory(self.problem,self.domains,order=order,max_placements=100000,
            max_frontier=len(self.problem.edges),max_resources=self.problem.n,
            **({'max_ports':max_ports} if owner.target=='gold' else {}))
        self.stats['regional_compiler_model_builds']=1
        self.stats['regional_compiler_model_placements']=self.model.stats['compact_factor_placements']
        self.order=order;self.existence_only=existence_only
        self.level=0;self.index=0;self.choice=0;self.steps=0;self.created=1
        self.layer=[(0,self.model.root_state)];self.following={};self.nodes={};self.terminals=()
        self.complete=False;self.blocked=False;self._result=None
        self.stats['regional_compiler_states']=1
        if resume is not None:self._restore(resume)

    def _assert_scope(self):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Regional compiler source changed.')

    def advance(self, transitions=512, check=lambda:None):
        if type(transitions)is not int or transitions<1:raise ValueError('Invalid compiler lease.')
        self._assert_scope();check();spent=0
        if self.complete or self.blocked:return self.complete
        if self.model.missing:self.layer=[];self.complete=True;return True
        while not self.complete and not self.blocked:
            check()
            if self.index==len(self.layer):
                following=self.following
                if self.existence_only and self.owner.target=='gold' and self.level<self.problem.n-1:
                    groups=defaultdict(list)
                    for state,ident in following.items():
                        endpoints=tuple(sorted(v for pair in state[2] for v in pair))
                        groups[(state[:2],endpoints)].append((state,ident))
                    retained={}
                    for (_,endpoints),group in groups.items():
                        check();self.stats['regional_compiler_rank_rows']+=len(group)
                        if len(group)>1 and 0<len(endpoints)<=16:
                            selected,work=cut_representatives([state[2] for state,ident in group],check,max_rows=max(1,len(group)))
                            self.stats['regional_compiler_rank_entries']+=work['cut_entries']
                            self.stats['regional_compiler_rank_xors']+=work['xors']
                            keep=set(selected);group=[(state,ident) for state,ident in group if state[2] in keep]
                        self.stats['regional_compiler_rank_kept']+=len(group)
                        retained.update(group)
                    following=retained
                check();self.layer=[(ident,state) for state,ident in following.items()]
                self.following={};self.level+=1;self.index=0;self.choice=0
                if not self.layer or self.level==self.problem.n:
                    self.complete=True;self.terminals=tuple(ident for ident,state in self.layer)
                    return True
                continue
            codes=self.model.choices[self.level]
            if self.choice==len(codes):self.index+=1;self.choice=0;continue
            if self.steps>=self.policy[2] or self.created>self.policy[1]:self.blocked=True;return False
            if spent==transitions:return False
            ident,state=self.layer[self.index];code=codes[self.choice]
            self.stats['regional_compiler_transition_attempts']+=1
            before=dict(self.model.stats)
            try:child=self.model.transition(self.level,state,code,check)
            finally:
                for target,source in (('regional_compiler_model_seam_tests','compact_factor_seam_tests'),
                    ('regional_compiler_model_forget_checks','compact_factor_forget_checks'),
                    ('regional_compiler_model_pair_splices','compact_gold_pair_splices')):
                    self.stats[target]+=self.model.stats[source]-before[source]
            # No cooperative call after this point until the entire transition commits.
            if child is not None:
                target=self.following.get(child)
                if target is None:
                    target=self.created;self.created+=1;self.following[child]=target
                    self.stats['regional_compiler_states']+=1
                else:self.stats['regional_compiler_merges']+=1
                self.nodes.setdefault(ident,(self.level,[]))[1].append((code,target))
            self.nodes.setdefault(ident,(self.level,[]))
            self.steps+=1;self.choice+=1;spent+=1;self.stats['regional_compiler_transitions']+=1
        return self.complete

    def result(self,check=lambda:None):
        self._assert_scope();check()
        if not self.complete:return None
        if self._result is not None:return self._result
        counts={ident:1 for ident in self.terminals};supports=[0]*self.problem.n
        for ident,(level,edges) in sorted(self.nodes.items(),key=lambda item:(-item[1][0],item[0])):
            number=0
            for code,child in edges:
                check();self.stats['regional_compiler_backward_edges']+=1
                if counts.get(child,0):number+=counts[child];supports[self.order[level]]|=1<<code
            counts[ident]=number
        witness=None
        if counts.get(0,0):
            witness=[0]*self.problem.n;node=0
            while node not in self.terminals:
                check();level,edges=self.nodes[node]
                code,node=next((code,child) for code,child in edges if counts.get(child,0))
                witness[self.order[level]]=code
            witness=tuple(witness)
            if not self.problem.validate(witness).get('valid') and self.owner.target=='gold':
                raise RuntimeError('Regional compiler Gold witness failed independent validation.')
        check();self._result={'complete':True,'existence_only':self.existence_only,
            'cells':self.order,'nodes':{n:(l,tuple(e)) for n,(l,e) in self.nodes.items()},
            'root':0,'terminals':self.terminals,'witness':witness,'feasible':bool(counts.get(0,0)),
            'rows':None if self.existence_only else counts.get(0,0),
            'supports':None if self.existence_only else tuple(supports)}
        return self._result

    def snapshot(self):
        self._assert_scope()
        body={'schema':1,'binding':self.binding,'level':self.level,'index':self.index,'choice':self.choice,
          'steps':self.steps,'created':self.created,'layer':self.layer,
          'following':[(state,ident) for state,ident in self.following.items()],
          'nodes':[(ident,level,edges) for ident,(level,edges) in self.nodes.items()],
          'terminals':self.terminals,'complete':self.complete,'blocked':self.blocked,'stats':self.stats}
        record={'body':body,'sha256':_digest(body)}
        encoded=json.dumps(record,separators=(',',':'),allow_nan=False)
        if _json_size(record)>MAX_BYTES:raise CompactLimit('regional continuation byte cap')
        return json.loads(encoded)

    def _restore(self,record):
        try:
            if type(record)is not dict or set(record)!={'body','sha256'}:raise ValueError
            if _json_size(record)>MAX_BYTES:raise ValueError
            body=record['body']
            keys={'schema','binding','level','index','choice','steps','created','layer','following','nodes','terminals','complete','blocked','stats'}
            if type(body)is not dict or set(body)!=keys or body['schema']!=1 or body['binding']!=self.binding or record['sha256']!=_digest(body):raise ValueError
            for key in ('level','index','choice','steps','created'):
                if type(body[key])is not int or body[key]<0:raise ValueError
            if type(body['complete'])is not bool or type(body['blocked'])is not bool:raise ValueError
            if body['level']>self.problem.n or body['steps']>self.policy[2] or not 1<=body['created']<=self.policy[1]+1:raise ValueError
            stats=body['stats']
            if type(stats)is not dict or set(stats)!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in stats.values()):raise ValueError
            created=body['created'];level=body['level']
            def node_id(x):
                if type(x)is not int or not 0<=x<created:raise ValueError
                return x
            layer=[(node_id(ident),_tuple(state)) for ident,state in body['layer']]
            following={_tuple(state):node_id(ident) for state,ident in body['following']}
            if len(following)!=len(body['following']) or len({n for n,s in layer})!=len(layer):raise ValueError
            if body['index']>len(layer) or body['choice']>(len(self.model.choices[level]) if level<self.problem.n else 0):raise ValueError
            nodes={}
            for ident,lev,edges in body['nodes']:
                node_id(ident)
                if ident in nodes or type(lev)is not int or not 0<=lev<self.problem.n:raise ValueError
                choices=set();parsed=[]
                for code,child in edges:
                    if type(code)is not int or code not in self.model.choices[lev] or code in choices:raise ValueError
                    choices.add(code);parsed.append((code,node_id(child)))
                nodes[ident]=(lev,parsed)
            terminals=tuple(node_id(n) for n in body['terminals'])
            if len(terminals)!=len(set(terminals)) or (terminals and (not body['complete'] or level!=self.problem.n)):raise ValueError
            for ident,(lev,edges) in nodes.items():
                for code,child in edges:
                    if child in nodes and nodes[child][0]!=lev+1:raise ValueError
            if body['complete'] and body['blocked']:raise ValueError
            def validate_state(state,depth):
                if len(state)!=(3 if self.owner.target=='gold' else 2) or type(state[0])is not int or not 0<=state[0]<(1<<self.problem.n):raise ValueError
                if state[0]&~self.model.future[depth] or len(state[1])!=len(self.model.frontiers[depth]) or any(type(mask)is not int or not 0<=mask<2048 for mask in state[1]):raise ValueError
                if self.owner.target=='gold':
                    pairs=state[2];endpoints=[]
                    if type(pairs)is not tuple or pairs!=tuple(sorted(pairs)):raise ValueError
                    for pair in pairs:
                        if len(pair)!=2 or any(type(v)is not int or not 0<=v<33*self.problem.n for v in pair) or pair[0]>=pair[1]:raise ValueError
                        endpoints.extend(pair)
                    if len(endpoints)!=len(set(endpoints)):raise ValueError
                    expected=[];members=set(self.order[:depth])
                    for eid,mask in zip(self.model.frontiers[depth],state[1]):
                        a,sa,b,sb=self.problem.edges[eid]
                        expected.extend((33*a+11*sa+p) if a in members else (33*b+11*sb+10-p) for p in values(mask))
                    if set(endpoints)!=set(expected):raise ValueError
            for ident,state in layer:validate_state(state,level)
            if level==self.problem.n and following:raise ValueError
            for state,ident in following.items():validate_state(state,level+1)
            if body['complete'] and level<self.problem.n and layer:raise ValueError
            if body['complete'] and level==0 and not self.model.missing:raise ValueError
            if body['blocked'] and body['steps']<self.policy[2] and created<=self.policy[1]:raise ValueError
            if stats['regional_compiler_transitions']!=body['steps'] or stats['regional_compiler_transition_attempts']<body['steps']:raise ValueError
            self.level=level;self.index=body['index'];self.choice=body['choice'];self.steps=body['steps'];self.created=created
            self.layer=layer;self.following=following;self.nodes=nodes;self.terminals=terminals
            self.complete=body['complete'];self.blocked=body['blocked']
            setup={name:self.stats[name] for name in ('regional_order_candidates','regional_order_cost_evaluations','regional_compiler_model_builds','regional_compiler_model_placements')}
            self.stats=dict(stats)
            for name,value in setup.items():self.stats[name]+=value
        except (TypeError,KeyError,IndexError,ValueError,OverflowError,RecursionError) as error:
            raise ValueError('Invalid or incompatible trusted regional continuation.') from error


def add_dag_cp_sat(diagram,model,codes,*,prefix='regional_dag',check=lambda:None,work=None):
    """Unit flow over an exact DAG; no concrete path expansion."""
    if not diagram['complete'] or diagram['existence_only']:raise ValueError('Exact completed diagram required.')
    incoming=defaultdict(list);outgoing=defaultdict(list);arcs=[];constraints=0
    work={} if work is None else work;work.update(arcs=0,constraints=0,expanded_histories=0)
    for node,(level,edges) in diagram['nodes'].items():
        for code,child in edges:
            check();arc=model.new_bool_var(f'{prefix}_{node}_{code}');arcs.append(arc);work['arcs']+=1
            incoming[child].append(arc);outgoing[node].append(arc)
            model.add(codes[diagram['cells'][level]]==code).only_enforce_if(arc);constraints+=1;work['constraints']+=1
    terminals=set(diagram['terminals'])
    for node in set(incoming)|set(outgoing)|{diagram['root']}:
        check()
        if node==diagram['root']:model.add(sum(outgoing[node])==1)
        elif node not in terminals:model.add(sum(incoming[node])==sum(outgoing[node]))
        else:continue
        constraints+=1;work['constraints']+=1
    model.add(sum(v for n in terminals for v in incoming[n])==1);constraints+=1;work['constraints']+=1
    return {'arcs':len(arcs),'constraints':constraints,'expanded_histories':0}


def add_dag_sat(diagram,solver,literal,new_variable,*,check=lambda:None,work=None):
    """Boolean path encoding on existing exactly-one cell literals."""
    if not diagram['complete'] or diagram['existence_only']:raise ValueError('Exact completed diagram required.')
    incoming=defaultdict(list);outgoing=defaultdict(list);arcs=clauses=0
    work={} if work is None else work;work.update(arcs=0,constraints=0,expanded_histories=0)
    def add(row):
        nonlocal clauses
        solver.add_clause(row);clauses+=1;work['constraints']+=1
    for node,(level,edges) in diagram['nodes'].items():
        for code,child in edges:
            check();arc=new_variable();arcs+=1;work['arcs']+=1;incoming[child].append(arc);outgoing[node].append(arc)
            add([-arc,literal(diagram['cells'][level],code)])
    terminals=set(diagram['terminals']);root=diagram['root']
    for node in set(incoming)|set(outgoing)|{root}:
        check();ins=incoming[node];outs=outgoing[node]
        if node==root:add(outs)
        else:
            for arc in outs:add([-arc,*ins])
            if node not in terminals:
                for arc in ins:add([-arc,*outs])
        for index,arc in enumerate(outs):
            for other in outs[index+1:]:add([-arc,-other])
    add([v for n in terminals for v in incoming[n]])
    return {'arcs':arcs,'constraints':clauses,'expanded_histories':0}


class RegionalCompiler:
    def __init__(self,owner,*,limit=4):
        self.owner=owner;self.semantic=source_binding(owner);self.limit=limit;self.sessions=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
        self.solution=None;self.directory=None;self._last={}
    def configure_cache_dir(self,path):
        self.directory=None if path is None else Path(path)/'regional-compiler-v1'
    def _sync(self,key,session):
        previous=self._last.get(key,{})
        for name,value in session.stats.items():self.stats[name]+=max(0,value-previous.get(name,0))
        self._last[key]=dict(session.stats)
    def get(self,domains,check=lambda:None,*,existence_only=False,lease=2048,max_states=20000,max_steps=200000):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Regional compiler cache source changed.')
        domains=_domains(self.owner.problem,domains);key=(domains,existence_only,max_states,max_steps)
        self.stats['regional_compiler_queries']+=1
        if key in self.sessions:
            session=self.sessions[key];self.sessions.move_to_end(key)
            self.stats['regional_compiler_cache_hits' if session.complete else 'regional_compiler_resumes']+=1
        else:
            session=RegionalCompilerSession(self.owner,domains,existence_only=existence_only,max_states=max_states,max_steps=max_steps)
            if self.directory is not None:
                path=self.directory/(session.binding+'.json')
                try:
                    if path.is_file() and path.stat().st_size<=MAX_BYTES:
                        record=json.loads(path.read_text(encoding='utf-8'))
                        session._restore(record)
                        # Restored history is not new work; model/order setup is.
                        self._last[key]=dict(record['body']['stats']);self.stats['regional_compiler_loaded']+=1
                except (OSError,ValueError,CompactLimit):self.stats['regional_compiler_io_failures']+=1
            self.sessions[key]=session
            while len(self.sessions)>self.limit:
                old,_=self.sessions.popitem(last=False);self._last.pop(old,None)
        try:
            was=session.complete;session.advance(lease,check);result=session.result(check)
            if result is not None and not was:self.stats['regional_compiler_completed']+=1
            if result is None:self.stats['regional_compiler_unknown']+=1
            return result
        finally:
            self._sync(key,session)
            if self.directory is not None:
                try:
                    from io_utils import atomic_json
                    self.directory.mkdir(parents=True,exist_ok=True)
                    atomic_json(self.directory/(session.binding+'.json'),session.snapshot())
                    self.stats['regional_compiler_saved']+=1
                except (OSError,ValueError,CompactLimit):self.stats['regional_compiler_io_failures']+=1
    def prune(self,domains,check=lambda:None,**kwargs):
        self.solution=None;result=self.get(domains,check,**kwargs)
        if result is None:return False
        if result['existence_only']:
            updated=list(domains) if result['feasible'] else [0]*len(domains)
        else:updated=[old&support for old,support in zip(domains,result['supports'])]
        check();self.stats['regional_compiler_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,updated))
        changed=list(domains)!=updated;domains[:]=updated;self.solution=result['witness'];return changed
