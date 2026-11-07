"""Exact multi-region integer master with branch-compatible DAG pricing (F13).

The master selects integral regional interface columns. Columns are generated
by exact DAG pricing, not history expansion. Replayed weighted inventory cuts
bound every omitted column; connectivity is separated at every exact signature
join. Completed pricing, generated columns, branch cursors and checked bounds
survive trusted local checkpoints. Caps and interruptions remain unknown.
"""
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
import json
from constraint_region_compiler_round26 import _domains,_digest,_source,_json_size,MAX_BYTES
from constraint_region_incremental import source_binding
from constraint_region_interfaces_round26 import ParametricEnvelope
from constraint_region_network import joined_signature
from constraint_region_tree_round26 import _pack,_unpack
from constraint_pruning import values
from constraint_separator import _Budget

STAT_KEYS=('integer_region_queries','integer_region_completed','integer_region_unknown','integer_region_compiler_transitions',
 'integer_region_compiler_states','integer_region_price_queries','integer_region_price_edges','integer_region_columns',
 'integer_region_column_reuses','integer_region_branches','integer_region_connectivity_rejections',
 'integer_region_bound_queries','integer_region_bound_replays','integer_region_bound_hits',
 'integer_region_bound_failures','integer_region_candidate_checks','integer_region_candidate_fixings',
 'integer_region_validations','integer_region_closure_checks','integer_region_closure_states','integer_region_failed_state_hits','integer_region_saved','integer_region_loaded','integer_region_io_failures')


def price_diagram(owner,diagram,domains,weights,*,prefix_cells=(),prefix_signature=None,excluded=(),check=lambda:None,stats=None,code_weights=None):
    """Exact integer maximum over all branch-compatible paths, with a witness."""
    if len(weights)!=owner.problem.n or any(type(v)is not int and not isinstance(v,Fraction) for v in weights):raise ValueError('Pricing requires exact rational piece weights.')
    code_weights={} if code_weights is None else code_weights
    if any(type(v)is not int and not isinstance(v,Fraction) for v in code_weights.values()):raise ValueError('Pricing requires exact rational arc weights.')
    excluded=set(excluded);memo={};terminal_results={}
    def visit(node):
        check()
        if node in memo:return memo[node]
        if node in diagram['terminals']:
            signature=diagram['terminals'][node]
            if signature in excluded:return None
            merged=signature if prefix_signature is None else joined_signature(owner,prefix_cells,prefix_signature,diagram['cells'],signature,domains,check)
            if merged is None:
                if stats is not None:stats['integer_region_connectivity_rejections']+=1
                return None
            terminal_results[signature]=merged
            return 0,(),signature
        level,edges=diagram['nodes'][node];cell=diagram['cells'][level];best=None
        for code,child in edges:
            check()
            if stats is not None:stats['integer_region_price_edges']+=1
            if not domains[cell]>>code&1:continue
            answer=visit(child)
            if answer is None:continue
            score,path,signature=answer;candidate=(score+weights[code//3]+code_weights.get((cell,code),0),(code,*path),signature)
            if best is None or candidate>best:best=candidate
        memo[node]=best;return best
    if stats is not None:stats['integer_region_price_queries']+=1
    result=None if diagram['root'] is None else visit(diagram['root'])
    return None if result is None else (*result,terminal_results[result[2]])


def verify_weighted_cut(owner,diagrams,domains,remaining,prefix_cells,prefix_signature,record,check=lambda:None,stats=None):
    weights=record['weights'];maxima=record['maxima'];required=record['required']
    if type(required)is not int or any(type(v)is not int or v<0 for v in weights) or len(weights)!=owner.problem.n or len(maxima)!=len(remaining):raise ValueError('Invalid regional weighted cut.')
    used=0 if prefix_signature is None else prefix_signature[0]
    if required!=sum(w for tile,w in enumerate(weights) if not used>>tile&1):raise ValueError('Weighted inventory obligation differs.')
    calculated=[]
    for region in remaining:
        result=price_diagram(owner,diagrams[region],domains,weights,prefix_cells=prefix_cells,prefix_signature=prefix_signature,check=check,stats=stats)
        calculated.append(None if result is None else result[0])
    if tuple(calculated)!=tuple(maxima):raise ValueError('Weighted cut does not cover every omitted column.')
    return any(v is None for v in calculated) or sum(calculated)<required


class _ClosureAudit:
    """Independent forward signature closure, separate from priced DFS choices.

    The compiler's exact completed relations are inputs. This checks exhaustive
    integer termination with a different traversal, including columns that the
    priced search never materialized. Its pair cursor itself is resumable.
    """
    def __init__(self,owner,diagrams,domains,*,max_states=20000,state=None):
        self.owner=owner;self.diagrams=diagrams;self.domains=domains;self.max_states=max_states
        self.level=0;self.members=();self.layer=[None];self.following=set();self.i=0;self.j=0
        self.checks=0;self.states=1;self.complete=False;self.feasible=False;self.blocked=False
        if state is not None:
            required={'level','members','layer','following','i','j','checks','states','complete','feasible','blocked'}
            if type(state)is not dict or set(state)!=required or any(type(state[k])is not int or state[k]<0 for k in ('level','i','j','checks','states')) or any(type(state[k])is not bool for k in ('complete','feasible','blocked')):raise ValueError('Invalid closure audit cursor.')
            if state['level']>len(diagrams) or len(state['layer'])>max_states or len(state['following'])>max_states or state['i']>len(state['layer']):raise ValueError('Closure audit bounds differ.')
            expected=tuple(sorted(c for d in diagrams[:state['level']] for c in d['cells']))
            if tuple(state['members'])!=expected:raise ValueError('Closure audit region scope differs.')
            if state['complete'] and state['level']<len(diagrams) and state['layer']:raise ValueError('Closure audit completion is premature.')
            if state['complete'] and state['blocked'] or state['feasible'] and not state['complete']:raise ValueError('Invalid closure audit outcome.')
            if state['level']<len(diagrams):
                choices=len(set(diagrams[state['level']]['terminals'].values()))
                if state['j']>=max(1,choices):raise ValueError('Closure audit column cursor exceeds relation.')
            for key,value in state.items():setattr(self,key,value)
            self.layer=list(self.layer);self.following=set(self.following)
    def advance(self,check):
        check()
        if self.complete or self.blocked:return
        if self.level==len(self.diagrams):self.complete=True;self.feasible=bool(self.layer);return
        diagram=self.diagrams[self.level];right=tuple(sorted(set(diagram['terminals'].values())))
        if self.i==len(self.layer) or not right:
            self.layer=sorted(self.following);self.following=set();self.members=tuple(sorted((*self.members,*diagram['cells'])))
            self.level+=1;self.i=0;self.j=0
            if not self.layer:self.complete=True;self.feasible=False
            return
        prefix=self.layer[self.i];signature=right[self.j]
        merged=signature if prefix is None else joined_signature(self.owner,self.members,prefix,diagram['cells'],signature,self.domains,check)
        if merged is not None:
            self.states+=int(merged not in self.following);self.following.add(merged)
            if len(self.following)>self.max_states:self.blocked=True;return
        self.checks+=1;self.j+=1
        if self.j==len(right):self.i+=1;self.j=0
    def snapshot(self):
        return {key:(tuple(value) if key=='following' else value) for key,value in self.__dict__.items() if key in
          ('level','members','layer','following','i','j','checks','states','complete','feasible','blocked')}


class IntegerRegionalSession:
    def __init__(self,owner,domains,*,region_size=4,max_states=20000,max_steps=200000,resume=None,columns=(),cuts=(),dual_hint=None):
        self.owner=owner;self.problem=owner.problem;self.semantic=source_binding(owner);self.domains=_domains(self.problem,domains)
        if any(type(v)is not int or v<1 for v in (region_size,max_states,max_steps)):raise ValueError('Invalid integer regional policy.')
        self.policy=(region_size,max_states,max_steps)
        pending=set(range(self.problem.n));parts=[]
        while pending:
            first=min(pending);part=[first];pending.remove(first)
            while len(part)<region_size:
                neighbors=set().union(*(owner.neighbors[c] for c in part))&pending
                if not neighbors:break
                nxt=min(neighbors);part.append(nxt);pending.remove(nxt)
            parts.append(tuple(sorted(part)))
        self.parts=tuple(parts);self.binding=_digest((self.semantic,self.domains,self.policy,self.parts,_source(),Path(__file__).read_bytes().hex()))
        self.envelopes=[ParametricEnvelope(owner,cells,self.domains,max_states=max_states,max_steps=max_steps) for cells in self.parts]
        self.diagrams=[];self.compiling=0;self.stack=[];self.columns={};self.bounds={};self.fixings={};self.stats=dict.fromkeys(STAT_KEYS,0)
        self.complete=False;self.blocked=False;self.witness=None;self.steps=0;self.exhausted=False;self.root_verified=False;self.verifier=None
        self.failed=OrderedDict()
        self.lp_nodes=OrderedDict();self.lp_pool=set(columns);self.lp_cuts=list(cuts);self.lp_stats={};self.lp_duals={} if dual_hint is None else dict(dual_hint)
        if resume is not None:self._restore(resume)
    def _frame(self,level,members,signature,witness):return {'level':level,'members':members,'signature':signature,'witness':witness,'excluded':set(),'bound_checked':False,'lp_complete':False}
    def _cuts(self,domains,level,members,signature,check):
        remaining=tuple(range(level,len(self.parts)));used=0 if signature is None else signature[0]
        key=(tuple(domains),level,members,signature)
        if key in self.bounds:self.stats['integer_region_bound_hits']+=1;return self.bounds[key]['impossible']
        frequency=[0]*self.problem.n
        for region in remaining:
            possible=0
            for terminal in self.diagrams[region]['terminals'].values():possible|=terminal[0]
            for tile in values(possible&~used):frequency[tile]+=1
        vectors=[]
        for threshold in range(min(3,len(remaining))+1):
            weights=tuple(int(not used>>tile&1 and frequency[tile]<=threshold) for tile in range(self.problem.n))
            if any(weights) and weights not in vectors:vectors.append(weights)
        certificates=[];impossible=False
        for weights in vectors:
            maxima=[];self.stats['integer_region_bound_queries']+=1
            for region in remaining:
                result=price_diagram(self.owner,self.diagrams[region],domains,weights,prefix_cells=members,prefix_signature=signature,check=check,stats=self.stats)
                maxima.append(None if result is None else result[0])
            required=sum(weights)
            if any(value is None for value in maxima) or sum(maxima)<required:
                record={'weights':weights,'maxima':tuple(maxima),'required':required}
                self.stats['integer_region_bound_replays']+=1
                if not verify_weighted_cut(self.owner,self.diagrams,domains,remaining,members,signature,record,check,self.stats):raise RuntimeError('Generated regional cut failed exact replay.')
                certificates.append(record);impossible=True;self.stats['integer_region_bound_failures']+=1;break
        check();self.bounds[key]={'impossible':impossible,'certificates':certificates}
        while len(self.bounds)>256:self.bounds.pop(next(iter(self.bounds)))
        return impossible
    def _reject(self,frame,verified_bound=False):
        self.failed[frame['level'],frame['signature']]=True
        while len(self.failed)>4096:self.failed.popitem(last=False)
        self.stack.pop()
        if frame['level']==0 and verified_bound:self.root_verified=True
    def advance(self,lease=128,check=lambda:None):
        if type(lease)is not int or lease<1:raise ValueError('Invalid integer regional lease.')
        if source_binding(self.owner)!=self.semantic:raise ValueError('Integer regional source changed.')
        for _ in range(lease):
            check()
            if self.complete or self.blocked:return self.complete
            if self.steps>=self.policy[2]:self.blocked=True;return False
            self.steps+=1
            if self.exhausted:
                old=self.verifier.checks;old_states=self.verifier.states;self.verifier.advance(check)
                self.stats['integer_region_closure_checks']+=self.verifier.checks-old
                self.stats['integer_region_closure_states']+=self.verifier.states-old_states
                if self.verifier.blocked:self.blocked=True;return False
                if self.verifier.complete:
                    if self.verifier.feasible:raise RuntimeError('Independent closure contradicts integer-master infeasibility.')
                    self.complete=True;return True
                continue
            if self.compiling<len(self.envelopes):
                env=self.envelopes[self.compiling];before=(env.session.steps,env.session.states)
                try:done=env.advance(1,check)
                except _Budget:self.blocked=True;return False
                finally:
                    self.stats['integer_region_compiler_transitions']+=env.session.steps-before[0]
                    self.stats['integer_region_compiler_states']+=env.session.states-before[1]
                if done:
                    diagram=env.view(self.domains,check,fixed_chains=True);self.diagrams.append(diagram);self.compiling+=1
                    if not diagram['rows']:self.complete=True;return True
                if self.compiling==len(self.envelopes):self.stack=[self._frame(0,(),None,{})]
                continue
            if not self.stack:
                if self.root_verified:self.complete=True;return True
                self.exhausted=True;self.verifier=_ClosureAudit(self.owner,self.diagrams,self.domains,max_states=self.policy[1]);continue
            frame=self.stack[-1];level=frame['level']
            if (level,frame['signature']) in self.failed:
                self.stats['integer_region_failed_state_hits']+=1;self.stack.pop();continue
            if level==len(self.parts):
                witness=tuple(frame['witness'][c] for c in range(self.problem.n))
                self.stats['integer_region_validations']+=1
                if self.owner.target=='gold' and not self.problem.validate(witness).get('valid'):raise RuntimeError('Integer regional witness failed Gold validation.')
                check();self.witness=witness;self.complete=True;return True
            if not frame['lp_complete']:
                from constraint_region_branch_price_round26 import BranchPriceNode
                lpkey=(level,frame['members'],frame['signature'],tuple(sorted(frame['witness'].items())))
                if lpkey not in self.lp_nodes:
                    self.lp_nodes[lpkey]=BranchPriceNode(self.owner,self.diagrams,self.domains,level,frame['members'],frame['signature'],frame['witness'],columns=self.lp_pool,cuts=self.lp_cuts,dual_hint=self.lp_duals)
                node=self.lp_nodes[lpkey];self.lp_nodes.move_to_end(lpkey)
                try:node.advance(check)
                finally:
                    old=self.lp_stats.get(lpkey,{})
                    for name,value in node.stats.items():self.stats[name]+=value-old.get(name,0)
                    self.lp_stats[lpkey]=dict(node.stats)
                    self.lp_pool.update(node.columns)
                    if node.dual is not None:self.lp_duals=dict(zip(node.rows,node.dual))
                    if len(self.lp_pool)>4096:self.lp_pool=set(sorted(self.lp_pool)[:4096])
                    for cut in node.cuts:
                        if cut not in self.lp_cuts and len(self.lp_cuts)<32:self.lp_cuts.append(cut)
                    while len(self.lp_nodes)>64:
                        evicted,_=self.lp_nodes.popitem(last=False);self.lp_stats.pop(evicted,None)
                if not node.complete:continue
                frame['lp_complete']=True
                if node.status=='infeasible':self._reject(frame,verified_bound=True);continue
                if node.witness is not None:
                    check();self.witness=node.witness;self.complete=True;self.stats['integer_region_validations']+=1;return True
                continue
            if not frame['bound_checked']:
                if self._cuts(self.domains,level,frame['members'],frame['signature'],check):self._reject(frame,verified_bound=True);continue
                frame['bound_checked']=True
            # Rare-piece weights order complete columns, never change authority.
            weights=tuple(1 for _ in range(self.problem.n))
            result=price_diagram(self.owner,self.diagrams[level],self.domains,weights,prefix_cells=frame['members'],prefix_signature=frame['signature'],excluded=frame['excluded'],check=check,stats=self.stats)
            if result is None:self._reject(frame);continue
            score,path,signature,merged=result;check()
            key=(level,signature)
            if key in self.columns:self.stats['integer_region_column_reuses']+=1
            else:self.columns[key]=path;self.stats['integer_region_columns']+=1
            frame['excluded'].add(signature);self.stats['integer_region_branches']+=1
            witness={**frame['witness'],**dict(zip(self.parts[level],path))}
            self.stack.append(self._frame(level+1,tuple(sorted((*frame['members'],*self.parts[level]))),merged,witness))
        return self.complete
    def candidate_fixings(self,check=lambda:None,limit=8):
        if self.compiling<len(self.envelopes):return {}
        if type(limit)is not int or limit<0:raise ValueError('Invalid candidate-fixing limit.')
        result={};tested=0
        for cell,mask in enumerate(self.domains):
            if mask.bit_count()<2:continue
            for code in values(mask):
                key=(cell,code)
                if key not in self.fixings:
                    if tested>=limit:return result
                    trial=list(self.domains);trial[cell]=1<<code;self.stats['integer_region_candidate_checks']+=1;tested+=1
                    impossible=self._cuts(trial,0,(),None,check);check();self.fixings[key]=impossible
                    self.stats['integer_region_candidate_fixings']+=int(impossible)
                if self.fixings[key]:result[cell]=result.get(cell,0)|(1<<code)
        return result
    def result(self):
        return None if not self.complete else {'complete':True,'feasible':self.witness is not None,'witness':self.witness}
    def snapshot(self):
        stack=[{**frame,'excluded':tuple(frame['excluded'])} for frame in self.stack]
        body={'schema':1,'binding':self.binding,'compiling':self.compiling,'stack':stack,'columns':self.columns,'bounds':self.bounds,'fixings':self.fixings,
              'stats':self.stats,'complete':self.complete,'blocked':self.blocked,'witness':self.witness,'steps':self.steps,'envelopes':[e.snapshot() for e in self.envelopes],
              'lp_nodes':{key:node.snapshot() for key,node in self.lp_nodes.items()},'lp_pool':tuple(sorted(self.lp_pool)),'lp_cuts':tuple(self.lp_cuts),'lp_duals':self.lp_duals,'failed':tuple(self.failed),'exhausted':self.exhausted,'root_verified':self.root_verified,'verifier':None if self.verifier is None else self.verifier.snapshot()}
        packed=_pack(body);record={'body':packed,'sha256':_digest(packed)}
        if _json_size(record)>MAX_BYTES:raise _Budget
        return record
    def _restore(self,record):
        if type(record)is not dict or set(record)!={'body','sha256'} or record['sha256']!=_digest(record['body']) or _json_size(record)>MAX_BYTES:raise ValueError('Invalid integer master checkpoint.')
        body=_unpack(record['body']);keys={'schema','binding','compiling','stack','columns','bounds','fixings','stats','complete','blocked','witness','steps','envelopes','lp_nodes','lp_pool','lp_cuts','lp_duals','failed','exhausted','root_verified','verifier'}
        if type(body)is not dict or set(body)!=keys or body['schema']!=1 or body['binding']!=self.binding:raise ValueError('Integer master checkpoint source/scope differs.')
        if any(type(body[k])is not bool for k in ('complete','blocked')) or body['complete'] and body['blocked'] or type(body['compiling'])is not int or not 0<=body['compiling']<=len(self.parts) or type(body['steps'])is not int or not 0<=body['steps']<=self.policy[2]:raise ValueError('Invalid integer master status.')
        if set(body['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in body['stats'].values()) or len(body['envelopes'])!=len(self.parts):raise ValueError('Invalid integer master counters/regions.')
        envelopes=[ParametricEnvelope(self.owner,cells,self.domains,max_states=self.policy[1],max_steps=self.policy[2],resume=saved) for cells,saved in zip(self.parts,body['envelopes'])]
        diagrams=[]
        for i in range(body['compiling']):
            if not envelopes[i].session.complete:raise ValueError('Integer master consumes unfinished region.')
            diagrams.append(envelopes[i].view(self.domains,fixed_chains=True))
        stack=[]
        for frame in body['stack']:
            if set(frame)!={'level','members','signature','witness','excluded','bound_checked','lp_complete'} or type(frame['level'])is not int or not 0<=frame['level']<=len(self.parts) or type(frame['bound_checked'])is not bool or type(frame['lp_complete'])is not bool:raise ValueError('Invalid integer branch cursor.')
            expected=tuple(sorted(c for cells in self.parts[:frame['level']] for c in cells))
            if frame['members']!=expected or set(frame['witness'])!=set(expected) or any(type(v)is not int or not self.domains[c]>>v&1 for c,v in frame['witness'].items()):raise ValueError('Integer branch escapes scope.')
            stack.append({**frame,'excluded':set(frame['excluded'])})
        if any(a['level']>=b['level'] for a,b in zip(stack,stack[1:])):raise ValueError('Invalid integer branch depth order.')
        witness=body['witness']
        if witness is not None and (len(witness)!=self.problem.n or any(type(v)is not int or not d>>v&1 for d,v in zip(self.domains,witness)) or self.owner.target=='gold' and not self.problem.validate(witness).get('valid')):raise ValueError('Invalid resumed integer Gold witness.')
        from constraint_region_branch_price_round26 import BranchPriceNode,validate_cut
        lp_nodes=OrderedDict()
        if len(body['lp_nodes'])>64 or len(body['lp_pool'])>4096 or len(body['lp_cuts'])>32:raise ValueError('Saved branch-price cache exceeds policy.')
        for key,state in body['lp_nodes'].items():
            if type(key)is not tuple or len(key)!=4:raise ValueError('Invalid branch-price scope key.')
            level,members,signature,fixed=key
            if type(level)is not int or not 0<=level<len(self.parts) or members!=tuple(sorted(c for part in self.parts[:level] for c in part)) or set(dict(fixed))!=set(members):raise ValueError('Invalid branch-price scope.')
            lp_nodes[key]=BranchPriceNode(self.owner,diagrams,self.domains,level,members,signature,dict(fixed),state=state)
        lp_cuts=[validate_cut(self.problem,cut) for cut in body['lp_cuts']]
        failed=body['failed']
        if type(failed)is not tuple or len(failed)>4096 or any(type(item)is not tuple or len(item)!=2 or type(item[0])is not int or not 0<=item[0]<len(self.parts) for item in failed):raise ValueError('Invalid completed integer suffix cache.')
        if type(body['exhausted'])is not bool or type(body['root_verified'])is not bool:raise ValueError('Invalid integer termination status.')
        verifier=None if body['verifier'] is None else _ClosureAudit(self.owner,diagrams,self.domains,max_states=self.policy[1],state=body['verifier'])
        if body['exhausted'] and verifier is None or verifier is not None and not body['exhausted']:raise ValueError('Missing independent closure cursor.')
        if body['complete'] and body['witness'] is None and body['compiling']<len(self.parts):
            if not diagrams or diagrams[-1]['rows']!=0:raise ValueError('Integer completion precedes a complete empty region.')
        if body['complete'] and body['witness'] is None and body['compiling']==len(self.parts):
            if body['root_verified']:
                root_lp=any(key[0]==0 and node.status=='infeasible' and node.certificate is not None for key,node in lp_nodes.items())
                root_bound=body['bounds'].get((self.domains,0,(),None),{})
                if not root_lp and not (root_bound.get('impossible') is True and root_bound.get('certificates')):raise ValueError('Verified integer root has no checked bound.')
            if not body['root_verified'] and (verifier is None or not verifier.complete or verifier.feasible):raise ValueError('Unverified integer exhaustion.')
        self.exhausted=body['exhausted'];self.root_verified=body['root_verified'];self.verifier=verifier
        self.failed=OrderedDict((item,True) for item in failed)
        self.lp_nodes=lp_nodes;self.lp_pool=set(body['lp_pool']);self.lp_cuts=lp_cuts;self.lp_duals=dict(body['lp_duals']);self.lp_stats={key:dict(node.stats) for key,node in lp_nodes.items()}
        self.envelopes=envelopes;self.diagrams=diagrams;self.compiling=body['compiling'];self.stack=stack;self.columns=body['columns'];self.bounds=body['bounds'];self.fixings=body['fixings'];self.stats=body['stats'];self.complete=body['complete'];self.blocked=body['blocked'];self.witness=witness;self.steps=body['steps']


class IntegerRegionalMaster:
    def __init__(self,owner,limit=4):
        self.owner=owner;self.semantic=source_binding(owner);self.limit=limit;self.cache=OrderedDict();self.last={};self.stats=dict.fromkeys(STAT_KEYS,0);self.directory=None;self.solution=None;self.column_pool=set();self.cut_pool=[];self.dual_pool={};self.completed_failures=OrderedDict()
    def configure_cache_dir(self,path):self.directory=None if path is None else Path(path)/'integer-regions-v1'
    def prune(self,domains,check=lambda:None,*,lease=1024,**policy):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Integer master cache source changed.')
        _domains(self.owner.problem,domains)
        self.solution=None;key=(tuple(domains),tuple(sorted(policy.items())));self.stats['integer_region_queries']+=1
        for scope in self.completed_failures:
            if all(not current&~old for current,old in zip(domains,scope)):
                check();self.stats['integer_region_bound_hits']+=1;domains[0]=0;return True
        if key not in self.cache:
            session=IntegerRegionalSession(self.owner,domains,columns=self.column_pool,cuts=self.cut_pool,dual_hint=self.dual_pool,**policy)
            if self.directory is not None:
                path=self.directory/(session.binding+'.json')
                try:
                    if path.is_file() and path.stat().st_size<=MAX_BYTES:session._restore(json.loads(path.read_text(encoding='utf-8')));self.last[key]=dict(session.stats);self.stats['integer_region_loaded']+=1
                except (OSError,ValueError,RecursionError,OverflowError,TypeError,_Budget):self.stats['integer_region_io_failures']+=1
            self.cache[key]=session
        session=self.cache[key];self.cache.move_to_end(key)
        try:
            was=session.complete;session.advance(lease,check);result=session.result()
            updated=list(domains)
            if result is None:
                self.stats['integer_region_unknown']+=1
                for cell,removed in session.candidate_fixings(check,4).items():updated[cell]&=~removed
            else:
                self.stats['integer_region_completed']+=int(not was)
                if not result['feasible']:
                    updated[0]=0;self.completed_failures[tuple(domains)]=True
                    while len(self.completed_failures)>64:self.completed_failures.popitem(last=False)
                self.solution=result['witness']
            check();changed=updated!=list(domains);domains[:]=updated;return changed
        finally:
            self.column_pool.update(session.lp_pool)
            if len(self.column_pool)>4096:self.column_pool=set(sorted(self.column_pool)[:4096])
            for cut in session.lp_cuts:
                if cut not in self.cut_pool and len(self.cut_pool)<32:self.cut_pool.append(cut)
            self.dual_pool=dict(session.lp_duals)
            old=self.last.get(key,{})
            for name,value in session.stats.items():self.stats[name]+=max(0,value-old.get(name,0))
            self.last[key]=dict(session.stats)
            if self.directory is not None:
                try:
                    from io_utils import atomic_json
                    self.directory.mkdir(parents=True,exist_ok=True);atomic_json(self.directory/(session.binding+'.json'),session.snapshot());self.stats['integer_region_saved']+=1
                except (OSError,ValueError,RecursionError,OverflowError,TypeError,_Budget):self.stats['integer_region_io_failures']+=1
            while len(self.cache)>self.limit:
                old,_=self.cache.popitem(last=False);self.last.pop(old,None)


# Imported after price_diagram is defined: the LP controller reuses that exact
# rational DAG kernel and the integration layer exposes all its work counters.
from constraint_region_branch_price_round26 import STAT_KEYS as BP_STAT_KEYS
STAT_KEYS=tuple((*STAT_KEYS,*BP_STAT_KEYS))
