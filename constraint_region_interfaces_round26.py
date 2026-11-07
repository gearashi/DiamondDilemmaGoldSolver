"""Boundary-parametric exact local DAG envelopes and correlated messages.

Outside domains and fixed-chain changes filter complete interface signatures;
they are never compiled into the envelope. Thus widening outside the region is
safe. Inventory, seam masks and every Gold pairing remain correlated.
"""
from collections import OrderedDict
from constraint_region_certificate import _Envelope
from constraint_region_cursor import PortableRegionSession
from constraint_region_network import assignment_signature,joined_signature
from constraint_region_incremental import source_binding
from constraint_region_compiler_round26 import _domains,_digest,_source,_json_size,MAX_BYTES
from constraint_pruning import values
from constraint_separator import _Budget
import json

STAT_KEYS=('regional_envelope_builds','regional_envelope_reuses','regional_envelope_transitions',
 'regional_envelope_states','regional_envelope_terminal_checks','regional_envelope_filter_edges',
 'regional_envelope_unknown','regional_envelope_prunes','regional_envelope_message_merges')

class ParametricEnvelope:
    def __init__(self,owner,cells,domains,*,max_states=20000,max_steps=200000,resume=None):
        self.owner=owner;self.cells=tuple(sorted(cells));self.domains=_domains(owner.problem,domains)
        if not self.cells or len(set(self.cells))!=len(self.cells) or any(type(c)is not int or not 0<=c<owner.problem.n for c in self.cells):
            raise ValueError('Invalid boundary-parametric region.')
        self.envelope=_Envelope(owner);self.semantic=source_binding(owner)
        # Outside masks deliberately take the original envelope. No outside
        # constraint can affect the compiled transitions through _Envelope.
        self.envelope_domains=tuple(self.domains[c] if c in self.cells else sum(1<<v for v in set(owner.problem.domains[c])) for c in range(owner.problem.n))
        self.binding=_digest((self.semantic,_source(),self.cells,self.envelope_domains,max_states,max_steps))
        self.limits=(max_states,max_steps);self.stats=dict.fromkeys(STAT_KEYS,0)
        self.session=PortableRegionSession(self.envelope,self.cells,self.envelope_domains,max_states,max_steps)
        if resume is not None:
            if type(resume)is not dict or set(resume)!={'binding','cursor','sha256'} or resume['binding']!=self.binding or resume['sha256']!=_digest(resume['cursor']):raise ValueError('Invalid parametric continuation binding.')
            if _json_size(resume)>MAX_BYTES:raise ValueError('Parametric continuation byte cap.')
            self.session=PortableRegionSession._from_local_payload(self.envelope,resume['cursor'],self.cells,self.envelope_domains,*self.limits)
        else:self.stats['regional_envelope_builds']=1
    def advance(self,lease=1024,check=lambda:None):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Parametric region source changed.')
        before=(self.session.steps,self.session.states)
        try:return self.session.advance(lease,check)
        finally:
            self.stats['regional_envelope_transitions']+=self.session.steps-before[0]
            self.stats['regional_envelope_states']+=self.session.states-before[1]
    def snapshot(self):
        cursor=json.loads(json.dumps(self.session._local_payload()))
        result={'binding':self.binding,'cursor':cursor,'sha256':_digest(cursor)}
        if _json_size(result)>MAX_BYTES:raise _Budget
        return result
    def view(self,domains,check=lambda:None,*,fixed_chains=True):
        domains=_domains(self.owner.problem,domains)
        if source_binding(self.owner)!=self.semantic:raise ValueError('Parametric region source changed.')
        if any(domains[c]&~self.envelope_domains[c] for c in self.cells):raise ValueError('Inside domain widened beyond the envelope.')
        if not self.session.complete:return None
        fixed=tuple(c for c,d in enumerate(domains) if c not in self.cells and d.bit_count()==1) if fixed_chains else ()
        fixed_signature=assignment_signature(self.envelope,fixed,tuple(next(values(domains[c])) for c in fixed),domains,check) if fixed else None
        accepted={}
        for node,signature in self.session.terminals.items():
            check();self.stats['regional_envelope_terminal_checks']+=1
            used,masks,pairs,cycles=signature;frontier=dict(zip(self.owner._frontier(set(self.cells)),masks))
            boundary=tuple((c,domains[c]) for c in sorted(set().union(*(self.owner.neighbors[c] for c in self.cells))-set(self.cells)))
            possible=self.owner._outside_possible(used,frontier,boundary,set(self.cells))
            if fixed:
                possible=possible and fixed_signature is not None and joined_signature(self.owner,self.cells,signature,fixed,fixed_signature,domains,check) is not None
            if possible:accepted[node]=signature
        counts={node:1 for node in accepted};edges={}
        for node,(level,choices) in sorted(self.session.nodes.items(),key=lambda row:-row[1][0]):
            row=[];number=0;cell=self.cells[level]
            for code,child in choices:
                check();self.stats['regional_envelope_filter_edges']+=1
                if domains[cell]>>code&1 and counts.get(child,0):row.append((code,child));number+=counts[child]
            edges[node]=(level,tuple(row));counts[node]=number
        root=self.session.root
        # Keep only reachable terminal classes; stale acceptable terminals whose
        # prefixes were removed must not leak into regional messages.
        reachable=set();pending=[] if root is None or not counts.get(root,0) else [root]
        supports=[0]*len(self.cells)
        while pending:
            node=pending.pop()
            if node in reachable:continue
            reachable.add(node)
            if node in accepted:continue
            level,row=edges[node]
            for code,child in row:supports[level]|=1<<code;pending.append(child)
        self.stats['regional_envelope_reuses']+=1
        return {'complete':True,'existence_only':False,'cells':self.cells,'root':root,
          'nodes':{node:row for node,row in edges.items() if node in reachable},
          'terminals':{node:signature for node,signature in accepted.items() if node in reachable},
          'rows':counts.get(root,0),'supports':tuple(supports)}


def signature_messages(diagram,check=lambda:None,max_entries=20000):
    """Correlated exact terminal classes, supports and one witness per class.

    This aggregates a DAG by dynamic programming; it never enumerates all paths.
    Each class is sufficient for any disjoint external Gold/inventory join.
    """
    terminals=diagram['terminals'];memo={};work=0
    def visit(node):
        nonlocal work
        check()
        if node in memo:return memo[node]
        if node in terminals:
            result={terminals[node]:{'count':1,'supports':{},'witness':{}}}
        else:
            result={};level,edges=diagram['nodes'][node];cell=diagram['cells'][level]
            for code,child in edges:
                for signature,entry in visit(child).items():
                    check();work+=1
                    if work>max_entries:raise _Budget
                    if signature not in result:result[signature]={'count':0,'supports':{},'witness':{cell:code,**entry['witness']}}
                    current=result[signature];current['count']+=entry['count'];current['supports'][cell]=current['supports'].get(cell,0)|(1<<code)
                    for c,mask in entry['supports'].items():current['supports'][c]=current['supports'].get(c,0)|mask
        memo[node]=result;return result
    result={} if diagram['root'] is None or not diagram['rows'] else visit(diagram['root'])
    return result,work


def join_messages(owner,left_cells,left,right_cells,right,domains,check=lambda:None,*,max_steps=200000):
    """Exact correlated multiway building block, returning provenance links."""
    if set(left_cells)&set(right_cells):raise ValueError('Correlated messages require disjoint interiors.')
    result={};links={};work=0
    for ls,le in left.items():
        for rs,re in right.items():
            check();work+=1
            if work>max_steps:raise _Budget
            merged=joined_signature(owner,left_cells,ls,right_cells,rs,domains,check)
            if merged is None:continue
            if merged not in result:result[merged]={'count':0,'supports':{},'witness':{**le['witness'],**re['witness']}};links[merged]=[]
            entry=result[merged];entry['count']+=le['count']*re['count'];links[merged].append((ls,rs))
            for source in (le,re):
                for c,mask in source['supports'].items():entry['supports'][c]=entry['supports'].get(c,0)|mask
    return result,links,work


class EnvelopeCache:
    def __init__(self,owner,limit=16):
        self.owner=owner;self.semantic=source_binding(owner);self.limit=limit;self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0);self._last={}
    def get(self,cells,domains,check=lambda:None,*,lease=2048,max_states=20000,max_steps=200000):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Parametric envelope cache source changed.')
        cells=tuple(sorted(cells));domains=_domains(self.owner.problem,domains)
        hit=next((key for key,env in reversed(self.cache.items()) if env.cells==cells and env.limits==(max_states,max_steps)
                  and all(not domains[c]&~env.envelope_domains[c] for c in cells)),None)
        if hit is None:
            hit=(cells,tuple(domains[c] for c in cells),max_states,max_steps)
            self.cache[hit]=ParametricEnvelope(self.owner,cells,domains,max_states=max_states,max_steps=max_steps)
        env=self.cache[hit];self.cache.move_to_end(hit)
        try:
            env.advance(lease,check)
            return env.view(domains,check)
        except _Budget:self.stats['regional_envelope_unknown']+=1;return None
        finally:
            old=self._last.get(hit,{})
            for name,value in env.stats.items():self.stats[name]+=value-old.get(name,0)
            self._last[hit]=dict(env.stats)
            while len(self.cache)>self.limit:
                key,_=self.cache.popitem(last=False);self._last.pop(key,None)


class OverlapConsistency:
    """Exact shared-cell consistency using paired DAG states, never support unions."""
    def __init__(self,owner):
        self.owner=owner;self.semantic=source_binding(owner);self.cache=OrderedDict();self.checks=0;self.state_steps=0;self.hits=0
    def supports(self,left,right,domains,check=lambda:None,max_steps=200000):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Overlap cache source changed.')
        from types import SimpleNamespace
        from constraint_region_symbolic import join_diagrams
        if not set(left['cells'])&set(right['cells']):raise ValueError('Overlap consistency requires shared cells.')
        key=(left['cells'],right['cells'],tuple(domains),_digest(left),_digest(right))
        if key in self.cache:self.hits+=1;return self.cache[key]
        self.checks+=1
        supports,work=join_diagrams(self.owner,SimpleNamespace(**left),SimpleNamespace(**right),domains,check,max_steps)
        count=work['histories']
        self.state_steps+=work['steps'] if isinstance(work,dict) else work
        cells=tuple(sorted(set(left['cells'])|set(right['cells'])))
        check();result=(count,cells,tuple(supports[c] for c in cells));self.cache[key]=result
        while len(self.cache)>8:self.cache.popitem(last=False)
        return result
