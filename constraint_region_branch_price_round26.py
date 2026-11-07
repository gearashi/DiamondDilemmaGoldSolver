"""Restricted-master LP branch-and-price beneath the exact regional master.

GLOP only proposes primal/dual values. A negative requires rational Farkas
checking plus fresh exact pricing of every omitted path. Gold cuts condition
on occupied anchors, so a solution may legally avoid a candidate loop's ports.
No floating tolerance authorizes exclusion or validates a solution.
"""
from fractions import Fraction
import copy
from constraint_region_master import phase_one,check_farkas
from constraint_region_network import joined_signature
from constraint_pruning import values
from validator import reverse_mask

STAT_KEYS=('integer_bp_master_requests','integer_bp_native_builds','integer_bp_native_calls','integer_bp_native_unknown',
 'integer_bp_native_retries','integer_bp_pricing_queries','integer_bp_pricing_edges',
 'integer_bp_generated_columns','integer_bp_reused_columns','integer_bp_exact_replays',
 'integer_bp_certified_failures','integer_bp_fractional_nodes','integer_bp_integral_candidates',
 'integer_bp_connectivity_cuts','integer_bp_dual_reuses','integer_bp_completed_price_reuses')


from constraint_region_integer_round26 import price_diagram

def cut_coefficient(owner,cell,code,cut):
    """cross(W)-2*occupied(anchor_in)-2*occupied(anchor_out), additively."""
    ports,inside,outside=cut;ports=set(ports);value=0;occupied=set()
    for a,b in owner.pairs[code]:
        left=33*cell+a;right=33*cell+b;occupied.update((left,right))
        value+=int((left in ports)!=(right in ports))
    value-=2*int(inside in occupied)+2*int(outside in occupied)
    for a,sa,b,sb in owner.problem.edges:
        if cell!=min(a,b):continue
        for position in values(owner.masks[code][sa] if cell==a else reverse_mask(owner.masks[code][sb])):
            left=33*a+11*sa+position;right=33*b+11*sb+10-position
            value+=int((left in ports)!=(right in ports))
    return value


def loop_cut(owner,codes):
    """Generate an always-valid conditional cut from a verified multi-loop board."""
    report=owner.problem.validate(codes)
    if owner.target!='gold' or not report['unique_tiles'] or report['matched_edges']!=report['total_edges'] or report['line_components'] is None or report['line_components']<2:return None
    graph={}
    def edge(a,b):graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    for cell,code in enumerate(codes):
        for a,b in owner.pairs[code]:edge(33*cell+a,33*cell+b)
    for a,sa,b,sb in owner.problem.edges:
        for position in values(owner.masks[codes[a]][sa]):edge(33*a+11*sa+position,33*b+11*sb+10-position)
    start=min(graph);seen={start};queue=[start]
    for node in queue:
        for other in graph[node]-seen:seen.add(other);queue.append(other)
    outside=set(graph)-seen
    return (tuple(sorted(seen)),min(seen),min(outside)) if outside else None


def validate_cut(problem,cut):
    if type(cut)not in (tuple,list) or len(cut)!=3:raise ValueError('Invalid Gold boundary cut.')
    ports,inside,outside=cut
    if type(ports)not in (tuple,list) or not ports or len(ports)>33*problem.n or any(type(p)is not int or not 0<=p<33*problem.n for p in ports) or tuple(ports)!=tuple(sorted(set(ports))) or type(inside)is not int or type(outside)is not int or inside not in ports or outside in ports or not 0<=outside<33*problem.n:raise ValueError('Invalid conditional Gold anchors.')
    return tuple(ports),inside,outside


class BranchPriceNode:
    """One durable restricted master, with exact per-region pricing progress."""
    def __init__(self,owner,diagrams,domains,level,members,signature,fixed_codes,*,columns=(),cuts=(),dual_hint=None,max_rounds=24,max_columns=4096,time_ms=50,state=None):
        if any(type(v)is not int or v<0 for v in (max_rounds,max_columns,time_ms)):raise ValueError('Invalid bounded LP policy.')
        self.owner=owner;self.diagrams=diagrams;self.domains=tuple(domains);self.level=level;self.members=tuple(members);self.signature=signature;self.fixed=dict(fixed_codes)
        self.remaining=tuple(range(level,len(diagrams)));self.max_rounds=max_rounds;self.max_columns=max_columns;self.time_ms=time_ms
        self.stats=dict.fromkeys(STAT_KEYS,0);self.columns=[];self.cuts=[];self.seen=set();self.rounds=0
        self.phase='native';self.dual=None;self.primal=None;self.price_index=0;self.added=False;self.priced=[];self.pending_native=False
        self.complete=False;self.status='unknown';self.witness=None;self.certificate=None;self.hint_active=False
        self._build_rows()
        for cut in cuts:self._add_cut(cut)
        for region,path in columns:
            if self._accept_column(region,tuple(path)):self.stats['integer_bp_reused_columns']+=1
        if dual_hint:
            candidate=tuple(dual_hint.get(key,0) for key in self.rows)
            if any(type(v)is not int and not isinstance(v,Fraction) for v in candidate):raise ValueError('Invalid exact dual hint.')
            if any(candidate):
                self.dual=candidate;self.phase='pricing';self.hint_active=True;self.stats['integer_bp_dual_reuses']+=1
        if state is not None:self._restore(state)
    def _build_rows(self):
        self.rows={('region',r):i for i,r in enumerate(self.remaining)};self.rhs=[1]*len(self.remaining)
        used=0 if self.signature is None else self.signature[0]
        for tile in range(self.owner.problem.n):
            if not used>>tile&1:self.rows['tile',tile]=len(self.rhs);self.rhs.append(1)
        region_of={cell:r for r in self.remaining for cell in self.diagrams[r]['cells']}
        for eid,(a,sa,b,sb) in enumerate(self.owner.problem.edges):
            if a not in region_of or b not in region_of or region_of[a]==region_of[b]:continue
            patterns={self.owner.masks[v][sa] for v in values(self.domains[a])}|{reverse_mask(self.owner.masks[v][sb]) for v in values(self.domains[b])}
            for pattern in sorted(patterns):self.rows['seam',eid,pattern]=len(self.rhs);self.rhs.append(0)
    def _add_cut(self,cut):
        cut=validate_cut(self.owner.problem,cut)
        if cut in self.cuts:return False
        index=len(self.cuts);self.cuts.append(cut);self.rows['cut',index]=len(self.rhs)
        self.rhs.append(-2-sum(cut_coefficient(self.owner,c,v,cut) for c,v in self.fixed.items()))
        return True
    def _path_signature(self,region,path):
        if region not in self.remaining or len(path)!=len(self.diagrams[region]['cells']):return None
        node=self.diagrams[region]['root']
        for cell,code in zip(self.diagrams[region]['cells'],path):
            if type(code)is not int or not self.domains[cell]>>code&1 or node not in self.diagrams[region]['nodes']:return None
            edges=dict(self.diagrams[region]['nodes'][node][1]);node=edges.get(code)
            if node is None:return None
        return self.diagrams[region]['terminals'].get(node)
    def _accept_column(self,region,path):
        signature=self._path_signature(region,path)
        if signature is None:return False
        if self.signature is not None and joined_signature(self.owner,self.members,self.signature,self.diagrams[region]['cells'],signature,self.domains,lambda:None) is None:return False
        key=(region,path)
        if key in self.seen:return False
        self.seen.add(key);self.columns.append(key);return True
    def _arc(self,region,cell,code):
        row={}
        for tile in (code//3,):
            if ('tile',tile) in self.rows:row[self.rows['tile',tile]]=1
        for eid,(a,sa,b,sb) in enumerate(self.owner.problem.edges):
            if cell not in (a,b):continue
            if a in self.diagrams[region]['cells'] and b in self.diagrams[region]['cells']:continue
            mask=self.owner.masks[code][sa] if cell==a else reverse_mask(self.owner.masks[code][sb]);key=('seam',eid,mask)
            if key in self.rows:row[self.rows[key]]=1 if cell==a else -1
        for i,cut in enumerate(self.cuts):
            value=cut_coefficient(self.owner,cell,code,cut)
            if value:row[self.rows['cut',i]]=value
        return row
    def _column(self,region,path):
        column={self.rows['region',region]:1}
        for cell,code in zip(self.diagrams[region]['cells'],path):
            for row,value in self._arc(region,cell,code).items():column[row]=column.get(row,0)+value
        return {row:value for row,value in column.items() if value}
    def _columns(self):
        return [self._column(region,path) for region,path in self.columns]+[{self.rows['cut',i]:-1} for i in range(len(self.cuts))]
    def _price(self,region,dual,check):
        code_weights={}
        for cell in self.diagrams[region]['cells']:
            for code in values(self.domains[cell]):code_weights[cell,code]=sum(dual[row]*value for row,value in self._arc(region,cell,code).items())
        local={'integer_region_price_queries':0,'integer_region_price_edges':0,'integer_region_connectivity_rejections':0}
        try:
            answer=price_diagram(self.owner,self.diagrams[region],self.domains,(0,)*self.owner.problem.n,prefix_cells=self.members,prefix_signature=self.signature,check=check,stats=local,code_weights=code_weights)
            return None if answer is None else (dual[self.rows['region',region]]+answer[0],answer[1],answer[2])
        finally:
            self.stats['integer_bp_pricing_queries']+=local['integer_region_price_queries'];self.stats['integer_bp_pricing_edges']+=local['integer_region_price_edges']
    def _verify_negative(self,dual,check):
        if not check_farkas(self._columns(),tuple(self.rhs),dual):return False
        self.stats['integer_bp_exact_replays']+=1
        for region in self.remaining:
            result=self._price(region,dual,check)
            if result is not None and result[0]>0:return False
        return True
    def advance(self,check=lambda:None):
        check()
        if self.complete:return True
        if self.rounds>=self.max_rounds or len(self.columns)>self.max_columns or self.time_ms==0:self.complete=True;self.status='unknown';return True
        if self.phase=='native':
            self.stats['integer_bp_master_requests']+=1
            self.stats['integer_bp_native_retries']+=int(self.pending_native);self.pending_native=True
            telemetry={}
            def commit(result):
                self.pending_native=False;self.rounds+=1
                if result.get('status')!='optimal':self.stats['integer_bp_native_unknown']+=1;self.complete=True;return
                self.dual=tuple(Fraction(v) for v in result['dual']);self.primal=tuple(result['primal'])
                self.phase='pricing';self.price_index=0;self.added=False;self.priced=[];self.hint_active=False
            try:result=phase_one(self._columns(),tuple(self.rhs),check,time_ms=self.time_ms,telemetry=telemetry)
            except BaseException:
                if 'result' in telemetry:commit(telemetry['result'])
                else:self.pending_native=telemetry.get('calls',0)>0
                raise
            finally:
                self.stats['integer_bp_native_builds']+=telemetry.get('builds',0)
                self.stats['integer_bp_native_calls']+=telemetry.get('calls',0)
            commit(result);return self.complete
        if self.price_index<len(self.remaining):
            region=self.remaining[self.price_index];result=self._price(region,self.dual,check)
            if result is not None and result[0]>0 and self._accept_column(region,result[1]):
                self.stats['integer_bp_generated_columns']+=1;self.added=True
            self.priced.append(None if result is None else result[0]);self.price_index+=1;return False
        if self.added:self.phase='native';return False
        # Restricted-model status alone never closes this integer node.
        if self._verify_negative(self.dual,check):
            self.certificate={'dual':self.dual,'rhs':tuple(self.rhs),'rows':tuple(self.rows),'cuts':tuple(self.cuts)}
            self.stats['integer_bp_certified_failures']+=1;self.status='infeasible';self.complete=True;return True
        if self.hint_active:self.hint_active=False;self.phase='native';return False
        selected={}
        for index,(region,path) in enumerate(self.columns):
            value=self.primal[index] if self.primal is not None and index<len(self.primal) else 0
            if value>.5 and (region not in selected or value>selected[region][0]):selected[region]=(value,path)
        if len(selected)==len(self.remaining):
            codes=dict(self.fixed)
            for region,(_,path) in selected.items():codes.update(zip(self.diagrams[region]['cells'],path))
            if len(codes)==self.owner.problem.n:
                candidate=tuple(codes[c] for c in range(self.owner.problem.n));self.stats['integer_bp_integral_candidates']+=1
                report=self.owner.problem.validate(candidate)
                if report['valid'] or self.owner.target=='edges' and report['unique_tiles'] and report['matched_edges']==report['total_edges']:
                    check();self.witness=candidate;self.status='solved';self.complete=True;return True
                cut=loop_cut(self.owner,candidate)
                if cut is not None and self._add_cut(cut):
                    self.stats['integer_bp_connectivity_cuts']+=1;self.phase='native';self.dual=None;self.primal=None;return False
        self.stats['integer_bp_fractional_nodes']+=1;self.status='possible';self.complete=True;return True
    def snapshot(self):
        return copy.deepcopy({'columns':tuple(self.columns),'cuts':tuple(self.cuts),'rounds':self.rounds,'phase':self.phase,
           'dual':self.dual,'primal':None if self.primal is None else tuple(str(x) for x in self.primal),
           'price_index':self.price_index,'added':self.added,'priced':tuple(self.priced),'pending_native':self.pending_native,
           'hint_active':self.hint_active,'complete':self.complete,'status':self.status,'witness':self.witness,'certificate':self.certificate,'stats':dict(self.stats)})
    def _restore(self,state):
        state=copy.deepcopy(state)
        import math
        required={'columns','cuts','rounds','phase','dual','primal','price_index','added','priced','pending_native','hint_active','complete','status','witness','certificate','stats'}
        if type(state)is not dict or set(state)!=required:raise ValueError('Invalid branch-price cursor schema.')
        if type(state['rounds'])is not int or not 0<=state['rounds']<=self.max_rounds or type(state['price_index'])is not int or not 0<=state['price_index']<=len(self.remaining) or state['phase'] not in ('native','pricing') or state['status'] not in ('unknown','possible','infeasible','solved'):raise ValueError('Invalid branch-price status.')
        if any(type(state[k])is not bool for k in ('added','pending_native','hint_active','complete')) or set(state['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in state['stats'].values()):raise ValueError('Invalid branch-price counters.')
        self.columns=[];self.seen=set();self.cuts=[];self._build_rows()
        for cut in state['cuts']:
            if not self._add_cut(cut):raise ValueError('Duplicate saved Gold cut.')
        for region,path in state['columns']:
            if not self._accept_column(region,tuple(path)):raise ValueError('Invalid saved regional LP column.')
        if len(self.columns)>self.max_columns+len(self.remaining):raise ValueError('Saved LP column cap exceeded.')
        dual=state['dual']
        if dual is not None and (len(dual)!=len(self.rhs) or any(type(v)is not int and not isinstance(v,Fraction) for v in dual)):raise ValueError('Invalid saved rational dual.')
        if state['phase']=='pricing' and dual is None:raise ValueError('Pricing cursor has no complete native dual.')
        primal=None if state['primal'] is None else tuple(float(x) for x in state['primal'])
        if primal is not None and any(not math.isfinite(x) for x in primal):raise ValueError('Invalid saved native hint.')
        for key in ('rounds','phase','price_index','added','pending_native','hint_active','complete','status','witness','certificate','stats'):setattr(self,key,state[key])
        self.dual=dual;self.primal=primal;self.priced=list(state['priced'])
