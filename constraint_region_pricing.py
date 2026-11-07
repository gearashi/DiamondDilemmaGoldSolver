"""Gold/seam-aware rational regional column generation (round25 #23).

Local history columns carry inventory and complete interface class. Coupling
columns connect only compatible interfaces of a complete two-region partition.
Only exact pricing plus replayed rational Farkas inequalities proves failure;
fractional feasibility is never a Gold solution.
"""
from fractions import Fraction
from collections import OrderedDict
from constraint_region_incremental import source_binding
from constraint_region_symbolic import DirectRegionSession,DirectRegionDiagram
from constraint_region_network import joined_signature
from constraint_region_master import phase_one,check_farkas
from constraint_pruning import values
from constraint_separator import _Budget

STAT_KEYS=('gold_pricing_queries','gold_pricing_completed','gold_pricing_unknown','gold_pricing_admission_skips',
 'gold_pricing_cache_hits','gold_pricing_states','gold_pricing_transitions','gold_pricing_interfaces',
 'gold_pricing_compatibility_tests','gold_pricing_coupling_columns','gold_pricing_rounds','gold_pricing_columns',
 'gold_pricing_dag_edges','gold_pricing_exact_checks','gold_pricing_failures','gold_pricing_work_ticks')

class PricingLimit(Exception):pass
class _Work:
    def __init__(self,limit,check,stats):self.limit,self.check,self.stats=limit,check,stats;self.steps=0
    def tick(self):
        self.check();self.steps+=1;self.stats['gold_pricing_work_ticks']+=1
        if self.steps>self.limit:raise PricingLimit


def exact_interface_price(diagram,piece_weights,terminal_weights,check=lambda:None):
    if any(type(value)is not int and not isinstance(value,Fraction) for value in (*piece_weights.values(),*terminal_weights.values())):
        raise ValueError('Pricing requires exact rational weights.')
    memo={};edges=0
    def visit(node):
        nonlocal edges
        check()
        if node in memo:return memo[node]
        if node in diagram.terminals:
            signature=diagram.terminals[node];return Fraction(terminal_weights.get(signature[1:],0)),(),signature
        candidates=[]
        for code,child in diagram.nodes[node][1]:
            check();edges+=1;score,path,signature=visit(child)
            candidates.append((score+Fraction(piece_weights.get(code//3,0)),(code,*path),signature))
        answer=max(candidates,key=lambda row:(row[0],row[1]));memo[node]=answer;return answer
    if diagram.root is None:return None,(),None,edges
    score,path,signature=visit(diagram.root);return score,path,signature,edges


def gold_priced_master(owner,domains,diagrams,work,stats,*,max_rounds=32,max_columns=2048,time_ms=100):
    if len(diagrams)!=2 or set(diagrams[0].cells)&set(diagrams[1].cells) or set(diagrams[0].cells)|set(diagrams[1].cells)!=set(range(owner.problem.n)):
        raise ValueError('Gold pricing requires a complete two-region partition.')
    if time_ms==0:return {'status':'unknown'}
    n=owner.problem.n;classes=[tuple(sorted({sig[1:] for sig in d.terminals.values()})) for d in diagrams]
    stats['gold_pricing_interfaces']+=sum(map(len,classes));rows={};rhs=[1]*(2+n)
    for region,group in enumerate(classes):
        for interface in group:
            work.tick();rows[region,interface]=len(rhs);rhs.append(0)
    if len(rhs)>2048:raise PricingLimit
    columns=[];seen=set()
    for left in classes[0]:
        for right in classes[1]:
            work.tick();stats['gold_pricing_compatibility_tests']+=1
            if joined_signature(owner,diagrams[0].cells,(0,*left),diagrams[1].cells,(0,*right),domains,work.tick) is not None:
                columns.append({rows[0,left]:-1,rows[1,right]:-1});stats['gold_pricing_coupling_columns']+=1
                if len(columns)>max_columns:raise PricingLimit
    for iteration in range(max_rounds):
        work.tick();stats['gold_pricing_rounds']+=1
        candidate=phase_one(columns,tuple(rhs),work.tick,time_ms)
        if candidate.get('status')!='optimal':return {'status':'unknown'}
        y=candidate.get('dual')
        if not isinstance(y,(tuple,list)) or len(y)!=len(rhs) or any(type(v)is not int and not isinstance(v,Fraction) for v in y):return {'status':'unknown'}
        y=tuple(Fraction(v) for v in y);added=False
        for region,diagram in enumerate(diagrams):
            score,path,signature,edges=exact_interface_price(diagram,{t:y[2+t] for t in range(n)},
                {interface:y[rows[region,interface]] for interface in classes[region]},work.tick)
            stats['gold_pricing_dag_edges']+=edges
            if score is None or y[region]+score<=0:continue
            key=(region,signature)
            if key in seen:continue
            seen.add(key);inventory=signature[0]
            columns.append({region:1,rows[region,signature[1:]]:1,**{2+t:1 for t in values(inventory)}})
            stats['gold_pricing_columns']+=1;added=True
            if len(columns)>max_columns:raise PricingLimit
        if added:continue
        # A second exact DP replay checks all omitted history columns, not only
        # the restricted LP's currently generated subset.
        valid=check_farkas(columns,tuple(rhs),y);stats['gold_pricing_exact_checks']+=1
        if valid:
            for region,diagram in enumerate(diagrams):
                score,_,_,edges=exact_interface_price(diagram,{t:y[2+t] for t in range(n)},
                    {interface:y[rows[region,interface]] for interface in classes[region]},work.tick)
                stats['gold_pricing_dag_edges']+=edges
                if score is not None and y[region]+score>0:valid=False;break
        if valid:return {'status':'infeasible','certificate':tuple((v.numerator,v.denominator) for v in y),
                         'rhs':tuple(rhs),'interface_rows':tuple(rows),'columns':tuple(columns)}
        return {'status':'possible' if candidate.get('objective',1)<=0 else 'unknown'}
    return {'status':'unknown'}


class GoldPricingPruner:
    def __init__(self,owner):
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0);self.last_certificate=None
        self.original=tuple(sum(1<<code for code in set(row)) for row in self.problem.domains)
    def prune(self,domains,check=lambda:None,*,max_states=4096,max_steps=50000,max_rounds=32,max_columns=2048,time_ms=100):
        if source_binding(self.owner)!=self.binding:raise ValueError('Gold pricing source binding changed.')
        limits=(max_states,max_steps,max_rounds,max_columns,time_ms)
        if any(type(bound)is not int or bound<0 for bound in limits):raise ValueError('Invalid Gold pricing bounds.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Gold pricing domains escape the original root.')
        if not all(domains):return False
        if self.owner.target!='gold' or self.problem.n<2 or self.problem.n>12 and sum(mask.bit_count()>1 for mask in domains)>8:
            self.stats['gold_pricing_admission_skips']+=1;return False
        check();key=(tuple(domains),limits);self.last_certificate=None
        if key in self.cache:self.stats['gold_pricing_cache_hits']+=1;result=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['gold_pricing_queries']+=1;work=_Work(max_steps,check,self.stats);middle=self.problem.n//2;diagrams=[]
            try:
                for cells in (tuple(range(middle)),tuple(range(middle,self.problem.n))):
                    session=DirectRegionSession(self.owner,cells,domains,max_states,max_steps)
                    try:
                        while not session.advance(128,work.tick):pass
                    finally:
                        self.stats['gold_pricing_states']+=session.states;self.stats['gold_pricing_transitions']+=session.steps
                    diagram=object.__new__(DirectRegionDiagram);diagram.__dict__.update(session.result());diagrams.append(diagram)
                result=gold_priced_master(self.owner,domains,diagrams,work,self.stats,max_rounds=max_rounds,max_columns=max_columns,time_ms=time_ms)
            except (PricingLimit,_Budget):result={'status':'unknown'}
            check();self.cache[key]=result
            while len(self.cache)>8:self.cache.popitem(last=False)
            self.stats['gold_pricing_unknown' if result['status']=='unknown' else 'gold_pricing_completed']+=1
        if result['status']!='infeasible':return False
        check();self.last_certificate={'domains':tuple(domains),'binding':self.binding,**result};domains[0]=0
        self.stats['gold_pricing_failures']+=1;return True
