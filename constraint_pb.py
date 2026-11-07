"""Checked bounded pseudo-Boolean learning and mixed obstruction discovery.

Rows have integer coefficients on placement indicators and mean sum <= rhs.
Primitive rows follow from cell ownership, physical-piece capacity, and seam
signatures. Positive combinations plus integer gcd rounding produce valid cuts.
Discovered clique/odd-cycle rules are verified against exact incompatibilities.
These are bounded rule families, not unrestricted automatic theorem discovery.
"""
from collections import OrderedDict,defaultdict,deque
from itertools import combinations
from math import gcd
from constraint_relations import bits
from validator import reverse_mask

STAT_KEYS=('pb_queries', 'pb_reuses', 'pb_rows', 'pb_derived', 'pb_pair_checks', 'pb_prunes', 'obstruction_rules', 'obstruction_checks')


def row(terms,rhs):
    merged=defaultdict(int)
    for cell,code,coefficient in terms:merged[cell,code]+=int(coefficient)
    merged={key:value for key,value in merged.items() if value}
    divisor=0
    for value in merged.values():divisor=gcd(divisor,abs(value))
    if divisor>1:merged={key:value//divisor for key,value in merged.items()};rhs//=divisor
    return tuple((cell,code,value) for (cell,code),value in sorted(merged.items())),int(rhs)


def combine(left,right,a,b):
    if type(a)is not int or type(b)is not int or a<=0 or b<=0:raise ValueError('Positive integer multipliers required.')
    return row([(c,v,k*a) for c,v,k in left[0]]+[(c,v,k*b) for c,v,k in right[0]],a*left[1]+b*right[1])


def derive_rows(base,check=lambda:None,*,max_pairs=128,max_new=64):
    rows=list(dict.fromkeys(base));known=set(rows);steps=[];examined=0
    # Newly learned cuts participate, but finite budgets keep every pass bounded.
    index=0
    while index<len(rows) and len(steps)<max_new and examined<max_pairs:
        left=rows[index];a_terms={(c,v):k for c,v,k in left[0]}
        for j in range(index):
            check();examined+=1
            if examined>max_pairs:break
            right=rows[j];b_terms={(c,v):k for c,v,k in right[0]}
            pivots=sorted(key for key in a_terms.keys()&b_terms.keys() if a_terms[key]*b_terms[key]<0)
            if not pivots:continue
            pivot=min(pivots,key=lambda key:(abs(a_terms[key])+abs(b_terms[key]),key))
            a,b=abs(b_terms[pivot]),abs(a_terms[pivot]);cut=combine(left,right,a,b)
            if not cut[0] and cut[1]>=0:continue
            if cut not in known:
                known.add(cut);rows.append(cut);steps.append((index,j,a,b,cut))
                if len(steps)>=max_new:break
        index+=1
    return tuple(rows),tuple(steps),min(examined,max_pairs)


def is_integer_row(value,*,max_bits=None):
    """Check canonical PB row types without coercing imported numbers.

    Python considers 1, 1.0 and True equal. A proof row must retain actual
    integers throughout later combinations and lower-bound evaluation.
    """
    if (type(value)is not tuple or len(value)!=2 or type(value[0])is not tuple
            or type(value[1])is not int):return False
    if max_bits is not None and value[1].bit_length()>max_bits:return False
    previous=None
    for term in value[0]:
        if (type(term)is not tuple or len(term)!=3 or any(type(x)is not int for x in term)
                or term[0]<0 or term[1]<0 or not term[2]):return False
        if max_bits is not None and any(x.bit_length()>max_bits for x in term):return False
        key=term[:2]
        if previous is not None and key<=previous:return False
        previous=key
    return True


def verify_derivation(base,steps,check=lambda:None):
    if not isinstance(base,(tuple,list)) or not isinstance(steps,(tuple,list)):return False
    try:
        for item in base:
            check()
            if not is_integer_row(item):return False
        rows=list(dict.fromkeys(base))
        for step in steps:
            check()
            if not isinstance(step,(tuple,list)) or len(step)!=5:return False
            left,right,a,b,claimed=step
            if type(left)is not int or type(right)is not int or not 0<=left<len(rows) or not 0<=right<len(rows):return False
            if not is_integer_row(claimed):return False
            expected=combine(rows[left],rows[right],a,b)
            if expected!=claimed:return False
            # Future ancestors are the independently computed exact row.
            rows.append(expected)
        return True
    except (ValueError,TypeError,IndexError,OverflowError):return False


def filter_rows(domains,rows,check=lambda:None,*,max_passes=4,reason=None):
    current=list(domains)
    for _ in range(max_passes):
        old=tuple(current)
        for row_index,(terms,rhs) in enumerate(rows):
            check();before=tuple(current) if reason is not None else None;by_cell=defaultdict(dict)
            for cell,code,coefficient in terms:by_cell[cell][code]=coefficient
            minima={cell:min((coefficients.get(code,0) for code in bits(current[cell])),default=0)
                    for cell,coefficients in by_cell.items()}
            total=sum(minima.values())
            if total>rhs:
                cell=next(iter(by_cell),0);current[cell]=0
                if reason is not None:reason(before,cell,before[cell],row_index)
                break
            for cell,coefficients in by_cell.items():
                lower=total-minima[cell]
                current[cell]&=sum(1<<code for code in bits(current[cell]) if lower+coefficients.get(code,0)<=rhs)
            if reason is not None:
                for cell,(old_mask,new_mask) in enumerate(zip(before,current)):
                    if old_mask!=new_mask:reason(before,cell,old_mask&~new_mask,row_index)
            if not all(current):break
        if not all(current) or tuple(current)==old:break
    check();changed=current!=domains;domains[:]=current;return changed


def primitive_rows(problem,scope):
    scopes=dict(scope);result=[];by_tile=defaultdict(list)
    for cell,mask in scope:
        terms=[(cell,code,1) for code in bits(mask)]
        result.extend((row(terms,1),row([(c,v,-k) for c,v,k in terms],-1)))
        for c,v,k in terms:by_tile[v//3].append((c,v,k))
    for terms in by_tile.values():
        result.append(row(terms,1))
        if len(scope)==problem.n:result.append(row([(c,v,-k) for c,v,k in terms],-1))
    for a,sa,b,sb in problem.edges:
        if a not in scopes or b not in scopes:continue
        left=defaultdict(list);right=defaultdict(list)
        for code in bits(scopes[a]):left[int(problem.masks[code,sa])].append((a,code,1))
        for code in bits(scopes[b]):right[reverse_mask(int(problem.masks[code,sb]))].append((b,code,-1))
        for mask in sorted(left.keys()|right.keys()):
            terms=left[mask]+right[mask]
            result.extend((row(terms,0),row([(c,v,-k) for c,v,k in terms],0)))
    return tuple(dict.fromkeys(result))


class ObstructionMiner:
    def __init__(self,problem):
        self.problem=problem;self.links=defaultdict(list)
        for a,sa,b,sb in problem.edges:
            self.links[a,b].append((sa,sb));self.links[b,a].append((sb,sa))

    def incompatible(self,left,right):
        a,am=left;b,bm=right
        if a==b:return not am&bm
        return all(x//3==y//3 or any(int(self.problem.masks[x,sa])!=reverse_mask(int(self.problem.masks[y,sb]))
                   for sa,sb in self.links[a,b]) for x in bits(am) for y in bits(bm))

    def verify(self,record):
        try:
            events=tuple(tuple(event) for event in record['events']);kind=record['kind']
            if len(set(events))!=len(events) or not events:return False
            for cell,mask in events:
                if type(cell)is not int or not 0<=cell<self.problem.n or type(mask)is not int or mask<=0:return False
                root=sum(1<<v for v in self.problem.domains[cell])
                if mask&~root:return False
            if kind=='clique':pairs=combinations(events,2);rhs=1
            elif kind=='odd_cycle' and len(events)>=5 and len(events)%2:
                pairs=zip(events,events[1:]+events[:1]);rhs=len(events)//2
            else:return False
            if not all(self.incompatible(a,b) for a,b in pairs):return False
            expected=row([(cell,code,1) for cell,mask in events for code in bits(mask)],rhs)
            return expected==record['row']
        except (TypeError,ValueError,KeyError,IndexError):return False

    def discover(self,scope,check=lambda:None,*,max_events=24,max_rules=32,max_walks=2048):
        groups_by_cell=[]
        for cell,mask in scope:
            grouped=defaultdict(int)
            for code in bits(mask):
                grouped[('tile',code//3)]|=1<<code
                for side in range(3):grouped[('side',side,int(self.problem.masks[code,side]))]|=1<<code
            groups_by_cell.append([(cell,group) for group in sorted(set(grouped.values()),key=lambda m:(-m.bit_count(),m)) if group!=mask])
        events=[]
        for index in range(max((len(row) for row in groups_by_cell),default=0)):
            for group in groups_by_cell:
                if index<len(group):events.append(group[index])
        events=tuple(events[:max_events]);adj=[set() for _ in events]
        for i,j in combinations(range(len(events)),2):
            check()
            if self.incompatible(events[i],events[j]):adj[i].add(j);adj[j].add(i)
        found=[];known=set();walks=0
        def publish(kind,indices):
            selected=tuple(events[i] for i in indices)
            rhs=1 if kind=='clique' else len(indices)//2
            cut=row([(c,code,1) for c,mask in selected for code in bits(mask)],rhs)
            if cut in known:return
            record={'kind':kind,'events':selected,'row':cut}
            if not self.verify(record):raise RuntimeError('Discovered obstruction failed independent verification.')
            known.add(cut);found.append(record)
        for i in range(len(events)):
            for j in sorted(v for v in adj[i] if v>i):
                for k in sorted(v for v in adj[i]&adj[j] if v>j):
                    check();publish('clique',(i,j,k))
                    if len(found)>=max_rules:return tuple(found)
        def walk(path):
            nonlocal walks
            check();walks+=1
            if walks>max_walks or len(found)>=max_rules:return
            if len(path)==5:
                if path[0] in adj[path[-1]] and path[1]<path[-1]:publish('odd_cycle',path)
                return
            for nxt in sorted(adj[path[-1]]):
                if nxt>path[0] and nxt not in path:walk((*path,nxt))
                if walks>max_walks or len(found)>=max_rules:return
        for start in range(len(events)):
            walk((start,))
            if walks>max_walks or len(found)>=max_rules:break
        return tuple(found)


class PBLearner:
    def __init__(self,problem,*,derive=True,discover=False,limit=32,indexed=False,explain=False):
        self.explain=bool(explain)
        if type(explain)is not bool or type(indexed)is not bool:raise ValueError('Invalid indexed PB policy.')
        from constraint_arithmetic_round26 import STAT_KEYS as QUEUE_KEYS
        self.indexed=indexed;self.queue_stats=dict.fromkeys(QUEUE_KEYS,0);self.pending=deque(maxlen=128)
        self.problem=problem;self.derive=derive;self.discover=discover;self.limit=limit
        self.records=OrderedDict();self.miner=ObstructionMiner(problem)
        self.stats=dict.fromkeys(('pb_queries','pb_reuses','pb_rows','pb_derived','pb_pair_checks',
                                 'pb_prunes','obstruction_rules','obstruction_checks'),0)

    def compile(self,scope,check=lambda:None):
        scope=tuple(sorted(scope))
        if scope in self.records:
            self.stats['pb_reuses']+=1;self.records.move_to_end(scope);return self.records[scope]
        self.stats['pb_queries']+=1;base=primitive_rows(self.problem,scope)
        rules=self.miner.discover(scope,check) if self.discover else ()
        self.stats['obstruction_checks']+=bool(self.discover)
        def dominated(cut):
            coefficients={(c,v):k for c,v,k in cut[0]}
            for terms,rhs in base:
                if rhs>cut[1] or any(k<0 for c,v,k in terms):continue
                other={(c,v):k for c,v,k in terms}
                if all(other.get(key,0)>=value for key,value in coefficients.items()):return True
            return False
        rules=tuple(record for record in rules if not dominated(record['row']))
        self.stats['obstruction_rules']+=len(rules)
        base=tuple(dict.fromkeys((*base,*(record['row'] for record in rules))))
        if self.derive:
            rows,steps,checks=derive_rows(base,check)
            if not verify_derivation(base,steps):raise RuntimeError('PB derivation failed replay.')
        else:rows,steps,checks=base,(),0
        check();record={'scope':scope,'rows':rows,'steps':steps,'rules':rules}
        record['proof_steps']=tuple({'parents':[a,b],'rule':'add','args':[x,y],'row':cut} for a,b,x,y,cut in steps)
        if self.indexed:
            from constraint_arithmetic_round26 import IndexedPBPropagation
            record['propagator']=IndexedPBPropagation(rows,stats=self.queue_stats)
            record['proof_steps']=tuple({'parents':[a,b],'rule':'add','args':[x,y],'row':cut} for a,b,x,y,cut in steps)
        self.records[scope]=record
        while len(self.records)>self.limit:self.records.popitem(last=False)
        self.stats['pb_rows']+=len(rows);self.stats['pb_derived']+=len(steps);self.stats['pb_pair_checks']+=checks
        return record

    def _apply(self,domains,record,check):
        if not self.indexed:
            def capture(before,cell,removed,index):
                for code in bits(removed):self.pending.append((before,cell,code,record,index))
            return filter_rows(domains,record['rows'],check,reason=capture if self.explain else None)
        worker=record['propagator']
        try:return worker.filter(domains,check)
        finally:
            while worker.reasons:
                reason=worker.reasons.popleft()
                for code in bits(reason['removed']):self.pending.append((reason['domains'],reason['cell'],code,record,reason['row']))

    def drain_deduction_chains(self,target='gold',limit=8):
        from constraint_arithmetic_round26 import export_chains
        return export_chains(self.problem,self.pending,target,limit)

    def prune(self,domains,check=lambda:None,*,max_cells=5,max_values=8):
        current=list(domains)
        for record in tuple(self.records.values()):
            if all(not current[c]&~mask for c,mask in record['scope']):self._apply(current,record,check)
            if not all(current):break
        eligible=sorted((c for c,d in enumerate(current) if 1<d.bit_count()<=max_values),key=lambda c:(current[c].bit_count(),c))
        if all(current) and len(eligible)>=2:
            selected=eligible[:max_cells];scope=tuple((c,current[c]) for c in selected)
            record=self.compile(scope,check);self._apply(current,record,check)
        check();self.stats['pb_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
