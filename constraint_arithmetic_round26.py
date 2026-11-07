"""Exact checked PB rule extensions and resumable indexed row propagation."""
from collections import OrderedDict,defaultdict,deque
from itertools import combinations
from constraint_pb import row,is_integer_row,primitive_rows,combine,derive_rows,verify_derivation
from constraint_relations import bits

STAT_KEYS=('arithmetic_rule_checks','arithmetic_divisions','arithmetic_saturations','arithmetic_weakenings',
 'arithmetic_cover_assignments','arithmetic_covers','arithmetic_rows_kept','arithmetic_rows_tested',
 'arithmetic_seed_pair_checks','arithmetic_pair_checks','arithmetic_combination_terms','arithmetic_cell_minima','arithmetic_queue_reuses','arithmetic_context_reuses','arithmetic_prunes','arithmetic_compilations')


def normalized_literals(source):
    if not is_integer_row(source,max_bits=256):raise ValueError('Exact bounded PB row required.')
    terms,rhs=source;return tuple((c,v,1 if a>0 else -1,abs(a)) for c,v,a in terms),rhs-sum(a for _,_,a in terms if a<0)


def from_literals(terms,rhs):
    return row([(c,v,a*sign) for c,v,sign,a in terms],rhs-sum(a for c,v,sign,a in terms if sign<0))


def apply_rule(source,rule,args,check=lambda:None,stats=None):
    stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    if not is_integer_row(source,max_bits=256):raise ValueError('Exact bounded rule source required.')
    check();stats['arithmetic_rule_checks']+=1
    if rule=='divide':
        if type(args)is not int or not 1<=args<=65536:raise ValueError('Invalid PB divisor.')
        stats['arithmetic_divisions']+=1;return row([(c,v,a//args) for c,v,a in source[0]],source[1]//args)
    if rule=='weaken':
        if type(args)is not int or not 0<=args<len(source[0]):raise ValueError('Invalid weakening term.')
        c,v,a=source[0][args];stats['arithmetic_weakenings']+=1
        return row([term for i,term in enumerate(source[0]) if i!=args],source[1]+max(0,-a))
    literals,rhs=normalized_literals(source)
    if rule=='saturate':
        if args is not None:raise ValueError('Saturation has no parameters.')
        stats['arithmetic_saturations']+=1
        if rhs<0:return (),-1
        return from_literals([(c,v,sign,min(a,rhs+1)) for c,v,sign,a in literals],rhs)
    if rule!='cover' or type(args)is not dict or set(args)!={'cover','lift'}:raise ValueError('Unknown arithmetic rule.')
    if not 1<=len(literals)<=12 or rhs<0:raise ValueError('Lifted-cover source exceeds exact enumeration scope.')
    cover,lift=args['cover'],args['lift'];n=len(literals)
    if (not isinstance(cover,(tuple,list)) or not cover or not isinstance(lift,(tuple,list)) or
        any(type(v)is not int or not 0<=v<n for v in (*cover,*lift)) or len(set((*cover,*lift)))!=len(cover)+len(lift)):
        raise ValueError('Invalid cover/lifting order.')
    if sum(literals[i][3] for i in cover)<=rhs:raise ValueError('Claimed cover does not exceed capacity.')
    weights={i:1 for i in cover};bound=len(cover)-1
    for selected in lift:
        maximum=None
        for mask in range(1<<n):
            check();stats['arithmetic_cover_assignments']+=1
            if not mask>>selected&1:continue
            if sum(item[3] for i,item in enumerate(literals) if mask>>i&1)>rhs:continue
            value=sum(a for i,a in weights.items() if mask>>i&1);maximum=value if maximum is None else max(maximum,value)
        coefficient=bound+1 if maximum is None else bound-maximum
        if coefficient<0:raise ValueError('Invalid lifted-cover prefix.')
        if coefficient:weights[selected]=coefficient
    stats['arithmetic_covers']+=1
    return from_literals([(literals[i][0],literals[i][1],literals[i][2],a) for i,a in sorted(weights.items())],bound)


def replay(base,steps,check=lambda:None,stats=None):
    if not isinstance(base,(tuple,list)) or any(not is_integer_row(item,max_bits=256) for item in base):raise ValueError('Invalid arithmetic axioms.')
    if not isinstance(steps,(tuple,list)) or len(steps)>1024:raise ValueError('Arithmetic proof exceeds step cap.')
    stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    rows=list(dict.fromkeys(base))
    for step in steps:
        check()
        if type(step)is not dict or set(step)!={'parents','rule','args','row'}:raise ValueError('Invalid arithmetic proof step.')
        parents=step['parents']
        if not isinstance(parents,(tuple,list)) or not parents or any(type(i)is not int or not 0<=i<len(rows) for i in parents):raise ValueError('Invalid arithmetic ancestor.')
        if step['rule']=='add':
            if len(parents)!=2 or not isinstance(step['args'],(tuple,list)) or len(step['args'])!=2:raise ValueError('Invalid addition.')
            stats['arithmetic_rule_checks']+=1;stats['arithmetic_combination_terms']+=len(rows[parents[0]][0])+len(rows[parents[1]][0])
            result=combine(rows[parents[0]],rows[parents[1]],*step['args'])
        else:
            if len(parents)!=1:raise ValueError('Invalid unary arithmetic rule.')
            result=apply_rule(rows[parents[0]],step['rule'],step['args'],check,stats)
        claimed=step['row']
        if not is_integer_row(claimed,max_bits=256) or result!=claimed:raise ValueError('Arithmetic result differs from exact replay.')
        rows.append(result)
    return tuple(rows)


class IndexedPBPropagation:
    """An exact row queue; unfinished work is retained only under its input box."""
    def __init__(self,rows,*,limit=4,stats=None):
        if any(not is_integer_row(item,max_bits=256) for item in rows):raise ValueError('Exact PB rows required.')
        self.rows=tuple(rows);self.limit=limit;self.sessions=OrderedDict();self.reasons=deque(maxlen=128);self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        self.by_cell=defaultdict(set);self.terms=[]
        for index,(terms,rhs) in enumerate(self.rows):
            grouped=defaultdict(dict)
            for cell,code,a in terms:grouped[cell][code]=a;self.by_cell[cell].add(index)
            self.terms.append(dict(grouped))
    def filter(self,domains,check=lambda:None):
        key=tuple(domains)
        if not key or any(type(mask)is not int or mask<0 for mask in key):raise ValueError('Invalid arithmetic domains.')
        if any(cell>=len(key) for cell in self.by_cell):raise ValueError('PB row cell outside domains.')
        check();session=self.sessions.get(key)
        if session is None:
            ancestor=next((s for prior,s in reversed(self.sessions.items()) if s['done'] and all(not m&~old for m,old in zip(key,prior))),None)
            current=list(key);pending=set(range(len(self.rows)))
            if ancestor is not None:
                current=[a&b for a,b in zip(key,ancestor['current'])];changed=[c for c,(a,b) in enumerate(zip(current,ancestor['current'])) if a!=b]
                pending=set().union(*(self.by_cell[c] for c in changed)) if changed else set();self.stats['arithmetic_queue_reuses']+=len(self.rows)-len(pending)
            session={'current':current,'pending':deque(sorted(pending)),'queued':pending,'done':False};self.sessions[key]=session
            while len(self.sessions)>self.limit:self.sessions.popitem(last=False)
        else:self.sessions.move_to_end(key);self.stats['arithmetic_context_reuses']+=1
        while session['pending'] and all(session['current']):
            check();index=session['pending'][0];terms=self.terms[index];rhs=self.rows[index][1];current=session['current'];minima={}
            for cell,coefficients in terms.items():
                check();self.stats['arithmetic_cell_minima']+=1;minima[cell]=min((coefficients.get(code,0) for code in bits(current[cell])),default=0)
            total=sum(minima.values());updates={}
            if total>rhs:updates[next(iter(terms),0)]=0
            else:
                for cell,coefficients in terms.items():
                    check();keep=sum(1<<code for code in bits(current[cell]) if total-minima[cell]+coefficients.get(code,0)<=rhs)
                    if keep!=current[cell]:updates[cell]=keep
            check();session['pending'].popleft();session['queued'].remove(index);self.stats['arithmetic_rows_tested']+=1
            before=tuple(current)
            for cell,mask in updates.items():
                self.reasons.append({'domains':before,'cell':cell,'removed':current[cell]&~mask,'row':index})
                current[cell]=mask
                for affected in sorted(self.by_cell[cell]):
                    if affected not in session['queued']:session['queued'].add(affected);session['pending'].append(affected)
        session['done']=not session['pending'] or not all(session['current']);check()
        self.stats['arithmetic_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,session['current']));changed=list(domains)!=session['current'];domains[:]=session['current'];return changed


def export_chains(problem,pending,target='gold',limit=8):
    """Lease each proof until its consumer finishes; Stop retains the head."""
    from copy import deepcopy
    if target not in ('gold','edges') or type(limit)is not int or not 0<=limit<=128:raise ValueError('Invalid arithmetic proof export policy.')
    from constraint_deduction_chain_round26 import FORMAT
    emitted=0;original=[hex(sum(1<<code for code in row)) for row in problem.domains]
    while pending and emitted<limit:
        item=pending[0];domains,cell,code,record,index=item
        data={'scope':[[c,hex(mask)] for c,mask in record['scope']], 'steps':list(record.get('proof_steps',record['steps'])),'row_index':index}
        if record.get('rules'):data['obstructions']=list(record['rules'])
        yield deepcopy({'format':FORMAT,'fingerprint':problem.fingerprint,'target':target,'original':original,
          'root':[hex(mask) for mask in domains],'decisions':[[cell,hex(1<<code)]],'events':[],
          'terminal':{'rule':'pb','data':data}})
        # A close/exception at the yield never acknowledges this completed
        # producer record. Identity also avoids popping a later bounded entry.
        if pending and pending[0] is item:pending.popleft()
        emitted+=1


class ArithmeticPBLearner:
    def __init__(self,problem,*,limit=16):
        self.problem=problem;self.limit=limit;self.records=OrderedDict();self.pending=deque(maxlen=128);self.stats=dict.fromkeys(STAT_KEYS,0)
        self.binding=(problem.fingerprint,tuple(tuple(row) for row in problem.domains),tuple(tuple(map(int,row)) for row in problem.masks),tuple(problem.edges))
    def compile(self,scope,check=lambda:None,*,max_pairs=128,max_new=64):
        scope=tuple(scope)
        if not scope or len({c for c,m in scope})!=len(scope) or any(type(c)is not int or not 0<=c<self.problem.n or type(m)is not int or m<=0 or m&~sum(1<<v for v in self.problem.domains[c]) for c,m in scope):raise ValueError('Invalid arithmetic scope.')
        if type(max_pairs)is not int or not 0<=max_pairs<=4096 or type(max_new)is not int or not 0<=max_new<=128:raise ValueError('Invalid arithmetic compilation bound.')
        key=(scope,max_pairs,max_new)
        if key in self.records:self.records.move_to_end(key);self.stats['arithmetic_context_reuses']+=1;return self.records[key]
        base=primitive_rows(self.problem,scope);rows,old_steps,seed_checks=derive_rows(base,check,max_pairs=max_pairs,max_new=max_new)
        self.stats['arithmetic_seed_pair_checks']+=seed_checks
        self.stats['arithmetic_combination_terms']+=sum(len(rows[a][0])+len(rows[b][0]) for a,b,x,y,cut in old_steps)
        if not verify_derivation(base,old_steps,check):raise RuntimeError('PB seed derivation failed replay.')
        steps=[{'parents':[a,b],'rule':'add','args':[x,y],'row':cut} for a,b,x,y,cut in old_steps]
        known=set(rows);derived=list(rows)
        for index,source in enumerate(rows):
            check();operations=[('saturate',None)]
            divisors=sorted({abs(a) for _,_,a in source[0] if 1<abs(a)<=16}|{2,3})
            operations.extend(('divide',d) for d in divisors)
            literals,rhs=normalized_literals(source)
            if 1<=len(literals)<=8 and rhs>=0:
                cover=[];total=0
                for i in sorted(range(len(literals)),key=lambda i:(-literals[i][3],i)):
                    cover.append(i);total+=literals[i][3]
                    if total>rhs:break
                if total>rhs:operations.append(('cover',{'cover':cover,'lift':[i for i in range(len(literals)) if i not in cover]}))
            for rule,args in operations:
                cut=apply_rule(source,rule,args,check,self.stats)
                if cut in known or not cut[0] and cut[1]>=0:continue
                known.add(cut);derived.append(cut);steps.append({'parents':[index],'rule':rule,'args':args,'row':cut})
                if len(steps)>=max_new+128:break
            if len(steps)>=max_new+128:break
        # New rounded/cover consequences participate in further cancellation;
        # otherwise unary bound filtering on their parent row already sees many
        # of their individual exclusions. Preserve every new ancestor explicitly.
        seed_end=len(derived);examined=0
        for left in (range(len(rows),seed_end) if max_pairs else ()):
            a_terms={(c,v):a for c,v,a in derived[left][0]}
            for right in range(left):
                check();self.stats['arithmetic_pair_checks']+=1;b_terms={(c,v):a for c,v,a in derived[right][0]}
                pivots=[key for key in a_terms.keys()&b_terms.keys() if a_terms[key]*b_terms[key]<0]
                if not pivots:continue
                pivot=min(pivots,key=lambda key:(abs(a_terms[key])+abs(b_terms[key]),key))
                a,b=abs(b_terms[pivot]),abs(a_terms[pivot]);cut=combine(derived[left],derived[right],a,b);examined+=1;self.stats['arithmetic_combination_terms']+=len(derived[left][0])+len(derived[right][0])
                if cut not in known and (cut[0] or cut[1]<0) and is_integer_row(cut,max_bits=256):
                    known.add(cut);derived.append(cut);steps.append({'parents':[left,right],'rule':'add','args':[a,b],'row':cut})
                if examined>=max_pairs or len(steps)>=max_new+256:break
            if examined>=max_pairs or len(steps)>=max_new+256:break
        checked=replay(base,steps,check,self.stats)
        if checked!=tuple(derived):raise RuntimeError('Advanced arithmetic replay differs.')
        check();record={'scope':scope,'base':base,'steps':tuple(steps),'rows':checked,'propagator':IndexedPBPropagation(checked,stats=self.stats)}
        self.records[key]=record;self.stats['arithmetic_compilations']+=1;self.stats['arithmetic_rows_kept']+=len(checked)
        while len(self.records)>self.limit:self.records.popitem(last=False)
        return record
    def _collect(self,record):
        worker=record['propagator']
        while worker.reasons:
            reason=worker.reasons.popleft()
            for code in bits(reason['removed']):
                self.pending.append((reason['domains'],reason['cell'],code,record,reason['row']))

    def drain_deduction_chains(self,target='gold',limit=8):
        return export_chains(self.problem,self.pending,target,limit)

    def prune(self,domains,check=lambda:None,*,max_cells=5,max_values=8):
        if type(max_cells)is not int or not 2<=max_cells<=160 or type(max_values)is not int or not 2<=max_values<=480:raise ValueError('Invalid arithmetic scope bounds.')
        binding=(self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))
        if binding!=self.binding:raise ValueError('Arithmetic source changed.')
        original=[sum(1<<v for v in row) for row in self.problem.domains]
        if len(domains)!=len(original) or any(type(m)is not int or m<0 or m&~r for m,r in zip(domains,original)):raise ValueError('Arithmetic domains escape input.')
        current=list(domains)
        for record in tuple(self.records.values()):
            if all(not current[c]&~m for c,m in record['scope']):
                record['propagator'].filter(current,check);self._collect(record)
            if not all(current):break
        eligible=sorted((c for c,m in enumerate(current) if 1<m.bit_count()<=max_values),key=lambda c:(current[c].bit_count(),c))
        if all(current) and len(eligible)>=2:
            record=self.compile(tuple((c,current[c]) for c in eligible[:max_cells]),check);record['propagator'].filter(current,check);self._collect(record)
        check()
        if (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))!=self.binding:raise ValueError('Arithmetic source changed during propagation.')
        changed=current!=domains;domains[:]=current;return changed
