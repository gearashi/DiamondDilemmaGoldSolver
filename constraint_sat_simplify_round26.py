"""Bounded, independently replayed CNF simplification with exact model lifting.

Only non-interface variables may disappear. Every eliminated positive/negative
clause pair contributes its complete resolvent. Caps select less simplification;
they never establish UNSAT or omit a required resolvent.
"""
from collections import defaultdict

STAT_KEYS=('simplify_inputs','simplify_literal_visits','simplify_resolvents','simplify_subsumption_checks',
 'simplify_equivalences','simplify_eliminations','simplify_clauses_removed','simplify_replay_steps',
 'simplify_model_clauses','simplify_caps','simplify_cache_hits','simplify_pinned_auxiliaries')


def canonical(clauses):
    if not isinstance(clauses,(list,tuple)):raise ValueError('Expected a finite CNF.')
    out=set()
    for clause in clauses:
        if not isinstance(clause,(tuple,list)) or any(type(v)is not int or not v or abs(v)>10000000 for v in clause):raise ValueError('Invalid CNF literal.')
        row=frozenset(clause)
        if any(-v in row for v in row):continue
        out.add(tuple(sorted(row,key=lambda v:(abs(v),v))))
    return tuple(sorted(out,key=lambda row:(len(row),row)))


def _replace(clauses,var,lit):
    return canonical([[lit if v==var else -lit if v==-var else v for v in row] for row in clauses])


def _resolve(clauses,var,check,stats):
    positive=[row for row in clauses if var in row];negative=[row for row in clauses if -var in row]
    rows=[row for row in clauses if var not in row and -var not in row]
    for left in positive:
        for right in negative:
            check();stats['simplify_resolvents']+=1
            rows.append(tuple(v for v in left if v!=var)+tuple(v for v in right if v!=-var))
    return canonical(rows),tuple(positive),tuple(negative)


def replay(clauses,protected,steps,check=lambda:None,*,stats=None):
    """Structural replay, independent of the producer's choice heuristics."""
    stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
    current=canonical(clauses);protected=frozenset(protected);inverse=[]
    if any(type(v)is not int or v<=0 for v in protected):raise ValueError('Invalid protected variables.')
    if not isinstance(steps,(tuple,list)) or len(steps)>4096:raise ValueError('Invalid simplification trace.')
    for step in steps:
        check();stats['simplify_replay_steps']+=1
        if not isinstance(step,(tuple,list)) or not step:raise ValueError('Invalid simplification step.')
        if step[0]=='equiv':
            if len(step)!=3:raise ValueError('Invalid equivalence.')
            var,lit=step[1:]
            if type(var)is not int or type(lit)is not int or var<=0 or not lit or var==abs(lit) or var in protected:raise ValueError('Protected or invalid equivalence.')
            required=canonical(((-var,lit),(var,-lit)))
            if any(row not in current for row in required):raise ValueError('Equivalence lacks both implication clauses.')
            current=_replace(current,var,lit);inverse.append(('equiv',var,lit))
        elif step[0]=='eliminate':
            if len(step)!=2 or type(step[1])is not int or step[1]<=0 or step[1] in protected:raise ValueError('Invalid elimination.')
            var=step[1];current,pos,neg=_resolve(current,var,check,stats);inverse.append(('eliminate',var,pos,neg))
        elif step[0]=='subsume':
            if len(step)!=3:raise ValueError('Invalid subsumption.')
            short,long=tuple(step[1]),tuple(step[2])
            if short not in current or long not in current or short==long or not set(short)<=set(long):raise ValueError('Subsumption lacks a retained premise.')
            current=tuple(row for row in current if row!=long)
        else:raise ValueError('Unknown simplification rule.')
    check();return current,tuple(inverse)


class SimplifiedCNF:
    def __init__(self,clauses,protected,check=lambda:None,*,max_variables=64,max_resolvents=4096,max_subsumption=1024,pin_eliminated=True,stats=None):
        for limit in (max_variables,max_resolvents,max_subsumption):
            if type(limit)is not int or not 0<=limit<=100000:raise ValueError('Invalid simplification bound.')
        if type(pin_eliminated)is not bool:raise ValueError('Invalid eliminated-variable pin policy.')
        self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        self.original=canonical(clauses);self.protected=frozenset(protected)
        if any(type(v)is not int or v<=0 for v in self.protected):raise ValueError('Invalid protected variables.')
        self.stats['simplify_inputs']+=1;current=self.original;steps=[];remaining=max_resolvents;changed=0
        # Direct binary equivalences are exact and cheap; chained substitution
        # remains reconstructible because inverse records are replayed backwards.
        while changed<max_variables:
            check();pairs=set(row for row in current if len(row)==2);chosen=None
            for left,right in sorted(pairs):
                self.stats['simplify_literal_visits']+=2
                if tuple(sorted((-left,-right),key=lambda v:(abs(v),v))) not in pairs:continue
                for var,lit in ((abs(left),-right if left>0 else right),(abs(right),-left if right>0 else left)):
                    if var not in self.protected and var!=abs(lit):chosen=(var,lit);break
                if chosen:break
            if not chosen:break
            var,lit=chosen;current=_replace(current,var,lit);steps.append(('equiv',var,lit));changed+=1;self.stats['simplify_equivalences']+=1
        # Positive/negative occurrence product gives an exact pre-admission work
        # bound; non-growing resolution limits formula expansion.
        while changed<max_variables:
            check();pos=defaultdict(int);neg=defaultdict(int)
            for row in current:
                for lit in row:
                    self.stats['simplify_literal_visits']+=1
                    if abs(lit) not in self.protected:(pos if lit>0 else neg)[abs(lit)]+=1
            choices=sorted((pos[v]*neg[v],v) for v in pos.keys()|neg.keys() if pos[v]*neg[v]<=pos[v]+neg[v] and pos[v]*neg[v]<=remaining)
            if not choices:break
            cost,var=choices[0];remaining-=cost
            current,_,_=_resolve(current,var,check,self.stats);steps.append(('eliminate',var));changed+=1;self.stats['simplify_eliminations']+=1
        checks=0
        for long in tuple(reversed(current)):
            if checks>=max_subsumption:break
            for short in current:
                if len(short)>=len(long) or checks>=max_subsumption:break
                check();checks+=1;self.stats['simplify_subsumption_checks']+=1
                if set(short)<=set(long):
                    current=tuple(row for row in current if row!=long);steps.append(('subsume',short,long));break
        if changed==max_variables or remaining==0 or checks==max_subsumption:self.stats['simplify_caps']+=1
        verified,inverse=replay(self.original,self.protected,steps,check,stats=self.stats)
        if verified!=current:raise RuntimeError('Simplification replay disagrees.')
        self.projection_clauses=current;self.steps=tuple(steps);self.inverse=inverse
        # Numeric SAT IDs allocate all intervening variables. Pin variables that
        # have ceased to occur so the native solver need not branch on padding.
        # These are projection-equivalent pins, NOT original-formula lemmas.
        # Callers importing original auxiliary lemmas must disable this policy.
        eliminated={record[1] for record in inverse}
        self.clauses=canonical([*current,*[(-var,) for var in sorted(eliminated)]]) if pin_eliminated else current
        self.stats['simplify_pinned_auxiliaries']+=len(eliminated) if pin_eliminated else 0
        self.stats['simplify_clauses_removed']+=len(self.original)-len(self.clauses)

    def reconstruct(self,model,check=lambda:None):
        if not isinstance(model,(tuple,list)) or any(type(v)is not int or not v for v in model):raise ValueError('Invalid native model.')
        values={}
        for lit in model:
            if abs(lit) in values and values[abs(lit)]!=(lit>0):raise ValueError('Conflicting native model values.')
            values[abs(lit)]=lit>0
        def satisfied(row):return any(values.get(abs(lit),False)==(lit>0) for lit in row)
        if any(not satisfied(row) for row in self.clauses):raise ValueError('Native model violates simplified formula.')
        for record in reversed(self.inverse):
            check();kind,var=record[:2]
            if kind=='equiv':lit=record[2];values[var]=values.get(abs(lit),False)==(lit>0)
            else:
                pos,neg=record[2:];need_true=any(not satisfied(tuple(v for v in row if v!=var)) for row in pos)
                need_false=any(not satisfied(tuple(v for v in row if v!=-var)) for row in neg)
                if need_true and need_false:raise ValueError('Eliminated model has incompatible requirements.')
                values[var]=need_true
        for row in self.original:
            check();self.stats['simplify_model_clauses']+=1
            if not satisfied(row):raise ValueError('Reconstructed model violates original CNF.')
        return tuple(var if value else -var for var,value in sorted(values.items()))
