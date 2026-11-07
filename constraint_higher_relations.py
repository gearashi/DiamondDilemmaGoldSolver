"""Complete scoped n-ary relations from bounded projected SAT enumeration.

Temporary selector-guarded blocking clauses enumerate projection tuples without
constraining other assistant queries. A partial list is never an allowed table.
Only a final UNSAT query establishes completeness under the recorded domain box.
"""
from collections import OrderedDict
import math,time

STAT_KEYS=('higher_queries','higher_decisions','higher_models','higher_tables','higher_prunes',
           'higher_table_checks','higher_reuses','higher_caps','higher_unknown','higher_reserved_clauses',
           'higher_partial_queries','higher_partial_prunes','higher_partial_reuses')
MAX_GUARDED_CLAUSES=32768


class ProjectedRelations:
    def __init__(self,problem,initial,target='gold',*,limit=32,semijoins=False,guided=False,certificates=False,partial=False):
        self.problem,self.initial,self.target=problem,tuple(initial),target;self.limit=limit
        if type(limit)is not int or not 1<=limit<=128:raise ValueError('Invalid projected-table cache bound.')
        source=tuple(sum(1<<c for c in row) for row in problem.domains)
        if (target not in ('gold','edges') or len(self.initial)!=problem.n
                or any(type(m)is not int or m<=0 or m&~r for m,r in zip(self.initial,source))):raise ValueError('Invalid projection root.')
        self.binding=self._binding();self.tables=OrderedDict();self.sessions=OrderedDict();self.capped=OrderedDict()
        if type(semijoins)is not bool or type(guided)is not bool:raise ValueError('Invalid higher-relation policy.')
        self.semijoins=semijoins;self.guided=guided
        from constraint_relational_round26 import SemijoinNetwork
        self.relations=SemijoinNetwork(limit=limit)
        self.stats=dict.fromkeys(STAT_KEYS,0)
        self.table_witnesses={};self.table_gold={};self.certificates=None;self.certificate_attempted=set();self.partial=partial;self.partial_seen=OrderedDict()
        if type(certificates)is not bool or type(partial)is not bool:raise ValueError('Invalid relation lifecycle policy.')
        if certificates:
            from constraint_projection_certificate_round26 import ProjectionCertificates
            self.certificates=ProjectionCertificates(problem,initial,target)

    def _binding(self):
        return (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),
                tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))

    def _post(self,helper,clause):
        helper.solver.add_clause(clause);helper.clauses+=1;helper.stats['sat_assist_clauses']=helper.clauses

    def _drop(self,key):
        session=self.sessions.pop(key,None)
        if session is not None and session['helper'].solver is not None:
            self._post(session['helper'],[-session['guard']])

    def _cap(self,key,max_models):
        self._drop(key);self.capped[key,max_models]=None;self.stats['higher_caps']+=1
        while len(self.capped)>self.limit:self.capped.popitem(last=False)

    def close(self):
        for key in tuple(self.sessions):self._drop(key)
        self.tables.clear();self.capped.clear();self.table_witnesses.clear();self.table_gold.clear()
        if self.certificates is not None:self.certificates.close()

    def __del__(self):
        try:self.close()
        except Exception:pass

    def configure_cache_dir(self,directory,check=lambda:None):
        if self.certificates is None:return
        self.certificates.configure_cache_dir(directory)
        for key,rows,witnesses in self.certificates.load(check):
            self.tables[key]=rows;self.table_witnesses[key]=dict(zip(rows,witnesses));self.certificate_attempted.add(key)
        self._trim()

    def flush(self,check=lambda:None):
        if self.certificates is None:return
        for (scope,cells),session in tuple(self.sessions.items()):
            if not session.get('poisoned'):self.certificates.save_observed(scope,cells,session['witnesses'],check)
        self.persist_complete(check)

    def import_certificate(self,record,check=lambda:None):
        if self.certificates is None:raise ValueError('Relation certificates are disabled.')
        key,rows,witnesses=self.certificates.verify(record,check)
        self.tables[key]=rows;self.table_witnesses[key]=dict(zip(rows,witnesses));self.certificate_attempted.add(key);self._trim()
        return key

    def _trim(self):
        while len(self.tables)>self.limit:
            key,_=self.tables.popitem(last=False);self.table_witnesses.pop(key,None);self.table_gold.pop(key,None);self.certificate_attempted.discard(key)

    def persist_complete(self,check=lambda:None,*,seconds=.03):
        if self.certificates is None:return False
        for key,rows in tuple(self.tables.items()):
            if key in self.certificate_attempted or key not in self.table_witnesses:continue
            witnesses=self.table_witnesses[key]
            if any(row not in witnesses for row in rows):continue
            record=self.certificates.produce(*key,rows,[witnesses[row] for row in rows],self.table_gold.get(key,()),check,seconds=seconds)
            # A bounded unknown is not proof. It leaves the warm table usable
            # under its completed native query, but creates no disk authority.
            self.certificate_attempted.add(key)
            return False if record is None else self.certificates.save(record,check)
        return False

    def _partial_filter(self,domains,helper,check,seconds):
        if not self.partial or not all(domains):return False
        key=tuple(domains)
        if key in self.partial_seen:
            result=self.partial_seen[key];changed=tuple(domains)!=result;domains[:]=result;self.stats['higher_partial_reuses']+=1;return changed
        self.stats['higher_partial_queries']+=1
        # No enumeration selector is present: this is a separate ordinary
        # support query under the full domain box. Native selectors remain
        # existentially free and can be false, so blockers add no constraint.
        result=helper.filter_domains(domains,check,seconds=seconds,max_candidates=4)
        if result is None:return False
        result=tuple(result)
        if len(result)!=len(domains) or any(type(m)is not int or m<0 or m&~old for m,old in zip(result,domains)):raise ValueError('Partial relation support result widened domains.')
        changed=result!=tuple(domains);self.stats['higher_partial_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,result));domains[:]=result
        # A completed no-change call may merely have hit its cap. Keep it
        # eligible; the assistant retains its own supported/candidate cursor.
        if changed:
            self.partial_seen[key]=result
            while len(self.partial_seen)>self.limit:self.partial_seen.popitem(last=False)
        return changed

    def _filter(self,domains,check):
        if self.semijoins:
            before=tuple(domains);changed=self.relations.filter(self.tables,domains,check)
            self.stats['higher_prunes']+=sum((a&~b).bit_count() for a,b in zip(before,domains))
            return changed
        current=list(domains)
        for (scope,cells),rows in tuple(self.tables.items()):
            check()
            if any(mask&~old for mask,old in zip(current,scope)):continue
            self.stats['higher_reuses']+=1;supports=[0]*len(cells)
            for row in rows:
                check();self.stats['higher_table_checks']+=1
                if all(current[cell]>>code&1 for cell,code in zip(cells,row)):
                    for i,code in enumerate(row):supports[i]|=1<<code
            for cell,keep in zip(cells,supports):current[cell]&=keep
            if not all(current):break
        check();self.stats['higher_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed

    def prune(self,domains,helper,check=lambda:None,*,seconds=.01,max_calls=16,max_models=256,arity=3,max_values=12,projection_cells=None):
        if (self._binding()!=self.binding or helper.problem.fingerprint!=self.problem.fingerprint
                or helper.target!=self.target or tuple(helper.initial)!=self.initial):raise ValueError('Projection helper/root binding changed.')
        if (len(domains)!=len(self.initial) or any(type(m)is not int or m<0 or m&~r for m,r in zip(domains,self.initial))):raise ValueError('Projection domains escape root.')
        if (type(max_calls)is not int or max_calls<0 or type(max_models)is not int or max_models<0
                or type(arity)is not int or arity not in (range(3,9) if self.guided else (3,4)) or type(max_values)is not int or not 2<=max_values<=480
                or max_models>4096
                or isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0):raise ValueError('Invalid projection budget.')
        check();self.persist_complete(check);changed=self._filter(domains,check)
        if seconds and max_calls:changed=self._partial_filter(domains,helper,check,min(seconds,.01)) or changed
        if not all(domains) or not seconds or not max_calls:return changed
        if projection_cells is not None:
            if not self.guided:raise ValueError('Explicit projections require guided relation policy.')
            selected=tuple(projection_cells)
            if not 3<=len(selected)<=8 or len(set(selected))!=len(selected) or any(type(c)is not int or not 0<=c<len(domains) for c in selected):raise ValueError('Invalid explicit projection cells.')
        elif self.guided:
            from constraint_relational_round26 import guided_cells
            selected=guided_cells(domains,helper,arity,max_values,self.relations.stats,self.relations)
        else:selected=tuple(sorted((c for c,m in enumerate(domains) if 1<m.bit_count()<=max_values),key=lambda c:(domains[c].bit_count(),c))[:arity])
        if len(selected)<3:return changed
        key=(tuple(domains),selected)
        if key in self.tables or (key,max_models) in self.capped:return changed
        # Refining every narrower scope repeated more native proofs than it
        # saved in the finite comparison. Reuse a sound ancestor table here;
        # the ordinary assistant still checks any additional outside effects.
        if any((not self.guided or cells==selected) and all(not mask&~old for mask,old in zip(domains,scope)) for scope,cells in self.tables):
            return changed
        # A facade's private native owner uses wider placement IDs; the complete
        # job masks below remain explicit assumptions on every projection query.
        owner=getattr(helper,'owner',None);native=owner.helper if owner is not None else helper
        deadline=time.monotonic()+seconds
        if not native.prepare(check,seconds=max(0,deadline-time.monotonic())):return changed
        if not callable(getattr(native.solver,'get_model',None)):return changed
        session=self.sessions.get(key)
        if session is not None and (session['helper'] is not native or session.get('poisoned') or max_models>session['max_models']):self._drop(key);session=None
        if session is not None and len(session['rows'])>max_models:
            self._cap(key,max_models);return changed
        if session is None:
            reserved=getattr(native,'_higher_projection_budget_used',0)
            if reserved+max_models+2>MAX_GUARDED_CLAUSES:
                self._cap(key,max_models);return changed
            # Reserve lifetime capacity, including guard and retirement units.
            # Reservations are never refunded: the native owner may outlive jobs.
            native._higher_projection_budget_used=reserved+max_models+2
            self.stats['higher_reserved_clauses']+=max_models+2
            session={'helper':native,'guard':native.fresh(),'rows':set(),'witnesses':{},'poisoned':True,'max_models':max_models};self.sessions[key]=session
            first=selected[0]
            self._post(native,[-session['guard'],*[literal for code,literal in native.variables[first].items() if key[0][first]>>code&1]])
            if self.certificates is not None:
                observed=self.certificates.load_observed(*key,check)
                for row,witness in observed.items():
                    if len(session['rows'])>=max_models:break
                    if any(all(witness[c]==code for c,code in cut) for cut in native._gold_known):continue
                    self._post(native,[-session['guard'],*[-native.variables[cell][code] for cell,code in zip(selected,row)]])
                    session['rows'].add(row);session['witnesses'][row]=witness
            session['poisoned']=False
            while len(self.sessions)>8:self._drop(next(iter(self.sessions)))
        self.sessions.move_to_end(key)
        assumptions=native._domain_assumptions(key[0],check,deadline)
        if assumptions is None:return changed
        assumptions.append(session['guard'])
        for _ in range(max_calls):
            check()
            if time.monotonic()>=deadline:break
            before=int(native.solver.accum_stats().get('decisions',0))
            self.stats['higher_queries']+=1
            try:answer=native._domain_call(lambda:native.solver.solve_limited(assumptions=assumptions,expect_interrupt=True),check,deadline,conflicts=2000)
            finally:self.stats['higher_decisions']+=max(0,int(native.solver.accum_stats().get('decisions',0))-before)
            if answer is None:
                self.stats['higher_unknown']+=1;break
            if answer is False:
                rows=tuple(sorted(session['rows']));witnesses=dict(session['witnesses']);gold=tuple(sorted(native._gold_known));self._drop(key)
                self.tables[key]=rows;self.table_witnesses[key]=witnesses;self.table_gold[key]=gold;self.stats['higher_tables']+=1;self._trim()
                self.persist_complete(check)
                return self._filter(domains,check) or changed
            supports=native._model_supports(key[0],assumptions,check)
            if supports is None:raise RuntimeError('Projected model failed independent domain/seam/inventory validation.')
            codes=[mask.bit_length()-1 for mask in supports]
            row=tuple(codes[cell] for cell in selected)
            if row in session['rows']:raise RuntimeError('Native projection blocker was not respected.')
            if len(session['rows'])>=max_models:
                self._cap(key,max_models)
                return changed
            # No cancellation boundary separates a retained tuple and its blocker.
            session['poisoned']=True
            self._post(native,[-session['guard'],*[-native.variables[cell][code] for cell,code in zip(selected,row)]])
            session['rows'].add(row);session['witnesses'][row]=tuple(codes);self.stats['higher_models']+=1;session['poisoned']=False
        return changed
