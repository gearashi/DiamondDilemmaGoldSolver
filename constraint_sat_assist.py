"""Optional incremental SAT assistance for exact DFS.

The oracle encodes original domain, seam and physical-tile constraints only.
UNSAT under placement assumptions proves a reusable forbidden conjunction.
SAT means edge feasibility only; it is never accepted as a Gold solution.
Partial model construction, limits and interruption cannot prove UNSAT.
"""
from collections import OrderedDict,defaultdict
import threading,time
from validator import reverse_mask
STAT_KEYS=('sat_assist_build_steps','sat_assist_clauses','sat_assist_calls','sat_assist_unsat',
           'sat_assist_unknown','sat_assist_cache_hits','sat_assist_core_literals_saved','sat_assist_subset_hits')
class SatAssist:
    def __init__(self,problem,initial,target='gold',*,factory=None):
        self.problem,self.initial,self.target=problem,tuple(initial),target
        if len(self.initial)!=problem.n or any(type(value)is not int or value<0 or value>>(3*problem.n) for value in self.initial):
            raise ValueError('SAT initial domains must be valid orientation bitsets.')
        self.solver=None;self.ready=False;self.reason=None;self.status='building';self.variables=[]
        self.ids=0;self.clauses=0;self.cache=OrderedDict();self.cores=OrderedDict();self.candidates=OrderedDict();self.last_outcome='not_ready';self.stats=dict.fromkeys(STAT_KEYS,0)
        if problem.n>160 or sum(value.bit_count() for value in self.initial)>80000:
            self.status='unsupported';self.reason='SAT assistance is capped at 160 cells and 80,000 placements.';self.pending=None
            return
        try:
            if factory is None:
                from pysat.solvers import Glucose4
                factory=Glucose4
            self.solver=factory();self.pending=self._build()
        except ImportError as exc:self.reason=str(exc);self.status='unavailable';self.pending=None
    def close(self):
        if self.solver is not None:self.solver.delete();self.solver=None
        self.ready=False
    def __del__(self):
        try:self.close()
        except Exception:pass
    def fresh(self):self.ids+=1;return self.ids
    def _exactly_one(self,literals):
        yield literals
        if len(literals)<2:return
        previous=self.fresh();yield [-literals[0],previous]
        for literal in literals[1:-1]:
            current=self.fresh();yield [-literal,current];yield [-previous,current];yield [-literal,-previous];previous=current
        yield [-literals[-1],-previous]
    def _build(self):
        from constraint_pruning import values
        for domain in self.initial:
            self.variables.append({code:self.fresh() for code in values(domain)})
            yield None
        for row in self.variables:yield from self._exactly_one(list(row.values()))
        by_tile=[[] for _ in range(self.problem.n)]
        for row in self.variables:
            for code,literal in row.items():by_tile[code//3].append(literal)
        for row in by_tile:yield from self._exactly_one(row)
        for a,sa,b,sb in self.problem.edges:
            left,right=defaultdict(list),defaultdict(list)
            for code,literal in self.variables[a].items():left[int(self.problem.masks[code,sa])].append(literal)
            for code,literal in self.variables[b].items():right[reverse_mask(int(self.problem.masks[code,sb]))].append(literal)
            common=left.keys()&right.keys()
            for mask in left.keys()-common:
                for lit in left[mask]:yield [-lit]
            for mask in right.keys()-common:
                for lit in right[mask]:yield [-lit]
            for mask in sorted(common):
                sig=self.fresh()
                for supports in (left[mask],right[mask]):
                    yield [-sig,*supports]
                    for lit in supports:yield [-lit,sig]
    def prepare(self,check=lambda:None,*,seconds=.01):
        if self.solver is None:return False
        if self.ready:return True
        deadline=time.monotonic()+max(0,seconds)
        while time.monotonic()<deadline:
            check();self.stats['sat_assist_build_steps']+=1
            try:clause=next(self.pending)
            except StopIteration:self.ready=True;self.status='ready';return True
            if clause is not None:
                if self.clauses>=1000000:
                    self.close();self.pending=None;self.status='unsupported';self.reason='SAT assistance exceeded its one-million-clause limit.'
                    return False
                self.solver.add_clause(clause);self.clauses+=1;self.stats['sat_assist_clauses']=self.clauses
        return False
    def drain_candidates(self):
        """Return private proved cores as UNTRUSTED publication candidates.

        A caller must re-prove each core under original problem domains before
        exporting it outside this helper's possibly restricted initial domains.
        """
        result=list(self.candidates);self.candidates.clear();return result
    def query(self,placements,check=lambda:None,*,seconds=.01,conflicts=256):
        """Return None for SAT/unknown, or a proved core (possibly empty)."""
        self.last_outcome='not_ready'
        if not self.ready:return None
        decisions=tuple(sorted(set(tuple(x) for x in placements)))
        if len({c for c,_ in decisions})!=len(decisions):raise ValueError('Conflicting SAT placement assumptions.')
        for cell,code in decisions:
            if type(cell)is not int or type(code)is not int or not 0<=cell<self.problem.n or code not in self.variables[cell]:
                raise ValueError('SAT assumption outside original domains.')
        check()
        if decisions in self.cache:
            self.stats['sat_assist_cache_hits']+=1;core=self.cache[decisions]
            self.last_outcome='sat' if core is None else 'unsat';return core
        chosen=frozenset(decisions)
        for core in self.cores:
            if frozenset(core)<=chosen:
                self.stats['sat_assist_subset_hits']+=1;self.last_outcome='unsat';return core
        self.last_outcome='unknown';self.stats['sat_assist_calls']+=1
        if seconds<=0:self.stats['sat_assist_unknown']+=1;return None
        assumptions=[self.variables[cell][code] for cell,code in decisions]
        decode={literal:pair for literal,pair in zip(assumptions,decisions)}
        done=threading.Event();interrupted=[];deadline=time.monotonic()+seconds
        def monitor():
            while not done.wait(.002):
                try:check()
                except BaseException as exc:interrupted.append(exc);self.solver.interrupt();return
                if time.monotonic()>=deadline:interrupted.append(None);self.solver.interrupt();return
        self.solver.clear_interrupt();self.solver.conf_budget(conflicts)
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        try:answer=self.solver.solve_limited(assumptions=assumptions,expect_interrupt=True)
        finally:done.set();watcher.join();self.solver.clear_interrupt()
        if interrupted:
            if interrupted[0] is not None:raise interrupted[0]
            self.stats['sat_assist_unknown']+=1;return None
        check()
        if answer is False:
            raw=self.solver.get_core() or []
            if any(lit not in decode for lit in raw):raise RuntimeError('SAT returned an unrelated assumption core.')
            core=tuple(sorted(decode[lit] for lit in raw));self.stats['sat_assist_unsat']+=1
            self.stats['sat_assist_core_literals_saved']+=len(decisions)-len(core)
            self.cache[decisions]=core;self.last_outcome='unsat'
            self.cores[core]=None;self.candidates[core]=None
            while len(self.cores)>64:self.cores.popitem(last=False)
            while len(self.candidates)>64:self.candidates.popitem(last=False)
        elif answer is True:self.cache[decisions]=None;core=None;self.last_outcome='sat'
        else:self.stats['sat_assist_unknown']+=1;return None
        while len(self.cache)>64:self.cache.popitem(last=False)
        return core
