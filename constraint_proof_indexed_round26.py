"""Explicit-state watched RUP checking and optional ordered proof hints.

Public proof input is always independently checked. Local cursor snapshots are
ONLY trusted local continuation state, never certificates: source/base/proof/
context bindings and a digest detect accidental changes, not malicious edits.
Every cancellation point is before a complete atomic checker microstep.
"""
from collections import defaultdict
import copy,hashlib,json
from pathlib import Path
from constraint_proof import _clause,_hash,ProofLimit,MAX_PROOF_BYTES,MAX_PROOF_STEPS
from constraint_proof_cursor import _ValidatedRUPBase

SOURCE_FILES=tuple(str(Path(__file__).with_name(name)) for name in
    ('constraint_proof_indexed_round26.py','constraint_proof.py','constraint_proof_cursor.py','constraint_proof_reuse_round26.py'))
SOURCE={Path(path).name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCE_FILES}
MAX_LOCAL_BYTES=32*1024*1024
STATS=('proof_steps','proof_clause_visits','proof_propagations','proof_resumptions',
       'proof_index_build_literals','proof_watch_moves','proof_hint_visits','proof_microsteps','proof_shared_steps')


def parse_line(raw):
    words=raw.split()
    if not words:return None,False
    deleted=words[0]=='d'
    if deleted:words=words[1:]
    try:values=[int(x) for x in words]
    except ValueError:raise ValueError('Noninteger RUP literal.') from None
    if not values or values[-1]!=0 or 0 in values[:-1]:raise ValueError('Missing RUP terminator.')
    return _clause(values[:-1]),deleted


class IndexedRUPChecker:
    def __init__(self,clauses,proof,*,extra_clauses=(),hints=None,context=None,
                 max_steps=MAX_PROOF_STEPS,max_visits=2000000,reuse=False):
        if type(max_steps)is not int or max_steps<0 or type(max_visits)is not int or max_visits<0:raise ValueError('Invalid proof budget.')
        if (not isinstance(proof,(list,tuple)) or len(proof)>max_steps
                or any(type(line)is not str or len(line)>16384 for line in proof)
                or sum(map(len,proof))>MAX_PROOF_BYTES):raise ValueError('Malformed or oversized proof.')
        base=clauses if type(clauses)is _ValidatedRUPBase else _ValidatedRUPBase(clauses)
        self.base=base.clauses;self.extra_clauses=tuple(_clause(row) for row in extra_clauses)
        self.proof=tuple(proof);self.max_visits=max_visits;self.max_steps=max_steps
        if hints is None:hints=[[] for _ in proof]
        if (not isinstance(hints,(list,tuple)) or len(hints)!=len(proof)
                or any(not isinstance(row,(list,tuple)) or len(row)>100000
                       or any(type(x)is not int or x<0 for x in row) for row in hints)
                or sum(map(len,hints))>1000000):raise ValueError('Malformed ordered clause hints.')
        self.hints=tuple(tuple(row) for row in hints)
        self.context=json.loads(json.dumps(context,sort_keys=True,allow_nan=False))
        self.binding=_hash({'base':self.base,'extra':self.extra_clauses,'proof':self.proof,'hints':self.hints,
                            'context':self.context,'source':SOURCE,'max_visits':max_visits,'max_steps':max_steps})
        self.active=list(self.base+self.extra_clauses);self.positions=[[-1,-1] for _ in self.active]
        self.watchers=defaultdict(list);self.units=[];self.index_cursor=0
        self.proof_cursor=0;self.phase='build';self.candidate=None;self.generated_hints=[]
        self.assigned={};self.queue=[];self.event_cursor=0;self.event=None;self.watch_cursor=0
        self.unit_cursor=0;self.hint_cursor=0;self.reason_ids=[]
        self.finished=False;self.result=None;self.limited=False;self.broken=False;self.seen_empty=False
        self.stats=dict.fromkeys(STATS,0)
        if type(reuse)is not bool:raise ValueError('Invalid proof-prefix policy.')
        self.reuse=reuse;self.prefix_binding=None
        if reuse:
            from constraint_proof_reuse_round26 import scope,lookup
            self.prefix_binding=scope(self.base,self.extra_clauses,SOURCE);known=lookup(self.prefix_binding,self.proof)
            if known is not None:
                prefix,clauses,hints,empty,_=known;self.proof_cursor=len(prefix);self.active.extend(clauses);self.positions.extend([[-1,-1] for _ in clauses]);self.generated_hints=list(hints);self.seen_empty=empty;self.stats['proof_shared_steps']=len(clauses)

    def _source_check(self):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE[Path(path).name] for path in SOURCE_FILES):
            raise ValueError('Checker source changed during the process.')

    def _index(self,index):
        row=self.active[index];self.stats['proof_index_build_literals']+=len(row)
        if len(row)<2:self.units.append(index)
        else:
            self.positions[index]=[0,1]
            self.watchers[row[0]].append(index);self.watchers[row[1]].append(index)
        self.index_cursor=index+1

    def _value(self,lit):
        value=self.assigned.get(abs(lit))
        return None if value is None else value==(lit>0)

    def _assign(self,lit):
        old=self.assigned.get(abs(lit))
        if old is not None:return old==(lit>0)
        self.assigned[abs(lit)]=lit>0;self.queue.append(lit);return True

    def _visit(self):
        if self.stats['proof_clause_visits']>=self.max_visits:raise ProofLimit('Indexed RUP work cap reached.')
        self.stats['proof_clause_visits']+=1

    def _commit(self):
        self.active.append(self.candidate);self.positions.append([-1,-1]);self._index(len(self.active)-1)
        self.stats['proof_steps']+=1;self.generated_hints.append(tuple(self.reason_ids))
        if not self.candidate:self.seen_empty=True
        self.proof_cursor+=1;self.candidate=None;self.phase='proof'
        self.assigned={};self.queue=[];self.event_cursor=0;self.event=None;self.watch_cursor=0
        self.unit_cursor=0;self.hint_cursor=0;self.reason_ids=[]
        if self.reuse and (self.proof_cursor%64==0 or self.proof_cursor==len(self.proof) or self.seen_empty):self._remember_prefix()

    def _remember_prefix(self):
        from constraint_proof_reuse_round26 import remember
        remember(self.prefix_binding,self.proof[:self.proof_cursor],self.active[len(self.base)+len(self.extra_clauses):],self.generated_hints,self.seen_empty)

    def _evaluate_hint(self,index):
        if index>=len(self.active):raise ValueError('Hint references an unproved clause.')
        self._visit();row=self.active[index];missing=[]
        for lit in row:
            value=self._value(lit)
            if value is True:return
            if value is None:missing.append(lit)
        if not missing:
            self.reason_ids.append(index);self._commit()
        elif len(missing)==1:
            self._assign(missing[0]);self.reason_ids.append(index);self.stats['proof_propagations']+=1

    def _step(self):
        if self.phase=='build':
            if self.index_cursor<len(self.active):self._index(self.index_cursor)
            else:self.phase='proof'
            return
        if self.phase=='proof':
            if self.proof_cursor==len(self.proof):
                self.finished=True;self.result=self.seen_empty;return
            row,deletion=parse_line(self.proof[self.proof_cursor])
            if row is None or deletion:
                self.proof_cursor+=1;self.generated_hints.append(());return
            self.candidate=row;self.phase='hints' if self.hints[self.proof_cursor] else 'units'
            for lit in row:
                if not self._assign(-lit):self._commit();break
            return
        if self.phase=='hints':
            hints=self.hints[self.proof_cursor]
            if self.hint_cursor==len(hints):self.phase='units';return
            index=hints[self.hint_cursor];self._evaluate_hint(index)
            if self.phase=='hints':self.hint_cursor+=1
            self.stats['proof_hint_visits']+=1;return
        if self.phase=='units':
            if self.unit_cursor==len(self.units):self.phase='watch';return
            index=self.units[self.unit_cursor];self._evaluate_hint(index)
            if self.phase=='units':self.unit_cursor+=1
            return
        if self.phase!='watch':raise ValueError('Invalid checker phase.')
        if self.event is None:
            if self.event_cursor==len(self.queue):
                self.finished=True;self.result=False;return
            self.event=-self.queue[self.event_cursor];self.event_cursor+=1;self.watch_cursor=0
        watching=self.watchers[self.event]
        if self.watch_cursor==len(watching):self.event=None;return
        self._visit();index=watching[self.watch_cursor];row=self.active[index];positions=self.positions[index]
        side=0 if row[positions[0]]==self.event else 1
        if row[positions[side]]!=self.event:raise ValueError('Watch index lost its clause.')
        other=row[positions[1-side]]
        if self._value(other) is True:self.watch_cursor+=1;return
        replacement=next((k for k,lit in enumerate(row) if k not in positions and self._value(lit) is not False),None)
        if replacement is not None:
            positions[side]=replacement;watching.pop(self.watch_cursor);self.watchers[row[replacement]].append(index)
            self.stats['proof_watch_moves']+=1;return
        if self._value(other) is False:
            self.reason_ids.append(index);self._commit();return
        self._assign(other);self.reason_ids.append(index);self.stats['proof_propagations']+=1;self.watch_cursor+=1

    def advance(self,check=lambda:None,*,ticks=256):
        if type(ticks)is not int or ticks<0:raise ValueError('Invalid checker slice.')
        if self.broken:raise RuntimeError('An incomplete checker mutation cannot resume.')
        if self.finished:return self.result
        self.stats['proof_resumptions']+=1
        for _ in range(ticks):
            check();self.broken=True
            try:self._step()
            except ProofLimit:self.finished=True;self.limited=True
            except ValueError:self.finished=True;self.result=False
            except BaseException:raise
            else:self.stats['proof_microsteps']+=1
            self.broken=False
            if self.finished:return self.result
        return None

    def snapshot(self):
        """Local continuation only; callers must never offer this as a proof."""
        self._source_check()
        if self.broken:raise RuntimeError('Cannot save a broken checker mutation.')
        state={key:copy.deepcopy(getattr(self,key)) for key in ('positions','index_cursor','proof_cursor','phase','candidate',
            'queue','event_cursor','event','watch_cursor','unit_cursor','hint_cursor','reason_ids',
            'finished','result','limited','seen_empty','stats','generated_hints')}
        state['candidate']=None if self.candidate is None else list(self.candidate)
        state['generated_hints']=[list(row) for row in self.generated_hints]
        state['assigned']=[[k,v] for k,v in sorted(self.assigned.items())]
        state['watchers']=[[lit,list(rows)] for lit,rows in sorted(self.watchers.items()) if rows]
        out={'format':'diamond-local-watched-rup-v1','authority':'trusted_local_continuation_not_a_proof',
             'binding':self.binding,'state':state};out['digest']=_hash(out)
        if len(json.dumps(out,separators=(',',':')).encode())>MAX_LOCAL_BYTES:raise ProofLimit('Local checker snapshot byte cap.')
        return out

    @classmethod
    def restore_trusted_local(cls,clauses,proof,payload,**kwargs):
        """Restore an application-owned local cache, never arbitrary proof input.

        The previously checked prefix is trusted exactly like the application's
        local search stack. A rehashed malicious state is outside this contract;
        public proof import always constructs a new checker instead.
        """
        obj=cls(clauses,proof,**kwargs);obj._source_check()
        if (not isinstance(payload,dict) or set(payload)!={'format','authority','binding','state','digest'}
                or payload['format']!='diamond-local-watched-rup-v1'
                or payload['authority']!='trusted_local_continuation_not_a_proof'
                or payload['binding']!=obj.binding or payload['digest']!=_hash({k:v for k,v in payload.items() if k!='digest'})
                or len(json.dumps(payload,separators=(',',':')).encode())>MAX_LOCAL_BYTES):raise ValueError('Local proof cursor identity/integrity differs.')
        s=copy.deepcopy(payload['state']);expected=set(obj.snapshot()['state'])
        if not isinstance(s,dict) or set(s)!=expected:raise ValueError('Malformed local checker state.')
        for key in ('index_cursor','proof_cursor','event_cursor','watch_cursor','unit_cursor','hint_cursor'):
            if type(s[key])is not int or s[key]<0:raise ValueError('Malformed local checker cursor.')
        if not 0<=s['proof_cursor']<=len(obj.proof):raise ValueError('Proof cursor exceeds input.')
        obj.active=list(obj.base+obj.extra_clauses)
        for raw in obj.proof[:s['proof_cursor']]:
            row,deletion=parse_line(raw)
            if row is not None and not deletion:obj.active.append(row)
        if not 0<=s['index_cursor']<=len(obj.active):raise ValueError('Index cursor exceeds formula.')
        if s['phase'] not in ('build','proof','hints','units','watch'):raise ValueError('Malformed local checker phase.')
        if s['phase']!='build' and s['index_cursor']!=len(obj.active):raise ValueError('Unindexed proof formula.')
        if s['phase']=='build' and s['candidate'] is not None:raise ValueError('Proof began before indexing.')
        if type(s['positions'])is not list or len(s['positions'])!=len(obj.active):raise ValueError('Malformed watch positions.')
        expected_watches=defaultdict(list);obj.units=[]
        for index,(row,positions) in enumerate(zip(obj.active,s['positions'])):
            if type(positions)is not list or len(positions)!=2 or any(type(v)is not int for v in positions):raise ValueError('Invalid watch positions.')
            if index>=s['index_cursor'] or len(row)<2:
                if positions!=[-1,-1]:raise ValueError('Unexpected watches.')
                if index<s['index_cursor']:obj.units.append(index)
            else:
                if len(set(positions))!=2 or any(not 0<=v<len(row) for v in positions):raise ValueError('Invalid watched offsets.')
                for pos in positions:expected_watches[row[pos]].append(index)
        if type(s['watchers'])is not list:raise ValueError('Invalid watch lists.')
        watchers={}
        for item in s['watchers']:
            if (type(item)is not list or len(item)!=2 or type(item[0])is not int or not item[0] or item[0] in watchers
                    or type(item[1])is not list or any(type(v)is not int for v in item[1])):raise ValueError('Malformed watch list.')
            watchers[item[0]]=item[1]
        if {k:sorted(v) for k,v in watchers.items()}!={k:sorted(v) for k,v in expected_watches.items()}:raise ValueError('Watch lists do not cover the indexed formula.')
        assigned={}
        for item in s['assigned']:
            if (type(item)is not list or len(item)!=2 or type(item[0])is not int or item[0]<=0
                    or type(item[1])is not bool or item[0] in assigned):raise ValueError('Invalid local assignment.')
            assigned[item[0]]=item[1]
        if (type(s['queue'])is not list or any(type(lit)is not int or not lit for lit in s['queue'])
                or len(s['queue'])!=len(assigned) or {abs(v):v>0 for v in s['queue']}!=assigned):raise ValueError('Malformed propagation trail.')
        if s['event_cursor']>len(s['queue']) or s['unit_cursor']>len(obj.units):raise ValueError('Propagation cursor out of range.')
        if s['event'] is not None and (type(s['event'])is not int or not s['event_cursor'] or s['event']!=-s['queue'][s['event_cursor']-1]
                or s['watch_cursor']>len(watchers.get(s['event'],()))):raise ValueError('Invalid active propagation event.')
        if any(type(s[k])is not bool for k in ('finished','limited','seen_empty')) or s['result'] not in (None,False,True) or type(s['result'])not in (type(None),bool):raise ValueError('Invalid checker outcome.')
        if type(s['stats'])is not dict or set(s['stats'])!=set(STATS) or any(type(v)is not int or v<0 for v in s['stats'].values()):raise ValueError('Invalid checker statistics.')
        if s['stats']['proof_clause_visits']>obj.max_visits:raise ValueError('Checker exceeded its bound.')
        if s['candidate'] is not None:
            if s['proof_cursor']==len(obj.proof):raise ValueError('Candidate beyond proof end.')
            candidate,deletion=parse_line(obj.proof[s['proof_cursor']])
            if deletion or list(candidate)!=s['candidate']:raise ValueError('Candidate differs from exact proof line.')
            s['candidate']=candidate
        if s['phase'] in ('units','hints','watch') and s['candidate'] is None:raise ValueError('Missing live candidate.')
        if s['hint_cursor']>(len(obj.hints[s['proof_cursor']]) if s['proof_cursor']<len(obj.proof) else 0):raise ValueError('Hint cursor outside input.')
        if type(s['reason_ids'])is not list or any(type(v)is not int or not 0<=v<len(obj.active) for v in s['reason_ids']):raise ValueError('Invalid checked reason ids.')
        if type(s['generated_hints'])is not list or len(s['generated_hints'])!=s['proof_cursor'] or any(not isinstance(row,(list,tuple)) or any(type(v)is not int or not 0<=v<len(obj.active) for v in row) for row in s['generated_hints']):raise ValueError('Invalid retained hints.')
        expected_empty=any(not row for row in obj.active[len(obj.base)+len(obj.extra_clauses):])
        if s['seen_empty']!=expected_empty or s['result'] is True and (not s['finished'] or s['proof_cursor']!=len(obj.proof) or not expected_empty):raise ValueError('Unfinished cursor claims acceptance.')
        for key,value in s.items():
            if key not in ('assigned','watchers'):setattr(obj,key,value)
        obj.assigned=assigned;obj.watchers=defaultdict(list,watchers);return obj
