"""Bounded conflict-directed exact PB cancellation (round25 candidate 10)."""
from collections import defaultdict,OrderedDict
import heapq
from constraint_pb import combine,verify_derivation,is_integer_row,primitive_rows,filter_rows
from constraint_relations import bits

STAT_KEYS=('conflict_pb_trials','conflict_pb_pair_scans','conflict_pb_combinations',
           'conflict_pb_score_terms','conflict_pb_derived','conflict_pb_hits','conflict_pb_prunes',
           'conflict_pb_scope_builds','conflict_pb_cache_hits','conflict_pb_rows_kept','conflict_pb_no_cut',
           'conflict_pb_replay_rows','conflict_pb_storage_caps','conflict_pb_proof_rows_kept')


def slack(row,domains,stats):
    by_cell=defaultdict(dict)
    for cell,code,coefficient in row[0]:by_cell[cell][code]=coefficient
    lower=0
    for cell,coefficients in by_cell.items():
        costs=[]
        for code in bits(domains[cell]):
            stats['conflict_pb_score_terms']+=1;costs.append(coefficients.get(code,0))
        lower+=min(costs,default=0)
    return row[1]-lower


def _validated_rows(base,trial):
    if not isinstance(trial,(tuple,list)) or not trial or len(trial)>480 or any(type(mask)is not int or mask<0 or mask.bit_length()>480 for mask in trial):
        raise ValueError('Exact bounded nonnegative trial masks required.')
    if not isinstance(base,(tuple,list)) or any(not is_integer_row(item,max_bits=256) for item in base):
        raise ValueError('Exact bounded PB rows required.')
    if any(cell>=len(trial) or code>=480 for terms,rhs in base for cell,code,k in terms):
        raise ValueError('PB terms escape trial coordinates.')
    return list(dict.fromkeys(base))


def guided_cut_parent_priority(base,trial,check=lambda:None,*,max_pairs=256,max_new=64,stats=None):
    """Trial masks rank work only; every row descends from unchanged base axioms."""
    if type(max_pairs)is not int or not 0<=max_pairs<=16384 or type(max_new)is not int or not 0<=max_new<=256:raise ValueError('Invalid PB guidance budget.')
    validated=_validated_rows(base,trial)
    stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    stats['conflict_pb_trials']+=1
    if not max_pairs or not max_new:return None
    rows=validated;known=set(rows);steps=[];heap=[]
    if any(not is_integer_row(item,max_bits=256) for item in rows):raise ValueError('Exact bounded PB rows required.')
    terms=[{(c,v):k for c,v,k in item[0]} for item in rows]
    scores=[slack(item,trial,stats) for item in rows]
    def pairs(index):
        left=terms[index]
        for other in range(index):
            check();stats['conflict_pb_pair_scans']+=1;right=terms[other]
            pivots=[key for key in left.keys()&right.keys() if left[key]*right[key]<0]
            if not pivots:continue
            pivot=min(pivots,key=lambda key:(abs(left[key])+abs(right[key]),key))
            a,b=abs(right[pivot]),abs(left[pivot])
            priority=(a*scores[index]+b*scores[other],len(left)+len(right),index,other)
            heapq.heappush(heap,(priority,index,other,a,b))
    for index in range(len(rows)):pairs(index)
    work=0
    while heap and work<max_pairs and len(steps)<max_new:
        check();_,left,right,a,b=heapq.heappop(heap);work+=1;stats['conflict_pb_combinations']+=1
        cut=combine(rows[left],rows[right],a,b)
        if cut in known or len(cut[0])>256 or not is_integer_row(cut,max_bits=256):continue
        if not cut[0] and cut[1]>=0:continue
        known.add(cut);index=len(rows);rows.append(cut);steps.append((left,right,a,b,cut))
        stats['conflict_pb_derived']+=1;terms.append({(c,v):k for c,v,k in cut[0]});scores.append(slack(cut,trial,stats))
        if scores[-1]<0:
            stats['conflict_pb_replay_rows']+=len(base)+len(steps)
            if not verify_derivation(base,steps,check):raise RuntimeError('Guided PB proof failed independent exact replay.')
            stats['conflict_pb_hits']+=1;return cut,tuple(steps)
        pairs(index)
    return None


def guided_cut(base,trial,check=lambda:None,*,max_pairs=2048,max_new=64,stats=None):
    """Best-first exact cuts, including same-sign combinations for gcd cuts."""
    if type(max_pairs)is not int or not 0<=max_pairs<=16384 or type(max_new)is not int or not 0<=max_new<=256:raise ValueError('Invalid PB guidance budget.')
    validated=_validated_rows(base,trial)
    stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    stats['conflict_pb_trials']+=1
    if not max_pairs or not max_new:return None
    rows=validated;known=set(rows);queued=set();steps=[];heap=[];combinations=0
    if any(not is_integer_row(item,max_bits=256) for item in rows):raise ValueError('Exact bounded PB rows required.')
    terms=[{(c,v):k for c,v,k in item[0]} for item in rows]
    cells=[{c for c,v,k in item[0]} for item in rows]
    def pairs(index):
        nonlocal combinations
        left=terms[index]
        for other in range(index):
            check()
            if combinations>=max_pairs:return
            stats['conflict_pb_pair_scans']+=1;right=terms[other]
            if not cells[index]&cells[other]:continue
            pivots=[key for key in left.keys()&right.keys() if left[key]*right[key]<0]
            multipliers=[(1,1)]
            if pivots:
                pivot=min(pivots,key=lambda key:(abs(left[key])+abs(right[key]),key))
                pair=(abs(right[pivot]),abs(left[pivot]))
                if pair!=(1,1):multipliers.append(pair)
            for a,b in multipliers:
                if combinations>=max_pairs:return
                combinations+=1;stats['conflict_pb_combinations']+=1
                cut=combine(rows[index],rows[other],a,b)
                if cut in known or cut in queued or len(cut[0])>256 or not is_integer_row(cut,max_bits=256):continue
                if not cut[0] and cut[1]>=0:continue
                score=slack(cut,trial,stats)
                # An intermediate cut may not yet violate the trial, but a
                # tighter or shorter row can yield a later exact gcd cut.
                queued.add(cut)
                heapq.heappush(heap,((score,len(cut[0]),index,other,a,b),cut))
    for index in range(len(rows)):pairs(index)
    while heap and len(steps)<max_new:
        check();(_,_,left,right,a,b),cut=heapq.heappop(heap)
        if cut in known:continue
        known.add(cut);index=len(rows);rows.append(cut);steps.append((left,right,a,b,cut))
        stats['conflict_pb_derived']+=1;terms.append({(c,v):k for c,v,k in cut[0]});cells.append({c for c,v,k in cut[0]})
        if slack(cut,trial,stats)<0:
            stats['conflict_pb_replay_rows']+=len(base)+len(steps)
            if not verify_derivation(base,steps,check):raise RuntimeError('Guided PB proof failed independent exact replay.')
            stats['conflict_pb_hits']+=1;return cut,tuple(steps)
        pairs(index)
    return None


def compact_steps(base,steps):
    count=len(dict.fromkeys(base));needed=set()
    def visit(index):
        if index<count or index in needed:return
        needed.add(index);left,right,*_=steps[index-count];visit(left);visit(right)
    if not steps:return ()
    visit(count+len(steps)-1);mapping={i:i for i in range(count)};result=[]
    for old in sorted(needed):
        left,right,a,b,cut=steps[old-count]
        mapping[old]=count+len(result);result.append((mapping[left],mapping[right],a,b,cut))
    return tuple(result)


class ConflictPBLearner:
    def __init__(self,problem,*,limit=16):
        if type(limit)is not int or not 1<=limit<=64:raise ValueError('Invalid conflict-PB cache bound.')
        self.problem=problem;self.limit=limit;self.records=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
        self.binding=self._binding()

    def _binding(self):
        return (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),
                tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))

    def prune(self,domains,check=lambda:None,*,max_trials=64,max_pairs=2048,max_new=64):
        if self._binding()!=self.binding:raise ValueError('Conflict-PB source binding changed.')
        if (type(max_trials)is not int or not 0<=max_trials<=128 or type(max_pairs)is not int or not 0<=max_pairs<=16384
                or type(max_new)is not int or not 0<=max_new<=256):raise ValueError('Invalid conflict-PB trial budget.')
        if len(domains)!=self.problem.n or any(type(m)is not int or m<0 or m&~sum(1<<c for c in self.problem.domains[i]) for i,m in enumerate(domains)):raise ValueError('Conflict-PB domains escape original root.')
        current=list(domains);check()
        for scope,record in tuple(self.records.items()):
            if all(not current[c]&~mask for c,mask in scope):filter_rows(current,record['cuts'],check)
            if not all(current):break
        selected=tuple(sorted((c for c,m in enumerate(current) if 1<m.bit_count()<=12),key=lambda c:(current[c].bit_count(),c))[:5])
        if all(current) and len(selected)>=2 and max_trials:
            scope=next((old for old in self.records if set(c for c,m in old)==set(selected)
                        and all(not current[c]&~mask for c,mask in old)),None)
            if scope is None:
                scope=tuple((c,current[c]) for c in sorted(selected))
                self.records[scope]={'base':primitive_rows(self.problem,scope),'cuts':[],'proofs':[],'proof_terms':0,'attempted':OrderedDict()}
                self.stats['conflict_pb_scope_builds']+=1
                while len(self.records)>self.limit:self.records.popitem(last=False)
            else:self.stats['conflict_pb_cache_hits']+=1
            record=self.records[scope];self.records.move_to_end(scope);used=0
            candidates=[(c,code) for c in selected for code in bits(current[c])]
            for cell,code in candidates:
                if used>=max_trials or not all(current) or len(record['cuts'])>=64:break
                if not current[cell]>>code&1:continue
                trial=current.copy();trial[cell]=1<<code
                signature=(tuple(trial[c] for c,m in scope),max_pairs,max_new)
                if signature in record['attempted']:continue
                check();hit=guided_cut(record['base'],trial,check,max_pairs=max_pairs,max_new=max_new,stats=self.stats)
                # Only completely finished bounded attempts are memoized. A
                # capped no-cut result is not an infeasibility statement.
                def remember_attempt():
                    record['attempted'][signature]=None
                    while len(record['attempted'])>128:record['attempted'].popitem(last=False)
                used+=1
                if hit is None:
                    remember_attempt();self.stats['conflict_pb_no_cut']+=1;continue
                cut,steps=hit;steps=compact_steps(record['base'],steps)
                proof_terms=sum(len(step[4][0]) for step in steps)
                if record['proof_terms']+proof_terms>8192:
                    self.stats['conflict_pb_storage_caps']+=1;remember_attempt();continue
                self.stats['conflict_pb_replay_rows']+=len(record['base'])+len(steps)
                if not verify_derivation(record['base'],steps,check):raise RuntimeError('Retained PB cut failed exact replay.')
                reduced=current.copy();filter_rows(reduced,[cut],check)
                if reduced==current:remember_attempt();continue
                record['cuts'].append(cut);record['proofs'].append(steps);record['proof_terms']+=proof_terms
                self.stats['conflict_pb_proof_rows_kept']+=len(steps)
                self.stats['conflict_pb_rows_kept']+=1;current=reduced;remember_attempt()
        check();self.stats['conflict_pb_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
