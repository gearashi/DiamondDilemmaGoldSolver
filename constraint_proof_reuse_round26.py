"""Bounded shared checked RUP prefixes and dependency-based proof compaction.

Only completed checker steps enter the process-local pool. Exact actual base
clauses and proof prefix bytes bind each reuse; arbitrary serialized state is
never accepted here. Streaming commits at64 completed lines keep a failed or
cancelled suffix from invalidating proved prefix work. Eviction loses reuse.
"""
from collections import OrderedDict
from constraint_proof import _hash,ProofLimit

MAX_BYTES=8*1024*1024
MAX_RECORDS=32
_PREFIXES=OrderedDict()

def scope(base,extra,source):return _hash({'base':base,'extra':extra,'source':source})

def lookup(binding,proof):
    for key,row in reversed(tuple(_PREFIXES.items())):
        source,prefix=key
        if source==binding and len(prefix)<=len(proof) and tuple(proof[:len(prefix)])==prefix:
            _PREFIXES.move_to_end(key);return row
    return None

def remember(binding,proof,clauses,hints,seen_empty):
    if not proof:return
    prefix=tuple(proof);clauses=tuple(tuple(row) for row in clauses);hints=tuple(tuple(row) for row in hints)
    size=sum(len(line) for line in prefix)+8*sum(map(len,clauses))+8*sum(map(len,hints))+64*len(prefix)
    if size>MAX_BYTES:return
    key=(binding,prefix);_PREFIXES[key]=(prefix,clauses,hints,bool(seen_empty),size);_PREFIXES.move_to_end(key)
    # A longer verified prefix subsumes its shorter exact ancestor.
    for old in tuple(_PREFIXES):
        if old!=key and old[0]==binding and len(old[1])<=len(prefix) and prefix[:len(old[1])]==old[1]:del _PREFIXES[old]
    while len(_PREFIXES)>MAX_RECORDS or sum(row[-1] for row in _PREFIXES.values())>MAX_BYTES:_PREFIXES.popitem(last=False)

def clear():_PREFIXES.clear()

def compact(base,proof,check=lambda:None,*,extra_clauses=(),stats=None):
    """Independently check, keep actual used antecedents, then check again.

    No claim of less total work in a one-off import: this trades an extra
    replay for smaller repeated transport. Returns None on an incomplete cap.
    """
    from constraint_proof_indexed_round26 import IndexedRUPChecker,parse_line
    stats={} if stats is None else stats
    for key in ('proof_compaction_input_steps','proof_compaction_output_steps','proof_compaction_checks','proof_compaction_visits','proof_compaction_index_literals'):stats.setdefault(key,0)
    first=IndexedRUPChecker(base,proof,extra_clauses=extra_clauses);second=None;callback=[None]
    stats['proof_compaction_input_steps']+=len(proof);stats['proof_compaction_checks']+=1
    def poll():
        try:check()
        except BaseException as exc:callback[0]=exc;raise
    try:
        while not first.finished:first.advance(poll,ticks=4096)
        if first.result is not True:return None
        size=len(first.base)+len(first.extra_clauses);by_id={};next_id=size;empty=None
        for index,raw in enumerate(proof):
            row,deletion=parse_line(raw)
            if row is not None and not deletion:
                by_id[next_id]=index;next_id+=1
                if not row and empty is None:empty=index
        needed=set();pending=[empty]
        while pending:
            poll();index=pending.pop()
            if index in needed:continue
            needed.add(index)
            pending.extend(by_id[antecedent] for antecedent in first.generated_hints[index] if antecedent in by_id)
        result=[line for index,line in enumerate(proof) if index in needed]
        second=IndexedRUPChecker(base,result,extra_clauses=extra_clauses)
        while not second.finished:second.advance(poll,ticks=4096)
        if second.result is True:stats['proof_compaction_output_steps']+=len(result);return result
        return None
    except ProofLimit:
        if callback[0] is not None:raise
        return None
    finally:
        stats['proof_compaction_index_literals']+=first.stats['proof_index_build_literals']
        stats['proof_compaction_visits']+=first.stats['proof_clause_visits']
        if second is not None:
            stats['proof_compaction_index_literals']+=second.stats['proof_index_build_literals']
            stats['proof_compaction_checks']+=1;stats['proof_compaction_visits']+=second.stats['proof_clause_visits']
