"""Direct justified IPASIR-UP propagation and checked semantic restart clauses.

Native assignments and learned clauses are not proof authority. Every published
clause is independently replayed as a proper-loop, Hall, or affine contradiction
under its negation, relative to this exact session root. Only unit clauses under
the current trail yield direct literals. Reasons remain immutable while assigned.
"""
from collections import deque,OrderedDict
from types import SimpleNamespace
from constraint_sat_connectivity import gold_propagator
from constraint_knowledge import revalidate_witness
from constraint_work import digest

STAT_KEYS=('theory_clause_replays','theory_loop_replays','theory_hall_replays','theory_affine_replays',
           'theory_verified_clauses','theory_rejected_clauses','theory_direct_literals','theory_reason_requests',
           'theory_restart_clauses','theory_cache_hits','theory_budget_stops','theory_affine_equations','theory_affine_pattern_checks','theory_affine_xors')

EXTERNAL_KEYS=('sat_external_checks','sat_external_partial_conflicts','sat_external_model_conflicts','sat_external_clauses','sat_external_hall_checks','sat_external_hall_conflicts','sat_external_affine_checks','sat_external_affine_clauses')

def direct_propagator(problem,variables,*,target='gold',check=lambda:None,direct=True):
    if type(direct)is not bool:raise ValueError('Invalid direct theory policy.')
    if target not in ('gold','edges'):raise ValueError('Invalid theory target.')
    initial=tuple(sum(1<<code for code in row) for row in variables)
    def cancelled():
        try:base.replay_check();return False
        except Exception as exc:
            from constraint_sat_session import SessionInterrupted
            if isinstance(exc,SessionInterrupted):return True
            raise
    base=gold_propagator(problem,variables,run=SimpleNamespace(cancelled=cancelled,error=None),
                         theory_hall=True,theory_affine=True)
    base_propagate=base.propagate;base_check_model=base.check_model
    decode=base.decode
    verified=OrderedDict();reasons={};output=deque();stats=dict.fromkeys(STAT_KEYS,0)
    def semantic_source():
        # Fingerprints are cached; bind the actual puzzle and derived geometry.
        return (problem.fingerprint, problem.n, tuple(tuple(row) for row in problem.domains),
                tuple(problem.edges), tuple(tuple(row) for row in problem.board.edges),
                problem.masks.tobytes(), problem.neighbors.tobytes(), problem.sides.tobytes(),
                digest(problem.tiles), problem.total_segments, problem._segment_counts,
                problem._segment_ports, tuple(tuple(sorted(row.items())) for row in variables))
    source=semantic_source()
    def assert_source():
        if semantic_source()!=source:raise ValueError('SAT theory semantic source changed.')
    binding=digest((problem.fingerprint,target,initial,tuple(tuple(sorted(row.items())) for row in variables),
                    [value.hex() if isinstance(value,bytes) else value for value in source]))
    base.replay_check=check
    def local_check():base.replay_check()
    # The legacy generator receives the current slice cancellation callback too.
    def theory_check():
        from constraint_sat_session import SessionInterrupted
        from constraint_sat_connectivity import _TheoryStop
        try:local_check()
        except SessionInterrupted:raise _TheoryStop() from None
    base._theory_check=theory_check
    def verify(clause):
        assert_source()
        clause=tuple(dict.fromkeys(clause));stats['theory_clause_replays']+=1
        if len(clause)>sum(map(len,variables)) or any(type(lit)is not int or abs(lit) not in decode for lit in clause):return False
        if any(-lit in clause for lit in clause):return True
        domains=list(initial)
        for lit in clause:
            cell,code=decode[abs(lit)];domains[cell]&=~(1<<code) if lit>0 else 1<<code
        if not all(domains):return True
        if target=='gold':
            codes=[mask.bit_length()-1 if mask.bit_count()==1 else -1 for mask in domains]
            if problem.partial_loop_impossible(codes):
                revalidate_witness(problem,problem.gold_nogood(codes),target='gold')
                stats['theory_loop_replays']+=1;return True
        from constraint_deep import hall_certificate
        from constraint_pruning import _tile_choices
        stats['theory_hall_replays']+=1
        witness=hall_certificate(tuple(_tile_choices(mask) for mask in domains),local_check)
        if witness is not None:
            cells,pieces=witness
            union=set().union(*({code//3 for code in range(3*problem.n) if domains[c]>>code&1} for c in cells))
            if len(union)<len(set(cells)):return True
        from constraint_affine_reasons_round26 import produce_affine_conflict,verify_affine_conflict
        stats['theory_affine_replays']+=1
        local={}
        try:
            certificate=produce_affine_conflict(problem,domains,local_check,stats=local)
            return False if certificate is None else verify_affine_conflict(problem,domains,certificate,local_check,stats=local)
        finally:
            stats['theory_affine_equations']+=local.get('affine_reason_equations',0)
            stats['theory_affine_pattern_checks']+=local.get('affine_reason_pattern_checks',0)
            stats['theory_affine_xors']+=local.get('affine_reason_xors',0)
    def admit(raw):
        assert_source()
        clause=tuple(dict.fromkeys(raw))
        if clause in verified:stats['theory_cache_hits']+=1;return clause
        if not verify(clause):stats['theory_rejected_clauses']+=1;return None
        local_check();verified[clause]=True;stats['theory_verified_clauses']+=1
        while len(verified)>256:verified.popitem(last=False)
        return clause
    def collect():
        while base.pending:
            raw=base.pending[0]
            clause=admit(raw)
            # An interrupted independent replay must retain the generator claim.
            base.pending.popleft()
            if clause is not None:output.append(clause)
    def propagate():
        try:
            assert_source();local_check()
            if target=='gold':base_propagate()
            elif base.dirty:base.dirty=False;base._theory()
            if base.error is not None:return []
            collect();literals=[];retained=deque()
            while output:
                clause=output.popleft();unassigned=[];satisfied=False
                for lit in clause:
                    assigned=base.assignments.get(abs(lit))
                    if assigned==lit:satisfied=True;break
                    if assigned is None:unassigned.append(lit)
                if satisfied:continue
                if direct and len(unassigned)==1:
                    lit=unassigned[0];reasons[lit]=clause;literals.append(lit);stats['theory_direct_literals']+=1
                else:retained.append(clause)
            output.extend(retained);return list(dict.fromkeys(literals))
        except Exception as exc:
            # Stop is an incomplete optional callback, not a native contradiction.
            from constraint_sat_session import SessionInterrupted
            if isinstance(exc,SessionInterrupted):stats['theory_budget_stops']+=1
            else:base.error=exc
            return []
    def check_model(model):
        try:
            assert_source()
            if target!='gold':return True
            accepted=base_check_model(model)
            if base.error is None:collect()
            return accepted
        except Exception as exc:base.error=exc;return True
    def provide_reason(lit):
        try:assert_source()
        except ValueError as exc:base.error=exc;return [lit,-lit]
        if lit not in reasons:base.error=RuntimeError('Native solver requested an unproved literal.');return [lit,-lit]
        stats['theory_reason_requests']+=1
        return [lit,*[other for other in reasons[lit] if other!=lit]]
    def has_clause():return bool(output)
    def add_clause():
        if not output:return []
        base.stats['sat_external_clauses']+=1;return list(output.popleft())
    def snapshot():
        assert_source()
        return {'version':1,'binding':binding,'clauses':[list(row) for row in verified]}
    def restore(record):
        assert_source()
        if not isinstance(record,dict) or set(record)!={'version','binding','clauses'} or type(record['version'])is not int or record['version']!=1 or record['binding']!=binding or not isinstance(record['clauses'],list) or len(record['clauses'])>256:raise ValueError('Theory clause source/root differs.')
        rows=[]
        for raw in record['clauses']:
            if not isinstance(raw,list):raise ValueError('Invalid saved theory clause.')
            clause=admit(raw)
            if clause is None:raise ValueError('Saved theory clause failed independent replay.')
            rows.append(clause);stats['theory_restart_clauses']+=1
        return rows
    base.propagate=propagate;base.check_model=check_model;base.provide_reason=provide_reason
    base.has_clause=has_clause;base.add_clause=add_clause;base.semantic_snapshot=snapshot;base.semantic_restore=restore
    base.theory_stats=stats;base.verify_clause=verify;base.assert_semantic_source=assert_source
    return base
