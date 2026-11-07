"""Whole-board CP-SAT variables selecting exact regional alternatives.

A deterministic disjoint partition covers all cells. Each table contains only
complete concrete regional histories, not merged placement-support unions.
An over-budget multi-cell table is discarded and those cells get singleton
regions; this loses strength without excluding a possible arrangement. Existing
cross-region seam, physical-inventory and Gold constraints remain in force.
"""
from constraint_regions import HierarchicalRegionPruner
from constraint_region_network import ExactRegionDiagram
from constraint_separator import _Budget


def add_regional_alternatives(problem,domains,model,codes,*,target='gold',check=lambda:None,
                              max_cells=3,max_rows=4096,max_steps=200000):
    if type(max_cells)is not int or not 1<=max_cells<=16:raise ValueError('Invalid model region size.')
    if type(max_rows)is not int or max_rows<1 or type(max_steps)is not int or max_steps<1:
        raise ValueError('Invalid model regional table budget.')
    masks=[sum(1<<code for code in row) for row in domains]
    owner=HierarchicalRegionPruner(problem,target);pending=set(range(problem.n));index=0
    stats={'regional_model_regions':0,'regional_model_joined_regions':0,
           'regional_model_rows':0,'regional_model_diagram_nodes':0,'regional_model_budget_fallbacks':0}
    while pending:
        check();first=min(pending,key=lambda c:(len(domains[c]),c));cells=[first]
        while len(cells)<max_cells:
            candidates=set().union(*(owner.neighbors[c] for c in cells))&pending-set(cells)
            if not candidates:break
            cells.append(min(candidates,key=lambda c:(len(domains[c]),c)))
        selected=None
        while len(cells)>1:
            try:selected=ExactRegionDiagram(owner,cells,masks,check,max_rows=max_rows,max_steps=max_steps);break
            except _Budget:
                stats['regional_model_budget_fallbacks']+=1;cells.pop()
        if selected is None:
            cells=[first];rows=[(code,) for code in domains[first]]
        else:
            cells=list(selected.cells);rows=[row for row,signature in selected.histories(check)]
            stats['regional_model_diagram_nodes']+=len(selected.nodes)+len(selected.terminals)
            stats['regional_model_joined_regions']+=1
        choice=model.new_int_var(0,max(0,len(rows)-1),f'region_choice_{index}')
        # The relation exposes the regional alternative as a real model
        # variable. Whole physical-tile and seam constraints link the selected
        # cell values across regions; arbitrary mixed rows are never admitted.
        model.add_allowed_assignments([choice,*[codes[c] for c in cells]],
                                      [(i,*row) for i,row in enumerate(rows)])
        stats['regional_model_regions']+=1;stats['regional_model_rows']+=len(rows)
        pending.difference_update(cells);index+=1
    return stats



# Source/query-bound local compilation only; native model variables never enter
# this cache. New native solvers necessarily replay their flow constraints.
_NATIVE_REGION_SESSIONS={}
import threading
_NATIVE_REGION_LOCK=threading.RLock()


def _add_regional_dags(problem,domains,*,backend,model=None,codes=None,solver=None,
                      literal=None,new_variable=None,target='gold',check=lambda:None,
                      max_cells=4,max_states=4096,max_steps=50000,cache_dir=None,stats_sink=None):
    """F09 native regional paths; no concrete-history expansion or tuple table.

    The ordinary native model still owns global piece/seam/Gold constraints.
    Capped diagrams are omitted, losing strength only; no partial diagram is
    encoded. Both backends consume the same independently tested relation.
    """
    from constraint_region_interfaces_round26 import ParametricEnvelope
    from constraint_region_compiler_round26 import add_dag_cp_sat,add_dag_sat,_digest,_source,MAX_BYTES
    from constraint_region_incremental import source_binding
    from pathlib import Path
    from io_utils import atomic_json
    import json
    if backend not in ('sat','cp-sat') or type(max_cells)is not int or not 1<=max_cells<=16:
        raise ValueError('Invalid native regional DAG policy.')
    owner=HierarchicalRegionPruner(problem,target);masks=[sum(1<<v for v in set(row)) for row in domains]
    pending=set(range(problem.n));stats={name:0 for name in ('regional_dag_regions','regional_dag_arcs',
       'regional_dag_constraints','regional_dag_compile_transitions','regional_dag_compile_states',
       'regional_dag_cap_skips','regional_dag_expanded_histories','regional_dag_compile_reuses',
       'regional_dag_compile_resumes','regional_dag_cursor_loads','regional_dag_cursor_saves',
       'regional_dag_cursor_io_failures','regional_dag_compile_evictions','regional_dag_encoder_replays')}
    flushed={}
    def flush_stats():
        if stats_sink is not None:
            for name,value in stats.items():stats_sink[name]=stats_sink.get(name,0)+value-flushed.get(name,0)
            flushed.update(stats)
    binding=(source_binding(owner),_source())
    directory=None if cache_dir is None else Path(cache_dir)/'native-regional-dags-v1'
    while pending:
        check();first=min(pending,key=lambda c:(len(domains[c]),c));cells=[first];pending.remove(first)
        while len(cells)<max_cells:
            neighbors=set().union(*(owner.neighbors[c] for c in cells))&pending
            if not neighbors:break
            nxt=min(neighbors,key=lambda c:(len(domains[c]),c));cells.append(nxt);pending.remove(nxt)
        key=_digest((binding,tuple(sorted(cells)),tuple(masks),max_states,max_steps))
        path=None if directory is None else directory/(key+'.json')
        envelope=_NATIVE_REGION_SESSIONS.pop(key,None)
        if envelope is None:
            saved=None
            if path is not None:
                try:
                    if path.is_file() and path.stat().st_size<=MAX_BYTES:saved=json.loads(path.read_text(encoding='utf-8'))
                except (OSError,ValueError):stats['regional_dag_cursor_io_failures']+=1
            try:
                envelope=ParametricEnvelope(owner,cells,masks,max_states=max_states,max_steps=max_steps,resume=saved)
                stats['regional_dag_cursor_loads']+=int(saved is not None)
            except ValueError:
                if saved is None:raise
                stats['regional_dag_cursor_io_failures']+=1
                envelope=ParametricEnvelope(owner,cells,masks,max_states=max_states,max_steps=max_steps)
        else:
            stats['regional_dag_compile_reuses' if envelope.session.complete else 'regional_dag_compile_resumes']+=1
            stats['regional_dag_encoder_replays']+=int(envelope.session.complete)
        _NATIVE_REGION_SESSIONS[key]=envelope
        before=(envelope.session.steps,envelope.session.states)
        try:
            while not envelope.advance(128,check):pass
            diagram=envelope.view(masks,check)
        except _Budget:
            stats['regional_dag_cap_skips']+=1;continue
        finally:
            stats['regional_dag_compile_transitions']+=envelope.session.steps-before[0]
            stats['regional_dag_compile_states']+=envelope.session.states-before[1]
            if path is not None:
                try:
                    payload=envelope.snapshot();path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,payload)
                    stats['regional_dag_cursor_saves']+=1
                except (OSError,ValueError,_Budget):stats['regional_dag_cursor_io_failures']+=1
            while len(_NATIVE_REGION_SESSIONS)>128 or sum(env.session.states for env in _NATIVE_REGION_SESSIONS.values())>50000:
                _NATIVE_REGION_SESSIONS.pop(next(iter(_NATIVE_REGION_SESSIONS)));stats['regional_dag_compile_evictions']+=1
            flush_stats()
        work={}
        try:
            if backend=='cp-sat':add_dag_cp_sat(diagram,model,codes,prefix=f'regional_dag_{stats["regional_dag_regions"]}',check=check,work=work)
            else:add_dag_sat(diagram,solver,literal,new_variable,check=check,work=work)
            stats['regional_dag_regions']+=1
        finally:
            # Failed model construction is rebuilt, and its actual work remains visible.
            stats['regional_dag_arcs']+=work.get('arcs',0);stats['regional_dag_constraints']+=work.get('constraints',0)
            flush_stats()
    flush_stats()
    return stats if stats_sink is None else {name:stats_sink[name] for name in stats}



def add_regional_dags(problem,domains,*,check=lambda:None,**kwargs):
    """Serialize mutation of shared compiler cursors; native encodings stay local."""
    while not _NATIVE_REGION_LOCK.acquire(timeout=.05):check()
    try:return _add_regional_dags(problem,domains,check=check,**kwargs)
    finally:_NATIVE_REGION_LOCK.release()
