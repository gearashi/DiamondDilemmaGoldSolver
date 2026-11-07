"""Exact Gold repair queries and constructive round-24 neighborhoods.

A repair failure is scoped to its explicit outside assumptions and original
cell domains. Optimization failures, exhausted attempt budgets and unavailable
engines never become exclusions. All proposed boards pass the independent
validator before a Gold success is returned.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import random
import time
from constraint_lns import score, _Control, _Cancelled, _watch
from validator import reverse_mask

class GoldRepairOracle:
    def __init__(self,problem,control):
        control.check()
        from ortools.sat.python import cp_model
        from constraint_circuit import build_circuit_model
        self.problem,self.control,self.cp_model=problem,control,cp_model
        self.binding=(problem.fingerprint,tuple(tuple(map(int,row)) for row in problem.domains))
        self.model=build_circuit_model(problem,[list(row) for row in problem.domains],check=control.check)
        self.stats={'oracle_queries':0,'oracle_branches':0,'oracle_conflicts':0,'oracle_unknown':0,'oracle_core_cells':0}
        self.failures=[]
    def query(self,assumptions=(),*,seconds=None):
        self.control.check()
        raw_assumptions=list(assumptions)
        if len({cell for cell,code in raw_assumptions})!=len(raw_assumptions):
            raise ValueError('Repair assumption cells must be distinct.')
        if self.binding!=(self.problem.fingerprint,tuple(tuple(map(int,row)) for row in self.problem.domains)):
            raise ValueError('Repair oracle inputs changed after construction.')
        assumptions=tuple(sorted(raw_assumptions))
        if any(type(cell) is not int or not 0<=cell<self.problem.n or type(code) is not int or not 0<=code<3*self.problem.n for cell,code in assumptions):
            raise ValueError('Invalid scoped repair assumptions.')
        given=dict(assumptions)
        for core in self.failures:
            if all(given.get(cell)==code for cell,code in core):
                return {'status':'infeasible','core':core,'cached':True,'binding':self.binding}
        for cell,code in assumptions:
            if code not in self.model.placements[cell]:
                return {'status':'infeasible','core':((cell,code),),'cached':False,'binding':self.binding}
        self.model.model.clear_assumptions()
        literals=[self.model.placements[cell][code] for cell,code in assumptions]
        self.model.model.add_assumptions(literals)
        mapping={literal.index:item for literal,item in zip(literals,assumptions)}
        self.control.check();solver=self.cp_model.CpSolver()
        solver.parameters.num_search_workers=1
        # The pinned native version can return presolved rather than original
        # assumption IDs on circuit constraints. Keep original IDs for cores.
        solver.parameters.cp_model_presolve=False
        deadline=self.control.deadline if seconds is None else min(self.control.deadline,time.monotonic()+seconds)
        solver.parameters.max_time_in_seconds=max(.000001,deadline-time.monotonic())
        self.stats['oracle_queries']+=1
        with _watch(self.control,deadline,solver):status=solver.solve(self.model.model)
        self.stats['oracle_branches']+=int(solver.num_branches);self.stats['oracle_conflicts']+=int(solver.num_conflicts)
        if status in (self.cp_model.FEASIBLE,self.cp_model.OPTIMAL):
            codes=[int(solver.value(value)) for value in self.model.codes]
            report=self.problem.validate(codes)
            if not report['valid'] or any(codes[cell]!=code for cell,code in assumptions) or any(code not in self.problem.domains[cell] for cell,code in enumerate(codes)):
                raise RuntimeError('Repair oracle failed independent Gold/domain/assumption validation.')
            return {'status':'solved','codes':codes,'validation':report,'binding':self.binding}
        if status==self.cp_model.INFEASIBLE:
            raw=solver.sufficient_assumptions_for_infeasibility()
            # Full assumptions are still a proved conflict if a native
            # backend supplies an unmappable smaller core. Never guess IDs.
            core=assumptions if any(index not in mapping for index in raw) else tuple(sorted(mapping[index] for index in raw));self.stats['oracle_core_cells']+=len(core)
            self.failures.append(core)
            self.failures=[old for old in self.failures if not (old!=core and set(core)<set(old))][-256:]
            return {'status':'infeasible','core':core,'cached':False,'binding':self.binding}
        if status==self.cp_model.MODEL_INVALID:raise RuntimeError('Invalid exact Gold repair model: '+self.model.model.validate())
        self.stats['oracle_unknown']+=1
        return {'status':'unknown','binding':self.binding}
    def export_exclusions(self):
        # A receiving process must verify these claims with its own scoped
        # oracle; this export is not advertised as a native proof certificate.
        return {'kind':'gold_repair_assumption_cores','fingerprint':self.problem.fingerprint,
                'domains':[list(row) for row in self.problem.domains],
                'cores':[[list(item) for item in core] for core in self.failures]}

def _minimum_hitting_set(cores,cells,control,cp_model):
    model=cp_model.CpModel();release={cell:model.new_bool_var(f'release_{cell}') for cell in cells}
    for core in cores:
        row=[release[cell] for cell in core if cell in release]
        if not row:return None,'infeasible'
        model.add_bool_or(row)
    model.minimize(sum(release.values()))
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=max(.000001,control.deadline-time.monotonic())
    with _watch(control,control.deadline,solver):status=solver.solve(model)
    if status==cp_model.OPTIMAL:return {cell for cell,var in release.items() if solver.value(var)},int(solver.num_branches)
    return None,'unknown'

def core_guided_repair(problem,codes,*,seconds=10,stop=None,max_queries=128,durable=False,resume=None):
    """Find a minimum-Hamming-distance Gold board through assumption cores."""
    if type(durable) is not bool:raise ValueError('durable must be boolean.')
    if durable or resume is not None:
        from constraint_repair_resume_round25 import core_guided_repair as continued
        return continued(problem,codes,seconds=seconds,stop=stop,max_queries=max_queries,resume=resume)
    if type(max_queries)is not int or max_queries<1:raise ValueError('Positive repair query limit required.')
    score(problem,codes);control=_Control(seconds,stop);seed=list(codes)
    result={'status':'unknown','scope':'minimum_change_repair','best_board':seed,'cores':[],
            'stats':{'hitting_set_solves':0,'hitting_set_branches':0,'queries':0},'minimum_changes':None}
    try:
        oracle=GoldRepairOracle(problem,control)
        cells=[cell for cell in range(problem.n) if cell not in problem.fixed]
        cores=[]
        for _ in range(max_queries):
            control.check();released,work=_minimum_hitting_set(cores,cells,control,oracle.cp_model)
            result['stats']['hitting_set_solves']+=1
            if released is None:result['status']=work;break
            result['stats']['hitting_set_branches']+=work
            answer=oracle.query((cell,seed[cell]) for cell in cells if cell not in released)
            result['stats']['queries']+=1
            if answer['status']=='solved':
                actual=sum(a!=b for a,b in zip(seed,answer['codes']))
                if actual!=len(released):raise RuntimeError('Exact core-guided repair violated its minimum-change bound.')
                result.update(status='solved',best_board=answer['codes'],validation=answer['validation'],minimum_changes=actual);break
            if answer['status']!='infeasible':break
            core=frozenset(cell for cell,code in answer['core'])
            if not core:result['status']='infeasible';break
            if core in cores:raise RuntimeError('Core-guided repair failed to advance its hitting set.')
            cores.append(core)
        result['cores']=[sorted(core) for core in cores];result['exclusions']=oracle.export_exclusions()
        result['stats'].update(oracle.stats)
    except _Cancelled:result['status']=control.reason or 'unknown'
    except ImportError:result['status']='unavailable'
    except Exception as exc:
        from constraint_circuit import CircuitLimit
        if not isinstance(exc,CircuitLimit):raise
        result.update(status='unknown',reason=str(exc))
    result['elapsed_seconds']=time.monotonic()-control.began
    return result

def _patch_histories(problem,current,patch,check,stats,*,max_states=200000):
    """Exact boundary-preserving patch enumeration; cap raises, never proves UNSAT."""
    patch=tuple(patch)
    if any(type(cell)is not int or not 0<=cell<problem.n for cell in patch):raise ValueError('Invalid repair patch cell.')
    if type(max_states)is not int or max_states<1:raise ValueError('Positive patch state limit required.')
    cells=tuple(sorted(set(patch)));inside=set(cells);inventory={current[cell]//3 for cell in cells}
    if any(cell in problem.fixed for cell in cells):
        cells=tuple(cell for cell in cells if cell not in problem.fixed);inside=set(cells);inventory={current[cell]//3 for cell in cells}
    choices={}
    for cell in cells:
        choices[cell]=[code for code in problem.domains[cell] if code//3 in inventory and
            all(int(problem.masks[code,side])==reverse_mask(problem.masks[current[int(other)],int(problem.sides[cell,side])])
                for side,other in enumerate(problem.neighbors[cell]) if other>=0 and int(other) not in inside)]
    order=sorted(cells,key=lambda c:(len(choices[c]),c));assignment={};states=0
    def walk(depth,used):
        nonlocal states
        check()
        if depth==len(order):
            answer=list(current)
            for cell,code in assignment.items():answer[cell]=code
            stats['trade_histories']+=1;yield answer;return
        cell=order[depth]
        for code in choices[cell]:
            check();states+=1;stats['trade_states']+=1
            if states>max_states:raise _PatchCap
            if code//3 in used:continue
            if any(int(other) in assignment and int(problem.masks[code,side])!=reverse_mask(problem.masks[assignment[int(other)],int(problem.sides[cell,side])]) for side,other in enumerate(problem.neighbors[cell]) if other>=0):continue
            assignment[cell]=code
            yield from walk(depth+1,used|{code//3})
            del assignment[cell]
    yield from walk(0,set())

class _PatchCap(Exception):pass

def loop_merge_trade(problem,codes,patch,*,seconds=2,stop=None,max_states=200000,durable=False,resume=None):
    if type(durable) is not bool:raise ValueError('durable must be boolean.')
    if durable or resume is not None:
        from constraint_repair_resume_round25 import loop_merge_trade as continued
        return continued(problem,codes,patch,seconds=seconds,stop=stop,max_states=max_states,resume=resume)
    checked=score(problem,codes);report=problem.validate(codes)
    if checked['domain_violations'] or checked['matched_edges']!=len(problem.edges):
        raise ValueError('Loop trades require an edge-perfect legal board.')
    control=_Control(seconds,stop);best=list(codes);components=report['line_components']
    stats={'trade_states':0,'trade_histories':0,'trade_improvements':0}
    complete=False;status='unknown'
    try:
        for candidate in _patch_histories(problem,codes,patch,control.check,stats,max_states=max_states):
            validation=problem.validate(candidate)
            if validation['matched_edges']!=len(problem.edges) or not validation['unique_tiles']:
                raise RuntimeError('Trade failed independent seam/inventory validation.')
            if validation['line_components']<components:
                best,components=candidate,validation['line_components'];stats['trade_improvements']+=1
                if validation['valid']:status='solved';break
        else:complete=True;status='completed'
    except (_Cancelled,_PatchCap):pass
    return {'status':status,'scope':'fixed_patch_boundary','complete_local_query':complete,
            'best_board':best,'validation':problem.validate(best),'stats':stats,
            'elapsed_seconds':time.monotonic()-control.began}

def matched_fragments(problem,boards,max_fragment=8):
    """Connected, internally matched archive fragments; no outside assumptions."""
    if type(max_fragment)is not int or max_fragment<1:raise ValueError('Positive fragment size required.')
    fragments=[]
    for source,board in enumerate(boards):
        score(problem,board);unseen={c for c in range(problem.n) if c not in problem.fixed}
        while unseen:
            root=min(unseen);unseen.remove(root);patch=[root]
            for cell in patch:
                if len(patch)>=max_fragment:break
                for side,other in enumerate(problem.neighbors[cell]):
                    other=int(other)
                    if other not in unseen:continue
                    if int(problem.masks[board[cell],side])!=reverse_mask(problem.masks[board[other],int(problem.sides[cell,side])]):continue
                    patch.append(other);unseen.remove(other)
                    if len(patch)>=max_fragment:break
            fragments.append({'source':source,'placements':tuple((cell,int(board[cell])) for cell in patch)})
    return fragments

def fragment_crossover(problem,boards,*,seconds=10,stop=None,max_fragment=8,max_queries=32,durable=False,resume=None):
    if type(durable) is not bool:raise ValueError('durable must be boolean.')
    if durable or resume is not None:
        from constraint_repair_resume_round25 import fragment_crossover as continued
        return continued(problem,boards,seconds=seconds,stop=stop,max_fragment=max_fragment,max_queries=max_queries,resume=resume)
    if not boards:raise ValueError('Crossover needs candidate boards.')
    if type(max_queries)is not int or max_queries<1:raise ValueError('Positive crossover query limit required.')
    control=_Control(seconds,stop);best=list(boards[0]);stats={'fragment_queries':0,'fragment_master_branches':0,'fragments':0}
    result={'status':'unknown','scope':'archive_fragment_recombination','best_board':best,'stats':stats}
    try:
        oracle=GoldRepairOracle(problem,control);cp=oracle.cp_model
        fragments=matched_fragments(problem,boards,max_fragment=max_fragment);stats['fragments']=len(fragments)
        model=cp.CpModel();select=[model.new_bool_var(f'fragment_{i}') for i in range(len(fragments))]
        by_cell={};by_tile={}
        for index,fragment in enumerate(fragments):
            for cell,code in fragment['placements']:
                by_cell.setdefault(cell,[]).append(select[index]);by_tile.setdefault(code//3,[]).append(select[index])
        for row in [*by_cell.values(),*by_tile.values()]:model.add_at_most_one(row)
        model.maximize(sum(len(fragment['placements'])*select[i] for i,fragment in enumerate(fragments)))
        for _ in range(max_queries):
            control.check();solver=cp.CpSolver();solver.parameters.num_search_workers=1
            solver.parameters.max_time_in_seconds=max(.000001,control.deadline-time.monotonic())
            with _watch(control,control.deadline,solver):status=solver.solve(model)
            stats['fragment_master_branches']+=int(solver.num_branches)
            if status not in (cp.FEASIBLE,cp.OPTIMAL):break
            chosen=[i for i,var in enumerate(select) if solver.value(var)]
            assumptions=[item for i in chosen for item in fragments[i]['placements']]
            answer=oracle.query(assumptions);stats['fragment_queries']+=1
            if answer['status']=='solved':
                result.update(status='solved',best_board=answer['codes'],validation=answer['validation'],sources=sorted({fragments[i]['source'] for i in chosen}));break
            if answer['status']!='infeasible':break
            core=set(answer['core'])
            conflict=[select[i] for i in chosen if core.intersection(fragments[i]['placements'])]
            if not conflict:result['status']='infeasible';break
            model.add_bool_or([var.Not() for var in conflict])
        result['exclusions']=oracle.export_exclusions();stats.update(oracle.stats)
    except _Cancelled:result['status']=control.reason or 'unknown'
    except ImportError:result['status']='unavailable'
    except Exception as exc:
        from constraint_circuit import CircuitLimit
        if not isinstance(exc,CircuitLimit):raise
        result.update(status='unknown',reason=str(exc))
    result['elapsed_seconds']=time.monotonic()-control.began
    return result

def scoped_repair_failure(problem,codes,patch,*,seconds=5,stop=None,durable=False,resume=None):
    """Query feasibility under outside assignments, without optimization exclusions."""
    if type(durable) is not bool:raise ValueError('durable must be boolean.')
    if durable or resume is not None:
        from constraint_repair_resume_round25 import scoped_repair_failure as continued
        return continued(problem,codes,patch,seconds=seconds,stop=stop,resume=resume)
    score(problem,codes);control=_Control(seconds,stop)
    patch=set(patch)
    if any(type(cell)is not int or not 0<=cell<problem.n for cell in patch):raise ValueError('Invalid repair patch cell.')
    try:
        oracle=GoldRepairOracle(problem,control)
        answer=oracle.query((cell,code) for cell,code in enumerate(codes) if cell not in patch)
        answer['exclusions']=oracle.export_exclusions();answer['stats']=oracle.stats;return answer
    except _Cancelled:return {'status':control.reason or 'unknown'}
    except ImportError:return {'status':'unavailable'}
    except Exception as exc:
        from constraint_circuit import CircuitLimit
        if not isinstance(exc,CircuitLimit):raise
        return {'status':'unknown','reason':str(exc)}
