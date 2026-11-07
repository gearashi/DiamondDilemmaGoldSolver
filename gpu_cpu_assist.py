"""Scoped CPU reasoning and durable exact endgames for GPU-owned DFS jobs.

A transferred current subtree retains its GPU stack and ownership. Its lane is
excluded from dispatch while CPU DFS owns that subtree. UNKNOWN keeps the CPU
checkpoint; only completed infeasibility backtracks the GPU stack. This avoids
racing two exact engines over the same active subtree.
"""
from __future__ import annotations
import copy
import json
import math
import time
from collections import OrderedDict

from constraint_pruning import Pruner
from search_problem import SearchProblem
from validator import reverse_mask


class _SliceExpired(Exception): pass


class CPUAssist:
    def __init__(self, data, fingerprint, *, probe=False, endgame=False, max_open=12,
                 seconds=.02, interval=64, proof_targeted=False):
        self.problem = SearchProblem(data)
        self.proof_targeted=bool(proof_targeted)
        self.fingerprint = fingerprint
        self.probe_enabled, self.endgame_enabled = bool(probe), bool(endgame)
        self.max_open, self.seconds, self.interval = int(max_open), float(seconds), int(interval)
        if not 1<=self.max_open<=32 or not math.isfinite(self.seconds) or not 0<self.seconds<=2 or self.interval<1:
            raise ValueError('CPU assistance needs positive bounded scheduling limits.')
        self.turn = 0
        self.entries = OrderedDict()
        self.completed_probes = OrderedDict()
        self.candidates = []
        self.stats = {'cpu_prefix_probes':0,'cpu_prefix_failures':0,'cpu_prefix_unknown':0,
                      'cpu_prefix_values_removed':0,'cpu_endgame_calls':0,'cpu_endgame_nodes':0,
                      'cpu_endgame_infeasible':0,'cpu_endgame_solutions':0,'cpu_endgame_unknown':0,
                      'cpu_endgame_held_lanes':0,'cpu_probe_selection_contexts':0,'cpu_probe_hall_targets':0}
        self.signature_codes = [{}, {}, {}]
        for code,row in enumerate(self.problem.masks):
            for side,mask in enumerate(row):
                self.signature_codes[side][int(mask)] = self.signature_codes[side].get(int(mask),0) | (1<<code)

    def _validate_assignment(self, assignment):
        if not isinstance(assignment,(list,tuple)) or len(assignment)!=160 or any(type(code) is not int or not -1<=code<480 for code in assignment):
            raise ValueError('CPU/GPU prefix binding is malformed.')
        used=[code//3 for code in assignment if code>=0]
        if len(set(used))!=len(used):raise ValueError('CPU/GPU prefix repeats a physical piece.')
        for a,sa,b,sb in self.problem.edges:
            if assignment[a]>=0 and assignment[b]>=0 and int(self.problem.masks[assignment[a],sa]) != reverse_mask(self.problem.masks[assignment[b],sb]):
                raise ValueError('CPU/GPU prefix has an incompatible seam.')
        return tuple(assignment)

    def probe(self, assignment, *, seconds=None):
        """Return a completed scope-bound propagation result; a cap is unknown."""
        key=self._validate_assignment(assignment)
        if key in self.completed_probes:
            return copy.deepcopy(self.completed_probes[key])
        deadline=time.monotonic()+(self.seconds if seconds is None else seconds)
        def check():
            if time.monotonic()>=deadline:raise _SliceExpired
        initial=[1<<code if code>=0 else (1<<480)-1 for code in key]
        domains=initial.copy();self.stats['cpu_prefix_probes']+=1
        try:
            check()
            if self.problem.partial_loop_impossible(list(key)):
                domains[0]=0
            else:
                pruner=Pruner(self.problem,check,neighborhood=False,parity=False,native='auto')
                while all(domains):
                    check();before=domains.copy()
                    for a,sa,b,sb in self.problem.edges:
                        check()
                        for left,ls,right,rs in ((a,sa,b,sb),(b,sb,a,sa)):
                            possible=0
                            for mask,codes in self.signature_codes[rs].items():
                                if domains[right]&codes:possible |= self.signature_codes[ls].get(reverse_mask(mask),0)
                            domains[left] &= possible
                            if not domains[left]:break
                        if not all(domains):break
                    if all(domains):pruner.prune_hall(domains)
                    if domains==before:break
            result={'status':'infeasible' if not all(domains) else 'propagated',
                    'assignment':list(key),'domains':domains,'fingerprint':self.fingerprint,
                    'removed':sum((a&~b).bit_count() for a,b in zip(initial,domains))}
            self.stats['cpu_prefix_values_removed']+=result['removed']
            self.stats['cpu_prefix_failures']+=int(result['status']=='infeasible')
            self.completed_probes[key]=copy.deepcopy(result)
            while len(self.completed_probes)>64:self.completed_probes.popitem(last=False)
            return result
        except _SliceExpired:
            self.stats['cpu_prefix_unknown']+=1
            return {'status':'unknown','assignment':list(key),'fingerprint':self.fingerprint}

    def obstruction_priority(self, assignment):
        from itertools import combinations
        from constraint_pruning import _tile_choices
        self.stats['cpu_probe_selection_contexts']+=1
        used=0
        for code in assignment:
            if code>=0:used |= 7<<(3*(code//3))
        domains=[]
        for cell,code in enumerate(assignment):
            if code>=0:continue
            domain=((1<<480)-1)&~used
            for side,other in enumerate(self.problem.neighbors[cell]):
                other=int(other)
                if other>=0 and assignment[other]>=0:
                    needed=reverse_mask(self.problem.masks[assignment[other],int(self.problem.sides[cell,side])])
                    domain &= self.signature_codes[side].get(needed,0)
            pieces=_tile_choices(domain)
            domains.append((pieces.bit_count(),cell,pieces))
        selected=sorted(domains)[:12]
        shortage=False;minimum_slack=160
        for size in range(4,min(6,len(selected))+1):
            for group in combinations(selected,size):
                union=0
                for _,_,pieces in group:union|=pieces
                slack=union.bit_count()-size
                minimum_slack=min(minimum_slack,slack)
                if slack<0:shortage=True;break
            if shortage:break
        return (int(shortage),-minimum_slack,sum(size<=4 for size,_,_ in selected))

    def _entry_lane(self, gpu, entry):
        owner=tuple(entry['owner_prefix'])
        try:lane=gpu.prefixes.index(owner)
        except ValueError:return None
        if int(gpu.states[lane].get()) not in (0,1):return None
        current=list(map(int,gpu.boards[:,lane].get()))
        if current!=entry['assignment']:
            raise ValueError('GPU moved a subtree owned by its CPU endgame.')
        return lane

    def before_step(self, gpu):
        self.turn+=1
        held=[]
        # At most one endgame is admitted, bounding both CPU memory and launch delay.
        for key,entry in list(self.entries.items()):
            lane=self._entry_lane(gpu,entry)
            if lane is None:continue # A paused job remains owned by the SSD lane bank.
            from constraint_dfs import solve
            if '_problem' not in entry:
                entry['_problem']=SearchProblem(self.problem.tiles,self.problem.board,
                    fixed={cell:code for cell,code in enumerate(entry['assignment']) if code>=0})
            self.stats['cpu_endgame_calls']+=1
            answer=solve(entry['_problem'],seconds=entry.get('slice_seconds',self.seconds),target='gold',
                         resume=entry.get('checkpoint'))
            old_nodes=entry.get('nodes',0);entry['nodes']=int(answer.get('nodes',0))
            self.stats['cpu_endgame_nodes']+=max(0,entry['nodes']-old_nodes)
            if answer['status']=='infeasible':
                gpu.reject_current_subtree(lane,entry['assignment'])
                self.stats['cpu_endgame_infeasible']+=1;del self.entries[key]
            elif answer['status']=='solved':
                codes=list(map(int,answer['codes']))
                report=self.problem.validate(codes)
                if not report['valid'] or any(code>=0 and codes[cell]!=code for cell,code in enumerate(entry['assignment'])):
                    raise RuntimeError('CPU endgame failed independent Gold/scope validation.')
                gpu.install_cpu_candidate(lane,entry['assignment'],codes)
                self.candidates.append(codes);self.stats['cpu_endgame_solutions']+=1;del self.entries[key]
            else:
                self.stats['cpu_endgame_unknown']+=1
                if int(answer.get('nodes',0)) == old_nodes:
                    entry['slice_seconds']=min(2.0,2*entry.get('slice_seconds',self.seconds))
                if answer.get('checkpoint') is not None:entry['checkpoint']=answer['checkpoint']
                held.append(lane)
        if self.entries or self.turn%self.interval:
            self.stats['cpu_endgame_held_lanes']=len(held);return held
        states=gpu.states.get();depths=gpu.depths.get()
        # A dispatch can stop just after the final placement, before its next
        # iteration marks the board pending. Leave that depth160 boundary to
        # the normal GPU path; it has no current cursor or open CPU subtree.
        lanes=[int(lane) for lane in range(gpu.n) if states[lane]==0 and 16<=depths[lane]<160]
        if not lanes:return held
        lane=max(lanes,key=lambda value:int(depths[value]))
        assignment=None
        if self.proof_targeted and self.probe_enabled:
            candidates=sorted(lanes,key=lambda value:(-int(depths[value]),value))[:8]
            # Bounded selection only: the chosen obstruction is re-proved by
            # probe(), and this ranking never rejects a GPU subtree itself.
            ranked=[]
            for candidate in candidates:
                values=list(map(int,gpu.boards[:,candidate].get()))
                priority=self.obstruction_priority(values)
                ranked.append((priority,int(depths[candidate]),-candidate,candidate,values))
            priority,_,_,lane,assignment=max(ranked)
            if priority[0]:self.stats['cpu_probe_hall_targets']+=1
        if assignment is None:assignment=list(map(int,gpu.boards[:,lane].get()))
        key=tuple(assignment)
        if self.probe_enabled and key not in self.completed_probes:
            proof=self.probe(assignment)
            if proof['status']=='infeasible':
                gpu.reject_current_subtree(lane,assignment);return held
            if proof['status']=='propagated':gpu._install_cpu_restrictions(proof)
        if self.endgame_enabled and sum(code<0 for code in assignment)<=self.max_open and int(gpu.cursors[int(depths[lane]),lane].get())==65535:
            self.entries[key]={'assignment':assignment,'owner_prefix':list(gpu.prefixes[lane]),'nodes':0}
            held.append(lane)
        self.stats['cpu_endgame_held_lanes']=len(held)
        return held

    @staticmethod
    def validate_ownership(saved, payloads):
        if not isinstance(saved,dict) or saved.get('version')!=1 or not isinstance(saved.get('entries'),list) or len(saved['entries'])>1:
            raise ValueError('Malformed CPU/GPU ownership metadata.')
        owners={}
        for payload in payloads:
            for lane,state in enumerate(payload['states']):
                if int(state) not in (0,1):continue
                floor=int(payload['floors'][lane])
                prefix=tuple(map(int,payload['prefix_codes'][lane,:floor]))
                if prefix in owners:raise ValueError('Duplicate CPU/GPU job owner.')
                owners[prefix]=(int(state),list(map(int,payload['boards'][:,lane])),
                                int(payload['depths'][lane]),payload['cursors'][:,lane])
        for entry in saved['entries']:
            if not isinstance(entry,dict) or not isinstance(entry.get('owner_prefix'),list):
                raise ValueError('Malformed CPU/GPU owner prefix.')
            owner=owners.get(tuple(entry['owner_prefix']))
            if owner is None or owner[0]!=0 or owner[1]!=entry.get('assignment'):
                raise ValueError('CPU endgame names an orphaned or changed GPU subtree.')
            if owner[2]>=160 or int(owner[3][owner[2]])!=65535:
                raise ValueError('CPU endgame does not own a fresh GPU frame.')

    def snapshot(self):
        return {'version':1,'fingerprint':self.fingerprint,
                'entries':[{key:copy.deepcopy(value) for key,value in entry.items() if not key.startswith('_')}
                           for entry in self.entries.values()], 'stats':dict(self.stats)}

    def restore(self, saved):
        if not isinstance(saved,dict) or saved.get('version')!=1 or saved.get('fingerprint')!=self.fingerprint:
            raise ValueError('CPU/GPU assistance checkpoint binding differs.')
        rows=saved.get('entries')
        if not isinstance(rows,list) or len(rows)>1:raise ValueError('Malformed CPU/GPU endgame checkpoint.')
        incoming=OrderedDict()
        for entry in rows:
            if not isinstance(entry,dict):raise ValueError('Malformed CPU endgame record.')
            key=self._validate_assignment(entry.get('assignment'))
            owner=entry.get('owner_prefix')
            if not isinstance(owner,list) or any(type(code)is not int or not 0<=code<480 for code in owner):
                raise ValueError('Malformed CPU endgame owner.')
            row=copy.deepcopy(entry)
            if 'solved' in row:
                raise ValueError('CPU Gold certificates must use ordinary pending GPU boards.')
            amount=row.get('slice_seconds',self.seconds)
            if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or not 0<amount<=2:
                raise ValueError('CPU endgame slice exceeds its bounded scheduling policy.')
            if type(row.get('nodes',0)) is not int or row.get('nodes',0)<0:
                raise ValueError('Malformed CPU endgame node counter.')
            incoming[key]=row
        self.entries=incoming
        if isinstance(saved.get('stats'),dict):
            for key in self.stats:
                value=saved['stats'].get(key,0)
                if type(value)is int and value>=0:self.stats[key]=value
