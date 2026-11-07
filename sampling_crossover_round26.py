"""Durable exact optimization of bounded recombination neighborhoods (F16).

Every result is a heuristic candidate for the original puzzle, never a search
exclusion. A completed neighborhood means only its explicitly defined parental
cycle-closure/rotation product has been optimized. Work leases, eviction and
interrupted partial factor queries do not establish global exhaustion.
"""
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import itertools,json
from sampling_crossover_round25 import _board,_digest,_encoded,_integer
from validator import reverse_mask
from constraint_region_branch_price_round26 import loop_cut,cut_coefficient,validate_cut
from constraint_regions import HierarchicalRegionPruner

VERSION=1
MAX_SNAPSHOT_BYTES=32*1024*1024
STAT_KEYS=('recombination_rounds','recombination_parent_scores','recombination_parent_cache_hits',
 'recombination_parent_duplicates','recombination_pair_selections','recombination_pair_evictions',
 'recombination_models_built','recombination_model_rebuilds','recombination_models_reused','recombination_models_evicted',
 'recombination_model_completions','recombination_rotation_models','recombination_multiparent_models',
 'recombination_component_visits','recombination_pattern_options','recombination_nodes','recombination_bound_prunes',
 'recombination_seam_factors','recombination_seam_pair_checks','recombination_factor_cache_hits',
 'recombination_factor_cache_evictions','recombination_candidate_scores','recombination_duplicate_child_hits',
 'recombination_gold_validations','recombination_gold_cuts','recombination_gold_cut_checks','recombination_cut_factor_evaluations','recombination_cut_factor_cache_hits','recombination_cut_factor_cache_evictions','recombination_cut_bounds',
 'recombination_improvements','recombination_gold_solutions','recombination_offers','recombination_injections',
 'recombination_history_evictions','recombination_outcome_reuses','recombination_outcome_evictions',
 'recombination_offer_score_checks','recombination_source_checks','recombination_restores','recombination_adaptive_expansions')
MODEL_KEYS=tuple(k for k in STAT_KEYS if k in ('recombination_component_visits','recombination_pattern_options',
 'recombination_nodes','recombination_bound_prunes','recombination_seam_factors','recombination_seam_pair_checks',
 'recombination_factor_cache_hits','recombination_factor_cache_evictions','recombination_gold_cuts',
 'recombination_gold_cut_checks','recombination_cut_factor_evaluations','recombination_cut_factor_cache_hits','recombination_cut_factor_cache_evictions','recombination_cut_bounds'))


def _values(mask):
    while mask:
        bit=mask&-mask;yield bit.bit_length()-1;mask-=bit


def closure_components(parents):
    """Smallest cell blocks with identical physical-piece sets in every parent."""
    n=len(parents[0]);owner={code//3:cell for cell,code in enumerate(parents[0])};dsu=list(range(n))
    def find(v):
        while dsu[v]!=v:dsu[v]=dsu[dsu[v]];v=dsu[v]
        return v
    for row in parents[1:]:
        for cell,code in enumerate(row):dsu[find(cell)]=find(owner[code//3])
    groups={}
    for cell in range(n):groups.setdefault(find(cell),[]).append(cell)
    return tuple(sorted(tuple(cells) for cells in groups.values()))


def _legal(problem,cell,code):
    return code in problem.domains[cell] and (cell not in problem.fixed or problem.fixed[cell]==code) and all(problem.neighbors[cell,side]>=0 or not int(problem.masks[code,side]) for side in range(3))


def _rank(metrics,gold,gold_aware):return (metrics['gold_objective' if gold_aware else 'objective'],int(gold))


def _validate_metrics(problem,metrics,gold_aware):
    keys={'matched_edges','required_edges','domain_violations','objective'}
    if gold_aware:keys|={'closed_proper_loops','gold_objective'}
    if type(metrics)is not dict or set(metrics)!=keys or any(type(value)is not int or value<0 for value in metrics.values()):raise ValueError('Invalid cached objective values.')
    if not 0<=metrics['matched_edges']<=len(problem.edges) or metrics['required_edges']!=len(problem.edges) or metrics['domain_violations']>problem.n or metrics['objective']!=(problem.n-metrics['domain_violations'])*(len(problem.edges)+1)+metrics['matched_edges']:raise ValueError('Inconsistent cached edge objective.')
    if gold_aware and (metrics['closed_proper_loops']>problem.total_segments or metrics['gold_objective']!=metrics['objective']*(problem.total_segments+1)+problem.total_segments-metrics['closed_proper_loops']):raise ValueError('Inconsistent cached Gold objective.')


class Neighborhood:
    """Exact retained branch-and-bound; static factors rebuild, search does not."""
    def __init__(self,problem,parents,rotations,gold_aware,*,state=None):
        self.problem=problem;self.parents=tuple(parents);self.rotations=rotations;self.gold_aware=gold_aware
        self.owner=HierarchicalRegionPruner(problem,'gold');self.stats=dict.fromkeys(MODEL_KEYS,0)
        self.components=closure_components(parents);self.groups=[];root=[0]*problem.n
        self.stats['recombination_component_visits']=problem.n*len(parents)
        for cells in self.components:
            choices=[]
            for row in parents:
                pattern=tuple(sum(1<<code for code in (range(3*(row[cell]//3),3*(row[cell]//3)+3) if rotations else (row[cell],)) if _legal(problem,cell,code)) for cell in cells)
                self.stats['recombination_pattern_options']+=1
                if all(pattern) and pattern not in choices:choices.append(pattern)
            choices.sort();self.groups.append((cells,tuple(choices)))
            for index,cell in enumerate(cells):
                for choice in choices:root[cell]|=choice[index]
        self.root=tuple(root)
        self.remaining=tuple(i for i,(_,choices) in enumerate(self.groups) if len(choices)>1)
        self.stack=[self._frame(root,self.remaining)];self.best=None;self.complete=False;self.cuts=[];self.factor_cache=OrderedDict();self.cut_factors=OrderedDict()
        if state is not None:self._restore(state)
    @staticmethod
    def _frame(masks,remaining):return {'masks':list(masks),'remaining':list(remaining),'phase':'bound','edge':0,'matched':0,'cut':0,'no_gold':False,'branch':None,'next':0}
    def _factor(self,eid,left,right,check):
        key=(eid,left,right)
        if key in self.factor_cache:
            self.stats['recombination_factor_cache_hits']+=1;self.factor_cache.move_to_end(key);return self.factor_cache[key]
        self.stats['recombination_seam_factors']+=1
        _,sa,_,sb=self.problem.edges[eid];answer=0
        for a in _values(left):
            for b in _values(right):
                check();self.stats['recombination_seam_pair_checks']+=1
                if int(self.problem.masks[a,sa])==reverse_mask(int(self.problem.masks[b,sb])):answer=1;break
            if answer:break
        self.factor_cache[key]=answer
        if len(self.factor_cache)>8192:self.factor_cache.popitem(last=False);self.stats['recombination_factor_cache_evictions']+=1
        return answer
    def _cut_maximum(self,masks,index,check):
        result=0;cut=self.cuts[index]
        for cell,domain in enumerate(masks):
            check();self.stats['recombination_gold_cut_checks']+=1;key=(index,cell,domain)
            if key in self.cut_factors:
                value=self.cut_factors[key];self.cut_factors.move_to_end(key);self.stats['recombination_cut_factor_cache_hits']+=1
            else:
                value=None
                for code in _values(domain):
                    check();self.stats['recombination_cut_factor_evaluations']+=1
                    coefficient=cut_coefficient(self.owner,cell,code,cut)
                    value=coefficient if value is None else max(value,coefficient)
                self.cut_factors[key]=value
                if len(self.cut_factors)>8192:self.cut_factors.popitem(last=False);self.stats['recombination_cut_factor_cache_evictions']+=1
            result+=value
        return result
    def _upper(self,frame):
        base=self.problem.n*(len(self.problem.edges)+1)+frame['matched'];total=self.problem.total_segments
        score=base*(total+1)+total if self.gold_aware else base
        gold=not frame['no_gold']
        if frame['no_gold'] and self.gold_aware and total>=2:
            # Every legal edge-perfect board has closed strands. If it is not
            # one Gold cycle, at least two proper cycles remain. With an
            # unmatched edge its base objective is already lower by total+1.
            maximum=(self.problem.n*(len(self.problem.edges)+1)+len(self.problem.edges))*(total+1)+total
            score=min(score,maximum-2)
        return score,int(gold)
    def seed(self,records):
        for row,record in records.items():
            if all(mask>>code&1 for mask,code in zip(self.root,row)):
                candidate={'codes':list(row),'metrics':dict(record['metrics']),'gold':record['gold']}
                if self.best is None or _rank(candidate['metrics'],candidate['gold'],self.gold_aware)>_rank(self.best['metrics'],self.best['gold'],self.gold_aware):self.best=candidate
    def _pop(self):
        self.stack.pop()
        if not self.stack:self.complete=True
    def advance(self,lease,score,check=lambda:None):
        for _ in range(lease):
            check()
            if self.complete:return
            if not self.stack:self.complete=True;return
            frame=self.stack[-1];masks=frame['masks']
            if frame['phase']=='bound':
                if not all(masks):self._pop();self.stats['recombination_bound_prunes']+=1;continue
                while frame['edge']<len(self.problem.edges):
                    eid=frame['edge'];a,_,b,_=self.problem.edges[eid]
                    value=self._factor(eid,masks[a],masks[b],check)
                    frame['matched']+=value;frame['edge']+=1
                while frame['cut']<len(self.cuts):
                    if self._cut_maximum(masks,frame['cut'],check)<-2:
                        frame['no_gold']=True;self.stats['recombination_cut_bounds']+=1
                    frame['cut']+=1
                self.stats['recombination_nodes']+=1
                if self.best is not None and self._upper(frame)<=_rank(self.best['metrics'],self.best['gold'],self.gold_aware):
                    self._pop();self.stats['recombination_bound_prunes']+=1;continue
                if not frame['remaining'] and all(mask.bit_count()==1 for mask in masks):
                    codes=tuple(mask.bit_length()-1 for mask in masks);record=score(codes)
                    if record['metrics']['matched_edges']!=frame['matched'] or record['metrics']['domain_violations']:
                        raise RuntimeError('Recombination seam factor/independent score disagrees.')
                    candidate={'codes':list(codes),'metrics':dict(record['metrics']),'gold':record['gold']}
                    if self.best is None or _rank(candidate['metrics'],candidate['gold'],self.gold_aware)>_rank(self.best['metrics'],self.best['gold'],self.gold_aware):self.best=candidate
                    if not candidate['gold'] and frame['matched']==len(self.problem.edges):
                        cut=loop_cut(self.owner,codes)
                        if cut is not None and cut not in self.cuts and len(self.cuts)<32:self.cuts.append(cut);self.stats['recombination_gold_cuts']+=1
                    self._pop();continue
                if frame['remaining']:
                    group=max(frame['remaining'],key=lambda g:(len(self.groups[g][0]),len(self.groups[g][1]),-g))
                    frame['branch']=['group',group]
                else:
                    cell=min((c for c,d in enumerate(masks) if d.bit_count()>1),key=lambda c:(masks[c].bit_count(),c))
                    frame['branch']=['rotation',cell]
                frame['phase']='branch'
            kind,index=frame['branch']
            choices=self.groups[index][1] if kind=='group' else tuple(1<<code for code in _values(masks[index]))
            if frame['next']==len(choices):self._pop();continue
            choice=choices[frame['next']];child=masks.copy();remaining=frame['remaining'].copy()
            if kind=='group':
                for cell,mask in zip(self.groups[index][0],choice):child[cell]&=mask
                remaining.remove(index)
            else:child[index]=choice
            frame['next']+=1;self.stack.append(self._frame(child,remaining))
        if not self.stack:self.complete=True
    def snapshot(self):
        return deepcopy({'stack':self.stack,'best':self.best,'complete':self.complete,'cuts':self.cuts,'stats':self.stats,
          'factors':[[*key,value] for key,value in self.factor_cache.items()],
          'cut_factors':[[*key,value] for key,value in self.cut_factors.items()]})
    def _restore(self,state):
        state=deepcopy(state)
        if type(state)is not dict or set(state)!={'stack','best','complete','cuts','stats','factors','cut_factors'} or type(state['complete'])is not bool:raise ValueError('Invalid optimizer continuation.')
        if type(state['stack'])is not list or len(state['stack'])>2*self.problem.n+1 or state['complete']!=(not state['stack']):raise ValueError('Invalid optimizer frontier/completion.')
        if type(state['stats'])is not dict or set(state['stats'])!=set(MODEL_KEYS) or any(not _integer(v,0,2**53-1) for v in state['stats'].values()):raise ValueError('Invalid optimizer work counters.')
        previous=None
        for frame in state['stack']:
            if type(frame)is not dict or set(frame)!=set(self.stack[0]) or frame['phase'] not in ('bound','branch') or type(frame['no_gold'])is not bool:raise ValueError('Invalid optimizer frame.')
            masks=frame['masks'];remaining=frame['remaining']
            if type(masks)is not list or len(masks)!=self.problem.n or any(type(d)is not int or d<0 or d&~self.root[c] for c,d in enumerate(masks)):raise ValueError('Optimizer frame escapes model.')
            if type(remaining)is not list or len(remaining)!=len(set(remaining)) or any(type(g)is not int or g not in self.remaining for g in remaining):raise ValueError('Invalid remaining component set.')
            if previous is not None and (any(d&~old for d,old in zip(masks,previous['masks'])) or not set(remaining)<=set(previous['remaining'])):raise ValueError('Optimizer frame is not a descendant.')
            if not _integer(frame['edge'],0,len(self.problem.edges)) or not _integer(frame['matched'],0,frame['edge']) or not _integer(frame['cut'],0,len(state['cuts'])) or not _integer(frame['next'],0,24):raise ValueError('Invalid optimizer step.')
            branch=frame['branch']
            if frame['phase']=='branch':
                if type(branch)is not list or len(branch)!=2 or branch[0] not in ('group','rotation') or type(branch[1])is not int:raise ValueError('Invalid optimizer branch.')
                kind,index=branch
                if kind=='group':
                    if index not in remaining or frame['next']>len(self.groups[index][1]):raise ValueError('Invalid component branch position.')
                elif not 0<=index<self.problem.n or frame['next']>masks[index].bit_count():raise ValueError('Invalid rotation branch position.')
            elif branch is not None or frame['next']!=0:raise ValueError('Unevaluated optimizer branch.')
            previous=frame
        cuts=[validate_cut(self.problem,cut) for cut in state['cuts']]
        if len(cuts)>32 or len(set(cuts))!=len(cuts):raise ValueError('Invalid optimizer cut family.')
        best=state['best']
        if best is not None:
            if type(best)is not dict or set(best)!={'codes','metrics','gold'} or type(best['gold'])is not bool:raise ValueError('Invalid optimizer incumbent.')
            row=_board(self.problem,best['codes']);_validate_metrics(self.problem,best['metrics'],self.gold_aware)
            if any(not _legal(self.problem,c,v) for c,v in enumerate(row)):raise ValueError('Optimizer incumbent is not legal.')
        factors=OrderedDict()
        if type(state['factors'])is not list or len(state['factors'])>8192:raise ValueError('Factor cache cap exceeded.')
        for row in state['factors']:
            if type(row)is not list or len(row)!=4 or not _integer(row[0],0,len(self.problem.edges)-1) or any(type(v)is not int or v<0 for v in row[1:3]) or not _integer(row[3],0,1):raise ValueError('Invalid seam factor cache.')
            eid,left,right,value=row;a,_,b,_=self.problem.edges[eid]
            if left&~self.root[a] or right&~self.root[b] or (eid,left,right) in factors:raise ValueError('Factor cache source differs.')
            factors[eid,left,right]=value
        cut_factors=OrderedDict()
        if type(state['cut_factors'])is not list or len(state['cut_factors'])>8192:raise ValueError('Cut factor cache cap exceeded.')
        for row in state['cut_factors']:
            if type(row)is not list or len(row)!=4 or not _integer(row[0],0,len(cuts)-1) or not _integer(row[1],0,self.problem.n-1) or type(row[2])is not int or row[2]<=0 or row[2]&~self.root[row[1]] or type(row[3])is not int or not -4<=row[3]<=3*self.problem.total_segments or tuple(row[:3]) in cut_factors:raise ValueError('Invalid exact cut factor cache.')
            cut_factors[tuple(row[:3])]=row[3]
        self.cut_factors=cut_factors
        self.stack=state['stack'];self.best=best;self.complete=state['complete'];self.cuts=cuts;self.stats=state['stats'];self.factor_cache=factors


class OptimizingCrossover:
    def __init__(self,problem,*,gold_aware=False,max_nodes=256,max_models=8,history_limit=256,max_parents=4,resume=None):
        if not _integer(problem.n,1,160) or type(gold_aware)is not bool or not _integer(max_nodes,1,65536) or not _integer(max_models,1,64) or not _integer(history_limit,1,4096) or not _integer(max_parents,2,8):raise ValueError('Invalid recombination policy.')
        self.problem=problem;self.policy={'gold_aware':gold_aware,'max_nodes':max_nodes,'max_models':max_models,'history_limit':history_limit,'max_parents':max_parents}
        self.binding=self._identity();self._stats=dict.fromkeys(STAT_KEYS,0);self.models=OrderedDict();self.last={};self.pairs=OrderedDict();self.history=OrderedDict();self.outcomes=OrderedDict();self.pending=False;self.pending_offer=None
        if resume is not None:self._restore(resume)
    def _identity(self):
        from constraint_region_compiler_round26 import _source
        p=self.problem
        sources=dict(_source())
        for name in ('sampling_crossover_round26.py','sampling_crossover_round25.py','gpu_sampling.py','constraint_lns.py','search_problem.py','validator.py','geometry.py'):
            sources[name]=sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return _digest({'version':VERSION,'policy':self.policy,'fingerprint':p.fingerprint,'tiles':p.tiles,'edges':p.edges,'board_edges':p.board.edges,'fixed':sorted(p.fixed.items()),'domains':p.domains,'masks':p.masks.tolist(),'neighbors':p.neighbors.tolist(),'sides':p.sides.tolist(),'segments':p.total_segments,'segment_ports':p._segment_ports,'sources':sources})
    def _check(self):
        if self._identity()!=self.binding:raise ValueError('Recombination source/input/policy changed.')
        self._stats['recombination_source_checks']+=1
    @property
    def stats(self):return dict(self._stats)
    def report(self):return self.stats
    def _score(self,codes,parent=False):
        if codes in self.history:
            self._stats['recombination_parent_cache_hits' if parent else 'recombination_duplicate_child_hits']+=1
            self.history.move_to_end(codes);return self.history[codes]
        from gpu_sampling import gold_score,score_board
        metrics=(gold_score if self.policy['gold_aware'] else score_board)(self.problem,list(codes))
        self._stats['recombination_parent_scores' if parent else 'recombination_candidate_scores']+=1
        gold=False
        if not metrics['domain_violations'] and metrics['matched_edges']==len(self.problem.edges):
            gold=bool(self.problem.validate(codes)['valid']);self._stats['recombination_gold_validations']+=1
        self._check();record={'metrics':metrics,'gold':gold,'offered':False}
        self.history[codes]=record
        while len(self.history)>self.policy['history_limit']:self.history.popitem(last=False);self._stats['recombination_history_evictions']+=1
        return record
    @staticmethod
    def _key(parents,rotations):return tuple(parents),rotations
    def _sync(self,key,model):
        before=self.last.get(key,{})
        for name,value in model.stats.items():self._stats[name]+=value-before.get(name,0)
        self.last[key]=dict(model.stats)
    def _offer(self,best,threshold):
        if best is None:return None,None,None
        row=tuple(best['codes']);record=self._score(row)
        if record['offered'] or (not record['gold'] and record['metrics']['gold_objective' if self.policy['gold_aware'] else 'objective']<=threshold):return None,None,None
        if any(not _legal(self.problem,c,v) for c,v in enumerate(row)):raise RuntimeError('Optimizer offered an illegal placement.')
        from gpu_sampling import gold_score,score_board
        verified=(gold_score if self.policy['gold_aware'] else score_board)(self.problem,list(row))
        self._stats['recombination_offer_score_checks']+=1
        verified_gold=bool(self.problem.validate(row)['valid']) if verified['matched_edges']==len(self.problem.edges) else False
        self._stats['recombination_gold_validations']+=int(verified['matched_edges']==len(self.problem.edges))
        if verified!=record['metrics'] or verified_gold!=record['gold']:raise ValueError('Saved candidate failed independent publication validation.')
        self._check();record['offered']=True;self.pending=True;self._stats['recombination_offers']+=1
        if record['gold']:self._stats['recombination_gold_solutions']+=1
        if record['metrics']['gold_objective' if self.policy['gold_aware'] else 'objective']>threshold:self._stats['recombination_improvements']+=1
        return list(row),dict(record['metrics']),list(row) if record['gold'] else None
    def propose(self,archive,check=lambda:None):
        self._check()
        if type(archive)is not list or len(archive)>8:raise ValueError('Recombination archive is bounded at eight parents.')
        rows=[]
        for row in archive:
            if type(row)is not dict or 'codes' not in row:raise ValueError('Recombination archive rows require codes.')
            rows.append(_board(self.problem,row['codes']))
        unique=tuple(sorted(set(rows)));check();records={}
        for row in unique:check();records[row]=self._score(row,parent=True)
        if self.pending_offer is not None:
            best,threshold=self.pending_offer;check();candidate,metrics,solution=self._offer(best,threshold);self.pending_offer=None
            return {'candidate':candidate,'metrics':metrics,'solution':solution,'stats':self.stats}
        self._stats['recombination_rounds']+=1;self._stats['recombination_parent_duplicates']+=len(rows)-len(unique);self.pending=False
        pairs=tuple(itertools.combinations(unique,2));present=set(pairs)
        for pair in tuple(self.pairs):
            if pair not in present:del self.pairs[pair];self._stats['recombination_pair_evictions']+=1
        for pair in pairs:self.pairs.setdefault(pair,{'stage':0,'completed':0,'gain':0,'cost':0,'last':0,'pending':None})
        stages=2*(min(self.policy['max_parents'],len(unique))-1)
        eligible=[pair for pair in pairs if self.pairs[pair]['stage']<stages or self.pairs[pair]['pending'] is not None]
        if not eligible:return {'candidate':None,'metrics':None,'solution':None,'stats':self.stats}
        roundno=self._stats['recombination_rounds']
        if roundno%4==1:pair=min(eligible,key=lambda pair:(self.pairs[pair]['last'],pair))
        else:pair=max(eligible,key=lambda pair:(Fraction(self.pairs[pair]['gain'],max(1,self.pairs[pair]['cost'])),-self.pairs[pair]['last'],tuple(-v for row in pair for v in row)))
        record=self.pairs[pair];record['last']=roundno;self._stats['recombination_pair_selections']+=1
        if record['pending'] is None:
            count=2+record['stage']//2
            extras=sorted((row for row in unique if row not in pair),key=lambda row:(-records[row]['metrics']['gold_objective' if self.policy['gold_aware'] else 'objective'],row))
            parents=tuple(sorted((*pair,*extras[:count-2])));rotations=bool(record['stage']%2)
            record['pending']={'parents':[list(row) for row in parents],'rotations':rotations}
        else:parents=tuple(tuple(row) for row in record['pending']['parents']);rotations=record['pending']['rotations']
        key=self._key(parents,rotations);threshold=max(self._score(row,parent=True)['metrics']['gold_objective' if self.policy['gold_aware'] else 'objective'] for row in parents)
        if key in self.outcomes:
            self._stats['recombination_outcome_reuses']+=1;outcome=self.outcomes[key];self.outcomes.move_to_end(key);model=None
        else:
            if key not in self.models:
                model=Neighborhood(self.problem,parents,rotations,self.policy['gold_aware']);model.seed({row:self._score(row,parent=True) for row in parents})
                self.models[key]=model;self._stats['recombination_models_built']+=1
                self._stats['recombination_rotation_models']+=int(rotations);self._stats['recombination_multiparent_models']+=int(len(parents)>2)
            else:model=self.models[key];self._stats['recombination_models_reused']+=1
            self.models.move_to_end(key)
            try:model.advance(self.policy['max_nodes'],self._score,check)
            finally:
                self._sync(key,model)
                while len(self.models)>self.policy['max_models']:
                    old,_=self.models.popitem(last=False);self.last.pop(old,None);self._stats['recombination_models_evicted']+=1
            outcome={'best':deepcopy(model.best),'nodes':model.stats['recombination_nodes'],'complete':model.complete}
            if model.complete:
                self._stats['recombination_model_completions']+=1;self.outcomes[key]=deepcopy(outcome)
                while len(self.outcomes)>64:self.outcomes.popitem(last=False);self._stats['recombination_outcome_evictions']+=1
            while len(self.models)>self.policy['max_models']:
                old,_=self.models.popitem(last=False);self.last.pop(old,None);self._stats['recombination_models_evicted']+=1
        if outcome['complete']:
            gain=0 if outcome['best'] is None else max(0,outcome['best']['metrics']['gold_objective' if self.policy['gold_aware'] else 'objective']-threshold)
            record['completed']+=1;record['gain']+=gain;record['cost']+=outcome['nodes'];record['stage']+=1;record['pending']=None
            if record['stage']<stages:self._stats['recombination_adaptive_expansions']+=1
        self.pending_offer=[deepcopy(outcome['best']),threshold];check()
        candidate,metrics,solution=self._offer(outcome['best'],threshold);self.pending_offer=None
        return {'candidate':candidate,'metrics':metrics,'solution':solution,'stats':self.stats}
    def record_injection(self):
        self._check()
        if not self.pending:raise RuntimeError('No pending optimized candidate injection.')
        self.pending=False;self._stats['recombination_injections']+=1
    def snapshot(self):
        self._check()
        state={'stats':self.stats,'pending':self.pending,'pending_offer':deepcopy(self.pending_offer),'pairs':[[[list(row) for row in key],deepcopy(value)] for key,value in self.pairs.items()],
         'history':[[list(row),deepcopy(record)] for row,record in self.history.items()],
         'models':[{'parents':[list(row) for row in key[0]],'rotations':key[1],'state':model.snapshot()} for key,model in self.models.items()],
         'outcomes':[{'parents':[list(row) for row in key[0]],'rotations':key[1],'outcome':deepcopy(value)} for key,value in self.outcomes.items()]}
        result={'optimizing_crossover_version':VERSION,'binding':self.binding,'policy':dict(self.policy),'state':state};result['digest']=_digest(result)
        if len(_encoded(result))>MAX_SNAPSHOT_BYTES:raise ValueError('Recombination continuation exceeds byte cap.')
        return result
    def _restore(self,value):
        try:
            if len(_encoded(value))>MAX_SNAPSHOT_BYTES:raise ValueError('Recombination continuation exceeds byte cap.')
            value=json.loads(_encoded(value))
        except (TypeError,OverflowError,RecursionError) as error:raise ValueError('Malformed recombination continuation.') from error
        if type(value)is not dict or set(value)!={'optimizing_crossover_version','binding','policy','state','digest'} or not _integer(value['optimizing_crossover_version'],VERSION,VERSION) or value['binding']!=self.binding or type(value['policy'])is not dict or set(value['policy'])!=set(self.policy) or any(type(value['policy'][k])is not type(v) or value['policy'][k]!=v for k,v in self.policy.items()) or value['digest']!=_digest({k:v for k,v in value.items() if k!='digest'}):raise ValueError('Recombination source/scope/policy/digest differs.')
        state=value['state']
        if type(state)is not dict or set(state)!={'stats','pending','pending_offer','pairs','history','models','outcomes'} or type(state['pending'])is not bool or type(state['stats'])is not dict or set(state['stats'])!=set(STAT_KEYS) or any(not _integer(v,0,2**53-1) for v in state['stats'].values()):raise ValueError('Invalid recombination controller state.')
        if not state['stats']['recombination_injections']<=state['stats']['recombination_offers'] or state['pending'] and state['stats']['recombination_injections']>=state['stats']['recombination_offers']:raise ValueError('Invalid recombination injection state.')
        def parents_of(rows):
            if type(rows)is not list or not 2<=len(rows)<=self.policy['max_parents']:raise ValueError('Invalid neighborhood parent count.')
            result=tuple(_board(self.problem,row) for row in rows)
            if result!=tuple(sorted(set(result))):raise ValueError('Noncanonical neighborhood parents.')
            return result
        pairs=OrderedDict()
        if type(state['pairs'])is not list or len(state['pairs'])>28:raise ValueError('Pair policy cap exceeded.')
        for key,record in state['pairs']:
            pair=parents_of(key)
            if len(pair)!=2 or pair in pairs or type(record)is not dict or set(record)!={'stage','completed','gain','cost','last','pending'} or any(not _integer(record[k],0,2**53-1) for k in ('completed','gain','cost','last')) or not _integer(record['stage'],0,2*(self.policy['max_parents']-1)):raise ValueError('Invalid adaptive pair history.')
            if record['completed']!=record['stage'] or record['last']>state['stats']['recombination_rounds']:raise ValueError('Adaptive history lacks a completed outcome.')
            pending=record['pending']
            if pending is not None:
                if type(pending)is not dict or set(pending)!={'parents','rotations'} or type(pending['rotations'])is not bool or not set(pair)<=set(parents_of(pending['parents'])) or len(pending['parents'])!=2+record['stage']//2 or pending['rotations']!=bool(record['stage']%2):raise ValueError('Invalid pending neighborhood.')
            pairs[pair]=record
        history=OrderedDict()
        if type(state['history'])is not list or len(state['history'])>self.policy['history_limit']:raise ValueError('History cap exceeded.')
        objective='gold_objective' if self.policy['gold_aware'] else 'objective'
        def valid_record(record):
            if type(record)is not dict or set(record)!={'metrics','gold','offered'} or type(record['gold'])is not bool or type(record['offered'])is not bool or type(record['metrics'])is not dict or objective not in record['metrics'] or any(type(v)is not int or v<0 for v in record['metrics'].values()):raise ValueError('Invalid completed score record.')
            _validate_metrics(self.problem,record['metrics'],self.policy['gold_aware'])
        for row,record in state['history']:
            board=_board(self.problem,row);valid_record(record)
            if board in history:raise ValueError('Duplicate completed child record.')
            history[board]=record
        models=OrderedDict();last={};setup={k:0 for k in MODEL_KEYS}
        if type(state['models'])is not list or len(state['models'])>self.policy['max_models']:raise ValueError('Model cap exceeded.')
        for item in state['models']:
            if type(item)is not dict or set(item)!={'parents','rotations','state'} or type(item['rotations'])is not bool:raise ValueError('Invalid retained model.')
            parents=parents_of(item['parents']);key=self._key(parents,item['rotations'])
            if key in models:raise ValueError('Duplicate retained model.')
            model=Neighborhood(self.problem,parents,item['rotations'],self.policy['gold_aware'])
            for name,count in model.stats.items():setup[name]+=count
            model._restore(item['state']);models[key]=model;last[key]=dict(model.stats)
        outcomes=OrderedDict()
        if type(state['outcomes'])is not list or len(state['outcomes'])>64:raise ValueError('Outcome cap exceeded.')
        for item in state['outcomes']:
            if type(item)is not dict or set(item)!={'parents','rotations','outcome'} or type(item['rotations'])is not bool:raise ValueError('Invalid completed neighborhood.')
            parents=parents_of(item['parents']);key=self._key(parents,item['rotations']);outcome=item['outcome']
            if key in outcomes or type(outcome)is not dict or set(outcome)!={'best','nodes','complete'} or outcome['complete'] is not True or not _integer(outcome['nodes'],0,2**53-1):raise ValueError('Invalid completed optimizer outcome.')
            best=outcome['best']
            if best is not None:
                if type(best)is not dict or set(best)!={'codes','metrics','gold'} or type(best['gold'])is not bool:raise ValueError('Invalid completed incumbent.')
                _board(self.problem,best['codes']);valid_record({'metrics':best['metrics'],'gold':best['gold'],'offered':False})
            outcomes[key]=outcome
        pending_offer=state['pending_offer']
        if pending_offer is not None:
            if type(pending_offer)is not list or len(pending_offer)!=2 or not _integer(pending_offer[1],0,2**53-1):raise ValueError('Invalid pending candidate publication.')
            best=pending_offer[0]
            if best is not None:
                if type(best)is not dict or set(best)!={'codes','metrics','gold'} or type(best['gold'])is not bool:raise ValueError('Invalid pending incumbent.')
                _board(self.problem,best['codes']);valid_record({'metrics':best['metrics'],'gold':best['gold'],'offered':False})
        self.pending_offer=pending_offer
        self._stats=state['stats'];self.pending=state['pending'];self.pairs=pairs;self.history=history;self.models=models;self.last=last;self.outcomes=outcomes
        self._stats['recombination_restores']+=1;self._stats['recombination_model_rebuilds']+=len(models)
        for name,count in setup.items():self._stats[name]+=count
