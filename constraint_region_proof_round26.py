"""Checked sufficient-assumption regional failure cores (F11).

Pure local failures ignore every outside domain. Contracted failures additionally
bind every omitted cell to its checked physical placement; every strand of those
placements is reconstructed independently. A contained descendant query may
have fewer active cells than the proof, without graph-isomorphism assumptions.
Certificates only prune; importing/evicting them never marks search work done.
"""
from collections import OrderedDict
from pathlib import Path
import json
from constraint_region_certificate import build_certificate,verify_certificate,CertificateError
from constraint_region_incremental import source_binding
from constraint_region_compiler_round26 import _domains,_digest,_json_size,MAX_BYTES
from constraint_pruning import values
from constraint_separator import _Budget
from io_utils import atomic_json

STAT_KEYS=('regional_core_queries','regional_core_index_checks','regional_core_verified_import_hits','regional_core_checks','regional_core_verified_states',
 'regional_core_verified_transitions','regional_core_build_transitions','regional_core_build_states',
 'regional_core_generalization_trials','regional_core_generalized_assumptions','regional_core_reuses',
 'regional_core_contractions','regional_core_terminal_checks','regional_core_unknown',
 'regional_core_imported','regional_core_rejected','regional_core_evicted','regional_core_saved','regional_core_io_failures')


def _fixed_completion(owner,cells,signature,fixed,check):
    """Independent graph reconstruction of a full contracted Gold completion."""
    problem=owner.problem;members=set(cells);used,masks,pairs,cycles=signature
    chosen=dict(fixed)
    if set(chosen)!=set(range(problem.n))-members:return False
    if len({code//3 for code in chosen.values()})!=len(chosen) or any(used>>(code//3)&1 for code in chosen.values()):return False
    parent={};degree={};total_cycles=cycles
    def find(p):
        while parent[p]!=p:p=parent[p]
        return p
    def edge(a,b):
        nonlocal total_cycles
        for p in (a,b):parent.setdefault(p,p);degree[p]=degree.get(p,0)+1
        x,y=find(a),find(b)
        if x==y:total_cycles+=1
        else:parent[x]=y
    # Signature pairs represent paths, never crossings in a drawing.
    for a,b in pairs:edge(a,b)
    local_masks={}
    for cell,code in chosen.items():
        check();tile=problem.tiles[code//3];rotation=code%3;row=[0,0,0]
        for left,right in tile['segments']:
            a=((left[0]+rotation)%3)*11+left[1]-1;b=((right[0]+rotation)%3)*11+right[1]-1
            row[a//11]|=1<<(a%11);row[b//11]|=1<<(b%11);edge(33*cell+a,33*cell+b)
        local_masks[cell]=row
        if any(problem.neighbors[cell,s]<0 and row[s] for s in range(3)):return False
    frontier=tuple(i for i,(a,sa,b,sb) in enumerate(problem.edges) if (a in members)!=(b in members))
    boundary=dict(zip(frontier,masks))
    for eid,(a,sa,b,sb) in enumerate(problem.edges):
        check()
        if a in members and b in members:continue
        am=boundary[eid] if a in members else local_masks[a][sa]
        bm=boundary[eid] if b in members else sum(1<<(10-p) for p in range(11) if local_masks[b][sb]>>p&1)
        if am!=bm:return False
        for position in range(11):
            if am>>position&1:edge(33*a+11*sa+position,33*b+11*sb+10-position)
    if any(d!=2 for d in degree.values()):return False
    return owner.target!='gold' or total_cycles==1 and bool(parent) and len({find(p) for p in parent})==1


def verify_core(owner,record,check=lambda:None,*,max_states=50000,max_steps=1000000,stats=None):
    if type(record)is not dict or set(record)!={'schema','binding','cells','assumptions','fixed','certificate','sha256'} or record['schema']!=1:
        raise CertificateError('Malformed regional failure core.')
    payload={k:v for k,v in record.items() if k!='sha256'}
    if record['sha256']!=_digest(payload) or record['binding']!=_digest(source_binding(owner)) or _json_size(record)>MAX_BYTES:
        raise CertificateError('Regional core source, digest or size differs.')
    cells=tuple(record['cells']);original=tuple(sum(1<<v for v in set(row)) for row in owner.problem.domains)
    if not cells or cells!=tuple(sorted(set(cells))) or any(type(c)is not int or not 0<=c<owner.problem.n for c in cells):raise CertificateError('Invalid core cells.')
    assumptions=[];seen=set();domains=list(original)
    for item in record['assumptions']:
        if type(item)not in (tuple,list) or len(item)!=2:raise CertificateError('Invalid core assumption.')
        cell,mask=item
        if type(cell)is not int or not 0<=cell<owner.problem.n or cell in seen or type(mask)is not int or mask<=0 or mask&~original[cell] or mask==original[cell]:raise CertificateError('Invalid sufficient assumption scope.')
        seen.add(cell);assumptions.append((cell,mask));domains[cell]=mask
    if assumptions!=sorted(assumptions):raise CertificateError('Core assumptions are not canonical.')
    fixed=[]
    for item in record['fixed']:
        if type(item)not in (tuple,list) or len(item)!=2 or any(type(v)is not int for v in item):raise CertificateError('Invalid checked contraction.')
        cell,code=item
        if not 0<=cell<owner.problem.n or cell in cells or domains[cell]!=(1<<code):raise CertificateError('Contraction is not a forced physical placement.')
        fixed.append((cell,code))
    if len({c for c,v in fixed})!=len(fixed) or fixed!=sorted(fixed):raise CertificateError('Duplicate contracted cell.')
    if fixed and set(c for c,v in fixed)!=set(range(owner.problem.n))-set(cells):raise CertificateError('A contraction must account for every omitted cell.')
    if not fixed and any(c not in cells for c,m in assumptions):raise CertificateError('Pure local core has irrelevant outside assumptions.')
    work={'region_certificate_verified_states':0,'region_certificate_verified_transitions':0}
    try:checked=verify_certificate(owner,cells,domains,record['certificate'],check,max_states=max_states,max_steps=max_steps,stats=work)
    finally:
        if stats is not None:
            stats['regional_core_verified_states']+=work['region_certificate_verified_states']
            stats['regional_core_verified_transitions']+=work['region_certificate_verified_transitions']
    if not fixed:
        if checked['count']!=0:raise CertificateError('Local region has a completion.')
    else:
        for state in record['certificate']['states']:
            if state['level']==len(cells):
                check()
                if stats is not None:stats['regional_core_terminal_checks']+=1
                signature=state['signature']
                if _fixed_completion(owner,cells,signature,fixed,check):raise CertificateError('Contracted region has a Gold completion.')
    check();return tuple(assumptions)


def build_core(owner,cells,domains,check=lambda:None,*,max_states=20000,max_steps=200000,generalize=8,stats=None):
    callback=check;callback_errors=[]
    def check():
        try:callback()
        except BaseException as error:callback_errors.append(error);raise
    domains=_domains(owner.problem,domains);cells=tuple(sorted(cells));original=tuple(sum(1<<v for v in set(row)) for row in owner.problem.domains)
    if any(not d for d in domains):return None
    if type(generalize)is not int or generalize<0:raise ValueError('Invalid core generalization budget.')
    outside=tuple(c for c in range(owner.problem.n) if c not in cells)
    contracted=bool(outside) and all(domains[c].bit_count()==1 for c in outside)
    fixed=tuple((c,next(values(domains[c]))) for c in outside) if contracted else ()
    scope=list(original)
    for c in cells:scope[c]=domains[c]
    for c,v in fixed:scope[c]=1<<v
    def prove(query):
        buildstats={'region_certificate_compiler_transitions':0,'region_certificate_compiler_states':0,'region_certificate_export_transitions':0}
        try:certificate=build_certificate(owner,cells,query,check,max_states=max_states,max_steps=max_steps,stats=buildstats)
        finally:
            if stats is not None:
                stats['regional_core_build_transitions']+=buildstats['region_certificate_compiler_transitions']+buildstats['region_certificate_export_transitions']
                stats['regional_core_build_states']+=buildstats['region_certificate_compiler_states']
        payload={'schema':1,'binding':_digest(source_binding(owner)),'cells':cells,
                 'assumptions':tuple((c,d) for c,d in enumerate(query) if d!=original[c]),'fixed':fixed,'certificate':certificate}
        result={**payload,'sha256':_digest(payload)}
        try:verify_core(owner,result,check,max_states=max_states,max_steps=max_steps*8,stats=stats)
        except CertificateError:
            if callback_errors:raise
            return None
        return result
    try:
        result=prove(scope)
        if result is None:return None
        if stats is not None:stats['regional_core_contractions']+=int(bool(fixed))
        candidates=[c for c in cells if scope[c]!=original[c]]
        for cell in candidates[:generalize]:
            check();trial=scope.copy();trial[cell]=original[cell]
            if stats is not None:stats['regional_core_generalization_trials']+=1
            try:broader=prove(trial)
            except _Budget:
                if callback_errors:raise
                continue
            if broader is not None:
                scope=trial;result=broader
                if stats is not None:stats['regional_core_generalized_assumptions']+=1
        return result
    except _Budget:
        if callback_errors:raise
        if stats is not None:stats['regional_core_unknown']+=1
        return None


class RegionalProofReuse:
    def __init__(self,owner,limit=64):
        self.owner=owner;self.semantic=source_binding(owner);self.limit=limit;self.records=OrderedDict();self.index={};self.directory=None
        self.stats=dict.fromkeys(STAT_KEYS,0);self.solution=None;self.attempted=set()
    def configure_cache_dir(self,path):self.directory=None if path is None else Path(path)/'regional-cores-v1'
    def import_certificate(self,record,check=lambda:None):
        callback=check;callback_errors=[]
        def check():
            try:callback()
            except BaseException as error:callback_errors.append(error);raise
        if source_binding(self.owner)!=self.semantic:raise ValueError('Regional proof source changed.')
        key=record.get('sha256') if isinstance(record,dict) else None
        if key in self.records and _digest(record)==_digest(self.records[key][1]):
            check();self.records.move_to_end(key);self.stats['regional_core_verified_import_hits']+=1;return True
        try:assumptions=verify_core(self.owner,record,check,stats=self.stats)
        except (CertificateError,ValueError,TypeError,KeyError,IndexError,OverflowError):
            if callback_errors:raise
            self.stats['regional_core_rejected']+=1;return False
        key=record['sha256'];check()
        # Detached immutable-by-ownership artifact; caller mutation cannot alter authority.
        self.records[key]=(assumptions,json.loads(json.dumps(record)));self.records.move_to_end(key)
        anchor=None if not assumptions else min(assumptions,key=lambda item:(item[1].bit_count(),item[0]))
        self.index.setdefault(anchor,set()).add(key)
        self.stats['regional_core_imported']+=1
        while len(self.records)>self.limit:
            old,_=self.records.popitem(last=False);self.stats['regional_core_evicted']+=1
            for anchor in tuple(self.index):
                self.index[anchor].discard(old)
                if not self.index[anchor]:del self.index[anchor]
        return True
    def export_certificates(self):return [json.loads(json.dumps(record)) for assumptions,record in self.records.values()]
    def infeasible(self,domains,check=lambda:None,*,max_active=12,generalize=8):
        domains=_domains(self.owner.problem,domains);self.stats['regional_core_queries']+=1
        if source_binding(self.owner)!=self.semantic:raise ValueError('Regional proof source changed.')
        candidates=set()
        for anchor,keys in self.index.items():
            check();self.stats['regional_core_index_checks']+=1
            if anchor is None or not domains[anchor[0]]&~anchor[1]:candidates.update(keys)
        for key in tuple(self.records):
            if key not in candidates:continue
            assumptions,record=self.records[key]
            check();self.stats['regional_core_checks']+=1
            if all(not domains[c]&~mask for c,mask in assumptions):
                self.records.move_to_end(key);self.stats['regional_core_reuses']+=1;return True
        cells=tuple(c for c,d in enumerate(domains) if d.bit_count()>1)
        if not cells or len(cells)>max_active:return False
        key=(cells,domains)
        if key in self.attempted:return False
        record=None
        path=None if self.directory is None else self.directory/(_digest((self.semantic,key))+'.json')
        if path is not None:
            imported=None
            try:
                if path.is_file() and path.stat().st_size<=MAX_BYTES:imported=json.loads(path.read_text(encoding='utf-8'))
            except (OSError,ValueError):self.stats['regional_core_io_failures']+=1
            # Semantic replay and user cancellation are outside the IO fallback.
            if imported is not None and self.import_certificate(imported,check):record=imported
        if record is None:record=build_core(self.owner,cells,domains,check,generalize=generalize,stats=self.stats)
        check();self.attempted.add(key)
        if len(self.attempted)>128:self.attempted.pop()
        if record is None:return False
        if not self.import_certificate(record,check):raise RuntimeError('Generated regional core failed independent replay.')
        if path is not None:
            try:path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,record);self.stats['regional_core_saved']+=1
            except OSError:self.stats['regional_core_io_failures']+=1
        return True

    def prune(self,domains,check=lambda:None,**kwargs):
        if not self.infeasible(domains,check,**kwargs):return False
        check();domains[0]=0;return True
