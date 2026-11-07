"""Independently checked complete regional DAG artifacts (round25 #25).

Every domain choice, including rejected choices and dead reachable children,
is present. The verifier rebuilds transitions with a separate DSU/segment
implementation; hashes bind inputs but never replace semantic checking.
This format is distinct from trusted-local unfinished compiler continuations.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,os
from constraint_region_incremental import source_binding
from constraint_region_symbolic import DirectRegionSession,DirectRegionDiagram
from constraint_region_network import assignment_signature,joined_signature
from constraint_pruning import values
from constraint_separator import _Budget
from io_utils import atomic_json,read_json_shared

STAT_KEYS=('region_certificate_queries','region_certificate_completed','region_certificate_unknown',
 'region_certificate_memory_hits','region_certificate_loads','region_certificate_rejected',
 'region_certificate_saves','region_certificate_write_failures','region_certificate_saved_bytes',
 'region_certificate_compiler_transitions','region_certificate_export_transitions','region_certificate_compiler_states','region_certificate_verified_states',
 'region_certificate_verified_transitions','region_certificate_prunes')

class CertificateError(ValueError):pass


def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode('utf-8')).hexdigest()

def _freeze(value,depth=0):
    if depth>6:raise CertificateError('Regional proof nesting exceeds its schema.')
    if type(value)is int:return value
    if value is None:return None
    if isinstance(value,(tuple,list)):
        if len(value)>65536:raise CertificateError('Regional proof sequence exceeds its schema.')
        return tuple(_freeze(x,depth+1) for x in value)
    raise CertificateError('Nonintegral regional proof data.')

def _validated_query(owner,cells,domains):
    cells=tuple(cells);domains=tuple(domains)
    if not cells or any(type(c)is not int or not 0<=c<owner.problem.n for c in cells) or cells!=tuple(sorted(set(cells))):
        raise CertificateError('Invalid proof region.')
    if len(domains)!=owner.problem.n or any(type(mask)is not int or mask<0 or mask&~sum(1<<code for code in set(owner.problem.domains[c])) for c,mask in enumerate(domains)):
        raise CertificateError('Proof domains escape the original root.')
    return cells,domains


class _Envelope:
    def __init__(self,owner):self.owner=owner
    def __getattr__(self,name):return getattr(self.owner,name)
    def _outside_possible(self,*args):return True


def build_certificate(owner,cells,domains,check=lambda:None,*,max_states=4096,max_steps=50000,stats=None):
    cells,domains=_validated_query(owner,cells,domains)
    if any(type(bound)is not int or bound<0 for bound in (max_states,max_steps)):raise ValueError('Invalid regional proof compiler bounds.')
    if any(a==b for a,_,b,_ in owner.problem.edges):raise _Budget
    envelope=_Envelope(owner);cells=tuple(cells);session=DirectRegionSession(envelope,cells,domains,max_states,max_steps)
    try:
        while not session.advance(128,check):pass
    finally:
        if stats is not None:
            stats['region_certificate_compiler_transitions']+=session.steps;stats['region_certificate_compiler_states']+=session.states
    result=session.result();keys=tuple(sorted(session._memo,key=lambda key:(key[0],repr(key[1]))));ident={key:i for i,key in enumerate(keys)};states=[]
    for level,signature in keys:
        check();edges=[]
        if level<len(cells):
            cell=cells[level]
            for code in values(domains[cell]):
                check()
                if stats is not None:stats['region_certificate_export_transitions']+=1
                single=assignment_signature(envelope,(cell,),(code,),domains,check)
                following=None if single is None else single if level==0 else joined_signature(envelope,cells[:level],signature,(cell,),single,domains,check)
                child=None if following is None else ident[level+1,following]
                edges.append([code,child])
        states.append({'level':level,'signature':signature,'edges':edges})
    diagram=object.__new__(DirectRegionDiagram);diagram.__dict__.update(result)
    return {'schema':1,'binding':digest(source_binding(owner)),'cells':cells,'domains':tuple(domains),
            'root':ident[0,None],'states':states,'count':diagram.rows,'supports':diagram.supports(check)}


class _IndependentTransitions:
    def __init__(self,owner,cells,check):
        self.problem=owner.problem;self.target=owner.target;self.cells=cells;self.check=check;self.edges=tuple(self.problem.edges)
        if any(a==b for a,_,b,_ in self.edges):raise CertificateError('Self-seam proofs are not supported.')
        self.pairs=[];self.masks=[]
        for tile in self.problem.tiles:
            for rotation in range(3):
                check();pairs=[];masks=[0,0,0]
                for a,b in tile['segments']:
                    left=((a[0]+rotation)%3)*11+a[1]-1;right=((b[0]+rotation)%3)*11+b[1]-1
                    pairs.append((left,right));masks[left//11]|=1<<(left%11);masks[right//11]|=1<<(right%11)
                self.pairs.append(tuple(pairs));self.masks.append(tuple(masks))
        self.frontiers=[]
        for level in range(len(cells)+1):
            members=set(cells[:level]);self.frontiers.append(tuple(i for i,(a,_,b,_) in enumerate(self.edges) if (a in members)!=(b in members)))
    @staticmethod
    def reverse(mask):return sum(1<<(10-p) for p in range(11) if mask>>p&1)
    def next(self,level,signature,code):
        self.check();cell=self.cells[level];members=set(self.cells[:level]);used,masks,pairs,cycles=(0,(),(),0) if signature is None else signature
        tile=code//3
        if used>>tile&1:return None
        local=self.masks[code]
        if any(self.problem.neighbors[cell,side]<0 and local[side] for side in range(3)):return None
        frontier=dict(zip(self.frontiers[level],masks));parent={};consumed=set()
        def find(p):
            while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
            return p
        def pair(a,b):
            if a in parent or b in parent or a==b:raise CertificateError('Invalid regional exposed matching.')
            parent[a]=a;parent[b]=a
        for a,b in pairs:pair(a,b)
        for a,b in self.pairs[code]:pair(33*cell+a,33*cell+b)
        for eid,(a,sa,b,sb) in enumerate(self.edges):
            if cell not in (a,b):continue
            self.check();other=b if cell==a else a;pattern=local[sa] if cell==a else self.reverse(local[sb])
            if other in members:
                if frontier.get(eid)!=pattern:return None
                del frontier[eid]
                for p in range(11):
                    if not pattern>>p&1:continue
                    left=33*a+11*sa+p;right=33*b+11*sb+10-p
                    if left not in parent or right not in parent:raise CertificateError('Seam endpoint absent from exact strand relation.')
                    ra,rb=find(left),find(right)
                    if ra==rb:cycles+=1
                    else:parent[ra]=rb
                    consumed.update((left,right))
            else:frontier[eid]=pattern
        groups={}
        for p in parent:
            if p not in consumed:groups.setdefault(find(p),[]).append(p)
        if any(len(row)!=2 for row in groups.values()):raise CertificateError('Invalid open component in regional transition.')
        following_pairs=tuple(sorted(tuple(sorted(row)) for row in groups.values()))
        if self.target=='gold' and cycles and not (level+1==self.problem.n and cycles==1 and not following_pairs):return None
        return (used|(1<<tile),tuple(frontier[e] for e in self.frontiers[level+1]),following_pairs,cycles)


def verify_certificate(owner,cells,domains,certificate,check=lambda:None,*,max_states=4096,max_steps=50000,stats=None):
    # Exact explicit limits are part of the parser, not instructions supplied by
    # the artifact. No serialized object is executed or deserialized as pickle.
    if any(type(bound)is not int or bound<0 for bound in (max_states,max_steps)):raise ValueError('Invalid proof verification bounds.')
    cells=tuple(cells);domains=tuple(domains);allowed={'schema','binding','cells','domains','root','states','count','supports'}
    if not isinstance(certificate,dict) or set(certificate)!=allowed or type(certificate['schema'])is not int or certificate['schema']!=1:
        raise CertificateError('Invalid regional proof schema.')
    if certificate['binding']!=digest(source_binding(owner)) or _freeze(certificate['cells'])!=cells or _freeze(certificate['domains'])!=domains:
        raise CertificateError('Regional proof source or exact query differs.')
    if not cells or cells!=tuple(sorted(set(cells))) or any(type(c)is not int or not 0<=c<owner.problem.n for c in cells):raise CertificateError('Invalid proof region.')
    if len(domains)!=owner.problem.n or any(type(mask)is not int or mask<0 or mask&~sum(1<<code for code in set(owner.problem.domains[c])) for c,mask in enumerate(domains)):
        raise CertificateError('Proof domains escape the original root.')
    states=certificate['states'];root=certificate['root'];count=certificate['count'];supports=_freeze(certificate['supports'])
    if not isinstance(states,list) or not states or len(states)>max_states:raise CertificateError('Invalid proof state count.')
    if type(root)is not int or not 0<=root<len(states) or type(count)is not int or count<0 or count.bit_length()>4096:
        raise CertificateError('Invalid proof root or result count.')
    if not isinstance(supports,tuple) or len(supports)!=len(cells) or any(type(mask)is not int or mask<0 or mask&~domains[cell] for cell,mask in zip(cells,supports)):
        raise CertificateError('Invalid claimed regional supports.')
    parsed=[];seen=set();steps=0
    def tick():
        nonlocal steps
        check();steps+=1
        if steps>max_steps:raise _Budget
    for record in states:
        tick()
        if not isinstance(record,dict) or set(record)!={'level','signature','edges'}:raise CertificateError('Invalid regional state record.')
        level=record['level'];signature=_freeze(record['signature']);edges=_freeze(record['edges'])
        if type(level)is not int or not 0<=level<=len(cells) or not isinstance(edges,tuple):raise CertificateError('Invalid state level/edges.')
        if signature is not None:
            if not isinstance(signature,tuple) or len(signature)!=4:raise CertificateError('Invalid regional signature.')
            used,masks,pairs,cycles=signature
            if type(used)is not int or used<0 or used>>owner.problem.n or type(cycles)is not int or not 0<=cycles<=33*owner.problem.n:
                raise CertificateError('Invalid regional inventory/cycles.')
            if not isinstance(masks,tuple) or any(type(mask)is not int or not 0<=mask<2048 for mask in masks) or not isinstance(pairs,tuple):raise CertificateError('Invalid regional seam/matching.')
            endpoints=[]
            for pair in pairs:
                if not isinstance(pair,tuple) or len(pair)!=2 or any(type(v)is not int or not 0<=v<33*owner.problem.n for v in pair) or pair[0]>=pair[1]:raise CertificateError('Invalid exposed endpoint pair.')
                endpoints.extend(pair)
            if len(endpoints)!=len(set(endpoints)) or pairs!=tuple(sorted(pairs)):raise CertificateError('Invalid exposed matching order.')
        key=(level,signature)
        if key in seen:raise CertificateError('Duplicate semantic regional state.')
        seen.add(key);parsed.append((level,signature,edges))
    if parsed[root][:2]!=(0,None):raise CertificateError('Proof root must be the empty region.')
    transition=_IndependentTransitions(owner,cells,tick);reachable=set();memo={}
    def visit(ident):
        tick()
        if ident in memo:return memo[ident]
        reachable.add(ident);level,signature,edges=parsed[ident]
        if stats is not None:stats['region_certificate_verified_states']+=1
        if level==len(cells):
            if edges:raise CertificateError('Terminal state has outgoing choices.')
            result=(1,(0,)*len(cells));memo[ident]=result;return result
        expected=tuple(values(domains[cells[level]]))
        if len(edges)!=len(expected) or any(not isinstance(edge,tuple) or len(edge)!=2 or type(edge[0])is not int or edge[0]!=code for edge,code in zip(edges,expected)):
            raise CertificateError('Proof omits or duplicates a domain choice.')
        total=0;union=[0]*len(cells)
        for code,child in edges:
            tick()
            if stats is not None:stats['region_certificate_verified_transitions']+=1
            following=transition.next(level,signature,code)
            if following is None:
                if child is not None:raise CertificateError('Rejected transition has a child.')
                continue
            if type(child)is not int or not 0<=child<len(parsed) or parsed[child][:2]!=(level+1,following):
                raise CertificateError('Legal transition is missing its exact child state.')
            number,keep=visit(child);total+=number
            if number:
                union[level]|=1<<code;union=[a|b for a,b in zip(union,keep)]
        result=total,tuple(union);memo[ident]=result;return result
    result=visit(root)
    if len(reachable)!=len(states) or result!=(count,supports):raise CertificateError('Regional completeness/count/support claim differs.')
    check();return {'count':count,'supports':supports,'states':len(states),'verification_ticks':steps}


class CertifiedRegionCache:
    def __init__(self,owner):
        self.owner=owner;self.binding=source_binding(owner);self.cache=OrderedDict();self.directory=None;self.stats=dict.fromkeys(STAT_KEYS,0)
    def configure_cache_dir(self,directory):self.directory=None if directory is None else Path(directory)/'regional-certificates-v1'
    def prune(self,cells,domains,check=lambda:None,*,max_states=4096,max_steps=50000):
        if source_binding(self.owner)!=self.binding:raise ValueError('Regional certificate source binding changed.')
        if any(type(bound)is not int or bound<0 for bound in (max_states,max_steps)):raise ValueError('Invalid regional certificate limits.')
        cells,validated=_validated_query(self.owner,cells,domains)
        if not all(validated):return False
        key=(cells,tuple(domains));path=None if self.directory is None else self.directory/(digest((self.binding,key))+'.json')
        check()
        if key in self.cache:self.stats['region_certificate_memory_hits']+=1;result=self.cache[key]
        else:
            self.stats['region_certificate_queries']+=1;certificate=None;result=None;loaded=False
            try:
                if path is not None and path.exists():
                    try:
                        if path.stat().st_size>16*1024*1024:raise CertificateError('Regional proof file too large.')
                        certificate=read_json_shared(path)
                    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RecursionError):
                        self.stats['region_certificate_rejected']+=1;certificate=None
                    if certificate is not None:
                        try:
                            result=verify_certificate(self.owner,cells,domains,certificate,check,max_states=max_states,max_steps=max_steps,stats=self.stats)
                            self.stats['region_certificate_loads']+=1;loaded=True
                        except CertificateError:
                            self.stats['region_certificate_rejected']+=1;result=None
                if result is None:
                    certificate=build_certificate(self.owner,cells,domains,check,max_states=max_states,max_steps=max_steps,stats=self.stats)
                    result=verify_certificate(self.owner,cells,domains,certificate,check,max_states=max_states,max_steps=max_steps,stats=self.stats)
                check();self.cache[key]=result
                while len(self.cache)>4:self.cache.popitem(last=False)
                self.stats['region_certificate_completed']+=1
            except _Budget:self.stats['region_certificate_unknown']+=1;return False
            if not loaded and path is not None:
                try:
                    payload=json.dumps(certificate,indent=2).replace('\n',os.linesep).encode('utf-8')
                    if len(payload)<=16*1024*1024 and (path.exists() or not path.parent.exists() or len(tuple(path.parent.glob('*.json')))<16):
                        path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,certificate)
                        self.stats['region_certificate_saves']+=1;self.stats['region_certificate_saved_bytes']+=len(payload)
                except OSError:self.stats['region_certificate_write_failures']+=1
        current=list(domains)
        for cell,mask in zip(cells,result['supports']):current[cell]&=mask
        check();self.stats['region_certificate_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
