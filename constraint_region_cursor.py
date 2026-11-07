"""Bounded trusted-local regional compiler continuation (round25 #15).

JSON checksums detect corruption; they are NOT mathematical certificates. These
files are created and consumed only by the local regional cache. External proof
and portable frontier imports cannot supply this continuation state.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,os
from constraint_region_network import assignment_signature,joined_signature
from constraint_region_incremental import source_binding,CURSOR_STAT_KEYS as STAT_KEYS
from constraint_pruning import values
from constraint_separator import _Budget
from io_utils import atomic_json,read_json_shared

VERSION='diamond-local-region-cursor-1'



def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def nested_tuple(value):return tuple(nested_tuple(x) for x in value) if isinstance(value,list) else value

def exact_int(value,low=0,high=None):
    if type(value)is not int or value<low or high is not None and value>high:raise ValueError('Malformed bounded cursor integer.')
    return value


class PortableRegionSession:
    """Explicit deterministic transition stack; only complete results escape."""
    def __init__(self,owner,cells,domains,max_rows=4096,max_steps=50000):
        self.owner=owner;raw_cells=tuple(cells);self.domains=tuple(domains)
        if (any(type(c)is not int or not 0<=c<owner.problem.n for c in raw_cells)
                or len(self.domains)!=owner.problem.n or any(type(mask)is not int or mask<0
                   or mask&~sum(1<<code for code in set(owner.problem.domains[cell])) for cell,mask in enumerate(self.domains))):
            raise ValueError('Invalid exact regional cursor domains.')
        self.cells=tuple(sorted(set(raw_cells)))
        self.max_rows=exact_int(max_rows);self.max_steps=exact_int(max_steps)
        if not self.cells or any(type(c)is not int or not 0<=c<owner.problem.n for c in self.cells):raise ValueError('Invalid cursor region.')
        self.choices=tuple(tuple(values(self.domains[c])) for c in self.cells)
        self.nodes={};self.terminals={};self._counts={};self._memo={};self._intern={};self._singletons={};self._next_id=0
        self.steps=0;self.states=0;self.memo_hits=0;self.ticks=0;self.rows=0;self.root=None;self.complete=False;self.failed=False;self.broken=False
        self.stack=[self._frame(0,None)]
    @staticmethod
    def _frame(level,signature):return {'level':level,'signature':signature,'stage':'enter','index':0,'edges':[],'pending':None}
    def _return(self,result):
        self.stack.pop()
        if not self.stack:
            self.root=result;self.rows=0 if result is None else self._counts[result];self.complete=True;return
        parent=self.stack[-1]
        if result is not None:parent['edges'].append((parent['pending'],result))
        parent['pending']=None;parent['stage']='choices'
    def _step(self):
        self.ticks+=1;frame=self.stack[-1];level=frame['level'];signature=frame['signature'];key=(level,signature)
        if frame['stage']=='enter':
            if key in self._memo:self.memo_hits+=1;self._return(self._memo[key]);return
            self.states+=1
            if self.states>self.max_rows:raise _Budget
            if level==len(self.cells):
                structural=('terminal',signature);ident=self._intern.get(structural)
                if ident is None:
                    ident=self._next_id;self._next_id+=1;self._intern[structural]=ident
                    self.terminals[ident]=signature;self._counts[ident]=1
                self._memo[key]=ident;self._return(ident);return
            frame['stage']='choices';return
        if frame['stage']!='choices':raise RuntimeError('Invalid live regional frame.')
        if frame['index']==len(self.choices[level]):
            edges=tuple(frame['edges'])
            if not edges:self._memo[key]=None;self._return(None);return
            structural=(level,edges);ident=self._intern.get(structural)
            if ident is None:
                ident=self._next_id;self._next_id+=1;self._intern[structural]=ident
                self.nodes[ident]=(level,edges);self._counts[ident]=sum(self._counts[c] for _,c in edges)
            self._memo[key]=ident;self._return(ident);return
        code=self.choices[level][frame['index']];frame['index']+=1;self.steps+=1
        if self.steps>self.max_steps:raise _Budget
        cell=self.cells[level];skey=(cell,code)
        if skey not in self._singletons:self._singletons[skey]=assignment_signature(self.owner,(cell,),(code,),self.domains,lambda:None)
        single=self._singletons[skey]
        if single is None:return
        following=single if level==0 else joined_signature(self.owner,self.cells[:level],signature,(cell,),single,self.domains,lambda:None)
        if following is None:return
        frame['pending']=code;frame['stage']='child';self.stack.append(self._frame(level+1,following))
    def advance(self,quantum=256,check=lambda:None):
        if type(quantum)is not int or quantum<1:raise ValueError('A cursor slice must be positive.')
        if self.failed:raise _Budget
        if self.broken:raise ValueError('An aborted internal transition cannot be resumed.')
        for _ in range(quantum):
            check()
            if self.complete:return True
            try:self._step()
            except _Budget:self.failed=True;raise
            except BaseException:self.broken=True;raise
        return self.complete
    def result(self):
        if not self.complete:raise ValueError('An unfinished regional cursor is not authoritative.')
        return {name:getattr(self,name) for name in ('cells','nodes','terminals','rows','steps','states','memo_hits','root','_counts')}
    def _local_payload(self):
        if self.broken:raise ValueError('An aborted internal transition cannot be saved.')
        return {'cells':self.cells,'domains':self.domains,'max_rows':self.max_rows,'max_steps':self.max_steps,
                'nodes':list(self.nodes.items()),'terminals':list(self.terminals.items()),
                'memo':list(self._memo.items()),'singletons':list(self._singletons.items()),'stack':self.stack,
                'steps':self.steps,'states':self.states,'memo_hits':self.memo_hits,'ticks':self.ticks,
                'root':self.root,'complete':self.complete,'failed':self.failed}
    @classmethod
    def _from_local_payload(cls,owner,payload,cells,domains,max_rows,max_steps):
        # This checks bounded structural invariants of trusted local state.
        # It intentionally does not claim independent transition completeness.
        required={'cells','domains','max_rows','max_steps','nodes','terminals','memo','singletons','stack',
                  'steps','states','memo_hits','ticks','root','complete','failed'}
        if not isinstance(payload,dict) or set(payload)!=required:raise ValueError('Malformed local regional cursor.')
        if (not isinstance(payload['cells'],list) or any(type(v)is not int for v in payload['cells'])
                or not isinstance(payload['domains'],list) or any(type(v)is not int for v in payload['domains'])
                or nested_tuple(payload['cells'])!=tuple(cells) or nested_tuple(payload['domains'])!=tuple(domains)
                or type(payload['max_rows'])is not int or type(payload['max_steps'])is not int
                or payload['max_rows']!=max_rows or payload['max_steps']!=max_steps):raise ValueError('Cursor query or limits differ.')
        result=cls(owner,cells,domains,max_rows,max_steps);n=len(cells)
        result.steps=exact_int(payload['steps'],0,max_steps+1);result.states=exact_int(payload['states'],0,max_rows+1)
        result.memo_hits=exact_int(payload['memo_hits'],0,max_steps+1)
        result.ticks=exact_int(payload['ticks'],0,2*max_rows+3*max_steps+8)
        if type(payload['complete'])is not bool or type(payload['failed'])is not bool:raise ValueError('Malformed cursor status.')
        result.complete=payload['complete'];result.failed=payload['failed']
        if result.complete and result.failed:raise ValueError('A failed cursor cannot be complete.')
        exceeded=result.steps>max_steps or result.states>max_rows
        if result.failed!=exceeded:raise ValueError('Cursor failure status differs from its work limits.')
        def signature(raw,members):
            item=nested_tuple(raw)
            if not isinstance(item,tuple) or len(item)!=4:raise ValueError('Malformed cursor signature.')
            used,masks,pairs,cycles=item;exact_int(used,0,(1<<owner.problem.n)-1);exact_int(cycles,0,33*owner.problem.n)
            if used.bit_count()!=len(members) or not isinstance(masks,tuple) or len(masks)!=len(owner._frontier(set(members))):raise ValueError('Cursor signature dimensions differ.')
            for mask in masks:exact_int(mask,0,2047)
            if not isinstance(pairs,tuple) or len(pairs)>33*owner.problem.n//2:raise ValueError('Malformed cursor pairing.')
            ports=set()
            for pair in pairs:
                if not isinstance(pair,tuple) or len(pair)!=2:raise ValueError('Malformed cursor endpoint.')
                a,b=pair;exact_int(a,0,33*owner.problem.n-1);exact_int(b,0,33*owner.problem.n-1)
                if a>=b or a in ports or b in ports or a//33 not in members or b//33 not in members:raise ValueError('Malformed cursor endpoint ownership.')
                ports.update(pair)
            if tuple(sorted(pairs))!=pairs:raise ValueError('Noncanonical cursor pairing.')
            return item
        def pairs(field,limit):
            rows=payload[field]
            if not isinstance(rows,list) or len(rows)>limit:raise ValueError('Cursor table exceeds its bound.')
            for row in rows:
                if not isinstance(row,list) or len(row)!=2:raise ValueError('Malformed cursor table row.')
            return rows
        for ident,raw in pairs('terminals',max_rows):
            exact_int(ident,0,max_rows)
            if ident in result.terminals:raise ValueError('Duplicate cursor terminal.')
            result.terminals[ident]=signature(raw,cells);result._counts[ident]=1
            key=('terminal',result.terminals[ident])
            if key in result._intern:raise ValueError('Duplicate structural cursor terminal.')
            result._intern[key]=ident
        for ident,raw in pairs('nodes',max_rows):
            exact_int(ident,0,max_rows)
            if ident in result.nodes or ident in result.terminals:raise ValueError('Duplicate cursor node.')
            raw=nested_tuple(raw)
            if not isinstance(raw,tuple) or len(raw)!=2:raise ValueError('Malformed cursor node.')
            level,edges=raw;exact_int(level,0,n-1)
            if not isinstance(edges,tuple) or not edges or len(edges)>len(result.choices[level]):raise ValueError('Malformed cursor outgoing edges.')
            prior=-1
            for edge in edges:
                if not isinstance(edge,tuple) or len(edge)!=2:raise ValueError('Malformed cursor edge.')
                code,child=edge;exact_int(code,0,3*owner.problem.n-1);exact_int(child,0,max_rows)
                if code<=prior or code not in result.choices[level] or child>=ident:raise ValueError('Invalid cursor edge ordering.')
                prior=code
            result.nodes[ident]=(level,edges)
        identifiers=set(result.nodes)|set(result.terminals)
        if len(identifiers)>max_rows or identifiers!=set(range(len(identifiers))):raise ValueError('Cursor node identifiers are incomplete.')
        result._next_id=len(identifiers)
        def destination(ident,level):
            if ident is None:return
            exact_int(ident,0,max_rows)
            if (ident not in identifiers or (level==n and ident not in result.terminals)
                    or (level<n and (ident not in result.nodes or result.nodes[ident][0]!=level))):raise ValueError('Cursor destination level differs.')
        for ident in sorted(result.nodes):
            level,edges=result.nodes[ident]
            for _,child in edges:destination(child,level+1)
            result._counts[ident]=sum(result._counts[child] for _,child in edges)
            key=(level,edges)
            if key in result._intern:raise ValueError('Duplicate structural cursor node.')
            result._intern[key]=ident
        for raw,ident in pairs('memo',max_rows):
            raw=nested_tuple(raw)
            if not isinstance(raw,tuple) or len(raw)!=2:raise ValueError('Malformed cursor memo key.')
            level,value=raw;exact_int(level,0,n)
            if level==0:
                if value is not None:raise ValueError('Malformed cursor root state.')
            else:value=signature(value,cells[:level])
            key=(level,value);destination(ident,level)
            if key in result._memo:raise ValueError('Duplicate cursor memo entry.')
            result._memo[key]=ident
        for raw,value in pairs('singletons',sum(map(len,result.choices))):
            raw=nested_tuple(raw)
            if not isinstance(raw,tuple) or len(raw)!=2:raise ValueError('Malformed singleton key.')
            cell,code=raw;exact_int(cell,0,owner.problem.n-1);exact_int(code,0,3*owner.problem.n-1)
            if cell not in cells or not domains[cell]>>code&1 or raw in result._singletons:raise ValueError('Invalid cursor singleton.')
            result._singletons[raw]=None if value is None else signature(value,(cell,))
        stack=payload['stack']
        if not isinstance(stack,list) or len(stack)>n+1:raise ValueError('Invalid cursor stack depth.')
        result.stack=[]
        for level,raw in enumerate(stack):
            if not isinstance(raw,dict) or set(raw)!={'level','signature','stage','index','edges','pending'}:raise ValueError('Malformed cursor frame.')
            if type(raw['level'])is not int or raw['level']!=level:raise ValueError('Cursor frame order differs.')
            frame=dict(raw);frame['signature']=None if level==0 else signature(raw['signature'],cells[:level])
            if level==0 and raw['signature'] is not None:raise ValueError('Invalid root frame.')
            if level==n and frame['stage']!='enter':raise ValueError('A terminal cursor frame must be unentered.')
            if frame['stage'] not in ('enter','choices','child'):raise ValueError('Invalid cursor stage.')
            index=exact_int(frame['index'],0,len(result.choices[level]) if level<n else 0)
            frame['edges']=nested_tuple(raw['edges'])
            if not isinstance(frame['edges'],tuple) or len(frame['edges'])>index:raise ValueError('Invalid partial cursor edges.')
            prior=-1
            for code,child in frame['edges']:
                exact_int(code);destination(child,level+1)
                if child is None or code<=prior or code not in result.choices[level][:index-1 if frame['stage']=='child' else index]:raise ValueError('Invalid completed cursor edge.')
                prior=code
            frame['edges']=list(frame['edges'])
            if level+1<len(stack):
                if frame['stage']!='child' or not index or type(frame['pending'])is not int or frame['pending']!=result.choices[level][index-1]:raise ValueError('Invalid awaiting-child frame.')
            elif frame['stage']=='child' or frame['pending'] is not None:raise ValueError('Cursor child is missing.')
            if frame['stage']=='enter' and (index or frame['edges']):raise ValueError('An unentered frame has completed work.')
            result.stack.append(frame)
        entered=sum(frame['stage']!='enter' for frame in result.stack)
        over_state=int(result.failed and result.states>max_rows)
        if result.states!=len(result._memo)+entered+over_state:raise ValueError('Cursor state accounting differs.')
        if result.complete:
            if stack:raise ValueError('Complete cursor still has live frames.')
            destination(payload['root'],0);result.root=payload['root']
            if result._memo.get((0,None),'missing')!=result.root:raise ValueError('Complete cursor has no matching root result.')
            result.rows=0 if result.root is None else result._counts[result.root]
        elif not stack or payload['root'] is not None:raise ValueError('Unfinished cursor has invalid root state.')
        return result


class LocalRegionCursorStore:
    def __init__(self,directory,owner,*,max_bytes=16*1024*1024,max_entries=16):
        self.directory=Path(directory)/'regional-cursors-v1';self.owner=owner
        exact_int(max_bytes);exact_int(max_entries)
        self.max_bytes=max_bytes;self.max_entries=max_entries;self.stats=dict.fromkeys(STAT_KEYS,0);self.last_error=None
        folder=Path(__file__).resolve().parent
        names=('constraint_region_cursor.py','constraint_region_incremental.py','constraint_region_symbolic.py',
               'constraint_region_network.py','constraint_regions.py','constraint_separator.py',
               'constraint_pruning.py','validator.py','search_problem.py','geometry.py','io_utils.py')
        self.producer=tuple((name,hashlib.sha256((folder/name).read_bytes()).hexdigest()) for name in names)
    def _identity(self,key):return {'version':VERSION,'source':source_binding(self.owner),'producer':self.producer,'query':key}
    def _path(self,key):return self.directory/(hashlib.sha256(encoded(self._identity(key))).hexdigest()+'.json')
    def load(self,key,check=lambda:None):
        check();path=self._path(key)
        try:
            if not path.exists():return None
            if path.stat().st_size>self.max_bytes:raise ValueError('Local cursor exceeds its byte bound.')
            record=read_json_shared(path)
            if not isinstance(record,dict) or set(record)!={'payload','sha256'}:raise ValueError('Malformed local cursor record.')
            payload=record['payload']
            if not isinstance(payload,dict) or set(payload)!={'identity','cursor'}:raise ValueError('Malformed local cursor payload.')
            if record['sha256']!=hashlib.sha256(encoded(payload)).hexdigest():raise ValueError('Local cursor checksum differs.')
            if encoded(payload['identity'])!=encoded(self._identity(key)):raise ValueError('Local cursor binding differs.')
            cells,domains,max_rows,max_steps=key
            result=PortableRegionSession._from_local_payload(self.owner,payload['cursor'],cells,domains,max_rows,max_steps)
            pass
        except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RecursionError):
            self.stats['region_cursor_rejections']+=1;return None
        check();self.stats['region_cursor_loads']+=1;self.stats['region_cursor_loaded_ticks']+=result.ticks
        self.stats['region_cursor_loaded_transitions']+=result.steps;self.stats['region_cursor_limited_hits']+=int(result.failed)
        return result
    def save(self,key,session):
        if self.max_entries<=0:self.stats['region_cursor_store_caps']+=1;return False
        try:
            payload={'identity':self._identity(key),'cursor':session._local_payload()}
            record={'payload':payload,'sha256':hashlib.sha256(encoded(payload)).hexdigest()}
            # Match atomic_json's indented UTF-8 text AND native newline
            # translation; the compact digest encoding is not the file size.
            size=len(json.dumps(record,indent=2).replace('\n',os.linesep).encode('utf-8'))
            if size>self.max_bytes:self.stats['region_cursor_store_caps']+=1;return False
            self.directory.mkdir(parents=True,exist_ok=True);path=self._path(key)
            if not path.exists() and sum(1 for _ in self.directory.glob('*.json'))>=self.max_entries:
                self.stats['region_cursor_store_caps']+=1;return False
            atomic_json(path,record);self.stats['region_cursor_writes']+=1;self.stats['region_cursor_saved_bytes']+=size;return True
        except (OSError,ValueError,TypeError,OverflowError,RecursionError) as error:
            self.last_error=str(error);self.stats['region_cursor_write_failures']+=1;return False
